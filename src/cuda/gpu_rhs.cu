// gpu_rhs.cu — P8.3: per-block GPU RHS kernels (WENO5-Z + HLLC-ES + viscous)
//
// Three-kernel pipeline per exec() call (one CUDA block per leaf):
//   k_prim_duc  — conservative → prim + µ (Sutherland) + Ducros φ → d_scratch
//   k_rhs_conv  — WENO5-Z/KEP/HLLC-ES face-centred convective flux (atomicAdd)
//   k_rhs_visc  — face-averaged µ viscous divergence (direct write, cell-centred)
//
// GpuRhsList::exec() zeros d_rhs_pool, launches the three kernels in order,
// then returns.  download_rhs() DtoH-copies the RHS back to CellBlock::Q.

#include "../../include/cuda/gpu_rhs.cuh"
#include "../../include/cuda/gpu_hllc.cuh"
#include "../../include/cuda/gpu_check.cuh"
#include "../../include/cuda/gpu_meta_buffer.cuh"
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
double gpu_log_mean(double a, double b) noexcept {
    const double xi = a / b;
    const double f  = (xi - 1.0) / (xi + 1.0);
    const double u2 = f * f;
    const double F  = (u2 < 1.0e-4)
                    ? 1.0 + u2 * (1.0/3.0 + u2 * (1.0/5.0 + u2 / 7.0))
                    : log(xi) / (2.0 * f);
    return (a + b) / (2.0 * F);
}

// Entropy-stable HLLC-ES flux (Chandrashekar 2013)
__device__ __forceinline__
void gpu_hllc_es_flux(const GPrim& L, const GPrim& R, int axis,
                      double F[GPU_NVAR]) noexcept {
    const double rho_a  = 0.5*(L.rho + R.rho);
    const double u_a    = 0.5*(L.u   + R.u  );
    const double v_a    = 0.5*(L.v   + R.v  );
    const double w_a    = 0.5*(L.w   + R.w  );
    const double beta_L = L.rho / (2.0*L.p);
    const double beta_R = R.rho / (2.0*R.p);
    const double beta_a = 0.5*(beta_L + beta_R);
    const double rho_ln  = gpu_log_mean(L.rho,  R.rho );
    const double beta_ln = gpu_log_mean(beta_L, beta_R);
    const double p_hat   = rho_a / (2.0 * beta_a);
    const double un_L = (axis==0)?L.u:(axis==1)?L.v:L.w;
    const double un_R = (axis==0)?R.u:(axis==1)?R.v:R.w;
    const double un_a = 0.5*(un_L + un_R);
    const double mass = rho_ln * un_a;
    const double KE_hat = 0.5*(u_a*u_a + v_a*v_a + w_a*w_a);
    const double H_hat  = 1.0/(2.0*(GPU_GAMMA-1.0)*beta_ln) + KE_hat + p_hat/rho_ln;
    F[0] = mass;
    F[1] = mass*u_a + (axis==0 ? p_hat : 0.0);
    F[2] = mass*v_a + (axis==1 ? p_hat : 0.0);
    F[3] = mass*w_a + (axis==2 ? p_hat : 0.0);
    F[4] = mass*H_hat;
    const double lam = fmax(fabs(un_L)+L.c, fabs(un_R)+R.c);
    const double E_L = L.p/(GPU_GAMMA-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R = R.p/(GPU_GAMMA-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    F[0] -= 0.5*lam*(R.rho     - L.rho    );
    F[1] -= 0.5*lam*(R.rho*R.u - L.rho*L.u);
    F[2] -= 0.5*lam*(R.rho*R.v - L.rho*L.v);
    F[3] -= 0.5*lam*(R.rho*R.w - L.rho*L.w);
    F[4] -= 0.5*lam*(E_R       - E_L      );
}

// KE-preserving flux (Pirozzoli 2011)
__device__ __forceinline__
void gpu_kep_flux(const GPrim& L, const GPrim& R, int axis,
                  double F[GPU_NVAR]) noexcept {
    const double rho_a = 0.5*(L.rho + R.rho);
    const double u_a   = 0.5*(L.u   + R.u  );
    const double v_a   = 0.5*(L.v   + R.v  );
    const double w_a   = 0.5*(L.w   + R.w  );
    const double p_a   = 0.5*(L.p   + R.p  );
    const double E_L   = L.p/(GPU_GAMMA-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R   = R.p/(GPU_GAMMA-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    const double H_a   = 0.5*((E_L+L.p)/L.rho + (E_R+R.p)/R.rho);
    const double un_a  = (axis==0)?u_a:(axis==1)?v_a:w_a;
    const double mass  = rho_a * un_a;
    F[0] = mass;
    F[1] = mass*u_a + (axis==0 ? p_a : 0.0);
    F[2] = mass*v_a + (axis==1 ? p_a : 0.0);
    F[3] = mass*w_a + (axis==2 ? p_a : 0.0);
    F[4] = mass*H_a;
}

// One-sided WENO5-Z upwind reconstruction (Borges et al. 2008).
// Stencil [a,b,c,d,e] = [vm2,vm1,v0,vp1,vp2] for the left state.
// For the right state, pass the mirrored stencil [vp3,vp2,vp1,v0,vm1].
__device__ __forceinline__
double weno5z_upwind(double a, double b, double c, double d, double e) noexcept {
    constexpr double eps = 1.0e-36;
    constexpr double d0 = 0.1, d1 = 0.6, d2 = 0.3;
    const double s0 = ( 2.0*a -  7.0*b + 11.0*c) * (1.0/6.0);
    const double s1 = (    -b +  5.0*c +  2.0*d) * (1.0/6.0);
    const double s2 = ( 2.0*c +  5.0*d -      e) * (1.0/6.0);
    const double b0 = (13.0/12.0)*(a-2.0*b+c)*(a-2.0*b+c)
                    +  (1.0/ 4.0)*(a-4.0*b+3.0*c)*(a-4.0*b+3.0*c);
    const double b1 = (13.0/12.0)*(b-2.0*c+d)*(b-2.0*c+d)
                    +  (1.0/ 4.0)*(b-d)*(b-d);
    const double b2 = (13.0/12.0)*(c-2.0*d+e)*(c-2.0*d+e)
                    +  (1.0/ 4.0)*(3.0*c-4.0*d+e)*(3.0*c-4.0*d+e);
    const double tau5 = fabs(b0 - b2);
    const double a0 = d0*(1.0+(tau5/(b0+eps))*(tau5/(b0+eps)));
    const double a1 = d1*(1.0+(tau5/(b1+eps))*(tau5/(b1+eps)));
    const double a2 = d2*(1.0+(tau5/(b2+eps))*(tau5/(b2+eps)));
    return (a0*s0 + a1*s1 + a2*s2) / (a0 + a1 + a2);
}

// WENO5-Z scalar reconstruction (Borges et al. 2008)
__device__ __forceinline__
void gpu_weno5z_scalar(double vm2, double vm1, double v0,
                       double vp1, double vp2, double vp3,
                       double& vL, double& vR) noexcept {
    vL = weno5z_upwind(vm2, vm1, v0,  vp1, vp2);   // left state
    vR = weno5z_upwind(vp3, vp2, vp1, v0,  vm1);   // right state (mirrored)
}

// WENO5 face reconstruction with Roe characteristic decomposition.
// Reads prim from d_scratch (comp-major: sp[comp*NCELL + flat]).
// (i,j,k) = left cell of face; axis = normal direction.
// Requires NG=2: stencil offset d ∈ {-2,-1,0,+1,+2,+3} all in-bounds.
__device__ __forceinline__
void gpu_weno5_face(const double* __restrict__ sp,
                    int i, int j, int k, int axis,
                    GPrim& qL_out, GPrim& qR_out) noexcept {
    // ── Stencil ───────────────────────────────────────────────────────────────
    auto sidx = [&](int d) -> int {
        if (axis == 0) return gpu_cell_idx(i+d, j, k);
        if (axis == 1) return gpu_cell_idx(i, j+d, k);
        return                gpu_cell_idx(i, j, k+d);
    };

    // Conservative stencil Q[m], m=0(d=-2)..5(d=+3)
    double Q[6][GPU_NVAR];
    for (int m = 0; m < 6; ++m) {
        int flat = sidx(m-2);
        const double rho = sp[0*GPU_NCELL+flat];
        const double u   = sp[1*GPU_NCELL+flat];
        const double v   = sp[2*GPU_NCELL+flat];
        const double w   = sp[3*GPU_NCELL+flat];
        const double p   = sp[4*GPU_NCELL+flat];
        Q[m][0] = rho;
        Q[m][1] = rho*u; Q[m][2] = rho*v; Q[m][3] = rho*w;
        Q[m][4] = p/(GPU_GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }

    // ── Roe average between cells i (m=2) and i+1 (m=3) ─────────────────────
    int f2 = sidx(0), f3 = sidx(1);
    const double rL = sp[0*GPU_NCELL+f2], uL = sp[1*GPU_NCELL+f2];
    const double vLs= sp[2*GPU_NCELL+f2], wL = sp[3*GPU_NCELL+f2];
    const double pL = sp[4*GPU_NCELL+f2], TL = sp[5*GPU_NCELL+f2], cL = sp[6*GPU_NCELL+f2];
    const double rR = sp[0*GPU_NCELL+f3], uR = sp[1*GPU_NCELL+f3];
    const double vR = sp[2*GPU_NCELL+f3], wR = sp[3*GPU_NCELL+f3];
    const double pR = sp[4*GPU_NCELL+f3], TR = sp[5*GPU_NCELL+f3], cR = sp[6*GPU_NCELL+f3];

    const double sqL   = sqrt(rL), sqR = sqrt(rR), denom = sqL+sqR;
    const double u_roe = (sqL*uL + sqR*uR)/denom;
    const double v_roe = (sqL*vLs+ sqR*vR)/denom;
    const double w_roe = (sqL*wL + sqR*wR)/denom;
    const double HL    = (Q[2][4]+pL)/rL;
    const double HR    = (Q[3][4]+pR)/rR;
    const double H_roe = (sqL*HL+sqR*HR)/denom;
    const double KE    = 0.5*(u_roe*u_roe+v_roe*v_roe+w_roe*w_roe);
    const double c2    = fmax((GPU_GAMMA-1.0)*(H_roe-KE), 1.0e-300);
    const double c_roe = sqrt(c2);

    const double un   = (axis==0)?u_roe:(axis==1)?v_roe:w_roe;
    const double ut1  = (axis==0)?v_roe:(axis==1)?u_roe:u_roe;
    const double ut2  = (axis==0)?w_roe:(axis==1)?w_roe:v_roe;
    const int    nidx = 1+axis;
    const int  t1idx  = (axis==0)?2:1;
    const int  t2idx  = (axis==2)?2:3;
    const double bv   = (GPU_GAMMA-1.0)/c2;
    const double b2v  = bv*KE;
    const double ioc  = 1.0/c_roe;

    // ── Characteristic projection ─────────────────────────────────────────────
    double W[5][6];
    for (int m = 0; m < 6; ++m) {
        const double rho = Q[m][0];
        const double qn  = Q[m][nidx];
        const double qt1 = Q[m][t1idx];
        const double qt2 = Q[m][t2idx];
        const double E   = Q[m][4];
        const double inner   = b2v*rho - bv*(un*qn+ut1*qt1+ut2*qt2) + bv*E;
        const double delta_n = ioc*(un*rho - qn);
        W[0][m] = 0.5*(inner + delta_n);
        W[1][m] = (1.0-b2v)*rho + bv*(un*qn+ut1*qt1+ut2*qt2) - bv*E;
        W[2][m] = -ut1*rho + qt1;
        W[3][m] = -ut2*rho + qt2;
        W[4][m] = 0.5*(inner - delta_n);
    }

    double wL_w[5], wR_w[5];
    for (int kk = 0; kk < 5; ++kk)
        gpu_weno5z_scalar(W[kk][0],W[kk][1],W[kk][2],
                          W[kk][3],W[kk][4],W[kk][5],
                          wL_w[kk], wR_w[kk]);

    // ── Back-project ──────────────────────────────────────────────────────────
    double QL[GPU_NVAR], QR[GPU_NVAR];
    auto back_project = [&](const double w[5], double Qrec[GPU_NVAR]) {
        const double w014 = w[0]+w[1]+w[4];
        const double dw04 = w[4]-w[0];
        Qrec[0]     = w014;
        Qrec[nidx]  = w014*un  + dw04*c_roe;
        Qrec[t1idx] = w014*ut1 + w[2];
        Qrec[t2idx] = w014*ut2 + w[3];
        Qrec[4]     = (w[0]+w[4])*H_roe + dw04*un*c_roe
                    + w[1]*KE + w[2]*ut1 + w[3]*ut2;
    };
    back_project(wL_w, QL);
    back_project(wR_w, QR);

    // ── Convert to prim; fall back to cell-center if non-physical ────────────
    GPrim fbL; fbL.rho=rL; fbL.u=uL; fbL.v=vLs; fbL.w=wL;
               fbL.p=pL;   fbL.T=TL; fbL.c=cL;
    GPrim fbR; fbR.rho=rR; fbR.u=uR; fbR.v=vR;  fbR.w=wR;
               fbR.p=pR;   fbR.T=TR; fbR.c=cR;

    auto safe_prim = [](const double Qc[GPU_NVAR], const GPrim& fb) -> GPrim {
        const double rho = Qc[0]; if (rho <= 0.0) return fb;
        const double u = Qc[1]/rho, v = Qc[2]/rho, w = Qc[3]/rho;
        const double p = (GPU_GAMMA-1.0)*(Qc[4]-0.5*rho*(u*u+v*v+w*w));
        if (p <= 0.0) return fb;
        GPrim q; q.rho=rho; q.u=u; q.v=v; q.w=w;
        q.p=p; q.T=p/(rho*GPU_R_GAS); q.c=sqrt(GPU_GAMMA*p/rho);
        return q;
    };
    qL_out = safe_prim(QL, fbL);
    qR_out = safe_prim(QR, fbR);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_prim_duc: conservative → primitives + µ + Ducros φ → d_scratch
// Grid: (n_leaves)  Block: (GPU_NB2, GPU_NB2) = 144 threads
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_prim_duc(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    int i = threadIdx.x, j = threadIdx.y;

    // ── Pass 1: prim + µ ─────────────────────────────────────────────────────
    for (int k = 0; k < GPU_NB2; ++k) {
        int flat = gpu_cell_idx(i, j, k);
        GPrim q = gpu_cons_to_prim(
            m.d_Q[0*GPU_NCELL+flat], m.d_Q[1*GPU_NCELL+flat],
            m.d_Q[2*GPU_NCELL+flat], m.d_Q[3*GPU_NCELL+flat],
            m.d_Q[4*GPU_NCELL+flat]);
        m.d_scratch[0*GPU_NCELL+flat] = q.rho;
        m.d_scratch[1*GPU_NCELL+flat] = q.u;
        m.d_scratch[2*GPU_NCELL+flat] = q.v;
        m.d_scratch[3*GPU_NCELL+flat] = q.w;
        m.d_scratch[4*GPU_NCELL+flat] = q.p;
        m.d_scratch[5*GPU_NCELL+flat] = q.T;
        m.d_scratch[6*GPU_NCELL+flat] = q.c;
        m.d_scratch[7*GPU_NCELL+flat] = gpu_sutherland(q.T);
    }
    __syncthreads();

    // ── Pass 2: Ducros sensor (reads neighbour prim from d_scratch) ───────────
    constexpr double eps_duc = 1.0e-30;
    for (int k = 0; k < GPU_NB2; ++k) {
        double duc = 0.0;
        if (i >= 1 && i < GPU_NB2-1 && j >= 1 && j < GPU_NB2-1
            && k >= 1 && k < GPU_NB2-1) {
            const double* sp = m.d_scratch;
            const double ih2 = 0.5 / m.h;
            auto U = [=](int ii,int jj,int kk){ return sp[1*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto V = [=](int ii,int jj,int kk){ return sp[2*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto W = [=](int ii,int jj,int kk){ return sp[3*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto P = [=](int ii,int jj,int kk){ return sp[4*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };

            const double dudx = ih2*(U(i+1,j,k)-U(i-1,j,k));
            const double dudy = ih2*(U(i,j+1,k)-U(i,j-1,k));
            const double dudz = ih2*(U(i,j,k+1)-U(i,j,k-1));
            const double dvdx = ih2*(V(i+1,j,k)-V(i-1,j,k));
            const double dvdy = ih2*(V(i,j+1,k)-V(i,j-1,k));
            const double dvdz = ih2*(V(i,j,k+1)-V(i,j,k-1));
            const double dwdx = ih2*(W(i+1,j,k)-W(i-1,j,k));
            const double dwdy = ih2*(W(i,j+1,k)-W(i,j-1,k));
            const double dwdz = ih2*(W(i,j,k+1)-W(i,j,k-1));
            const double divu = dudx + dvdy + dwdz;
            const double ox = dwdy-dvdz, oy = dudz-dwdx, oz = dvdx-dudy;
            const double d2 = divu*divu;
            const double c2 = ox*ox+oy*oy+oz*oz;
            const double phi_vel = d2/(d2+c2+eps_duc);

            const double pC  = P(i,j,k);
            const double dpx = fmax(fabs(P(i+1,j,k)-pC), fabs(P(i-1,j,k)-pC));
            const double dpy = fmax(fabs(P(i,j+1,k)-pC), fabs(P(i,j-1,k)-pC));
            const double dpz = fmax(fabs(P(i,j,k+1)-pC), fabs(P(i,j,k-1)-pC));
            const double phi_p = fmax(dpx, fmax(dpy,dpz)) / (pC+eps_duc);
            const double phi_p_cl = fmin(1.0, fmax(0.0, (phi_p-0.1)*10.0));

            duc = fmax(phi_vel, phi_p_cl);
        }
        m.d_scratch[8*GPU_NCELL + gpu_cell_idx(i,j,k)] = duc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rhs_conv: hybrid WENO5-Z/KEP/HLLC-ES face-centred convective flux
// Grid: (n_leaves)  Block: (192) flat threads
// Iterates over all 3×(NB+1)×NB² = 1728 faces per leaf.
// Uses atomicAdd because each face writes to two independent cells.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rhs_conv(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const int  ilo    = GPU_NG;
    const int  ihi    = GPU_NG + GPU_NB - 1;
    const double ih   = 1.0 / m.h;
    constexpr double kep_thr = 1.0e-8;

    // Face counts per axis: (NB+1)*NB*NB = 576 ; total = 1728
    constexpr int NF   = GPU_NB + 1;   // 9 faces along normal axis
    constexpr int FPA  = NF * GPU_NB * GPU_NB;  // 576 per axis
    constexpr int FTOT = 3 * FPA;               // 1728

    auto load_prim = [&](int flat, GPrim& q) {
        q.rho = sp[0*GPU_NCELL+flat]; q.u = sp[1*GPU_NCELL+flat];
        q.v   = sp[2*GPU_NCELL+flat]; q.w = sp[3*GPU_NCELL+flat];
        q.p   = sp[4*GPU_NCELL+flat]; q.T = sp[5*GPU_NCELL+flat];
        q.c   = sp[6*GPU_NCELL+flat];
    };

    for (int fid = threadIdx.x; fid < FTOT; fid += blockDim.x) {
        const int axis  = fid / FPA;
        const int fi    = fid % FPA;
        const int f0    = fi % NF;          // face along normal: 0..NB
        const int fa    = (fi / NF) % GPU_NB; // transverse dim 1: 0..NB-1
        const int fb    = fi / (NF * GPU_NB); // transverse dim 2: 0..NB-1

        // Face normal coordinate: f0=0 → ghost face (ilo-1), f0=NB → ghost face (ihi)
        const int fn = ilo - 1 + f0;   // NG-1 .. NG+NB-1
        const int ta = ilo + fa;
        const int tb = ilo + fb;

        // Left/right cell flat indices (axis-dependent)
        int idxL, idxR;
        if (axis == 0) {
            idxL = gpu_cell_idx(fn,   ta, tb);
            idxR = gpu_cell_idx(fn+1, ta, tb);
        } else if (axis == 1) {
            idxL = gpu_cell_idx(ta, fn,   tb);
            idxR = gpu_cell_idx(ta, fn+1, tb);
        } else {
            idxL = gpu_cell_idx(ta, tb, fn  );
            idxR = gpu_cell_idx(ta, tb, fn+1);
        }

        const bool bL = (fn   >= ilo);
        const bool bR = (fn+1 <= ihi);
        if (!bL && !bR) continue;  // ghost-ghost face

        GPrim pL, pR;
        load_prim(idxL, pL); load_prim(idxR, pR);

        const double ducL = sp[8*GPU_NCELL+idxL];
        const double ducR = sp[8*GPU_NCELL+idxR];
        const double theta = fmax(ducL, ducR);
        const bool is_bnd = (fn < ilo || fn+1 > ihi);

        // KEP flux
        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, axis, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            // Pure KEP: smooth interior
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            // Check wall face: p_L == p_R (exact) and anti-symmetric tangential vel
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) -> bool {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }

            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                // Interior: WENO5-Z + HLLC-ES
                GPrim qL, qR;
                if (axis == 0) gpu_weno5_face(sp, fn, ta, tb, 0, qL, qR);
                else if (axis == 1) gpu_weno5_face(sp, ta, fn, tb, 1, qL, qR);
                else               gpu_weno5_face(sp, ta, tb, fn, 2, qL, qR);
                gpu_hllc_es_flux(qL, qR, axis, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, axis, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih*F[v]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Viscous face helper — used by k_rhs_visc
// ─────────────────────────────────────────────────────────────────────────────

// Scratch velocity accessor: comp=0→u,1→v,2→w (scratch offsets 1,2,3).
__device__ __forceinline__ static double
sp_vel(const double* __restrict__ sp, int comp, int ii, int jj, int kk) {
    return sp[(comp + 1) * GPU_NCELL + gpu_cell_idx(ii, jj, kk)];
}

// Viscous stress + energy flux at face (ax, dn∈{+1,-1}) of cell (i,j,k).
// t1=(ax+1)%3, t2=(ax+2)%3.
// dn sign convention: grad_nn = ih*dn*(v_nb - v_c) is positive outward for both faces.
// tnn: normal-normal stress; tnt1/tnt2: shear on (ax,t1)/(ax,t2) planes.
// Fe: τ·u + κ∇T at face (energy flux, outward positive).
__device__ __forceinline__ static void face_visc(
    const double* __restrict__ sp,
    int ax, int dn, int i, int j, int k,
    double mu, double kc, double ih, double ihs,
    double& tnn, double& tnt1, double& tnt2, double& Fe)
{
    const int t1 = (ax + 1) % 3, t2 = (ax + 2) % 3;
    const int ni = i + (ax == 0 ? dn : 0);
    const int nj = j + (ax == 1 ? dn : 0);
    const int nk = k + (ax == 2 ? dn : 0);
    const int d1i = (t1 == 0), d1j = (t1 == 1), d1k = (t1 == 2);
    const int d2i = (t2 == 0), d2j = (t2 == 1), d2k = (t2 == 2);

    // Normal velocity gradients at this face (sign absorbed by dn)
    const double dnn  = ih * dn * (sp_vel(sp, ax, ni, nj, nk) - sp_vel(sp, ax, i, j, k));
    const double dtn1 = ih * dn * (sp_vel(sp, t1, ni, nj, nk) - sp_vel(sp, t1, i, j, k));
    const double dtn2 = ih * dn * (sp_vel(sp, t2, ni, nj, nk) - sp_vel(sp, t2, i, j, k));

    // Tangential gradients: face-averaged between neighbor and center cells
    const double d_ax_dt1 = ihs*(sp_vel(sp,ax,ni+d1i,nj+d1j,nk+d1k)-sp_vel(sp,ax,ni-d1i,nj-d1j,nk-d1k)
                                +sp_vel(sp,ax,i +d1i,j +d1j,k +d1k)-sp_vel(sp,ax,i -d1i,j -d1j,k -d1k));
    const double d_ax_dt2 = ihs*(sp_vel(sp,ax,ni+d2i,nj+d2j,nk+d2k)-sp_vel(sp,ax,ni-d2i,nj-d2j,nk-d2k)
                                +sp_vel(sp,ax,i +d2i,j +d2j,k +d2k)-sp_vel(sp,ax,i -d2i,j -d2j,k -d2k));
    const double d_t1_dt1 = ihs*(sp_vel(sp,t1,ni+d1i,nj+d1j,nk+d1k)-sp_vel(sp,t1,ni-d1i,nj-d1j,nk-d1k)
                                +sp_vel(sp,t1,i +d1i,j +d1j,k +d1k)-sp_vel(sp,t1,i -d1i,j -d1j,k -d1k));
    const double d_t2_dt2 = ihs*(sp_vel(sp,t2,ni+d2i,nj+d2j,nk+d2k)-sp_vel(sp,t2,ni-d2i,nj-d2j,nk-d2k)
                                +sp_vel(sp,t2,i +d2i,j +d2j,k +d2k)-sp_vel(sp,t2,i -d2i,j -d2j,k -d2k));

    const double divu = dnn + d_t1_dt1 + d_t2_dt2;
    tnn  = mu * (2.0 * dnn - (2.0 / 3.0) * divu);
    tnt1 = mu * (d_ax_dt1 + dtn1);
    tnt2 = mu * (d_ax_dt2 + dtn2);

    Fe = tnn  * 0.5 * (sp_vel(sp, ax, ni, nj, nk) + sp_vel(sp, ax, i, j, k))
       + tnt1 * 0.5 * (sp_vel(sp, t1, ni, nj, nk) + sp_vel(sp, t1, i, j, k))
       + tnt2 * 0.5 * (sp_vel(sp, t2, ni, nj, nk) + sp_vel(sp, t2, i, j, k))
       + kc * ih * dn * (sp[5*GPU_NCELL + gpu_cell_idx(ni,nj,nk)]
                        - sp[5*GPU_NCELL + gpu_cell_idx(i, j, k)]);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rhs_visc: face-averaged µ viscous divergence (B5 conservative form)
// Grid: (n_leaves)  Block: (GPU_NB, GPU_NB) = 64 threads
// Each thread (di,dj) handles all k ∈ [ilo,ihi]; direct write (no atomics).
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rhs_visc(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp = m.d_scratch;
    double*      rhs = m.d_RHS;
    const int    ilo = GPU_NG;
    const int    ihi = GPU_NG + GPU_NB - 1;
    const double ih  = 1.0 / m.h;
    const double ihs = 0.5 * ih;

    const int i = GPU_NG + threadIdx.x;
    const int j = GPU_NG + threadIdx.y;

    auto MU = [&](int ii,int jj,int kk){ return sp[7*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };

    for (int k = ilo; k <= ihi; ++k) {
        const double mu_c = MU(i,j,k);
        const double mu_p[3] = {
            0.5*(mu_c + MU(i+1,j,  k  )),
            0.5*(mu_c + MU(i,  j+1,k  )),
            0.5*(mu_c + MU(i,  j,  k+1))
        };
        const double mu_m[3] = {
            0.5*(mu_c + MU(i-1,j,  k  )),
            0.5*(mu_c + MU(i,  j-1,k  )),
            0.5*(mu_c + MU(i,  j,  k-1))
        };

        double acc[3] = {0.0, 0.0, 0.0};
        double Fe_acc = 0.0;

        for (int ax = 0; ax < 3; ++ax) {
            const int t1 = (ax + 1) % 3, t2 = (ax + 2) % 3;
            double tnn_p, tnt1_p, tnt2_p, Fe_p;
            double tnn_m, tnt1_m, tnt2_m, Fe_m;
            face_visc(sp, ax, +1, i, j, k,
                      mu_p[ax], mu_p[ax] * GPU_CP / GPU_PR, ih, ihs,
                      tnn_p, tnt1_p, tnt2_p, Fe_p);
            face_visc(sp, ax, -1, i, j, k,
                      mu_m[ax], mu_m[ax] * GPU_CP / GPU_PR, ih, ihs,
                      tnn_m, tnt1_m, tnt2_m, Fe_m);
            acc[ax] += ih * (tnn_p  - tnn_m );
            acc[t1] += ih * (tnt1_p - tnt1_m);
            acc[t2] += ih * (tnt2_p - tnt2_m);
            Fe_acc  += ih * (Fe_p   - Fe_m  );
        }

        const int flat = gpu_cell_idx(i,j,k);
        rhs[1*GPU_NCELL+flat] += acc[0];
        rhs[2*GPU_NCELL+flat] += acc[1];
        rhs[3*GPU_NCELL+flat] += acc[2];
        rhs[4*GPU_NCELL+flat] += Fe_acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuRhsList methods
// ─────────────────────────────────────────────────────────────────────────────

GpuRhsList::~GpuRhsList() {
    if (d_metas)        { cudaFree(d_metas);        d_metas        = nullptr; }
    if (d_scratch_pool) { cudaFree(d_scratch_pool); d_scratch_pool = nullptr; }
    if (d_rhs_pool)     { cudaFree(d_rhs_pool);     d_rhs_pool     = nullptr; }
}

void GpuRhsList::build(const BlockTree& tree, const GpuPool& pool) {
    // Free previous large-pool allocations (d_metas freed inside gpu_upload_meta)
    cudaFree(d_scratch_pool); d_scratch_pool = nullptr;
    cudaFree(d_rhs_pool);     d_rhs_pool     = nullptr;

    const auto& leaves = tree.leaf_indices();
    n_leaves = (int)leaves.size();
    if (n_leaves == 0) return;

    const size_t scratch_bytes = (size_t)SCRATCH_NCOMP * NCELL * n_leaves * sizeof(double);
    const size_t rhs_bytes     = (size_t)NVAR           * NCELL * n_leaves * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_scratch_pool, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs_pool,     rhs_bytes    ));

    std::vector<GpuLeafRhsMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const BlockNode& nd = tree.nodes[leaves[li]];
        GpuLeafRhsMeta& meta = h_metas[li];
        meta.d_Q      = pool.d_Q(nd.block.get());
        meta.d_RHS    = d_rhs_pool     + (size_t)li * NVAR           * NCELL;
        meta.d_scratch= d_scratch_pool + (size_t)li * SCRATCH_NCOMP  * NCELL;
        meta.h        = nd.block->h;
        meta._pad[0]  = meta._pad[1] = meta._pad[2] = meta._pad[3] = 0;
    }

    gpu_upload_meta(d_metas, h_metas);
}

// k_zero_rhs: zero d_rhs_pool as a proper kernel node (graph-capture safe).
// Using a kernel instead of cudaMemsetAsync guarantees explicit stream ordering
// in the captured CUDA graph — no reliance on memset helper streams.
__global__
void k_zero_rhs(double* __restrict__ pool, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        pool[i] = 0.0;
}

void GpuRhsList::exec(cudaStream_t stream, bool zero_rhs) const {
    if (n_leaves == 0) return;

    // Optional zeroing: callers that use cudaMemsetAsync on the same stream
    // BEFORE launching this (e.g. the per-stage graph replay path) pass false
    // to avoid a redundant zero inside any CUDA graph node.
    if (zero_rhs) {
        const int total = NVAR * NCELL * n_leaves;
        const int nblks = (total + 255) / 256;
        k_zero_rhs<<<nblks, 256, 0, stream>>>(d_rhs_pool, total);
    }

    k_prim_duc<<<dim3(n_leaves), dim3(GPU_NB2, GPU_NB2), 0, stream>>>(d_metas);
    k_rhs_conv <<<dim3(n_leaves), dim3(192),              0, stream>>>(d_metas);
    k_rhs_visc <<<dim3(n_leaves), dim3(GPU_NB, GPU_NB),   0, stream>>>(d_metas);
}

void GpuRhsList::download_rhs(const BlockTree& tree) const {
    if (n_leaves == 0) return;
    const auto& leaves = tree.leaf_indices();
    const size_t blk_bytes = (size_t)NVAR * NCELL * sizeof(double);
    static thread_local double h_buf[NVAR * NCELL];
    for (int li = 0; li < n_leaves; ++li) {
        CellBlock* blk = tree.nodes[leaves[li]].block.get();
        if (!blk) continue;
        CUDA_CHECK(cudaMemcpy(h_buf, d_rhs_pool + (size_t)li * NVAR * NCELL,
                              blk_bytes, cudaMemcpyDeviceToHost));
        for (int v = 0; v < NVAR; ++v)
            blk->Q[v].assign_from_flat(h_buf + v * NCELL);
    }
}
