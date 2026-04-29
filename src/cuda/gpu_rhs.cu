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

// WENO5-Z scalar reconstruction (Borges et al. 2008)
__device__ __forceinline__
void gpu_weno5z_scalar(double vm2, double vm1, double v0,
                       double vp1, double vp2, double vp3,
                       double& vL, double& vR) noexcept {
    constexpr double eps = 1.0e-36;
    constexpr double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    // ── Left state ───────────────────────────────────────────────────────────
    const double L0 = ( 2.0*vm2 -  7.0*vm1 + 11.0*v0 ) * (1.0/6.0);
    const double L1 = (    -vm1 +  5.0*v0  +  2.0*vp1) * (1.0/6.0);
    const double L2 = ( 2.0*v0  +  5.0*vp1 -      vp2) * (1.0/6.0);
    const double b0L = (13.0/12.0)*(vm2-2.0*vm1+v0)*(vm2-2.0*vm1+v0)
                     +  (1.0/ 4.0)*(vm2-4.0*vm1+3.0*v0)*(vm2-4.0*vm1+3.0*v0);
    const double b1L = (13.0/12.0)*(vm1-2.0*v0+vp1)*(vm1-2.0*v0+vp1)
                     +  (1.0/ 4.0)*(vm1-vp1)*(vm1-vp1);
    const double b2L = (13.0/12.0)*(v0-2.0*vp1+vp2)*(v0-2.0*vp1+vp2)
                     +  (1.0/ 4.0)*(3.0*v0-4.0*vp1+vp2)*(3.0*v0-4.0*vp1+vp2);
    const double tau5L = fabs(b0L - b2L);
    const double a0L = d0*(1.0+(tau5L/(b0L+eps))*(tau5L/(b0L+eps)));
    const double a1L = d1*(1.0+(tau5L/(b1L+eps))*(tau5L/(b1L+eps)));
    const double a2L = d2*(1.0+(tau5L/(b2L+eps))*(tau5L/(b2L+eps)));
    vL = (a0L*L0 + a1L*L1 + a2L*L2) / (a0L + a1L + a2L);

    // ── Right state (mirrored stencil) ───────────────────────────────────────
    const double R0 = ( 2.0*vp3 -  7.0*vp2 + 11.0*vp1) * (1.0/6.0);
    const double R1 = (    -vp2 +  5.0*vp1 +  2.0*v0 ) * (1.0/6.0);
    const double R2 = ( 2.0*vp1 +  5.0*v0  -      vm1) * (1.0/6.0);
    const double b0R = (13.0/12.0)*(vp1-2.0*vp2+vp3)*(vp1-2.0*vp2+vp3)
                     +  (1.0/ 4.0)*(3.0*vp1-4.0*vp2+vp3)*(3.0*vp1-4.0*vp2+vp3);
    const double b1R = (13.0/12.0)*(v0-2.0*vp1+vp2)*(v0-2.0*vp1+vp2)
                     +  (1.0/ 4.0)*(v0-vp2)*(v0-vp2);
    const double b2R = (13.0/12.0)*(vm1-2.0*v0+vp1)*(vm1-2.0*v0+vp1)
                     +  (1.0/ 4.0)*(vm1-4.0*v0+3.0*vp1)*(vm1-4.0*v0+3.0*vp1);
    const double tau5R = fabs(b0R - b2R);
    const double a0R = d0*(1.0+(tau5R/(b0R+eps))*(tau5R/(b0R+eps)));
    const double a1R = d1*(1.0+(tau5R/(b1R+eps))*(tau5R/(b1R+eps)));
    const double a2R = d2*(1.0+(tau5R/(b2R+eps))*(tau5R/(b2R+eps)));
    vR = (a0R*R0 + a1R*R1 + a2R*R2) / (a0R + a1R + a2R);
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
    const double ihs = 0.5 * ih;   // 1/(2h) for face tangential gradients

    const int i = GPU_NG + threadIdx.x;
    const int j = GPU_NG + threadIdx.y;

    auto UF = [&](int ii,int jj,int kk) { return sp[1*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
    auto VF = [&](int ii,int jj,int kk) { return sp[2*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
    auto WF = [&](int ii,int jj,int kk) { return sp[3*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
    auto TF = [&](int ii,int jj,int kk) { return sp[5*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
    auto MU = [&](int ii,int jj,int kk) { return sp[7*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };

    for (int k = ilo; k <= ihi; ++k) {
        // Face-averaged µ at 6 faces of cell (i,j,k)
        const double mu_xp = 0.5*(MU(i,j,k)+MU(i+1,j,  k  ));
        const double mu_xm = 0.5*(MU(i,j,k)+MU(i-1,j,  k  ));
        const double mu_yp = 0.5*(MU(i,j,k)+MU(i,  j+1,k  ));
        const double mu_ym = 0.5*(MU(i,j,k)+MU(i,  j-1,k  ));
        const double mu_zp = 0.5*(MU(i,j,k)+MU(i,  j,  k+1));
        const double mu_zm = 0.5*(MU(i,j,k)+MU(i,  j,  k-1));

        // ── x±½ velocity gradients ───────────────────────────────────────────
        const double dudx_xp = ih *(UF(i+1,j,k)-UF(i,j,k));
        const double dvdx_xp = ih *(VF(i+1,j,k)-VF(i,j,k));
        const double dwdx_xp = ih *(WF(i+1,j,k)-WF(i,j,k));
        const double dudy_xp = ihs*(UF(i+1,j+1,k)-UF(i+1,j-1,k)+UF(i,j+1,k)-UF(i,j-1,k));
        const double dudz_xp = ihs*(UF(i+1,j,k+1)-UF(i+1,j,k-1)+UF(i,j,k+1)-UF(i,j,k-1));
        const double dvdy_xp = ihs*(VF(i+1,j+1,k)-VF(i+1,j-1,k)+VF(i,j+1,k)-VF(i,j-1,k));
        const double dwdz_xp = ihs*(WF(i+1,j,k+1)-WF(i+1,j,k-1)+WF(i,j,k+1)-WF(i,j,k-1));
        const double divu_xp = dudx_xp + dvdy_xp + dwdz_xp;

        const double dudx_xm = ih *(UF(i,j,k)-UF(i-1,j,k));
        const double dvdx_xm = ih *(VF(i,j,k)-VF(i-1,j,k));
        const double dwdx_xm = ih *(WF(i,j,k)-WF(i-1,j,k));
        const double dudy_xm = ihs*(UF(i-1,j+1,k)-UF(i-1,j-1,k)+UF(i,j+1,k)-UF(i,j-1,k));
        const double dudz_xm = ihs*(UF(i-1,j,k+1)-UF(i-1,j,k-1)+UF(i,j,k+1)-UF(i,j,k-1));
        const double dvdy_xm = ihs*(VF(i-1,j+1,k)-VF(i-1,j-1,k)+VF(i,j+1,k)-VF(i,j-1,k));
        const double dwdz_xm = ihs*(WF(i-1,j,k+1)-WF(i-1,j,k-1)+WF(i,j,k+1)-WF(i,j,k-1));
        const double divu_xm = dudx_xm + dvdy_xm + dwdz_xm;

        // ── y±½ velocity gradients ───────────────────────────────────────────
        const double dudy_yp = ih *(UF(i,j+1,k)-UF(i,j,k));
        const double dvdx_yp = ihs*(VF(i+1,j+1,k)-VF(i-1,j+1,k)+VF(i+1,j,k)-VF(i-1,j,k));
        const double dvdy_yp = ih *(VF(i,j+1,k)-VF(i,j,k));
        const double dvdz_yp = ihs*(VF(i,j+1,k+1)-VF(i,j+1,k-1)+VF(i,j,k+1)-VF(i,j,k-1));
        const double dwdy_yp = ih *(WF(i,j+1,k)-WF(i,j,k));
        const double dudx_yp = ihs*(UF(i+1,j+1,k)-UF(i-1,j+1,k)+UF(i+1,j,k)-UF(i-1,j,k));
        const double dwdz_yp = ihs*(WF(i,j+1,k+1)-WF(i,j+1,k-1)+WF(i,j,k+1)-WF(i,j,k-1));
        const double divu_yp = dudx_yp + dvdy_yp + dwdz_yp;

        const double dudy_ym = ih *(UF(i,j,k)-UF(i,j-1,k));
        const double dvdx_ym = ihs*(VF(i+1,j-1,k)-VF(i-1,j-1,k)+VF(i+1,j,k)-VF(i-1,j,k));
        const double dvdy_ym = ih *(VF(i,j,k)-VF(i,j-1,k));
        const double dvdz_ym = ihs*(VF(i,j-1,k+1)-VF(i,j-1,k-1)+VF(i,j,k+1)-VF(i,j,k-1));
        const double dwdy_ym = ih *(WF(i,j,k)-WF(i,j-1,k));
        const double dudx_ym = ihs*(UF(i+1,j-1,k)-UF(i-1,j-1,k)+UF(i+1,j,k)-UF(i-1,j,k));
        const double dwdz_ym = ihs*(WF(i,j-1,k+1)-WF(i,j-1,k-1)+WF(i,j,k+1)-WF(i,j,k-1));
        const double divu_ym = dudx_ym + dvdy_ym + dwdz_ym;

        // ── z±½ velocity gradients ───────────────────────────────────────────
        const double dudz_zp = ih *(UF(i,j,k+1)-UF(i,j,k));
        const double dvdz_zp = ih *(VF(i,j,k+1)-VF(i,j,k));
        const double dwdz_zp = ih *(WF(i,j,k+1)-WF(i,j,k));
        const double dwdx_zp = ihs*(WF(i+1,j,k+1)-WF(i-1,j,k+1)+WF(i+1,j,k)-WF(i-1,j,k));
        const double dwdy_zp = ihs*(WF(i,j+1,k+1)-WF(i,j-1,k+1)+WF(i,j+1,k)-WF(i,j-1,k));
        const double dudx_zp = ihs*(UF(i+1,j,k+1)-UF(i-1,j,k+1)+UF(i+1,j,k)-UF(i-1,j,k));
        const double dvdy_zp = ihs*(VF(i,j+1,k+1)-VF(i,j-1,k+1)+VF(i,j+1,k)-VF(i,j-1,k));
        const double divu_zp = dudx_zp + dvdy_zp + dwdz_zp;

        const double dudz_zm = ih *(UF(i,j,k)-UF(i,j,k-1));
        const double dvdz_zm = ih *(VF(i,j,k)-VF(i,j,k-1));
        const double dwdz_zm = ih *(WF(i,j,k)-WF(i,j,k-1));
        const double dwdx_zm = ihs*(WF(i+1,j,k-1)-WF(i-1,j,k-1)+WF(i+1,j,k)-WF(i-1,j,k));
        const double dwdy_zm = ihs*(WF(i,j+1,k-1)-WF(i,j-1,k-1)+WF(i,j+1,k)-WF(i,j-1,k));
        const double dudx_zm = ihs*(UF(i+1,j,k-1)-UF(i-1,j,k-1)+UF(i+1,j,k)-UF(i-1,j,k));
        const double dvdy_zm = ihs*(VF(i,j+1,k-1)-VF(i,j-1,k-1)+VF(i,j+1,k)-VF(i,j-1,k));
        const double divu_zm = dudx_zm + dvdy_zm + dwdz_zm;

        // ── Face stresses ────────────────────────────────────────────────────
        const double txx_xp = mu_xp*(2.0*dudx_xp-(2.0/3.0)*divu_xp);
        const double txy_xp = mu_xp*(dudy_xp+dvdx_xp);
        const double txz_xp = mu_xp*(dudz_xp+dwdx_xp);
        const double txx_xm = mu_xm*(2.0*dudx_xm-(2.0/3.0)*divu_xm);
        const double txy_xm = mu_xm*(dudy_xm+dvdx_xm);
        const double txz_xm = mu_xm*(dudz_xm+dwdx_xm);

        const double tyx_yp = mu_yp*(dudy_yp+dvdx_yp);
        const double tyy_yp = mu_yp*(2.0*dvdy_yp-(2.0/3.0)*divu_yp);
        const double tyz_yp = mu_yp*(dvdz_yp+dwdy_yp);
        const double tyx_ym = mu_ym*(dudy_ym+dvdx_ym);
        const double tyy_ym = mu_ym*(2.0*dvdy_ym-(2.0/3.0)*divu_ym);
        const double tyz_ym = mu_ym*(dvdz_ym+dwdy_ym);

        const double tzx_zp = mu_zp*(dudz_zp+dwdx_zp);
        const double tzy_zp = mu_zp*(dvdz_zp+dwdy_zp);
        const double tzz_zp = mu_zp*(2.0*dwdz_zp-(2.0/3.0)*divu_zp);
        const double tzx_zm = mu_zm*(dudz_zm+dwdx_zm);
        const double tzy_zm = mu_zm*(dvdz_zm+dwdy_zm);
        const double tzz_zm = mu_zm*(2.0*dwdz_zm-(2.0/3.0)*divu_zm);

        // ── Momentum divergences ─────────────────────────────────────────────
        const double ax = ih*((txx_xp-txx_xm)+(tyx_yp-tyx_ym)+(tzx_zp-tzx_zm));
        const double ay = ih*((txy_xp-txy_xm)+(tyy_yp-tyy_ym)+(tzy_zp-tzy_zm));
        const double az = ih*((txz_xp-txz_xm)+(tyz_yp-tyz_ym)+(tzz_zp-tzz_zm));

        // ── Energy: conservative face-flux form div(τ·u + κ∇T) ─────────────
        const double Uc = UF(i,j,k), Vc = VF(i,j,k), Wc = WF(i,j,k);
        const double kxp = mu_xp*GPU_CP/GPU_PR, kxm = mu_xm*GPU_CP/GPU_PR;
        const double kyp = mu_yp*GPU_CP/GPU_PR, kym = mu_ym*GPU_CP/GPU_PR;
        const double kzp = mu_zp*GPU_CP/GPU_PR, kzm = mu_zm*GPU_CP/GPU_PR;

        const double Fex_p = txx_xp*0.5*(UF(i+1,j,k)+Uc) + txy_xp*0.5*(VF(i+1,j,k)+Vc)
                           + txz_xp*0.5*(WF(i+1,j,k)+Wc) + kxp*ih*(TF(i+1,j,k)-TF(i,j,k));
        const double Fex_m = txx_xm*0.5*(UF(i-1,j,k)+Uc) + txy_xm*0.5*(VF(i-1,j,k)+Vc)
                           + txz_xm*0.5*(WF(i-1,j,k)+Wc) + kxm*ih*(TF(i,j,k)-TF(i-1,j,k));
        const double Fey_p = tyx_yp*0.5*(UF(i,j+1,k)+Uc) + tyy_yp*0.5*(VF(i,j+1,k)+Vc)
                           + tyz_yp*0.5*(WF(i,j+1,k)+Wc) + kyp*ih*(TF(i,j+1,k)-TF(i,j,k));
        const double Fey_m = tyx_ym*0.5*(UF(i,j-1,k)+Uc) + tyy_ym*0.5*(VF(i,j-1,k)+Vc)
                           + tyz_ym*0.5*(WF(i,j-1,k)+Wc) + kym*ih*(TF(i,j,k)-TF(i,j-1,k));
        const double Fez_p = tzx_zp*0.5*(UF(i,j,k+1)+Uc) + tzy_zp*0.5*(VF(i,j,k+1)+Vc)
                           + tzz_zp*0.5*(WF(i,j,k+1)+Wc) + kzp*ih*(TF(i,j,k+1)-TF(i,j,k));
        const double Fez_m = tzx_zm*0.5*(UF(i,j,k-1)+Uc) + tzy_zm*0.5*(VF(i,j,k-1)+Vc)
                           + tzz_zm*0.5*(WF(i,j,k-1)+Wc) + kzm*ih*(TF(i,j,k)-TF(i,j,k-1));

        const int flat = gpu_cell_idx(i,j,k);
        rhs[1*GPU_NCELL+flat] += ax;
        rhs[2*GPU_NCELL+flat] += ay;
        rhs[3*GPU_NCELL+flat] += az;
        rhs[4*GPU_NCELL+flat] += ih*(Fex_p-Fex_m) + ih*(Fey_p-Fey_m) + ih*(Fez_p-Fez_m);
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

void GpuRhsList::build(const BlockTree& tree) {
    // Free previous allocations
    cudaFree(d_metas);        d_metas        = nullptr;
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
        meta.d_Q      = nd.block->d_Q;
        meta.d_RHS    = d_rhs_pool     + (size_t)li * NVAR           * NCELL;
        meta.d_scratch= d_scratch_pool + (size_t)li * SCRATCH_NCOMP  * NCELL;
        meta.h        = nd.block->h;
        meta._pad[0]  = meta._pad[1] = meta._pad[2] = meta._pad[3] = 0;
    }

    CUDA_CHECK(cudaMalloc(&d_metas, n_leaves * sizeof(GpuLeafRhsMeta)));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas.data(),
                          n_leaves * sizeof(GpuLeafRhsMeta),
                          cudaMemcpyHostToDevice));
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
