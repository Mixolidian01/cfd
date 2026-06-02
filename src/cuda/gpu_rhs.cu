// gpu_rhs.cu — P8.3: per-block GPU RHS kernels (WENO5-Z + HLLC-ES + viscous)
//
// Three-kernel pipeline per exec() call (one CUDA block per leaf):
//   k_prim_duc  — conservative → prim + µ (Sutherland) + Ducros φ → d_scratch
//   k_rhs_conv  — WENO5-Z/KEP/HLLC-ES face-centred convective flux (atomicAdd)
//   k_rhs_visc  — face-averaged µ viscous divergence (direct write, cell-centred)
//
// GpuRhsList::exec() zeros d_rhs_pool, launches the three kernels in order,
// then returns.  download_rhs() DtoH-copies the RHS back to CellBlock::Q.

#include "cuda/gpu_rhs_recon.cuh"   // reconstruction helpers (weno5z, teno5, teno7, Roe)
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "physics/face_interp.hpp"
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// k_prim_duc: conservative → primitives + µ + Ducros φ → d_scratch
// Grid: (n_leaves)  Block: (GPU_NB2, GPU_NB2) = 144 threads
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_prim_duc(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    int i = threadIdx.x, j = threadIdx.y;

    // ── Pass 1: prim + µ ─────────────────────────────────────────────────────
    // P_FLOOR: the positivity floor (k_positivity_floor) clamps d_Q so that
    // p >= 1e-12 in conserved space.  However, gpu_cons_to_prim recomputes ke
    // from d_Q via a divide-then-multiply pattern that can differ by ~1 ULP
    // from the floor's multiply-then-divide form, leaving p=0 in the scratch.
    // Matching EPS_POS (1e-12) here ensures the scratch pressure is consistent
    // with the conserved-space floor and keeps HLLC-ES (which divides by p
    // via log_mean) from encountering p=0.
    constexpr double P_FLOOR = 1.0e-12;
    for (int k = 0; k < GPU_NB2; ++k) {
        int flat = gpu_cell_idx(i, j, k);
        GPrim q = gpu_cons_to_prim(
            m.d_Q[0*GPU_NCELL+flat], m.d_Q[1*GPU_NCELL+flat],
            m.d_Q[2*GPU_NCELL+flat], m.d_Q[3*GPU_NCELL+flat],
            m.d_Q[4*GPU_NCELL+flat]);
        if (q.p < P_FLOOR) {
            q.p = P_FLOOR;
            q.T = P_FLOOR / (q.rho * GPU_R_GAS);
            q.c = sqrt(GPU_GAMMA * P_FLOOR / q.rho);
        }
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
            const double ih2x = 0.5 / m.hx;
            const double ih2y = 0.5 / m.hy;
            const double ih2z = 0.5 / m.hz;
            auto U = [=](int ii,int jj,int kk){ return sp[1*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto V = [=](int ii,int jj,int kk){ return sp[2*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto W = [=](int ii,int jj,int kk){ return sp[3*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto P = [=](int ii,int jj,int kk){ return sp[4*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };

            const double dudx = ih2x*(U(i+1,j,k)-U(i-1,j,k));
            const double dudy = ih2y*(U(i,j+1,k)-U(i,j-1,k));
            const double dudz = ih2z*(U(i,j,k+1)-U(i,j,k-1));
            const double dvdx = ih2x*(V(i+1,j,k)-V(i-1,j,k));
            const double dvdy = ih2y*(V(i,j+1,k)-V(i,j-1,k));
            const double dvdz = ih2z*(V(i,j,k+1)-V(i,j,k-1));
            const double dwdx = ih2x*(W(i+1,j,k)-W(i-1,j,k));
            const double dwdy = ih2y*(W(i,j+1,k)-W(i,j-1,k));
            const double dwdz = ih2z*(W(i,j,k+1)-W(i,j,k-1));
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
            const double phi_p_cl = fmin(1.0, fmax(0.0, (phi_p-m.duc_p_thr)*m.duc_blend_inv));

            duc = fmax(phi_vel, phi_p_cl);
        }
        m.d_scratch[8*GPU_NCELL + gpu_cell_idx(i,j,k)] = duc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D0.5: shared-memory reconstruction helper for Y and Z faces (WENO5-Z or TENO5-A).
// Reads the 6-point stencil from the pre-loaded i-plane in shared memory.
// AXIS=1 (Y): face between y=fn and y=fn+1, x=xi fixed, z=tb fixed.
// AXIS=2 (Z): face between z=fn and z=fn+1, x=xi fixed, y=tb fixed.
// s layout: s[comp * NB22 + j * GPU_NB2 + k], NB22 = GPU_NB2*GPU_NB2.
// ─────────────────────────────────────────────────────────────────────────────
template<int AXIS, bool USE_TENO>
__device__ __forceinline__
void gpu_recon_shmem(const double* __restrict__ s,
                     int fn, int tb,
                     GPrim& qL_out, GPrim& qR_out) noexcept {
    // Shmem layout (padded): jk = k * PAD + j  where PAD = NB2+1 = 13.
    // Stride-13 is coprime with 32 → zero bank conflicts for Z-stencil reads.
    constexpr int PAD  = GPU_NB2 + 1;         // 13
    constexpr int NB2P = GPU_NB2 * PAD;        // 156 — per-comp shmem stride

    // jk index for stencil cell d ∈ {-2,-1,0,+1,+2,+3}
    auto jkd = [&](int d) -> int {
        if constexpr (AXIS == 1) return tb        * PAD + (fn + d);  // Y: k=tb, j=fn+d
        else                     return (fn + d)  * PAD + tb;         // Z: k=fn+d, j=tb
    };
    auto jkL = jkd(0), jkR = jkd(1);  // left / right cell of the face

    const double rL = s[0*NB2P+jkL], uL = s[1*NB2P+jkL];
    const double vL = s[2*NB2P+jkL], wL = s[3*NB2P+jkL];
    const double pL = s[4*NB2P+jkL], TL = s[5*NB2P+jkL], cL = s[6*NB2P+jkL];
    const double rR = s[0*NB2P+jkR], uR = s[1*NB2P+jkR];
    const double vR = s[2*NB2P+jkR], wR = s[3*NB2P+jkR];
    const double pR = s[4*NB2P+jkR], TR = s[5*NB2P+jkR], cR = s[6*NB2P+jkR];
    const GpuRoeState rs = gpu_roe_from_prim(rL,uL,vL,wL,pL, rR,uR,vR,wR,pR, AXIS);

    double Q[6][GPU_NVAR];
    for (int m = 0; m < 6; ++m) {
        const int jk = jkd(m - 2);
        const double rho = s[0*NB2P+jk], u = s[1*NB2P+jk];
        const double v   = s[2*NB2P+jk], w = s[3*NB2P+jk];
        const double p   = s[4*NB2P+jk];
        Q[m][0] = rho; Q[m][1] = rho*u; Q[m][2] = rho*v; Q[m][3] = rho*w;
        Q[m][4] = p/(GPU_GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }

    double W[5][6];
    for (int m = 0; m < 6; ++m) { double Wm[5]; gpu_char_proj(Q[m], rs, Wm); for (int c=0;c<5;++c) W[c][m]=Wm[c]; }

    double wL_w[5], wR_w[5];
    for (int kk = 0; kk < 5; ++kk)
        if constexpr (USE_TENO)
            physics_teno5_scalar(W[kk][0],W[kk][1],W[kk][2],W[kk][3],W[kk][4],W[kk][5], wL_w[kk], wR_w[kk]);
        else
            gpu_weno5z_scalar(W[kk][0],W[kk][1],W[kk][2],W[kk][3],W[kk][4],W[kk][5], wL_w[kk], wR_w[kk]);

    double QL[GPU_NVAR], QR[GPU_NVAR];
    gpu_back_proj(wL_w, rs, QL);
    gpu_back_proj(wR_w, rs, QR);
    GPrim fbL; fbL.rho=rL; fbL.u=uL; fbL.v=vL; fbL.w=wL; fbL.p=pL; fbL.T=TL; fbL.c=cL;
    GPrim fbR; fbR.rho=rR; fbR.u=uR; fbR.v=vR; fbR.w=wR; fbR.p=pR; fbR.T=TR; fbR.c=cR;
    qL_out = gpu_safe_prim_f(QL, fbL);
    qR_out = gpu_safe_prim_f(QR, fbR);
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
    const double ihx  = 1.0 / m.hx;
    const double ihy  = 1.0 / m.hy;
    const double ihz  = 1.0 / m.hz;
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

        const double ih_face = (axis == 0) ? ihx : (axis == 1) ? ihy : ihz;
        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih_face*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih_face*F[v]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D3: k_rhs_conv_teno<USE_TENO7> — TENO5-A or TENO7-A / KEP / HLLC-ES.
// USE_TENO7=false → TENO5-A (is_bnd ignores periodicity).
// USE_TENO7=true  → TENO7-A (is_bnd suppressed on periodic domains; 7-point stencil).
// ─────────────────────────────────────────────────────────────────────────────
template<bool USE_TENO7>
__global__
void k_rhs_conv_teno(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const int  ilo    = GPU_NG;
    const int  ihi    = GPU_NG + GPU_NB - 1;
    const double ihx  = 1.0 / m.hx;
    const double ihy  = 1.0 / m.hy;
    const double ihz  = 1.0 / m.hz;
    constexpr double kep_thr = 1.0e-8;

    constexpr int NF   = GPU_NB + 1;
    constexpr int FPA  = NF * GPU_NB * GPU_NB;
    constexpr int FTOT = 3 * FPA;

    auto load_prim = [&](int flat, GPrim& q) {
        q.rho = sp[0*GPU_NCELL+flat]; q.u = sp[1*GPU_NCELL+flat];
        q.v   = sp[2*GPU_NCELL+flat]; q.w = sp[3*GPU_NCELL+flat];
        q.p   = sp[4*GPU_NCELL+flat]; q.T = sp[5*GPU_NCELL+flat];
        q.c   = sp[6*GPU_NCELL+flat];
    };

    for (int fid = threadIdx.x; fid < FTOT; fid += blockDim.x) {
        const int axis  = fid / FPA;
        const int fi    = fid % FPA;
        const int f0    = fi % NF;
        const int fa    = (fi / NF) % GPU_NB;
        const int fb    = fi / (NF * GPU_NB);

        const int fn = ilo - 1 + f0;
        const int ta = ilo + fa;
        const int tb = ilo + fb;

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
        if (!bL && !bR) continue;

        GPrim pL, pR;
        load_prim(idxL, pL); load_prim(idxR, pR);

        const double ducL = sp[8*GPU_NCELL+idxL];
        const double ducR = sp[8*GPU_NCELL+idxR];
        const double theta = fmax(ducL, ducR);
        const bool at_bnd = (fn < ilo || fn+1 > ihi);
        const int  face_d = 2*axis + (fn < ilo ? 0 : 1);
        const bool at_cf  = at_bnd && ((m.cf_bnd_mask >> face_d) & 1u);
        const bool is_bnd = USE_TENO7 ? (at_cf || (!m.is_periodic && at_bnd)) : at_bnd;

        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, axis, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
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
                GPrim qL, qR;
                if constexpr (USE_TENO7) {
                    if (axis == 0) gpu_teno7_face(sp, fn, ta, tb, 0, qL, qR);
                    else if (axis == 1) gpu_teno7_face(sp, ta, fn, tb, 1, qL, qR);
                    else               gpu_teno7_face(sp, ta, tb, fn, 2, qL, qR);
                } else {
                    if (axis == 0) gpu_teno5_face(sp, fn, ta, tb, 0, qL, qR);
                    else if (axis == 1) gpu_teno5_face(sp, ta, fn, tb, 1, qL, qR);
                    else               gpu_teno5_face(sp, ta, tb, fn, 2, qL, qR);
                }
                gpu_hllc_es_flux(qL, qR, axis, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, axis, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        const double ih_face = (axis == 0) ? ihx : (axis == 1) ? ihy : ihz;
        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih_face*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih_face*F[v]);
    }
}
template __global__ void k_rhs_conv_teno<false>(const GpuLeafRhsMeta*);
template __global__ void k_rhs_conv_teno<true> (const GpuLeafRhsMeta*);

// ─────────────────────────────────────────────────────────────────────────────
// gpu_face_flux<USE_TENO7>: single-face flux helper for k_rhs_conv_cell.
// fn = left-cell normal index; ta, tb = tangential cell indices; axis = 0/1/2.
// Mirrors k_rhs_conv_teno loop body exactly; result written to F[GPU_NVAR].
// ─────────────────────────────────────────────────────────────────────────────
template<bool USE_TENO7>
__device__ __forceinline__
void gpu_face_flux(const double* __restrict__ sp,
                   int fn, int ta, int tb, int axis,
                   bool is_periodic, uint8_t cf_bnd_mask,
                   double F[GPU_NVAR]) noexcept
{
    const int ilo = GPU_NG, ihi = GPU_NG + GPU_NB - 1;
    constexpr double kep_thr = 1.0e-8;

    int idxL, idxR;
    if      (axis == 0) { idxL = gpu_cell_idx(fn,   ta, tb); idxR = gpu_cell_idx(fn+1, ta, tb); }
    else if (axis == 1) { idxL = gpu_cell_idx(ta, fn,   tb); idxR = gpu_cell_idx(ta, fn+1, tb); }
    else                { idxL = gpu_cell_idx(ta, tb, fn  ); idxR = gpu_cell_idx(ta, tb, fn+1); }

    GPrim pL, pR;
    pL.rho=sp[0*GPU_NCELL+idxL]; pL.u=sp[1*GPU_NCELL+idxL]; pL.v=sp[2*GPU_NCELL+idxL];
    pL.w  =sp[3*GPU_NCELL+idxL]; pL.p=sp[4*GPU_NCELL+idxL]; pL.T=sp[5*GPU_NCELL+idxL]; pL.c=sp[6*GPU_NCELL+idxL];
    pR.rho=sp[0*GPU_NCELL+idxR]; pR.u=sp[1*GPU_NCELL+idxR]; pR.v=sp[2*GPU_NCELL+idxR];
    pR.w  =sp[3*GPU_NCELL+idxR]; pR.p=sp[4*GPU_NCELL+idxR]; pR.T=sp[5*GPU_NCELL+idxR]; pR.c=sp[6*GPU_NCELL+idxR];

    const double ducL  = sp[8*GPU_NCELL+idxL];
    const double ducR  = sp[8*GPU_NCELL+idxR];
    const double theta = fmax(ducL, ducR);
    const bool at_bnd  = (fn < ilo || fn+1 > ihi);
    const int  face_d  = 2*axis + (fn < ilo ? 0 : 1);
    const bool at_cf   = at_bnd && ((cf_bnd_mask >> face_d) & 1u);
    const bool is_bnd  = USE_TENO7 ? (at_cf || (!is_periodic && at_bnd)) : at_bnd;

    double Fk[GPU_NVAR];
    gpu_kep_flux(pL, pR, axis, Fk);

    if (!is_bnd && theta < kep_thr) {
        for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        return;
    }

    double Fs[GPU_NVAR];
    bool wall = is_bnd;
    if (wall) {
        auto antisym = [](double a, double b) {
            return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
        };
        wall = (pL.p == pR.p) && antisym(pL.u,pR.u) && antisym(pL.v,pR.v) && antisym(pL.w,pR.w);
    }
    if (wall) {
        for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
    } else if (!is_bnd) {
        GPrim qL, qR;
        if constexpr (USE_TENO7) {
            if      (axis == 0) gpu_teno7_face(sp, fn, ta, tb, 0, qL, qR);
            else if (axis == 1) gpu_teno7_face(sp, ta, fn, tb, 1, qL, qR);
            else                gpu_teno7_face(sp, ta, tb, fn, 2, qL, qR);
        } else {
            if      (axis == 0) gpu_teno5_face(sp, fn, ta, tb, 0, qL, qR);
            else if (axis == 1) gpu_teno5_face(sp, ta, fn, tb, 1, qL, qR);
            else                gpu_teno5_face(sp, ta, tb, fn, 2, qL, qR);
        }
        gpu_hllc_es_flux(qL, qR, axis, Fs);
    } else {
        gpu_hllc_es_flux(pL, pR, axis, Fs);
    }
    const double th = is_bnd ? 1.0 : theta;
    const double om = 1.0 - th;
    for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rhs_conv_cell<USE_TENO7> — cell-centric convective RHS; no atomics.
// NOT USED in exec() — retained as a reference and for future experimentation.
//
// Intended benefit: eliminates 30 FP64 atomicAdds per cell.
// Measured result: 25× SLOWER than k_rhs_conv_teno on RTX 3070 Laptop.
// Root cause: TENO7 per-face reconstruction inlines Q[7][5]+GpuRoeState 6 times
// per thread, exhausting the register file → massive local-memory spills.
// k_rhs_conv_teno is already FP64-compute-bound at near-peak GFLOPS; the
// atomicAdd overhead is negligible compared to Roe/TENO7 arithmetic.
// See docs/tech_debt.md "k_rhs_conv_cell reverted" for the full analysis.
// ─────────────────────────────────────────────────────────────────────────────
template<bool USE_TENO7>
__global__
void k_rhs_conv_cell(const GpuLeafRhsMeta* __restrict__ metas)
{
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const double  ihx = 1.0 / m.hx;
    const double  ihy = 1.0 / m.hy;
    const double  ihz = 1.0 / m.hz;
    const bool    ip  = (m.is_periodic != 0);
    const uint8_t cfm = m.cf_bnd_mask;

    const int i = GPU_NG + (int)threadIdx.x;
    const int j = GPU_NG + (int)threadIdx.y;

    for (int k = GPU_NG; k < GPU_NG + GPU_NB; ++k) {
        double acc[GPU_NVAR] = {};
        double F[GPU_NVAR];

        // X-left face (i-1 | i): this cell is right → +F/hx
        gpu_face_flux<USE_TENO7>(sp, i-1, j, k, 0, ip, cfm, F);
        for (int v = 0; v < GPU_NVAR; ++v) acc[v] += ihx * F[v];

        // X-right face (i | i+1): this cell is left → -F/hx
        gpu_face_flux<USE_TENO7>(sp, i,   j, k, 0, ip, cfm, F);
        for (int v = 0; v < GPU_NVAR; ++v) acc[v] -= ihx * F[v];

        // Y-left face (j-1 | j): this cell is right → +F/hy
        gpu_face_flux<USE_TENO7>(sp, j-1, i, k, 1, ip, cfm, F);
        for (int v = 0; v < GPU_NVAR; ++v) acc[v] += ihy * F[v];

        // Y-right face (j | j+1): this cell is left → -F/hy
        gpu_face_flux<USE_TENO7>(sp, j,   i, k, 1, ip, cfm, F);
        for (int v = 0; v < GPU_NVAR; ++v) acc[v] -= ihy * F[v];

        // Z-left face (k-1 | k): this cell is right → +F/hz
        gpu_face_flux<USE_TENO7>(sp, k-1, i, j, 2, ip, cfm, F);
        for (int v = 0; v < GPU_NVAR; ++v) acc[v] += ihz * F[v];

        // Z-right face (k | k+1): this cell is left → -F/hz
        gpu_face_flux<USE_TENO7>(sp, k,   i, j, 2, ip, cfm, F);
        for (int v = 0; v < GPU_NVAR; ++v) acc[v] -= ihz * F[v];

        const int flat = gpu_cell_idx(i, j, k);
        for (int v = 0; v < GPU_NVAR; ++v)
            rhs[v * GPU_NCELL + flat] = acc[v];
    }
}
// Not instantiated: exec() uses k_rhs_conv_teno for TENO5A/TENO7A.
// Explicit instantiation would compile the TENO7A path, which spills ~120 doubles
// to local memory and inflates ptxas register usage for the entire translation unit.
// template __global__ void k_rhs_conv_cell<false>(const GpuLeafRhsMeta*);
// template __global__ void k_rhs_conv_cell<true> (const GpuLeafRhsMeta*);

// ─────────────────────────────────────────────────────────────────────────────
// D0.5 — k_rhs_conv_tiled<USE_TENO>: Y/Z faces use i-plane shmem (bank-conflict-free).
// USE_TENO=false → WENO5-Z;  USE_TENO=true → TENO5-A.
//
// Block: 144 = NB2² threads.  Iterates over NB2/2 = 6 pairs of i-planes:
//   1. Load both xi_a and xi_b slices into padded shmem s[2][8][NB2][PAD]
//      (PAD=NB2+1=13; stride-13 coprime with 32 → zero bank conflicts for Z).
//      tid 0..71  handle xi_a;  tid 72..143 handle xi_b.
//   2. Y-face pass: ALL 144 threads active — 72 for xi_a, 72 for xi_b.
//   3. __syncthreads()
//   4. Z-face pass: ALL 144 threads active — 72 for xi_a, 72 for xi_b.
//   5. __syncthreads()
// X-faces (576 total): global memory, 4 iters/thread.
// ─────────────────────────────────────────────────────────────────────────────
template<bool USE_TENO>
__global__
void k_rhs_conv_tiled(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const int     ilo = GPU_NG;
    const int     ihi = GPU_NG + GPU_NB - 1;
    const double  ihx = 1.0 / m.hx;
    const double  ihy = 1.0 / m.hy;
    const double  ihz = 1.0 / m.hz;
    constexpr double kep_thr = 1.0e-8;
    constexpr int NF   = GPU_NB + 1;           // 9 faces along normal axis
    constexpr int NF72 = NF * GPU_NB;           // 72 = faces per axis per xi-plane
    constexpr int PAD  = GPU_NB2 + 1;           // 13 (bank-conflict-free stride)
    constexpr int NB2P = GPU_NB2 * PAD;         // 156 per-comp size in padded shmem

    // Shared: 2 xi-planes × 8 comps × NB2 × PAD = 2496 doubles = 19968 bytes.
    // Comp 7 stores Ducros sensor (from scratch comp 8).
    __shared__ double s[2 * 8 * NB2P];

    const int tid     = threadIdx.x;
    const int xi_sel  = tid / (GPU_NB2 * GPU_NB2 / 2);  // 0 or 1
    const int tid_loc = tid % (GPU_NB2 * GPU_NB2 / 2);  // 0..71 within the group

    // Base pointer into shmem for each xi-plane selection
    auto s_xi = [&](int sel) { return s + sel * 8 * NB2P; };

    // Load GPrim from padded shmem at jk = k*PAD+j
    auto sload = [&](const double* sb, int jk, GPrim& q) {
        q.rho = sb[0*NB2P+jk]; q.u = sb[1*NB2P+jk];
        q.v   = sb[2*NB2P+jk]; q.w = sb[3*NB2P+jk];
        q.p   = sb[4*NB2P+jk]; q.T = sb[5*NB2P+jk];
        q.c   = sb[6*NB2P+jk];
    };

    // Shared flux helper (same logic for Y and Z, templated by axis at callsite)
    auto do_yz_face = [&](const double* sb, int ta, int fn, int tb, int axis) {
        const bool is_y   = (axis == 1);
        // jk = k*PAD+j layout:
        //   Y-face (fn,tb): cell (j=fn,k=tb) → jk = tb*PAD + fn
        //   Z-face (tb,fn): cell (j=tb,k=fn) → jk = fn*PAD + tb
        const int jkL = is_y ? tb * PAD + fn     : fn       * PAD + tb;
        const int jkR = is_y ? tb * PAD + (fn+1) : (fn+1)   * PAD + tb;

        const int idxL = is_y ? gpu_cell_idx(ta, fn,   tb) : gpu_cell_idx(ta, tb, fn  );
        const int idxR = is_y ? gpu_cell_idx(ta, fn+1, tb) : gpu_cell_idx(ta, tb, fn+1);
        const bool bL  = (fn   >= ilo);
        const bool bR  = (fn+1 <= ihi);

        GPrim pL, pR;
        sload(sb, jkL, pL);
        sload(sb, jkR, pR);

        const double ducL  = sb[7*NB2P + jkL];
        const double ducR  = sb[7*NB2P + jkR];
        const double theta = fmax(ducL, ducR);
        const bool is_bnd  = (fn < ilo || fn+1 > ihi);

        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, axis, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }
            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                GPrim qL, qR;
                if (axis == 1) gpu_recon_shmem<1, USE_TENO>(sb, fn, tb, qL, qR);
                else           gpu_recon_shmem<2, USE_TENO>(sb, fn, tb, qL, qR);
                gpu_hllc_es_flux(qL, qR, axis, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, axis, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        const double ih_face = (axis == 1) ? ihy : ihz;
        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih_face*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih_face*F[v]);
    };

    // ── Y/Z faces: loop over pairs of i-planes ────────────────────────────────
    for (int xi_pair = 0; xi_pair < GPU_NB2 / 2; ++xi_pair) {
        const int xi_a = 2 * xi_pair;
        const int xi_b = xi_a + 1;
        const int xi   = (xi_sel == 0) ? xi_a : xi_b;

        // Load: 2 iters per thread (tid_loc covers 0..71; NB2²=144 cells total).
        {
            double* sb = s_xi(xi_sel);
            for (int cell = tid_loc; cell < GPU_NB2 * GPU_NB2; cell += NF72) {
                const int j  = cell % GPU_NB2;
                const int k  = cell / GPU_NB2;
                const int jk = k * PAD + j;
                const int flat = gpu_cell_idx(xi, j, k);
                sb[0*NB2P+jk] = sp[0*GPU_NCELL+flat];
                sb[1*NB2P+jk] = sp[1*GPU_NCELL+flat];
                sb[2*NB2P+jk] = sp[2*GPU_NCELL+flat];
                sb[3*NB2P+jk] = sp[3*GPU_NCELL+flat];
                sb[4*NB2P+jk] = sp[4*GPU_NCELL+flat];
                sb[5*NB2P+jk] = sp[5*GPU_NCELL+flat];
                sb[6*NB2P+jk] = sp[6*GPU_NCELL+flat];
                sb[7*NB2P+jk] = sp[8*GPU_NCELL+flat];  // Ducros at scratch[8]
            }
        }
        __syncthreads();

        // Y-face pass: all 144 threads — each does 1 Y-face for its xi-plane
        {
            const int fn_rel = tid_loc % NF;
            const int tb_rel = tid_loc / NF;
            const int fn     = ilo - 1 + fn_rel;
            const int tb     = ilo + tb_rel;
            do_yz_face(s_xi(xi_sel), xi, fn, tb, 1);
        }
        __syncthreads();  // all Y atomicAdds complete before Z starts

        // Z-face pass: all 144 threads — each does 1 Z-face for its xi-plane
        {
            const int fn_rel = tid_loc % NF;
            const int tb_rel = tid_loc / NF;
            const int fn     = ilo - 1 + fn_rel;
            const int tb     = ilo + tb_rel;
            do_yz_face(s_xi(xi_sel), xi, fn, tb, 2);
        }
        __syncthreads();  // before next pair load
    }

    // ── X-faces: global memory, stride-1 (4 iters/thread) ───────────────────
    constexpr int FX = NF * GPU_NB * GPU_NB;  // 576
    auto gload = [&](int flat, GPrim& q) {
        q.rho = sp[0*GPU_NCELL+flat]; q.u = sp[1*GPU_NCELL+flat];
        q.v   = sp[2*GPU_NCELL+flat]; q.w = sp[3*GPU_NCELL+flat];
        q.p   = sp[4*GPU_NCELL+flat]; q.T = sp[5*GPU_NCELL+flat];
        q.c   = sp[6*GPU_NCELL+flat];
    };
    for (int fid = tid; fid < FX; fid += blockDim.x) {
        const int fn_rel = fid % NF;
        const int fa     = (fid / NF) % GPU_NB;
        const int fb     = fid / (NF * GPU_NB);
        const int fn     = ilo - 1 + fn_rel;
        const int ta     = ilo + fa;
        const int tb     = ilo + fb;

        const int idxL = gpu_cell_idx(fn,   ta, tb);
        const int idxR = gpu_cell_idx(fn+1, ta, tb);
        const bool bL  = (fn   >= ilo);
        const bool bR  = (fn+1 <= ihi);

        GPrim pL, pR;
        gload(idxL, pL); gload(idxR, pR);

        const double theta = fmax(sp[8*GPU_NCELL+idxL], sp[8*GPU_NCELL+idxR]);
        const bool is_bnd  = (fn < ilo || fn+1 > ihi);

        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, 0, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }
            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                GPrim qL, qR;
                if constexpr (USE_TENO)
                    gpu_teno5_face(sp, fn, ta, tb, 0, qL, qR);
                else
                    gpu_weno5_face(sp, fn, ta, tb, 0, qL, qR);
                gpu_hllc_es_flux(qL, qR, 0, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, 0, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ihx*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ihx*F[v]);
    }
}

// Explicit instantiations required for __global__ function templates.
template __global__ void k_rhs_conv_tiled<false>(const GpuLeafRhsMeta*);
template __global__ void k_rhs_conv_tiled<true> (const GpuLeafRhsMeta*);

// ─────────────────────────────────────────────────────────────────────────────
// Viscous face helper — used by k_rhs_visc
// ─────────────────────────────────────────────────────────────────────────────

// Scratch velocity accessor: comp=0→u,1→v,2→w (scratch offsets 1,2,3).
__device__ __forceinline__ static double
sp_vel(const double* __restrict__ sp, int comp, int ii, int jj, int kk) {
    return sp[(comp + 1) * GPU_NCELL + gpu_cell_idx(ii, jj, kk)];
}

// Viscous stress + energy flux at face AX±½ of cell (i,j,k).
// AX: compile-time axis (0=X,1=Y,2=Z). dn ∈ {+1,-1}: +½ or -½ face.
// dn sign convention: grad_nn = ih_nn*dn*(v_nb - v_c) is positive outward.
// tnn: normal-normal stress; tnt1/tnt2: shear on (AX,t1)/(AX,t2) planes.
// Fe: τ·u + κ∇T at face (energy flux, outward positive).
// ih_nn: inverse cell size along the face-normal axis (AX).
// ihs_t1: 0.25/h_t1 (quarter-cell tangential factor for axis t1=(AX+1)%3).
// ihs_t2: 0.25/h_t2 (quarter-cell tangential factor for axis t2=(AX+2)%3).
template<int AX>
__device__ __forceinline__ static void face_visc(
    const double* __restrict__ sp,
    int dn, int i, int j, int k,
    double mu, double kc, double ih_nn, double ihs_t1, double ihs_t2,
    double& tnn, double& tnt1, double& tnt2, double& Fe)
{
    constexpr int t1 = (AX + 1) % 3;
    constexpr int t2 = (AX + 2) % 3;
    const int ni = i + (AX == 0 ? dn : 0);
    const int nj = j + (AX == 1 ? dn : 0);
    const int nk = k + (AX == 2 ? dn : 0);
    constexpr int d1i = (t1 == 0), d1j = (t1 == 1), d1k = (t1 == 2);
    constexpr int d2i = (t2 == 0), d2j = (t2 == 1), d2k = (t2 == 2);

    // Normal velocity gradients at this face (sign absorbed by dn)
    const double dnn  = ih_nn * dn * (sp_vel(sp, AX, ni, nj, nk) - sp_vel(sp, AX, i, j, k));
    const double dtn1 = ih_nn * dn * (sp_vel(sp, t1, ni, nj, nk) - sp_vel(sp, t1, i, j, k));
    const double dtn2 = ih_nn * dn * (sp_vel(sp, t2, ni, nj, nk) - sp_vel(sp, t2, i, j, k));

    // Tangential gradients: face-averaged between neighbor and center cells
    const double d_ax_dt1 = ihs_t1*(sp_vel(sp,AX,ni+d1i,nj+d1j,nk+d1k)-sp_vel(sp,AX,ni-d1i,nj-d1j,nk-d1k)
                                   +sp_vel(sp,AX,i +d1i,j +d1j,k +d1k)-sp_vel(sp,AX,i -d1i,j -d1j,k -d1k));
    const double d_ax_dt2 = ihs_t2*(sp_vel(sp,AX,ni+d2i,nj+d2j,nk+d2k)-sp_vel(sp,AX,ni-d2i,nj-d2j,nk-d2k)
                                   +sp_vel(sp,AX,i +d2i,j +d2j,k +d2k)-sp_vel(sp,AX,i -d2i,j -d2j,k -d2k));
    const double d_t1_dt1 = ihs_t1*(sp_vel(sp,t1,ni+d1i,nj+d1j,nk+d1k)-sp_vel(sp,t1,ni-d1i,nj-d1j,nk-d1k)
                                   +sp_vel(sp,t1,i +d1i,j +d1j,k +d1k)-sp_vel(sp,t1,i -d1i,j -d1j,k -d1k));
    const double d_t2_dt2 = ihs_t2*(sp_vel(sp,t2,ni+d2i,nj+d2j,nk+d2k)-sp_vel(sp,t2,ni-d2i,nj-d2j,nk-d2k)
                                   +sp_vel(sp,t2,i +d2i,j +d2j,k +d2k)-sp_vel(sp,t2,i -d2i,j -d2j,k -d2k));

    const double divu = dnn + d_t1_dt1 + d_t2_dt2;
    tnn  = mu * (2.0 * dnn - (2.0 / 3.0) * divu);
    tnt1 = mu * (d_ax_dt1 + dtn1);
    tnt2 = mu * (d_ax_dt2 + dtn2);

    // Face velocity: FaceInterp<AX> at the lower-index cell of this face.
    // dn=+1 → lower index = (i,j,k); dn=-1 → lower index = (ni,nj,nk).
    const int li = (AX == 0 ? (dn == 1 ? i : ni) : i);
    const int lj = (AX == 1 ? (dn == 1 ? j : nj) : j);
    const int lk = (AX == 2 ? (dn == 1 ? k : nk) : k);
    auto vel_ax = [sp](int ii,int jj,int kk){ return sp_vel(sp, AX, ii, jj, kk); };
    auto vel_t1 = [sp](int ii,int jj,int kk){ return sp_vel(sp, t1, ii, jj, kk); };
    auto vel_t2 = [sp](int ii,int jj,int kk){ return sp_vel(sp, t2, ii, jj, kk); };
    constexpr FaceInterp<static_cast<Axis>(AX)> fi;
    Fe = tnn  * fi(vel_ax, li, lj, lk)
       + tnt1 * fi(vel_t1, li, lj, lk)
       + tnt2 * fi(vel_t2, li, lj, lk)
       + kc * ih_nn * dn * (sp[5*GPU_NCELL + gpu_cell_idx(ni,nj,nk)]
                           - sp[5*GPU_NCELL + gpu_cell_idx(i, j, k)]);
}

// Accumulate viscous stress divergence for one axis AX into acc[] and Fe_acc.
// ih[3]: per-axis inverse cell sizes {ihx, ihy, ihz}.
// ihs[3]: quarter-cell tangential factors {0.25*ihx, 0.25*ihy, 0.25*ihz}.
template<int AX>
__device__ __forceinline__ static void visc_axis(
    const double* __restrict__ sp,
    const double ih[3], const double ihs[3],
    int i, int j, int k, double mu_p, double mu_m,
    double (&acc)[3], double& Fe_acc)
{
    constexpr int t1 = (AX + 1) % 3;
    constexpr int t2 = (AX + 2) % 3;
    double tnn_p, tnt1_p, tnt2_p, Fe_p;
    double tnn_m, tnt1_m, tnt2_m, Fe_m;
    face_visc<AX>(sp, +1, i, j, k, mu_p, mu_p*GPU_CP/GPU_PR,
                  ih[AX], ihs[t1], ihs[t2],
                  tnn_p, tnt1_p, tnt2_p, Fe_p);
    face_visc<AX>(sp, -1, i, j, k, mu_m, mu_m*GPU_CP/GPU_PR,
                  ih[AX], ihs[t1], ihs[t2],
                  tnn_m, tnt1_m, tnt2_m, Fe_m);
    acc[AX] += ih[AX] * (tnn_p  - tnn_m );
    acc[t1] += ih[AX] * (tnt1_p - tnt1_m);
    acc[t2] += ih[AX] * (tnt2_p - tnt2_m);
    Fe_acc  += ih[AX] * (Fe_p   - Fe_m  );
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
    const double ih[3]  = { 1.0 / m.hx, 1.0 / m.hy, 1.0 / m.hz };
    const double ihs[3] = { 0.25 * ih[0], 0.25 * ih[1], 0.25 * ih[2] };

    const int i = GPU_NG + threadIdx.x;
    const int j = GPU_NG + threadIdx.y;

    auto MU = [sp](int ii,int jj,int kk){ return sp[7*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
    constexpr FaceInterp<Axis::X> fi_x;
    constexpr FaceInterp<Axis::Y> fi_y;
    constexpr FaceInterp<Axis::Z> fi_z;

    for (int k = ilo; k <= ihi; ++k) {
        // µ face averaging via FaceInterp<DIR, ArithmeticMean>
        const double mu_xp = fi_x(MU, i,   j,   k  );
        const double mu_xm = fi_x(MU, i-1, j,   k  );
        const double mu_yp = fi_y(MU, i,   j,   k  );
        const double mu_ym = fi_y(MU, i,   j-1, k  );
        const double mu_zp = fi_z(MU, i,   j,   k  );
        const double mu_zm = fi_z(MU, i,   j,   k-1);

        double acc[3] = {0.0, 0.0, 0.0};
        double Fe_acc = 0.0;
        visc_axis<0>(sp, ih, ihs, i, j, k, mu_xp, mu_xm, acc, Fe_acc);
        visc_axis<1>(sp, ih, ihs, i, j, k, mu_yp, mu_ym, acc, Fe_acc);
        visc_axis<2>(sp, ih, ihs, i, j, k, mu_zp, mu_zm, acc, Fe_acc);

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

    std::vector<int> local;
    for (int idx : tree.leaf_indices())
        if (tree.nodes[idx].has_block()) local.push_back(idx);
    n_leaves = (int)local.size();
    if (n_leaves == 0) return;

    const size_t scratch_bytes = (size_t)SCRATCH_NCOMP * NCELL * n_leaves * sizeof(double);
    const size_t rhs_bytes     = (size_t)NVAR           * NCELL * n_leaves * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_scratch_pool, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs_pool,     rhs_bytes    ));

    std::vector<GpuLeafRhsMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const BlockNode& nd = tree.nodes[local[li]];
        GpuLeafRhsMeta& meta = h_metas[li];
        meta.d_Q          = pool.d_Q(nd.block.get());
        meta.d_RHS        = d_rhs_pool     + (size_t)li * NVAR          * NCELL;
        meta.d_scratch    = d_scratch_pool + (size_t)li * SCRATCH_NCOMP * NCELL;
        meta.hx           = nd.block->h;
        meta.hy           = nd.block->hy;
        meta.hz           = nd.block->hz;
        meta.duc_p_thr    = duc_p_thr_;
        meta.duc_blend_inv= duc_blend_inv_;
        meta.is_periodic  = (tree.is_fully_periodic() && n_leaves == 1) ? 1u : 0u;
        meta.cf_bnd_mask  = 0u;
        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            if (ni >= 0 && tree.nodes[ni].has_block()
                        && tree.nodes[ni].level > nd.level)
                meta.cf_bnd_mask |= (uint8_t)(1u << d);
        }
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

// ─────────────────────────────────────────────────────────────────────────────
// k_body_force: add uniform body-force acceleration (fx, fy, fz) to the RHS.
// Grid: (n_leaves)  Block: 256 flat threads
// Only interior cells [NG..NG+NB)³ are updated; ghost cells are skipped.
// ─────────────────────────────────────────────────────────────────────────────
__global__ static void k_body_force(const GpuLeafRhsMeta* __restrict__ metas,
                                     double fx, double fy, double fz)
{
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    for (int flat = threadIdx.x; flat < GPU_NCELL; flat += blockDim.x) {
        const int k_ = flat / (GPU_NB2 * GPU_NB2);
        const int j_ = (flat / GPU_NB2) % GPU_NB2;
        const int i_ = flat % GPU_NB2;
        if (i_ < GPU_NG || i_ >= GPU_NG + GPU_NB ||
            j_ < GPU_NG || j_ >= GPU_NG + GPU_NB ||
            k_ < GPU_NG || k_ >= GPU_NG + GPU_NB) continue;
        const double rho  = m.d_Q[0 * GPU_NCELL + flat];
        const double rhou = m.d_Q[1 * GPU_NCELL + flat];
        const double rhov = m.d_Q[2 * GPU_NCELL + flat];
        const double rhow = m.d_Q[3 * GPU_NCELL + flat];
        m.d_RHS[1 * GPU_NCELL + flat] += fx * rho;
        m.d_RHS[2 * GPU_NCELL + flat] += fy * rho;
        m.d_RHS[3 * GPU_NCELL + flat] += fz * rho;
        m.d_RHS[4 * GPU_NCELL + flat] += fx * rhou + fy * rhov + fz * rhow;
    }
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
    switch (scheme) {
    case GpuReconScheme::TENO7A:
        k_rhs_conv_teno<true>        <<<dim3(n_leaves), 192, 0, stream>>>(d_metas);
        break;
    case GpuReconScheme::TENO5A:
        k_rhs_conv_teno<false>       <<<dim3(n_leaves), 192, 0, stream>>>(d_metas);
        break;
    case GpuReconScheme::WENO5Z_TILED:
        k_rhs_conv_tiled<false>      <<<dim3(n_leaves), 144, 0, stream>>>(d_metas);
        break;
    case GpuReconScheme::TENO5A_TILED:
        k_rhs_conv_tiled<true>       <<<dim3(n_leaves), 144, 0, stream>>>(d_metas);
        break;
    default:  // WENO5Z
        k_rhs_conv                   <<<dim3(n_leaves), 192, 0, stream>>>(d_metas);
        break;
    }
    k_rhs_visc<<<dim3(n_leaves), dim3(GPU_NB, GPU_NB), 0, stream>>>(d_metas);
    if (force_x_ != 0.0 || force_y_ != 0.0 || force_z_ != 0.0)
        k_body_force<<<dim3(n_leaves), 256, 0, stream>>>(d_metas, force_x_, force_y_, force_z_);
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
