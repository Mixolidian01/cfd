// gpu_bn.cu — G6: GPU Baer-Nunziato two-phase solver.
//
// Mirrors BNSolver::advance() on GPU.
// SoA layout: d_Q[v * GPU_NCELL + flat], v=0..6.
//   v=0: α₁ρ₁  v=1: α₂ρ₂  v=2: ρu  v=3: ρv  v=4: ρw  v=5: E  v=6: α₁

#include "cuda/gpu_bn.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_constants.cuh"
#include "models/bn_model.hpp"
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Device structs and helpers
// ─────────────────────────────────────────────────────────────────────────────

struct BNPrim {
    double alpha1;
    double rho1, rho2, rho;
    double u, v, w;
    double p, c_mix;
};

struct BNFlux6 {
    double F[6];
    double s_star;
};

__device__ __forceinline__
BNPrim bn_load_prim(const double* __restrict__ Q, int flat,
                    float g1f, float g2f, float pi1f, float pi2f)
{
    constexpr double eps = 1.0e-14;
    const double g1 = g1f, g2 = g2f, pi1 = pi1f, pi2 = pi2f;
    const double a1r1  = Q[0 * GPU_NCELL + flat];
    const double a2r2  = Q[1 * GPU_NCELL + flat];
    const double rhou  = Q[2 * GPU_NCELL + flat];
    const double rhov  = Q[3 * GPU_NCELL + flat];
    const double rhow  = Q[4 * GPU_NCELL + flat];
    const double E     = Q[5 * GPU_NCELL + flat];
    const double alpha1 = Q[6 * GPU_NCELL + flat];
    const double alpha2 = 1.0 - alpha1;

    BNPrim q;
    q.alpha1 = alpha1;
    q.rho    = a1r1 + a2r2;
    q.u      = rhou / q.rho;
    q.v      = rhov / q.rho;
    q.w      = rhow / q.rho;

    const double KE    = 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
    const double rho_e = E - KE;

    q.rho1 = (alpha1 > eps) ? a1r1 / alpha1 : 0.0;
    q.rho2 = (alpha2 > eps) ? a2r2 / alpha2 : 0.0;

    const double A = alpha1 / (g1 - 1.0);
    const double B = alpha2 / (g2 - 1.0);
    const double C = alpha1 * g1 * pi1 / (g1 - 1.0);
    const double D = alpha2 * g2 * pi2 / (g2 - 1.0);
    const double denom = A + B;
    q.p = (denom > eps) ? (rho_e - C - D) / denom : 0.0;

    const double c1sq = (q.rho1 > eps) ? g1 * (q.p + pi1) / q.rho1 : 0.0;
    const double c2sq = (q.rho2 > eps) ? g2 * (q.p + pi2) / q.rho2 : 0.0;

    double inv = 0.0;
    if (q.rho1 > eps && c1sq > 0.0) inv += alpha1 / (q.rho1 * c1sq);
    if (q.rho2 > eps && c2sq > 0.0) inv += alpha2 / (q.rho2 * c2sq);
    q.c_mix = (q.rho > eps && inv > 0.0) ? sqrt(1.0 / (q.rho * inv)) : 0.0;
    return q;
}

__device__ __forceinline__
double bn_total_energy(const BNPrim& q, float g1f, float g2f, float pi1f, float pi2f)
{
    const double g1 = g1f, g2 = g2f, pi1 = pi1f, pi2 = pi2f;
    const double a2 = 1.0 - q.alpha1;
    const double KE = 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
    const double rho_e = q.alpha1*(q.p + g1*pi1)/(g1-1.0)
                       + a2      *(q.p + g2*pi2)/(g2-1.0);
    return KE + rho_e;
}

__device__ __forceinline__
BNFlux6 bn_hllc_flux(const BNPrim& L, const BNPrim& R, int axis,
                     float g1f, float g2f, float pi1f, float pi2f)
{
    const double g1 = g1f, g2 = g2f, pi1 = pi1f, pi2 = pi2f;
    const double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
    const double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

    const double sqL = sqrt(L.rho), sqR = sqrt(R.rho);
    const double isq = 1.0 / (sqL + sqR);
    const double uh  = (sqL*L.u + sqR*R.u)*isq;
    const double vh  = (sqL*L.v + sqR*R.v)*isq;
    const double wh  = (sqL*L.w + sqR*R.w)*isq;
    const double u_h = (axis==0)?uh:(axis==1)?vh:wh;
    const double gmL = 1.0 + 1.0/(L.alpha1/(g1-1.0)+(1.0-L.alpha1)/(g2-1.0));
    const double gmR = 1.0 + 1.0/(R.alpha1/(g1-1.0)+(1.0-R.alpha1)/(g2-1.0));
    const double gm_h = 0.5*(gmL + gmR);
    const double EL = bn_total_energy(L, g1f, g2f, pi1f, pi2f);
    const double ER = bn_total_energy(R, g1f, g2f, pi1f, pi2f);
    const double HL = (EL + L.p) / L.rho;
    const double HR = (ER + R.p) / R.rho;
    const double H_h = (sqL*HL + sqR*HR)*isq;
    const double c2h = (gm_h - 1.0)*(H_h - 0.5*(uh*uh + vh*vh + wh*wh));
    const double ch  = (c2h > 0.0) ? sqrt(c2h) : 0.5*(L.c_mix + R.c_mix);

    const double sL = fmin(uL - L.c_mix, u_h - ch);
    const double sR = fmax(uR + R.c_mix, u_h + ch);

    const double numer = R.p - L.p + L.rho*(sL-uL)*uL - R.rho*(sR-uR)*uR;
    const double denom2 = L.rho*(sL-uL) - R.rho*(sR-uR);
    const double sStar = (fabs(denom2) > 1.0e-300) ? numer/denom2 : 0.5*(uL+uR);

    // Physical flux for a given prim state
    auto phys_F = [&](const BNPrim& q, double E_q) {
        BNFlux6 f;
        const double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        const double a2 = 1.0 - q.alpha1;
        f.F[0] = q.alpha1 * q.rho1 * un;
        f.F[1] = a2       * q.rho2 * un;
        f.F[2] = q.rho * q.u * un + (axis==0 ? q.p : 0.0);
        f.F[3] = q.rho * q.v * un + (axis==1 ? q.p : 0.0);
        f.F[4] = q.rho * q.w * un + (axis==2 ? q.p : 0.0);
        f.F[5] = (E_q + q.p) * un;
        f.s_star = sStar;
        return f;
    };

    auto star_F = [&](const BNPrim& q, double E_q, double sK) {
        const double un  = (axis==0)?q.u:(axis==1)?q.v:q.w;
        const double a2  = 1.0 - q.alpha1;
        const double cff = (sK - un) / (sK - sStar);
        const double a1r1_s = q.alpha1 * q.rho1 * cff;
        const double a2r2_s = a2       * q.rho2 * cff;
        const double rho_s  = q.rho * cff;
        const double rhou_s = rho_s * (axis==0 ? sStar : q.u);
        const double rhov_s = rho_s * (axis==1 ? sStar : q.v);
        const double rhow_s = rho_s * (axis==2 ? sStar : q.w);
        const double E_s    = rho_s * (E_q/q.rho + (sStar-un)*(sStar + q.p/(q.rho*(sK-un))));
        const auto fq = phys_F(q, E_q);
        BNFlux6 f;
        f.F[0] = fq.F[0] + sK*(a1r1_s  - q.alpha1*q.rho1);
        f.F[1] = fq.F[1] + sK*(a2r2_s  - a2*q.rho2);
        f.F[2] = fq.F[2] + sK*(rhou_s  - q.rho*q.u);
        f.F[3] = fq.F[3] + sK*(rhov_s  - q.rho*q.v);
        f.F[4] = fq.F[4] + sK*(rhow_s  - q.rho*q.w);
        f.F[5] = fq.F[5] + sK*(E_s     - E_q);
        f.s_star = sStar;
        return f;
    };

    if      (sL >= 0.0)      return phys_F(L, EL);
    else if (sR <= 0.0)      return phys_F(R, ER);
    else if (sStar >= 0.0)   return star_F(L, EL, sL);
    else                     return star_F(R, ER, sR);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_bn_ghost_fill
// Grid: dim3(n_leaves), Block: dim3(NB2*NB2=144)
// Fills all 6 faces for one leaf using d_nb pointers.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_bn_ghost_fill(const GpuBnLeafMeta* __restrict__ metas)
{
    const GpuBnLeafMeta& m = metas[blockIdx.x];
    double* Q = m.d_Q;
    const int tid = threadIdx.x;  // 0..143
    const int j_  = tid % GPU_NB2;
    const int k_  = tid / GPU_NB2;

    // ── X-faces ──
    for (int v = 0; v < GPU_BN_NVAR; ++v) {
        const double* nbXM = m.d_nb[0];
        const double* nbXP = m.d_nb[1];
        for (int g = 0; g < GPU_NG; ++g) {
            // XMINUS ghost: dst=(g, j_, k_), src=(NB+g, j_, k_) from nbXM
            const int dst0 = gpu_cell_idx(g,               j_, k_);
            const int src0 = gpu_cell_idx(GPU_NB + g,      j_, k_);
            Q[v * GPU_NCELL + dst0] = nbXM
                ? nbXM[v * GPU_NCELL + src0]
                : Q[v * GPU_NCELL + gpu_cell_idx(GPU_NG, j_, k_)];

            // XPLUS ghost: dst=(NB+NG+g, j_, k_), src=(NG+g, j_, k_) from nbXP
            const int dst1 = gpu_cell_idx(GPU_NB + GPU_NG + g, j_, k_);
            const int src1 = gpu_cell_idx(GPU_NG + g,          j_, k_);
            Q[v * GPU_NCELL + dst1] = nbXP
                ? nbXP[v * GPU_NCELL + src1]
                : Q[v * GPU_NCELL + gpu_cell_idx(GPU_NB + GPU_NG - 1, j_, k_)];
        }
    }
    __syncthreads();

    // ── Y-faces: reinterpret tid as (i, k) ──
    const int i_y = tid % GPU_NB2;
    const int k_y = tid / GPU_NB2;
    for (int v = 0; v < GPU_BN_NVAR; ++v) {
        const double* nbYM = m.d_nb[2];
        const double* nbYP = m.d_nb[3];
        for (int g = 0; g < GPU_NG; ++g) {
            const int dst0 = gpu_cell_idx(i_y, g,               k_y);
            const int src0 = gpu_cell_idx(i_y, GPU_NB + g,      k_y);
            Q[v * GPU_NCELL + dst0] = nbYM
                ? nbYM[v * GPU_NCELL + src0]
                : Q[v * GPU_NCELL + gpu_cell_idx(i_y, GPU_NG, k_y)];

            const int dst1 = gpu_cell_idx(i_y, GPU_NB + GPU_NG + g, k_y);
            const int src1 = gpu_cell_idx(i_y, GPU_NG + g,          k_y);
            Q[v * GPU_NCELL + dst1] = nbYP
                ? nbYP[v * GPU_NCELL + src1]
                : Q[v * GPU_NCELL + gpu_cell_idx(i_y, GPU_NB + GPU_NG - 1, k_y)];
        }
    }
    __syncthreads();

    // ── Z-faces: reinterpret tid as (i, j) ──
    const int i_z = tid % GPU_NB2;
    const int j_z = tid / GPU_NB2;
    for (int v = 0; v < GPU_BN_NVAR; ++v) {
        const double* nbZM = m.d_nb[4];
        const double* nbZP = m.d_nb[5];
        for (int g = 0; g < GPU_NG; ++g) {
            const int dst0 = gpu_cell_idx(i_z, j_z, g              );
            const int src0 = gpu_cell_idx(i_z, j_z, GPU_NB + g     );
            Q[v * GPU_NCELL + dst0] = nbZM
                ? nbZM[v * GPU_NCELL + src0]
                : Q[v * GPU_NCELL + gpu_cell_idx(i_z, j_z, GPU_NG)];

            const int dst1 = gpu_cell_idx(i_z, j_z, GPU_NB + GPU_NG + g);
            const int src1 = gpu_cell_idx(i_z, j_z, GPU_NG + g         );
            Q[v * GPU_NCELL + dst1] = nbZP
                ? nbZP[v * GPU_NCELL + src1]
                : Q[v * GPU_NCELL + gpu_cell_idx(i_z, j_z, GPU_NB + GPU_NG - 1)];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// k_bn_rhs
// Grid: n_leaves blocks, Block: NB^3=512 threads.
// Cell-centred: each thread computes flux divergence for one interior cell.
// RHS must be zeroed before launch.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_bn_rhs(const GpuBnLeafMeta* __restrict__ metas)
{
    const GpuBnLeafMeta& m = metas[blockIdx.x];
    const double* Q  = m.d_Q;
    double*       RHS = m.d_RHS;

    const int tid = threadIdx.x;
    const int ii = tid % GPU_NB + GPU_NG;
    const int jj = (tid / GPU_NB) % GPU_NB + GPU_NG;
    const int kk = tid / (GPU_NB * GPU_NB) + GPU_NG;
    const int fc = gpu_cell_idx(ii, jj, kk);
    const double ih = 1.0 / (double)m.h;

    const BNPrim pc = bn_load_prim(Q, fc, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
    const double a1c = Q[6 * GPU_NCELL + fc];

    double rhs_v[GPU_BN_NVAR];
    for (int v = 0; v < GPU_BN_NVAR; ++v) rhs_v[v] = 0.0;

    // ── X i-1/2: left face — this cell is R ──
    {
        const int fl = gpu_cell_idx(ii-1, jj, kk);
        const BNPrim pL = bn_load_prim(Q, fl, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        const BNFlux6 res = bn_hllc_flux(pL, pc, 0, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        for (int v = 0; v < 6; ++v) rhs_v[v] += ih * res.F[v];
        const double a1up = (res.s_star >= 0.0) ? Q[6*GPU_NCELL + fl] : a1c;
        rhs_v[6] += ih * res.s_star * (a1up - a1c);
    }
    // ── X i+1/2: right face — this cell is L ──
    {
        const int fr = gpu_cell_idx(ii+1, jj, kk);
        const BNPrim pR = bn_load_prim(Q, fr, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        const BNFlux6 res = bn_hllc_flux(pc, pR, 0, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        for (int v = 0; v < 6; ++v) rhs_v[v] -= ih * res.F[v];
        const double a1up = (res.s_star >= 0.0) ? a1c : Q[6*GPU_NCELL + fr];
        rhs_v[6] += ih * res.s_star * (a1c - a1up);
    }
    // ── Y j-1/2 ──
    {
        const int fl = gpu_cell_idx(ii, jj-1, kk);
        const BNPrim pL = bn_load_prim(Q, fl, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        const BNFlux6 res = bn_hllc_flux(pL, pc, 1, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        for (int v = 0; v < 6; ++v) rhs_v[v] += ih * res.F[v];
        const double a1up = (res.s_star >= 0.0) ? Q[6*GPU_NCELL + fl] : a1c;
        rhs_v[6] += ih * res.s_star * (a1up - a1c);
    }
    // ── Y j+1/2 ──
    {
        const int fr = gpu_cell_idx(ii, jj+1, kk);
        const BNPrim pR = bn_load_prim(Q, fr, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        const BNFlux6 res = bn_hllc_flux(pc, pR, 1, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        for (int v = 0; v < 6; ++v) rhs_v[v] -= ih * res.F[v];
        const double a1up = (res.s_star >= 0.0) ? a1c : Q[6*GPU_NCELL + fr];
        rhs_v[6] += ih * res.s_star * (a1c - a1up);
    }
    // ── Z k-1/2 ──
    {
        const int fl = gpu_cell_idx(ii, jj, kk-1);
        const BNPrim pL = bn_load_prim(Q, fl, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        const BNFlux6 res = bn_hllc_flux(pL, pc, 2, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        for (int v = 0; v < 6; ++v) rhs_v[v] += ih * res.F[v];
        const double a1up = (res.s_star >= 0.0) ? Q[6*GPU_NCELL + fl] : a1c;
        rhs_v[6] += ih * res.s_star * (a1up - a1c);
    }
    // ── Z k+1/2 ──
    {
        const int fr = gpu_cell_idx(ii, jj, kk+1);
        const BNPrim pR = bn_load_prim(Q, fr, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        const BNFlux6 res = bn_hllc_flux(pc, pR, 2, m.gamma1, m.gamma2, m.pinf1, m.pinf2);
        for (int v = 0; v < 6; ++v) rhs_v[v] -= ih * res.F[v];
        const double a1up = (res.s_star >= 0.0) ? a1c : Q[6*GPU_NCELL + fr];
        rhs_v[6] += ih * res.s_star * (a1c - a1up);
    }

    for (int v = 0; v < GPU_BN_NVAR; ++v)
        RHS[v * GPU_NCELL + fc] = rhs_v[v];
}

// ─────────────────────────────────────────────────────────────────────────────
// SSP-RK3 update kernels
// Grid: n_leaves, Block: GPU_BN_NVAR * (NB^3/GPU_BN_NVAR) ~ 512 threads
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_bn_save_qn(const GpuBnLeafMeta* __restrict__ metas)
{
    const GpuBnLeafMeta& m = metas[blockIdx.x];
    const int flat = threadIdx.x + blockDim.x * (int)blockIdx.y;
    if (flat >= GPU_BN_NVAR * GPU_NCELL) return;
    m.d_Qn[flat] = m.d_Q[flat];
}

// Stage 1: Q = Qn + dt*RHS
__global__ void k_bn_rk3s1(const GpuBnLeafMeta* __restrict__ metas, double dt)
{
    const GpuBnLeafMeta& m = metas[blockIdx.x];
    const int flat = threadIdx.x + blockDim.x * (int)blockIdx.y;
    if (flat >= GPU_BN_NVAR * GPU_NCELL) return;
    // Only update interior cells: check that flat encodes an interior cell
    // v = flat / GPU_NCELL, cell_flat = flat % GPU_NCELL
    const int v    = flat / GPU_NCELL;
    const int cf   = flat % GPU_NCELL;
    const int i_   = cf % GPU_NB2;
    const int j_   = (cf / GPU_NB2) % GPU_NB2;
    const int k_   = cf / (GPU_NB2 * GPU_NB2);
    if (i_ < GPU_NG || i_ >= GPU_NB + GPU_NG) return;
    if (j_ < GPU_NG || j_ >= GPU_NB + GPU_NG) return;
    if (k_ < GPU_NG || k_ >= GPU_NB + GPU_NG) return;
    (void)v;
    m.d_Q[flat] = m.d_Qn[flat] + dt * m.d_RHS[flat];
}

// Stages 2/3: Q = alpha*Qn + beta*(Q + dt*RHS)
__global__ void k_bn_rk3s23(const GpuBnLeafMeta* __restrict__ metas,
                              double dt, double alpha, double beta)
{
    const GpuBnLeafMeta& m = metas[blockIdx.x];
    const int flat = threadIdx.x + blockDim.x * (int)blockIdx.y;
    if (flat >= GPU_BN_NVAR * GPU_NCELL) return;
    const int cf   = flat % GPU_NCELL;
    const int i_   = cf % GPU_NB2;
    const int j_   = (cf / GPU_NB2) % GPU_NB2;
    const int k_   = cf / (GPU_NB2 * GPU_NB2);
    if (i_ < GPU_NG || i_ >= GPU_NB + GPU_NG) return;
    if (j_ < GPU_NG || j_ >= GPU_NB + GPU_NG) return;
    if (k_ < GPU_NG || k_ >= GPU_NB + GPU_NG) return;
    m.d_Q[flat] = alpha * m.d_Qn[flat] + beta * (m.d_Q[flat] + dt * m.d_RHS[flat]);
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuBnList — host-side lifecycle
// ─────────────────────────────────────────────────────────────────────────────

GpuBnList::~GpuBnList()
{
    if (d_metas)    { cudaFree(d_metas);    d_metas    = nullptr; }
    if (d_Q_pool)   { cudaFree(d_Q_pool);   d_Q_pool   = nullptr; }
    if (d_Qn_pool)  { cudaFree(d_Qn_pool);  d_Qn_pool  = nullptr; }
    if (d_RHS_pool) { cudaFree(d_RHS_pool); d_RHS_pool = nullptr; }
}

void GpuBnList::build(double h, int n_leaves_in, const BNEosParams& eos)
{
    n_leaves = n_leaves_in;
    const size_t bytes = (size_t)n_leaves * GPU_BN_NVAR * GPU_NCELL * sizeof(double);

    if (d_Q_pool)   cudaFree(d_Q_pool);
    if (d_Qn_pool)  cudaFree(d_Qn_pool);
    if (d_RHS_pool) cudaFree(d_RHS_pool);
    if (d_metas)    cudaFree(d_metas);

    CUDA_CHECK(cudaMalloc(&d_Q_pool,   bytes));
    CUDA_CHECK(cudaMalloc(&d_Qn_pool,  bytes));
    CUDA_CHECK(cudaMalloc(&d_RHS_pool, bytes));
    CUDA_CHECK(cudaMemset(d_Q_pool,   0, bytes));
    CUDA_CHECK(cudaMemset(d_Qn_pool,  0, bytes));
    CUDA_CHECK(cudaMemset(d_RHS_pool, 0, bytes));

    const size_t stride = (size_t)GPU_BN_NVAR * GPU_NCELL;
    std::vector<GpuBnLeafMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        auto& m = h_metas[li];
        m.d_Q   = d_Q_pool   + li * stride;
        m.d_Qn  = d_Qn_pool  + li * stride;
        m.d_RHS = d_RHS_pool + li * stride;
        // Periodic: all neighbours point to self
        for (int f = 0; f < 6; ++f) m.d_nb[f] = m.d_Q;
        m.h      = (float)h;
        m.gamma1 = (float)eos.gamma1;
        m.gamma2 = (float)eos.gamma2;
        m.pinf1  = (float)eos.pinf1;
        m.pinf2  = (float)eos.pinf2;
        for (int f = 0; f < 6; ++f) m.bc_type[f] = 0;
        m._pad[0] = m._pad[1] = 0;
    }

    CUDA_CHECK(cudaMalloc(&d_metas, n_leaves * sizeof(GpuBnLeafMeta)));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas.data(),
                          n_leaves * sizeof(GpuBnLeafMeta),
                          cudaMemcpyHostToDevice));
}

void GpuBnList::upload(const std::vector<BNCellBlock>& blocks)
{
    const size_t stride = (size_t)GPU_BN_NVAR * GPU_NCELL;
    std::vector<double> buf(stride);
    for (int li = 0; li < n_leaves && li < (int)blocks.size(); ++li) {
        const BNCellBlock& blk = blocks[li];
        for (int v = 0; v < GPU_BN_NVAR; ++v)
            for (int flat = 0; flat < GPU_NCELL; ++flat)
                buf[v * GPU_NCELL + flat] = blk.Q[v][flat];
        CUDA_CHECK(cudaMemcpy(d_Q_pool + li * stride, buf.data(),
                              stride * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void GpuBnList::download(std::vector<BNCellBlock>& blocks) const
{
    const size_t stride = (size_t)GPU_BN_NVAR * GPU_NCELL;
    std::vector<double> buf(stride);
    for (int li = 0; li < n_leaves && li < (int)blocks.size(); ++li) {
        CUDA_CHECK(cudaMemcpy(buf.data(), d_Q_pool + li * stride,
                              stride * sizeof(double), cudaMemcpyDeviceToHost));
        BNCellBlock& blk = blocks[li];
        for (int v = 0; v < GPU_BN_NVAR; ++v)
            for (int flat = 0; flat < GPU_NCELL; ++flat)
                blk.Q[v][flat] = buf[v * GPU_NCELL + flat];
    }
}

void GpuBnList::advance(double dt, cudaStream_t stream)
{
    const size_t rhs_bytes = (size_t)n_leaves * GPU_BN_NVAR * GPU_NCELL * sizeof(double);

    // Block dims for save/update kernels: 512 threads cover NB^3 cells per var
    // Use a 2D grid: gridX=n_leaves, gridY covers GPU_BN_NVAR*GPU_NCELL / 512 chunks
    constexpr int TPB = 512;
    const int chunks = (GPU_BN_NVAR * GPU_NCELL + TPB - 1) / TPB;
    dim3 grid_update(n_leaves, chunks);

    // Stage 1
    k_bn_save_qn<<<grid_update, TPB, 0, stream>>>(d_metas);
    CUDA_CHECK(cudaMemsetAsync(d_RHS_pool, 0, rhs_bytes, stream));
    k_bn_ghost_fill<<<n_leaves, GPU_NB2 * GPU_NB2, 0, stream>>>(d_metas);
    k_bn_rhs<<<n_leaves, GPU_NB * GPU_NB * GPU_NB, 0, stream>>>(d_metas);
    k_bn_rk3s1<<<grid_update, TPB, 0, stream>>>(d_metas, dt);

    // Stage 2
    CUDA_CHECK(cudaMemsetAsync(d_RHS_pool, 0, rhs_bytes, stream));
    k_bn_ghost_fill<<<n_leaves, GPU_NB2 * GPU_NB2, 0, stream>>>(d_metas);
    k_bn_rhs<<<n_leaves, GPU_NB * GPU_NB * GPU_NB, 0, stream>>>(d_metas);
    k_bn_rk3s23<<<grid_update, TPB, 0, stream>>>(d_metas, dt, 0.75, 0.25);

    // Stage 3
    CUDA_CHECK(cudaMemsetAsync(d_RHS_pool, 0, rhs_bytes, stream));
    k_bn_ghost_fill<<<n_leaves, GPU_NB2 * GPU_NB2, 0, stream>>>(d_metas);
    k_bn_rhs<<<n_leaves, GPU_NB * GPU_NB * GPU_NB, 0, stream>>>(d_metas);
    k_bn_rk3s23<<<grid_update, TPB, 0, stream>>>(d_metas, dt, 1.0/3.0, 2.0/3.0);

    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    else         CUDA_CHECK(cudaStreamSynchronize(stream));
}
