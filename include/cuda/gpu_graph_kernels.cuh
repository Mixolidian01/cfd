#pragma once
// gpu_graph_kernels.cuh — SSP-RK3 update device kernels for GpuGraphSolver.
//
//   k_save_qn          : d_Qn ← d_Q  (one CUDA block per leaf, 256 threads)
//   k_rk3s1            : stage 1 — Q = Qn + dt * RHS
//   k_rk3s23           : stages 2 & 3 — Q = α*Qn + β*(Q + dt*RHS)
//   k_positivity_floor : clamp ρ≥EPS and p≥EPS on interior cells (P16.1)
//
// Thin include chain: only gpu_rk3_meta.cuh (struct) + gpu_constants.cuh (GPU_*)
// are needed here.  gpu_graph.cu includes gpu_graph.cuh first, which transitively
// provides both, so this header is self-contained via the #pragma once guards.

#include "cuda/gpu_rk3_meta.cuh"
#include "cuda/gpu_constants.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// k_save_qn: d_Qn ← d_Q  (all GPU_NVAR * GPU_NCELL scalars per leaf)
// gridDim.x = n_leaves,  blockDim.x = 256
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_save_qn(const GpuRk3LeafMeta* __restrict__ metas) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    constexpr int total = GPU_NVAR * GPU_NCELL;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        m.d_Qn[i] = m.d_Q[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rk3s1: stage 1 — Q = Qn + dt * RHS
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rk3s1(const GpuRk3LeafMeta* __restrict__ metas,
             const double* __restrict__ d_dt) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    const double dt = *d_dt;
    constexpr int total = GPU_NVAR * GPU_NCELL;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        m.d_Q[i] = m.d_Qn[i] + dt * m.d_RHS[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rk3s23: stages 2 & 3 — Q = α*Qn + β*(Q + dt*RHS)
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rk3s23(const GpuRk3LeafMeta* __restrict__ metas,
              const double* __restrict__ d_dt,
              double alpha, double beta) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    const double dt = *d_dt;
    constexpr int total = GPU_NVAR * GPU_NCELL;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        m.d_Q[i] = alpha * m.d_Qn[i] + beta * (m.d_Q[i] + dt * m.d_RHS[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_positivity_floor: mirrors CPU apply_positivity_floor (P16.1).
// Clamps ρ≥EPS_POS and p≥EPS_POS on interior cells after each RK3 stage.
// gridDim.x = n_leaves,  blockDim.x = 256
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_positivity_floor(const GpuRk3LeafMeta* __restrict__ metas) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    constexpr int NB2   = GPU_NB2;
    constexpr int NCELL = GPU_NCELL;
    constexpr double EPS_POS = 1.0e-12;

    const int n_int = GPU_NB * GPU_NB * GPU_NB;  // 512 interior cells
    for (int idx = threadIdx.x; idx < n_int; idx += blockDim.x) {
        const int ii = idx % GPU_NB + GPU_NG;
        const int jj = (idx / GPU_NB) % GPU_NB + GPU_NG;
        const int kk = idx / (GPU_NB * GPU_NB) + GPU_NG;
        const int c  = ii + NB2 * (jj + NB2 * kk);

        double rho  = m.d_Q[0 * NCELL + c];
        double rhou = m.d_Q[1 * NCELL + c];
        double rhov = m.d_Q[2 * NCELL + c];
        double rhow = m.d_Q[3 * NCELL + c];
        double E    = m.d_Q[4 * NCELL + c];

        if (rho < EPS_POS) {
            m.d_Q[0 * NCELL + c] = rho = EPS_POS;
        }
        // Compute ke using the same divide-then-multiply pattern as gpu_cons_to_prim,
        // so the subsequent prim conversion (k_prim_duc) gives p >= EPS_POS exactly.
        // The old (a²+b²+c²)/rho pattern diverges from a*(a/rho)+... by ~1 ULP,
        // allowing catastrophic cancellation E-ke=0 (p=0) after the floor is applied.
        const double inv_rho = 1.0 / rho;
        const double u = rhou * inv_rho;
        const double v = rhov * inv_rho;
        const double w = rhow * inv_rho;
        const double ke = 0.5 * (rhou*u + rhov*v + rhow*w);
        if ((GPU_GAMMA - 1.0) * (E - ke) < EPS_POS) {
            m.d_Q[4 * NCELL + c] = ke + EPS_POS / (GPU_GAMMA - 1.0);
        }
    }
}
