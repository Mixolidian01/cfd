#pragma once
// D7: GPU-resident WMLES — algebraic Reichardt wall model applied as a kernel.
//
// k_wmles_apply processes all NB² wall-adjacent cells of one block face in a
// single 8×8 thread block.  Newton inversion of the Reichardt composite law
// runs on-device; no D2H round-trip for ghost-cell filling.
//
// Per-step call sequence (after GpuGraphSolver::advance()):
//   list.exec_apply(nu, cfg, stream)   — fill ghost cells for all wall leaves
//
// WallModelCfg is host-side; constants are passed as scalars to the kernel.

#include "models/wall_model.hpp"   // WallModelCfg, WM_UTAU_MIN
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// Per-leaf wall-face metadata (host-built, kept host-side for simplicity).
struct GpuWmlesLeafMeta {
    double* d_Q;      // device Q [NVAR*NCELL]
    double  h;        // cell spacing
    int8_t  wall_ax;  // 0=x, 1=y, 2=z wall normal
    int8_t  side;     // 0=low-index wall, 1=high-index wall
};

// GPU WMLES list — one entry per wall-face leaf.
struct GpuWmlesList {
    std::vector<GpuWmlesLeafMeta> metas;   // host-side; small, no device copy needed

    GpuWmlesList() = default;

    // Register a wall leaf. wall_ax=axis perpendicular to wall; side=0/1.
    void add_leaf(double* d_Q, double h, int wall_ax, int side);

    // Build from a BlockTree — registers all leaves that have a wall BC.
    // wall_ax and side are uniform for the channel test.
    void build_from_tree(const BlockTree& tree, const GpuPool& pool,
                         int wall_ax, int side);

    // Launch GPU wall-model kernel for all registered leaves.
    void exec_apply(double nu, WallModelCfg cfg,
                    cudaStream_t stream = nullptr) const;
};

// ── Device-callable Reichardt composite law ────────────────────────────────
__host__ __device__ inline double d_reichardt_uplus(double yp, double kappa) {
    return (1.0/kappa)*log(1.0 + kappa*yp)
         + 7.8*(1.0 - exp(-yp/11.0) - (yp/11.0)*exp(-yp/3.0));
}

// ── Device-callable Newton inversion: u_τ from (u_t, y_m, ν) ──────────────
__host__ __device__ inline double d_wm_log_law(
    double u_t, double y_m, double nu, double kappa, double B, double tol)
{
    if (u_t <= WM_UTAU_MIN) return 0.0;
    double utau = u_t * kappa / (log(u_t * y_m / nu + 1.0) + kappa * B);
    utau = (utau < WM_UTAU_MIN) ? WM_UTAU_MIN : utau;

    for (int iter = 0; iter < 60; ++iter) {
        double yp  = y_m * utau / nu;
        double up  = d_reichardt_uplus(yp, kappa);
        double F   = utau * up - u_t;
        if (fabs(F) < tol * u_t + 1e-15) break;
        double eps = 1e-6 * utau + 1e-20;
        double yp2 = y_m * (utau + eps) / nu;
        double up2 = d_reichardt_uplus(yp2, kappa);
        double dF  = ((utau + eps)*up2 - utau*up) / eps;
        if (fabs(dF) < 1e-20) break;
        utau -= F / dF;
        utau  = (utau < WM_UTAU_MIN) ? WM_UTAU_MIN : utau;
    }
    return utau;
}
