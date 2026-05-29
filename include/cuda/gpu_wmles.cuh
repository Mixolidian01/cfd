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

// ── Device-callable ODE mixing-length model (equilibrium TBLE, van Driest) ──
// Picard iteration: integrate u_pred = ∫₀^{y_m} u_τ²/(ν+ν_t) dy,
// update u_τ ← u_τ · sqrt(u_t / u_pred) until converged.
// Uses N=128 trapezoidal points (same as CPU default).
__host__ __device__ inline double d_wm_ode_ml(
    double u_t, double y_m, double nu, double kappa, double A_plus, double tol)
{
    if (u_t <= WM_UTAU_MIN) return 0.0;

    // Start from algebraic guess (kappa=0.41, B=5.2)
    double utau = d_wm_log_law(u_t, y_m, nu, kappa, 5.2, tol);
    if (utau <= WM_UTAU_MIN) return 0.0;

    constexpr int N  = 128;
    const double  dy = y_m / N;

    for (int iter = 0; iter < 60; ++iter) {
        double u_pred = 0.0;
        double f_prev = 1.0;  // f = u_τ²/(ν+ν_t) at y=0 → ν/(ν+0) = 1 × u_τ²/u_τ² = 1
        for (int i = 1; i <= N; ++i) {
            const double y    = i * dy;
            const double yp   = y * utau / nu;
            const double D    = 1.0 - exp(-yp / A_plus);
            const double lm   = kappa * y * D;
            // Solve ν_t² + ν·ν_t - lm²·u_τ² = 0 (quadratic in ν_t)
            const double disc = nu*nu + 4.0*lm*lm*utau*utau;
            const double nut  = 0.5*(-nu + sqrt(disc));
            const double f_cur = utau*utau / (nu + nut);
            u_pred += 0.5*(f_prev + f_cur) * dy;
            f_prev  = f_cur;
        }
        const double ratio    = u_t / (u_pred + 1e-300);
        const double utau_new = utau * sqrt(ratio);
        if (fabs(utau_new - utau) < tol * utau + 1e-15) {
            utau = utau_new;
            break;
        }
        utau = 0.5*utau + 0.5*utau_new;
        utau = (utau < WM_UTAU_MIN) ? WM_UTAU_MIN : utau;
    }
    return utau;
}
