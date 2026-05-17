// D7: GPU-resident WMLES — algebraic Reichardt wall model.
// Mirrors CPU wm_apply_ghost() exactly; one 8×8 thread block per wall face.

#include "cuda/gpu_wmles.cuh"
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include <cmath>

// ── k_wmles_apply: NB×NB threads, one per wall-adjacent cell ─────────────────
// Fills ghost layers gl=0,1 (NG=2) for one block face.
// Kernel signature passes kappa/B/tol as scalars — WallModelCfg is host-only.
__global__ static void k_wmles_apply(
    double* __restrict__ d_Q,
    double  h,
    int8_t  wall_ax,
    int8_t  side,
    double  nu,
    double  kappa,
    double  B_const,
    double  tol)
{
    const int tx = (int)threadIdx.x;   // 0..NB-1
    const int ty = (int)threadIdx.y;   // 0..NB-1

    // Map thread indices to interior wall-adjacent cell (ci,cj,ck).
    int ci, cj, ck;
    if (wall_ax == 0) {
        ci = (side == 0) ? NG : NG + NB - 1;
        cj = NG + tx;
        ck = NG + ty;
    } else if (wall_ax == 1) {
        ci = NG + tx;
        cj = (side == 0) ? NG : NG + NB - 1;
        ck = NG + ty;
    } else {
        ci = NG + tx;
        cj = NG + ty;
        ck = (side == 0) ? NG : NG + NB - 1;
    }

    const int idx_int = cell_idx(ci, cj, ck);

    const double rho = d_Q[0 * NCELL + idx_int];
    if (rho < 1e-10) return;

    const double y_m = 0.5 * h;
    double u[3] = {
        d_Q[1 * NCELL + idx_int] / rho,
        d_Q[2 * NCELL + idx_int] / rho,
        d_Q[3 * NCELL + idx_int] / rho
    };

    // Isolate wall-normal component; compute wall-parallel magnitude.
    const int wax  = (int)wall_ax;
    const double u_n = u[wax];
    u[wax] = 0.0;
    const double u_t = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

    const double utau  = d_wm_log_law(u_t, y_m, nu, kappa, B_const, tol);
    const double tau_w = rho * utau * utau;
    const double mu    = rho * nu;

    // Fill NG ghost layers stepping away from the interior cell toward the wall.
    for (int gl = 0; gl < NG; ++gl) {
        const int step = gl + 1;
        int gi = ci, gj = cj, gk = ck;
        if (side == 0) {
            if      (wax == 0) gi = ci - step;
            else if (wax == 1) gj = cj - step;
            else               gk = ck - step;
        } else {
            if      (wax == 0) gi = ci + step;
            else if (wax == 1) gj = cj + step;
            else               gk = ck + step;
        }
        const int idx_g = cell_idx(gi, gj, gk);

        // Density: copy from interior.
        d_Q[0 * NCELL + idx_g] = rho;

        // Wall-parallel: set so FD derivative gives τ_w/μ.
        // u_ghost = u_int − step·h·τ_w/μ  (element-wise, preserving direction).
        const double scale = (u_t > WM_UTAU_MIN)
                             ? (1.0 - (step * h * tau_w / mu) / u_t)
                             : 0.0;
        double u_g[3] = { u[0] * scale, u[1] * scale, u[2] * scale };
        // Wall-normal: image method (no penetration).
        u_g[wax] = -u_n;

        d_Q[1 * NCELL + idx_g] = rho * u_g[0];
        d_Q[2 * NCELL + idx_g] = rho * u_g[1];
        d_Q[3 * NCELL + idx_g] = rho * u_g[2];
        // Energy: adiabatic wall.
        d_Q[4 * NCELL + idx_g] = d_Q[4 * NCELL + idx_int];
    }
}

// ── GpuWmlesList ──────────────────────────────────────────────────────────────
void GpuWmlesList::add_leaf(double* d_Q, double h, int wall_ax, int side) {
    metas.push_back({ d_Q, h, (int8_t)wall_ax, (int8_t)side });
}

void GpuWmlesList::build_from_tree(const BlockTree& tree, const GpuPool& pool,
                                    int wall_ax, int side) {
    for (int li : tree.leaf_indices()) {
        const CellBlock* blk = tree.nodes[li].block.get();
        if (!blk || !pool.has_device(blk)) continue;
        add_leaf(pool.d_Q(blk), blk->h, wall_ax, side);
    }
}

void GpuWmlesList::exec_apply(double nu, WallModelCfg cfg,
                               cudaStream_t stream) const {
    const dim3 block(NB, NB);
    const dim3 grid(1);
    for (const auto& m : metas) {
        k_wmles_apply<<<grid, block, 0, stream>>>(
            m.d_Q, m.h, m.wall_ax, m.side,
            nu, cfg.kappa, cfg.B, cfg.tol);
    }
}
