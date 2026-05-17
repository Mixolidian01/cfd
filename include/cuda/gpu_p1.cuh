#pragma once
// D6: GPU P1 radiation transport — per-leaf elliptic CG solve + energy coupling.
//
// GpuP1List manages one d_G[NCELL] per leaf (ghost-padded, same layout as d_Q).
//
// Per-step call sequence:
//   list.exec_ghost_g(stream)           — fill G ghost cells (periodic / Dirichlet)
//   list.exec_cg(params, h, stream)     — solve −∇·(D∇G)+κG = κaT⁴ per leaf (CG)
//   list.exec_couple(dt, params, stream)— add κ(aT⁴−G)·dt to Q[4] (energy)
//
// BC types for G:
//   bc_type = 0  — all-periodic (default)
//   bc_type = 1  — Dirichlet at left x-face (G=G_left) and right x-face (G=G_right);
//                  y,z faces periodic.  Used for 1D Marshak wave gate.

#include "physics/p1_radiation.hpp"
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

// ── Per-leaf metadata (host-built, uploaded to device) ────────────────────────
struct GpuP1LeafMeta {
    double* d_Q;          // leaf Q [NVAR*NCELL] (read for T; Q[4] written by couple)
    double* d_G;          // this leaf's G [NCELL] (ghost-padded)
    double* d_Gnb[6];     // neighbour d_G for periodic ghost fill; null → self-wrap
    double  h;            // cell spacing
    int8_t  bc_type;      // 0=periodic, 1=Dirichlet x-faces
    double  G_left;       // Dirichlet G at left  x-face (bc_type=1)
    double  G_right;      // Dirichlet G at right x-face (bc_type=1)
};

// Result of GPU CG solve (aggregated over all leaves).
struct GpuP1CgResult {
    int    max_iters;   // worst-case iteration count across leaves
    double max_rel_res; // worst-case final ‖r‖/‖b‖
};

// ── P1 radiation list ─────────────────────────────────────────────────────────
struct GpuP1List {
    GpuP1LeafMeta* d_metas  = nullptr;
    int            n_leaves = 0;

    // Host-side G pointer registry (CellBlock* → d_G).
    std::unordered_map<const CellBlock*, double*> g_map_;

    cublasHandle_t cublas_ = nullptr;

    GpuP1List();
    GpuP1List(const GpuP1List&) = delete;
    GpuP1List& operator=(const GpuP1List&) = delete;
    ~GpuP1List();

    // Rebuild after regrid.  Allocates d_G per leaf if not already done.
    // bc_type: 0=periodic, 1=Dirichlet (caller sets G_left/G_right via set_bc()).
    void build(const BlockTree& tree, const GpuPool& pool, int bc_type = 0);

    // Set Dirichlet BC values for all leaves (call before exec_ghost_g).
    void set_bc(double G_left, double G_right);

    // Fill G ghost cells from neighbours.
    void exec_ghost_g(cudaStream_t stream = nullptr) const;

    // Solve −∇·(D∇G) + κG = κaT⁴ per leaf with GPU-resident CG.
    // h: uniform cell spacing; G initial guess used from d_G interior.
    GpuP1CgResult exec_cg(RadiationParams params, double h,
                           int    max_iter = 100,
                           double tol      = 1e-8,
                           cudaStream_t stream = nullptr);

    // Energy coupling: Q[4] += κ(aT⁴ − G)·dt  [operator-split after RK3].
    void exec_couple(double dt, RadiationParams params,
                     cudaStream_t stream = nullptr) const;

    // Upload/download G interior cells (NB³ per leaf, Morton order of leaves).
    void upload_g(const std::vector<double>& h_G) const;
    void download_g(std::vector<double>& h_G) const;

    int n_interior_g() const noexcept { return n_leaves * NB * NB * NB; }
};
