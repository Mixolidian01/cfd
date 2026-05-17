#pragma once
// D5: GPU reactive flow — species transport (scalar Y) and Arrhenius chemistry.
//
// GpuArrheniusList manages one d_Y[NCELL] array per leaf block.
//
// Per-step call sequence (after GpuGraphSolver::advance()):
//   list.exec_ghost_y(stream)    — fill Y ghost cells from neighbours (periodic)
//   list.exec_advect(dt, stream) — 1st-order upwind species advection (3D)
//   list.exec_rk4(dt, params, stream) — subcycled explicit RK4 chemistry

#include "physics/arrhenius.hpp"
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

// ── Per-leaf metadata (host-built, uploaded to device) ────────────────────────
struct GpuSourceLeafMeta {
    double* d_Q;        // leaf Q [NVAR*NCELL] (read for velocity; E written by chem)
    double* d_Y;        // this leaf's Y [NCELL] (all cells incl. ghosts)
    double* d_Ynb[6];   // neighbour d_Y for ghost fill; null → periodic self-wrap
    double  h;          // cell spacing
    int8_t  bc_int;     // 0=periodic, 1=wall (zero-grad for Y)
};

// ── Species transport + chemistry list ───────────────────────────────────────
struct GpuArrheniusList {
    GpuSourceLeafMeta* d_metas = nullptr;
    int                n_leaves = 0;

    // Host-side Y pointer registry (CellBlock* → d_Y), kept for neighbour lookup.
    std::unordered_map<const CellBlock*, double*> y_map_;

    GpuArrheniusList() = default;
    GpuArrheniusList(const GpuArrheniusList&) = delete;
    GpuArrheniusList& operator=(const GpuArrheniusList&) = delete;
    ~GpuArrheniusList();

    // Rebuild after regrid. Allocates d_Y per leaf if not already done.
    // Caller is responsible for initialising d_Y content after build().
    void build(const BlockTree& tree, const GpuPool& pool, int bc_type = 0);

    // Fill Y ghost cells from neighbours (periodic or wall zero-grad).
    void exec_ghost_y(cudaStream_t stream = nullptr) const;

    // 1st-order upwind advection of Y in all three directions using Q velocity.
    void exec_advect(double dt, cudaStream_t stream = nullptr) const;

    // Subcycled explicit RK4 Arrhenius chemistry — updates d_Y and d_Q[E].
    void exec_rk4(double dt, ArrheniusParams params,
                  cudaStream_t stream = nullptr) const;

    // Upload a flat host Y vector (NB³ interior cells per leaf, Morton order of leaves)
    // into the device d_Y arrays.
    void upload_y(const std::vector<double>& h_Y) const;

    // Download all interior Y cells (NB³ per leaf) into h_Y (caller sizes it).
    void download_y(std::vector<double>& h_Y) const;

    // Number of interior Y cells across all leaves.
    int n_interior_y() const noexcept { return n_leaves * NB * NB * NB; }
};
