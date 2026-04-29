#pragma once
// P8.6: CUDA Graph re-capture on regrid.
//
// GpuGraphSolver wraps the full SSP-RK3 loop (ghost fill + WENO5-Z RHS +
// RK3 update) into three per-stage CUDA sub-graphs (s1, s2, s3) that are:
//   • Captured once (after the first explicit step) for each topology.
//   • Replayed every subsequent step with explicit cudaMemsetAsync on
//     stream_ between stages — zeroing d_rhs_pool outside the graphs so
//     the captured nodes never include a memset node (which proved
//     unreliable across repeated replays in CUDA 13.x with Global mode).
//   • Invalidated and re-captured whenever build() is called (regrid).
//
// Per-stage graph content (no RHS zeroing inside):
//   graph_s1_: k_save_qn + ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s1
//   graph_s2_: ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s23(0.75, 0.25)
//   graph_s3_: ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s23(1/3, 2/3)
//
// Replay sequence in advance():
//   cfl_list_.exec()                   (async on stream_, updates d_dt)
//   cudaMemsetAsync(d_rhs_pool, 0, stream_)
//   cudaGraphLaunch(graph_s1_, stream_)
//   cudaMemsetAsync(d_rhs_pool, 0, stream_)
//   cudaGraphLaunch(graph_s2_, stream_)
//   cudaMemsetAsync(d_rhs_pool, 0, stream_)
//   cudaGraphLaunch(graph_s3_, stream_)
//   cudaStreamSynchronize(stream_)

#include "../cell_block.hpp"
#include "../block_tree.hpp"
#include "gpu_ghost_fill.cuh"
#include "gpu_rhs.cuh"
#include "gpu_cfl.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// ── Per-leaf RK3 update metadata ─────────────────────────────────────────────
struct GpuRk3LeafMeta {
    double*       d_Q;    // current state (in/out per stage)
    double*       d_Qn;   // saved Q^n (written by k_save_qn, read by update kernels)
    const double* d_RHS;  // from GpuRhsList.d_rhs_pool (read-only)
};

// ── CUDA Graph solver ─────────────────────────────────────────────────────────
struct GpuGraphSolver {
    // Component lists (rebuilt on each build())
    GpuGhostFillList ghost_list_;
    GpuRhsList       rhs_list_;
    GpuCflList       cfl_list_;

    // Per-leaf RK3 metadata and Qn pool
    GpuRk3LeafMeta* d_rk3_metas_ = nullptr;
    double*          d_Qn_pool_   = nullptr;
    int              n_leaves_    = 0;

    // CUDA Graph state — three per-stage sub-graphs
    cudaStream_t    stream_      = nullptr;
    cudaGraphExec_t graph_s1_    = nullptr;
    cudaGraphExec_t graph_s2_    = nullptr;
    cudaGraphExec_t graph_s3_    = nullptr;
    bool            graph_valid_ = false;

    GpuGraphSolver();
    GpuGraphSolver(const GpuGraphSolver&) = delete;
    GpuGraphSolver& operator=(const GpuGraphSolver&) = delete;
    ~GpuGraphSolver();

    // Rebuild all component lists from the tree; invalidates any captured graphs.
    // bc_type: 0=periodic, 1=wall, 2=open.
    void build(const BlockTree& tree, int bc_type = 0);

    // Run one SSP-RK3 step.  If graphs are invalid (first step after build or
    // regrid), runs explicitly and then captures three per-stage sub-graphs.
    // Otherwise replays the sub-graphs with explicit inter-stage zeroing.
    // Returns the CFL-limited dt.
    double advance(const BlockTree& tree, double cfl);

    // Copy device Q back to CPU CellBlocks for all leaves (for verification).
    void download_q(const BlockTree& tree) const;

private:
    void _run_rk3_explicit(cudaStream_t s) const;
    void _capture_graphs();
    void _destroy_graphs();
};
