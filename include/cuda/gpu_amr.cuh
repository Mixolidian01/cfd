#pragma once
// P8.4: GPU AMR prolongation and restriction kernels.
//
// k_prolong: piecewise-constant coarse→fine injection.
//   Grid: n_prolong blocks (one per (coarse,fine) pair).
//   Block: dim3(GPU_NB, GPU_NB, GPU_NB) = 512 threads.
//   Thread (li,lj,lk) writes fine interior cell (NG+li, NG+lj, NG+lk)
//   from the corresponding coarse parent cell.
//
// k_restrict: volume-weighted average of 8 fine children → coarse.
//   Grid: n_restrict blocks (one per coarse block).
//   Block: dim3(GPU_NB, GPU_NB, GPU_NB) = 512 threads.
//   Thread (li,lj,lk) averages the 2³ fine cells that map to coarse
//   interior cell (NG+li, NG+lj, NG+lk).
//
// Octant bit layout: bit0=x-high, bit1=y-high, bit2=z-high.
// Matches BlockTree convention in block_tree.hpp.

#include "mesh/cell_block.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// ── Per-operation metadata ────────────────────────────────────────────────────

struct GpuProlongMeta {
    const double* d_coarse_Q;  // source (read-only)
    double*       d_fine_Q;    // destination (write interior only)
    int32_t       oct;         // child octant 0..7
    int32_t       _pad;
};
static_assert(sizeof(GpuProlongMeta) == 24, "GpuProlongMeta size mismatch");

struct alignas(8) GpuRestrictMeta {
    double*       d_coarse_Q;        // destination (write interior)
    const double* d_children_Q[8];   // sources: child 0..7 (read interior)
};
static_assert(sizeof(GpuRestrictMeta) == 72, "GpuRestrictMeta size mismatch");

// ── AMR operation list ────────────────────────────────────────────────────────
struct GpuAmrList {
    GpuProlongMeta*  d_prolong  = nullptr;
    int              n_prolong  = 0;
    GpuRestrictMeta* d_restrict = nullptr;
    int              n_restrict = 0;

    GpuAmrList() = default;
    GpuAmrList(const GpuAmrList&) = delete;
    GpuAmrList& operator=(const GpuAmrList&) = delete;
    ~GpuAmrList();

    // Upload a batch of prolong operations to device.
    void build_prolong(const std::vector<GpuProlongMeta>& ops);

    // Upload a batch of restrict operations to device.
    void build_restrict(const std::vector<GpuRestrictMeta>& ops);

    // Launch k_prolong for all registered pairs; synchronises if stream==nullptr.
    void exec_prolong(cudaStream_t stream = nullptr) const;

    // Launch k_restrict for all registered pairs; synchronises if stream==nullptr.
    void exec_restrict(cudaStream_t stream = nullptr) const;
};
