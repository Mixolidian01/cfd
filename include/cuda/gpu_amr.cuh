#pragma once
// P8.4 / D1: GPU AMR prolongation, restriction, and refinement sensor kernels.
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

// ── Refinement sensor ─────────────────────────────────────────────────────────
// k_refine_sensor: one block per leaf.
//   Block: dim3(GPU_NB, GPU_NB, GPU_NB) = 512 threads.
//   Computes max(|∇ρ|·h / |ρ|) over interior cells and writes to d_sensor[blockIdx.x].
//   Used by GpuGraphSolver::gpu_regrid() to decide refine/coarsen without D2H of Q.
struct GpuSensorMeta {
    const double* d_Q;     // device Q array for this leaf (NVAR × GPU_NCELL)
    float         h;       // cell size
    int32_t       _pad;
};
static_assert(sizeof(GpuSensorMeta) == 16, "GpuSensorMeta size mismatch");

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

// ── Refinement sensor utility ─────────────────────────────────────────────────
// Evaluate the AMR refinement sensor for each leaf on GPU.
//   d_sensor[leaf_idx] = max(|∇ρ|·h / |ρ|) over interior cells.
// n_leaves leaves with their Q arrays at d_Q_ptrs[i] and cell sizes h_vals[i].
// Result is in d_sensor (device, float, size = n_leaves).
void gpu_eval_refine_sensor(
    const double* const* d_Q_ptrs,  // host array of n_leaves device pointers
    const float*         h_vals,    // host array of n_leaves cell sizes
    int                  n_leaves,
    float*               d_sensor,  // device output (float per leaf)
    cudaStream_t         stream = nullptr);
