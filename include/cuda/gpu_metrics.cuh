#pragma once
#include "gpu_constants.cuh"
#include "gpu_rhs.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// ── GpuResidualList ───────────────────────────────────────────────────────────
// Pinned host output buffer for L2 residual partial sums.
// Layout: h_out[li * GPU_NVAR + var] = per-leaf partial sum of RHS[var]^2 over
// interior cells (512 cells). CPU folds after sync.
struct GpuResidualList {
    double*  h_out    = nullptr;  // pinned host [n_leaves * GPU_NVAR]
    int      n_leaves = 0;

    GpuResidualList() = default;
    ~GpuResidualList();
    GpuResidualList(const GpuResidualList&) = delete;
    GpuResidualList& operator=(const GpuResidualList&) = delete;

    void build(int n_leaves_max);
    // Launch k_residual_norm onto stream; no sync.
    void exec(const GpuLeafRhsMeta* d_rhs_metas, cudaStream_t s) const;
    // CPU fold after sync: writes l2[GPU_NVAR].
    void fold(double* l2_out) const;
};

// GPU kernel declaration
__global__ void k_residual_norm(
    const GpuLeafRhsMeta* __restrict__ metas,
    double* __restrict__ d_out,
    int n_leaves);
