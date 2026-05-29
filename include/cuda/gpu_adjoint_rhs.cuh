#pragma once
// G2: GPU adjoint convective RHS — device-side L*(Q)·λ (PCM+HLLC-ES, frozen-lam).
// One CUDA thread-block (NB2×NB2 = 144 threads) per leaf.

#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <vector>

// Per-leaf adjoint metadata (device array).
struct alignas(64) GpuAdjMeta {
    const double* d_Q;       // frozen state [NVAR*NCELL], device ptr
    double        hx, hy, hz;
};

struct GpuAdjointRhsList {
    GpuAdjMeta* d_metas = nullptr;
    int         n_leaves = 0;

    GpuAdjointRhsList() = default;
    GpuAdjointRhsList(const GpuAdjointRhsList&) = delete;
    GpuAdjointRhsList& operator=(const GpuAdjointRhsList&) = delete;
    ~GpuAdjointRhsList();

    // Rebuild from tree + pool (call after each regrid).
    void build(const BlockTree& tree, const GpuPool& pool);

    // Launch L*(Q)·λ on stream s.
    // d_lam_in  [n_leaves * NVAR * NCELL] — input adjoint seeds (read-only)
    // d_lam_out [n_leaves * NVAR * NCELL] — output (zeroed, then accumulated)
    void exec(const double* d_lam_in, double* d_lam_out, cudaStream_t s) const;

    // Synchronous host wrapper: upload h_lam → device, run exec, download result.
    void exec_sync(const std::vector<double>& h_lam,
                   std::vector<double>&       h_out) const;
};
