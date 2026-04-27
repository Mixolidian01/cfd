#pragma once
// P8.1: GPU memory pool for AMR leaf CellBlocks.
//
// Manages a free-list of CUDA device allocations.  Each slot holds
// NVAR * NCELL float64 values in flat SoA order:
//   d_Q[v * NCELL + flat_idx]
//
// CPU layout : AoSoA  — tile * NVAR * W + v * W + lane
// GPU layout : flat SoA — v * NCELL + cell_idx(i,j,k)
// upload/download transparently convert between the two via FieldProxy helpers.
//
// This header compiles under both g++ and nvcc.
// CUDA API calls live entirely in src/cuda/gpu_pool.cu.
//
// Typical NSSolver flow (cfg.use_gpu = true):
//   init():   for each leaf: pool.alloc(blk); pool.upload(blk);
//   refine(): on_block_free_(parent) → on_block_alloc_(8 children)
//   coarsen():on_block_alloc_(parent) → on_block_free_(8 children)
//   advance():pool.upload(blk) after ghost fill; RHS on GPU; pool.download for diag

#include "cell_block.hpp"
#include <vector>
#include <cstddef>

struct GpuPool {
    GpuPool() = default;
    GpuPool(const GpuPool&) = delete;
    GpuPool& operator=(const GpuPool&) = delete;
    ~GpuPool();

    // Assign a device buffer to blk->d_Q from the pool (cudaMalloc if empty).
    // Does NOT copy data — call upload() after filling CPU data.
    void alloc(CellBlock* blk);

    // Return blk->d_Q to the pool for reuse.  Sets blk->d_Q = nullptr.
    void free(CellBlock* blk);

    // AoSoA CPU data → flat SoA device buffer (cudaMemcpy H→D).
    void upload(const CellBlock* blk);

    // Flat SoA device buffer → AoSoA CPU data (cudaMemcpy D→H).
    void download(CellBlock* blk) const;

    // Bytes per device slot.
    static constexpr size_t slot_bytes() noexcept {
        return static_cast<size_t>(NVAR) * NCELL * sizeof(double);
    }

    // Number of currently live (non-pooled) device buffers.
    int n_active() const noexcept { return active_; }

    // Total device memory currently allocated (live + pooled).
    size_t total_allocated_bytes() const noexcept {
        return static_cast<size_t>(active_ + (int)free_list_.size()) * slot_bytes();
    }

private:
    std::vector<double*> free_list_;  // device pointers ready for reuse
    int                  active_ = 0; // count of live (non-pooled) slots
};
