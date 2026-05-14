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
// P10-A1: device pointers live in GpuPool::ptrs_ (keyed by CellBlock*), not
// in CellBlock itself.  This removes the Layer 1 → Layer 3 pointer violation.
// Use d_Q(blk) / has_device(blk) to query the device state of a block.
//
// This header compiles under both g++ and nvcc.
// CUDA API calls live entirely in src/cuda/gpu_pool.cu.

#include "mesh/cell_block.hpp"
#include <unordered_map>
#include <vector>
#include <cstddef>

struct GpuPool {
    GpuPool() = default;
    GpuPool(const GpuPool&) = delete;
    GpuPool& operator=(const GpuPool&) = delete;
    ~GpuPool();

    // Assign a device buffer for blk from the pool (cudaMalloc if empty).
    // Does NOT copy data — call upload() after filling CPU data.
    void alloc(CellBlock* blk);

    // Return the device buffer for blk to the pool for reuse.
    void free(CellBlock* blk);

    // AoSoA CPU data → flat SoA device buffer (cudaMemcpy H→D).
    void upload(const CellBlock* blk);

    // Flat SoA device buffer → AoSoA CPU data (cudaMemcpy D→H).
    void download(CellBlock* blk) const;

    // Return the device pointer for blk, or nullptr if not allocated.
    double* d_Q(const CellBlock* blk) const noexcept;

    // True iff blk has a live device buffer in this pool.
    bool has_device(const CellBlock* blk) const noexcept;

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
    std::unordered_map<const CellBlock*, double*> ptrs_;     // block → device ptr
    std::vector<double*>                          free_list_; // ready for reuse
    int                                           active_ = 0;
};
