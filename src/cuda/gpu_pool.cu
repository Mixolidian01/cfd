// P8.1: GpuPool — CUDA implementation
//
// Device layout: d_Q[v * NCELL + flat_idx]  (flat SoA)
// Host layout  : AoSoA — FieldProxy::copy_to_flat / assign_from_flat
//
// upload: AoSoA → staging h_buf (thread_local) → cudaMemcpy H→D
// download: cudaMemcpy D→H → h_buf → AoSoA
//
// The staging buffer is 69,120 bytes (5 * 1728 * 8). thread_local avoids
// contention when multiple CPU threads drive concurrent blocks.

#include "../../include/gpu_pool.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <string>

static void check(cudaError_t err, const char* where) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " +
                                 cudaGetErrorString(err));
}

// =============================================================================
GpuPool::~GpuPool() {
    for (double* p : free_list_) cudaFree(p);
    free_list_.clear();
}

// =============================================================================
void GpuPool::alloc(CellBlock* blk) {
    assert(blk != nullptr);
    assert(blk->d_Q == nullptr && "double alloc — call free() first");

    if (!free_list_.empty()) {
        blk->d_Q = free_list_.back();
        free_list_.pop_back();
    } else {
        double* ptr = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&ptr), slot_bytes()),
              "GpuPool::alloc cudaMalloc");
        blk->d_Q = ptr;
    }
    ++active_;
}

// =============================================================================
void GpuPool::free(CellBlock* blk) {
    assert(blk != nullptr);
    if (blk->d_Q == nullptr) return;
    free_list_.push_back(blk->d_Q);
    blk->d_Q = nullptr;
    --active_;
}

// =============================================================================
void GpuPool::upload(const CellBlock* blk) {
    assert(blk != nullptr && blk->d_Q != nullptr);
    // Staging buffer: flat SoA — d_Q[v * NCELL + flat]
    static thread_local double h_buf[NVAR * NCELL];
    for (int v = 0; v < NVAR; ++v)
        blk->Q[v].copy_to_flat(h_buf + v * NCELL);
    check(cudaMemcpy(blk->d_Q, h_buf, slot_bytes(),
                     cudaMemcpyHostToDevice), "GpuPool::upload");
}

// =============================================================================
void GpuPool::download(CellBlock* blk) const {
    assert(blk != nullptr && blk->d_Q != nullptr);
    static thread_local double h_buf[NVAR * NCELL];
    check(cudaMemcpy(h_buf, blk->d_Q, slot_bytes(),
                     cudaMemcpyDeviceToHost), "GpuPool::download");
    for (int v = 0; v < NVAR; ++v)
        blk->Q[v].assign_from_flat(h_buf + v * NCELL);
}
