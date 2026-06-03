#pragma once
// gpu_array.cuh — minimal RAII wrapper for owned device allocations.
// Satisfies CLAUDE.md rule 6 ("No raw owning pointers — std::unique_ptr or GpuArray<T>")
// for members that don't fit the GpuPool::d_Q(blk) pattern (e.g. scratch
// buffers, per-leaf metadata arrays).
//
// Usage:
//   GpuArray<double> d_buf_{};
//   d_buf_.alloc(n);                              // (re)allocates n elements
//   double* p = d_buf_.get();                     // raw device pointer
//   d_buf_.upload(h_vec);                         // alloc(h_vec.size()) + H→D copy
//   // automatic cudaFree on destruction
//
// Non-copyable, movable.

#include "gpu_check.cuh"
#include <cstddef>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

template <typename T>
struct GpuArray {
    GpuArray() = default;
    ~GpuArray() { reset(); }

    GpuArray(const GpuArray&) = delete;
    GpuArray& operator=(const GpuArray&) = delete;

    GpuArray(GpuArray&& o) noexcept : ptr_(o.ptr_), n_(o.n_) { o.ptr_ = nullptr; o.n_ = 0; }
    GpuArray& operator=(GpuArray&& o) noexcept {
        if (this != &o) { reset(); ptr_ = o.ptr_; n_ = o.n_; o.ptr_ = nullptr; o.n_ = 0; }
        return *this;
    }

    // (Re)allocate to hold n elements.  Frees previous allocation.  No-op if n==0.
    void alloc(std::size_t n) {
        reset();
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        n_ = n;
    }

    // alloc(h.size()) then host->device copy.
    void upload(const std::vector<T>& h) {
        alloc(h.size());
        if (!h.empty())
            CUDA_CHECK(cudaMemcpy(ptr_, h.data(), h.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
    }

    void reset() noexcept {
        if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; }
        n_ = 0;
    }

    T*          get()       noexcept { return ptr_; }
    const T*    get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return n_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*          ptr_ = nullptr;
    std::size_t n_   = 0;
};
