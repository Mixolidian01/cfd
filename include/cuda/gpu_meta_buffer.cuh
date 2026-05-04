#pragma once
// P10-B2: gpu_upload_meta — replaces the cudaFree/cudaMalloc/cudaMemcpy
// boilerplate that appears identically in every GPU list build() method.
//
// Usage:
//   gpu_upload_meta(d_metas, h_metas);  // free old, alloc new, H→D copy
//
// The three-line pattern:
//   cudaFree(ptr); ptr = nullptr;
//   CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
//   CUDA_CHECK(cudaMemcpy(ptr, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
// collapses to one call and one RAII destructor-style null-out.

#include "gpu_check.cuh"
#include <vector>
#include <cuda_runtime.h>

// Upload a host vector to a device pointer, freeing the old allocation first.
// Sets ptr = nullptr when h is empty.
template <typename T>
inline void gpu_upload_meta(T*& ptr, const std::vector<T>& h) {
    cudaFree(ptr);
    ptr = nullptr;
    if (h.empty()) return;
    CUDA_CHECK(cudaMalloc(&ptr, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(ptr, h.data(), h.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
}

// Upload a single scalar to a device pointer, freeing the old allocation first.
template <typename T>
inline void gpu_upload_scalar(T*& ptr, const T& val) {
    cudaFree(ptr);
    ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(T)));
    CUDA_CHECK(cudaMemcpy(ptr, &val, sizeof(T), cudaMemcpyHostToDevice));
}
