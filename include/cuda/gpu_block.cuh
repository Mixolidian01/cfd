#pragma once
// gpu_block.cuh — GPU SoA block layout
//
// One GPUBlock corresponds to one CellBlock on the CPU.
// Layout: 5 arrays of GPU_NCELL doubles, one per conserved variable.
// All arrays are device-resident; the struct itself can live on host or device.
//
// Memory layout (for one block of N=10^3=1000 cells):
//   rho[0..999], rhou[0..999], rhov[0..999], rhow[0..999], E[0..999]
//   Total: 5 * 1000 * 8 bytes = 40 KB per block
//
// For a 128-block tree: 128 * 40 KB = 5 MB — fits in L2 on any GPU.
//
// Pointer-of-arrays layout (PoA) for coalesced access:
//   Q[var][cell_idx]  — threads accessing the same variable in adjacent cells
//   access Q[v][base + threadIdx.x] → coalesced warp access.

#include "gpu_constants.cuh"
#include <cstddef>

// ── Flat SoA device buffer ────────────────────────────────────────────────────
// Stores N_blocks * GPU_NVAR * GPU_NCELL doubles in one contiguous allocation.
// Stride between variable v of block b and variable v+1 of block b:
//   GPU_NCELL * N_blocks  (variable-major, block-minor layout)
// This lets a kernel over all blocks share a single pointer + strides.
//
// Access pattern: Q[v * stride_var + b * GPU_NCELL + cell_idx(i,j,k)]
// where stride_var = N_blocks * GPU_NCELL.

struct GPUBlockBuffer {
    double* data;            // device pointer, size = NVAR * N_blocks * NCELL
    int     n_blocks;        // number of blocks
    size_t  stride_var;      // = n_blocks * GPU_NCELL

    // Returns pointer to variable v of block b
    __host__ __device__ __forceinline__
    double* var(int v, int b) noexcept {
        return data + (size_t)v * stride_var + (size_t)b * GPU_NCELL;
    }
    __host__ __device__ __forceinline__
    const double* var(int v, int b) const noexcept {
        return data + (size_t)v * stride_var + (size_t)b * GPU_NCELL;
    }

    // Total bytes
    size_t bytes() const noexcept {
        return (size_t)GPU_NVAR * stride_var * sizeof(double);
    }
};

// ── Primitive struct (registers only — never stored in global memory) ─────────
struct GPrim {
    double rho, u, v, w, p, T, c;
};

// ── Inline EOS ────────────────────────────────────────────────────────────────
__host__ __device__ __forceinline__
GPrim gpu_cons_to_prim(double rho, double rhou, double rhov,
                       double rhow, double E) noexcept {
    GPrim q;
    q.rho = rho;
    double inv_rho = 1.0 / rho;
    q.u = rhou * inv_rho;
    q.v = rhov * inv_rho;
    q.w = rhow * inv_rho;
    q.p = (GPU_GAMMA - 1.0) * (E - 0.5*(rhou*q.u + rhov*q.v + rhow*q.w));
    q.T = q.p * inv_rho / GPU_R_GAS;
    q.c = sqrt(GPU_GAMMA * q.p * inv_rho);
    return q;
}

__host__ __device__ __forceinline__
void gpu_prim_to_cons(const GPrim& q,
                      double& rho, double& rhou, double& rhov,
                      double& rhow, double& E) noexcept {
    rho  = q.rho;
    rhou = q.rho * q.u;
    rhov = q.rho * q.v;
    rhow = q.rho * q.w;
    E    = q.p / (GPU_GAMMA - 1.0)
         + 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
}
