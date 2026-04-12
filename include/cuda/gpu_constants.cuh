#pragma once
// gpu_constants.cuh — compile-time constants mirroring cell_block.hpp
// Must stay in sync with cell_block.hpp (static_assert enforced in test_gpu.cu).

#include <cstdint>

// Block geometry — must match cell_block.hpp
static constexpr int GPU_NB  = 8;      // cells per block per axis (interior)
static constexpr int GPU_NG  = 1;      // ghost layers per side
static constexpr int GPU_NB2 = GPU_NB + 2*GPU_NG;   // 10
static constexpr int GPU_NCELL = GPU_NB2 * GPU_NB2 * GPU_NB2; // 1000

// Number of conserved variables: rho, rhou, rhov, rhow, E
static constexpr int GPU_NVAR = 5;

// Thermodynamic constants — must match cell_block.hpp
static constexpr double GPU_GAMMA = 1.4;
static constexpr double GPU_R_GAS = 287.058;
static constexpr double GPU_MU_REF = 1.716e-5;
static constexpr double GPU_T_REF  = 273.15;
static constexpr double GPU_S_SUTH = 110.4;
static constexpr double GPU_PR     = 0.72;
static constexpr double GPU_CP     = GPU_GAMMA * GPU_R_GAS / (GPU_GAMMA - 1.0);

// Index mapping: (i,j,k) → flat index in [0, GPU_NCELL)
__host__ __device__ __forceinline__
int gpu_cell_idx(int i, int j, int k) noexcept {
    return i + GPU_NB2 * (j + GPU_NB2 * k);
}

// Interior range helpers
__host__ __device__ __forceinline__ int gpu_ilo() { return GPU_NG; }
__host__ __device__ __forceinline__ int gpu_ihi() { return GPU_NB + GPU_NG - 1; }

// Sutherland viscosity
__host__ __device__ __forceinline__
double gpu_sutherland(double T) noexcept {
    return GPU_MU_REF * (GPU_T_REF + GPU_S_SUTH) / (T + GPU_S_SUTH)
         * (T / GPU_T_REF) * sqrt(T / GPU_T_REF);
}
