#pragma once
// G6: GPU Baer-Nunziato two-phase solver.
// Mirrors BNSolver::advance() (src/models/bn_solver.cpp) on GPU.
// Layout: SoA d_Q[v * GPU_NCELL + flat], v=0..6.
//   v=0: α₁ρ₁  v=1: α₂ρ₂  v=2: ρu  v=3: ρv  v=4: ρw  v=5: E  v=6: α₁

#include "models/bn_model.hpp"
#include "cuda/gpu_constants.cuh"
#include "mesh/block_tree.hpp"
#include <cuda_runtime.h>
#include <vector>

static constexpr int GPU_BN_NVAR = 7;

// Per-leaf metadata (single-block gate: all d_nb set to d_Q for periodic).
struct alignas(64) GpuBnLeafMeta {
    double*       d_Q;              // [GPU_BN_NVAR * GPU_NCELL]
    double*       d_Qn;             // [GPU_BN_NVAR * GPU_NCELL]
    double*       d_RHS;            // [GPU_BN_NVAR * GPU_NCELL]
    const double* d_nb[NFACES];     // neighbour d_Q (= d_Q itself for periodic)
    float         h;
    float         gamma1, gamma2;
    float         pinf1,  pinf2;
    int8_t        bc_type[NFACES];  // 0 = periodic, 1 = wall
    int8_t        _pad[2];
};

struct GpuBnList {
    GpuBnLeafMeta* d_metas    = nullptr;
    double*        d_Q_pool   = nullptr;
    double*        d_Qn_pool  = nullptr;
    double*        d_RHS_pool = nullptr;
    int            n_leaves   = 0;

    GpuBnList() = default;
    GpuBnList(const GpuBnList&) = delete;
    GpuBnList& operator=(const GpuBnList&) = delete;
    ~GpuBnList();

    // Build for n_leaves single-block periodic leaves of cell size h.
    void build(double h, int n_leaves_in, const BNEosParams& eos);

    // Upload BNCellBlock AoSoA → GPU SoA.
    void upload(const std::vector<BNCellBlock>& blocks);

    // Download GPU SoA → BNCellBlock AoSoA.
    void download(std::vector<BNCellBlock>& blocks) const;

    // One SSP-RK3 step with given dt (periodic BC; synchronises on return).
    void advance(double dt, cudaStream_t stream = nullptr);
};
