#pragma once
// G5: GPU Berger-Oliger LTS integrator.
//
// GpuLtsIntegrator executes one Berger-Oliger LTS step:
//   1. r fine sub-steps at dt_f (fine leaves only updated)
//   2. 1 coarse step at dt_c = r × dt_f (coarse leaves only updated)
//   3. Berger-Colella flux register correction at C/F interfaces
//
// All ghost fills and RHS evaluations run over ALL leaves; only the
// RK3 update kernels are level-filtered via separate meta arrays.
// No CUDA graphs (level-split launch cannot be captured in a static graph).
//
// Usage:
//   GpuLtsIntegrator lts;
//   lts.build(tree, pool, bc_type, r=2);
//   double dt_c = lts.step(cfl);

#include "cuda/gpu_graph.cuh"       // GpuRk3LeafMeta
#include "cuda/gpu_cfl.cuh"
#include "cuda/gpu_cf.cuh"
#include "cuda/gpu_ghost_fill.cuh"
#include "cuda/gpu_rhs.cuh"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>

struct GpuLtsIntegrator {
    GpuCflList       cfl_fine;       // CFL constrained to fine leaves
    GpuCflList       cfl_coarse;     // CFL constrained to coarse leaves
    GpuGhostFillList ghost_all;      // ghost fill for ALL leaves
    GpuRhsList       rhs_all;        // RHS for ALL leaves
    GpuCfList        cf;             // Berger-Colella C/F correction

    GpuRk3LeafMeta*  d_fine_metas   = nullptr;
    GpuRk3LeafMeta*  d_coarse_metas = nullptr;
    double*          d_Qn_fine      = nullptr;   // Qn buffer for fine leaves
    double*          d_Qn_coarse    = nullptr;   // Qn buffer for coarse leaves
    int              n_fine         = 0;
    int              n_coarse       = 0;
    int              lts_r          = 2;          // refinement ratio
    cudaStream_t     stream         = nullptr;

    GpuLtsIntegrator();
    ~GpuLtsIntegrator();

    // Build from a 2-level BlockTree.
    // Must be called after every regrid.  r is the time-step refinement ratio.
    void build(const BlockTree& tree, const GpuPool& pool, int bc_type, int r = 2);

    // Execute one Berger-Oliger LTS step.  Returns dt_c (coarse dt).
    double step(double cfl);
};
