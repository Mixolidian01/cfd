#pragma once
// P8.3: GPU per-block RHS kernel (convective + viscous).
//
// Data flow:
//   GpuRhsList::build(tree)      — rebuild after each regrid
//   GpuRhsList::exec(stream)     — called once per RK3 stage
//                                  (after GpuGhostFillList::exec)
//
// Kernel pipeline per leaf:
//   k_prim_duc   — primitive variables + Ducros sensor → d_scratch
//   k_rhs_conv   — WENO5-Z / KEP / HLLC-ES convective flux (atomicAdd)
//   k_rhs_visc   — viscous stress divergence (direct write, no atomics)
//
// Scratch layout: d_scratch[comp * NCELL + flat]
//   comp 0..6 : rho, u, v, w, p, T, c
//   comp 7    : dynamic viscosity µ (Sutherland)
//   comp 8    : Ducros sensor φ
//   SCRATCH_NCOMP = 9
//
// d_RHS is flat SoA (same layout as d_Q): d_RHS[v*NCELL + flat]
// Only interior cells [ilo..ihi]³ are written; ghost cells are untouched.

#include "../cell_block.hpp"
#include "../block_tree.hpp"
#include "../gpu_pool.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

static constexpr int SCRATCH_NCOMP = 9;  // rho,u,v,w,p,T,c,mu,duc

// ── Per-leaf RHS metadata ─────────────────────────────────────────────────────
struct alignas(64) GpuLeafRhsMeta {
    const double* d_Q;       // source block d_Q (with ghost cells filled)
    double*       d_RHS;     // output RHS (flat SoA NVAR*NCELL; interior written)
    double*       d_scratch; // 9*NCELL doubles: prim[0..6], mu[7], duc[8]
    double        h;         // cell size
    int8_t        _pad[4];
};
static_assert(sizeof(GpuLeafRhsMeta) <= 64, "GpuLeafRhsMeta too large");

// ── RHS list ─────────────────────────────────────────────────────────────────
struct GpuRhsList {
    GpuLeafRhsMeta* d_metas   = nullptr;
    double*         d_scratch_pool = nullptr;  // one contiguous alloc for all leaves
    double*         d_rhs_pool     = nullptr;  // d_RHS per leaf
    int             n_leaves  = 0;

    GpuRhsList() = default;
    GpuRhsList(const GpuRhsList&) = delete;
    GpuRhsList& operator=(const GpuRhsList&) = delete;
    ~GpuRhsList();

    // Rebuild after regrid.
    void build(const BlockTree& tree, const GpuPool& pool);

    // Launch the three RHS kernels on the given stream.
    // Prerequisites: d_Q must have ghost cells filled (call GpuGhostFillList::exec first).
    // zero_rhs: if true, zeros d_rhs_pool first (safe for explicit use).
    //           Pass false when the caller has already zeroed d_rhs_pool via
    //           cudaMemsetAsync on the same stream (e.g. inside a CUDA graph stage).
    void exec(cudaStream_t stream = nullptr, bool zero_rhs = true) const;

    // Copy d_RHS back to a CPU CellBlock for each leaf.
    // Called after exec() to verify results or feed the CPU time integrator.
    void download_rhs(const BlockTree& tree) const;
};
