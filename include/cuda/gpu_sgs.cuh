#pragma once
// gpu_sgs.cuh — P-SGS-GPU: Smagorinsky SGS model as a GPU operator-split kernel.
//
// Called AFTER the full SSP-RK3 cycle (and a ghost-fill refresh) on every step.
// Mirrors NSSolver's CPU sgs->apply() operator-split: reads Q^{n+1} from d_Q,
// computes mu_t = rho*Cs²h²*|S̄|, applies conservative stress divergence in-place.
//
// k_sgs_smag launch config:
//   Grid: dim3(n_leaves), Block: dim3(GPU_NB, GPU_NB) = 64 threads
//   Shared: GPU_NCELL doubles for mu_t = 13824 bytes  (fits on any sm_8x device)

#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>

// ── Per-leaf SGS metadata (32 bytes) ─────────────────────────────────────────
struct alignas(32) GpuSgsMeta {
    double* d_Q;     // flat SoA Q[NVAR][NCELL] — read and written in-place
    double  h;       // cell width
    double  Cs2h2;   // Cs² · h²  (precomputed; constant for this build)
    double  kap_fac; // Cp / Pr_t  (SGS thermal diffusivity scale)
};
static_assert(sizeof(GpuSgsMeta) == 32, "GpuSgsMeta size changed");

// ── SGS list ─────────────────────────────────────────────────────────────────
struct GpuSgsList {
    GpuSgsMeta* d_metas  = nullptr;
    int         n_leaves = 0;

    GpuSgsList() = default;
    GpuSgsList(const GpuSgsList&)            = delete;
    GpuSgsList& operator=(const GpuSgsList&) = delete;
    ~GpuSgsList();

    // Rebuild after regrid.  Cs and Pr_t are stored pre-multiplied into GpuSgsMeta.
    void build(const BlockTree& tree, const GpuPool& pool, double Cs, double Pr_t);

    // Launch k_sgs_smag.  d_dt must point to device dt (from GpuCflList::d_dt).
    void exec(const double* d_dt, cudaStream_t stream) const;
};
