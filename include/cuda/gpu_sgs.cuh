#pragma once
// gpu_sgs.cuh — SGS model GPU operator-split kernels:
//   - GpuSgsList      : static Smagorinsky (k_sgs_smag)
//   - GpuDynSgsList   : dynamic Smagorinsky / Germano+Lilly (k_dyn_sgs_germano)
//
// Both called AFTER the full SSP-RK3 cycle (and a ghost-fill refresh) on every step.

#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>

// ── Per-leaf static Smagorinsky metadata (32 bytes) ──────────────────────────
struct alignas(32) GpuSgsMeta {
    double* d_Q;     // flat SoA Q[NVAR][NCELL] — read and written in-place
    double  h;       // cell width
    double  Cs2h2;   // Cs² · h²  (precomputed; constant for this build)
    double  kap_fac; // Cp / Pr_t  (SGS thermal diffusivity scale)
};
static_assert(sizeof(GpuSgsMeta) == 32, "GpuSgsMeta size changed");

// ── Static Smagorinsky list ───────────────────────────────────────────────────
struct GpuSgsList {
    GpuSgsMeta* d_metas  = nullptr;
    int         n_leaves = 0;

    GpuSgsList() = default;
    GpuSgsList(const GpuSgsList&)            = delete;
    GpuSgsList& operator=(const GpuSgsList&) = delete;
    ~GpuSgsList();

    void build(const BlockTree& tree, const GpuPool& pool, double Cs, double Pr_t);
    void exec(const double* d_dt, cudaStream_t stream) const;
};

// ── Dynamic Smagorinsky (Germano + Lilly LS) ──────────────────────────────────
//
// k_dyn_sgs_germano: single kernel per leaf, 64 threads (NB×NB).
//   Phase 1: Sij + Smag → global per-leaf scratch
//   Phase 2: 3×3×3 box test-filter → u_tf, uiuj_tf, SmSij_tf in scratch
//   Phase 3: Germano/Lilly LS via 64-thread reduction → Cs²
//   Phase 4: mu_t = rho · Cs² · Δ² · |S̄| → smu_t shared array
//   Phase 5: conservative stress divergence (same as k_sgs_smag Phase 2)
//
// Scratch layout per leaf (doubles):
//   [0                  ) : Sij[6][NCELL]      = 10368 doubles
//   [6*NCELL            ) : Smag[NCELL]         =  1728 doubles
//   [7*NCELL            ) : u_tf[NCELL]          =  1728 doubles
//   [8*NCELL            ) : v_tf[NCELL]          =  1728 doubles
//   [9*NCELL            ) : w_tf[NCELL]          =  1728 doubles
//   [10*NCELL           ) : uiuj_tf[6][NCELL]   = 10368 doubles
//   [16*NCELL           ) : SmSij_tf[6][NCELL]  = 10368 doubles
//   Total                 :                       38016 doubles = 304128 bytes

static constexpr int DSM_SCRATCH_PER_LEAF = 22 * 1728;  // 38016 doubles

struct alignas(32) GpuDynSgsMeta {
    double* d_Q;       // flat SoA Q[NVAR][NCELL]
    double* d_scratch; // per-leaf scratch (DSM_SCRATCH_PER_LEAF doubles)
    double  h;         // cell width
    double  kap_fac;   // Cp / Pr_t
};
static_assert(sizeof(GpuDynSgsMeta) == 32, "GpuDynSgsMeta size changed");

struct GpuDynSgsList {
    GpuDynSgsMeta* d_metas   = nullptr;
    double*        d_scratch  = nullptr;  // n_leaves × DSM_SCRATCH_PER_LEAF doubles
    int            n_leaves   = 0;

    GpuDynSgsList() = default;
    GpuDynSgsList(const GpuDynSgsList&)            = delete;
    GpuDynSgsList& operator=(const GpuDynSgsList&) = delete;
    ~GpuDynSgsList();

    void build(const BlockTree& tree, const GpuPool& pool, double Pr_t);
    void exec(const double* d_dt, cudaStream_t stream) const;
};
