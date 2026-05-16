#pragma once
// P14.4: GPU Berger-Colella flux-register correction at AMR C/F interfaces.
//
// Mirrors the CPU sequence in operators.cpp / block_tree.cpp:
//   undo_cf_face_flux        → GpuCfList::undo_coarse_flux (k_cf_undo kernel)
//   accumulate_cf_fine_fluxes → GpuCfList::accum_fine_flux  (k_cf_accum kernel)
//   apply_flux_correction(dt) → GpuCfList::apply_correction  (k_cf_apply kernel)
//
// Call sequence per RK3 step (inside _advance_amr):
//   cf_list.zero_regs(stream)                — once before stage 1
//   for each stage s in {1,2,3}:
//     ghost_list.exec + rhs_list.exec        — fills d_scratch with fresh prims
//     cf_list.undo_coarse_flux(stream)       — removes wrong PCM+HLLC-ES CF flux
//     cf_list.accum_fine_flux(stream, w_s)  — accumulates fine flux × weight
//     k_rk3s... / k_positivity_floor         — advance Q
//   cf_list.apply_correction(stream, dt)     — once after stage 3
//
// Stage weights w_s (SSP-RK3 Berger-Colella quadrature): 1/6, 1/6, 2/3.

#include "mesh/block_tree.hpp"
#include "gpu_rhs.cuh"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// ── Per coarse-leaf CF face metadata (for undo and apply) ────────────────────
// One entry per (coarse_leaf, face_dir_toward_fine) pair.
struct GpuCfCoarseMeta {
    double*       d_RHS;     // coarse d_RHS [NVAR*NCELL] — modified by undo
    double*       d_Q;       // coarse d_Q  [NVAR*NCELL] — modified by apply
    const double* d_scratch; // coarse d_scratch [9*NCELL] — read by undo
    double*       d_reg;     // flux register [NVAR*NB*NB] — written by accum, read by apply
    float         h_coarse;
    int8_t        face_dir;  // 0-5: XMINUS,XPLUS,YMINUS,YPLUS,ZMINUS,ZPLUS (coarse→fine)
    int8_t        _pad[3];
};
static_assert(sizeof(GpuCfCoarseMeta) == 40, "GpuCfCoarseMeta size check");

// ── Per fine-leaf CF face metadata (for accum) ───────────────────────────────
// One entry per (fine_leaf, face_dir_toward_coarse) pair.
struct GpuCfFineMeta {
    const double* d_scratch; // fine d_scratch [9*NCELL] — read by accum
    double*       d_reg;     // points into coarse d_reg_pool [NVAR*NB*NB]
    float         h_fine;
    int8_t        face_dir;  // direction from fine toward coarse (0-5)
    int8_t        off1;      // octant offset along first transverse axis (0 or 1)
    int8_t        off2;      // octant offset along second transverse axis (0 or 1)
    int8_t        _pad;
};
static_assert(sizeof(GpuCfFineMeta) == 24, "GpuCfFineMeta size check");

// ── CF correction list ────────────────────────────────────────────────────────
struct GpuCfList {
    GpuCfCoarseMeta* d_coarse   = nullptr;
    int              n_coarse   = 0;
    GpuCfFineMeta*   d_fine     = nullptr;
    int              n_fine     = 0;
    double*          d_reg_pool = nullptr;  // contiguous: n_coarse × NVAR×NB×NB doubles

    GpuCfList() = default;
    GpuCfList(const GpuCfList&) = delete;
    GpuCfList& operator=(const GpuCfList&) = delete;
    ~GpuCfList();

    // Build from tree topology + per-leaf device pointers.
    // d_rhs_pool   : leaf li at offset li * NVAR * NCELL
    // d_scratch_pool: leaf li at offset li * 9   * NCELL
    // Must be called after GpuRhsList::build() on the same tree.
    void build(const BlockTree& tree, const GpuPool& pool,
               double* d_rhs_pool, double* d_scratch_pool);

    // Zero all flux registers; call once before stage 1.
    void zero_regs(cudaStream_t stream = nullptr) const;

    // Per-stage operations (called after rhs_list.exec()):
    void undo_coarse_flux(cudaStream_t stream = nullptr) const;
    void accum_fine_flux(cudaStream_t stream, double stage_weight) const;
    void apply_correction(cudaStream_t stream, double dt) const;
};
