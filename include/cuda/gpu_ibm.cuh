#pragma once
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include "gpu_bvh.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

struct GpuIbmMeta {
    double*  d_Q;
    int8_t*  d_cell_type; // [NCELL] 0=FLUID, 1=SOLID, 2=IBM_GHOST
    float*   d_sdf;       // [NCELL] signed distance (+ = fluid)
    float*   d_wnx;       // [NCELL] wall normal x
    float*   d_wny;       // [NCELL] wall normal y
    float*   d_wnz;       // [NCELL] wall normal z
    float    ox, oy, oz;
    float    hx, hy, hz;
};

// GhostEntry is forward-declared here; full definition added in Task 4.
struct GhostEntry;

struct GpuIbmList {
    GpuIbmMeta* d_metas          = nullptr; // [n_leaves]
    int8_t*     d_cell_type_pool = nullptr; // [n_leaves * NCELL]
    float*      d_sdf_pool       = nullptr; // [n_leaves * NCELL]
    float*      d_wnorm_pool     = nullptr; // [n_leaves * 3 * NCELL]
    GhostEntry* d_ghosts         = nullptr; // [n_ghosts]
    int         n_leaves         = 0;
    int         n_ghosts         = 0;

    uint8_t  wall_bc = 0;  // 0=NoSlip, 1=Adiabatic, 2=Isothermal
    float    u_wall = 0.f, v_wall = 0.f, w_wall = 0.f, T_wall = 300.f;

    GpuIbmList() = default;
    GpuIbmList(const GpuIbmList&) = delete;
    GpuIbmList& operator=(const GpuIbmList&) = delete;
    ~GpuIbmList();

    void build(const BlockTree& tree, const GpuPool& pool, const GpuBvh& bvh);
    void exec(cudaStream_t stream = nullptr) const;
};
