#pragma once
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include "gpu_bvh.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Forward declaration — defined in fsi/rigid_body.hpp.
// Included here only when FSI wire-in is needed (gpu_graph.cu).
struct RigidBody6DOF;

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

// One entry per IBM ghost cell.  Pointers encode:
//   ghost_ptr  = d_Q_block + flat_ghost   (access var v: ghost_ptr + v*GPU_NCELL)
//   stencil[s] = d_Q_blockS + flat_s      (access var v: stencil[s] + v*GPU_NCELL)
struct alignas(16) GhostEntry {
    double*  ghost_ptr;      // base pointer into ghost cell's block d_Q
    double*  stencil[8];     // base pointers into stencil cells' d_Q arrays
    float    w[8];           // trilinear weights (sum ~1)
    uint8_t  wall_bc;        // 0=NoSlip/Adiabatic, 2=Isothermal, 3=SolidFill (copy Q_I to SOLID cell)
    uint8_t  _pad[3];
    float    u_wall, v_wall, w_wall, T_wall;
    // FSI-1: nearest surface point (world coords).  Used by k_apply_moving_wall.
    float    x_surf, y_surf, z_surf;
    float    _pad2;
};

// Per-leaf metadata needed by k_surface_forces_ibm.
// Uses the prim-scratch buffer (d_scratch = rhs_list scratch pool, stores primitive
// variables after k_prim_duc; scratch layout comp=4 is pressure).
struct GpuIbmForceMeta {
    const double* d_scratch;   // prim scratch for this leaf (SCRATCH_NCOMP*NCELL doubles)
    const int8_t* d_cell_type; // [GPU_NCELL] same as GpuIbmMeta::d_cell_type
    const float*  d_sdf;       // [GPU_NCELL]
    const float*  d_wnx;       // [GPU_NCELL] outward wall normal x
    const float*  d_wny;
    const float*  d_wnz;
    float ox, oy, oz;          // block origin
    float hx, hy, hz;          // cell size
};

// FSI-1: rigid-body state passed to ghost-cell kernel for moving-wall BC.
// All values zero → stationary wall (backward-compatible).
struct IbmRigidState {
    double v_cm[3] = {};  // centre-of-mass velocity (world frame)
    double omega[3] = {}; // angular velocity (world frame)
    double x_cm[3] = {};  // centre-of-mass position (world frame)
};

struct GpuIbmList {
    GpuIbmMeta* d_metas          = nullptr; // [n_leaves]
    int8_t*     d_cell_type_pool = nullptr; // [n_leaves * NCELL]
    float*      d_sdf_pool       = nullptr; // [n_leaves * NCELL]
    float*      d_wnorm_pool     = nullptr; // [n_leaves * 3 * NCELL]
    GhostEntry* d_ghosts         = nullptr; // [n_ghosts]  IBM_GHOST fill entries
    GhostEntry* d_solid_fills    = nullptr; // [n_solid_fills] SOLID cell suppression entries
    int         n_leaves         = 0;
    int         n_ghosts         = 0;
    int         n_solid_fills    = 0;

    uint8_t  wall_bc = 0;  // 0=NoSlip/Adiabatic, 2=Isothermal
    float    u_wall = 0.f, v_wall = 0.f, w_wall = 0.f, T_wall = 300.f;

    // FSI-1: rigid-body state for moving-wall BC (defaults = stationary).
    IbmRigidState rigid;

    GpuIbmList() = default;
    GpuIbmList(const GpuIbmList&) = delete;
    GpuIbmList& operator=(const GpuIbmList&) = delete;
    ~GpuIbmList();

    void build(const BlockTree& tree, const GpuPool& pool, const GpuBvh& bvh);
    void exec(cudaStream_t stream = nullptr) const;
    // FSI-1: update ghost-cell wall velocity from rigid-body state (no rebuild needed).
    void update_rigid(const IbmRigidState& s) { rigid = s; }
};

// FSI-1: accumulate pressure force and torque on the immersed surface.
// d_wrench[6] = {Fx,Fy,Fz,Tx,Ty,Tz}; must be zeroed before launch.
// Uses atomicAdd; one thread per IBM_GHOST cell.
__global__
void k_surface_forces_ibm(
    const GpuIbmForceMeta* __restrict__ metas,
    int n_leaves,
    double x_cm_x, double x_cm_y, double x_cm_z,
    double* __restrict__ d_wrench);
