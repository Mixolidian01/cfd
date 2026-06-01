#pragma once
#include "gpu_constants.cuh"
#include "gpu_rhs.cuh"
#include "gpu_ibm.cuh"
#include "gpu_scalars.cuh"
#include "gpu_snapshot.hpp"
#include "solver/ns_solver.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <string>

// ── GpuResidualList ───────────────────────────────────────────────────────────
// Pinned host output buffer for L2 residual partial sums.
// Layout: h_out[li * GPU_NVAR + var] = per-leaf partial sum of RHS[var]^2 over
// interior cells (512 cells). CPU folds after sync.
struct GpuResidualList {
    double*  h_out    = nullptr;  // pinned host [n_leaves * GPU_NVAR]
    int      n_leaves = 0;

    GpuResidualList() = default;
    ~GpuResidualList();
    GpuResidualList(const GpuResidualList&) = delete;
    GpuResidualList& operator=(const GpuResidualList&) = delete;

    void build(int n_leaves_max);
    // Launch k_residual_norm onto stream; no sync.
    void exec(const GpuLeafRhsMeta* d_rhs_metas, cudaStream_t s) const;
    // CPU fold after sync: writes l2[GPU_NVAR].
    void fold(double* l2_out) const;
};

// GPU kernel declaration
__global__ void k_residual_norm(
    const GpuLeafRhsMeta* __restrict__ metas,
    double* __restrict__ d_out,
    int n_leaves);

// ── GpuSurfaceEntry ──────────────────────────────────────────────────────────
// Per IBM ghost cell; built once at GpuSurfaceList::build() time.
struct alignas(16) GpuSurfaceEntry {
    double*  ghost_ptr;       // base ptr into ghost cell's d_Q block
    double*  stencil[8];      // base ptrs for trilinear image-point stencil
    float    w[8];            // trilinear weights (sum ≈ 1)
    float    nx, ny, nz;      // outward wall normal
    float    d;               // |sdf| — half image-point distance
    float    h;               // cell width
    float    cx, cy, cz;      // ghost cell physical centre
    uint8_t  wall_bc;         // 0=NoSlip/Adiabatic, 2=Isothermal
    float    u_wall, v_wall, w_wall;
};

// ── GpuSurfaceList ────────────────────────────────────────────────────────────
struct GpuSurfaceList {
    GpuSurfaceEntry* d_entries = nullptr;
    double*          d_acc     = nullptr;  // device [6] = {Fx,Fy,Fz,Mx,My,Mz}
    double*          h_acc     = nullptr;  // pinned [6]
    int              n_entries = 0;

    std::array<double,3> ref_point = {};
    double rho_ref = 0.0, u_ref = 0.0, A_ref = 0.0;
    std::string name;

    GpuSurfaceList() = default;
    ~GpuSurfaceList();
    GpuSurfaceList(const GpuSurfaceList&) = delete;
    GpuSurfaceList& operator=(const GpuSurfaceList&) = delete;

    // Build from ibm_list: D2H downloads pools, scans IBM_GHOST cells, uploads entries.
    void build(const GpuIbmList& ibm_list, const SolverConfig::SurfaceConfig& cfg);

    // Launch k_surface_forces onto stream; zeroes d_acc first; D2H copies to h_acc.
    void exec(cudaStream_t s, float mu) const;

    const double* results() const { return h_acc; }
};

// GPU kernel declaration
__global__ void k_surface_forces(
    const GpuSurfaceEntry* __restrict__ entries, int n_entries,
    double* __restrict__ acc,
    float mu,
    double ref_x, double ref_y, double ref_z);

// ── GpuProbeEntry ─────────────────────────────────────────────────────────────
struct GpuProbeEntry {
    float px, py, pz;  // physical probe location
    int   var_id;       // snap_scalar_val var_id (0-11)
};

// ── GpuProbeList ─────────────────────────────────────────────────────────────
struct GpuProbeList {
    GpuProbeEntry* d_probes   = nullptr;
    double*        d_results  = nullptr;  // device [n_probes]
    double*        h_results  = nullptr;  // pinned [n_probes]
    int            n_probes   = 0;

    // For plane_avg (Task 5):
    double* d_slab_sum   = nullptr;
    int*    d_slab_cnt   = nullptr;
    double* h_slab_sum   = nullptr;
    int*    h_slab_cnt   = nullptr;
    int     n_slabs      = 0;
    int     plane_axis   = 1;
    int     plane_var_id = 0;
    float   slab_lo_     = 0.f;
    float   slab_hi_     = 1.f;

    GpuProbeList() = default;
    ~GpuProbeList();
    GpuProbeList(const GpuProbeList&) = delete;
    GpuProbeList& operator=(const GpuProbeList&) = delete;

    // Point probe: single point at (px, py, pz), quantity var_id.
    void build_point(float px, float py, float pz, int var_id);

    // Plane-averaged profile (Task 5 — stub only for now).
    void build_plane(int n_slabs, int axis, float slab_lo, float slab_hi, int var_id);

    // Launch k_probe_interp on stream; no sync.
    void exec(const SnapLeafMeta* d_metas, int n_leaves, cudaStream_t s) const;

    // Launch k_plane_avg on stream; no sync (Task 5).
    void exec_plane(const SnapLeafMeta* d_metas, int n_leaves, cudaStream_t s) const;
};

// GPU kernel declarations
__global__ void k_probe_interp(
    const SnapLeafMeta* __restrict__ metas, int n_leaves,
    const GpuProbeEntry* __restrict__ probes, int n_probes,
    double* __restrict__ results);

__global__ void k_plane_avg(
    const SnapLeafMeta* __restrict__ metas, int n_leaves,
    int axis, int n_slabs, float slab_lo, float slab_hi, int var_id,
    double* __restrict__ d_sum, int* __restrict__ d_cnt);
