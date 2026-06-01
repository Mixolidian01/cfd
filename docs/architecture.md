# GPU CFD Solver — Architecture Reference

> **Scope:** GPU subsystem only. The CPU layer (linalg → cell_block → operators → ns_solver) is documented in `CLAUDE.md` §"Reference architecture".

---

## 1. Solver Layers

```
┌─────────────────────────────────────────────────────────────┐
│  apps/simulate_gpu.cu   apps/simulate.cpp                   │  Entry points
├─────────────────────────────────────────────────────────────┤
│  IGpuSolver  (include/solver/ns_solver.hpp)                 │  Interface
│  └─ GpuGraphSolver  (include/cuda/gpu_graph.cuh)            │  Concrete impl
├───────────────────────┬─────────────────────────────────────┤
│  GpuPool              │  GpuSnapshotBuffer                  │  Device memory
│  (gpu_pool.hpp/.cu)   │  (gpu_snapshot.hpp/.cu)             │
├───────────────────────┴─────────────────────────────────────┤
│  "List" objects — one per physics subsystem (see §3)        │  Physics lists
├─────────────────────────────────────────────────────────────┤
│  __global__ kernels  (src/cuda/*.cu)                        │  CUDA kernels
├─────────────────────────────────────────────────────────────┤
│  Device helpers  (include/cuda/gpu_constants.cuh, gpu_bvh.cuh, ...) │
├─────────────────────────────────────────────────────────────┤
│  CellBlock SoA  (include/mesh/cell_block.hpp)               │  Data layout
│  BlockTree      (include/mesh/block_tree.hpp)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. CellBlock Memory Layout

```
NB=8  NG=2  NB2=12  NCELL=1728(=12³)  NVAR=5  GAMMA=1.4  R_GAS=287.058

data_[NTILE * NVAR * W]          // 216 * 5 * 8 = 8640 doubles = 69 120 B
  NTILE = NCELL/W = 216
  W = 8 (AVX-512 lanes)
  index: tile*(NVAR*W) + var*W + lane
         tile = flat >> 3,  lane = flat & 7
  vars:  [0]=ρ  [1]=ρu  [2]=ρv  [3]=ρw  [4]=E

phi_data_[NCELL]                 // flat double array, separate from Q

Interior range per axis:  ilo()=2 … ihi()=9  (ghost layers at 0,1 and 10,11)
Flat index: k*NB2² + j*NB2 + i   (i fastest)
```

On device, Q is stored as a flat `double*` with stride `NCELL` between variables:  
`d_Q[v * NCELL + flat]` — same layout as `data_[]` but pointer into `GpuPool`.

---

## 3. The "List" Pattern

Every physics subsystem follows the same three-phase protocol:

```
build(tree, pool, …)   — called from GpuGraphSolver::build(); allocates device
                          arrays, classifies topology, uploads metadata
exec(stream)           — called every RK3 stage; launches one or more kernels
~destructor            — cudaFree all device arrays
```

| List | Header | Source | Owns (device) | Notes |
|------|--------|--------|---------------|-------|
| `GpuGhostFillList` | gpu_ghost_fill.cuh | gpu_ghost_fill.cu | `GpuLeafGhostMeta* d_metas` | Same-level + C/F ghost fill; 4 build() overloads for BC/MPI variants |
| `GpuRhsList` | gpu_rhs.cuh | gpu_rhs.cu | `GpuLeafRhsMeta* d_metas`, `double* d_scratch_pool`, `double* d_rhs_pool` | k_prim_duc → k_rhs_conv → k_rhs_visc pipeline; `exec(s, zero_rhs)` |
| `GpuCflList` | gpu_cfl.cuh | gpu_cfl.cu | `GpuLeafCflMeta* d_metas`, `ull* d_dt_bits`, `double* d_dt` | Returns dt via warp-shuffle reduction; `exec(cfl,s)→double` |
| `GpuCfList` | gpu_cf.cuh | gpu_cf.cu | `GpuCfCoarseMeta* d_coarse`, `GpuCfFineMeta* d_fine`, `double* d_reg_pool` | Berger-Colella flux registers; 4-method exec (zero/undo/accum/apply) |
| `GpuSgsList` | gpu_sgs.cuh | gpu_sgs.cu | `GpuSgsMeta* d_metas` | Static Smagorinsky operator-split |
| `GpuDynSgsList` | gpu_sgs.cuh | gpu_sgs.cu | `GpuDynSgsMeta* d_metas`, `double* d_scratch` | Germano+Lilly dynamic SGS; 38016 doubles scratch per leaf |
| `GpuMpiHaloList` | gpu_mpi_halo.cuh | gpu_mpi_halo.cu | `vector<FaceEntry>`, `vector<RankBuf>` | D2H→MPI→H2D per stage; `active()` guards all calls |
| `GpuAcdiList` + `GpuPhiPool` | gpu_acdi.cuh | gpu_acdi.cu | `GpuAcdiLeafMeta* d_metas`, `double* d_phin_pool`, `double* d_rhs_pool` | Interface-compression phi transport; phi stored in GpuPhiPool (map CellBlock*→double*) |
| `GpuIbmList` | gpu_ibm.cuh | gpu_ibm.cu | `GpuIbmMeta* d_metas`, `int8_t* d_cell_type_pool`, `float* d_sdf/wnorm_pool`, `GhostEntry* d_ghosts`, `GhostEntry* d_solid_fills` | STL ghost-cell IBM; two-pass exec (SolidFill first, then ghost fill) |
| `GpuAmrList` | gpu_amr.cuh | gpu_amr.cu | prolong/restrict meta arrays | Built transiently inside gpu_regrid(); not held by GpuGraphSolver |

---

## 4. GpuGraphSolver Anatomy

**Defined in:** `include/cuda/gpu_graph.cuh`, `src/cuda/gpu_graph.cu`

### 4.1 Fields

```cpp
// ── Physics lists (all rebuilt on build()) ───────────────────────────
GpuGhostFillList ghost_list;
GpuRhsList       rhs_list;
GpuCflList       cfl_list;
GpuCfList        cf_list;         // Berger-Colella C/F (only active if AMR tree has C/F faces)
GpuSgsList       sgs_list;        // static Smagorinsky (enabled by set_gpu_sgs())
GpuDynSgsList    dyn_sgs_list_;   // dynamic Smagorinsky (enabled by set_gpu_dyn_sgs())
GpuMpiHaloList   mpi_halo_;       // MPI halos (enabled by set_mpi())
GpuAcdiList      acdi_list_;      // ACDI phi (enabled by set_gpu_acdi())
GpuPhiPool       phi_pool_;
GpuIbmList       ibm_list_;       // IBM ghost-cell (enabled by set_gpu_ibm())

// ── Feature flags & config ───────────────────────────────────────────
bool   acdi_enabled_ = false;   double acdi_ceps_ = 0.0;
bool   ibm_enabled_ = false;    GpuBvh* ibm_bvh_ptr_ = nullptr;
bool   sgs_enabled = false;     double sgs_Cs_ = 0.16;   double sgs_Pr_t_ = 0.9;
bool   dyn_sgs_enabled_ = false; double dyn_sgs_Pr_t_ = 0.9;
double duc_p_thr_ = 0.1;        double duc_blend_inv_ = 10.0;
MpiPartition* mpi_part_ = nullptr;
GpuSnapshotBuffer* snap_buf_ = nullptr;
std::array<int,6> bc_types_ = {0,0,0,0,0,0};

// ── Per-leaf RK3 state ───────────────────────────────────────────────
GpuRk3LeafMeta* d_rk3_metas = nullptr;   // [n_leaves] on device
double*          d_Qn_pool   = nullptr;   // Q^n snapshots for SSP-RK3
int              n_leaves    = 0;
std::vector<std::pair<CellBlock*, double*>> download_pairs;

// ── CUDA runtime objects ─────────────────────────────────────────────
cudaStream_t    stream      = nullptr;
cudaGraphExec_t graph_s1/s2/s3 = nullptr;  // three per-stage sub-graphs
bool            graph_valid = false;
```

### 4.2 Lifecycle

```
Constructor → create stream
set_gpu_*()  → configure feature flags (call BEFORE build())
build()      → rebuild all lists, allocate d_rk3_metas + d_Qn_pool,
               _destroy_graphs() (invalidates stale graphs)
advance()    → choose path below, return dt
download_q() → D2H copy all leaves after advance() if CPU access needed
upload_q()   → H2D copy (before GPU advance when CPU path ran previous step)
Destructor   → _destroy_graphs(), cudaFree all, destroy stream
```

---

## 5. Advance Loop — Three Paths

### 5.1 Graph replay path (fast; no AMR C/F faces, no MPI, no ACDI, no IBM)

```
advance()
├─ cfl_list.exec(cfl, stream) → dt
├─ [Stage 1]
│   cudaMemsetAsync(d_rhs_pool, 0, stream)
│   cudaGraphLaunch(graph_s1, stream)       ← contains: k_save_qn + ghost_fill + rhs + k_rk3s1 + positivity
├─ [Stage 2]
│   cudaMemsetAsync(d_rhs_pool, 0, stream)
│   cudaGraphLaunch(graph_s2, stream)       ← contains: ghost_fill + rhs + k_rk3s23(0.75, 0.25) + positivity
├─ [Stage 3]
│   cudaMemsetAsync(d_rhs_pool, 0, stream)
│   cudaGraphLaunch(graph_s3, stream)       ← contains: ghost_fill + rhs + k_rk3s23(1/3, 2/3) + positivity
├─ [Optional] SGS operator-split:  ghost_list.exec → sgs/dyn_sgs.exec
├─ [Optional] snapshot launch
└─ cudaStreamSynchronize(stream)
```

Graphs are **captured once** after the first explicit step. Graph capture is
**skipped** (falls back to explicit) when:
`mpi_halo_.active() || acdi_enabled_ || dyn_sgs_enabled_ || ibm_enabled_`

### 5.2 Explicit path (first step, or any feature that blocks capture)

```
_run_rk3_explicit(stream):
  acdi_list_.save_phin(stream)  [ACDI only]
  k_save_qn<<<n_leaves,256,0,stream>>>(d_rk3_metas)
  for stage in {s1, s2, s3}:
    cudaMemsetAsync(d_rhs_pool, 0, stream)
    mpi_halo_.exchange(stream)
    ghost_list.exec(stream)
    ibm_list_.exec(stream)           [IBM only: SolidFill then GhostFill]
    rhs_list.exec(stream, false)
    k_rk3s1 or k_rk3s23<<<n_leaves,256,0,stream>>>
    k_positivity_floor<<<n_leaves,256,0,stream>>>
    [ACDI stage updates]
```

### 5.3 AMR C/F path (`_advance_amr`; active when `cf_list.n_coarse > 0`)

```
_advance_amr(cfl):
  [destroys any cached graphs]
  cfl_list.exec → dt   (MPI allreduce if multi-rank)
  cf_list.zero_regs(stream)
  acdi_list_.save_phin(stream)  [ACDI only]
  for stage in {s1, s2, s3}:
    cudaMemsetAsync(d_rhs_pool, 0, stream)
    [Stage 1 only] k_save_qn
    mpi_halo_.exchange(stream)
    ghost_list.exec(stream)
    ibm_list_.exec(stream)           [IBM only]
    rhs_list.exec(stream, false)
    cf_list.undo_coarse_flux(stream)
    cf_list.accum_fine_flux(stream, stage_weight)
    k_rk3s1 or k_rk3s23
    k_positivity_floor
    [ACDI stage updates]
  cf_list.apply_correction(stream, dt)
  [Optional] SGS operator-split
  snapshot launch
  cudaStreamSynchronize(stream)
```

### 5.4 Decision tree

```
advance()
  └─ n_leaves == 0?  → return 1e300
  └─ cf_list.n_coarse > 0?  → _advance_amr()
  └─ graph_valid && !mpi_halo_.active()?  → graph replay
  └─ else  → _run_rk3_explicit(); [capture graphs if conditions allow]
```

---

## 6. IBM Subsystem Detail

```
GpuBvh  (include/cuda/gpu_bvh.cuh + src/cuda/gpu_bvh.cu)
  build(StlMesh)  — CPU BVH build (median-split), upload BVH nodes + triangle SoA to device
  d_nodes         — BvhNode flat array (root at index 0)
  d_v{0,1,2}{x,y,z}, d_n{x,y,z}  — triangle geometry SoA

  __device__ bvh_sdf(nodes, v0x..v2z, tnx..tnz, px,py,pz, &nx,&ny,&nz) → float
    Iterative DFS, stack[64], returns signed distance + outward wall normal
    Sign: positive = fluid/exterior, negative = solid/interior

GpuIbmList  (include/cuda/gpu_ibm.cuh + src/cuda/gpu_ibm.cu)
  build(tree, pool, bvh)
    1. Run bvh_sdf per cell → d_cell_type_pool (0=FLUID, 1=SOLID, 2=IBM_GHOST)
    2. Pass 1: build GhostEntry for each IBM_GHOST cell
         image point: I = G − 2·sdf·n_outward  (into fluid)
         trilinear 8-cell stencil at I, wall_bc = 0 (NoSlip) or 2 (Isothermal)
    3. Pass 2: build GhostEntry for each SOLID cell
         image point: I = G − 2·max(sdf, -1.5h)·n_outward (clamped, into fluid)
         wall_bc = 3 (SolidFill: copy Q_I directly)
    Upload d_ghosts, d_solid_fills to device

  exec(stream)
    1. k_ghost_fill_ibm<<<n_solid_fills/256, 256, 0, stream>>>(d_solid_fills, n_solid_fills)
    2. k_ghost_fill_ibm<<<n_ghosts/256, 256, 0, stream>>>(d_ghosts, n_ghosts)
    (SolidFill first so stencil reads for IBM_GHOST see fresh fluid values)

GhostEntry:
  ghost_ptr     — base of ghost cell in d_Q  (stride: NCELL between vars)
  stencil[8]    — base pointers of 8 trilinear stencil cells
  w[8]          — trilinear weights
  wall_bc       — 0=NoSlip+Adiabatic, 2=Isothermal, 3=SolidFill
  u/v/w/T_wall  — prescribed wall values
```

Wall BC application in `k_ghost_fill_ibm`:
- **bc=3 (SolidFill):** `ghost[v] = Q_I[v]` (copy interpolated image-point state)
- **bc=0 (NoSlip+Adiabatic):** `u_g = 2u_w − u_I`, `T_g = T_I` (zero heat flux)
- **bc=2 (Isothermal):** `u_g = 2u_w − u_I`, `T_g = 2T_w − T_I`

IBM is **excluded from graph capture** — pointer arrays are rebuilt on regrid.

---

## 7. File Index (GPU subsystem)

### Headers (`include/cuda/`)

| File | Defines | Notes |
|------|---------|-------|
| `gpu_constants.cuh` | `GPU_NB/NG/NCELL/NVAR/GAMMA`; `gpu_cell_idx()`, `gpu_ilo/ihi()`, `gpu_sutherland()` | Include in every kernel file |
| `gpu_pool.hpp` | `GpuPool` | Device memory arena; maps CellBlock*→double* |
| `gpu_ghost_fill.cuh` | `GpuLeafGhostMeta`, `GpuGhostFillList` | Same-level + C/F BC |
| `gpu_rhs.cuh` | `GpuLeafRhsMeta`, `GpuRhsList`, `GpuReconScheme` enum | WENO5-Z / TENO5-A / TENO7-A dispatch |
| `gpu_cfl.cuh` | `GpuLeafCflMeta`, `GpuCflList` | CFL reduction |
| `gpu_cf.cuh` | `GpuCfCoarseMeta`, `GpuCfFineMeta`, `GpuCfList` | Berger-Colella registers |
| `gpu_sgs.cuh` | `GpuSgsMeta`, `GpuSgsList`, `GpuDynSgsMeta`, `GpuDynSgsList` | Static + dynamic Smagorinsky |
| `gpu_mpi_halo.cuh` | `GpuMpiHaloList`, `FaceEntry`, `RankBuf` | GPU-buffer MPI halos |
| `gpu_acdi.cuh` | `GpuPhiPool`, `GpuAcdiLeafMeta`, `GpuAcdiList` | ACDI phi transport |
| `gpu_bvh.cuh` | `BvhNode`, `GpuBvh`; `bvh_sdf()` (device inline) | STL BVH; header-inline device code |
| `gpu_ibm.cuh` | `GpuIbmMeta`, `GhostEntry`, `GpuIbmList` | IBM ghost-cell lists |
| `gpu_amr.cuh` | `GpuProlongMeta`, `GpuRestrictMeta`, `GpuSensorMeta`, `GpuAmrList` | Prolongation/restriction kernels |
| `gpu_snapshot.hpp` | `GpuSnapshotBuffer`, `GpuBlockMetrics`, `SnapLeafMeta` | Slice extraction + metrics |
| `gpu_graph.cuh` | `GpuRk3LeafMeta`, `GpuGraphSolver` | Master solver; includes all above |
| `gpu_check.cuh` | `CUDA_CHECK` macro | Error checking |
| `gpu_meta_buffer.cuh` | `gpu_upload_meta<T>()`, `gpu_upload_scalar<T>()` | Metadata upload helpers |
| `gpu_adjoint_rhs.cuh` | adjoint RHS kernel decl | D10 |
| `gpu_bn.cuh` | Baer-Nunziato two-phase | G6 |
| `gpu_lts.cuh` | Berger-Oliger LTS | G5 |
| `gpu_gmres.cuh` | GPU GMRES | D4 |
| `gpu_p1.cuh` | P1 radiation | D6 |
| `gpu_source.cuh` | Arrhenius + species | D5 |
| `gpu_wmles.cuh` | WMLES algebraic wall model | D7 |

### Sources (`src/cuda/`)

Each `.cu` implements its matching `.cuh`. Notable extras:

| File | Notes |
|------|-------|
| `gpu_graph.cu` | `GpuGraphSolver` — build, advance, _capture_graphs, _advance_amr |
| `gpu_pool.cu` | `GpuPool` alloc/free/upload/download |
| `gpu_snapshot.cu` | Slice + metric reduction kernels |
| `gpu_imex.cu` | `advance_imex()` — IMEX-Euler (NOT in `_GPU_NS`; linked only for t32) |
| `gpu_solver.cu` | Solver dispatch utilities |

---

## 8. Key Structs Quick Reference

```cpp
// ── Grid ─────────────────────────────────────────────────────────────
// CellBlock (include/mesh/cell_block.hpp)
  double data_[216*5*8];   // AoSoA: [tile][var][lane]  69 120 B
  double phi_data_[1728];  // ACDI phase field
  double ox,oy,oz,h,hy,hz; // origin + cell sizes

// ── GPU per-leaf metadata (uploaded at build()) ──────────────────────
// GpuRk3LeafMeta (gpu_graph.cuh)
  double* d_Q;    // current state (in/out)
  double* d_Qn;   // Q^n checkpoint
  const double* d_RHS;

// GpuIbmMeta (gpu_ibm.cuh)
  double* d_Q;
  int8_t* d_cell_type;  // [1728]  0=FLUID 1=SOLID 2=IBM_GHOST
  float*  d_sdf;        // [1728]  signed distance (+=fluid)
  float*  d_wnx/y/z;   // [1728]  outward wall normals
  float   ox,oy,oz, hx,hy,hz;

// GhostEntry (gpu_ibm.cuh)
  double*  ghost_ptr;     // base of ghost cell's d_Q
  double*  stencil[8];    // trilinear stencil cell bases
  float    w[8];          // trilinear weights
  uint8_t  wall_bc;       // 0=NoSlip+Adiab, 2=Isotherm, 3=SolidFill
  float    u/v/w/T_wall;

// BvhNode (gpu_bvh.cuh)
  float aabb_min[3], aabb_max[3];
  int   left;   // >=0: child idx; <0: leaf, tri=~left
  int   right;
```

---

## 9. IGpuSolver Interface

```cpp
// include/solver/ns_solver.hpp
struct IGpuSolver : TimeIntegrator {
    // Core
    virtual double advance(const BlockTree&, double cfl) = 0;
    virtual void   build(const BlockTree&, const GpuPool&, int bc_type=0) = 0;
    virtual void   build_faces(const BlockTree&, const GpuPool&,
                               const std::array<int,6>& bc_types);
    virtual void   download_q(const BlockTree&) const = 0;
    virtual void   upload_q() const = 0;

    // Physics (all default no-op)
    virtual void set_gpu_sgs(double Cs, double Pr_t) {}
    virtual void set_gpu_dyn_sgs(double Pr_t) {}
    virtual void set_gpu_acdi(double ceps) {}
    virtual void set_gpu_ibm(GpuBvh*, uint8_t bc,
                             float uw, float vw, float ww, float Tw) {}
    virtual void set_ducros(double p_thr, double blend_inv) {}
    virtual void set_mpi(MpiPartition*) {}
    virtual void set_snapshot_buffer(GpuSnapshotBuffer*) {}

    // AMR (default false)
    virtual bool gpu_regrid(BlockTree&, GpuPool&, int bc_type,
                            int cfg_max_level,
                            float refine_thr=0.05f,
                            float coarsen_thr=0.01f) { return false; }
};
```

`set_gpu_*()` must be called **before** `build()`. They only set flags/params.
`build()` is the only method that allocates device memory and uploads metadata.

---

## 10. Gate Test Map (GPU)

| Target | Gate | What it tests |
|--------|------|---------------|
| t24 | P8.6 | CUDA Graph re-capture on regrid |
| t25 | P9.1 | GPU vs CPU solution correctness |
| t26 | P10-A3 | NSSolver GPU dispatch |
| t27 | P-SGS-GPU | Static Smagorinsky |
| t28 | P-MPI-GPU | MPI halo exchange |
| t29 | D1 | GPU-native AMR (no D2H for Q) |
| t30 | D2 | CUDA-aware MPI halos |
| t31 | D3a | TENO5-A reconstruction |
| t32 | D4 | GPU GMRES (IMEX) |
| t33 | D5 | Arrhenius reactive flow |
| t34 | D6 | P1 radiation |
| t35 | D7 | WMLES algebraic wall model |
| t36 | — | GPU snapshot (slice + metrics) |
| t37 | D3b | TENO7-A reconstruction |
| t40 | D9 | NSCBC outflow/inflow BC |
| t43 | G1 | ACDI phi transport |
| t44 | G2 | Adjoint convective RHS |
| t45 | G3 | Dynamic Smagorinsky (Germano+Lilly) |
| t46 | G4 | ODE mixing-length wall model |
| t47 | G5 | Berger-Oliger LTS |
| t48 | G6 | Baer-Nunziato two-phase |
| t49 | I5–I9 | IBM STL import + ghost-cell BC |

Note: t28, t30 are MPI-gated (built only when `HAVE_MPI`). GPU targets are **not** in the `ba` (build-all) target.

---

## 11. Build System Sketch

```cmake
# _GPU_NS = full GPU solver source set
set(_GPU_NS
  src/cuda/gpu_graph.cu  src/cuda/gpu_ghost_fill.cu  src/cuda/gpu_rhs.cu
  src/cuda/gpu_cfl.cu    src/cuda/gpu_pool.cu         src/cuda/gpu_cf.cu
  src/cuda/gpu_sgs.cu    src/cuda/gpu_mpi_halo.cu     src/cuda/gpu_amr.cu
  src/cuda/gpu_snapshot.cu  src/cuda/gpu_acdi.cu
  src/cuda/gpu_bvh.cu    src/cuda/gpu_ibm.cu          src/models/stl_loader.cpp
  src/solver/ns_solver.cpp  src/mesh/block_tree.cpp  ...
)

# add_nvcc_gate(TARGET t49 BIN t49_ibm_gpu SRCS test_t49_gpu_ibm.cu ${_GPU_NS})
# simulate_gpu links _GPU_NS
```

---

*Last updated: 2026-06-01 (IBM STL, t49 complete)*
