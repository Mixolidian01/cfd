# Metrics & Monitoring System — Design Spec

**Date:** 2026-06-01  
**Status:** Approved, pending implementation

---

## Goal

Add a composable metrics and monitoring system to the GPU CFD solver covering:
- Residual / convergence tracking
- Surface force / pressure / shear extraction on IBM bodies
- Spatial probes (point, line, plane-averaged profile, volume integral)
- Runtime CSV output + binary field dumps for post-processing

No external dependencies beyond the existing codebase. Output format: plain binary + CSV, readable from Python via `numpy.fromfile`.

---

## Architecture

### Pattern: composable per-category monitors + thin bus

Each category is its own class following the existing `build()/exec()` List pattern. A thin `MetricsBus` aggregates them and is called at two hook points per advance step:

```
advance()
  ├── [post-RHS]  bus.on_rhs(d_RHS, t, dt)       → ResidualMonitor
  └── [post-step] bus.on_step(d_Q, t, dt, step)   → SurfaceMonitor
                                                   → ProbeMonitor
                                                   → FieldDumper
```

```
MetricsBus
├── ResidualMonitor   →  GpuResidualList   (k_residual_norm)
├── SurfaceMonitor    →  GpuSurfaceList    (k_surface_forces)
├── ProbeMonitor      →  GpuProbeList      (k_probe_interp, k_plane_avg,
│                                           k_enstrophy_reduce)
└── FieldDumper       →  (CPU-side; reads pinned staging buffer after sync)

Shared output layer:
├── CsvWriter         →  header-on-construct, append-per-step, flush-per-line
└── BinDumper         →  binary field file + fields.csv index
```

`MetricsBus` is owned by `NSSolver` (CPU path) and `GpuGraphSolver` (GPU path). Built from `SolverConfig::MetricsConfig` in `init()` / `build()`.

### New files

| File | Responsibility |
|------|---------------|
| `include/metrics/metrics_bus.hpp` | `MetricsBus`, `IMonitor` interface |
| `include/metrics/residual_monitor.hpp` | `ResidualMonitor` |
| `include/metrics/surface_monitor.hpp` | `SurfaceMonitor` |
| `include/metrics/probe_monitor.hpp` | `ProbeMonitor` |
| `include/metrics/field_dumper.hpp` | `FieldDumper`, `CsvWriter`, `BinDumper` |
| `include/cuda/gpu_metrics.cuh` | `GpuResidualList`, `GpuSurfaceList`, `GpuProbeList` — structs + kernel declarations |
| `src/metrics/metrics_bus.cpp` | `MetricsBus` impl |
| `src/cuda/gpu_metrics.cu` | All GPU kernels |

### Modified files

| File | Change |
|------|--------|
| `include/solver/ns_solver.hpp` | Add `MetricsConfig`, `SurfaceConfig`, `ProbeConfig` to `SolverConfig`; add `MetricsBus` member to `NSSolver` |
| `src/solver/ns_solver.cpp` | Call `bus_.on_rhs()` and `bus_.on_step()` in advance loop |
| `src/cuda/gpu_graph.cu` | Same two hook calls in `GpuGraphSolver::advance()` |
| `apps/simulate.cpp` | Parse `"metrics"` block from JSON |
| `apps/template.json` | Add `"metrics"` section |
| `CMakeLists.txt` | Add `t51` gate; add `src/metrics/` and `src/cuda/gpu_metrics.cu` to build |

---

## Component 1: ResidualMonitor

### What it computes

L2 norm of the RHS after each RHS evaluation, one value per conserved variable:

$$\|R\|_2^{(v)} = \sqrt{\frac{1}{N_\text{cells}} \sum_\text{cells} \text{RHS}[v][i]^2}$$

Also tracks the **relative residual** normalised by the norm at step 0:

$$\|R\|_\text{rel}^{(v)} = \|R\|_2^{(v)} / \|R\|_2^{(v)}\big|_{t=0}$$

For steady-state problems, relative residuals decay to a small ε. For time-accurate runs, they remain O(1) — this is expected.

### GPU kernel

`k_residual_norm(d_RHS, n_leaves, partial_sums[n_leaves * NVAR])`:
- One block per leaf, NVAR passes (or one pass over NVAR in registers)
- Each thread accumulates `RHS[v][flat]^2` for its cells
- Block reduction via cooperative groups → writes per-leaf partial sum to pinned host memory
- CPU folds the `n_leaves` partial sums and takes sqrt

Shares the existing `GpuSnapshotBuffer` pinned allocation pattern.

### Output

`<output_dir>/residuals.csv`:
```
step, t, L2_rho, L2_rhou, L2_rhov, L2_rhow, L2_E, L2_rho_rel, L2_rhou_rel, L2_rhov_rel, L2_rhow_rel, L2_E_rel
```

Appended every `residual_interval` steps.

---

## Component 2: SurfaceMonitor

### Approach

Ghost-cell IBM force integration. For each `IBM_GHOST` cell, the wall normal, wall-to-image-point distance, and cell width are precomputed at build time from the BVH. At exec time, `k_surface_forces` computes per-ghost-cell contributions:

| Quantity | Formula |
|----------|---------|
| Wall pressure | `p_w = p(Q_I)` — pressure at image point I (in fluid) |
| Wall shear | `τ_w = μ · (u_I − u_wall) / d` — tangential component |
| Projected area | `A = h²` where `h = L / (NB · 2^level)` |
| Pressure force | `f_p = −p_w · n · A` |
| Viscous force | `f_v = τ_w · A` |
| Moment arm | `r = x_ghost − ref_point` |
| Moment contribution | `m = r × (f_p + f_v)` |

Contributions atomically accumulate into a 6-double per-body device array `{Fx, Fy, Fz, Mx, My, Mz}`. One D2H copy after `exec`.

If `SurfaceConfig::rho_ref`, `u_ref`, `A_ref` are non-zero, non-dimensional coefficients are also written:
`Cx = Fx / (0.5 · rho_ref · u_ref² · A_ref)` etc.

### `GpuSurfaceList` build-time data

For each `IBM_GHOST` cell, stores:
- `n[3]` — outward wall normal (from BVH at build time)
- `d` — `|sdf_eff|` — half the image-point distance
- `h` — cell width at this leaf level
- `ghost_Q_ptr` — pointer into `d_Q` for this ghost cell
- `stencil[8]`, `w[8]` — image-point trilinear stencil (same as `GpuGhostEntry`)

**Known limitation:** area weighting uses `h²` per ghost cell, not the actual projected STL triangle area. This is first-order accurate and standard for ghost-cell IBM. A triangle-integration pass can replace it later without changing the interface.

### Output

`<output_dir>/forces_<name>.csv`:
```
step, t, Fx, Fy, Fz, Mx, My, Mz[, Cx, Cy, Cz]
```

Appended every `surface_interval` steps.

---

## Component 3: ProbeMonitor

### Probe types

#### Point probe

User specifies `(x, y, z)`. GPU kernel: linear scan over leaf meta-array to find the containing leaf; trilinear interpolation into `d_Q`. Any quantity from the table below. One row per step.

Output: `probe_<name>.csv` — `step, t, <quantity>`

#### Line probe

N evenly spaced points between `(x0,y0,z0)` and `(x1,y1,z1)`. Runs as N point probes in one kernel launch.

Output: `probe_<name>.csv` — `step, t, s, <quantity>` (s = arclength ∈ [0,1]), N rows per step.

#### Plane-averaged profile

Averages all interior cells in `n_slabs` slabs perpendicular to a chosen axis. One block per leaf; in-slab cells reduce to shared memory; CPU accumulates across leaves.

Produces mean(quantity) as a function of slab centre position. The channel flow t50 profile extraction moves here.

Output: `profile_<name>.csv` — `step, t, pos, mean_<quantity>`, `n_slabs` rows per step.

#### Volume integral

Integrates `f(Q)` over a user-specified AABB (or full domain if AABB is not set).

Output: `integral_<name>.csv` — `step, t, value`

### Quantity table

| Name | Description | Stencil |
|------|-------------|---------|
| `rho`, `u`, `v`, `w` | Density, velocity components | 1 cell |
| `p`, `T`, `mach` | Pressure, temperature, Mach | 1 cell |
| `omega_x`, `omega_y`, `omega_z` | Vorticity components | 3×3 FD |
| `omega_mag` | Vorticity magnitude `|ω|` | 3×3 FD |
| `Q_cr` | Q-criterion: `½(‖Ω‖² − ‖S‖²)` | 3×3 FD |
| `schlieren` | Numerical shadowgraph `exp(−k|∇ρ|/max)` | 3×3 FD |
| `enstrophy` | `|ω|²` (volume integral only) | 3×3 FD |
| `ke` | Kinetic energy `½ρ|u|²` | 1 cell |
| `mass` | Density `ρ` (volume integral: ∫ρ dV) | 1 cell |

Quantities requiring a stencil (`omega_*`, `Q_cr`, `schlieren`, `enstrophy`) reuse the existing `snap_scalar_val` logic from `gpu_snapshot.cu`.

### GPU kernels

| Kernel | Purpose |
|--------|---------|
| `k_probe_interp` | One thread per probe point; leaf scan + trilinear interpolation; handles stencil quantities by reading 3×3×3 neighbourhood |
| `k_plane_avg` | One block per leaf; shared-memory slab reduction across in-slab cells |
| `k_enstrophy_reduce` | One block per leaf; 6-point FD for ω, then ω² partial sum |

---

## Component 4: FieldDumper

### Binary field dump format

```
Header (64 bytes total, little-endian):
  uint32  magic    = 0xCFD10001      (offset  0, 4 bytes)
  uint32  step                       (offset  4, 4 bytes)
  double  t                          (offset  8, 8 bytes)
  uint32  n_leaves                   (offset 16, 4 bytes)
  uint32  nvar                       (offset 20, 4 bytes)  5 or 7
  uint32  NB2   = 12                 (offset 24, 4 bytes)
  uint32  NCELL = 1728               (offset 28, 4 bytes)
  uint8   reserved[32]               (offset 32, 32 bytes) — pad to 64

Per-leaf block × n_leaves:
  double  origin[3]          (x0, y0, z0 of leaf corner)
  double  h                  (cell width at this level)
  double  Q[nvar * NCELL]    (variable-major: Q[v*NCELL + flat_index])
                             vars 0-4: ρ, ρu, ρv, ρw, E  (always)
                             var  5:   |ω|               (if dump_derived)
                             var  6:   Q-criterion        (if dump_derived)
```

D2H uses a pre-allocated pinned staging buffer (reuses the checkpoint path in `simulate.cpp`). One `cudaMemcpyAsync` per leaf into staging buffer, then a single `fwrite`.

### Index file

`<output_dir>/fields.csv`:
```
step, t, filename
0, 0.000, fields_0000000.bin
1000, 0.412, fields_0001000.bin
...
```

### Python loader (3 lines)

```python
import numpy as np, struct
def load_field(path):
    with open(path, 'rb') as f:
        magic, step, t, n_leaves, nvar, NB2, NCELL = struct.unpack('<IIdIIII', f.read(32))
        f.seek(64)  # skip 32-byte reserved pad
        leaves = []
        for _ in range(n_leaves):
            origin = np.frombuffer(f.read(24), dtype=np.float64)
            h = struct.unpack('<d', f.read(8))[0]
            Q = np.frombuffer(f.read(nvar * NCELL * 8), dtype=np.float64).reshape(nvar, NB2, NB2, NB2)
            leaves.append({'origin': origin, 'h': h, 'Q': Q})
    return leaves
```

### Runtime CSV files

All monitors share the thin `CsvWriter` helper (constructor writes header; `append(row...)` formats + appends + flushes):

| File | Columns | Source |
|------|---------|--------|
| `globals.csv` | `step, t, dt, mass, ke, etot, rho_min, rho_max` | Extends `StepDiag` |
| `residuals.csv` | `step, t, L2_rho, …, L2_E, L2_rho_rel, …` | ResidualMonitor |
| `forces_<name>.csv` | `step, t, Fx, Fy, Fz, Mx, My, Mz[, Cx, Cy, Cz]` | SurfaceMonitor |
| `probe_<name>.csv` | `step, t[, s], <quantity>` | ProbeMonitor |
| `profile_<name>.csv` | `step, t, pos, mean_<quantity>` | ProbeMonitor |
| `integral_<name>.csv` | `step, t, value` | ProbeMonitor |
| `fields.csv` | `step, t, filename` | FieldDumper |

---

## SolverConfig additions

```cpp
// in include/solver/ns_solver.hpp

struct SurfaceConfig {
    std::string name;
    std::array<double,3> ref_point = {0,0,0};
    double rho_ref = 0.0, u_ref = 0.0, A_ref = 0.0;  // 0 = skip Cx/Cy/Cz
};

struct ProbeConfig {
    enum class Type { POINT, LINE, PLANE_AVG, VOLUME_INTEGRAL };
    Type        type      = Type::POINT;
    std::string name;
    std::string quantity  = "rho";
    std::array<double,3> p0 = {}, p1 = {};
    int N       = 1;
    int axis    = 1;      // 0=x, 1=y, 2=z for PLANE_AVG
    int n_slabs = 32;
};

struct MetricsConfig {
    int  global_interval   = 10;
    int  residual_interval = 0;   // 0 = disabled
    int  surface_interval  = 0;
    int  probe_interval    = 0;
    int  dump_interval     = 0;
    bool dump_derived      = false;
    std::string output_dir = ".";
    std::vector<SurfaceConfig> surfaces;
    std::vector<ProbeConfig>   probes;
};
```

`template.json` additions:

```json
"metrics": {
    "global_interval"   : 10,
    "residual_interval" : 0,
    "surface_interval"  : 0,
    "probe_interval"    : 0,
    "dump_interval"     : 0,
    "dump_derived"      : false,
    "output_dir"        : ".",
    "surfaces": [],
    "probes":   []
}
```

---

## Testing — Gate t51

File: `tests/cuda/test_t51_metrics.cu`  
CMake target: `add_nvcc_gate(t51 tests/cuda/test_t51_metrics.cu)`

| Sub-gate | Setup | Pass criterion |
|----------|-------|---------------|
| M1 ResidualMonitor | Isentropic vortex, 5 steps | All 5 L2 norms finite and positive; relative residuals ≤ 1.0 at step 0 |
| M2 SurfaceMonitor | Sphere IBM, uniform pressure p₀ everywhere | `|Fx| + |Fy| + |Fz| < 0.01 · p₀ · A_sphere` (zero net force by divergence theorem) |
| M3a ProbeMonitor point | IC: `ρ = 1 + 0.1·sin(πx)sin(πy)sin(πz)`, probe at grid-aligned point | Extracted value matches IC to 1e-10 |
| M3b ProbeMonitor plane_avg | Channel-like IC `u = tanh(y)` | Plane-averaged u(y) matches input to 1e-10 |
| M4 FieldDumper | Dump at step 0, reread with `std::fread` | File size = `64 + n_leaves·(4 + NVAR·NCELL)·8` bytes; magic = `0xCFD10001`; Q arrays bit-identical |

---

## What this does NOT include

- Time-averaging / Reynolds stress tensors (second-moment statistics) — deferred; needs accumulator arrays proportional to n_leaves
- Spectral diagnostics (kinetic energy spectrum) — deferred; requires FFT (cuFFT dependency)
- Triangle-accurate surface integration (replace h² area weighting with projected STL area) — deferred; noted in tech_debt.md
- Adjoint/sensitivity output — separate concern
