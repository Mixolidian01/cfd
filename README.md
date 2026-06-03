# GPU-Native Compressible CFD Solver

A production-grade, fully GPU-native solver for 3-D compressible Navier-Stokes on octree AMR grids. All physics, AMR decisions, flux corrections, and I/O run on the GPU; there is no CPU round-trip in the advance loop.

For the full technical reference (governing equations, numerics, GPU architecture, developer APIs) see [`docs/reference.md`](docs/reference.md).

---

## Capabilities

| Category | Feature |
|---|---|
| **Schemes** | WENO5-Z, TENO5-A, TENO7-A reconstruction; HLLC-ES entropy-stable Riemann flux; KEP (Pirozzoli 2010) |
| **Time integration** | SSP-RK3; CFL-limited dt; Berger-Oliger local time stepping (LTS); IMEX-ARK for stiff viscous terms |
| **AMR** | GPU-native octree (max_level configurable); Berger-Colella flux correction at C/F faces; SBP-SAT penalty (optional) |
| **Turbulence** | Smagorinsky SGS; dynamic Smagorinsky (Germano–Lilly); algebraic Reichardt WMLES; ODE mixing-length TBLE WMLES |
| **IBM** | Ghost-cell IBM from STL files; BVH closest-point + winding-number sign (robust for non-watertight meshes); curvature-driven AMR sensor; startup pre-refinement |
| **Multiphase** | ACDI 5-equation compressible (Allaire 2002); Baer-Nunziato two-phase |
| **FSI** | Rigid-body 6-DOF IBM (t53); Dowell flutter benchmark (t54) |
| **Boundary conditions** | Periodic, no-slip wall (adiabatic / isothermal), open (zero-gradient), NSCBC (inflow/outflow) |
| **Adjoint** | Discrete adjoint RHS; reversed SSP-RK3; JAX `custom_vjp` wiring (Python) |
| **I/O** | In-situ 2-D live-stream (browser); VTK XML binary; checkpoint (ZFP-compressed optional); per-leaf metrics |
| **Multi-GPU** | CUDA-aware MPI halo exchange; NCCL collective reductions |
| **Radiation** | P1 diffusion (D6) |
| **Chemistry** | Single-step Arrhenius with IMEX sub-cycling (D5) |

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| CUDA toolkit | 12.0+ | `nvcc`, cuBLAS, Cooperative Groups |
| CMake | 3.22+ | |
| C++ compiler | GCC 12+ or Clang 16+ | C++20 required |
| MPI | OpenMPI 4+ or MPICH 4+ | Optional; enables t28, t30, multi-GPU |
| LZ4 | 1.9+ | Optional; checkpoint compression |
| ZFP | 0.5.5+ | Optional; lossy checkpoint compression |
| pybind11 | 2.11+ | Optional; Python bindings (D11) |
| ONNX Runtime | 1.16+ | Optional; neural SGS model (D11) |

CMake auto-detects optional libraries; missing ones disable the relevant feature silently.

---

## Build

```bash
# Minimal (GPU solver + CPU tests)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With all optional libraries
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLZ4_LIBRARY=/path/to/liblz4.so \
  -DZFP_DIR=/path/to/zfp \
  -Dpybind11_DIR=/path/to/pybind11
cmake --build build -j$(nproc)
```

Build products:
- `build/simulate_gpu` — GPU solver driver
- `build/simulate` — CPU solver driver
- `build/t01` … `build/t54` — individual gate tests

---

## Quick Start

### Taylor-Green vortex (GPU)

```json
{
  "domain_L": 6.283185307,
  "ic": "tgv",
  "ic_ma": 0.1,
  "cfl": 0.4,
  "t_end": 20.0,
  "bc": "periodic",
  "scheme": "teno7a",
  "max_level": 2,
  "regrid_interval": 10,
  "vtk_prefix": "tgv",
  "vtk_interval": 1.0,
  "checkpoint_save": true,
  "checkpoint_interval": 5.0
}
```

```bash
./build/simulate_gpu build/sim.json
```

### Sphere IBM (GPU + AMR)

```json
{
  "domain_L": 2.0,
  "ic": "uniform",
  "ic_rho0": 1.0,
  "ic_ma": 0.2,
  "cfl": 0.3,
  "t_end": 5.0,
  "bc": "open",
  "max_level": 3,
  "regrid_interval": 5,
  "ibm_enabled": true,
  "ibm_stl_path": "sphere.stl",
  "ibm_wall_bc": "noslip",
  "vtk_prefix": "sphere",
  "vtk_interval": 0.5
}
```

On startup the solver automatically pre-refines the mesh around IBM geometry until no further refinement is needed (curvature signal + flat-surface signal), then begins time integration.

---

## Configuration Reference

All keys are optional unless marked **required**. Unrecognised keys are silently ignored.

### Domain and IC

| Key | Type | Default | Description |
|---|---|---|---|
| `domain_L` | float | 1.0 | Domain side length [m] (cubic domain) |
| `ic` | string | `"uniform"` | Initial condition: `"uniform"`, `"tgv"` (Taylor-Green vortex) |
| `ic_ma` | float | 0.1 | Mach number for TGV IC |
| `ic_v0` | float | 1.0 | Reference velocity for IC |
| `ic_rho0` | float | 1.0 | Reference density for IC |
| `volume_size` | int | — | Override leaf count at initialisation |

### Time Integration

| Key | Type | Default | Description |
|---|---|---|---|
| `cfl` | float | 0.4 | Acoustic CFL number |
| `t_end` | float | 1.0 | End time [s] |
| `max_steps` | int | ∞ | Maximum number of timesteps |
| `use_lts` | bool | false | Enable Berger-Oliger local time stepping |
| `lts_ratio` | float | 2.0 | LTS refinement ratio per level |
| `use_imex` | bool | false | Enable IMEX-ARK (implicit viscous; requires GPU GMRES) |

### Boundary Conditions

| Key | Type | Default | Description |
|---|---|---|---|
| `bc` | string | `"periodic"` | Global BC applied to all 6 faces: `"periodic"`, `"wall"`, `"open"`, `"nscbc"` |
| `bc_xlo` / `bc_xhi` | string | — | Per-face BC override (same values as `bc`) |
| `bc_ylo` / `bc_yhi` | string | — | Per-face BC override |
| `bc_zlo` / `bc_zhi` | string | — | Per-face BC override |
| `nscbc_p_inf` | float | 101325 | NSCBC reference pressure [Pa] |

### AMR

| Key | Type | Default | Description |
|---|---|---|---|
| `max_level` | int | 0 | Maximum refinement level (0 = no AMR) |
| `regrid_interval` | int | 10 | Steps between regrid calls |
| `refine_levels` | int | — | Initial refinement depth at startup |

### Numerics

| Key | Type | Default | Description |
|---|---|---|---|
| `scheme` | string | `"teno7a"` | Reconstruction scheme: `"weno5z"`, `"teno5a"`, `"teno7a"` |
| `mu` | float | 0.0 | Dynamic viscosity [Pa·s] (0 = inviscid) |
| `sutherland` | bool | false | Use Sutherland viscosity law instead of constant μ |

### Physics: SGS

| Key | Type | Default | Description |
|---|---|---|---|
| `sgs` | string | `"none"` | SGS model: `"none"`, `"smag"`, `"dyn_smag"`, `"neural"` |
| `sgs_cs` | float | 0.1 | Smagorinsky constant (static Smag only) |
| `sgs_prt` | float | 0.9 | SGS Prandtl number |

### Physics: Body Force and WMLES

| Key | Type | Default | Description |
|---|---|---|---|
| `body_force_x` / `y` / `z` | float | 0.0 | Uniform body force components [N/m³] |
| `wmles_enabled` | bool | false | Enable wall-modelled LES |
| `wmles_nu` | float | 1.5e-5 | Kinematic viscosity for WMLES [m²/s] |
| `wmles_ym` | float | — | Matching-point height [m] |
| `wmles_use_ode` | bool | false | Use ODE TBLE wall model instead of algebraic Reichardt |
| `wmles_axis` | int | 1 | Wall-normal axis (0=x, 1=y, 2=z) |

### Physics: Chemistry

| Key | Type | Default | Description |
|---|---|---|---|
| `combustion` | bool | false | Enable single-step Arrhenius combustion |
| `combustion_A` | float | — | Pre-exponential factor [1/s] |
| `combustion_Tact` | float | — | Activation temperature [K] |
| `combustion_Q` | float | — | Heat release per unit mass [J/kg] |
| `combustion_nsub` | int | 4 | IMEX sub-steps per RK3 stage |

### Physics: ACDI Multiphase

| Key | Type | Default | Description |
|---|---|---|---|
| `acdi` | bool | false | Enable ACDI 5-equation compressible multiphase |
| `acdi_ceps` | float | 0.1 | Interface sharpening parameter ε |
| `acdi_gamma_a` / `_b` | float | — | Adiabatic indices for each phase |
| `acdi_pinf_a` / `_b` | float | — | Stiffened-gas reference pressures [Pa] |

### IBM (Immersed Boundary Method)

| Key | Type | Default | Description |
|---|---|---|---|
| `ibm_enabled` | bool | false | Enable GPU ghost-cell IBM |
| `ibm_stl_path` | string | — | Path to `.stl` geometry file (**required** when IBM enabled) |
| `ibm_wall_bc` | string | `"noslip"` | Wall BC: `"noslip"` (adiabatic), `"isothermal"`, `"solid_fill"` |
| `ibm_u` / `ibm_v` / `ibm_w` | float | 0.0 | Wall velocity components [m/s] (moving IBM) |
| `ibm_T_wall` | float | 300.0 | Wall temperature [K] (isothermal only) |
| `ibm_h_surf` | float | auto | Flat-surface target resolution [m]. If 0 (default), auto-computed as mean STL triangle edge length / 5. Set explicitly to override or to disable flat-surface refinement trigger. |

### FSI (Fluid-Structure Interaction)

| Key | Type | Default | Description |
|---|---|---|---|
| `fsi_mass` | float | — | Rigid body mass [kg] |
| `fsi_Ix` / `Iy` / `Iz` | float | — | Moments of inertia [kg·m²] |
| `fsi_prescribed` | bool | false | Prescribed (kinematic) motion; no fluid forces on body |

### Output

| Key | Type | Default | Description |
|---|---|---|---|
| `vtk_prefix` | string | — | VTK output file prefix (empty = no VTK) |
| `vtk_interval` | float | — | Physical time between VTK snapshots [s] |
| `metrics_dir` | string | — | Directory for per-step metrics CSV files |
| `metrics_interval` | int | 1 | Steps between metrics writes |
| `metrics_residual` | bool | false | Include residual norms in metrics |
| `checkpoint_save` | bool | false | Save binary checkpoint files |
| `checkpoint_interval` | float | — | Physical time between checkpoints [s] |
| `stream_port` | int | — | TCP port for in-situ browser stream (empty = disabled) |
| `stream_var` | int | 0 | Variable index to stream (0=ρ, 1=ρu, 4=E) |
| `stream_axis` | int | 1 | Slice axis (0=x, 1=y, 2=z) |
| `stream_pos` | float | 0.5 | Normalised slice position along axis |
| `stream_stride` | int | 1 | Pixel stride (subsampling) |

### Diagnostics

| Key | Type | Default | Description |
|---|---|---|---|
| `verbose` | bool | false | Extra console output each step |
| `diag_interval` | int | 100 | Steps between diagnostic prints |

---

## Validation Gates

Run the full CPU test suite (≈30 min):
```bash
cmake --build build -t ba
```

Run individual gates by target name:
```bash
cmake --build build -t t49   # IBM + BVH + winding-number sign
cmake --build build -t t50   # WMLES channel Re_τ=395
cmake --build build -t t52   # VTK XML binary writer
cmake --build build -t t53   # Moving-wall IBM + rigid 6-DOF, Theodorsen Cl
cmake --build build -t t54   # Dowell flutter onset
```

GPU gates are **not** included in `ba` (requires a CUDA device):
```bash
cmake --build build -t t24   # CUDA Graph re-capture on regrid
cmake --build build -t t25   # GPU vs CPU correctness
cmake --build build -t t29   # GPU-native AMR
cmake --build build -t t35   # WMLES GPU kernel
cmake --build build -t t49   # IBM BVH + AMR sensor (I5–I12, W5)
cmake --build build -t t50   # Channel WMLES (C50)
```

See `docs/reference.md §G10` for the complete gate-to-phase mapping.

---

## Repository Layout

```
apps/
  simulate_gpu.cu    GPU solver driver (JSON → GpuGraphSolver)
  simulate.cpp       CPU solver driver

include/
  cuda/              GPU headers: gpu_graph.cuh, gpu_ibm.cuh, gpu_bvh.cuh, …
  mesh/              BlockTree, CellBlock, AMR operators
  physics/           EOS, WENO5-Z, TENO, reconstruction schemes
  solver/            NSSolver, SolverConfig, BNSolver
  models/            StlMesh loader, chemistry, WMLES, SGS

src/
  cuda/              GPU implementations (*.cu)
  mesh/
  schemes/
  solver/
  models/

tests/
  cuda/              GPU gate tests (test_t24.cu … test_t54.cu)
  python/            Python binding tests
  *.cpp              CPU gate tests

docs/
  reference.md       Full technical reference (physics, numerics, GPU API)
  dev_phases.md      Phase status and commit hashes
  dev_log.md         Change log with measured performance
  perf/              Nsight Compute baseline profiles
  superpowers/
    plans/           Implementation plans (per session)
    specs/           Design specifications
```

---

## Branches

| Branch | Purpose |
|---|---|
| `to_refactor` | Architecture reference — do not modify |
| `to_develop` | GPU development (active) |

---

## Key References

- Borges et al. (2008) — WENO5-Z improved weighting
- Fu et al. (2019) — TENO7-A targeted essentially non-oscillatory
- Chandrashekar (2013) — Entropy-stable HLLC-ES flux
- Berger & Colella (1989) — AMR flux register correction
- Germano et al. (1991) + Lilly (1992) — Dynamic Smagorinsky
- Thompson (1987) + Poinsot & Lele (1992) — NSCBC
- Mittal & Iaccarino (2005) — Ghost-cell IBM
- Van Oosterom & Strackee (1983) — Winding-number solid angle (IBM sign)
- Baer & Nunziato (1986) — Two-phase BN solver
- Lee & Moser (2015) — DNS channel flow Re_τ=395 reference
