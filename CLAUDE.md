# CLAUDE.md — Refactor Branch

This file provides guidance to Claude Code (claude.ai/code) when working on the
`to_refactor` branch of this repository.

## Role

Act as an expert in: compressible CFD, multiphase flows, shock-capturing schemes,
turbulence modelling, C++20, CUDA/GPU programming, and modern HPC software design.
Numerical correctness and physical fidelity are non-negotiable.

## Branching Rules

- Working branch: `to_refactor` (diverges from `to_debug` at commit 2f16dae)
- `to_debug` is the correctness reference. Do NOT modify it.
- Every refactor phase must leave all gate tests green before committing.
- Gate command: `cmake --build build -t ba`

## Current Architecture (to_debug baseline)

The solver has a well-defined four-layer stack:

```
Layer 0  linalg.hpp/cpp          — Kahan BLAS-1, CG, multigrid
Layer 1  cell_block.hpp          — CellBlock (SoA, NB=8, NG=2, NB2=12, NCELL=1728)
         block_tree.hpp/cpp      — BlockTree octree, AMR prolong/restrict
         amr_operators.cpp       — fill_cf_ghosts (must stay in libblock)
Layer 2  operators.hpp/cpp       — HLLC-ES, WENO5-Z RHS, compute_rhs, tree_rhs
Layer 3  ns_solver.hpp/cpp       — SSP-RK3, regrid, flux correction
         gpu_graph.cuh/cu        — CUDA Graph RK3 (P8.6), positivity floor (P16.1)
         gpu_rhs.cuh/cu          — WENO5-Z GPU RHS
         gpu_ghost_fill.cuh/cu   — GPU ghost fill
```

Key constants (do not change without updating both CPU and GPU headers):
| NB=8 | NG=2 | NB2=12 | NCELL=1728 | NVAR=5 | GAMMA=1.4 |

## Target Architecture (to_refactor goal)

The refactor adds three orthogonal layers **on top of** the existing stack without
replacing it wholesale:

```
LAYER P — Physics functors  (include/physics/)
           Structs with operator(). __host__ __device__. template <Axis DIR>.
           Carry only physics state (γ, p_inf, μ…). No execution knowledge.

LAYER C — Concept contracts  (include/concepts.hpp)
           C++20 concept definitions. Applied at every template boundary.
           RiemannFlux, SPOperator, EquationOfState, BoundaryCondition.
           Property flags: is_entropy_stable, is_conservative, is_skew_symmetric.

LAYER E — Execution backend tags  (include/execution.hpp)
           CPUSerial, GPUCuda tags. Backend selected once at solver startup via
           factory. No physics or contract knowledge.
```

These three layers compose independently. A new flux scheme lives only in Layer P.
A new GPU backend lives only in Layer E. The mathematical guarantees live only in
Layer C and are independent of execution.

## Migration Phases (execute in order)

### R0 — Enable C++20
- `cmake -S . -B build -DCMAKE_CXX_STANDARD=20`
- Verify: nvcc 12.4+ with `--std=c++20`. Check `nvcc --version`.
- Zero functional change. All tests must still pass.

### R1 — Concept layer (additive, non-breaking)
- Create `include/concepts.hpp` with `RiemannFlux`, `EquationOfState`,
  `BoundaryCondition` concepts and compile-time property flags.
- Add `static_assert` checks at existing call sites in `operators.cpp` as
  verification only. No behavioral change.
- Gate: all t1–t26 pass.

### R2 — Physics functor extraction (refactor, not rewrite)
- Move `hllc_flux`, `hllc_es_flux_t`, `weno5_face_t` from `operators.cpp`
  into structs in `include/physics/`.
- Existing free functions become one-liner wrappers calling the functor.
- The concept from R1 is applied to each functor.
- Gate: all t1–t26 pass.

### R3 — BC variant dispatch (replaces if/else BC chains)
- Replace BC-type enum dispatch in ghost fill with
  `std::variant<PeriodicBC, WallBC, OpenBC, ContactAngleBC>`.
- `std::visit` replaces every BC if/else chain.
- Gate: all t1–t26 pass, contact angle T16a/T16b still pass.

### R4 — Backend tag dispatch (formalises existing CPU/GPU split)
- Create `include/execution.hpp` with `CPUSerial{}` and `GPUCuda{}` tags.
- Wrap existing CPU operator calls behind `CPUSerial` dispatch.
- Wrap existing GPU kernel calls behind `GPUCuda` dispatch.
- Factory: `make_solver(cfg)` reads scheme+backend from `SolverConfig` and
  returns a `std::unique_ptr<ISolver>` pointing to the correct pre-compiled type.
- Gate: all t1–t26 pass including GPU gates t19–t26.

### R5 — Instantiation matrix (Strategy 3 — enables runtime scheme selection)
- Pre-compile all supported (Flux × Recon × EOS) combinations in a dedicated
  translation unit `src/instantiation_matrix.cpp`.
- Factory dispatches to the correct specialisation once at startup.
- Eliminates all scheme-selection runtime branches from GPU kernels.
- Gate: all t1–t26 pass. Add new test verifying scheme selection from JSON.

### R6 — mdspan block access (optional, breaking — last)
- Replace `Q[v][idx]` raw access with `std::mdspan` views in `CellBlock`.
- Axis rotation for `template <Axis DIR>` kernels becomes a zero-copy
  layout policy instead of manual index arithmetic.
- This phase changes every block access in the codebase — do not start until
  R0–R5 are complete and stable.
- Gate: all t1–t26 pass. Check for regression in T08 convergence rate (≥1.8).

## Code Rules

### Mathematical / Numerical
1. Interior flux: entropy-conservative (Chandrashekar) or entropy-stable (HLLC-ES).
   Never plain Roe without entropy fix.
2. Convective operator: Pirozzoli (2010) split-form for the compressible case.
3. AMR C/F flux correction: undo_cf and accumulate_cf must use the same
   reconstruction as accumulate_face for exact Berger-Colella cancellation.
4. Positivity floor: ρ≥1e-12, p≥1e-12 after every RK3 stage on both CPU and GPU.
5. Regrid must run at the TOP of advance(), before zero_flux_registers.

### C++
6. No raw owning pointers — use `std::unique_ptr` or `GpuArray<T>`.
7. No axis-specific duplicate functions — one `template <Axis DIR>` only.
8. No scheme-selection `if` branches inside GPU kernels — dispatch at launch site.
9. No `virtual` in device-callable code — use CRTP or `std::variant`.
10. Concept constraints applied at every template boundary (kernel launch sites,
    factory). Never inside `__global__` kernels (CUDA limitation).

### CUDA
11. `cudaDeviceSynchronize()` forbidden in the advance loop.
12. Halo exchanges use `cudaMemcpyAsync` on the solver stream; TMA pipelines
    are a R6+ upgrade.
13. Kernel roofline target: ≥50% of peak memory bandwidth (Nsight Compute).

### Testing
14. Every new functor must have a `CPUSerial` unit test before any GPU kernel.
15. Sod shock tube must give bit-identical results for Axis::X, Axis::Y, Axis::Z.
16. T08 isentropic vortex convergence rate must remain ≥1.8 after every phase.
17. Mass conservation drift < 1e-12 over 20 steps (checked by t4 T04).

## Validation Gate Commands

```bash
cmake --build build -t ba       # all gates (t1–t26)
cmake --build build -t t3       # operators — T08 convergence rate
cmake --build build -t t4       # ns_solver — 28 sub-tests
cmake --build build -t t24      # CUDA graph (P8.6)
cmake --build build -t t25      # GPU vs CPU correctness (P9.1)
cmake --build build -t t26      # NSSolver GPU dispatch (P10-A3)
```

## Key References

- Pirozzoli (2010) — split-form compressible convective operator
- Chandrashekar (2013) — entropy-conservative flux
- Einfeldt et al. (1991) — positivity-preserving schemes
- Zhang & Shu (2010) — maximum-principle positivity floor
- Berger & Colella (1989) — AMR flux register correction
- Trias et al. (2014) — symmetry-preserving discretization
- Huang & Johnsen (2024) — consistent-conservative ACDI multiphase
- Del Rey Fernández et al. — SBP-SAT at AMR coarse-fine interfaces
