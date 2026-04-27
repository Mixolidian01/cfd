# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role

Act as an expert in Fluid mechanics, Mathematics. You also have strong and proven skills in C++, CUDA programming, HPC and numerical schemes and skills.

## Solver features

Aim to design a CFD solver, handling compressible single and multiphase flows, with/without shocks. It would entirely or for the most part works on GPU, with AMR block-structure, symmetry preserving features (in regards to F.X Trias works). It aims to out-perform or at least equal proven CFD codes

## Regarding your answers

All your answers must be numbered or indexed. In each of your answer, show first the number or index of the answer. If possible, detail your answer by giving the physical, mathematical and numerical aspects.

## Regarding code writing

Always ensure physical, mathematical and numerical correctness. The code must follow the best C++ practices of code writing and implementation. Check the compatibility with the code features against the best approaches that you will propose. Look at other domains such as computer aided graphics, video games, finance, HPC for approaches that can be applied to the code.


## Build & Test Commands

**One-time setup** (run from repo root):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

**Daily workflow:**
```bash
cmake --build build -t b     # incremental build of all targets
cmake --build build -t ba    # build + run all gate tests (verbose)
cmake --build build -t t1    # Step 1: test_linalg
cmake --build build -t t2    # Step 2: test_block
cmake --build build -t t3    # Step 3: test_operators
cmake --build build -t t4    # Step 4: test_ns (time loop + conservation)
cmake --build build -t t5    # Step 5: GPU gate (requires nvcc)
cmake --build build -t t6    # Step 6: test_amr6 (AMR conservation)
cmake --build build -t t7    # Step 7: test_step7 (SGS + checkpoint + VTK)
cmake --build build -t t12   # Step 12: test_streamer (P6.5 live-feed gate)
cmake --build build -t t13   # Step 13: test_mpi (P7.1 MPI 2-rank Sod)
cmake --build build -t t14   # Step 14: test_weno5_cf (P7.2 5th-order C/F ghost fill)
cmake --build build -t t15   # Step 15: test_ibm (P7.3 ghost-cell IBM)
cmake --build build -t t16   # Step 16: test_chemistry (P7.4 Arrhenius chemistry)
cmake --build build -t t17   # Step 17: test_p75_dns_tgv (P7.5 TGV DNS quantitative)
cmake --build build -t t18   # Step 18: test_wmles (P7.6 WMLES log-law + ODE)
```

**Running a simulation:**
```bash
cmake --build build -t sim                     # Sod shock tube demo (apps/sod.json)
build/simulate apps/taylor_green.json          # Taylor-Green vortex
build/simulate my_run.json                     # custom JSON config
```
Open `http://localhost:8080` for the 2D slice viewer, `http://localhost:8080/volume` for the WebGPU 3D volume renderer (Chrome 113+ required).

**Build types:**
- `RelWithDebInfo` (default): `-O2 -g` — recommended for development
- `Debug`: `-O0 -g -fsanitize=address,undefined`
- `Release`: `-O3 -march=native -ffast-math`

## Architecture

This is a compressible Navier-Stokes CFD solver with Adaptive Mesh Refinement (AMR) and optional GPU support. The design is strictly layered — each layer may only depend on layers below it.

### Layer 0 — Linear Algebra (`linalg.hpp/cpp`)
Kahan-compensated BLAS-1 primitives, a Conjugate Gradient solver, and a 3-level geometric multigrid V-cycle (Neumann BC). No physics knowledge. The multigrid is wired for future IMEX implicit viscous solves but is currently unused by the solver.

### Layer 1 — Block Tree & AMR (`cell_block.hpp`, `block_tree.hpp/cpp`, `amr_operators.cpp`)

**CellBlock** is a single 8×8×8 patch. Storage is SoA: one `std::vector<double>` per conserved variable (`NVAR=5`: ρ, ρu, ρv, ρw, E). Each axis has `NG=1` ghost layer, giving a 10×10×10 physical allocation (`NCELL=1000`).

**BlockTree** is an octree of `BlockNode`s. Internal nodes have no `CellBlock`; only leaves do. The 8 children of a node are stored at contiguous indices `first_child + oct` where `oct = ix | (iy<<1) | (iz<<2)` (bit0=x, bit1=y, bit2=z). Contiguity is guaranteed by `alloc_node_group(8)` — never individual `alloc_node()` calls.

**Ghost fill** is a 2-pass protocol: same-level faces copy directly; coarse-fine faces call `fill_cf_ghosts()` from `amr_operators.cpp` (which is why `amr_operators.cpp` is compiled into `libblock`, not `libns_solver`).

**Flux registers** (Berger-Colella): each leaf holds one `FluxRegister` per face. Fine-leaf fluxes accumulate during RK3 (weighted by SSP-RK3 coefficients 1/6, 1/6, 2/3). `apply_flux_correction(dt)` is the last write to every coarse boundary cell in each time step.

### Layer 2 — Discrete Operators (`operators.hpp/cpp`)

`hllc_flux(L, R, axis)` is the HLLC Riemann solver (1st-order). `compute_rhs(blk, rhs)` assembles convective + viscous RHS for one block. `tree_rhs(tree, rhs, periodic, w)` fills ghosts, loops all leaves, and accumulates fine fluxes with weight `w`. `tree_cfl_dt(tree, cfl)` returns the global minimum dt.

Viscous RHS uses Sutherland-law viscosity and a Newtonian stress tensor with central differences; ghost cells must be filled before this is called.

### Layer 3 — Time Loop, I/O & Streaming (`ns_solver.hpp/cpp`, `sgs.hpp/cpp`, `checkpoint.hpp/cpp`, `vtk_writer.hpp/cpp`, `live_streamer.hpp/cpp`)

`NSSolver::advance()` executes one time step in this exact order:
```
regrid(Q^n)  →  zero_flux_registers  →  RK3 stages  →  apply_flux_correction(dt)
```
The tree topology is frozen for the entire zero→accumulate→correct sequence. Regrid at the *end* is a hard bug (Rule 006).

SSP-RK3 (Shu-Osher form):
```
Q^(1) = Q^n + (1/6)·dt·L(Q^n)
Q^(2) = 3/4·Q^n + 1/4·Q^(1) + (1/6)·dt·L(Q^(1))
Q^(n+1) = 1/3·Q^n + 2/3·Q^(2) + (2/3)·dt·L(Q^(2))
```

`SmagorinskyModel` is a plug-in (`virtual SGSModel::apply()`), applied post-RK3 via operator splitting.

**LiveStreamer** (`include/live_streamer.hpp`, `src/live_streamer.cpp`) is an optional Phase 6 plugin. Attach via `solver.set_streamer(&streamer)` before `run()`. Endpoints:
- `GET /` → 2D slice viewer (viridis, canvas API)
- `GET /stream` → chunked binary 2D frames (magic `0xCFD00001`)
- `GET /volume` → WebGPU 3D ray-marched volume viewer (Chrome 113+)
- `GET /volume-stream` → chunked binary 3D frames (magic `0xCFD00003`, N³ r32float)
- `POST /config` → JSON `{var, axis, pos}` hot-config

Wire format: `[4-byte LE length][body]`. Both 2D and 3D use LZ4+uint16 compression when `HAVE_LZ4=1`. The 3D volume thread activates only when a volume client is connected; otherwise `snapshot()` skips `build_volume()`.

**`apps/simulate`** is the standalone JSON-driven runner. All `SolverConfig` fields, five named ICs, `refine_levels`, checkpoint in/out, and `LiveStreamer` are configurable from a flat JSON file. See `apps/sod.json` and `apps/taylor_green.json` for examples.

## CMake Library Layout

```
linalg  ←  block (block_tree.cpp + amr_operators.cpp)
                ←  operators
                        ←  ns_solver (ns_solver.cpp + flux_register.cpp + sgs.cpp
                                      + checkpoint.cpp + vtk_writer.cpp + live_streamer.cpp)
                                ←  simulate (apps/simulate.cpp)
```

`amr_operators.cpp` lives in `libblock` (not `libns_solver`) because `block_tree.cpp` calls `fill_cf_ghosts()` directly. Do not move it.

## Hard Rules (from `to_avoid_bugs.md`)

**Rule 001 — Include paths.** All `#include` for project headers in `src/*.cpp` must use `../include/<name>.hpp`. Never use a bare `"<name>.hpp"` path.

**Rule 002 — Library placement.** Compile a `.cpp` into the lowest-layer library that calls its symbols. Never list the same `.cpp` in multiple `add_library()` calls (ODR violation).

**Rule 003 — Contiguous allocation.** Whenever N objects must be addressed as `base + 0..N-1`, allocate them with `alloc_node_group(N)`. Individual `alloc_node()` calls corrupt the tree after a free-list recycle.

**Rule 004 — Ghost fill for derived arrays.** Any per-cell array derived from Q (e.g. `mu_t`) that is used in stencils must have its ghost cells filled with the same BC as Q, immediately after the interior loop.

**Rule 005 — Verify call sites.** After defining a protocol function (flux accumulation, ghost fill, restriction…), grep the codebase to confirm it has at least one call site before committing.

**Rule 006 — Regrid ordering.** `regrid()` must run at the **top** of `advance()`, before `zero_flux_registers` and the RK3 stages. Placing it after `apply_flux_correction()` causes `coarsen()` to overwrite flux-corrected values.

## Key Constants (`cell_block.hpp`, `gpu_constants.cuh`)

| Name | Value | Meaning |
|------|-------|---------|
| `NB` | 8 | Interior cells per axis |
| `NG` | 1 | Ghost layers per face |
| `NB2` | 10 | Total cells per axis (NB + 2·NG) |
| `NCELL` | 1000 | Total cells per block (NB2³) |
| `NVAR` | 5 | Conserved variables (ρ, ρu, ρv, ρw, E) |
| `GAMMA` | 1.4 | Ratio of specific heats (air) |
| `R_GAS` | 287.058 | Specific gas constant [J/(kg·K)] |

**WENO5 requires `NG=2`.** Changing `NG` propagates to `NB2`, `NCELL`, all GPU constants, all ghost-fill kernels, and all index arithmetic — treat it as a project-wide gate change (Phase 3).

## Active Development Context

- git address: `https://github.com/Mixolidian01/cfd.git`
- Current branch: `to_debug`
- All Phases 0–7 complete: 25 gate tests pass (t1–t7, t8–t12, tb1–tb9, t13–t18)
- `roadmap.md` is the authoritative Phase 0–4 plan
- `todo.md` tracks Phases 0–7 status (all ✅)
- `answers_register.md` logs session Q&A history
- `to_avoid_bugs.md` records all derived rules (append on each new misbehaviour)
