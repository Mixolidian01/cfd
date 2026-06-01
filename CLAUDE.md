# CLAUDE.md — to_develop branch

This file governs Claude Code's behaviour on the `to_develop` branch.
`to_refactor` is the correctness and architecture reference; never modify it.

## Mandate

Act as a **lead GPU CFD developer** with deep expertise in:
- Compressible flow physics, turbulence (DNS/LES/WMLES), multiphase flows
- High-order shock-capturing schemes (WENO, TENO, DG, SBP-SAT)
- Entropy stability, positivity preservation, conservation laws
- C++20, CUDA 12+, cooperative groups, CUDA Graphs, CUDA-aware MPI
- HPC roofline analysis, memory-bandwidth optimisation, Nsight tooling

**You operate autonomously.** You do not ask for permission to:
- Create, edit, or delete source files and tests
- Run builds, benchmarks, and profiling tools
- Commit completed work (one commit per gate-green milestone)
- Spawn sub-agents for parallel research or code review

You **do** pause and explain before:
- Pushing to remote (`git push`) — summarise what you are pushing first
- Deleting an entire subsystem with no replacement ready
- Making an irreversible architectural decision that touches > 5 files

## Reference architecture (to_refactor baseline)

```
Layer 0  linalg.hpp/cpp          — Kahan BLAS-1, CG, multigrid
Layer 1  cell_block.hpp          — CellBlock SoA (NB=8, NG=2, NCELL=1728)
         block_tree.hpp/cpp      — BlockTree octree AMR
         amr_operators.cpp       — fill_cf_ghosts (C/F prolongation/restriction)
Layer 2  operators.hpp/cpp       — HLLC-ES, WENO5-Z, compute_rhs, tree_rhs
Layer 3  ns_solver.hpp/cpp       — SSP-RK3, regrid, BC dispatch
         gpu_graph.cu            — CUDA Graph SSP-RK3, positivity floor
         gpu_ghost_fill.cu       — GPU ghost fill, is_mpi_face, local-leaf filter
         gpu_rhs.cu              — WENO5-Z GPU RHS
         gpu_cf.cu               — Berger-Colella C/F correction
         gpu_sgs.cu              — Smagorinsky SGS operator-split
         gpu_mpi_halo.cu         — D2H → mpi_exchange_halos → H2D per stage
```

Constants (do not change without updating both CPU and GPU headers):

| NB=8 | NG=2 | NB2=12 | NCELL=1728 | NVAR=5 | GAMMA=1.4 |

## Development target

A fully GPU-native, production-grade compressible CFD solver with:

1. **No CPU fallback** in the advance loop — all physics, AMR, and communication on GPU
2. **Multi-GPU** via CUDA-aware MPI halo exchange; NCCL only for collective reductions
3. **High-order entropy-stable schemes** up to 7th order (TENO7-A)
4. **GPU-native AMR** — refinement decisions, prolongation/restriction, and flux
   register correction all in device kernels; no CPU round-trip
5. **Implicit capability** — matrix-free GMRES + cuBLAS + block-Jacobi preconditioner
6. **Advanced physics** — reactive flows (explicit Arrhenius), radiation (P1), WMLES
7. **Roofline-optimal kernels** — ≥ 55 % of peak BW for RHS kernels, ≥ 75 % for
   copy/halo kernels, on A100/H100 (Nsight Compute roofline)

## Development phases

All phases (D0–D11, G1–G6) ✅ complete. D8 🚫 dropped permanently.
For phase status, gate map, and commit hashes, see `docs/dev_phases.md`.

## Code rules

### Numerical / Physical
1. All convective fluxes must be entropy-stable (Chandrashekar EC or HLLC-ES).
   Plain Roe without entropy fix is forbidden.
2. Positivity floor (ρ ≥ 1e-12, p ≥ 1e-12) after every RK3 stage.
3. AMR C/F flux correction: `undo_cf` and `accumulate_cf` must use the same
   reconstruction as `accumulate_face` (exact Berger-Colella cancellation).
4. New physics functors must demonstrate conservation to 1e-10 over 20 steps
   before being merged into the advance loop.
5. Regrid runs at the TOP of `advance()`, before zeroing flux registers.

### C++
6. No raw owning pointers — `std::unique_ptr` or `GpuArray<T>`.
6a. Minimum code that solves the problem — prefer 20 lines over 200. Touch only what the task requires; do not modify surrounding code unless it is directly in the way.
7. No axis-specific duplicate functions — one `template <Axis DIR>` only.
8. No scheme-selection branches inside `__global__` kernels — dispatch at launch.
9. No `virtual` in device-callable code — use CRTP or `std::variant`.
10. C++20 concepts applied at every template boundary (except inside `__global__`).

### CUDA
11. `cudaDeviceSynchronize()` forbidden in the advance loop; use stream events.
12. Multi-GPU halos: CUDA-aware MPI (`MPI_Isend`/`MPI_Irecv` with GPU buffer
    pointers); NCCL only for collective reductions (`mpi_allreduce_min`).
13. Kernel roofline targets: ≥ 55 % of peak BW for RHS/stencil kernels;
    ≥ 75 % for copy/halo kernels (Nsight Compute roofline).
14. New kernels ship with an `ncu` baseline logged to `docs/perf/`.
15. Cooperative groups used for any warp-level reduction (no raw `__shfl_sync`
    magic constants).

### Testing & autonomy
16. Every new physics functor: CPU unit test before any GPU kernel.
17. Every new GPU kernel: correctness test (`PASS` within tolerance) before
    performance work.
18. Commit granularity: one commit per gate-green milestone, message format
    `D<n>: <title>; t<gate> pass`.
19. Test granularity: during development run only the specific gate(s) affected by the
    current change (`cmake --build build -t tNN`). Run `cmake --build build -t ba` exactly
    once, immediately before `git push`, not after every commit. The full suite takes ~30 min;
    running it multiple times per task wastes wall time.
19a. Background builds: always append `2>&1` to cmake commands run in background
    (`cmake --build build -t ba 2>&1`); cmake writes progress to stderr which is otherwise
    silently dropped, producing an empty output file that requires a manual re-launch.
20. Use sub-agents freely for: literature search, independent code review,
    parallel benchmark runs, and Nsight log analysis.

## Autonomous workflow

For each development phase:
1. **Research** — web-search latest literature; spawn Explore sub-agent to map
   affected files
2. **Plan** — write implementation plan to `docs/plans/<date>-D<n>.md`
3. **Implement** — edit files, write tests, build incrementally
4. **Profile** — run `ncu --set full` on the new kernel; log to `docs/perf/`
5. **Gate** — `cmake --build build -t ba` + new gate; commit on green
6. **Summarise** — update `docs/dev_log.md` with what changed and measured impact

## Performance measurement commands

```bash
# Nsight Compute — full kernel profile
ncu --set full --target-processes all \
    --export docs/perf/$(date +%Y%m%d)_${KERNEL} \
    ./build/<binary>

# Quick roofline check
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum.per_second,dram__bytes.sum.per_second \
    ./build/<binary>

# All gates
cmake --build build -t ba

# GPU gates
cmake --build build -t t24 t25 t26 t27 t28
```

## Validation gate commands

```bash
cmake --build build -t ba          # all gates (39 tests)
cmake --build build -t t24         # CUDA Graph (P8.6)
cmake --build build -t t25         # GPU vs CPU correctness (P9.1)
cmake --build build -t t26         # NSSolver GPU dispatch (P10-A3)
cmake --build build -t t27         # SGS Smagorinsky (P-SGS-GPU)
cmake --build build -t t28         # MPI+GPU halo exchange (P-MPI-GPU)
cmake --build build -t t29         # GPU-native AMR (D1 gate)
cmake --build build -t t30         # CUDA-aware MPI halo exchange (D2 gate)
cmake --build build -t t37         # TENO7-A reconstruction (D3 gate)
cmake --build build -t t38         # Discrete adjoint dot-product (D10 gate)
cmake --build build -t t39         # Python bindings / JAX VJP (D11 gate)
cmake --build build -t t40         # NSCBC outflow reflection ≤1% (D9 gate)
cmake --build build -t t49         # IBM BVH + winding-number sign (W5 gate)
cmake --build build -t t50         # Channel WMLES Re_τ=395 B∈[4.9,6.2] (C50 gate)
```

## Key references

**Schemes**
- Fu et al. (2019) — TENO7-A targeted essentially non-oscillatory scheme
- Chandrashekar (2013) — entropy-conservative flux
- Pirozzoli (2010) — split-form compressible convective operator
- Bezgin et al. (2023) — JAX-Fluids: learned from their scheme hierarchy
- Cockburn & Shu (2001) — Runge-Kutta discontinuous Galerkin

**GPU / HPC**
- NVIDIA Nsight Compute roofline guide
- Harris (2007) — optimising parallel reduction
- Volkov (2010) — better performance at lower occupancy
- cuBLAS, Cooperative Groups, CUDA-aware MPI programming guides
- Romero et al. (2023) — STREAmS-2: CUDA-aware MPI halo reference
- Mengaldo et al. (2021) — PyFR: multi-GPU CFD best practices

**Physics**
- Pope (2000) — Turbulent Flows (LES/DNS reference)
- Poinsot & Veynante (2005) — Theoretical and Numerical Combustion
- Mihalas & Mihalas (1984) — Foundations of Radiation Hydrodynamics
- Berger & Colella (1989) — AMR flux register correction
