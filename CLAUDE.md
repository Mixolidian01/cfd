# CLAUDE.md — to_develop branch

This file governs Claude Code's behaviour on the `to_develop` branch.
`to_refactor` is the correctness and architecture reference; never modify it.

## Mandate

Act as a **lead GPU CFD developer** with deep expertise in:
- Compressible flow physics, turbulence (DNS/LES/WMLES), multiphase flows
- High-order shock-capturing schemes (WENO, TENO, DG, SBP-SAT)
- Entropy stability, positivity preservation, conservation laws
- C++20, CUDA 12+, cooperative groups, CUDA Graphs, NCCL
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
2. **Multi-GPU** via NCCL P2P, topology-aware halo exchange, no MPI CPU staging
3. **High-order entropy-stable schemes** up to 7th order (TENO7-A or DGSEM-p4)
4. **GPU-native AMR** — refinement decisions, prolongation/restriction, and flux
   register correction all in device kernels; no CPU round-trip
5. **Implicit capability** — Newton-Krylov + GPU-resident GMRES for stiff viscous/
   reactive problems
6. **Advanced physics** — reactive flows, radiation (P1), wall-modelled LES
7. **Roofline-optimal kernels** — ≥ 70 % of peak memory bandwidth on A100/H100

## Development phases (execute in priority order)

### D0 — Baseline verification (before any new feature)
- Ensure all gates from `to_refactor` pass on this branch: `cmake --build build -t ba`
- Establish GPU performance baseline: run `ncu` on `k_rhs_conv`, record achieved BW%
- Gate: 32/32 CPU tests + t24–t28 GPU tests pass; baseline BW% logged to
  `docs/perf/baseline_roofline.md`

### D1 — GPU-native AMR (eliminate CPU round-trip on regrid)
- Move refinement criteria evaluation to a GPU reduction kernel
- Move prolongation and restriction to `__global__` kernels in `gpu_amr.cu`
- `BlockTree::refine()` becomes a host-side tree-topology update only;
  data movement stays device-to-device
- Gate: all existing AMR tests pass; new `t29_gpu_amr_native` verifies
  refine/coarsen cycle with no `cudaMemcpy D2H` during advance

### D2 — NCCL multi-GPU halo exchange (replace MPI CPU staging)
- Replace `GpuMpiHaloList` D2H→CPU→H2D path with NCCL `ncclSend/ncclRecv`
  on peer streams; remove `cudaStreamSynchronize` from the hot path
- Keep MPI fallback for non-NVLink environments (`#ifdef HAVE_NCCL` guard)
- Gate: `t28` still passes; new `t30_nccl_halo` measures halo latency ≤ 50 % of
  MPI-CPU baseline on 2-GPU NVLink node

### D3 — TENO7-A reconstruction (higher spectral resolution)
- Implement `teno7_face_t` functor in `include/physics/teno7.hpp` following
  Fu et al. (2019); `__host__ __device__`, `template <Axis DIR>`
- Replace WENO5-Z face interpolation in `gpu_rhs.cu` via the R4 backend tag;
  WENO5-Z remains available for comparison
- Gate: T08 isentropic vortex convergence rate ≥ 3.8 (was ≥ 1.8); Sod shock
  tube bit-identical for X/Y/Z; no new positivity violations

### D4 — GPU-resident GMRES for implicit viscous solve
- Implement preconditioned GMRES in `src/cuda/gpu_gmres.cu` using cuBLAS
  for BLAS-1/2 and a block-Jacobi preconditioner
- Wire into IMEX-ARK path: explicit convective RHS stays on GPU stream;
  implicit viscous Helmholtz solve uses GPU-resident GMRES
- Gate: Poiseuille flow viscous solution matches analytical to 1e-6;
  GMRES iteration count ≤ 50 for μ = 1e-3, Re = 100

### D5 — Reactive flows (single-step Arrhenius)
- Add species transport scalar φ_s alongside ACDI phase field
- Arrhenius source S = A·ρ·Y·exp(-Ea/RT) as GPU functor in
  `include/physics/arrhenius.hpp`; operator-split after RK3 stage
- Gate: 1D detonation wave speed matches Chapman-Jouguet to 1 % over 200 steps

### D6 — P1 radiation transport
- Diffusion-limit radiation via elliptic G-equation solved each step with
  GPU-resident CG (reuse linalg.hpp patterns on device)
- Coupled energy source term ±κ(aT⁴ − G/c) in RK3 RHS
- Gate: Marshak wave penetration depth matches analytical within 2 %

### D7 — Wall-modelled LES (algebraic + ODE wall model)
- Algebraic Reichardt law wall model (`wmles_algebraic`) as default
- Optional ODE wall model (thin-boundary-layer equations on GPU)
- Gate: turbulent channel Re_τ = 395; u⁺ log-law intercept B ∈ [4.8, 5.5];
  wake-region u⁺ within 5 % of DNS at y⁺ > 50

### D8 — Tensor core acceleration (optional, H100 target)
- WGMMA / HMMA for the WENO reconstruction matrix multiply on H100
- Guard with `#if __CUDA_ARCH__ >= 900`
- Gate: throughput ≥ 2× WENO5-Z on H100 at same accuracy

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
7. No axis-specific duplicate functions — one `template <Axis DIR>` only.
8. No scheme-selection branches inside `__global__` kernels — dispatch at launch.
9. No `virtual` in device-callable code — use CRTP or `std::variant`.
10. C++20 concepts applied at every template boundary (except inside `__global__`).

### CUDA
11. `cudaDeviceSynchronize()` forbidden in the advance loop; use stream events.
12. All halo exchange: async on the solver stream; no blocking CPU MPI staging
    once D2 lands (NCCL P2P).
13. Kernel roofline target: ≥ 70 % of peak memory BW (Nsight Compute).
14. New kernels ship with an `ncu` baseline logged to `docs/perf/`.
15. Cooperative groups used for any warp-level reduction (no raw `__shfl_sync`
    magic constants).

### Testing & autonomy
16. Every new physics functor: CPU unit test before any GPU kernel.
17. Every new GPU kernel: correctness test (`PASS` within tolerance) before
    performance work.
18. Commit granularity: one commit per gate-green milestone, message format
    `D<n>: <title>; t<gate> pass`.
19. Run `cmake --build build -t ba` before every push; abort push if any test fails.
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
cmake --build build -t ba          # all CPU gates (32 tests)
cmake --build build -t t24         # CUDA Graph (P8.6)
cmake --build build -t t25         # GPU vs CPU correctness (P9.1)
cmake --build build -t t26         # NSSolver GPU dispatch (P10-A3)
cmake --build build -t t27         # SGS Smagorinsky (P-SGS-GPU)
cmake --build build -t t28         # MPI+GPU halo exchange (P-MPI-GPU)
cmake --build build -t t29         # GPU-native AMR (D1 gate)
cmake --build build -t t30         # NCCL halo exchange (D2 gate)
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
- cuBLAS, NCCL, Cooperative Groups programming guides

**Physics**
- Pope (2000) — Turbulent Flows (LES/DNS reference)
- Poinsot & Veynante (2005) — Theoretical and Numerical Combustion
- Mihalas & Mihalas (1984) — Foundations of Radiation Hydrodynamics
- Berger & Colella (1989) — AMR flux register correction
