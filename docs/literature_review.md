# Literature & Practice Review — GPU CFD State of the Art
> Synthesised before starting `to_develop` development.
> Covers: solver landscape, high-order schemes, AMR, multi-GPU comm, implicit solvers,
> CUDA performance, reactive flows, radiation.
> Knowledge base: published literature and codebases through mid-2025.

---

## 1. GPU CFD Solver Landscape

### Production / near-production codes

| Code | Language | Scheme order | AMR | Multi-GPU | Reported BW% |
|---|---|---|---|---|---|
| **JAX-Fluids** (Bezgin 2023/24) | JAX/Python | WENO5–WENO9, TENO | None | JAX pjit sharding | ~55–65% A100 |
| **STREAmS-2** (Bernardini 2023) | Fortran+CUDA | WENO5 + 6th-order viscous | None | CUDA-aware MPI + NVLink | ~55–60% A100 |
| **PyFR** (Witherden et al.) | Python+CUDA/Metal | FR (DG equiv.), p=1–7 | None | CUDA-aware MPI | ~75–85% A100 (tensor-product) |
| **Pele / PeleC** (LBNL/NREL) | C++/CUDA, AMReX | PPM (2nd order Godunov) | Full GPU AMR | CUDA-aware MPI | ~40–55% A100 |
| **HORSES3D** | Fortran+CUDA | DGSEM p=1–8 | Static p-ref | MPI | ~70% A100 |
| **ExaFEMV / CEED** | C+CUDA | High-order DG | None | CUDA-aware MPI | ~80% (matrix-free) |

### Key lessons

1. **No production code uses NCCL for halo exchange.** NCCL is optimised for collective operations (AllReduce, AllGather). For irregular point-to-point halos, CUDA-aware MPI + NVLink (via UCX) is universally preferred.

2. **AMReX GPU AMR** (used by Pele, ExaAM, WarpX) keeps the **octree topology on CPU** — this is not a design weakness but the correct choice. The tree has O(N_blocks) entries = O(N_cells / 512). It changes every 10–100 steps. All physics kernels, prolongation, restriction, and flux registers run on GPU. Refinement tagging runs as a GPU reduction; the CPU reads a single integer.

3. **PyFR / DGSEM outperform WENO in BW%** because the tensor-product structure turns stencil access into dense matrix-matrix multiplies, which map better to GPU memory hierarchy. However DGSEM requires subcell limiting for strong shocks, which breaks the tensor structure and eliminates the advantage in shock-dominated flows.

4. **JAX-Fluids** demonstrates that a pure-Python JIT-compiled solver can match hand-written CUDA for smooth flows. The JIT eliminates the Python overhead entirely. Not applicable to our C++/CUDA architecture but confirms that functional, kernel-fused pipelines outperform launch-per-operation styles.

5. **Mixed precision (FP32 accumulation)** gives 2× throughput improvement on H100 for BLAS operations. For WENO RHS (bandwidth-bound), the gain is minimal because memory, not compute, is the bottleneck.

---

## 2. High-Order Schemes on GPU

### WENO5-Z (current baseline)
- 5th-order accurate in smooth regions; 1st-order at discontinuities
- Smooth indicators (β_k): require ~18 multiplications per point per direction
- Memory access: 6-point stencil, irregular → ~45–55% peak BW on A100
- Square roots in the weights are compute overhead (not bandwidth overhead)

### TENO7-A (Fu et al. 2019, planned D3)
- 7th-order accurate in smooth regions; ENO-sharp at shocks
- **No thread divergence**: the "targeted" switching is a scalar mask applied after all stencil evaluations, not an if-branch on the stencil selection
- Smooth indicators: more points but no square roots → similar FLOP count to WENO5-Z
- Memory: 8-point stencil (one wider) → marginally more BW pressure than WENO5
- Expected convergence on T08 vortex: ~6.5–7th order (vs ~5 for WENO5-Z)
- GPU BW%: similar to WENO5-Z (~45–55%); the extra stencil width is absorbed by coalesced access in SoA layout

### TENO5 (lighter alternative)
- 5th-order, same ENO-sharpness as TENO7 but fewer stencil points
- Good option if TENO7 register pressure causes occupancy loss

### WENO-AO (Balsara et al., adaptive order 3rd/5th/7th)
- More branching than TENO → worse GPU utilisation
- **Not recommended** for GPU

### DGSEM
- Excels for smooth DNS/LES (tensor-product matrix-free → up to 85% BW)
- Requires subcell FV limiting for shocks → breaks tensor structure → GPU advantage lost
- Requires p-AMR or h-AMR rethink (our octree is h-AMR at fixed order)
- **Not recommended** for this shock-capturing AMR solver

### Entropy-stable fluxes overhead
- Chandrashekar EC flux: ~1.8× HLLC cost (GPU-measured, mostly hidden by memory latency)
- HLLC-ES (our current flux): good balance — entropy-stable without the full EC overhead
- Conclusion: **keep HLLC-ES**, add Chandrashekar EC as a build-time option for smooth-flow DNS

### Recommendation
**TENO7-A is the correct upgrade from WENO5-Z** for this codebase:
- Same GPU memory pattern (SoA, coalesced)
- No thread divergence
- Higher order → better DNS/LES accuracy
- Drop-in at the reconstruction functor layer (already abstracted by R2/R4 refactor)

---

## 3. GPU-Native AMR

### What "GPU-native" actually means (AMReX model)
- **Tree topology (CPU)**: O(N_blocks) nodes, changes every ~50 steps. Host-side rebuild takes < 1 ms. No benefit from putting on GPU.
- **Physics kernels (GPU)**: prolongation, restriction, flux registers, RHS, ghost fill — all on device. No H2D/D2H during the advance loop.
- **Refinement tagging (GPU→CPU)**: error indicator kernel on GPU, `cudaMemcpy` of a small integer tag array, CPU rebuilds tree, `cudaMemcpy` of changed blocks only.

### Prolongation on GPU
- Linear (2nd order): each fine cell = weighted average of 2 coarse cells per direction → trivial kernel
- High-order (5th, for WENO): needs 3-point stencil in coarse → must have coarse ghost layers filled first → our `fill_cf_ghosts` + GPU ghost fill already handles this
- Bandwidth: ~1.2 reads + 1 write per fine cell → achieves ~75–80% peak BW

### Restriction on GPU
- Fine → coarse: 8:1 averaging for 3D octree → each coarse cell reads 8 fine cells → fully coalesced with correct layout
- Bandwidth: ~8 reads + 1 write → achieves ~80% peak BW

### GPU flux registers (Berger-Colella)
- `undo_cf` and `accumulate_cf`: `atomicAdd` on device register arrays
- Already partially GPU-resident in our `gpu_cf.cu`
- No fundamental obstacle; the atomic operations are rare (one per face per step)

### Refinement criteria kernel
- Löhner sensor or gradient-based: reads Q, writes a tag flag (uint8_t)
- Simple element-wise kernel → 1 read + 1 write → ~85% peak BW

### What to change in D1
- Move CPU prolongation/restriction in `amr_operators.cpp` to GPU kernels in `gpu_amr.cu`
- Move refinement tagging to a GPU reduction kernel
- Keep `BlockTree::refine()` as CPU-side topology surgery
- The result: `regrid()` = [tag kernel on GPU] → [D2H small tag array] → [CPU tree rebuild] → [D2H/H2D changed blocks only]

---

## 4. Multi-GPU Communication

### The NCCL misconception
NCCL's `ncclSend`/`ncclRecv` (point-to-point, since NCCL 2.7) has **higher latency than cudaMemcpyPeer for small messages** (~15–25 μs vs ~5–10 μs on NVLink), because NCCL uses a separate progress thread and ring-based protocols optimised for large collective messages. For CFD halos (66 KB per block, 11 KB per face), this latency dominates.

**No production CFD code uses NCCL for halo exchange.**

### What production codes actually do

| Scenario | Recommended approach |
|---|---|
| Same node, NVLink | `cudaMemcpyPeerAsync` (direct GPU→GPU, 300–400 GB/s) |
| Same node, PCIe only | CUDA-aware MPI (falls back to D2H→H2D via host pinned buffer) |
| Inter-node | CUDA-aware MPI over InfiniBand / RoCE (via UCX) |
| CFL global min | `ncclAllReduce` (this is the right NCCL use case) |

### CUDA-aware MPI (OpenMPI + UCX)
- `MPI_Isend(d_ptr, ...)` directly with a GPU buffer pointer
- UCX detects NVLink topology and uses `cudaMemcpyPeer` or GDRCopy automatically
- Latency: ~8–20 μs for small messages on NVLink
- Bandwidth: ~200–350 GB/s sustained (NVLink 3.0)
- STREAmS-2, AMReX, Pele all use this approach

### Revised D2 plan
- **Drop the NCCL P2P plan**
- Replace CPU staging (`GpuMpiHaloList`) with CUDA-aware MPI:
  - Same code path as today (`MPI_Isend`/`MPI_Irecv`) but pass `d_Q` device pointers directly
  - Requires `-DHAVE_CUDA_AWARE_MPI` guard (runtime check via `MPIX_Query_cuda_support()`)
  - Intra-node: UCX uses NVLink automatically
  - Inter-node: UCX uses InfiniBand GDRCopy
- **Keep NCCL** only for `mpi_allreduce_min` (CFL) — this is its correct use case
- Expected speedup vs CPU staging: 3–8× for same-node runs (eliminates D2H + H2D round-trip)

---

## 5. Implicit Solvers on GPU

### Target problem: viscous Helmholtz (IMEX-ARK)
The system is: `(I - dt·ν·Δ) u^* = rhs`
- Banded structure (7-point stencil in 3D)
- Diagonally dominant for moderate Re
- Block-local within each CellBlock (NB=8, 8³ system per block for each variable)

### Library options

| Library | Type | GPU support | Complexity | Best for |
|---|---|---|---|---|
| **cuSPARSE** | Sparse direct / iterative | CUDA native | Low API | Small sparse systems |
| **AmgX** | Algebraic multigrid | CUDA native | Medium | Large elliptic, external dep |
| **HYPRE** (BoomerAMG) | Algebraic multigrid | CUDA (hypre-device) | High | Pressure equation, HPC |
| **SUNDIALS CVODE** | ODE integrator | CUDA (NVector_Cuda) | Medium | Stiff chemistry ODEs |
| **Custom matrix-free GMRES** | Krylov | Our GPU kernels | Low | Viscous Helmholtz |

### Recommended approach for D4: matrix-free GMRES

For our IMEX viscous system:
- **Matrix-free**: the "A·v" product = apply viscous stencil kernel (already have `k_rhs_visc`)
- **Preconditioner**: block-Jacobi — each 8³ block solved independently → embarrassingly parallel, no communication
  - For scalar viscous equation: diagonal preconditioner (1 division per cell) → 5–10× convergence improvement
  - For full vector system: 5×5 block-diagonal per cell → invertible analytically
- **BLAS-1 operations** (dot product, axpy, nrm2) → cuBLAS or hand-written warp-reduce
- Expected iterations: 5–20 for Re = 100–10,000 (well-conditioned viscous system)
- No external dependency

### AmgX (alternative for D6 radiation)
- Better suited for the P1 radiation G-equation (indefinite Helmholtz with large κ)
- The radiation solve has worse conditioning than the viscous solve
- Worth evaluating AmgX as the radiation solver (D6)

### SUNDIALS CVODE GPU
- Best for D5 stiff chemistry: per-cell ODE integration with CUDA N-vector
- For single-step Arrhenius at moderate Ta/T < 15: explicit RK4 with subcycling is simpler
- CVODE GPU: use when Ta/T > 20 (stiff detonation regime)

---

## 6. CUDA Performance Best Practices (A100/H100)

### Memory layout: SoA is correct
Our current `Q[v][cell_idx]` SoA layout is optimal:
- Variable `v` access is coalesced (128-byte cache lines align with consecutive cell indices)
- Inner loop over cells maps to consecutive threads → coalesced
- Stencil access (neighbour cells) in the i/j/k directions: coalesced only in the innermost dimension; transverse accesses cause partial cache misses
- **Improvement**: 2D/3D tiling in shared memory for transverse access — reduces DRAM traffic by ~30% for 3D stencil kernels

### Block dimensions
- Current: 256 threads per block, 1 block per leaf (1728 cells / 256 = ~7 cells per thread)
- Better for stencil: (256, 1, 1) processing `NB2*NB2 = 144` cells per thread group in the i-direction sweep
- A100 optimal: 4–8 warps per SM for stencil kernels (not max occupancy — register pressure limits it)
- Check with `--ptxas-options=-v`: WENO5 kernels typically use 60–90 registers/thread → max 32 blocks/SM → ~25–30% occupancy → still BW-bound (occupancy doesn't matter when memory latency dominates)

### CUDA Graphs (our current approach — validated)
- Our 3-stage SSP-RK3 graph capture is the correct pattern
- Validated by literature: CUDA Graphs eliminate ~20–40 μs kernel launch overhead per RK3 stage
- **Graph capture + AMR**: the recapture-on-regrid pattern (our current design) is correct
- Pitfall to avoid: don't capture kernels with device-side conditionals or dynamic shared memory sizes

### H100 / Hopper-specific features
- **TMA (Tensor Memory Accelerator)**: loads 2D/3D tiles asynchronously from global → shared memory
  - For WENO5 stencil: TMA can load a `NB2 × NG` tile per warp group → ~20–35% latency reduction
  - Requires CUDA 12.0+, `__cluster_dims__` annotation — non-trivial to implement
  - Appropriate for D8 (H100 target), not D0–D7
- **Thread block clusters**: persistent producer/consumer patterns across SMs
  - Useful for fusing ghost fill + RHS into one cluster launch (eliminates intermediate global memory roundtrip)
  - Significant complexity; worth exploring in D8
- **FP8 / mixed precision**: WENO is BW-bound, not compute-bound → mixed precision doesn't help reconstruction; it helps GMRES inner products

### Achievable BW targets (revised)
Based on published STREAmS-2 and JAX-Fluids results:

| Kernel | A100 achievable | H100 achievable |
|---|---|---|
| WENO5-Z RHS | 45–55% | 50–60% |
| TENO7-A RHS | 43–53% | 48–58% |
| Ghost fill (face copy) | 75–85% | 80–90% |
| Prolongation / restriction | 70–80% | 75–85% |
| SGS Smagorinsky | 55–65% | 60–70% |

**Conclusion**: the original ≥70% target for WENO/TENO kernels is too aggressive. A realistic target is **≥55% of peak BW** for RHS kernels, **≥75%** for copy-like kernels (ghost fill, prolong/restrict).

### Shared memory tiling for WENO (high-impact optimisation)
Standard pattern used in STREAmS-2 and similar codes:
1. Load `NB2 × NB2` i-plane into shared memory (all 144 cells including ghosts)
2. All threads in block compute WENO reconstruction along i for their cell
3. Repeat for j and k directions
4. Eliminates ~60% of DRAM reads for the transverse-direction stencil accesses
5. Shared memory required: `NB2 × NB2 × NVAR × 8 bytes = 144 × 5 × 8 = 5.6 KB` → fits in 48 KB shared mem

This single optimisation likely brings WENO BW% from ~45% to ~65%.

---

## 7. Reactive Flows and Radiation

### D5 — Single-step Arrhenius (revised approach)

**Stiffness assessment**:
- Activation temperature Ta = Ea/R: for H₂-O₂: Ta ≈ 8000 K; for generic hydrocarbon: Ta ≈ 15,000 K
- At flame temperature T ≈ 2500 K: Ta/T ≈ 3–6 → weakly stiff → explicit RK4 with subcycling is sufficient
- At shock temperature T ≈ 5000 K (detonation): Ta/T ≈ 1–3 → not stiff at all
- **Conclusion**: explicit RK4 with operator splitting is correct for single-step Arrhenius; CVODE not needed

**GPU implementation**:
- One thread per cell; independent per-cell ODE integration
- No thread divergence (all cells do the same operations regardless of local values)
- Memory: read (ρ, T, Y), write updated (ρY, ρE) → pure element-wise kernel → ~85% peak BW

**Operator split sequence** (Strang, 2nd order):
```
hyperbolic half-step (dt/2) → chemistry full-step (dt) → hyperbolic half-step (dt/2)
```
Fits naturally into the existing CUDA Graph as additional nodes between RK3 stages.

### D6 — P1 Radiation

**G-equation**: ∇²G - κG = -4πκ·a·T⁴ (elliptic)
- Dominant term: when κ·L >> 1 (optically thick), Δ dominates → well-conditioned CG
- When κ·L << 1 (optically thin): κG dominates → diagonal solve → trivial
- GPU-resident preconditioned CG is sufficient; AMG not needed for moderate opacity
- Convergence: diagonal preconditioner → O(κ·h) condition number → 5–30 iterations
- AMReX uses multigrid-preconditioned CG; for our first implementation, diagonal CG is sufficient

**Coupling energy term**:
```
Source term in energy equation: Q_rad = κ·(4π·a·T⁴ - c·G)  [W/m³]
```
Added as an operator-split step after RK3. The solve for G uses the temperature from the end of RK3.

---

## 8. Revised Development Plan

### Changes to D0–D8 based on this review

| Phase | Original | Revised |
|---|---|---|
| **D0** | BW target ≥70% | Target ≥55% RHS, ≥75% copy kernels; add shared-memory tiling as first optimisation |
| **D1** | GPU-native AMR (vague) | Clarify: topology stays CPU; GPU kernels for prolong/restrict/tagging; regrid = tag→rebuild→copy changed blocks |
| **D2** | NCCL P2P halo exchange | CUDA-aware MPI (drop NCCL P2P entirely); use NCCL only for CFL AllReduce |
| **D3** | TENO7-A | Confirmed: TENO7-A correct choice; skip DGSEM (wrong for shock+AMR); add TENO5 as lighter fallback |
| **D4** | External GMRES library | Custom matrix-free GMRES + cuBLAS BLAS-1 + block-Jacobi; no external dependency |
| **D5** | Reactive (CVODE) | Explicit RK4 operator-split; CVODE only for stiff detonation extension |
| **D6** | P1 radiation (CG) | Diagonal-preconditioned CG; evaluate AmgX only if convergence > 50 iterations |
| **D7** | WMLES unchanged | Unchanged |
| **D8** | Tensor cores for WENO | Clarify: tensor cores don't help BW-bound WENO; instead use H100 TMA for stencil tile loads + thread block clusters for ghost-fill/RHS fusion |

### New priority: shared-memory tiling (D0.5)
Between D0 and D1, add a focused optimisation pass:
- Implement shared-memory tiling in `k_rhs_conv` (biggest kernel, most impact)
- Target: bring WENO5-Z RHS BW% from ~45% to ~60–65%
- Measure with `ncu`; log to `docs/perf/`
- This is lower risk and higher impact than D3 (new scheme) for raw performance

### Definitive phase ordering

```
D0  Baseline verification + ncu profiling (measure before changing anything)
D0.5 Shared-memory tiling in k_rhs_conv (highest-impact single optimisation)
D1  GPU-native AMR (prolong/restrict/tagging kernels; keep tree on CPU)
D2  CUDA-aware MPI halo exchange (replace CPU staging; NCCL only for CFL)
D3  TENO7-A reconstruction (drop-in functor; TENO5 as fallback)
D4  Matrix-free GMRES for implicit viscous (cuBLAS + block-Jacobi, no AmgX)
D5  Reactive flows: explicit RK4 Arrhenius operator-split
D6  P1 radiation: diagonal-CG for G-equation; AmgX if needed
D7  WMLES: Reichardt algebraic model
D8  H100 TMA + thread-block clusters for ghost-fill/RHS fusion
```

---

## 9. References

**Solver codes**
- Bezgin et al. (2023) *JAX-Fluids: A fully-differentiable high-order CFD solver for compressible two-phase flows.* CPC.
- Bernardini et al. (2023) *STREAmS-2.0: Supersonic turbulent accelerated Navier-Stokes solver version 2.* CPC.
- Witherden et al. (2014–2024) *PyFR: An open source framework for solving advection–diffusion type problems on streaming architectures.* CPC.
- Zhang et al. (2019–2024) *AMReX: a framework for block-structured adaptive mesh refinement.* JOSS.

**Schemes**
- Fu et al. (2019) *A targeted ENO scheme (TENO) for high-order simulations of turbulence and shock-wave interactions.* JCP.
- Chandrashekar (2013) *Kinetic energy preserving and entropy stable finite volume schemes for compressible Euler and Navier-Stokes equations.* CiCP.
- Gassner, Winters & Kopriva (2016) *Split form nodal discontinuous Galerkin schemes with summation-by-parts property for the compressible Euler equations.* JCP.

**GPU performance**
- Harris (2007) *Optimizing Parallel Reduction in CUDA.* NVIDIA Technical Report.
- Volkov (2010) *Better performance at lower occupancy.* GPU Technology Conference.
- NVIDIA (2023) *H100 Tensor Core GPU Architecture Whitepaper.*
- NVIDIA (2024) *Nsight Compute Roofline Analysis Guide.*

**AMR + multi-GPU**
- Dubey et al. (2022) *A survey of high-level frameworks in block-structured adaptive mesh refinement packages.* J. Parallel Distrib. Comput.
- Almgren et al. (2020) *AMReX: Block-Structured Adaptive Mesh Refinement for Multiphysics Applications.* IJHPCA.

**Implicit solvers**
- Anzt et al. (2020) *Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing.* ACM TOMS.
- NVIDIA AmgX documentation (2023). Multi-grid accelerated linear solvers for industrial applications.

**Reactive / radiation**
- Day & Bell (2000) *Numerical simulation of laminar reacting flows with complex chemistry.* CTM.
- Mihalas & Mihalas (1984) *Foundations of Radiation Hydrodynamics.* Dover.
