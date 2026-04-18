# CFD Solver — To-Do List

> Generated: 2026-04-16  
> Based on `roadmap.md` and current codebase state.

Status legend: `✅ done` · `⚠️ partial` · `🔲 not started`

---

## Phase 0 — Emergency Fixes

| # | Status | Item |
|---|--------|------|
| P0.1 | ✅ | `restrict_conservative` stride `*2` restored |
| P0.2 | ✅ | `node.h` → `node.block->h` in test_amr6 |
| P0.3 | ✅ | A02 octant convention fixed (bit0=x) |
| P0.4 | ✅ | `sutherland()` uses `(T/T_ref)*sqrt(T/T_ref)` not `std::pow` |
| P0.5 | ✅ | `R_GAS=287.058` unified CPU/GPU |
| P0.6 | ✅ | `ke_prev_` promoted to member, reset in `init()` |
| P0.7 | ✅ | Dead ternary removed from `fill_ghosts_periodic` |
| P0.8 | ✅ | Remove dead `lap` lambda and dead stress tensor in `sgs.cpp` |

---

## Phase 1 — AMR Foundation

| # | Status | Item |
|---|--------|------|
| P1.1 | ✅ | `coarsen()` free-list allocator; `alloc_node_group(8)` |
| P1.2 | ✅ | `balance()` re-snapshots `leaf_indices()` after each `refine()` |
| P1.3 | ✅ | `fill_cf_ghosts()` integrated into ghost-fill path |
| P1.4 | ✅ | Flux registers: `accumulate_fine_flux` + `apply_flux_correction` |
| P1.5 | ✅ | `regrid()` full Berger-Colella protocol (tag → refine → coarsen → neighbours → ghosts) |
| P1.6 | ✅ | Leaf cache with dirty flag |
| —   | ✅ | **All gate tests T1–T4, T6, T7 pass** |

---

## Phase 2 — Spatial and CPU/GPU Correctness ← **current phase**

### CPU track (no hardware dependency)

| # | Status | Item | Why it matters |
|---|--------|------|----------------|
| P0.8 | ✅ | Remove dead `lap` lambda + stress tensor in `sgs.cpp` | Dead code executed every SGS call; misleads future readers |
| P2.2 | ✅ | Face-centered flux loop in `compute_rhs` | Halves HLLC evaluations; makes flux accumulation architecturally clean |
| P2.3 | ✅ | Pre-compute primitive variable cache `P[NCELL]` in `compute_rhs` | Reduces EOS inversions from 3584 to 1728 per block; enables SIMD |
| P2.4 | ✅ | OpenMP `#pragma omp parallel for` on leaf loop in `tree_rhs`; Morton-sort leaves before loop | Embarrassingly parallel after ghost fill; L3 cache locality from Morton sort |
| B5   | ✅ | Face-averaged viscosity `mu_face = 0.5*(mu_L + mu_R)` at coarse-fine interfaces in `operators.cpp` | Currently deferred from Phase 0; needed for correct viscous fluxes across AMR levels |

### GPU track

| # | Status | Item | Why it matters |
|---|--------|------|----------------|
| P2.5 | ✅ | On-device CFL reduction (`atomicMin` on uint64 reinterpretation) | Eliminates ~40 MB D→H transfer per step; prerequisite for P2.6 |
| P2.6 | ✅ | CUDA Graph capture of full RK3 loop — pointer-dt kernels + capture/replay | Eliminates ~30 s of kernel-launch overhead at 10⁶ steps |
| T5   | ✅ | GPU gate test (8 sub-tests; T05 tests CFL dt agreement + 1-step conservation) | Validates GPU path end-to-end |

### Gate for Phase 3 (last P2 item — widest blast radius)

| # | Status | Item | Propagation |
|---|--------|------|-------------|
| **P2.1** | ✅ | **`NG` 1→2 upgrade** | `NB2=12`, `NCELL=1728`, `gpu_constants.cuh`, all ghost-fill kernels, all index arithmetic, `static_assert` shared memory ≤ 96 KB on A100, AMR prolongation/restriction loops |

---

## Phase 3 — 5th-Order Accuracy + Symmetry Preservation *(hard gate: P2.1 must land first)*

| # | Status | Item | Key reference |
|---|--------|------|---------------|
| P3.1 | ✅ | WENO5-Z reconstruction with characteristic decomposition | Borges et al. (2008) |
| P3.2 | ✅ | Ducros sensor + hybrid skew-symmetric/WENO5 switch | Pirozzoli (2011), Morinishi (2010) |
| P3.3 | ✅ | Entropy-stable HLLC-ES flux (Chandrashekar 2013) | Chandrashekar (2013) |
| P3.4 | ✅ | Dynamic Smagorinsky SGS (Germano identity, Lilly LS) | Germano et al. (1991) |
| P3.5 | ✅ | IMEX-ARK + connect existing `MGSolver::vcycle()` for implicit viscous solve | Kennedy & Carpenter (2003) |

---

## Phase 4 — HPC & Advanced Features *(parallel items, no strict ordering among themselves)*

| # | Status | Item | Dependency |
|---|--------|------|------------|
| P4.1 | ✅ | Local time stepping — Berger-Oliger subcycling | P1.4 + P1.5 |
| P4.2 | ✅ | AoSoA memory layout for AVX-512 CPU SIMD | Stable code post-P2 |
| P4.3 | ✅ | Thread Block Clusters / DSMEM ghost exchange (H100/B200) | CUDA 12 + hardware |
| P4.4 | ✅ | Multiphase Baer-Nunziato diffuse interface (`NVAR→7`) | P2.1 + P3.1 |
| P4.5 | ✅ | ZFP compressed checkpointing (16–32× reduction) | None — drop-in in `checkpoint.cpp` |
| P4.6 | ✅ | Neural SGS closure (ONNX Runtime, drops into `SGSModel` interface) | P3.4 |

---

## Phase 5 — Physical Validation Benchmarks *(implement in order; each is a new gate test)*

| # | Status | Test case | Validates | Key metric | Ref |
|---|--------|-----------|-----------|-----------|-----|
| B.1 | 🔲 | Sod shock tube (1D, Euler) | HLLC Riemann solver | Shock x≈0.850, contact x≈0.685 at t=0.2 | Sod (1978) J.Comput.Phys. |
| B.2 | 🔲 | Shu-Osher shock-entropy (1D) | WENO5-Z fine-scale accuracy | Post-shock density oscillations preserved; 5th-order on smooth | Shu & Osher (1989) J.Comput.Phys. |
| B.3 | 🔲 | Taylor-Green vortex Re=1600 (3D) | SSP-RK3 + WENO5 turbulence decay | KE dissipation peak t*≈9; 128³ grid < 2% error vs DNS | HiOCFD4 (2016); Taylor & Green (1937) |
| B.4 | 🔲 | Two-phase water-air shock tube (1D, Allaire) | HLLC-BN flux, non-conservative α₁, stiffened-gas EOS | Sharp interface; no spurious pressure oscillations | Saurel & Abgrall (1999) J.Comput.Phys. 150 |
| B.5 | 🔲 | Woodward-Colella blast waves (1D, strong shocks) | HLLC robustness (p ratio 10⁵), AMR triggering | Shock/contact positions match converged reference at t=0.038 | Woodward & Colella (1984) J.Comput.Phys. 54 |
| B.6 | 🔲 | Kelvin-Helmholtz instability (2D) | AMR shear-layer refinement, Ducros sensor | Linear growth rate σ=√(kΔU/2); roll-up at t≈1–2 | Chandrashekar (2013); Pirozzoli (2011) |
| B.7 | 🔲 | Rayleigh-Taylor instability (2D, Allaire) | 5-equation model + AMR; α₁∈[0,1] | Bubble/spike growth match linear theory γ=√(gkA) | Allaire et al. (2002) J.Comput.Phys. |
| B.8 | 🔲 | Lid-driven cavity Re=1000 (2D) | Viscous NS, Sutherland law, low-Mach | u-centerline min ≈−0.33 at y≈0.45 vs Ghia et al. | Ghia, Ghia & Shin (1982); Bruneau & Saad (2006) |
| B.9 | 🔲 | Turbulent channel flow Re_τ=180 (3D, LES) | Dynamic Smagorinsky, Vreman, Neural SGS | Log-law u⁺=(1/0.41)ln(y⁺)+5.2; u′rms peak≈2.7 at y⁺≈14 | Moser, Kim & Mansour (1999) |
