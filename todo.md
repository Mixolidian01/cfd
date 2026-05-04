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
| B.1 | ✅ | Sod shock tube (1D, Euler) | HLLC Riemann solver | Shock x≈0.850, contact x≈0.685 at t=0.2 | Sod (1978) J.Comput.Phys. |
| B.2 | ✅ | Shu-Osher shock-entropy (1D) | WENO5-Z fine-scale accuracy | Post-shock density oscillations preserved; 5th-order on smooth | Shu & Osher (1989) J.Comput.Phys. |
| B.3 | ✅ | Taylor-Green vortex Re=1600 (3D) | SSP-RK3 + WENO5 turbulence decay | KE dissipation peak t*≈9; 128³ grid < 2% error vs DNS | HiOCFD4 (2016); Taylor & Green (1937) |
| B.4 | ✅ | Two-phase water-air shock tube (1D, Allaire) | HLLC-BN flux, non-conservative α₁, stiffened-gas EOS | Sharp interface; no spurious pressure oscillations | Saurel & Abgrall (1999) J.Comput.Phys. 150 |
| B.5 | ✅ | Woodward-Colella blast waves (1D, strong shocks) | HLLC robustness (p ratio 10⁵), AMR triggering | Shock/contact positions match converged reference at t=0.038 | Woodward & Colella (1984) J.Comput.Phys. 54 |
| B.6 | ✅ | Kelvin-Helmholtz instability (2D) | AMR shear-layer refinement, Ducros sensor | Linear growth rate σ=√(kΔU/2); roll-up at t≈1–2 | Chandrashekar (2013); Pirozzoli (2011) |
| B.7 | ✅ | Rayleigh-Taylor instability (2D, Allaire) | 5-equation model + AMR; α₁∈[0,1] | Bubble/spike growth match linear theory γ=√(gkA) | Allaire et al. (2002) J.Comput.Phys. |
| B.8 | ✅ | Lid-driven cavity Re=1000 (2D) | Viscous NS, Sutherland law, low-Mach | u-centerline min ≈−0.33 at y≈0.45 vs Ghia et al. | Ghia, Ghia & Shin (1982); Bruneau & Saad (2006) |
| B.9 | ✅ | Turbulent channel flow Re_τ=180 (3D, LES) | Dynamic Smagorinsky, Vreman, Neural SGS | Log-law u⁺=(1/0.41)ln(y⁺)+5.2; u′rms peak≈2.7 at y⁺≈14 | Moser, Kim & Mansour (1999) |

---

## Phase 6 — In-situ Browser Live Feed ← **current phase**

### Design: zero-dependency, direct-to-browser, AMR-aware 2D slice streaming

**Architecture** (Answer #43):
- `LiveStreamer` lives inside `NSSolver` as an optional plugin pointer (`streamer_`).
- `snapshot()` is called once per `advance()`, after `apply_flux_correction()` — the one guaranteed quiescent point where all leaf data is final.
- Double-buffer: solver writes `back_` (no lock), then atomically swaps with `front_` under `swap_mtx_`; stream thread steals `front_` into `work_` (O(1) `std::swap`) and serializes.
- HTTP server: minimal POSIX sockets, no external deps. `GET /` → embedded HTML. `GET /stream` → HTTP/1.1 chunked binary stream. `POST /config` → JSON var/axis/pos.
- Wire format: 4-byte length prefix + 32-byte frame header + n_blocks×16-byte block descriptors + n_blocks×NB×NB×float32 data.
- Browser viewer: single HTML page (embedded as C++ raw string literal), `fetch()` + `ReadableStream`, viridis colormap polynomial, pixel-level canvas drawing.

### Phase A — 2D axis-aligned slice

| # | Status | Item | Notes |
|---|--------|------|-------|
| P6.1 | ✅ | Double-buffer snapshot protocol in `NSSolver::advance()` | `back_`/`front_`/`work_` triple; solver writes without lock; swap under `swap_mtx_`; stream thread steals via `std::swap` |
| P6.2 | ✅ | `LiveStreamer` HTTP server (POSIX sockets, no external deps) | `run_accept()` thread spawns per-connection threads; `GET /` HTML; `GET /stream` chunked octet-stream; `POST /config` JSON update |
| P6.3 | ✅ | 2D slice extraction: vars ρ, p, T, \|u\|, ρu, ρv, ρw, E; axis X/Y/Z; float32 wire | `blk.prim()` for derived vars; `blk.Q[v][cell_idx]` for conserved |
| P6.4 | ✅ | Browser viewer (HTML+JS, embedded raw string literal) | viridis polynomial; canvas pixel fill; auto-reconnect; var/axis/pos controls |
| P6.5 | ✅ | Gate test: solver runs 50 steps with streamer, viewer receives ≥1 frame, parses cleanly | `tests/test_streamer.cpp` — S1–S4 pass |

### Phase B — 3D ray-marched volume rendering *(future)*

| # | Status | Item |
|---|--------|------|
| P6.6 | ✅ | WebGPU ray-marcher (render pipeline, WGSL, arcball camera); AMR blocks → resampled N³ r32float GPUTexture3D; `GET /volume` + `GET /volume-stream` |
| P6.7 | ✅ | Transfer function editor (canvas drag → opacity curve); colormap selector (Viridis/Hot/Cool/Gray); opacity slider; touch support; `POST /config` steering |
| P6.8 | ✅ | uint16 quantization + LZ4 compression (3–5× reduction vs raw float32) |

---

## Phase 7 — Scale, Accuracy & Physics Extensions ← **current phase**

### Priority list (Answer #47)

| # | Status | Item | Key reference |
|---|--------|------|---------------|
| P7.1 | ✅ | **MPI domain decomposition** — Morton-curve leaf partitioning across ranks; non-blocking halo exchange (`exchange_halos()`); global CFL `MPI_Allreduce`; global diagnostics reduce; gate test: 2-rank Sod, mass conserved to 10⁻¹⁰ | Berger & Oliger (1984); Colella et al. (1999) |
| P7.2 | ✅ | **WENO5 high-order prolongation at C/F interfaces** — replace piecewise-constant prolongation with 5th-order one-sided WENO stencil; restores O(h⁵) global accuracy in AMR regions | McCorquodale & Colella (2011) J.Comput.Phys. |
| P7.3 | ✅ | **Ghost-cell immersed boundary method** — level-set surface representation; ghost-cell velocity/temperature reconstruction for no-slip walls; STL surface loader | Mittal & Iaccarino (2005) Annu.Rev.Fluid Mech. |
| P7.4 | ✅ | **Multi-species reactive flow** — N_spec species conservation equations (NVAR += N_spec); Arrhenius finite-rate chemistry source; IMEX stiff chemistry sub-step | Poinsot & Veynante (2005) |
| P7.5 | ✅ | **Quantitative DNS: TGV Re=1600 at 128³** (requires P7.1) — compare ε(t*) to Brachet (1983) DNS and HiOCFD4 data; error table in `answers_register.md` | Brachet et al. (1983) J.Fluid Mech. 130 |
| P7.6 | ✅ | **Wall-modelled LES (WMLES)** — algebraic/ODE wall model, log-law reconstruction at first off-wall cell; enables Re_τ ≫ 180 | Larsson et al. (2016) CTR Annual Research Briefs |

---

## Phase 8 — Full GPU Integration ← **current phase**

### Goal: make every leaf block's primary residence on-device; CPU is a staging buffer.

| # | Status | Item | Key note |
|---|--------|------|----------|
| P8.1 | ✅ | **GPU memory pool** — `d_Q` pointer in `CellBlock`; lifecycle hooks `on_block_alloc_`/`on_block_free_` in `BlockTree`; `GpuPool` (CUDA free-list); `NSSolver::init()` wires the pool; gate test t19 (upload/download round-trip + lifecycle) | Foundation for all subsequent GPU items |
| P8.2 | ✅ | **GPU ghost fill** — `GpuLeafGhostMeta` + `GpuGhostFillList`; `k_fill_faces` (same-level copy, CF 5th-order Lagrange, wall/open/periodic BCs) + `k_fill_edges_corners` (interior-self-ref); gate test t20 (F1–F4: wall, periodic, same-level, CF) | P8.1 |
| P8.3 | ✅ | **GPU WENO5-Z RHS** — full `tree_rhs` loop on device; WENO5-Z + HLLC-ES + viscous/SGS in one kernel per block; shared-memory halo staging | P8.2 |
| P8.4 | ✅ | **GPU AMR refine/coarsen kernels** — `k_prolong` (piecewise-constant, 512-thread 3D block per pair) + `k_restrict` (volume-weighted average); `GpuAmrList` with `build_prolong`/`build_restrict`/`exec_prolong`/`exec_restrict`; gate test t23 (A1–A4: 4/4 pass) | P8.3 |
| P8.5 | ✅ | **GPU CFL + diagnostics reduce** — warp-shuffle tree reduction; `cudaMemcpyAsync` for dt only; zero D→H bandwidth on interior steps | P8.1 |
| P8.6 | ✅ | **CUDA Graph re-capture on regrid** — detect topology change, invalidate and rebuild graph; steady steps use graph replay; three per-stage sub-graphs (s1/s2/s3) with external cudaMemsetAsync zeroing; gate test t24 (G1–G4: 4/4 pass) | P8.3 + P8.4 |

---

## Phase 9 — GPU Correctness Validation & Code Quality

| # | Status | Item | Key note |
|---|--------|------|----------|
| P9.1 | ✅ | **GPU vs CPU correctness** — gate test t25 (N1–N4: 4/4 pass); GPU Q matches CPU NSSolver to 3.4e-15; GPU dt matches CPU dt to 4.1e-16; mass conserved to 1.97e-15; validates GpuGraphSolver as drop-in for uniform meshes | Flat periodic tree, no AMR |
| P9.2 | ✅ | **GPU code quality cleanup** — unified `CUDA_CHECK` macro in `include/cuda/gpu_check.cuh` (was duplicated in 5 TUs); `NVAR/NCELL` → `GPU_NVAR/GPU_NCELL` in `gpu_graph.cu`; compile-time consistency `static_assert`; removed 140-line dead `gpu_rhs_kernel`/`gpu_ghost_periodic_single` (from `gpu_solver.cu`, not compiled); fixed topology-detection UB in `advance()` (stale-pointer on same-count regrid) | Code quality / correctness |

---

## Phase 10 — GPU Code Quality, Architecture & NSSolver Integration *(remaining items from 2026-04-29 review)*

> Source: architecture review (agent a4d3e3ed7e34d2032) and code simplification audit (agent affaa432d02af1437), session 2026-04-29.  P9.2 closed the easy items; the below are the remaining open findings.

### 10-A — Architecture corrections

| # | Status | Item | File:line | Risk |
|---|--------|------|-----------|------|
| A1 | ✅ | **`d_Q` layer violation** — removed `d_Q` from `CellBlock`; `GpuPool` now owns an `unordered_map<const CellBlock*, double*>` (`ptrs_`); all build() methods accept `const GpuPool&`; `GpuGraphSolver` stores `download_pairs_` for download_q() without needing the pool after build(). All t19–t26 gate tests pass. | `include/gpu_pool.hpp`, `include/cell_block.hpp`, all GPU TUs + test files | done |
| A2 | ✅ | **`TimeIntegrator` interface added** — `TimeIntegrator::step(const BlockTree&, double cfl)` defined in `ns_solver.hpp`; `IGpuSolver` inherits from it via a `final` bridge that delegates `step()→advance()`; `GpuGraphSolver` (already implementing `IGpuSolver`) now satisfies `TimeIntegrator`; TODO note in header documents how CPU/LTS paths should also be extracted to `CpuRk3Integrator`/`LtsIntegrator`. | `include/ns_solver.hpp` | done |
| A3 | ✅ | **NSSolver GPU integration** — wired `GpuGraphSolver` into `NSSolver::advance()` via `set_gpu_solver()` / `set_gpu_pool()`; CPU and GPU `NSSolver` agree to 2e-12 over 20 steps (flat IC). Root cause of original t26 failure was a flawed `max_rel_err` metric (denominator `\|a\|+1e-12` gave false 9% error on analytically-zero rho*w); fixed to use `max(\|a\|,\|b\|,1e-8)` floor (matches P9.1 convention). Gate test t26: all 3 sub-tests pass. | `src/ns_solver.cpp`, `tests/cuda/test_p10a3_gpu_nssolver.cu` | done |

### 10-B — GPU kernel refactoring

| # | Status | Item | File:line | Effort |
|---|--------|------|-----------|--------|
| B1 | ✅ | **Axis-dispatch index duplication** — added `cidx_axis(axis, ax_val, a, b)` device helper in `gpu_ghost_fill.cu`; collapsed 6 × 4-line if/else dispatch blocks to single-line calls. t20 gate still 4/4 pass. | `src/cuda/gpu_ghost_fill.cu` | done |
| B2 | ✅ | **`gpu_meta_buffer<T>` RAII helper** — `gpu_upload_meta<T>(ptr, h_vec)` template in `include/cuda/gpu_meta_buffer.cuh`; replaces free/malloc/memcpy ritual in all four `build()` methods (`GpuGhostFillList`, `GpuRhsList`, `GpuCflList`, `GpuGraphSolver`); ~30 lines removed. All t19–t26 gate tests still 0 failures. | `include/cuda/gpu_meta_buffer.cuh`, `src/cuda/gpu_ghost_fill.cu`, `src/cuda/gpu_rhs.cu`, `src/cuda/gpu_cfl.cu`, `src/cuda/gpu_graph.cu` | done |
| B3 | ✅ | **`k_rhs_visc` refactored** — added `sp_vel()` + `face_visc()` `__device__ __forceinline__` helpers; `k_rhs_visc` body replaced by a 3-iteration axis loop calling `face_visc` for ± faces; kernel shrinks from ~140 to ~45 lines. All t21, t25 gate tests still 0 failures. | `src/cuda/gpu_rhs.cu` | done |
| B4 | ✅ | **`k_fill_faces` refactored** — added `fill_copy()`, `fill_zero_grad()`, `fill_wall()` `__device__ __forceinline__` helpers; discovered `fill_zero_grad` unifies CF coarse←fine and open BC (identical formula), `fill_copy` unifies same-level and periodic self-wrap. Kernel dispatch now 4 clean branches, each a named call. t20 gate still 4/4 pass. | `src/cuda/gpu_ghost_fill.cu` | done |
| B5 | ✅ | **`gpu_weno5z_scalar` mirrored reconstruction** — extracted `weno5z_upwind(a,b,c,d,e)` device helper; right state calls it with reversed stencil `(vp3,vp2,vp1,v0,vm1)`. Removed ~18 duplicate lines. t21 and t25 gates still 0 failures. | `src/cuda/gpu_rhs.cu` | done |

### 10-C — Naming / API consistency

| # | Status | Item | File | Effort |
|---|--------|------|------|--------|
| C1 | ✅ | **Trailing-underscore convention** — removed trailing `_` from all 12 public data members of `GpuGraphSolver` (`ghost_list`, `rhs_list`, `cfl_list`, `d_rk3_metas`, `d_Qn_pool`, `n_leaves`, `download_pairs`, `stream`, `graph_s1/s2/s3`, `graph_valid`); private methods keep leading `_`. Other list structs already correct. t24–t26 still pass. | `include/cuda/gpu_graph.cuh`, `src/cuda/gpu_graph.cu` | done |
| C2 | ✅ | **`d_dt_ptr()` vs `d_dt` mixed accessor** — removed `d_dt_ptr()` getter; all callers use `.d_dt` field directly. Comment in header updated. | `include/cuda/gpu_cfl.cuh` | done |
| C3 | ❌ N/A | **`download_q` / `download_rhs` duplicate `thread_local` buffer** — sharing via `gpu_check.cuh` not viable: nvcc device-compilation pass parses host functions and rejects namespace references that are conditionally hidden by `!defined(__CUDA_ARCH__)`. Buffers remain local (two separate `static thread_local` lines). | `src/cuda/gpu_graph.cu`, `src/cuda/gpu_rhs.cu` | — |

---

## Phase 11 — Correctness & Safety *(from Perplexity audit 2026-05-04)*

> Note: P11.1 was initially misdiagnosed as root cause of t26 A1 failure. Actual root cause was a flawed `max_rel_err` metric (2026-05-04 fix). Both CPU and GPU viscous energy implementations use the identical conservative face-flux form; no inconsistency exists. P11.1 is now a code-quality improvement (ensure doc matches code), not a correctness fix.

### 11-A — Critical correctness fixes

| # | Status | Item | File | Priority |
|---|--------|------|------|----------|
| P11.1 | ❌ N/A | **Viscous energy CPU/GPU** — validated: both use the same conservative face-flux form. t26 A1 passes (max error 2e-12, tol 1e-8); original 7.6% error was from a flawed `max_rel_err` metric (fixed in A3 gate). No code change needed. | — | — |
| P11.2 | ✅ | **No viscous CFL** — added $\Delta t_\text{visc} = h^2 / (2 C_\text{visc} \max_i(\mu_i/\rho_i))$ where $C_\text{visc}=\max(4/3,\gamma/\text{Pr})$ to `CellBlock::cfl_dt()` in `block_tree.cpp`; also mirrored in `k_cfl_reduce` in `gpu_cfl.cu`. Moved `PR=0.72` from per-file statics to `cell_block.hpp`. CPU–GPU dt agreement maintained (t26 A2: 4e-16). | `src/block_tree.cpp`, `src/cuda/gpu_cfl.cu`, `include/cell_block.hpp` | done |
| P11.3 | ✅ | **No positivity limiter** — added `apply_positivity_floor()` static helper in `ns_solver.cpp`; floors ρ ≥ 1e-12 and p ≥ 1e-12 (via E adjustment) on interior cells after each RK3 stage in `advance()`, `advance_imex()`. All 12 t4 sub-tests and t26 3/3 pass unchanged. | `src/ns_solver.cpp` | done |
| P11.4 | ✅ | **Sutherland temperature floor** — added `if (T < 1.0) T = 1.0;` to `sutherland()` in `cell_block.hpp` and `gpu_sutherland()` in `gpu_constants.cuh`. Prevents `sqrt` of negative T near rarefactions. All gates pass. | `include/cell_block.hpp`, `include/cuda/gpu_constants.cuh` | done |
| P11.5 | ✅ | **HLLC Roe-average wave speeds** — replaced Davis-Einfeldt bounds with Roe-average $\hat{u}, \hat{c}$ (Einfeldt-Roe form); $\hat{H} = c^2/(\gamma-1) + \tfrac{1}{2}|u|^2$ Roe-averaged via $\sqrt{\rho}$ weights; fallback to arithmetic mean if $\hat{c}^2 \le 0$. All t4/t7/t25/t26 gates pass. | `src/operators.cpp:hllc_flux` | done |
| P11.6 | ✅ | **Ducros pressure threshold configurable** — added `ducros_p_threshold=0.1` and `ducros_blend_width=0.1` to `SolverConfig`; added `set_ducros_thresholds()` to `operators.hpp`; `fill_ducros_cache` reads module-level globals; `NSSolver::advance()` propagates config before RK3. All gates pass. | `include/ns_solver.hpp`, `include/operators.hpp`, `src/operators.cpp`, `src/ns_solver.cpp` | done |

### 11-B — GPU completeness

| # | Status | Item | File | Priority |
|---|--------|------|------|----------|
| P11.7 | ✅ | **GPU Wall/Open BC ghost fill** — already resolved by P10-B4: `k_fill_faces` dispatches `fill_wall()` (bc_type=1), `fill_zero_grad()` (bc_type=2), periodic self-wrap (bc_type=0) in the `d_src==nullptr` branch. `GpuGhostFillList::build()` stores `bc_type` per-leaf in `GpuLeafGhostMeta`. No separate kernel needed. | `src/cuda/gpu_ghost_fill.cu` | done |
| P11.8 | 🔲 | **Berger-Colella flux correction on GPU** — implement GPU flux registers (`GpuFluxRegister`); accumulate fine fluxes during GPU RK3 stages; `k_apply_flux_correction` kernel; call after `gpu_solver_->advance()` in `NSSolver::advance()`. Required for mass/momentum/energy conservation on AMR+GPU grids. | `src/cuda/gpu_graph.cu`, `src/ns_solver.cpp` | 🟠 Medium (flat-tree: harmless) |

---

## Phase 12 — LiveStreamer Improvements *(from Perplexity analysis 2026-05-04)*

### 12-A — C++ side (metrics, probe)

| # | Status | Item | Notes |
|---|--------|------|-------|
| P12.1 | ✅ | **`GET /metrics` JSON endpoint** — `MetricsSnapshot` struct in `live_streamer.hpp` (step, t, dt, cfl, ke, mass, n_leaves, rho_min/max, wall_time_ms, leaves_per_level[8], gpu_active); `push_metrics()` called from `NSSolver::advance()` whenever `streamer_` non-null; `handle_get_metrics()` returns JSON; merged P12.4 metrics loop. t4, t7, t12 pass. | `include/live_streamer.hpp`, `src/live_streamer.cpp`, `src/ns_solver.cpp` | done |
| P12.2 | ✅ | **`POST /probe` endpoint** — canvas click → `{"x":norm_x,"y":norm_y}` (0-1 canvas coords) → JSON with all 8 vars + level + block; `probe` field added to `FrameBuffer` (8 floats/cell, filled in `build_frame()`); `handle_post_probe()` searches `front_.descs` under `swap_mtx_`; JS click handler updates info bar with ρ/p/T/|u|. t12 4/4 pass. | `include/live_streamer.hpp`, `src/live_streamer.cpp` | done |
| P12.3 | ✅ | **Conservation norms in MetricsSnapshot** — `mass_error`, `momentum_error`, `energy_error` (relative to step-0 baselines stored in `NSSolver::mass0_/mtm0_/energy0_`); computed in `advance()` metrics loop alongside P12.1 quantities; exposed in `GET /metrics` JSON. t4 (12/12), t12 (4/4) pass. | `include/live_streamer.hpp`, `include/ns_solver.hpp`, `src/ns_solver.cpp`, `src/live_streamer.cpp` | done |
| P12.4 | ✅ | **Structured JSON stdout per step** — enhanced `verbose_json` output in `NSSolver::advance()` to include `mass` (via `CellBlock::total_mass()`) and `leaves` count; aligned field names to `{"step":N,"t":T,"dt":DT,"cfl":CFL,"ke":KE,"mass":M,"leaves":L}`. | `src/ns_solver.cpp` | done |
| P12.5 | 🔲 | **Derived variables: Mach, vorticity, Q-criterion, schlieren** — add to `StreamVar` enum and `build_frame()`; local stencil operations on existing CellBlock fields | ~1 day C++ |

### 12-B — JS/HTML side (browser viewer)

| # | Status | Item | Notes |
|---|--------|------|-------|
| P12.6 | ✅ | **Manual vmin/vmax lock** — added lock checkbox + two number inputs to 2D viewer bar; `parseFrame` reads locked values when checked, auto-fills from frame range when unlocked; LZ4 path dequantizes from frame range correctly; shows 🔒 in info bar when locked. t12 still 4/4 pass. | `src/live_streamer.cpp` | done |
| P12.7 | 🔲 | **Live sparkline charts** — rolling history (last 2000 steps) for CFL, KE, mass, leaf count, wall_time; Canvas2D sparkline, no external libraries | ~half day |
| P12.8 | 🔲 | **AMR grid overlay** — checkbox to draw block outlines colour-coded by AMR level on a separate canvas layer; uses BlockDesc2D origin+cell_size | ~2 hours |
| P12.9 | ✅ | **Colormap selector** — Viridis/Inferno/Plasma/RdBu polynomial colormaps in JS dispatcher; `<select id="scm">` in viewer bar; `colormap()` replaces `viridis()`; `drawCells` updated. t12 4/4 pass. | `src/live_streamer.cpp` | done |
| P12.10 | ✅ | **Keyboard shortcuts** — `j`/`k` step slice position (0.005/step), `v` cycle vars, `a` cycle axes, `Space` pause/resume; guarded against focus on INPUT/SELECT; `paused` flag skips parseFrame but keeps stream alive. t12 4/4 pass. | `src/live_streamer.cpp` | done |

---

## Phase 13 — Architecture Refinement *(from Perplexity + literature survey)*

| # | Status | Item | Reference | Effort |
|---|--------|------|-----------|--------|
| P13.1 | 🔲 | **`template <Axis DIR>` flux refactor** — templated flux functions (HLLC-ES, KEP, WENO5-Z, viscous face-stress) on `enum class Axis : int {X,Y,Z}`; compile-time axis permutation; eliminates asymmetric-bug surface; extends Phase 10 B1 | Basilisk `foreach_dimension()`; Perplexity §5 | 2 weeks |
| P13.2 | 🔲 | **SP split-form compressible convective operator** — Kennedy-Gruber or Pirozzoli split form satisfying discrete KE identity; entropy-stable correction via Chandrashekar; extends current KEP+HLLC-ES hybrid | Pirozzoli (2011); Chandrashekar (2013); arXiv 2502.11567 | 2 weeks |
| P13.3 | 🔲 | **Entropy-stable inflow/outflow BCs** (Svärd-Gjesteland 2025) — SAT penalty terms at open boundaries; proven discrete entropy inequality $\frac{d}{dt}\int\eta\,dV \le 0$; replaces zero-gradient open BC | arXiv 2506.21065 | 3 days |
| P13.4 | 🔲 | **Entropy-stable no-slip wall BC** (Sayyari 2021) — viscous penalty enforcement of $\mathbf{u}=0, T=T_w$; configurable isothermal wall temperature in `BCType::Wall`; replaces adiabatic-only wall | arXiv 2110.10507 | 1 week |
| P13.5 | 🔲 | **SAT penalty at AMR C/F interfaces** — reformulate Berger-Colella reflux as matrix-free SBP-SAT penalty; proven conservative + energy stable; GPU-native (no global matrix) | Del Rey Fernández et al.; SC24 | 3 weeks |
| P13.6 | 🔲 | **LTS + Berger-Colella quadrature fix** — fine sub-step weights 1/2 (vs coarse 1/6, 1/6, 2/3); conservation error on AMR+LTS currently unaddressed | Berger-Oliger (1984) | 1 week |
| P13.7 | 🔲 | **TMA async halo exchanges** *(Hopper-exclusive: H100/H200 only)* — replace `cudaMemcpyAsync` ghost fill with TMA prefetch; hides latency while interior flux runs; zero-change for non-Hopper hardware | CUDA 12 + H100/H200 | 2 weeks |

---

## Phase 14 — Advanced Physics *(from Perplexity research survey)*

| # | Status | Item | Reference | Effort |
|---|--------|------|-----------|--------|
| P14.1 | 🔲 | **ACDI compressible multiphase** — Accurate Conservative Diffuse Interface model; 9th conserved field $\phi$; stiffened-gas EOS per phase; 2–4 cell sharp interface, no reinitialization; replaces/extends P4.4 Baer-Nunziato | Huang & Johnsen (2023–2024); MHIT36 (2025) | 3 months |
| P14.2 | 🔲 | **Wall contact angle BC for phase-field** — entropy-stable Cahn-Hilliard wall BC: $\nabla\phi\cdot\hat{n}|_\partial = -\cos\theta_w/\epsilon \cdot g'(\phi)$; configurable static contact angle $\theta_w$ | arXiv 1910.11252 | 2 weeks |
| P14.3 | 🔲 | **Neural SGS via JAX-Fluids adjoint + TensorRT** — train Dynamic Smagorinsky surrogate via end-to-end AD on JAX-Fluids; export ONNX, deploy via TensorRT into `SGSModel` virtual interface; extends P4.6 | JAX-Fluids 2.0; arXiv 2406.19494 | 2 months |
| P14.4 | 🔲 | **GPU AMR subcycling** — two CUDA streams (coarse/fine), CUDA event synchronisation, LTS-aware flux register weights on device; companion to P13.6 | AMReX GPU subcycling (2025) | 3 months |
| P14.5 | 🔲 | **GPU ensemble UQ + Ensemble Kalman Filter** — multiple simultaneous GPU solvers sharing pool; `MPI_Allreduce` for ensemble moments; posterior update at observation times | Finance-inspired UQ literature | 2 months |

---

## System Dependencies — Missing Packages (Answer #44)

> Audited: 2026-04-25. All packages below are **optional** — the solver builds and all 18 gate tests pass without them. Fallbacks are active.

| Package | Status | Used by | Fallback when absent | Install command |
|---------|--------|---------|----------------------|-----------------|
| `liblz4-dev` | ✅ installed — `/usr/lib/x86_64-linux-gnu/liblz4.so`, `/usr/include/lz4.h` | P6.8 — LZ4 stream frame compression | — | already done |
| `libonnxruntime-dev` | ❌ `.deb` downloaded to `/home/dkoffibi/` but not installed | P4.6 — Neural SGS ONNX inference | Vreman algebraic SGS | `sudo dpkg -i /home/dkoffibi/libonnxruntime1.21_1.21.0+dfsg-2_amd64.deb && sudo dpkg -i /home/dkoffibi/libonnxruntime-dev_1.21.0+dfsg-2_amd64.deb` |

**Fully installed (no action needed):**
- CUDA Toolkit 12.6 + 13.x — `nvcc` at `/usr/local/cuda/bin/nvcc`, GPU gates build
- ZFP 1.0.1 (`libzfp-dev`) — `/usr/include/zfp.h` present, P4.5 gate passes
- OpenMP 4.5 — leaf loop parallelised
- pthreads — `LiveStreamer` threads run
- POSIX socket headers (`arpa/inet.h`, `netinet/in.h`, `sys/socket.h`) — glibc

**After installing `liblz4-dev`**, add to `CMakeLists.txt` (in the `ns_solver` block):
```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 IMPORTED_TARGET liblz4)
if(LZ4_FOUND)
    target_link_libraries(ns_solver PUBLIC PkgConfig::LZ4)
    target_compile_definitions(ns_solver PUBLIC HAVE_LZ4=1)
endif()
```

**After installing `libonnxruntime-dev`**, re-run `cmake -S . -B build` and rebuild `t11`.
