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
| P7.6 | 🔲 | **Wall-modelled LES (WMLES)** — algebraic/ODE wall model, log-law reconstruction at first off-wall cell; enables Re_τ ≫ 180 | Larsson et al. (2016) CTR Annual Research Briefs |

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
