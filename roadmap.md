# CFD Solver — Master Roadmap (Answer \#6)

> Generated: 2026-04-12  
> Synthesises: Answer \#3 (Perplexity analysis), Answer \#4 (consolidated bug register),  
> Answer \#5 (cross-domain improvements), plus Claude analyses of 2026-04-09.

---

## Compatibility Notes

| Improvement | Conflicts | Dependency | Verdict |
|---|---|---|---|
| WENO5-Z | `NG=1` hardcoded everywhere | Requires `NG=2` gate (P2.1) | ⚠️ Blocking change |
| Skew-symmetric convection | HLLC in `operators.cpp` | Ducros sensor for hybrid switch | ✅ Swap `convective_rhs` only |
| Entropy-stable flux | `hllc_flux()` | New log-mean operators | ✅ Replaces `hllc_flux()` only |
| AoSoA layout | Current SoA `Q[NVAR][NCELL]` | All accessors, ghost fill, GPU | 🔴 Major refactor — Phase 4 |
| CUDA Graph time loop | GPU CFL bug M12 | On-device CFL fix first (P2.5) | ⚠️ CFL fix is prerequisite |
| Thread Block Clusters | Current `gpu_ghost_*` design | CUDA 12 + H100/B200 | ⚠️ Hardware-gated — Phase 4 |
| IMEX-ARK | Adds new time path | Connected MG solver needed | ⚠️ MG connection prerequisite |
| LTS (Berger-Oliger) | Requires regrid + flux registers | B3 fixes must land first | 🔴 Depends on P1 |
| Neural SGS | Replaces `SmagorinskyModel::apply()` | Fits existing virtual interface | ✅ Drop-in compatible |
| ZFP checkpoint | Binary checkpoint format | `checkpoint.cpp` refactor only | ✅ I/O only |
| Multiphase (Baer-Nunziato) | `NVAR=5` hardcoded, GPU kernels | `NVAR=6`, full kernel rewrite | 🔴 Deep refactor — Phase 4 |
| Dynamic Smagorinsky | Test filter needs 3×3×3 stencil | Tied to `NG=2` (P2.1) | ⚠️ Bundle with NG=2 |

**Critical incompatibility:** WENO5 requires `NG=2`. Changing `NG` in `cell_block.hpp` propagates to  
`NB2`, `NCELL`, all GPU constants, all ghost fill kernels, and all index arithmetic.  
This is the single Phase gate separating P2 from P3.

---

## Phase 0 — Emergency Fixes  *(< 1 day)*

| ID | Severity | File | Fix |
|---|---|---|---|
| P0.1 | 🔴 | `src/amr_operators.cpp` | Restore `fi = NG + (lf_i % half) * 2` (and `fj`, `fk`). Regression from FIX B3: restriction maps coarse cell `lc` to fine cells `2*lc` and `2*lc+1`; missing `*2` makes adjacent coarse cells overlap identical fine cells. |
| P0.2 | 🔴 | `tests/test_amr6.cpp` | Replace `node.h` → `node.block->h` (all occurrences). `BlockNode::h` was removed; compile error blocks all CI. |
| P0.3 | 🔴 | `tests/test_amr6.cpp` | Fix A02 expected octant: `int oct = (lf_i/half) \| ((lf_j/half)<<1) \| ((lf_k/half)<<2)`. Old test uses bit2=x; new canonical convention is bit0=x. |
| P0.4 | 🟡 | `include/cell_block.hpp` | `sutherland()`: replace `std::pow(T/T_ref, 1.5)` with `(T/T_ref)*std::sqrt(T/T_ref)`. Mathematically identical; ~10× faster (5 vs 50 cycles). Called `NB³ × N_leaves` times per viscous RHS. |
| P0.5 | 🟡 | `include/cell_block.hpp` + `include/cuda/gpu_constants.cuh` | Unify `R_GAS = 287.058` (CODATA dry air). Current CPU=287.0 vs GPU=287.058 silently produces different pressures at the same (ρ, T). |
| P0.6 | 🟡 | `src/ns_solver.cpp` | Move `static double ke_prev` to `NSSolver` member variable initialised in `init()`. Static local survives solver destruction; wrong residual in multi-instance or repeated test runs. |
| P0.7 | 🟢 | `src/block_tree.cpp` | Remove dead ternary in `fill_ghosts_periodic`: `(ni>=0 && block) ? ihi() : ihi()` always evaluates identically. |
| P0.8 | 🟢 | `src/sgs.cpp` | Remove dead `lap` lambda; remove dead `txx/tyy/tzz/txy/txz/tyz` computation. Variables computed and discarded every SGS call. |

---

## Phase 1 — AMR Foundation  *(1–2 weeks)*

All of Phase 2 onward depends on a functioning AMR loop.

### P1.1 — `coarsen()` Tombstone / Free-List Model

**Why:** `nodes.resize(fc)` corrupts the tree whenever children are not the last 8 entries.  
**Fix:** Replace flat vector append/truncate with a free-list allocator:
```cpp
static constexpr int DEAD_SENTINEL = -2;
std::vector<int> free_list_;
int  alloc_node();   // pop free_list or emplace_back
void free_node(int); // push to free_list, reset block
```
Also resolves the `refine()` use-after-reallocation UB since `nodes.resize()` is eliminated.

### P1.2 — `balance()` Work-Queue Model

**Why:** Loop iterates stale leaf list after `refine()`; internal nodes accessed.  
**Fix:** Re-snapshot `leaf_indices()` after every `refine()` call inside the balance loop; break and restart immediately when `changed=true`.

### P1.3 — `fill_cf_ghosts()` Integration

**Why:** Fine-block ghost cells adjacent to a coarser neighbour currently receive wrong values (direct copy assumes same resolution). All cross-level gradients are O(h_coarse) instead of O(h_fine).  
**Fix:** In `fill_ghosts_periodic`, per-face:
```cpp
if (nodes[ni].level == nodes[li].level)
    copy_same_level_ghost(blk, *nodes[ni].block, d);
else  // 2:1 guaranteed
    fill_cf_ghosts(blk, *nodes[ni].block, child_octant_of(li), d);
```

### P1.4 — Flux Register Implementation (Berger & Colella 1989)

**Why:** Without flux correction the scheme is not locally conservative at coarse-fine faces.  
**Mathematics:** After each fine-level RHS, accumulate fine face fluxes. After coarse update, apply:
$$\Delta Q_c^{corr} = \frac{\Delta t}{h_c}\left(\sum_{f \subset \text{coarse face}} F_f \frac{h_f^2}{h_c^2} - F_c\right)$$
**Fix:** Implement `accumulate_fine_flux()` and `apply_flux_correction()` in `flux_register.hpp/.cpp`.

### P1.5 — `regrid()` Full Implementation

**Why:** Currently a no-op — tree topology never changes at runtime.  
**Fix:** Full Berger-Colella protocol:
1. Tag cells by gradient sensor (density + pressure + vorticity)
2. Call `tree.refine(li)` for tagged leaves
3. Prolong conserved variables to new fine cells
4. Enforce 2:1 balance via `tree.balance()`
5. Call `tree.rebuild_neighbours()`
6. Fill all ghosts in new topology
7. `alloc_scratch()`

### P1.6 — Leaf Cache with Dirty Flag

**Why:** `leaf_indices()` allocates a new `std::vector<int>` on every call (~6× per RK stage).  
**Fix:** Dirty-flag cache inside `BlockTree`; set `leaf_dirty_=true` in `refine()`, `coarsen()`, `rebuild_neighbours()`.

---

## Phase 2 — Spatial and GPU Correctness  *(2–3 weeks, parallel tracks)*

### P2.1 — NG=2 Ghost Layer Upgrade  ⚠️ *Gate for Phase 3*

**Why:** WENO5-Z needs a 5-point stencil; dynamic SGS needs a 3×3×3 test filter.  
**Change:** `NG=1` → `NG=2` in `cell_block.hpp`.  
**Propagation:**
- `NB2 = 12`, `NCELL = 1728` (from 10, 1000)
- `gpu_constants.cuh`: `GPU_NB2=12`, `GPU_NCELL=1728`
- Ghost fill must fill 2 layers
- Shared memory per GPU block: 5×1728×8 = 69.1 KB — fits A100 (96 KB), add `static_assert`
- `amr_operators.cpp`: prolongation/restriction loops cover 2 ghost layers

**Storage cost:** +72.8% per block. Justified: WENO5 needs 125× fewer blocks for the same accuracy.

### P2.2 — Face-Centered Flux Loop

**Why:** Current cell loop computes every face flux twice (43.7% wasted HLLC evaluations).  
**Fix:** Iterate over faces, write to left and right cells simultaneously.  
**Bonus:** Flux register accumulation becomes trivially correct (face pointer captured inline).

### P2.3 — Primitive Variable Cache

**Why:** `prim(i,j,k)` called ~3584× per block per RHS (7 neighbours × 512 cells); each call does a full EOS inversion.  
**Fix:** Pre-compute `std::array<Prim, NCELL> P` once at start of `compute_rhs`; reduces EOS calls to 1728. Enables compiler SIMD auto-vectorisation on the inner i-loop.

### P2.4 — OpenMP on Leaf Loop

**Why:** Each `compute_rhs` is fully independent after ghost fill.  
**Fix:** `#pragma omp parallel for schedule(dynamic,4)` on the leaf loop in `tree_rhs`.  
**Bonus:** Sort leaves by Morton code before the loop for L3 cache reuse of spatially adjacent ghost data.

### P2.5 — GPU On-Device CFL Reduction

**Why:** Full D→H memcpy (~40 MB/step) performed solely to find `dt_min`; kills GPU throughput.  
**Fix:** `atomicCAS`-based `atomicMin_double` device reduction kernel; transfers **8 bytes** D→H per step.

### P2.6 — CUDA Graph Time Loop

**Why:** Per-stage kernel launch overhead ~5 µs × 6 launches × 10⁶ steps = 30 s wasted.  
**Requires:** P2.5 (on-device CFL).  
**Fix:** Capture full RK3 loop as a `cudaGraph_t`; replay with `cudaGraphLaunch()`. CPU-GPU sync only at diagnostic intervals.

---

## Phase 3 — Accuracy Upgrade: 5th-Order + Symmetry Preservation  *(3–4 weeks)*

**Hard prerequisite: P2.1 (NG=2).**

### P3.1 — WENO5-Z with Characteristic Decomposition

**Mathematics:** 5th-order ENO-quality reconstruction using WENO-Z weights with global smoothness indicator $\tau_5 = |\beta_0 - \beta_2|$. Applied in characteristic variables (eigenvector decomposition of $\partial F/\partial Q$ at Roe-averaged state) to prevent component-wise oscillations at shocks.  
**Integration:** Add `ReconstructionScheme` enum to `SolverConfig`; dispatch in `convective_rhs_faces()`.  
**Reference:** Borges et al. (2008), *J. Comput. Phys.* 227:3191.

### P3.2 — Hybrid Skew-Symmetric / WENO5 (Pirozzoli Switch)

**Mathematics:** Ducros sensor:
$$\theta = \frac{-(\nabla\cdot\mathbf{u})^2}{(\nabla\cdot\mathbf{u})^2 + |\nabla\times\mathbf{u}|^2 + \epsilon}$$
- $\theta < 0.65$: Morinishi skew-symmetric split (zero numerical dissipation, exact KE conservation)
- $\theta \geq 0.65$: WENO5-Z (TVD shock capturing)

**Why unique:** The only production solver simultaneously achieving symmetry preservation, TVD shock capturing, and block-structured AMR on GPU.  
**Reference:** Pirozzoli (2011), *J. Comput. Phys.* 230:3097; Morinishi (2010), *J. Comput. Phys.* 229:4world.

### P3.3 — Entropy-Stable HLLC-ES Flux

**Mathematics:** Replace HLLC with Chandrashekar (2013) entropy-conservative flux augmented by entropy-dissipative term, satisfying discrete entropy inequality:
$$\frac{d}{dt}\int U\,dV + \oint \mathbf{F}^{ES}\cdot\hat{n}\,dS \leq 0$$
Activates only in shock cells (where Ducros sensor triggers).  
**Why unique:** Mathematically provable non-increasing discrete entropy — not offered by PeleC, OpenFOAM, or FLASH.  
**Reference:** Chandrashekar (2013), *Commun. Comput. Phys.* 14:1252.

### P3.4 — Dynamic Smagorinsky SGS

**Mathematics:** Germano identity with test filter at $\hat{\Delta}=2\Delta$; Lilly (1992) least-squares contraction for local $C_s^2$. Test filter is a 3×3×3 box average — requires NG=2.  
**Integration:** `DynamicSmagorinsky` class replacing `SmagorinskyModel` via existing virtual interface.  
**References:** Germano et al. (1991), *Phys. Fluids A* 3:1760; Lilly (1992), *Phys. Fluids A* 4:633.

### P3.5 — IMEX-ARK + Connected Multigrid Solver

**Mathematics:** ARS(2,2,2) IMEX scheme (Pareschi & Russo 2005) treats viscous Laplacian implicitly:
$$(\mathbf{I} - \gamma\Delta t\,\mu\nabla^2)\mathbf{U}^{(i)} = \text{explicit terms}$$
Solve via `MGSolver::vcycle()` already implemented in `linalg.cpp` but never connected.  
**Fixes:** `MGLevel::hy/hz` dead fields; MG level count as parameter (not hardcoded 3).  
**Reference:** Kennedy & Carpenter (2003), *Appl. Numer. Math.* 44:139.

---

## Phase 4 — HPC & Advanced Features  *(4–6 weeks, parallel)*

### P4.1 — Local Time Stepping (Berger-Oliger Subcycling)
**Requires:** P1.5 + P1.4. **Impact:** 10–100× for 3+ AMR levels.  
Recursive `advance_level(int l, double dt_l)` with 2:1 subcycling; flux corrections applied after each coarse step.  
**Reference:** Berger & Oliger (1984).

### P4.2 — AoSoA Memory Layout for CPU SIMD
**Requires:** Stable code base after P2. **Impact:** 3–4× CPU throughput.  
Template-parameter layout selection: SoA for GPU (coalesced loads), AoSoA with W=8 for AVX-512 CPU.

### P4.3 — Thread Block Clusters / DSMEM Ghost Exchange
**Requires:** CUDA 12.x + H100/B200. **Impact:** ~10× ghost fill bandwidth.  
Cluster of 2 adjacent CFD blocks share face data via distributed shared memory; eliminates inter-block global memory ghost copies entirely.

### P4.4 — Multiphase: Baer-Nunziato Diffuse Interface
**Requires:** P2.1 (NG=2) + P3.1 (WENO5). `NVAR` → 7. Full GPU kernel rewrite.  
**Reference:** Saurel & Abgrall (1999), *J. Comput. Phys.* 150:425.

### P4.5 — ZFP Compressed Checkpointing
**Requires:** None. Drop-in in `checkpoint.cpp`.  
Fixed-accuracy ZFP at tolerance $10^{-8}$; 16–32× compression on smooth fields. CUDA backend for GPU-side compression before PCI-e transfer.

### P4.6 — Neural SGS Closure
**Requires:** P3.4. Train on dynamic Smagorinsky outputs; ONNX Runtime CUDA EP inference.  
Drop-in via `SGSModel` virtual interface.

---

## Dependency Graph

```
P0  (Emergency fixes — 1 day)
 |
 P1.1 coarsen tombstone
 P1.2 balance work-queue
 P1.3 fill_cf_ghosts
 P1.4 flux registers  ─────────────────────────────────┐
 P1.5 regrid() ←── P1.1 + P1.2 + P1.3                  │
 P1.6 leaf cache                                         │
 |                                                       │
P2  (parallel tracks)                                    │
 P2.1 NG=2  ──┐                                          │
 P2.2 face flux loop                                     │
 P2.3 prim cache                                         │
 P2.4 OpenMP leaves                                      │
 P2.5 GPU on-device CFL                                  │
 P2.6 CUDA Graph ←── P2.5                                │
 |                                                       │
P3  (requires P2.1 NG=2)                                 │
 P3.1 WENO5-Z ←── P2.1                                  │
 P3.2 Hybrid skew-sym ←── P3.1                          │
 P3.3 Entropy-stable flux                                │
 P3.4 Dynamic SGS ←── P2.1                              │
 P3.5 IMEX + MG solver                                   │
 |                                                       │
P4  (independent items)                                  ↓
 P4.1 LTS ──────────────────────────── P1.5 + P1.4 ──────┘
 P4.2 AoSoA CPU layout
 P4.3 TBC/DSMEM (H100)
 P4.4 Multiphase ←── P2.1 + P3.1
 P4.5 ZFP checkpoint
 P4.6 Neural SGS ←── P3.4
```

---

## Expected Performance After Each Phase

| After Phase | Spatial Order | GPU Throughput | AMR Status | vs. PeleC |
|---|---|---|---|---|
| **Current** | 1st | ~300 M cells/s (single block) | No-op | ≪ PeleC |
| **P0** | 1st | ~300 M cells/s | No-op (fixed tests) | ≪ PeleC |
| **P1** | 1st | ~300 M cells/s | ✅ Functional | On par (1st-order) |
| **P2** | 1st | **~5–10 B cells/s** | ✅ Functional | **Ahead on GPU** |
| **P3** | **5th + symm-pres** | ~3–6 B cells/s | ✅ Functional | **Ahead on accuracy** |
| **P4** | 5th + symm-pres | ~10–15 B cells/s | ✅ LTS | **Outperforms on all metrics** |

---

## Bug Register Reference (Answer \#4)

| ID | Severity | File | Issue |
|---|---|---|---|
| B1 | 🔴 | `amr_operators.cpp` | `restrict_conservative` regression: `*2` stride removed |
| B2 | 🔴 | `test_amr6.cpp` | `node.h` compile error |
| B3 | 🔴 | `ns_solver.cpp` | `regrid()` never calls `tree.refine()` |
| B4 | 🔴 | `test_amr6.cpp` | A02 expected values use old octant convention |
| B5 | 🔴 | `block_tree.cpp` | Ghost fill ignores coarse-fine level differences |
| B6 | 🔴 | `block_tree.cpp` | `coarsen()` tail-assumption corrupts tree |
| B7 | 🔴 | `block_tree.cpp` | `refine()` use-after-reallocation UB |
| B10 | 🔴 | `flux_register.cpp` | Flux register stubs — not implemented |
| B11 | 🔴 | `block_tree.cpp` | `fill_ghosts_wall` negates all momentum components |
| M2 | 🟡 | `ns_solver.cpp` | `static ke_prev` persists across instances |
| M4 | 🟡 | `cell_block.hpp` | `sutherland()` uses `std::pow` |
| M5 | 🟡 | `cell_block.hpp` / `gpu_constants.cuh` | `R_GAS` inconsistency |
| M12 | 🟡 | `gpu_solver.cu` | Full D→H memcpy per step for CFL |
| m2 | 🟢 | `sgs.cpp` | Dead `lap` lambda |
| m7 | 🟢 | `sgs.cpp` | Dead SGS stress tensor computation |
