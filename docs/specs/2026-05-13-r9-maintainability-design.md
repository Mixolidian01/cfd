# R9 Maintainability & Modifiability — Design Spec

## Goal

Systematically reduce the maintenance burden of the solver by eliminating global
mutable state, decomposing monolith files, formalising existing interface stubs,
and closing test-coverage gaps. No numerical behaviour changes.

## Motivation — Comparison with leading frameworks

| Practice | AMReX / OpenFOAM / FLEXI | This codebase |
|---|---|---|
| Configuration | Typed nested structs + `validate()` | 40-field flat `SolverConfig`, no validation |
| Global state | Zero — injected via constructors | 9 file-scope statics across 2 TUs |
| EOS state | Injected via field/mesh object | Two desync'd copies (CellBlock + operators.cpp) |
| Integrator | Abstract `TimeIntegrator` object | Interface stub exists; CPU/LTS bodies inlined |
| BC dispatch | Registered strategy / variant | 3-way if/else; params in `block_tree.cpp` globals |
| Monolith guard | < 600 LOC per file | Three files > 1000 LOC |
| GPU/CPU parity | Shared `__host__ __device__` kernel library | WENO5/Ducros duplicated |

---

## Phase A — Global state elimination

### A1: Remove duplicate EOS statics

**Problem:** `operators.cpp` has its own copy of the stiffened-gas EOS state
(`g_sg_active`, `g_sg_ga/gb`, `g_sg_pia/pib`) that is a near-duplicate of
`CellBlock::sg_active_/sg_ga_/sg_gb_/sg_pia_/sg_pib_`. Both are set by
`ns_solver.cpp::init()`. If they drift, silent numerical errors occur.

**Fix:** Remove the `g_sg_*` statics from `operators.cpp`; replace their use
with `CellBlock::sg_active_`, `CellBlock::sg_ga_`, etc. Remove `set_sg_eos()`
from `operators.hpp` (no longer needed; keep only `CellBlock::set_sg_eos()`).
`CellBlock::set_sg_eos()` is a new single-source setter:
```cpp
static void set_sg_eos(bool active, double ga, double gb, double pia, double pib) noexcept {
    sg_active_ = active; sg_ga_ = ga; sg_gb_ = gb; sg_pia_ = pia; sg_pib_ = pib;
}
```
`ns_solver.cpp::init()` calls only `CellBlock::set_sg_eos()`.

**Files:** `include/cell_block.hpp`, `src/operators.cpp`, `include/operators.hpp`,
`src/ns_solver.cpp`.

**Gate:** all tests pass. T9 (multiphase) exercises the SG-EOS path.

---

### A2: Replace BC globals with BCRuntimeConfig member on BlockTree

**Problem:** `block_tree.cpp` has four file-scope globals:
```
g_wall_T       — isothermal wall temperature
g_open_bc_p    — far-field pressure for subsonic outflow
g_wall_ca_cos  — contact angle cos(θ_w) for ACDI
g_wall_ca_ceps — ACDI compression coefficient at wall
```
Accessed via static setter methods on BlockTree (`set_wall_T()`,
`set_open_bc_pressure()`, `set_wall_contact_angle()`). These break
multi-tree or multi-configuration scenarios and are thread-unsafe.

**Fix:** Define in `include/block_tree.hpp`:
```cpp
struct BCRuntimeConfig {
    double wall_T       = 0.0;
    double open_bc_p    = 0.0;
    double wall_ca_cos  = 0.0;
    double wall_ca_ceps = 0.0;
};
```
Add `BCRuntimeConfig bc_cfg;` as a public member of `BlockTree`.
Replace all reads of `g_wall_T` → `bc_cfg.wall_T`, etc.
Replace `BlockTree::set_wall_T(T_w)` → `tree.bc_cfg.wall_T = T_w;` at all call sites.
Remove the static setter declarations from `block_tree.hpp`.

**Files:** `include/block_tree.hpp`, `src/block_tree.cpp`, `src/ns_solver.cpp`.

**Gate:** all tests pass. T16 (contact angle) verifies the ACDI ghost-fill path.

---

### A3: Replace Ducros globals with per-call parameters in operators

**Problem:** `operators.cpp` has two globals:
```
g_ducros_p_thr = 0.1
g_ducros_blend = 0.1
```
Set by `set_ducros_thresholds()`. These prevent per-call-site customisation.

**Fix:** Remove the globals and `set_ducros_thresholds()`. Add a
`DucrosConfig { double p_threshold, blend_width; }` struct to
`include/operators.hpp`. Thread it through `compute_rhs` and
`tree_rhs` as a `const DucrosConfig&` parameter read from
`SolverConfig::ducros_*` fields. The `SolverConfig` already holds
`ducros_p_threshold` and `ducros_blend_width`; they are now passed
directly rather than stored in globals.

**Files:** `include/operators.hpp`, `src/operators.cpp`, `src/ns_solver.cpp`,
`src/cuda/gpu_rhs.cu` (if Ducros thresholds are used there too — check).

**Gate:** all tests pass.

---

### A4: SolverConfig::validate()

**Problem:** `SolverConfig` has 30+ fields with no constructor and no validation.
Invalid combinations (cfl > 1, max_level < 0, gamma < 1, etc.) are silently
accepted and produce garbage results.

**Fix:** Add `void validate() const` to `SolverConfig` in `include/ns_solver.hpp`.
It throws `std::invalid_argument` on any invalid combination. Call it at the
start of `NSSolver::init()`.

Checked invariants:
```
cfl ∈ (0.0, 1.0]             — CFL > 1 violates stability
t_end > 0                    — simulation must have positive duration
max_steps > 0
max_level ∈ [0, 6]           — sane AMR depth (prevents 64× memory blow-up)
lts_ratio ∈ {1, 2, 4}        — must be power-of-two refinement ratio
acdi_ceps ≥ 0
sat_tau ≥ 0
ducros_p_threshold ∈ [0, 1]
ducros_blend_width ≥ 0
wall_T ≥ 0
gamma_a > 1, gamma_b > 1     — EOS requires γ > 1
p_inf_a ≥ 0, p_inf_b ≥ 0
mg_levels ∈ [1, 8]
```

**Files:** `include/ns_solver.hpp`, `src/ns_solver.cpp`, plus a test added to
`tests/test_ns.cpp` (or a new `tests/test_config.cpp`).

**Gate:** all tests pass; new test verifies that bad configs throw.

---

## Phase B — TimeIntegrator extraction (P10-A2)

**Problem:** The `TimeIntegrator` interface already exists in `ns_solver.hpp`
(struct + TODO comment). Three implementations exist but two are inlined in
`ns_solver.cpp`:
- CPU SSP-RK3: `advance()` (~100 lines, lines 200–310 of ns_solver.cpp)
- LTS SSP-RK3: `advance_lts()` + `lts_rk3_level()` (~150 lines)
- GPU: `IGpuSolver : TimeIntegrator` (already extracted in CUDA TU)

**Fix:**
1. Create `include/cpu_rk3.hpp` and `src/cpu_rk3.cpp` with:
   ```cpp
   struct CpuRk3Integrator : TimeIntegrator {
       double step(const BlockTree& tree, double cfl) override;
   };
   ```
   Move the `advance()` body into `CpuRk3Integrator::step()`.

2. Create `include/lts_integrator.hpp` and `src/lts_integrator.cpp` with:
   ```cpp
   struct LtsIntegrator : TimeIntegrator {
       int lts_ratio;
       explicit LtsIntegrator(int ratio);
       double step(const BlockTree& tree, double cfl) override;
   private:
       void lts_rk3_level(BlockTree& tree, int level,
                          double dt, double sub_weight, bool coarse_mode);
   };
   ```
   Move `advance_lts()` and `lts_rk3_level()` bodies.

3. `NSSolver` holds `std::unique_ptr<TimeIntegrator> integrator_`, created in
   `init()` from `cfg.use_lts` and `cfg.use_gpu`.

4. `NSSolver::advance()` becomes:
   ```cpp
   double NSSolver::advance() { return integrator_->step(tree, cfg.cfl); }
   ```

**Files:** new `include/cpu_rk3.hpp`, `src/cpu_rk3.cpp`, `include/lts_integrator.hpp`,
`src/lts_integrator.cpp`, `src/ns_solver.cpp`, `include/ns_solver.hpp`,
`CMakeLists.txt`.

**Gate:** all tests pass including GPU gates (t19–t26). `ns_solver.cpp` loses ~250 lines.

---

## Phase C — GhostFiller extraction

**Problem:** `block_tree.cpp` (1379 LOC) mixes AMR octree topology with ghost-fill
logic. The three BC fill functions (`fill_ghosts_periodic`, `fill_ghosts_wall`,
`fill_ghosts_open`) are near-copies. `fill_cf_ghosts` (5th-order Lagrange) has
no unit test.

**Fix:**
1. Create `include/ghost_filler.hpp` and `src/ghost_filler.cpp`.
2. `GhostFiller` takes a `const BCRuntimeConfig&` (from Phase A2) and dispatches
   via `std::visit` on the `BCVariant` (already a C++20 variant after R3).
3. `fill_cf_ghosts` moves from `amr_operators.cpp` into `ghost_filler.cpp`.
4. `block_tree.cpp` calls `GhostFiller::fill(tree, node_idx)` instead of
   three separate free-function calls.
5. Add `tests/test_ghost_fill.cpp` with a unit test: construct a 2-block tree,
   set a polynomial field in the coarse block, fill ghosts, assert that the
   Lagrange interpolant matches to O(h^5) in the ghost layer.

**Files:** new `include/ghost_filler.hpp`, `src/ghost_filler.cpp`,
`tests/test_ghost_fill.cpp`, `src/block_tree.cpp`, `src/amr_operators.cpp`,
`CMakeLists.txt`.

**Gate:** all tests pass; new test_ghost_fill passes.

---

## Phase E — Monolith file decomposition

### E1: live_streamer.cpp → three files

`src/live_streamer.cpp` (1897 LOC) is a monolith. Extract:
- `src/http_server.cpp` — POSIX socket listener, HTTP parser, response writer
  (keeps `include/http_server.hpp` with `HttpServer` struct)
- `src/frame_packer.cpp` — FrameBuffer extraction, LZ4 compression,
  WebSocket frame assembly
- `src/live_streamer.cpp` — only the `LiveStreamer` class wiring both together
  (~200 LOC after extraction)
- `src/viewer_html.cpp` — the large embedded HTML/CSS/JS string in its own TU
  (generated from `assets/viewer.html` via `xxd -i` at configure time, or kept
  as a large string literal in a dedicated file)

### E3: operators.cpp → four files

`src/operators.cpp` (1360 LOC) mixes unrelated concerns. Extract:
- `src/convective_rhs.cpp` — `convective_rhs_impl` (~250 LOC); add forward
  declaration to `include/operators.hpp` (already present)
- `src/viscous_rhs.cpp` — `viscous_rhs_impl`, `cf_visc_energy_flux` (~220 LOC)
- `src/rhs_sensors.cpp` — `fill_ducros_cache`, `phi_compression_rhs`,
  `tree_sat_penalty` (~180 LOC)
- `src/operators.cpp` — `tree_rhs`, `compute_rhs`, `tree_cfl_dt`, AMR flux
  correction glue (~220 LOC after extraction)

Update `CMakeLists.txt` to add new source files to the `operators` static library.

---

## Phase F — Test coverage gaps

### F1: IMEX integration test

`use_imex=true` is reachable but untested. Add `tests/test_imex.cpp`: set up a
stiff acoustic problem, advance 10 steps with `use_imex=true`, verify that the
L2 residual of the implicit solve converges and that total energy is conserved
to < 1e-10.

### F2: SBP-SAT penalty energy balance test

`tree_sat_penalty` adds a penalty to boundary cells but is only covered
indirectly. Add `tests/test_sat.cpp`: construct a two-block tree, apply a
penalty step with known `sat_tau`, verify that the energy injected into the
fine block exactly equals the energy removed from the coarse block (paired
conservative correction).

### F3: Ghost-fill unit test

Covered by Phase C above.

---

## Phase D — GPU/CPU kernel unification (post-R6)

**Problem:** WENO5-Z reconstruction and Ducros sensor logic exist in both
`operators.cpp` (CPU) and `gpu_rhs.cu` (GPU). Any bug fix must be mirrored.
This is already partially mitigated by `weno5z_scalar.hpp` for the scalar path.

**Fix (after R6/mdspan):** Move the remaining duplicated logic (Ducros, face
loop structure) into `__host__ __device__` template functions in
`include/physics/rhs_kernels.hpp`. Both `operators.cpp` and `gpu_rhs.cu`
include this header. The per-architecture TUs only contain kernel launch wrappers.

This is deferred until R6 is complete because R6 changes every block access
pattern (mdspan) and would require re-doing the unification work.

---

## Execution order

```
A4 (validate)      — isolated, no risk, immediate value
A1 (EOS dedup)     — critical correctness fix
A2 (BC globals)    — depends on A2 BCRuntimeConfig struct
A3 (Ducros globals)— independent
E3 (operators split) — independent, no behavior change
E1 (live_streamer split) — independent, no behavior change
B  (TimeIntegrator) — depends on A2 (needs clean BCRuntimeConfig to pass through)
C  (GhostFiller)   — depends on A2, B
F1, F2             — independent additive tests
D  — deferred until post-R6
```

## Non-goals

- No numerical algorithm changes.
- No mdspan migration (that is R6).
- No MPI scalability work.
- No JSON schema / external config library.
- No GPU AMR support (not in scope for R9).
