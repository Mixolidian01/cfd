# R9 Maintainability Refactoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate global mutable state, decompose three monolith files, formalise the TimeIntegrator interface stub, and close test-coverage gaps. Zero numerical behaviour changes — all 18/19 gate tests must remain green after every task.

**Architecture:** The solver has a 4-layer stack (linalg → block → operators → ns_solver) plus Layer P physics functors. R9 adds no new computation — it restructures how state flows through the existing stack.

**Tech Stack:** C++20, CMake, CUDA 13.2, GCC 14.2. Gate: `cmake --build build -t ba` (expects 18/19; multiphase BN04 pre-existing failure).

---

## Task A4: SolverConfig::validate()

**Context:** `SolverConfig` in `include/ns_solver.hpp` has 30+ fields and no validation. Invalid configs silently produce garbage. This is purely additive — no behavioral change for valid configs.

**Files:**
- Modify: `include/ns_solver.hpp` (add `validate()` declaration)
- Modify: `src/ns_solver.cpp` (add definition + call in `init()`)
- Create: `tests/test_config.cpp` (new gate test)
- Modify: `CMakeLists.txt` (add test_config target)

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.cpp`:
```cpp
#include "../include/ns_solver.hpp"
#include <cassert>
#include <stdexcept>

static void test_valid_config_passes() {
    SolverConfig cfg;
    cfg.validate();  // should not throw
}

static void test_bad_cfl_throws() {
    SolverConfig cfg;
    cfg.cfl = 1.5;
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

static void test_bad_max_level_throws() {
    SolverConfig cfg;
    cfg.max_level = -1;
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

static void test_bad_gamma_throws() {
    SolverConfig cfg;
    cfg.gamma_a = 0.5;
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

static void test_bad_lts_ratio_throws() {
    SolverConfig cfg;
    cfg.lts_ratio = 3;  // must be 1, 2, or 4
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}

int main() {
    test_valid_config_passes();
    test_bad_cfl_throws();
    test_bad_max_level_throws();
    test_bad_gamma_throws();
    test_bad_lts_ratio_throws();
    return 0;
}
```

- [ ] **Step 2: Add test to CMakeLists.txt**

Find the block in `CMakeLists.txt` where other test targets are defined (look for `add_executable(test_ns` or similar). Add:
```cmake
add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config PRIVATE ns_solver operators block linalg)
add_test(NAME config COMMAND test_config)
```
Also add to the `ba` (build-all) custom target alongside the other tests.

- [ ] **Step 3: Run test to verify it fails**

```bash
cmake --build build -t test_config 2>&1 | tail -20
```
Expected: compile error — `validate()` not yet declared.

- [ ] **Step 4: Add `validate()` declaration to SolverConfig**

In `include/ns_solver.hpp`, add at the end of the `SolverConfig` struct (before the closing `}`):
```cpp
    // Throws std::invalid_argument if any field has an invalid value.
    void validate() const;
```

Also add `#include <stdexcept>` at the top of `ns_solver.hpp` if not present.

- [ ] **Step 5: Implement `validate()` in ns_solver.cpp**

Add near the top of `src/ns_solver.cpp` (before `NSSolver::init()`):
```cpp
void SolverConfig::validate() const {
    if (cfl <= 0.0 || cfl > 1.0)
        throw std::invalid_argument("SolverConfig: cfl must be in (0, 1]");
    if (t_end <= 0.0)
        throw std::invalid_argument("SolverConfig: t_end must be > 0");
    if (max_steps <= 0)
        throw std::invalid_argument("SolverConfig: max_steps must be > 0");
    if (max_level < 0 || max_level > 6)
        throw std::invalid_argument("SolverConfig: max_level must be in [0, 6]");
    if (lts_ratio != 1 && lts_ratio != 2 && lts_ratio != 4)
        throw std::invalid_argument("SolverConfig: lts_ratio must be 1, 2, or 4");
    if (acdi_ceps < 0.0)
        throw std::invalid_argument("SolverConfig: acdi_ceps must be >= 0");
    if (sat_tau < 0.0)
        throw std::invalid_argument("SolverConfig: sat_tau must be >= 0");
    if (ducros_p_threshold < 0.0 || ducros_p_threshold > 1.0)
        throw std::invalid_argument("SolverConfig: ducros_p_threshold must be in [0, 1]");
    if (ducros_blend_width < 0.0)
        throw std::invalid_argument("SolverConfig: ducros_blend_width must be >= 0");
    if (wall_T < 0.0)
        throw std::invalid_argument("SolverConfig: wall_T must be >= 0");
    if (gamma_a <= 1.0)
        throw std::invalid_argument("SolverConfig: gamma_a must be > 1");
    if (gamma_b <= 1.0)
        throw std::invalid_argument("SolverConfig: gamma_b must be > 1");
    if (p_inf_a < 0.0)
        throw std::invalid_argument("SolverConfig: p_inf_a must be >= 0");
    if (p_inf_b < 0.0)
        throw std::invalid_argument("SolverConfig: p_inf_b must be >= 0");
    if (mg_levels < 1 || mg_levels > 8)
        throw std::invalid_argument("SolverConfig: mg_levels must be in [1, 8]");
}
```

- [ ] **Step 6: Call validate() at start of NSSolver::init()**

In `src/ns_solver.cpp`, find the `NSSolver::init()` function. Add as the first statement of the body:
```cpp
cfg.validate();
```

- [ ] **Step 7: Run the test**

```bash
cmake --build build -t test_config && ./build/test_config
```
Expected: all assertions pass, exit 0.

- [ ] **Step 8: Run full gate**

```bash
cmake --build build -t ba 2>&1 | tail -10
```
Expected: 18/19 pass (multiphase pre-existing failure only).

- [ ] **Step 9: Commit**

```bash
git add include/ns_solver.hpp src/ns_solver.cpp tests/test_config.cpp CMakeLists.txt
git commit -m "R9-A4: add SolverConfig::validate() with 15 invariant checks"
```

---

## Task A1: Remove duplicate EOS statics

**Context:** `operators.cpp` maintains its own copy of the stiffened-gas EOS state
(`g_sg_active`, `g_sg_ga`, etc.) that duplicates `CellBlock::sg_active_` etc.
Both are set by `ns_solver.cpp::init()`. If they drift, results are silently wrong.

The fix is: make `operators.cpp` read directly from `CellBlock::sg_*` statics.
Add `CellBlock::set_sg_eos()` as the single setter; remove `set_sg_eos()` from
`operators.hpp`.

**Files:**
- Modify: `include/cell_block.hpp` (add `set_sg_eos()` static method)
- Modify: `src/operators.cpp` (remove g_sg_* statics, use CellBlock::sg_*_)
- Modify: `include/operators.hpp` (remove `set_sg_eos()` declaration)
- Modify: `src/ns_solver.cpp` (call CellBlock::set_sg_eos() only)

- [ ] **Step 1: Add CellBlock::set_sg_eos()**

In `include/cell_block.hpp`, after the four `static inline` EOS statics (around line 264), add:
```cpp
    static void set_sg_eos(bool active, double ga, double gb,
                           double pia, double pib) noexcept {
        sg_active_ = active;
        sg_ga_ = ga; sg_gb_ = gb;
        sg_pia_ = pia; sg_pib_ = pib;
    }
```

- [ ] **Step 2: Remove g_sg_* statics and set_sg_eos() from operators.cpp**

In `src/operators.cpp`, locate lines:
```cpp
static bool   g_sg_active = false;
static double g_sg_ga  = GAMMA, g_sg_gb  = GAMMA;
static double g_sg_pia = 0.0,   g_sg_pib = 0.0;

void set_sg_eos(bool active, double ga, double gb, double pia, double pib) noexcept {
    g_sg_active = active;
    g_sg_ga = ga; g_sg_gb = gb;
    g_sg_pia = pia; g_sg_pib = pib;
```
Delete those lines entirely.

Then replace every use of `g_sg_active` with `CellBlock::sg_active_`, `g_sg_ga` with `CellBlock::sg_ga_`, etc. throughout `operators.cpp`.

- [ ] **Step 3: Remove set_sg_eos() from operators.hpp**

In `include/operators.hpp`, delete the line:
```cpp
void set_sg_eos(bool active, double ga, double gb, double pia, double pib) noexcept;
```

- [ ] **Step 4: Update ns_solver.cpp to call CellBlock::set_sg_eos()**

In `src/ns_solver.cpp::init()`, find:
```cpp
set_sg_eos(sg, cfg.gamma_a, cfg.gamma_b, cfg.p_inf_a, cfg.p_inf_b);
```
Replace with:
```cpp
CellBlock::set_sg_eos(sg, cfg.gamma_a, cfg.gamma_b, cfg.p_inf_a, cfg.p_inf_b);
```
(Remove any now-unused `#include "operators.hpp"` if it was only needed for `set_sg_eos`; but operators.hpp is needed for other things so keep it.)

- [ ] **Step 5: Build and verify**

```bash
cmake --build build 2>&1 | grep -E "error:|warning: unused" | head -20
```
Expected: no compile errors. If `g_sg_active` etc. are still referenced, fix those.

- [ ] **Step 6: Run full gate**

```bash
cmake --build build -t ba 2>&1 | tail -10
```
Expected: 18/19 (same as before).

- [ ] **Step 7: Commit**

```bash
git add include/cell_block.hpp src/operators.cpp include/operators.hpp src/ns_solver.cpp
git commit -m "R9-A1: consolidate SG-EOS statics to CellBlock; remove operators.cpp duplicate"
```

---

## Task A2: Replace BC globals with BCRuntimeConfig member on BlockTree

**Context:** `block_tree.cpp` has four file-scope statics for BC parameters
(`g_wall_T`, `g_open_bc_p`, `g_wall_ca_cos`, `g_wall_ca_ceps`), set via static
methods on BlockTree. This prevents multi-tree runs and is thread-unsafe.

The fix: add a `BCRuntimeConfig` struct to `block_tree.hpp` and store it as a
public member of `BlockTree`. Replace all static getters/setters.

**Files:**
- Modify: `include/block_tree.hpp`
- Modify: `src/block_tree.cpp`
- Modify: `src/ns_solver.cpp`

- [ ] **Step 1: Add BCRuntimeConfig struct to block_tree.hpp**

In `include/block_tree.hpp`, after the `FaceDir` enum (around line 46), add:
```cpp
// BC runtime parameters — stored per-tree instead of as globals.
// Set by NSSolver::init() from SolverConfig before any ghost fill.
struct BCRuntimeConfig {
    double wall_T       = 0.0;   // > 0 → isothermal wall; 0 → adiabatic
    double open_bc_p    = 0.0;   // > 0 → subsonic outflow pressure; 0 → transmissive
    double wall_ca_cos  = 0.0;   // cos(θ_w) for ACDI contact angle
    double wall_ca_ceps = 0.0;   // acdi_ceps at wall (0 → disabled)
};
```

- [ ] **Step 2: Add bc_cfg member to BlockTree**

In `include/block_tree.hpp`, in the `BlockTree` struct's public section, add:
```cpp
    BCRuntimeConfig bc_cfg;
```

- [ ] **Step 3: Remove static setter declarations from BlockTree**

In `include/block_tree.hpp`, delete:
```cpp
    static void set_wall_T(double T_w) noexcept;
    static void set_open_bc_pressure(double p_inf) noexcept;
    static void set_wall_contact_angle(double cos_theta_w, double acdi_ceps) noexcept;
```

- [ ] **Step 4: Update block_tree.cpp**

In `src/block_tree.cpp`:

a. Delete the four static globals and their setter definitions:
```cpp
static double g_wall_T = 0.0;
void BlockTree::set_wall_T(double T_w) noexcept { g_wall_T = T_w; }

static double g_open_bc_p = 0.0;
void BlockTree::set_open_bc_pressure(double p_inf) noexcept { g_open_bc_p = p_inf; }

static double g_wall_ca_cos  = 0.0;
static double g_wall_ca_ceps = 0.0;
void BlockTree::set_wall_contact_angle(double cos_theta_w, double ceps) noexcept {
    g_wall_ca_cos  = cos_theta_w;
    g_wall_ca_ceps = ceps;
}
```

b. Replace all reads:
- `g_wall_T`      → pass `bc_cfg.wall_T` through the ghost-fill call chain  
- `g_open_bc_p`   → `bc_cfg.open_bc_p`
- `g_wall_ca_cos` → `bc_cfg.wall_ca_cos`
- `g_wall_ca_ceps`→ `bc_cfg.wall_ca_ceps`

The ghost-fill functions are called with `*this` (BlockTree&) — update their
internal reads to use `bc_cfg.*` from the tree reference they receive.

- [ ] **Step 5: Update ns_solver.cpp call sites**

In `src/ns_solver.cpp::init()`, replace:
```cpp
BlockTree::set_wall_T(cfg.wall_T);
BlockTree::set_open_bc_pressure(ob->far_field_pressure);
BlockTree::set_open_bc_pressure(0.0);
BlockTree::set_wall_contact_angle(ca_cos, cfg.acdi_ceps);
```
With direct assignments:
```cpp
tree.bc_cfg.wall_T    = cfg.wall_T;
tree.bc_cfg.open_bc_p = ob ? ob->far_field_pressure : 0.0;
// contact angle:
if (auto* ca = std::get_if<ContactAngleBC>(&cfg.bc_variant)) {
    tree.bc_cfg.wall_ca_cos  = std::cos(ca->theta_deg * M_PI / 180.0);
    tree.bc_cfg.wall_ca_ceps = cfg.acdi_ceps;
}
```
(Adapt to match the actual if/else structure in ns_solver.cpp::init(); the pattern above is illustrative.)

- [ ] **Step 6: Build and verify**

```bash
cmake --build build 2>&1 | grep "error:" | head -20
```

- [ ] **Step 7: Run full gate**

```bash
cmake --build build -t ba 2>&1 | tail -10
```
Expected: 18/19.

- [ ] **Step 8: Commit**

```bash
git add include/block_tree.hpp src/block_tree.cpp src/ns_solver.cpp
git commit -m "R9-A2: replace BC globals with BCRuntimeConfig member on BlockTree"
```

---

## Task A3: Replace Ducros globals with per-call DucrosConfig

**Context:** `operators.cpp` has `g_ducros_p_thr` and `g_ducros_blend` set by
`set_ducros_thresholds()`. Thread both as a struct through `compute_rhs` so
each call site can use its own values.

**Files:**
- Modify: `include/operators.hpp`
- Modify: `src/operators.cpp`
- Modify: `src/ns_solver.cpp`
- Check: `src/cuda/gpu_rhs.cu` (does it read Ducros thresholds?)

- [ ] **Step 1: Check GPU usage**

```bash
grep -n "ducros\|g_ducros\|p_thr\|blend" /home/dkoffibi/dev/git_cfd/cfd/src/cuda/gpu_rhs.cu | head -20
```
If found: add `DucrosConfig` to the GPU kernel launch parameters too.

- [ ] **Step 2: Add DucrosConfig to operators.hpp**

In `include/operators.hpp`, add after the existing includes:
```cpp
struct DucrosConfig {
    double p_threshold  = 0.1;
    double blend_width  = 0.1;
};
```

Remove the line:
```cpp
void set_ducros_thresholds(double p_threshold, double blend_width) noexcept;
```

- [ ] **Step 3: Update compute_rhs / tree_rhs signatures**

In `include/operators.hpp`, update `compute_rhs` and `tree_rhs` signatures to
accept `const DucrosConfig& ducros` as a new parameter (append to the end of
the parameter list to minimise call-site churn).

- [ ] **Step 4: Update operators.cpp**

a. Delete `g_ducros_p_thr`, `g_ducros_blend`, and `set_ducros_thresholds()`.
b. Replace uses with `ducros.p_threshold` and `ducros.blend_width`.
c. Update function definitions to accept `const DucrosConfig& ducros`.

- [ ] **Step 5: Update ns_solver.cpp call sites**

Replace calls that previously used the global (e.g., `tree_rhs(...)`) with:
```cpp
DucrosConfig ducros{ cfg.ducros_p_threshold, cfg.ducros_blend_width };
tree_rhs(tree, rhs_, ducros);
```

- [ ] **Step 6: Build, run gate, commit**

```bash
cmake --build build 2>&1 | grep "error:" | head -20
cmake --build build -t ba 2>&1 | tail -10
git add include/operators.hpp src/operators.cpp src/ns_solver.cpp
git commit -m "R9-A3: thread DucrosConfig through compute_rhs; remove global Ducros setters"
```

---

## Task E3: Split operators.cpp into four focused files

**Context:** `operators.cpp` (1360 LOC) mixes: convective RHS, viscous RHS,
Ducros sensor + phi compression, and AMR flux-correction glue. Split into
4 focused TUs — no behavior changes.

All functions are already `static` free functions or declared in `operators.hpp`;
this is a mechanical extraction.

**Files:**
- Create: `src/convective_rhs.cpp`
- Create: `src/viscous_rhs.cpp`
- Create: `src/rhs_sensors.cpp`
- Modify: `src/operators.cpp` (keep only glue: tree_rhs, compute_rhs, tree_cfl_dt, AMR correction)
- Modify: `CMakeLists.txt` (add 3 new sources to `operators` static library)

- [ ] **Step 1: Identify the function boundaries**

```bash
grep -n "^static\|^void\|^double\|^int\|^bool\|^template" /home/dkoffibi/dev/git_cfd/cfd/src/operators.cpp | head -60
```
This gives the line numbers for each function start.

- [ ] **Step 2: Create src/convective_rhs.cpp**

Move from `operators.cpp`:
- `convective_rhs_impl` (the main face-flux loop, ~250 LOC)
- Any static helpers it calls that are not used elsewhere (HLLC wrappers,
  WENO reconstruction dispatch, wall detection logic)

Add at the top of the new file:
```cpp
#include "operators.hpp"
#include "concepts.hpp"
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
#include "physics/diff_ops.hpp"
#include <cmath>
#include <algorithm>
```

- [ ] **Step 3: Create src/viscous_rhs.cpp**

Move from `operators.cpp`:
- `viscous_rhs_impl` (~220 LOC)
- `cf_visc_energy_flux<AX>` template helper
- `undo_cf_viscous_energy`

Add same includes as convective_rhs.cpp plus any viscous-specific ones.

- [ ] **Step 4: Create src/rhs_sensors.cpp**

Move from `operators.cpp`:
- `fill_ducros_cache` (Ducros sensor computation)
- `phi_compression_rhs` (ACDI compression)
- `tree_sat_penalty` (SBP-SAT penalty)

- [ ] **Step 5: Verify operators.cpp is now glue-only**

After extraction, `operators.cpp` should contain only:
- `tree_rhs` (top-level loop: fill ghosts → compute_rhs → flux correction)
- `compute_rhs` (dispatches convective + viscous + sensors)
- `tree_cfl_dt`
- AMR flux correction calls (`accumulate_cf_fine_fluxes`, `undo_cf_face_flux`)
- Instantiation matrix includes

Check line count:
```bash
wc -l /home/dkoffibi/dev/git_cfd/cfd/src/operators.cpp
```
Target: < 350 lines.

- [ ] **Step 6: Update CMakeLists.txt**

Find the `add_library(operators STATIC ...)` block and add:
```cmake
    src/convective_rhs.cpp
    src/viscous_rhs.cpp
    src/rhs_sensors.cpp
```

- [ ] **Step 7: Build**

```bash
cmake --build build 2>&1 | grep "error:" | head -20
```
Fix any "undefined symbol" errors from functions that are now in a different TU
but were previously static — they need to be either made non-static (and declared
in a local header or in operators.hpp) or moved back to the same TU as their
caller.

- [ ] **Step 8: Run gate, commit**

```bash
cmake --build build -t ba 2>&1 | tail -10
git add src/convective_rhs.cpp src/viscous_rhs.cpp src/rhs_sensors.cpp src/operators.cpp CMakeLists.txt
git commit -m "R9-E3: split operators.cpp into convective, viscous, sensors, and glue TUs"
```

---

## Task E1: Split live_streamer.cpp

**Context:** `src/live_streamer.cpp` (1897 LOC) mixes HTTP socket handling,
WebSocket framing, frame packing, LZ4 compression, and a large embedded HTML
viewer string. Extract into 3 separate files.

**Files:**
- Create: `src/viewer_html.cpp` (the embedded HTML string only)
- Modify: `src/live_streamer.cpp` (remove HTML string, delegate to viewer_html.cpp)
- Modify: `CMakeLists.txt` (add viewer_html.cpp to ns_solver library)

Note: Full HTTP server and frame packer extraction is lower priority; the
embedded HTML extraction alone removes ~500 LOC from the monolith and makes
the viewer HTML independently editable.

- [ ] **Step 1: Find the HTML string in live_streamer.cpp**

```bash
grep -n "viewer_html\|<html\|<!DOCTYPE\|const char\|static.*html" /home/dkoffibi/dev/git_cfd/cfd/src/live_streamer.cpp | head -10
```
Note the function/variable name and line range.

- [ ] **Step 2: Create src/viewer_html.cpp**

Move the HTML string function (e.g., `static std::string viewer_html()`) to
`src/viewer_html.cpp`. Add a forward declaration to `include/live_streamer.hpp`:
```cpp
std::string viewer_html();
```

- [ ] **Step 3: Remove from live_streamer.cpp**

Delete the HTML string function from `live_streamer.cpp`.

- [ ] **Step 4: Update CMakeLists.txt**

Add `src/viewer_html.cpp` to the `ns_solver` (or `live_streamer`) library target.

- [ ] **Step 5: Build, run gate, commit**

```bash
cmake --build build 2>&1 | grep "error:" | head -10
cmake --build build -t ba 2>&1 | tail -10
git add src/viewer_html.cpp src/live_streamer.cpp include/live_streamer.hpp CMakeLists.txt
git commit -m "R9-E1: extract embedded HTML viewer string to viewer_html.cpp"
```

---

## Task B: Extract CpuRk3Integrator and LtsIntegrator (P10-A2)

**Context:** `TimeIntegrator` interface already exists in `ns_solver.hpp` (lines 37-40).
The TODO P10-A2 says: extract CPU and LTS implementations. The GPU is already
`IGpuSolver : TimeIntegrator`. This task extracts the two remaining inline
implementations.

**Files:**
- Create: `include/cpu_rk3.hpp`
- Create: `src/cpu_rk3.cpp`
- Create: `include/lts_integrator.hpp`
- Create: `src/lts_integrator.cpp`
- Modify: `src/ns_solver.cpp` (remove advance() and advance_lts() bodies)
- Modify: `include/ns_solver.hpp` (add integrator_ member)
- Modify: `CMakeLists.txt`

**NOTE:** This task is the highest-risk in R9. Read `src/ns_solver.cpp` completely
before starting. The CPU and LTS RK3 bodies access `rhs_`, `Qn_`, `Qs_` private
members of `NSSolver` — these need to either be moved to the integrator or passed
via reference. Prefer passing by reference (cleaner ownership). Read the full
advance() and advance_lts() implementations first.

- [ ] **Step 1: Read ns_solver.cpp advance() and advance_lts() bodies**

```bash
grep -n "NSSolver::advance\|NSSolver::lts\|NSSolver::save\|NSSolver::copy" /home/dkoffibi/dev/git_cfd/cfd/src/ns_solver.cpp
```
Note all private helpers called from advance() and advance_lts().

- [ ] **Step 2: Design the interface**

`CpuRk3Integrator::step()` needs access to:
- `BlockTree& tree` (passed in)
- `std::vector<CellBlock>& rhs_, Qn_, Qs_` (scratch — pass by reference or keep in NSSolver and pass)
- `SolverConfig& cfg` (pass const ref)
- `MpiPartition*`, `IGpuSolver*`, `LiveStreamer*` (optional, pass pointers)

Recommended: Keep scratch vectors in NSSolver; pass them by reference to `step()`.
Extend the `TimeIntegrator` signature or create an `NSSolverState` POD:
```cpp
struct RK3Scratch {
    std::vector<CellBlock>& rhs;
    std::vector<CellBlock>& Qn;
    std::vector<CellBlock>& Qs;
};
```

Pass `RK3Scratch` to `step()`. This keeps NSSolver as the owner.

- [ ] **Step 3: Create CpuRk3Integrator**

`include/cpu_rk3.hpp`:
```cpp
#pragma once
#include "ns_solver.hpp"  // TimeIntegrator, SolverConfig, RK3Scratch
struct CpuRk3Integrator : TimeIntegrator {
    const SolverConfig& cfg;
    MpiPartition*       mpi;
    explicit CpuRk3Integrator(const SolverConfig& cfg, MpiPartition* mpi = nullptr);
    double step(BlockTree& tree, double cfl,
                RK3Scratch scratch, LiveStreamer* streamer) override;
};
```

Move the `NSSolver::advance()` body into `CpuRk3Integrator::step()`.

- [ ] **Step 4: Create LtsIntegrator**

Similar: move `advance_lts()` and `lts_rk3_level()` into `LtsIntegrator::step()`.

- [ ] **Step 5: Wire into NSSolver**

Add to `NSSolver`:
```cpp
std::unique_ptr<TimeIntegrator> integrator_;
```
In `NSSolver::init()`, create the integrator:
```cpp
if (cfg.use_gpu && gpu_solver_)
    integrator_ = nullptr;  // GPU uses gpu_solver_ directly
else if (cfg.use_lts)
    integrator_ = std::make_unique<LtsIntegrator>(cfg, mpi_);
else
    integrator_ = std::make_unique<CpuRk3Integrator>(cfg, mpi_);
```

`NSSolver::advance()` becomes:
```cpp
double NSSolver::advance() {
    if (gpu_solver_) return gpu_solver_->step(tree, cfg.cfl);
    RK3Scratch s{rhs_, Qn_, Qs_};
    return integrator_->step(tree, cfg.cfl, s, streamer_);
}
```

- [ ] **Step 6: Build, run full gate, commit**

```bash
cmake --build build 2>&1 | grep "error:" | head -20
cmake --build build -t ba 2>&1 | tail -10
git add include/cpu_rk3.hpp src/cpu_rk3.cpp include/lts_integrator.hpp src/lts_integrator.cpp \
        include/ns_solver.hpp src/ns_solver.cpp CMakeLists.txt
git commit -m "R9-B: extract CpuRk3Integrator and LtsIntegrator (P10-A2)"
```

---

## Task F1: IMEX integration test

- [ ] **Step 1: Write test**

Create `tests/test_imex.cpp` with a simple stiff problem (acoustic oscillation
in a 1D periodic box). Use `cfg.use_imex = true`. Advance 10 steps. Assert:
- Total energy conserved to < 1e-8 relative error
- No NaN/Inf in any field
- The implicit residual converges in < 20 GMRES iterations

- [ ] **Step 2: Add to CMakeLists.txt and gate**

- [ ] **Step 3: Commit**

---

## Task F2: SBP-SAT penalty balance test

- [ ] **Step 1: Write test**

Create `tests/test_sat.cpp`. Construct a two-leaf tree (coarse + fine level).
Apply `tree_sat_penalty` with `sat_tau = 0.5`. Measure total energy before and
after. Assert: |ΔE_fine + ΔE_coarse| < 1e-14 (paired conservative correction).

- [ ] **Step 2: Add to CMakeLists.txt and gate**

- [ ] **Step 3: Commit**
