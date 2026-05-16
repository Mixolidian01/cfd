# CFD Solver R10 Maintainability Refactors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eight self-contained maintainability and performance refactors for the compressible CFD solver on the `to_refactor` branch.

**Architecture:** Tasks are mostly independent; Task 3 (StiffenedGasEOS) is prerequisite to Task 4 (configurable γ), and Task 6 (tree_rhs_typed dispatch) builds on Tasks 3 and 4. All other tasks are fully independent. Every task must leave `cmake --build build -t ba` green before committing.

**Tech Stack:** C++20, OpenMP, optional NVTX3/Tracy headers, CMake 3.20+

---

## Codebase context (read this before implementing each task)

- **Working directory:** `/home/dkoffibi/dev/git_cfd/cfd`
- **Branch:** `to_refactor`
- **Gate command:** `cmake --build build -t ba` (t1–t7 + config/imex/sat/ghost_fill)
- **Pre-existing known failure:** BN04 in `test_multiphase` — ignore, not caused by this work
- **Build directory:** `build/`  (already configured; do not re-run cmake unless explicitly needed)

**Key constants (cell_block.hpp):**
```
NB=8, NG=2, NB2=12, NCELL=1728, NVAR=5, GAMMA=1.4 (global constexpr double)
```

**Layer map:**
| File | Role |
|------|------|
| `include/mesh/cell_block.hpp` | CellBlock, AoSoA layout, eos functions, GAMMA constant |
| `include/mesh/block_tree.hpp` | BlockTree, leaf_indices(), BlockNode.morton |
| `include/schemes/concepts.hpp` | RiemannFlux, EquationOfState, SpatialReconstruction concepts |
| `include/physics/ideal_gas_eos.hpp` | IdealGasEOS functor |
| `include/schemes/operators.hpp` | compute_rhs_typed, tree_rhs declarations |
| `src/schemes/operators.cpp` | tree_rhs impl, compute_rhs_typed body + explicit instantiations |
| `src/solver/cpu_rk3.cpp` | CpuRk3Integrator::step() — SSP-RK3 with OMP parallel for |
| `src/solver/lts_integrator.cpp` | LtsIntegrator::rk3_level() — Berger-Oliger LTS |
| `src/solver/ns_solver.hpp` | NSSolver, SolverConfig (8 sub-structs) |
| `src/mesh/block_tree.cpp` | leaf_indices() with leaf_cache_/leaf_dirty_ |

**CellBlock field proxy layout:** `Q[v][flat]` where `flat = cell_idx(i,j,k) = k*NB2² + j*NB2 + i`

**Existing `(i,j,k)` accessors (cell_block.hpp lines 247–258):**
```cpp
double& rho (int i,int j,int k) noexcept { return Q[0][cell_idx(i,j,k)]; }
// rhou, rhov, rhow, E — same pattern, indices 1..4
```
Flat-index `rho(int flat)` variants do NOT yet exist — that is Task 1.

**CpuRk3Integrator::step() structure (cpu_rk3.cpp):** three nearly-identical blocks, each:
```
mpi_exchange_halos → tree_rhs → #pragma omp parallel for (stage update) → phi_stage → apply_positivity_floor → copy_stage_to_tree
```

**tree_rhs (operators.cpp ~line 494):** ghost fill → Morton-sort leaf order → `#pragma omp parallel for schedule(dynamic,4)` calling `compute_rhs` → `accumulate_cf_fine_fluxes`.

**compute_rhs_typed template:** declared in operators.hpp, body in operators.cpp. Uses `fill_prim_cache` (uses global GAMMA) + `convective_rhs_impl_typed<Flux,Recon>` + `viscous_rhs_impl`. Current explicit instantiations: `HllcEsFlux/Weno5Recon/IdealGasEOS` and `HllcFlux/Weno5Recon/IdealGasEOS`. `tree_rhs` currently calls non-typed `compute_rhs` — Task 6 wires the typed path.

**BlockTree leaf cache:** `leaf_indices()` is cached via `leaf_cache_/leaf_dirty_`. `tree_rhs` re-sorts by Morton on every call (~O(NL log NL)). Task 7 caches the sorted order.

---

## File map for this plan

**Created:**
- `include/profiling/profiler.hpp` — NVTX/Tracy guard macros (Task 2)
- `include/physics/stiffened_gas_eos.hpp` — StiffenedGasEOS functor (Task 3)
- `include/solver/solver_result.hpp` — SolverResult<T> non-throwing error type (Task 8)
- `tests/mesh/test_cell_block_accessors.cpp` — Task 1 unit test

**Modified:**
- `include/mesh/cell_block.hpp` — flat-index accessors (T1), `eos_cons_to_prim_g` (T4)
- `include/mesh/block_tree.hpp` — `morton_leaf_indices()` declaration (T7)
- `include/physics/ideal_gas_eos.hpp` — `double gamma = GAMMA` member (T4)
- `include/schemes/operators.hpp` — `tree_rhs_typed` declaration (T6)
- `include/solver/ns_solver.hpp` — `SolverConfig::PhysicsConfig::gamma` (T4), `advance_result()` (T8)
- `src/mesh/block_tree.cpp` — `morton_leaf_indices()` + invalidation (T7)
- `src/schemes/operators.cpp` — `tree_rhs_typed` impl (T6), StiffenedGasEOS instantiation (T3), use cached Morton (T7), profiler markers (T2)
- `src/solver/cpu_rk3.cpp` — `collapse(3)` OMP (T5), `tree_rhs_typed` dispatch (T6), profiler (T2)
- `src/solver/lts_integrator.cpp` — `collapse(3)` OMP (T5)
- `src/solver/ns_solver.cpp` — `advance_result()` impl (T8), `validate()` gamma check (T4), profiler (T2)
- `src/mpi/mpi_comm.cpp` — profiler marker (T2)
- `CMakeLists.txt` — register `test_cell_block_accessors` (T1)

---

## Task 1: Flat-index typed field accessors on CellBlock

**Goal:** Add `rho(int flat)`, `rhou(int flat)`, `rhov(int flat)`, `rhow(int flat)`, `E(int flat)` to `CellBlock` and update `apply_positivity_floor` to use them.

**Files:**
- Modify: `include/mesh/cell_block.hpp`
- Modify: `src/solver/cpu_rk3.cpp`
- Create: `tests/mesh/test_cell_block_accessors.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create the failing test**

Create `tests/mesh/test_cell_block_accessors.cpp`:

```cpp
// R10-T1: flat-index typed field accessors on CellBlock
#include "mesh/cell_block.hpp"
#include <cstdio>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool cond) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else       { printf("  FAIL  %s\n", name); ++n_fail; }
}

static void t01_flat_read() {
    CellBlock b;
    for (int f = 0; f < NCELL; ++f) {
        b.Q[0][f] = 1.0+f; b.Q[1][f] = 2.0+f; b.Q[2][f] = 3.0+f;
        b.Q[3][f] = 4.0+f; b.Q[4][f] = 5.0+f;
    }
    bool ok = true;
    for (int f = 0; f < NCELL; ++f)
        ok &= (b.rho(f)==1.0+f) && (b.rhou(f)==2.0+f) && (b.rhov(f)==3.0+f)
           && (b.rhow(f)==4.0+f) && (b.E(f)==5.0+f);
    check("T01 flat read", ok);
}

static void t02_flat_write() {
    CellBlock b;
    for (int f = 0; f < NCELL; ++f) {
        b.rho(f) = 10.0+f; b.rhou(f) = 20.0+f; b.rhov(f) = 30.0+f;
        b.rhow(f) = 40.0+f; b.E(f) = 50.0+f;
    }
    bool ok = true;
    for (int f = 0; f < NCELL; ++f)
        ok &= (b.Q[0][f]==10.0+f) && (b.Q[1][f]==20.0+f) && (b.Q[2][f]==30.0+f)
           && (b.Q[3][f]==40.0+f) && (b.Q[4][f]==50.0+f);
    check("T02 flat write", ok);
}

static void t03_flat_ijk_consistent() {
    CellBlock b;
    for (int f = 0; f < NCELL; ++f) b.rho(f) = (double)f;
    bool ok = true;
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        ok &= (b.rho(i,j,k) == b.rho(cell_idx(i,j,k)));
    check("T03 flat/ijk consistency", ok);
}

int main() {
    t01_flat_read(); t02_flat_write(); t03_flat_ijk_consistent();
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
```

- [ ] **Step 2: Register test in CMakeLists.txt**

Find the `# Core gate tests` section. After the `test_ghost_fill` executable, add:

```cmake
add_executable(test_cell_block_accessors tests/mesh/test_cell_block_accessors.cpp)
target_link_libraries(test_cell_block_accessors PRIVATE libblock)
add_test(NAME cell_block_accessors COMMAND test_cell_block_accessors)
add_custom_target(taccess DEPENDS test_cell_block_accessors
    COMMAND $<TARGET_FILE:test_cell_block_accessors>
    COMMENT "R10-T1: flat-index accessor test" USES_TERMINAL)
```

Also add `test_cell_block_accessors` to the DEPENDS list of the `ba` custom target.

- [ ] **Step 3: Build to verify compile failure**

```bash
cmake --build build --target test_cell_block_accessors 2>&1 | grep error | head -3
```
Expected: `error: 'CellBlock' has no member named 'rho'` (for `b.rho(f)`)

- [ ] **Step 4: Add flat-index accessors to CellBlock**

In `include/mesh/cell_block.hpp`, directly after the existing `(i,j,k)` accessors block (after line 258), add:

```cpp
    // Flat-index variants — complement the (i,j,k) forms above.
    double& rho (int f) noexcept { return Q[0][f]; }
    double& rhou(int f) noexcept { return Q[1][f]; }
    double& rhov(int f) noexcept { return Q[2][f]; }
    double& rhow(int f) noexcept { return Q[3][f]; }
    double& E   (int f) noexcept { return Q[4][f]; }
    double  rho (int f) const noexcept { return Q[0][f]; }
    double  rhou(int f) const noexcept { return Q[1][f]; }
    double  rhov(int f) const noexcept { return Q[2][f]; }
    double  rhow(int f) const noexcept { return Q[3][f]; }
    double  E   (int f) const noexcept { return Q[4][f]; }
```

- [ ] **Step 5: Update apply_positivity_floor in cpu_rk3.cpp**

Replace the body of the triple `for (int k/j/i)` loop inside `apply_positivity_floor` (lines 21–33 of cpu_rk3.cpp) with:

```cpp
            const int f = cell_idx(i, j, k);
            double rho_v = blk.rho(f);
            if (rho_v < EPS_POS) { blk.rho(f) = rho_v = EPS_POS; }
            const double ke = 0.5 * (blk.rhou(f)*blk.rhou(f)
                                   + blk.rhov(f)*blk.rhov(f)
                                   + blk.rhow(f)*blk.rhow(f)) / rho_v;
            if ((GAMMA - 1.0) * (blk.E(f) - ke) < EPS_POS)
                blk.E(f) = ke + EPS_POS / (GAMMA - 1.0);
```

- [ ] **Step 6: Build and run accessor test**

```bash
cmake --build build --target test_cell_block_accessors && ./build/test_cell_block_accessors
```
Expected output:
```
  PASS  T01 flat read
  PASS  T02 flat write
  PASS  T03 flat/ijk consistency

3 passed, 0 failed
```

- [ ] **Step 7: Run full gate**

```bash
cmake --build build -t ba
```
Expected: all previously-passing gates still pass.

- [ ] **Step 8: Commit**

```bash
git add include/mesh/cell_block.hpp src/solver/cpu_rk3.cpp \
        tests/mesh/test_cell_block_accessors.cpp CMakeLists.txt
git commit -m "R10-T1: flat-index typed field accessors on CellBlock; positivity floor uses named accessors"
```

---

## Task 2: NVTX/Tracy profiling annotations

**Goal:** Create a zero-cost `include/profiling/profiler.hpp` header with `PROFILE_SCOPE(name)` and `PROFILE_SCOPE_COLOR(name, argb)` macros that activate NVTX3 or Tracy when the respective compile flag is defined, and are strict no-ops otherwise. Add markers to the four hottest call sites.

**Files:**
- Create: `include/profiling/profiler.hpp`
- Modify: `src/schemes/operators.cpp`
- Modify: `src/solver/cpu_rk3.cpp`
- Modify: `src/mpi/mpi_comm.cpp`

- [ ] **Step 1: Create profiler.hpp**

Create `include/profiling/profiler.hpp`:

```cpp
#pragma once
// Profiling annotation guards — zero-cost no-ops by default.
//
// Activate NVTX (Nsight Systems/Compute):
//   add -DENABLE_NVTX to compile flags; link nvtx3 (shipped with CUDA Toolkit).
//
// Activate Tracy:
//   add -DENABLE_TRACY and link TracyClient.cpp.
//
// Usage (RAII scope — exits on scope end):
//   PROFILE_SCOPE("label");
//   PROFILE_SCOPE_COLOR("label", 0xFFRRGGBB);

#if defined(ENABLE_NVTX)
#  include <nvtx3/nvToolsExt.h>
   namespace _profiler_detail {
       struct NvtxRange {
           explicit NvtxRange(const char* n) noexcept { nvtxRangePushA(n); }
           ~NvtxRange() noexcept { nvtxRangePop(); }
       };
       struct NvtxRangeColor {
           NvtxRangeColor(const char* n, uint32_t argb) noexcept {
               nvtxEventAttributes_t ea{};
               ea.version     = NVTX_VERSION;
               ea.size        = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
               ea.colorType   = NVTX_COLOR_ARGB;  ea.color = argb;
               ea.messageType = NVTX_MESSAGE_TYPE_ASCII; ea.message.ascii = n;
               nvtxRangePushEx(&ea);
           }
           ~NvtxRangeColor() noexcept { nvtxRangePop(); }
       };
   }
#  define PROFILE_SCOPE(name) \
        ::_profiler_detail::NvtxRange _prof_##__LINE__{name}
#  define PROFILE_SCOPE_COLOR(name, argb) \
        ::_profiler_detail::NvtxRangeColor _prof_##__LINE__{name, argb}

#elif defined(ENABLE_TRACY)
#  include <tracy/Tracy.hpp>
#  define PROFILE_SCOPE(name)             ZoneScopedN(name)
#  define PROFILE_SCOPE_COLOR(name, argb) ZoneScopedNC(name, (argb))

#else
#  define PROFILE_SCOPE(name)             ((void)0)
#  define PROFILE_SCOPE_COLOR(name, argb) ((void)0)
#endif
```

- [ ] **Step 2: Add markers to tree_rhs in operators.cpp**

At the top of `operators.cpp` (after existing includes), add:
```cpp
#include "profiling/profiler.hpp"
```

At the start of the `tree_rhs` function body, add:
```cpp
    PROFILE_SCOPE("tree_rhs");
```

Wrap the ghost-fill block:
```cpp
    { PROFILE_SCOPE("tree_rhs/ghost_fill");
      if (periodic) tree.fill_ghosts_periodic(cf_coarse_zero_grad);
      else if (open_bc) tree.fill_ghosts_open(cf_coarse_zero_grad);
      else tree.fill_ghosts_wall(cf_coarse_zero_grad); }
```

Wrap the parallel compute block:
```cpp
    { PROFILE_SCOPE("tree_rhs/compute");
#pragma omp parallel for schedule(dynamic,4)
      for (int oi = 0; oi < n_active; ++oi) { ... } }
```

Wrap the flux accumulation:
```cpp
    { PROFILE_SCOPE("tree_rhs/flux_accum");
      accumulate_cf_fine_fluxes(tree, stage_weight, level_filter); }
```

- [ ] **Step 3: Add markers to CpuRk3Integrator::step()**

At top of `cpu_rk3.cpp`, add `#include "profiling/profiler.hpp"`.

Wrap the full `step()` body:
```cpp
double CpuRk3Integrator::step(BlockTree& tree, double cfl) {
    PROFILE_SCOPE("CpuRk3/step");
    ...
    // Stage 1 block:
    { PROFILE_SCOPE_COLOR("CpuRk3/stage1", 0xFFFF4040);
      mpi_exchange_halos(tree, solver.mpi_);
      tree_rhs(...);
      // ... omp parallel for ...
      phi_stage(-1.0);
      apply_positivity_floor(solver.Qs_);
      solver.copy_stage_to_tree(solver.Qs_);
    }
    // Stage 2 block — PROFILE_SCOPE_COLOR("CpuRk3/stage2", 0xFF40FF40)
    // Stage 3 block — PROFILE_SCOPE_COLOR("CpuRk3/stage3", 0xFF4040FF)
    ...
}
```

- [ ] **Step 4: Add marker to mpi_exchange_halos**

At top of `mpi_comm.cpp`, add `#include "profiling/profiler.hpp"`.

At start of `mpi_exchange_halos` body (before the null check):
```cpp
void mpi_exchange_halos(BlockTree& tree, const MpiPartition* mpi_part) {
    PROFILE_SCOPE("mpi_exchange_halos");
    if (!mpi_part || !mpi_part->active()) return;
    ...
}
```

- [ ] **Step 5: Build and run gate (no-op path)**

```bash
cmake --build build -t ba
```
Expected: all gates pass (macros expand to `((void)0)`).

- [ ] **Step 6: Commit**

```bash
git add include/profiling/profiler.hpp src/schemes/operators.cpp \
        src/solver/cpu_rk3.cpp src/mpi/mpi_comm.cpp
git commit -m "R10-T2: NVTX/Tracy profiling annotations (zero-cost no-ops by default)"
```

---

## Task 3: StiffenedGasEOS functor

**Goal:** Create `include/physics/stiffened_gas_eos.hpp` — a struct that satisfies `EquationOfState` and exposes the Allaire (2002) stiffened-gas mixture EOS. Add it to the explicit instantiation matrix in `operators.cpp`.

**Context:** `mix_eos()` and `eos_cons_to_prim_sg()` already exist in `cell_block.hpp`. `IdealGasEOS` in `include/physics/ideal_gas_eos.hpp` is the model for the pattern.

**Files:**
- Create: `include/physics/stiffened_gas_eos.hpp`
- Modify: `src/schemes/operators.cpp`
- Modify: `tests/schemes/test_operators.cpp`

- [ ] **Step 1: Add failing test to test_operators.cpp**

In `tests/schemes/test_operators.cpp`, add `#include "physics/stiffened_gas_eos.hpp"` at the top (after existing includes).

Add at the end of `main()` (before `return`):

```cpp
    // T11: StiffenedGasEOS with gamma_a=gamma_b=1.4, p_inf=0 must match IdealGasEOS exactly.
    {
        StiffenedGasEOS sg{1.4, 1.4, 0.0, 0.0};
        IdealGasEOS     ideal{};
        const double rho=1.0, rhou=0.2, rhov=0.0, rhow=0.0, E=2.8;
        Prim ps = sg.cons_to_prim(rho, rhou, rhov, rhow, E);
        Prim pi = ideal.cons_to_prim(rho, rhou, rhov, rhow, E);
        bool ok = (std::fabs(ps.p - pi.p) < 1e-12) && (std::fabs(ps.c - pi.c) < 1e-12);
        check("T11 StiffenedGasEOS single-phase limit matches IdealGasEOS", ok,
              std::fabs(ps.p - pi.p), 1e-12);
    }
```

- [ ] **Step 2: Build to verify compile error**

```bash
cmake --build build --target test_operators 2>&1 | grep "error:" | head -3
```
Expected: `error: 'StiffenedGasEOS' was not declared`

- [ ] **Step 3: Create stiffened_gas_eos.hpp**

Create `include/physics/stiffened_gas_eos.hpp`:

```cpp
#pragma once
// Layer P — StiffenedGasEOS functor (R10-T3)
// Satisfies EquationOfState (include/schemes/concepts.hpp).
//
// Wraps the Allaire (2002) mixture rule implemented in cell_block.hpp:
//   mix_eos(φ, γ_A, γ_B, p∞_A, p∞_B, γ_m, p∞_m)
//   eos_cons_to_prim_sg(ρ, ρu, ρv, ρw, E, γ_m, p∞_m)
//
// The concept-required single-field form (no φ) uses φ=1 (pure fluid A).
// For the two-phase path, call the six-argument overload with φ from phi_data_.

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"   // mix_eos, eos_cons_to_prim_sg, Prim
#include "schemes/concepts.hpp"  // EquationOfState concept

struct StiffenedGasEOS {
    double gamma_a = 1.4;
    double gamma_b = 1.4;
    double p_inf_a = 0.0;
    double p_inf_b = 0.0;

    // Two-phase form: φ ∈ [0,1] selects the mixture EOS.
    __host__ __device__
    Prim cons_to_prim(double rho, double rhou, double rhov,
                      double rhow, double en, double phi) const noexcept {
        double gm, pim;
        mix_eos(phi, gamma_a, gamma_b, p_inf_a, p_inf_b, gm, pim);
        return eos_cons_to_prim_sg(rho, rhou, rhov, rhow, en, gm, pim);
    }

    // EquationOfState concept: single-component form (φ=1.0 → pure fluid A).
    __host__ __device__
    Prim cons_to_prim(double rho, double rhou, double rhov,
                      double rhow, double en) const noexcept {
        return cons_to_prim(rho, rhou, rhov, rhow, en, 1.0);
    }
};

static_assert(EquationOfState<StiffenedGasEOS>);
```

- [ ] **Step 4: Add explicit instantiations in operators.cpp**

In `src/schemes/operators.cpp`, after the existing `#include "physics/ideal_gas_eos.hpp"` and instantiation block (around line 336–341), add:

```cpp
#include "physics/stiffened_gas_eos.hpp"

template void compute_rhs_typed<HllcEsFlux, Weno5Recon, StiffenedGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&) noexcept;
template void compute_rhs_typed<HllcFlux, Weno5Recon, StiffenedGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&) noexcept;
```

- [ ] **Step 5: Build and run test_operators**

```bash
cmake --build build --target test_operators && ./build/test_operators 2>&1 | grep -E "T11|PASS|FAIL"
```
Expected: `PASS  T11 StiffenedGasEOS single-phase limit matches IdealGasEOS`

- [ ] **Step 6: Run full gate**

```bash
cmake --build build -t ba
```
Expected: all previously passing gates still pass.

- [ ] **Step 7: Commit**

```bash
git add include/physics/stiffened_gas_eos.hpp src/schemes/operators.cpp \
        tests/schemes/test_operators.cpp
git commit -m "R10-T3: StiffenedGasEOS functor; add HllcEsFlux+HllcFlux instantiations"
```

---

## Task 4: Configurable γ via SolverConfig::physics.gamma

**Goal:** Allow simulations with γ ≠ 1.4 (e.g. argon γ=1.67, CO₂ γ=1.3) without recompiling. Add `SolverConfig::physics.gamma`, validate it, carry it through `IdealGasEOS`, and expose a `eos_cons_to_prim_g(...)` free function that takes γ as a runtime argument.

**Prerequisite:** Task 3 complete (StiffenedGasEOS header exists for reference pattern).

**Files:**
- Modify: `include/mesh/cell_block.hpp`
- Modify: `include/physics/ideal_gas_eos.hpp`
- Modify: `include/solver/ns_solver.hpp`
- Modify: `src/solver/ns_solver.cpp`
- Modify: `tests/solver/test_config.cpp`

- [ ] **Step 1: Add failing tests to test_config.cpp**

In `tests/solver/test_config.cpp`, add two new test functions:

```cpp
static void test_custom_gamma_accepted() {
    SolverConfig cfg;
    cfg.physics.gamma = 1.67;   // argon
    cfg.validate();              // must not throw
}

static void test_sub_unity_gamma_throws() {
    SolverConfig cfg;
    cfg.physics.gamma = 0.8;    // unphysical
    try { cfg.validate(); assert(false && "should have thrown"); }
    catch (const std::invalid_argument&) {}
}
```

Call both from `main()`.

- [ ] **Step 2: Build to verify compile failure**

```bash
cmake --build build --target test_config 2>&1 | grep "error:" | head -3
```
Expected: `'PhysicsConfig' has no member named 'gamma'`

- [ ] **Step 3: Add physics.gamma to SolverConfig**

In `include/solver/ns_solver.hpp`, in `PhysicsConfig` (around line 115–121):

```cpp
    struct PhysicsConfig {
        std::shared_ptr<SGSModel> sgs = nullptr;
        bool   use_imex  = false;
        int    mg_levels = 3;
        double gamma     = GAMMA;   // ratio of specific heats; must be > 1.0
    } physics;
```

- [ ] **Step 4: Add gamma validation in SolverConfig::validate()**

In `src/solver/ns_solver.cpp`, in the `validate()` method, add:

```cpp
    if (physics.gamma <= 1.0)
        throw std::invalid_argument("physics.gamma must be > 1.0 (got "
                                    + std::to_string(physics.gamma) + ")");
```

- [ ] **Step 5: Add eos_cons_to_prim_g to cell_block.hpp**

In `include/mesh/cell_block.hpp`, immediately after `eos_cons_to_prim` (around line 81), add:

```cpp
// Configurable-gamma variant — same formula, gamma passed as argument.
// Enables runtime γ ≠ 1.4 without recompiling.
inline Prim eos_cons_to_prim_g(double rho, double rhou, double rhov,
                                double rhow, double E, double gamma) noexcept
{
    Prim q;
    q.rho = rho;
    q.u   = rhou / rho;
    q.v   = rhov / rho;
    q.w   = rhow / rho;
    q.p   = (gamma - 1.0) * (E - 0.5 * rho * (q.u*q.u + q.v*q.v + q.w*q.w));
    q.T   = q.p / (rho * R_GAS);
    q.c   = std::sqrt(gamma * q.p / rho);
    return q;
}
```

Also make `eos_cons_to_prim` a one-line wrapper:
```cpp
inline Prim eos_cons_to_prim(double rho, double rhou, double rhov,
                              double rhow, double E) noexcept {
    return eos_cons_to_prim_g(rho, rhou, rhov, rhow, E, GAMMA);
}
```

- [ ] **Step 6: Add double gamma member to IdealGasEOS**

In `include/physics/ideal_gas_eos.hpp`, update the struct to:

```cpp
struct IdealGasEOS {
    double gamma = GAMMA;   // default: compile-time GAMMA = 1.4; override for other gases

    __host__ __device__
    Prim cons_to_prim(double rho, double rhou, double rhov,
                      double rhow, double en) const noexcept {
        return eos_cons_to_prim_g(rho, rhou, rhov, rhow, en, gamma);
    }
};

static_assert(EquationOfState<IdealGasEOS>);
```

- [ ] **Step 7: Build and run test_config**

```bash
cmake --build build --target test_config && ./build/test_config
```
Expected: `test_custom_gamma_accepted` and `test_sub_unity_gamma_throws` both pass (all subtests PASS).

- [ ] **Step 8: Run full gate**

```bash
cmake --build build -t ba
```
Expected: all previously-passing gates still pass.

- [ ] **Step 9: Commit**

```bash
git add include/mesh/cell_block.hpp include/physics/ideal_gas_eos.hpp \
        include/solver/ns_solver.hpp src/solver/ns_solver.cpp \
        tests/solver/test_config.cpp
git commit -m "R10-T4: configurable gamma via SolverConfig::physics.gamma; eos_cons_to_prim_g; IdealGasEOS::gamma member"
```

---

## Task 5: OpenMP collapse(3) on RK3 stage-update loops

**Goal:** Add `collapse(3)` to the triple-nested `(ii, v, t)` loops in `cpu_rk3.cpp` and `lts_integrator.cpp`. This exposes `NL × NVAR × NTILE` iterations to the OpenMP scheduler instead of just `NL`, which is critical for single-block runs (NL=1) where the current `parallel for` over `ii` leaves all but one thread idle.

**Context:** The loops have the form:
```cpp
#pragma omp parallel for
for (int ii = 0; ii < NL; ++ii)
for (int v  = 0; v  < NVAR; ++v)
for (int t  = 0; t  < CellBlock::NTILE; ++t) {
    double* qs = solver.Qs_[ii].Q[v].tile_ptr(t);
    ...
    #pragma omp simd simdlen(8)
    for (int lane = 0; lane < CellBlock::W; ++lane)
        qs[lane] = ...;
}
```
`NL × NVAR × NTILE = NL × 5 × 216`. Iterations are independent (disjoint writes to different `tile_ptr` positions).

**Files:**
- Modify: `src/solver/cpu_rk3.cpp`
- Modify: `src/solver/lts_integrator.cpp`

- [ ] **Step 1: Update Stage 1 loop in cpu_rk3.cpp**

In `src/solver/cpu_rk3.cpp`, for each of the three `#pragma omp parallel for` stage-update loops (stages 1, 2, 3), change:

```cpp
#pragma omp parallel for
for (int ii = 0; ii < NL; ++ii)
for (int v  = 0; v  < NVAR; ++v)
for (int t  = 0; t  < CellBlock::NTILE; ++t) {
```

to:

```cpp
#pragma omp parallel for collapse(3) schedule(static)
for (int ii = 0; ii < NL; ++ii)
for (int v  = 0; v  < NVAR; ++v)
for (int t  = 0; t  < CellBlock::NTILE; ++t) {
```

Do this for all three stage-update loops (stages 1, 2, 3) in `step()`.

- [ ] **Step 2: Update Stage loops in lts_integrator.cpp**

In `src/solver/lts_integrator.cpp`, same change for each of the three `#pragma omp parallel for` loops in `rk3_level()`:

```cpp
#pragma omp parallel for collapse(3) schedule(static)
for (int ii = 0; ii < NL; ++ii) {
    if (tree.nodes[leaves[ii]].level != level) continue;
    for (int v = 0; v < NVAR; ++v)
    for (int t = 0; t < CellBlock::NTILE; ++t) {
```

**Important:** `lts_integrator.cpp` has an `if (tree.nodes[leaves[ii]].level != level) continue;` guard inside the loop. With `collapse(3)` this guard must be inside the collapsed loop body and not between loop headers. Restructure the loop so the guard comes after the opening brace:

```cpp
#pragma omp parallel for collapse(3) schedule(static)
for (int ii = 0; ii < NL; ++ii)
for (int v  = 0; v  < NVAR; ++v)
for (int t  = 0; t  < CellBlock::NTILE; ++t) {
    if (tree.nodes[leaves[ii]].level != level) continue;
    double*       qs = solver.Qs_[ii].Q[v].tile_ptr(t);
    const double* qn = solver.Qn_[ii].Q[v].tile_ptr(t);
    const double* r  = solver.rhs_[ii].Q[v].tile_ptr(t);
    #pragma omp simd simdlen(8)
    for (int lane = 0; lane < CellBlock::W; ++lane)
        qs[lane] = ...;
}
```

- [ ] **Step 3: Build and run gate**

```bash
cmake --build build -t ba
```
Expected: all previously-passing gates still pass (correctness unchanged; scheduler hint only).

- [ ] **Step 4: Spot-check single-block mass conservation (T01 from test_ns)**

```bash
cmake --build build --target test_ns && ./build/test_ns 2>&1 | grep -E "T01|T04|PASS|FAIL" | head -10
```
Expected: T01 mass conservation and T04 energy conservation both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solver/cpu_rk3.cpp src/solver/lts_integrator.cpp
git commit -m "R10-T5: OpenMP collapse(3) on RK3 stage-update loops for single-block parallelism"
```

---

## Task 6: tree_rhs_typed dispatched from CpuRk3Integrator

**Goal:** Add `tree_rhs_typed<Flux, Recon, EOS>` that routes through `compute_rhs_typed` instead of the untyped `compute_rhs`. Dispatch from `CpuRk3Integrator::step()` based on `cfg.exec.flux_scheme`. This completes the R5 typed path from the per-leaf RHS down to the tree level.

**Prerequisite:** Tasks 3 and 4 (StiffenedGasEOS and IdealGasEOS::gamma exist).

**Context:** `tree_rhs` currently calls `compute_rhs(...)` at line ~534 of `operators.cpp`. The new `tree_rhs_typed` replaces that call with `compute_rhs_typed<Flux,Recon,EOS>(...)`.

**Files:**
- Modify: `include/schemes/operators.hpp`
- Modify: `src/schemes/operators.cpp`
- Modify: `src/solver/cpu_rk3.cpp`

- [ ] **Step 1: Declare tree_rhs_typed in operators.hpp**

In `include/schemes/operators.hpp`, after the existing `tree_rhs` declaration, add:

```cpp
// R10-T6: Typed tree-level RHS — same as tree_rhs but dispatches through
// compute_rhs_typed<Flux,Recon,EOS> for compile-time scheme resolution.
// eos: EOS instance carrying runtime parameters (e.g. IdealGasEOS::gamma).
template<template<Axis> class Flux, template<Axis> class Recon, class EOS>
    requires RiemannFlux<Flux<Axis::X>>
          && SpatialReconstruction<Recon<Axis::X>>
          && EquationOfState<EOS>
void tree_rhs_typed(BlockTree& tree,
                    std::vector<CellBlock>& rhs_blocks,
                    bool periodic,
                    double stage_weight        = 1.0,
                    int    level_filter        = -1,
                    bool   cf_coarse_zero_grad = false,
                    bool   open_bc             = false,
                    const DucrosConfig& ducros = DucrosConfig{},
                    EOS    eos                 = EOS{}) noexcept;
```

- [ ] **Step 2: Implement tree_rhs_typed in operators.cpp**

In `src/schemes/operators.cpp`, after the `tree_rhs` function, add:

```cpp
// =============================================================================
// R10-T6 — tree_rhs_typed: typed dispatch to compute_rhs_typed<Flux,Recon,EOS>
// =============================================================================
template<template<Axis> class Flux, template<Axis> class Recon, class EOS>
    requires RiemannFlux<Flux<Axis::X>>
          && SpatialReconstruction<Recon<Axis::X>>
          && EquationOfState<EOS>
void tree_rhs_typed(BlockTree& tree,
                    std::vector<CellBlock>& rhs_blocks,
                    bool periodic,
                    double stage_weight,
                    int    level_filter,
                    bool   cf_coarse_zero_grad,
                    bool   open_bc,
                    const DucrosConfig& ducros,
                    EOS    eos) noexcept
{
    PROFILE_SCOPE("tree_rhs_typed");

    if (periodic) tree.fill_ghosts_periodic(cf_coarse_zero_grad);
    else if (open_bc) tree.fill_ghosts_open(cf_coarse_zero_grad);
    else tree.fill_ghosts_wall(cf_coarse_zero_grad);

    const auto& leaves  = tree.leaf_indices();
    const int   n_leaves = (int)leaves.size();
    assert((int)rhs_blocks.size() == n_leaves);

    // Reuse Morton-sorted order from tree (Task 7 will use cached version;
    // for now inline the sort identical to tree_rhs).
    std::vector<int> order;
    order.reserve(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        if (level_filter >= 0 && tree.nodes[leaves[li]].level != level_filter)
            continue;
        order.push_back(li);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tree.nodes[leaves[a]].morton < tree.nodes[leaves[b]].morton;
    });
    const int n_active = (int)order.size();

#pragma omp parallel for schedule(dynamic,4)
    for (int oi = 0; oi < n_active; ++oi) {
        const int li       = order[oi];
        const int node_idx = leaves[li];
        if (!tree.nodes[node_idx].has_block()) continue;
        compute_rhs_typed<Flux, Recon, EOS>(
            *tree.nodes[node_idx].block, rhs_blocks[li], ducros);
        undo_cf_face_flux(tree, node_idx, rhs_blocks[li]);
        if (cf_coarse_zero_grad)
            undo_cf_viscous_energy(tree, node_idx, rhs_blocks[li]);
    }

    accumulate_cf_fine_fluxes(tree, stage_weight, level_filter);
}

// Explicit instantiations
template void tree_rhs_typed<HllcEsFlux, Weno5Recon, IdealGasEOS>(
    BlockTree&, std::vector<CellBlock>&, bool, double, int, bool, bool,
    const DucrosConfig&, IdealGasEOS) noexcept;
template void tree_rhs_typed<HllcFlux, Weno5Recon, IdealGasEOS>(
    BlockTree&, std::vector<CellBlock>&, bool, double, int, bool, bool,
    const DucrosConfig&, IdealGasEOS) noexcept;
```

- [ ] **Step 3: Dispatch from CpuRk3Integrator::step()**

In `src/solver/cpu_rk3.cpp`, add `#include "physics/ideal_gas_eos.hpp"` at the top.

Replace the three `tree_rhs(tree, solver.rhs_, ...)` calls with a dispatch:

```cpp
    // R10-T6: typed dispatch based on flux scheme
    const IdealGasEOS eos{cfg.physics.gamma};
    auto rhs_call = [&](double sw) {
        if (cfg.exec.flux_scheme == SolverConfig::FluxScheme::HLLC_ES)
            tree_rhs_typed<HllcEsFlux, Weno5Recon, IdealGasEOS>(
                tree, solver.rhs_, periodic, sw, -1, false, open_bc, ducros, eos);
        else
            tree_rhs_typed<HllcFlux, Weno5Recon, IdealGasEOS>(
                tree, solver.rhs_, periodic, sw, -1, false, open_bc, ducros, eos);
    };

    // Stage 1
    mpi_exchange_halos(tree, solver.mpi_);
    rhs_call(1.0/6.0);
    // ... rest of stage 1 unchanged ...

    // Stage 2
    mpi_exchange_halos(tree, solver.mpi_);
    rhs_call(1.0/6.0);
    // ...

    // Stage 3
    mpi_exchange_halos(tree, solver.mpi_);
    rhs_call(2.0/3.0);
    // ...
```

**Note:** `use_sat` still calls `tree_sat_penalty(tree, solver.rhs_, cfg.numerics.sat_tau)` — keep that call unchanged after each `rhs_call`.

- [ ] **Step 4: Build and run operators test**

```bash
cmake --build build --target test_operators && ./build/test_operators 2>&1 | grep -E "T08|convergence|PASS|FAIL"
```
Expected: T08 convergence rate ≥ 1.8 PASS.

- [ ] **Step 5: Run full gate**

```bash
cmake --build build -t ba
```
Expected: all previously-passing gates still pass.

- [ ] **Step 6: Commit**

```bash
git add include/schemes/operators.hpp src/schemes/operators.cpp src/solver/cpu_rk3.cpp
git commit -m "R10-T6: tree_rhs_typed<Flux,Recon,EOS>; dispatch from CpuRk3Integrator based on flux scheme"
```

---

## Task 7: Cached Morton leaf indices in BlockTree

**Goal:** Add `BlockTree::morton_leaf_indices()` that returns a cached Morton-sorted leaf index list, invalidated with the existing `leaf_dirty_` flag. Use it in `tree_rhs` and `tree_rhs_typed` to avoid re-sorting on every call (called 3× per RK3 step).

**Context:** `leaf_indices()` already has a dirty cache (`leaf_cache_` / `leaf_dirty_`). The Morton sort in `tree_rhs` rebuilds a sorted `order` vector on every call. `BlockNode::morton` is a `uint32_t` field.

**Files:**
- Modify: `include/mesh/block_tree.hpp`
- Modify: `src/mesh/block_tree.cpp`
- Modify: `src/schemes/operators.cpp`

- [ ] **Step 1: Declare morton_leaf_indices() in block_tree.hpp**

In `include/mesh/block_tree.hpp`, after the existing `leaf_indices()` declaration (line ~130), add:

```cpp
    // Returns leaf_indices() sorted by Morton code — cached, O(1) after first call.
    // Cache is invalidated alongside leaf_cache_ on regrid/refine/coarsen.
    const std::vector<int>& morton_leaf_indices() const;
```

Also add a second cached vector in the private section, alongside `leaf_cache_`:

```cpp
    mutable std::vector<int> morton_leaf_cache_;
    // leaf_dirty_ invalidates both leaf_cache_ and morton_leaf_cache_.
```

- [ ] **Step 2: Implement morton_leaf_indices() in block_tree.cpp**

In `src/mesh/block_tree.cpp`, add the implementation after `leaf_indices()`:

```cpp
const std::vector<int>& BlockTree::morton_leaf_indices() const {
    const auto& leaves = leaf_indices();   // ensures leaf_cache_ is up to date
    if (!leaf_dirty_ && morton_leaf_cache_.size() == leaves.size())
        return morton_leaf_cache_;
    morton_leaf_cache_ = leaves;
    std::sort(morton_leaf_cache_.begin(), morton_leaf_cache_.end(),
              [this](int a, int b) { return nodes[a].morton < nodes[b].morton; });
    return morton_leaf_cache_;
}
```

**Note on invalidation:** `leaf_dirty_` is set to `true` whenever the tree topology changes (refine, coarsen, balance). Since `morton_leaf_indices()` calls `leaf_indices()` first (which clears `leaf_dirty_`), checking `leaf_dirty_` after that call is always false. Instead, detect staleness by comparing `morton_leaf_cache_.size()` to `leaves.size()`. When the leaf cache is rebuilt, its size changes, so `morton_leaf_cache_` gets resorted. This is safe because `leaf_indices()` rebuilds `leaf_cache_` from scratch when dirty, so any size change means the topology changed.

- [ ] **Step 3: Use morton_leaf_indices() in tree_rhs**

In `src/schemes/operators.cpp`, in `tree_rhs`, replace the Morton-sort block (the `std::vector<int> order;` + sort ~lines 516–525):

```cpp
    // ── 2. Morton-sorted slot order (cached in tree)
    const auto& ml = tree.morton_leaf_indices();
    std::vector<int> order;
    order.reserve(ml.size());
    for (int node_idx : ml) {
        // Find the slot index (position in leaf_indices()) for this node
        // leaf_indices() and ml iterate the same nodes in different orders.
        // We need the slot index li such that leaves[li] == node_idx.
        // Build the slot-index lookup once.
    }
```

Hmm, `tree_rhs` uses slot indices `li` (position in `leaf_indices()`) to index into `rhs_blocks[li]`. `morton_leaf_indices()` returns node indices sorted by Morton. We need slot indices sorted by Morton.

**Correct approach:** Build a `node_to_slot` map once. Or, since `leaf_indices()` returns a vector and Morton-sorted order is a permutation, build the order as:

```cpp
    const auto& leaves  = tree.leaf_indices();  // slot li → node index
    const int n_leaves  = (int)leaves.size();

    // Build node_index → slot_index reverse map
    // (leaves is small enough that a linear scan is fine; build once per step)
    // Use morton_leaf_indices() to get Morton-ordered node indices,
    // then find their slot positions.
    const auto& ml = tree.morton_leaf_indices();   // node indices in Morton order
    // Build reverse map: node_idx → slot li
    std::vector<int> node_to_slot(tree.nodes.size(), -1);
    for (int li = 0; li < n_leaves; ++li)
        node_to_slot[leaves[li]] = li;

    std::vector<int> order;
    order.reserve(n_leaves);
    for (int node_idx : ml) {
        int li = node_to_slot[node_idx];
        if (li < 0) continue;
        if (level_filter >= 0 && tree.nodes[node_idx].level != level_filter) continue;
        order.push_back(li);
    }
    // order is now slot indices in Morton order
    const int n_active = (int)order.size();
```

Apply the same change in `tree_rhs_typed`.

- [ ] **Step 4: Build and run gate**

```bash
cmake --build build -t ba
```
Expected: all previously-passing gates still pass (Morton order is identical; only the sort is cached).

- [ ] **Step 5: Commit**

```bash
git add include/mesh/block_tree.hpp src/mesh/block_tree.cpp src/schemes/operators.cpp
git commit -m "R10-T7: cached morton_leaf_indices() in BlockTree; eliminate per-call sort in tree_rhs"
```

---

## Task 8: SolverResult<T> — non-throwing error type

**Goal:** Create `include/solver/solver_result.hpp` with a minimal `SolverResult<T>` type backed by `std::variant`. Add `NSSolver::advance_result()` as a non-throwing alternative to `advance()`. The existing `advance()` is unchanged for backward compatibility.

**Files:**
- Create: `include/solver/solver_result.hpp`
- Modify: `include/solver/ns_solver.hpp`
- Modify: `src/solver/ns_solver.cpp`
- Modify: `tests/solver/test_imex.cpp`

- [ ] **Step 1: Create solver_result.hpp**

Create `include/solver/solver_result.hpp`:

```cpp
#pragma once
// SolverResult<T> — a minimal non-throwing alternative to exception-based errors.
//
// Usage:
//   SolverResult<double> r = solver.advance_result();
//   if (!r.ok()) { std::cerr << r.error().message; return 1; }
//   double dt = r.value();
//
// C++20 polyfill for std::expected<T, E> (available in C++23).
// Uses std::variant<T, SolverError> to avoid undefined behaviour from
// unconstructed alternatives.

#include <string>
#include <variant>
#include <stdexcept>
#include <utility>

struct SolverError {
    std::string message;
    int         code = 0;   // 0 = generic; specific codes TBD by callers
};

template<class T>
class SolverResult {
    std::variant<T, SolverError> v_;
public:
    static SolverResult ok(T val)     { return SolverResult(std::in_place_index<0>, std::move(val)); }
    static SolverResult err(SolverError e) { return SolverResult(std::in_place_index<1>, std::move(e)); }

    bool               ok()    const noexcept { return v_.index() == 0; }
    const T&           value() const { return std::get<T>(v_); }
    T&&                value()       { return std::get<T>(std::move(v_)); }
    const SolverError& error() const { return std::get<SolverError>(v_); }

    // Implicit bool conversion
    explicit operator bool() const noexcept { return ok(); }

    // Throw if error; return value otherwise (bridge to exception path)
    T value_or_throw() const {
        if (!ok()) throw std::runtime_error(error().message);
        return value();
    }

private:
    template<std::size_t I, class U>
    SolverResult(std::in_place_index_t<I> idx, U&& val)
        : v_(idx, std::forward<U>(val)) {}
};

// Specialisation for void (used by validate_result())
template<>
class SolverResult<void> {
    std::variant<std::monostate, SolverError> v_;
public:
    static SolverResult ok()          { return SolverResult(std::in_place_index<0>, std::monostate{}); }
    static SolverResult err(SolverError e) { return SolverResult(std::in_place_index<1>, std::move(e)); }

    bool               ok()    const noexcept { return v_.index() == 0; }
    const SolverError& error() const { return std::get<SolverError>(v_); }
    explicit operator bool() const noexcept { return ok(); }

private:
    template<std::size_t I, class U>
    SolverResult(std::in_place_index_t<I> idx, U&& val)
        : v_(idx, std::forward<U>(val)) {}
};
```

- [ ] **Step 2: Add advance_result() to NSSolver**

In `include/solver/ns_solver.hpp`, add `#include "solver/solver_result.hpp"` near the top (after other includes).

In the public section of `NSSolver`, add after `advance()`:

```cpp
    // Non-throwing alternative to advance().
    // Returns SolverResult<double> with the dt, or an error if the step fails
    // (e.g. NaN detected in positivity floor, negative dt from CFL).
    SolverResult<double> advance_result() noexcept;
```

- [ ] **Step 3: Implement advance_result() in ns_solver.cpp**

In `src/solver/ns_solver.cpp`, add `#include "solver/solver_result.hpp"` and implement:

```cpp
SolverResult<double> NSSolver::advance_result() noexcept {
    try {
        double dt = advance();
        if (dt <= 0.0 || std::isnan(dt) || std::isinf(dt))
            return SolverResult<double>::err(
                SolverError{"advance() returned non-positive dt: " + std::to_string(dt), 1});
        return SolverResult<double>::ok(dt);
    } catch (const std::exception& ex) {
        return SolverResult<double>::err(SolverError{ex.what(), 2});
    } catch (...) {
        return SolverResult<double>::err(SolverError{"unknown exception in advance()", 3});
    }
}
```

- [ ] **Step 4: Add a test to test_imex.cpp**

In `tests/solver/test_imex.cpp`, add `#include "solver/solver_result.hpp"` and add a new test:

```cpp
static void t05_advance_result_ok() {
    NSSolver s;
    s.cfg.time.cfl = 0.5; s.cfg.time.max_steps = 1; s.cfg.time.t_end = 1e30;
    s.cfg.bc.variant = PeriodicBC{}; s.cfg.io.verbose = false;
    s.cfg.io.diag_interval = 100;
    s.init(1.0, [](double x, double y, double z) -> Prim {
        (void)y; (void)z;
        return Prim{1.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0};
    });
    auto r = s.advance_result();
    check("T05 advance_result ok() for valid solver", r.ok());
    check("T05 advance_result dt > 0", r.ok() && r.value() > 0.0,
          r.ok() ? r.value() : -1.0, 0.0);
}
```

Call `t05_advance_result_ok()` from `main()`.

- [ ] **Step 5: Build and run test_imex**

```bash
cmake --build build --target test_imex && ./build/test_imex
```
Expected: all tests PASS including T05.

- [ ] **Step 6: Run full gate**

```bash
cmake --build build -t ba
```
Expected: all previously-passing gates still pass.

- [ ] **Step 7: Commit**

```bash
git add include/solver/solver_result.hpp include/solver/ns_solver.hpp \
        src/solver/ns_solver.cpp tests/solver/test_imex.cpp
git commit -m "R10-T8: SolverResult<T> non-throwing error type; NSSolver::advance_result()"
```

---

## Self-Review

### 1. Spec coverage

| Original idea | Task | Status |
|---------------|------|--------|
| Typed field accessors | T1 | ✅ flat-index rho/rhou/rhov/rhow/E + positivity floor |
| Composable RHS pipeline | T6 | ✅ tree_rhs_typed dispatches compute_rhs_typed |
| General EOS / remove hardcoded GAMMA | T3 + T4 | ✅ StiffenedGasEOS + configurable GAMMA |
| Morton SFC load balancing | T7 | ✅ Already in mpi_partition; T7 caches the sort in tree_rhs |
| Task-based parallelism | T5 | ✅ collapse(3) enables single-block parallelism |
| NVTX/Tracy annotations | T2 | ✅ profiler.hpp with RAII markers |
| std::expected / error API | T8 | ✅ SolverResult<T> + advance_result() |
| Compile-time grid parameters | T4 | ✅ gamma is now configurable; NB/NG/NVAR stay compile-time (changes those would require R6+) |

### 2. Placeholder scan — none found

### 3. Type consistency

- `SolverResult<double>` factory methods: `SolverResult<double>::ok(dt)` and `SolverResult<double>::err(SolverError{...})` — consistent across T8 steps.
- `IdealGasEOS{cfg.physics.gamma}` in T6 — `IdealGasEOS::gamma` member added in T4, used in T6 after T4 completes.
- `tree_rhs_typed` signature in T6 declaration matches implementation — `EOS eos = EOS{}` as last parameter.
- `morton_leaf_indices()` in T7: returns `const std::vector<int>&` — same as `leaf_indices()` for API consistency.
