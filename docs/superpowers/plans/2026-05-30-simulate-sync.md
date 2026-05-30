# simulate + template.json Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `apps/simulate.cpp`, `apps/simulate_gpu.cu`, and `apps/template.json` in sync with the evolved codebase by adding config keys for scheme selection, ACDI, NSCBC, BN two-phase model, combustion, radiation, WMLES, and GPU dispatch.

**Architecture:** Extend the existing flat-JSON `struct Config` parser pattern in both simulate binaries; add a `ReconScheme` enum to `SolverConfig::ExecConfig` and physics params to `SolverConfig::PhysicsConfig`; add `build_bn_ic()` factory alongside the existing `build_ic()`; add a BN dispatch path in simulate.cpp that calls `BNSolver` instead of `NSSolver`.

**Tech Stack:** C++20, CUDA (simulate_gpu.cu), nlohmann-free flat-JSON parser, CMake, ctest

---

## File map

| File | Role |
|---|---|
| `include/solver/ns_solver.hpp` | Add `ReconScheme` enum + combustion/radiation/WMLES fields |
| `include/schemes/operators.hpp` | Add `extern template` suppressors for `Teno5Recon` specialisations |
| `src/solver/cpu_rk3.cpp` | Extend `select_scheme()` with TENO5A branch |
| `apps/simulate.cpp` | Add grouped config sections; extend `parse_bc_str`; add BN + GPU paths |
| `apps/simulate_gpu.cu` | Add same grouped config sections; wire ACDI + Dynamic SGS |
| `apps/initial_conditions.hpp` | Add `build_bn_ic()` + `reactive_blast` to `build_ic()` |
| `apps/template.json` | Add all new flat config keys with comment-block grouping |

---

### Task 1: SolverConfig additions

**Files:**
- Modify: `include/solver/ns_solver.hpp`

- [ ] **Step 1: Add includes** — insert three new `#include` lines after the existing includes at the top of `ns_solver.hpp` (after `#include <string>`):

```cpp
#include "physics/arrhenius.hpp"
#include "physics/p1_radiation.hpp"
#include "models/wall_model.hpp"
```

- [ ] **Step 2: Add `ReconScheme` enum to `SolverConfig`** — insert after the existing `enum class FluxScheme` declaration (around line 111, just before `struct ExecConfig {`):

```cpp
    enum class ReconScheme { WENO5Z, TENO5A };
```

- [ ] **Step 3: Add `recon` field to `ExecConfig`** — inside `struct ExecConfig { ... }`, add after `bool use_gpu = false;`:

```cpp
        ReconScheme recon = ReconScheme::WENO5Z;
```

The full updated `ExecConfig` block becomes:
```cpp
    struct ExecConfig {
        ExecutionBackend backend     = ExecutionBackend::CPU;
        FluxScheme       flux_scheme = FluxScheme::HLLC_ES;
        bool             use_gpu     = false;
        ReconScheme      recon       = ReconScheme::WENO5Z;
    } exec;
```

- [ ] **Step 4: Add combustion/radiation/WMLES fields to `PhysicsConfig`** — append after `double gamma = GAMMA;` inside `struct PhysicsConfig`:

```cpp
        // D5 Arrhenius combustion (GPU path; CPU path ignores with warning)
        bool            combustion_enabled = false;
        ArrheniusParams arrhenius{};

        // D6 P1 radiation (GPU path; CPU path ignores with warning)
        bool            radiation_enabled = false;
        RadiationParams rad_params{};

        // D7 WMLES (GPU path; CPU path ignores with warning)
        bool            wmles_enabled = false;
        WallModelCfg    wall_model{};
```

- [ ] **Step 5: Build check**

```bash
cmake --build build --target ns_solver 2>&1 | tail -5
```

Expected: zero errors and zero new warnings. Fix any include-order conflicts before continuing.

- [ ] **Step 6: Commit**

```bash
git add include/solver/ns_solver.hpp
git commit -m "cfg: add ReconScheme + combustion/radiation/wmles fields to SolverConfig"
```

---

### Task 2: Teno5Recon extern templates + cpu_rk3 scheme dispatch

**Files:**
- Modify: `include/schemes/operators.hpp`
- Modify: `src/solver/cpu_rk3.cpp`

- [ ] **Step 1: Add Teno5Recon extern templates in `operators.hpp`** — append the block below immediately after the last existing `extern template void compute_rhs_typed<HllcFlux, Weno5Recon, StiffenedGasEOS>(...)` line (around line 229):

```cpp
#include "physics/teno5_recon.hpp"
extern template void tree_rhs_typed<HllcEsFlux, Teno5Recon, IdealGasEOS>(
    BlockTree&, std::vector<CellBlock>&, const BCVariant&, double, int, bool,
    const DucrosConfig&, IdealGasEOS) noexcept;
extern template void tree_rhs_typed<HllcFlux, Teno5Recon, IdealGasEOS>(
    BlockTree&, std::vector<CellBlock>&, const BCVariant&, double, int, bool,
    const DucrosConfig&, IdealGasEOS) noexcept;
extern template void tree_rhs_typed<HllcEsFlux, Teno5Recon, StiffenedGasEOS>(
    BlockTree&, std::vector<CellBlock>&, const BCVariant&, double, int, bool,
    const DucrosConfig&, StiffenedGasEOS) noexcept;
extern template void tree_rhs_typed<HllcFlux, Teno5Recon, StiffenedGasEOS>(
    BlockTree&, std::vector<CellBlock>&, const BCVariant&, double, int, bool,
    const DucrosConfig&, StiffenedGasEOS) noexcept;
extern template void compute_rhs_typed<HllcEsFlux, Teno5Recon, IdealGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&, uint8_t) noexcept;
extern template void compute_rhs_typed<HllcFlux, Teno5Recon, IdealGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&, uint8_t) noexcept;
extern template void compute_rhs_typed<HllcEsFlux, Teno5Recon, StiffenedGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&, uint8_t) noexcept;
extern template void compute_rhs_typed<HllcFlux, Teno5Recon, StiffenedGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&, uint8_t) noexcept;
```

- [ ] **Step 2: Add `#include` for Teno5Recon in `cpu_rk3.cpp`** — append after the existing `#include "physics/weno5_recon.hpp"` line (line 7):

```cpp
#include "physics/teno5_recon.hpp"
```

- [ ] **Step 3: Extend `select_scheme()` in `cpu_rk3.cpp`** — find the function (starting at line 39). Replace the entire `select_scheme()` body with a version that dispatches on `cfg.exec.recon`. The existing logic is preserved as the WENO5Z path; a parallel TENO5A block is added:

```cpp
void CpuRk3Integrator::select_scheme() {
    const SolverConfig& cfg = solver.cfg;
    const bool sg = cfg.acdi.use_acdi &&
                    (cfg.acdi.gamma_a != cfg.acdi.gamma_b ||
                     cfg.acdi.p_inf_a != 0.0 || cfg.acdi.p_inf_b != 0.0);
    const bool es    = (cfg.exec.flux_scheme == SolverConfig::FluxScheme::HLLC_ES);
    const bool teno5 = (cfg.exec.recon       == SolverConfig::ReconScheme::TENO5A);

    if (teno5) {
        if (es && !sg) {
            const IdealGasEOS eos{cfg.physics.gamma};
            rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                            const BCVariant& bc, double sw, int lf, bool cz,
                            const DucrosConfig& d) noexcept {
                tree_rhs_typed<HllcEsFlux, Teno5Recon, IdealGasEOS>(t, r, bc, sw, lf, cz, d, eos);
            };
        } else if (!es && !sg) {
            const IdealGasEOS eos{cfg.physics.gamma};
            rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                            const BCVariant& bc, double sw, int lf, bool cz,
                            const DucrosConfig& d) noexcept {
                tree_rhs_typed<HllcFlux, Teno5Recon, IdealGasEOS>(t, r, bc, sw, lf, cz, d, eos);
            };
        } else if (es) {
            const StiffenedGasEOS eos{cfg.acdi.gamma_a, cfg.acdi.gamma_b,
                                       cfg.acdi.p_inf_a, cfg.acdi.p_inf_b};
            rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                            const BCVariant& bc, double sw, int lf, bool cz,
                            const DucrosConfig& d) noexcept {
                tree_rhs_typed<HllcEsFlux, Teno5Recon, StiffenedGasEOS>(t, r, bc, sw, lf, cz, d, eos);
            };
        } else {
            const StiffenedGasEOS eos{cfg.acdi.gamma_a, cfg.acdi.gamma_b,
                                       cfg.acdi.p_inf_a, cfg.acdi.p_inf_b};
            rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                            const BCVariant& bc, double sw, int lf, bool cz,
                            const DucrosConfig& d) noexcept {
                tree_rhs_typed<HllcFlux, Teno5Recon, StiffenedGasEOS>(t, r, bc, sw, lf, cz, d, eos);
            };
        }
        return;
    }

    // WENO5Z (default) — unchanged from original
    if (es && !sg) {
        const IdealGasEOS eos{cfg.physics.gamma};
        rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                        const BCVariant& bc, double sw, int lf, bool cz,
                        const DucrosConfig& d) noexcept {
            tree_rhs_typed<HllcEsFlux, Weno5Recon, IdealGasEOS>(t, r, bc, sw, lf, cz, d, eos);
        };
    } else if (!es && !sg) {
        const IdealGasEOS eos{cfg.physics.gamma};
        rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                        const BCVariant& bc, double sw, int lf, bool cz,
                        const DucrosConfig& d) noexcept {
            tree_rhs_typed<HllcFlux, Weno5Recon, IdealGasEOS>(t, r, bc, sw, lf, cz, d, eos);
        };
    } else if (es) {
        const StiffenedGasEOS eos{cfg.acdi.gamma_a, cfg.acdi.gamma_b,
                                   cfg.acdi.p_inf_a, cfg.acdi.p_inf_b};
        rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                        const BCVariant& bc, double sw, int lf, bool cz,
                        const DucrosConfig& d) noexcept {
            tree_rhs_typed<HllcEsFlux, Weno5Recon, StiffenedGasEOS>(t, r, bc, sw, lf, cz, d, eos);
        };
    } else {
        const StiffenedGasEOS eos{cfg.acdi.gamma_a, cfg.acdi.gamma_b,
                                   cfg.acdi.p_inf_a, cfg.acdi.p_inf_b};
        rhs_fn_ = [eos](BlockTree& t, std::vector<CellBlock>& r,
                        const BCVariant& bc, double sw, int lf, bool cz,
                        const DucrosConfig& d) noexcept {
            tree_rhs_typed<HllcFlux, Weno5Recon, StiffenedGasEOS>(t, r, bc, sw, lf, cz, d, eos);
        };
    }
}
```

- [ ] **Step 4: Run the full CPU gate suite**

```bash
cmake --build build -t ba 2>&1 | tail -20
```

Expected: `Results: ... passed, 0 failed` for all four layers. The new code is branch-free in the hot loop so it must not regress any existing test.

- [ ] **Step 5: Commit**

```bash
git add include/schemes/operators.hpp src/solver/cpu_rk3.cpp
git commit -m "scheme: add Teno5Recon dispatch in SolverConfig + cpu_rk3 select_scheme()"
```

---

### Task 3: BN IC factory in `initial_conditions.hpp`

**Files:**
- Modify: `apps/initial_conditions.hpp`

This task adds `build_bn_ic()` with two IC cases (`bn_sod_x`, `bn_uniform`) so the BN dispatch in simulate.cpp (Task 5) can call it.

- [ ] **Step 1: Add the required include** — insert at the top of `initial_conditions.hpp`, after the existing `#include` lines:

```cpp
#include "models/bn_model.hpp"   // BNCellBlock, BNEosParams, cell_idx, NB2, NG
```

- [ ] **Step 2: Add `build_bn_ic()` function** — append after the closing `}` of `build_ic()`:

```cpp
// IC factory for BNSolver — returns a void(BNCellBlock&, ox, oy, oz, h) lambda.
// Supported names: "bn_sod_x", "bn_uniform".
inline std::function<void(BNCellBlock&, double, double, double, double)>
build_bn_ic(const Config& cfg, const BNEosParams& eos)
{
    std::string name = cfg.str("ic", "bn_sod_x");
    const double L   = cfg.d("domain_L", 1.0);

    if (name == "bn_uniform") {
        const double a1 = cfg.d("ic_bn_alpha1_l", 0.5);
        const double a2 = 1.0 - a1;
        const double p  = cfg.d("ic_bn_p_l",      1.0);
        return [=](BNCellBlock& blk, double /*ox*/, double /*oy*/, double /*oz*/, double /*h*/) {
            const double e = a1 * (p + eos.gamma1 * eos.pinf1) / (eos.gamma1 - 1.0)
                           + a2 * (p + eos.gamma2 * eos.pinf2) / (eos.gamma2 - 1.0);
            for (int k = 0; k < NB2; ++k)
            for (int j = 0; j < NB2; ++j)
            for (int i = 0; i < NB2; ++i) {
                const int f = cell_idx(i, j, k);
                blk.Q[0][f] = a1; blk.Q[1][f] = a2;
                blk.Q[2][f] = 0.0; blk.Q[3][f] = 0.0; blk.Q[4][f] = 0.0;
                blk.Q[5][f] = e;  blk.Q[6][f] = a1;
            }
        };
    }

    // Default: bn_sod_x
    const double a1_l = cfg.d("ic_bn_alpha1_l", 0.9);
    const double a1_r = cfg.d("ic_bn_alpha1_r", 0.1);
    const double p_l  = cfg.d("ic_bn_p_l",      1.0);
    const double p_r  = cfg.d("ic_bn_p_r",      0.1);
    return [=](BNCellBlock& blk, double ox, double /*oy*/, double /*oz*/, double h) {
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            const double x = ox + (i - NG + 0.5) * h;
            const bool  lf = (x < 0.5 * L);
            const double a1 = lf ? a1_l : a1_r;
            const double a2 = 1.0 - a1;
            const double p  = lf ? p_l  : p_r;
            const double e  = a1 * (p + eos.gamma1 * eos.pinf1) / (eos.gamma1 - 1.0)
                            + a2 * (p + eos.gamma2 * eos.pinf2) / (eos.gamma2 - 1.0);
            const int f = cell_idx(i, j, k);
            blk.Q[0][f] = a1; blk.Q[1][f] = a2;
            blk.Q[2][f] = 0.0; blk.Q[3][f] = 0.0; blk.Q[4][f] = 0.0;
            blk.Q[5][f] = e;  blk.Q[6][f] = a1;
        }
    };
}
```

- [ ] **Step 3: Add `reactive_blast` case to `build_ic()`** — add the following block before the `fprintf(stderr, "simulate: unknown ic...")` line at the bottom of `build_ic()`:

```cpp
    if (name == "reactive_blast") {
        const double L         = cfg.d("domain_L",       1.0);   // each branch owns its L
        const double blast_r   = cfg.d("ic_blast_r",     0.1);
        const double T_hot     = cfg.d("ic_blast_T_hot", 4.0);
        const double rho0      = 1.0;
        const double xc = 0.5*L, yc = 0.5*L, zc = 0.5*L;
        return [=](double x, double y, double z) -> Prim {
            const double r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc)+(z-zc)*(z-zc));
            const bool   hot = (r < blast_r * L);
            Prim q{};
            q.rho = rho0;
            q.u = q.v = q.w = 0.0;
            q.T   = hot ? T_hot : 1.0;
            q.p   = q.rho * R_GAS * q.T;
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }
```

- [ ] **Step 4: Build check**

```bash
cmake --build build --target simulate 2>&1 | tail -5
```

Expected: zero errors. The `simulate` binary links `initial_conditions.hpp` implicitly.

- [ ] **Step 5: Commit**

```bash
git add apps/initial_conditions.hpp
git commit -m "ic: add build_bn_ic() (bn_sod_x, bn_uniform) + reactive_blast to build_ic()"
```

---

### Task 4: Add grouped config sections to `simulate.cpp`

**Files:**
- Modify: `apps/simulate.cpp`

This task adds the new config keys (scheme, ACDI, combustion, radiation, WMLES, NSCBC, mu, model, gpu) to the CPU simulation runner. The BN dispatch path is added in Task 5. For now, `model` and `gpu` are parsed but only used for informational prints/warnings.

- [ ] **Step 1: Add new headers at the top of `simulate.cpp`** — after the `#include "models/sgs.hpp"` line:

```cpp
#include "models/bn_model.hpp"      // BNEosParams (parsed even for ns path)
#include "models/bn_solver.hpp"     // BNSolver (used in Task 5 BN dispatch)
```

- [ ] **Step 2: Add `model` + `gpu` parse after the line `sc.physics.use_imex = cfg.b("use_imex", false);`** (around line 215):

```cpp
    // === Model selection ===
    const std::string model   = cfg.str("model", "ns");
    const bool        use_gpu = cfg.b("gpu", false);
    if (use_gpu)
        fprintf(stderr, "[INFO] simulate: gpu=true — run simulate_gpu for the GPU path\n");
```

- [ ] **Step 3: Extend `parse_bc_str` lambda to support NSCBC** — find the existing lambda definition (~line 218):

```cpp
        auto parse_bc_str = [](const std::string& s) -> BCVariant {
            if (s == "Wall")  return WallBC{};
            if (s == "Open")  return OpenBC{};
            return PeriodicBC{};
        };
```

Replace with a version that captures `cfg` by reference and handles `"NSCBC"`:

```cpp
        auto parse_bc_str = [&](const std::string& s) -> BCVariant {
            if (s == "Wall")  return WallBC{};
            if (s == "Open")  return OpenBC{};
            if (s == "NSCBC") return NscbcBC{ cfg.d("nscbc_p_inf", 1.0) };
            return PeriodicBC{};
        };
```

- [ ] **Step 4: Add grouped config sections after the SGS block** — insert the following block after the closing `}` of the SGS block (after `sc.physics.sgs = nullptr;` / the closing `}`):

```cpp
    // === Scheme selection ===
    {
        const std::string sch = cfg.str("scheme", "weno5z");
        if (sch == "teno5a")
            sc.exec.recon = SolverConfig::ReconScheme::TENO5A;
        else if (sch == "teno7a") {
            fprintf(stderr, "[WARN] simulate: scheme=teno7a not yet instantiated — falling back to teno5a\n");
            sc.exec.recon = SolverConfig::ReconScheme::TENO5A;
        }
        // else WENO5Z default
    }

    // === Viscosity ===
    // mu and sutherland are stored for documentation; CPU path relies on SGS/Sutherland model.
    {
        const double mu         = cfg.d("mu",         0.0);
        const bool   sutherland = cfg.b("sutherland", false);
        (void)mu; (void)sutherland;  // prevent unused-variable warnings
    }

    // === ACDI phase field ===
    sc.acdi.use_acdi  = cfg.b("acdi",         false);
    sc.acdi.acdi_ceps = cfg.d("acdi_ceps",    0.0);
    sc.acdi.gamma_a   = cfg.d("acdi_gamma_a", GAMMA);
    sc.acdi.gamma_b   = cfg.d("acdi_gamma_b", GAMMA);
    sc.acdi.p_inf_a   = cfg.d("acdi_pinf_a",  0.0);
    sc.acdi.p_inf_b   = cfg.d("acdi_pinf_b",  0.0);

    // === Combustion / Arrhenius ===
    sc.physics.combustion_enabled = cfg.b("combustion",      false);
    sc.physics.arrhenius.A        = cfg.d("combustion_A",    1e4);
    sc.physics.arrhenius.T_act    = cfg.d("combustion_Tact", 10.0);
    sc.physics.arrhenius.q_heat   = cfg.d("combustion_Q",    10.0);
    sc.physics.arrhenius.n_sub    = cfg.i("combustion_nsub", 8);
    if (sc.physics.combustion_enabled)
        fprintf(stderr, "[WARN] simulate: combustion=true — use simulate_gpu (CPU path ignores combustion)\n");

    // === Radiation / P1 ===
    sc.physics.radiation_enabled  = cfg.b("radiation",       false);
    sc.physics.rad_params.kappa   = cfg.d("radiation_kappa", 1.0);
    sc.physics.rad_params.a_rad   = cfg.d("radiation_arad",  1.0);
    if (sc.physics.radiation_enabled)
        fprintf(stderr, "[WARN] simulate: radiation=true — use simulate_gpu (CPU path ignores radiation)\n");

    // === WMLES ===
    sc.physics.wmles_enabled        = cfg.b("wmles",         false);
    sc.physics.wall_model.use_ode   = (cfg.str("wmles_model","reichardt") == "ode");
    if (sc.physics.wmles_enabled)
        fprintf(stderr, "[WARN] simulate: wmles=true — use simulate_gpu (CPU path ignores wmles)\n");

    // === BN EOS (read regardless; used in BN dispatch below) ===
    const BNEosParams bn_eos{
        cfg.d("bn_gamma1", 1.4),   // matches BNEosParams::gamma1 default
        cfg.d("bn_gamma2", 4.4),   // matches BNEosParams::gamma2 default (stiffened-gas water)
        cfg.d("bn_pinf1",  0.0),
        cfg.d("bn_pinf2",  6e8)    // matches BNEosParams::pinf2 default
    };
    const double bn_cfl = cfg.d("bn_cfl", 0.4);
    (void)bn_cfl;   // used in Task 5; silence warning for now
```

- [ ] **Step 5: Build and smoke-test**

```bash
cmake --build build --target simulate 2>&1 | tail -5
./build/simulate apps/template.json 2>&1 | head -20
```

Expected: binary builds; running with template.json prints config and starts (may exit quickly if t_end is short). Any `[WARN]` messages appear only when the relevant keys are set to non-default values.

- [ ] **Step 6: Run CPU gate suite to ensure no regression**

```bash
cmake --build build -t ba 2>&1 | tail -5
```

Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add apps/simulate.cpp
git commit -m "simulate: add scheme/ACDI/combustion/radiation/wmles/NSCBC/BN config sections"
```

---

### Task 5: BN dispatch path in `simulate.cpp`

**Files:**
- Modify: `apps/simulate.cpp`

The BN path uses `BNSolver` instead of `NSSolver`. It runs entirely within its own `if (model == "bn") { ... }` block inserted before the existing `NSSolver solver;` line. The existing NSSolver block becomes the `else` branch.

- [ ] **Step 1: Wrap the existing solver path in `else { ... }`** — find `NSSolver solver;` (around line 202). Insert before it:

```cpp
    if (model == "bn") {
        // ── BN two-phase path ──────────────────────────────────────────────────
        auto bn_ic = build_bn_ic(cfg, bn_eos);
        BNSolver bn_solver;
        bn_solver.bc = [&]() -> BCVariant {
            const std::string s = cfg.str("bc", "Periodic");
            if (s == "Wall")  return WallBC{};
            if (s == "Open")  return OpenBC{};
            if (s == "NSCBC") return NscbcBC{ cfg.d("nscbc_p_inf", 1.0) };
            return PeriodicBC{};
        }();
        bn_solver.init(domain_L, bn_ic, bn_eos, bn_solver.bc);
        if (mpi_rank == 0)
            printf("simulate: BN solver  t_end=%.4g  max_steps=%d  bn_cfl=%.3g\n",
                   cfg.d("t_end", 1.0), cfg.i("max_steps", 1000000), bn_cfl);
        bn_solver.run(cfg.d("t_end", 1.0), cfg.i("max_steps", 1000000));
        if (mpi_rank == 0) {
            printf("simulate: BN done  t=%.6e  step=%d\n", bn_solver.t, bn_solver.step);
        }
        return 0;
    } else {
```

Then add a matching closing `}` just before `return 0;` at the end of `main()`:

```cpp
    } // end else (model != "bn")
```

Note: `domain_L` is declared BEFORE `NSSolver solver;` in the existing code (`double domain_L = cfg.d("domain_L", 1.0);`) so it is in scope. `mpi_rank` and `bn_cfl` are also already declared. `bn_eos` and `bn_cfl` were declared in Task 4.

- [ ] **Step 2: Build and check BN path compiles**

```bash
cmake --build build --target simulate 2>&1 | tail -5
```

Expected: zero errors.

- [ ] **Step 3: Smoke-test the BN path** — create a minimal BN config and run it:

```bash
cat > /tmp/test_bn.json << 'EOF'
{
    "model": "bn",
    "ic": "bn_sod_x",
    "domain_L": 1.0,
    "t_end": 0.01,
    "max_steps": 5,
    "bn_gamma1": 1.4,
    "bn_gamma2": 1.6,
    "bn_pinf1": 0.0,
    "bn_pinf2": 0.0,
    "bn_cfl": 0.4
}
EOF
./build/simulate /tmp/test_bn.json
```

Expected: prints "BN solver" line and "BN done" line with a positive `t` value. No crash.

- [ ] **Step 4: Run CPU gate suite**

```bash
cmake --build build -t ba 2>&1 | tail -5
```

Expected: 0 failures. (BN path is triggered only when `model=bn`; default `ns` path is unchanged.)

- [ ] **Step 5: Commit**

```bash
git add apps/simulate.cpp
git commit -m "simulate: add BN two-phase dispatch path (model=bn → BNSolver)"
```

---

### Task 6: Add grouped config sections to `simulate_gpu.cu`

**Files:**
- Modify: `apps/simulate_gpu.cu`

`simulate_gpu.cu` already builds `NSSolver` + `GpuGraphSolver`. This task wires the same new config keys: scheme, ACDI, Dynamic SGS fix, combustion/radiation/WMLES/BN warnings.

- [ ] **Step 1: Add `model` + `gpu` parse (informational)** — after the line `sc.physics.use_imex = cfg.b("use_imex", false);` in `simulate_gpu.cu`:

```cpp
    // === Model selection (informational; GPU binary always uses NSSolver) ===
    const std::string model = cfg.str("model", "ns");
    if (model == "bn")
        fprintf(stderr, "[WARN] simulate_gpu: model=bn not yet wired for GPU — use simulate for BN\n");
```

- [ ] **Step 2: Extend `parse_bc_str` for NSCBC** — same as Task 4 Step 3, applied to the copy in `simulate_gpu.cu` (around line 153):

```cpp
        auto parse_bc_str = [&](const std::string& s) -> BCVariant {
            if (s == "Wall")  return WallBC{};
            if (s == "Open")  return OpenBC{};
            if (s == "NSCBC") return NscbcBC{ cfg.d("nscbc_p_inf", 1.0) };
            return PeriodicBC{};
        };
```

- [ ] **Step 3: Add grouped config sections** — insert after the SGS block (after `sc.physics.sgs = nullptr;`):

```cpp
    // === Scheme selection ===
    {
        const std::string sch = cfg.str("scheme", "weno5z");
        if (sch == "teno5a")
            sc.exec.recon = SolverConfig::ReconScheme::TENO5A;
        else if (sch == "teno7a") {
            fprintf(stderr, "[WARN] simulate_gpu: scheme=teno7a not yet instantiated — falling back to teno5a\n");
            sc.exec.recon = SolverConfig::ReconScheme::TENO5A;
        }
    }

    // === Viscosity ===
    {
        const double mu         = cfg.d("mu",         0.0);
        const bool   sutherland = cfg.b("sutherland", false);
        (void)mu; (void)sutherland;
    }

    // === ACDI phase field ===
    sc.acdi.use_acdi  = cfg.b("acdi",         false);
    sc.acdi.acdi_ceps = cfg.d("acdi_ceps",    0.0);
    sc.acdi.gamma_a   = cfg.d("acdi_gamma_a", GAMMA);
    sc.acdi.gamma_b   = cfg.d("acdi_gamma_b", GAMMA);
    sc.acdi.p_inf_a   = cfg.d("acdi_pinf_a",  0.0);
    sc.acdi.p_inf_b   = cfg.d("acdi_pinf_b",  0.0);

    // === Combustion / Arrhenius ===
    sc.physics.combustion_enabled = cfg.b("combustion",      false);
    sc.physics.arrhenius.A        = cfg.d("combustion_A",    1e4);
    sc.physics.arrhenius.T_act    = cfg.d("combustion_Tact", 10.0);
    sc.physics.arrhenius.q_heat   = cfg.d("combustion_Q",    10.0);
    sc.physics.arrhenius.n_sub    = cfg.i("combustion_nsub", 8);
    if (sc.physics.combustion_enabled)
        fprintf(stderr, "[WARN] simulate_gpu: combustion wired via gpu_source.cu — not yet in simulate_gpu.cu step loop\n");

    // === Radiation / P1 ===
    sc.physics.radiation_enabled  = cfg.b("radiation",       false);
    sc.physics.rad_params.kappa   = cfg.d("radiation_kappa", 1.0);
    sc.physics.rad_params.a_rad   = cfg.d("radiation_arad",  1.0);
    if (sc.physics.radiation_enabled)
        fprintf(stderr, "[WARN] simulate_gpu: radiation wired via gpu_p1.cu — not yet in simulate_gpu.cu step loop\n");

    // === WMLES ===
    sc.physics.wmles_enabled        = cfg.b("wmles",         false);
    sc.physics.wall_model.use_ode   = (cfg.str("wmles_model","reichardt") == "ode");
    if (sc.physics.wmles_enabled)
        fprintf(stderr, "[WARN] simulate_gpu: wmles wired via gpu_wmles.cu — not yet in simulate_gpu.cu step loop\n");
```

- [ ] **Step 4: Fix Dynamic SGS injection** — the existing `simulate_gpu.cu` only handles `SmagorinskyModel` via dynamic_cast. The block starting at line 231 reads:

```cpp
    if (sc.physics.sgs) {
        if (auto* sm = dynamic_cast<SmagorinskyModel*>(sc.physics.sgs.get()))
            graph_solver.set_gpu_sgs(sm->Cs, sm->Pr_t);
    }
```

Replace with a version that also handles `DynamicSmagorinskyModel`:

```cpp
    if (sc.physics.sgs) {
        if (auto* sm = dynamic_cast<SmagorinskyModel*>(sc.physics.sgs.get()))
            graph_solver.set_gpu_sgs(sm->Cs, sm->Pr_t);
        else if (auto* dm = dynamic_cast<DynamicSmagorinskyModel*>(sc.physics.sgs.get()))
            graph_solver.set_gpu_dyn_sgs(dm->Pr_t);
    }
```

- [ ] **Step 5: Wire ACDI to GpuGraphSolver** — after the SGS injection block (after Step 4's change), add:

```cpp
    if (sc.acdi.use_acdi)
        graph_solver.set_gpu_acdi(sc.acdi.acdi_ceps);
```

- [ ] **Step 6: Build simulate_gpu**

```bash
cmake --build build -t simulate_gpu 2>&1 | tail -5
```

Expected: zero errors. (GPU binary build is tested only when CUDA is available; if not, the nvcc target is skipped — that's expected.)

- [ ] **Step 7: Commit**

```bash
git add apps/simulate_gpu.cu
git commit -m "simulate_gpu: add scheme/ACDI/combustion/radiation/wmles config + Dynamic SGS fix"
```

---

### Task 7: Update `template.json`

**Files:**
- Modify: `apps/template.json`

Add all new flat config keys with comment-block grouping. They go after the `"sgs_prt"` block and before `"diag_interval"`. Also extend the IC section with BN and reactive-blast keys.

- [ ] **Step 1: Insert new sections after the `"sgs_prt"` line** — add the following block between `"sgs_prt"` and `"diag_interval"`:

```json
    // ── Model dispatch ────────────────────────────────────────────────────────
    "model"              : "ns",
    // Simulation model:
    //   "ns"  — single-phase NSSolver (default; all other keys apply)
    //   "bn"  — Baer-Nunziato two-phase BNSolver (uses bn_* keys below)
    "gpu"                : false,
    // GPU execution: true → use simulate_gpu binary (this key is informational in simulate).

    // ── Scheme selection ──────────────────────────────────────────────────────
    "scheme"             : "weno5z",
    // Convective reconstruction scheme for the CPU path:
    //   "weno5z"  — WENO5-Z (default; 5th-order shock-capturing)
    //   "teno5a"  — TENO5-A (better spectral resolution for smooth flows)
    // GPU path always uses the scheme compiled into gpu_rhs.cu (currently WENO5-Z).

    // ── Viscosity ─────────────────────────────────────────────────────────────
    "mu"                 : 0.0,     // Dynamic viscosity [Pa·s]. 0 = inviscid.
    "sutherland"         : false,   // Sutherland's law μ(T) — GPU only; ignored on CPU.

    // ── ACDI phase field (P14.1) ──────────────────────────────────────────────
    "acdi"               : false,   // Enable ACDI compressible interface (φ ∈ [0,1]).
    "acdi_ceps"          : 0.0,     // Compression coefficient Cε; 0 = pure advection.
    "acdi_gamma_a"       : 1.4,     // γ for fluid A (φ=1). E.g. 6.12 for liquid water.
    "acdi_gamma_b"       : 1.4,     // γ for fluid B (φ=0). 1.4 for air.
    "acdi_pinf_a"        : 0.0,     // p∞ for fluid A [Pa]. 3.43e8 for liquid water.
    "acdi_pinf_b"        : 0.0,     // p∞ for fluid B [Pa]. 0 for ideal gas.

    // ── Combustion / Arrhenius (D5) ───────────────────────────────────────────
    "combustion"         : false,   // Enable single-step Arrhenius chemistry.
                                    // Requires simulate_gpu (GPU-only for now).
    "combustion_A"       : 1e4,     // Pre-exponential factor [1/s].
    "combustion_Tact"    : 10.0,    // Activation temperature T_act [code units].
    "combustion_Q"       : 10.0,    // Heat release per unit mass of reactant.
    "combustion_nsub"    : 8,       // Chemistry substeps per fluid step.

    // ── Radiation / P1 (D6) ───────────────────────────────────────────────────
    "radiation"          : false,   // Enable P1 diffusion-limit radiation.
                                    // Requires simulate_gpu (GPU-only for now).
    "radiation_kappa"    : 1.0,     // Absorption opacity κ [1/length].
    "radiation_arad"     : 1.0,     // Radiation constant a_rad [energy/(volume·T⁴)].

    // ── WMLES (D7) ────────────────────────────────────────────────────────────
    "wmles"              : false,   // Enable wall-modelled LES ghost-fill.
                                    // Requires simulate_gpu (GPU-only for now).
    "wmles_model"        : "reichardt",
    // Wall model variant: "reichardt" (algebraic) | "ode" (thin-boundary-layer).

    // ── NSCBC ─────────────────────────────────────────────────────────────────
    // Set any face bc to "NSCBC" to enable characteristic boundary conditions.
    // Example: "bc_xhi": "NSCBC",  "bc_xlo": "Periodic"
    "nscbc_p_inf"        : 1.0,     // Subsonic outflow target static pressure.

    // ── BN two-phase model (model="bn") ───────────────────────────────────────
    "bn_gamma1"          : 1.4,     // γ₁ for phase 1 (e.g. 1.4 for air).
    "bn_gamma2"          : 4.4,     // γ₂ for phase 2 (stiffened-gas water; use 1.6 for ideal-gas tests).
    "bn_pinf1"           : 0.0,     // p∞₁ [Pa]. 0 for ideal gas.
    "bn_pinf2"           : 6e8,     // p∞₂ [Pa]. 6e8 for stiffened-gas water.
    "bn_cfl"             : 0.4,     // CFL for BNSolver (acoustic speeds are stiffer).
```

- [ ] **Step 2: Extend the IC section** — insert before the `"checkpoint_load"` entry:

```json
    // IC: bn_sod_x (BN two-phase 1D shock tube, x-aligned) ────────────────────
    "ic_bn_alpha1_l"     : 0.9,     // Left-state volume fraction α₁
    "ic_bn_alpha1_r"     : 0.1,     // Right-state volume fraction α₁
    "ic_bn_p_l"          : 1.0,     // Left-state pressure (also used for bn_uniform)
    "ic_bn_p_r"          : 0.1,     // Right-state pressure

    // IC: reactive_blast (spherical hot kernel for Arrhenius ignition) ─────────
    "ic_blast_r"         : 0.1,     // Kernel radius as fraction of domain_L
    "ic_blast_T_hot"     : 4.0,     // Hot-kernel temperature (cold = 1.0)
    "ic_blast_p"         : 1.0,     // Background pressure
```

- [ ] **Step 3: Update the IC comment block** — find the `// Supported IC names —` comment in the IC section and add the two new names:

```json
    //   "bn_sod_x"          — BN two-phase 1D Sod tube (ic_bn_alpha1_l/r, ic_bn_p_l/r)
    //   "bn_uniform"        — BN uniform state (uses ic_bn_alpha1_l, ic_bn_p_l)
    //   "reactive_blast"    — spherical hot kernel for Arrhenius ignition (ic_blast_*)
```

- [ ] **Step 4: Smoke-test with the updated template**

```bash
cmake --build build --target simulate 2>&1 | tail -3
./build/simulate apps/template.json 2>&1 | head -10
```

Expected: runs, prints config, exits normally (no crash, no unexpected warnings since all new keys default to "off").

- [ ] **Step 5: Run full CPU gate suite**

```bash
cmake --build build -t ba 2>&1 | tail -5
```

Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add apps/template.json
git commit -m "template.json: add model/gpu/scheme/ACDI/combustion/radiation/wmles/NSCBC/BN sections"
```

---

### Task 8: Final gate check + memory update

**Files:**
- Read-only gate run

- [ ] **Step 1: Full gate sweep**

```bash
cmake --build build -t ba 2>&1 | grep -E "passed|failed|PASS|FAIL" | tail -20
```

Expected: all layers pass (11 + 42 + 61 + 28 + 5 = 147 tests, 0 failures).

- [ ] **Step 2: Verify simulate smoke-test with non-default new keys**

```bash
cat > /tmp/test_acdi.json << 'EOF'
{
    "model": "ns",
    "scheme": "teno5a",
    "acdi": true,
    "acdi_ceps": 0.5,
    "acdi_gamma_a": 1.4,
    "acdi_gamma_b": 1.4,
    "t_end": 0.001,
    "max_steps": 2,
    "verbose": false
}
EOF
./build/simulate /tmp/test_acdi.json 2>&1 | head -10
```

Expected: no crash, scheme=teno5a reported, acdi=true parsed.

- [ ] **Step 3: Verify NSCBC parse**

```bash
cat > /tmp/test_nscbc.json << 'EOF'
{
    "bc_xlo": "Periodic",
    "bc_xhi": "NSCBC",
    "bc_ylo": "Periodic",
    "bc_yhi": "Periodic",
    "bc_zlo": "Periodic",
    "bc_zhi": "Periodic",
    "nscbc_p_inf": 0.9,
    "t_end": 0.001,
    "max_steps": 2,
    "verbose": false
}
EOF
./build/simulate /tmp/test_nscbc.json 2>&1 | head -10
```

Expected: no crash (NSCBC BC is parsed; ghost fill uses it on the xhi face).

- [ ] **Step 4: Final commit if any last-minute fixes needed**

```bash
cmake --build build -t ba 2>&1 | tail -3
git add -p   # stage only intentional changes
git commit -m "simulate-sync: final gate green; all 147 CPU tests pass"
```

---

## Summary of commits

After all tasks:

| Commit | Contents |
|---|---|
| `cfg: add ReconScheme + combustion/radiation/wmles fields` | `ns_solver.hpp` |
| `scheme: add Teno5Recon dispatch` | `operators.hpp`, `cpu_rk3.cpp` |
| `ic: add build_bn_ic() + reactive_blast` | `initial_conditions.hpp` |
| `simulate: add scheme/ACDI/combustion/radiation/wmles/NSCBC/BN config` | `simulate.cpp` |
| `simulate: add BN two-phase dispatch path` | `simulate.cpp` |
| `simulate_gpu: add scheme/ACDI/combustion/radiation/wmles config` | `simulate_gpu.cu` |
| `template.json: add all new sections` | `template.json` |
| `simulate-sync: final gate green` | (only if fixes needed) |
