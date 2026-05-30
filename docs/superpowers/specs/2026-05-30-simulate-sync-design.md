# simulate.cpp + template.json Sync Design

> **Goal:** Bring `apps/simulate.cpp` and `apps/template.json` in sync with the evolved
> codebase: scheme selection, WMLES, NSCBC, ACDI phase field, BN two-phase dispatch,
> combustion/Arrhenius, P1 radiation, GPU dispatch, and two-phase/reactive ICs.

---

## Scope

Three primary files change:

| File | Change |
|---|---|
| `apps/simulate.cpp` | Add 9 grouped config sections; BN + GPU dispatch paths |
| `apps/template.json` | Add matching flat keys with comment-block grouping |
| `apps/initial_conditions.hpp` | Add `bn_sod_x`, `bn_uniform`, `reactive_blast` ICs |

Two supporting files receive minimal additions:

| File | Change |
|---|---|
| `include/solver/ns_solver.hpp` | Add `ReconScheme` enum + combustion/radiation/WMLES fields to `SolverConfig` |
| `src/solver/cpu_rk3.cpp` | Extend `select_scheme()` with TENO5A branch |
| `include/schemes/operators.hpp` | Add `extern template` suppressors for Teno5Recon |

---

## Parser constraint

`apps/simulate.cpp` uses a hand-rolled flat-JSON parser (`struct Config`).
It parses only flat `"key": scalar_value` pairs; nested objects are not supported.
All new config keys are therefore **flat** — no nested objects anywhere.
Comment blocks in `template.json` provide the visual grouping that the parser cannot.

---

## Config key catalogue

Keys are listed with their flat name, type, default, and the SolverConfig field or
runtime action they drive.

### Model dispatch

| Key | Type | Default | Maps to |
|---|---|---|---|
| `"model"` | string | `"ns"` | Selects `NSSolver` (`"ns"`) or `BNSolver` (`"bn"`) |
| `"gpu"` | bool | `false` | `sc.exec.use_gpu = true`; builds `GpuGraphSolver` |

### Scheme selection (CPU + GPU)

| Key | Type | Default | Maps to |
|---|---|---|---|
| `"scheme"` | string | `"weno5z"` | `sc.exec.recon` (new enum); `"teno5a"` supported; `"teno7a"` logs warning (not yet instantiated) |

`SolverConfig::ExecConfig` gains:
```cpp
enum class ReconScheme { WENO5Z, TENO5A } recon = ReconScheme::WENO5Z;
```

`cpu_rk3.cpp::select_scheme()` gains a Teno5Recon branch alongside the existing
Weno5Recon dispatch (same flux × EOS matrix, new Recon axis).

### Viscosity (new keys, no SolverConfig field needed yet)

| Key | Type | Default | Notes |
|---|---|---|---|
| `"mu"` | double | `0.0` | Stored locally; passed to GPU path as `mu_lam`. CPU NSSolver ignores (viscous already handled via SGS) |
| `"sutherland"` | bool | `false` | Enable Sutherland's law `μ(T)`; GPU-only for now; logged as warning on CPU |

### SGS (extends existing keys)

The existing `"sgs"` key already dispatches `"none"` / `"Smagorinsky"` / `"Dynamic"`.
No new SGS keys — WMLES is a separate key (below).

### ACDI phase field (new keys → existing `SolverConfig::AcdiConfig` fields)

| Key | Type | Default | Maps to |
|---|---|---|---|
| `"acdi"` | bool | `false` | `sc.acdi.use_acdi` |
| `"acdi_ceps"` | double | `0.0` | `sc.acdi.acdi_ceps` |
| `"acdi_gamma_a"` | double | `1.4` | `sc.acdi.gamma_a` |
| `"acdi_gamma_b"` | double | `1.4` | `sc.acdi.gamma_b` |
| `"acdi_pinf_a"` | double | `0.0` | `sc.acdi.p_inf_a` |
| `"acdi_pinf_b"` | double | `0.0` | `sc.acdi.p_inf_b` |

### Combustion / Arrhenius (new keys → new `SolverConfig::PhysicsConfig` fields)

| Key | Type | Default | Maps to |
|---|---|---|---|
| `"combustion"` | bool | `false` | `sc.physics.combustion_enabled` |
| `"combustion_A"` | double | `1e4` | `sc.physics.arrhenius.A` |
| `"combustion_Tact"` | double | `10.0` | `sc.physics.arrhenius.T_act` |
| `"combustion_Q"` | double | `10.0` | `sc.physics.arrhenius.q_heat` |
| `"combustion_nsub"` | int | `8` | `sc.physics.arrhenius.n_sub` |

`SolverConfig::PhysicsConfig` gains:
```cpp
#include "physics/arrhenius.hpp"
bool           combustion_enabled = false;
ArrheniusParams arrhenius{};
```

**CPU path**: when `"combustion": true` and `"gpu": false`, simulate.cpp logs
`[WARN] combustion not yet wired for CPU path — enable gpu:true` and continues.

### Radiation / P1 (new keys → new `SolverConfig::PhysicsConfig` fields)

| Key | Type | Default | Maps to |
|---|---|---|---|
| `"radiation"` | bool | `false` | `sc.physics.radiation_enabled` |
| `"radiation_kappa"` | double | `1.0` | `sc.physics.rad_params.kappa` |
| `"radiation_arad"` | double | `1.0` | `sc.physics.rad_params.a_rad` |

`SolverConfig::PhysicsConfig` gains:
```cpp
#include "physics/p1_radiation.hpp"
bool            radiation_enabled = false;
RadiationParams rad_params{};
```

**CPU path**: same warning as combustion — GPU-only for now.

### WMLES (new keys → new `SolverConfig::PhysicsConfig` fields)

| Key | Type | Default | Maps to |
|---|---|---|---|
| `"wmles"` | bool | `false` | `sc.physics.wmles_enabled` |
| `"wmles_model"` | string | `"reichardt"` | `sc.physics.wall_model.use_ode` (false for "reichardt", true for "ode") |

`SolverConfig::PhysicsConfig` gains:
```cpp
#include "models/wall_model.hpp"
bool          wmles_enabled = false;
WallModelCfg  wall_model{};
```

**GPU wiring**: `GpuGraphSolver` already has `gpu_wmles.cu`; `set_gpu_sgs()` / `set_gpu_dyn_sgs()` paths — WMLES wired analogously via a new `set_gpu_wmles(WallModelCfg)` call (if exposed) or flag in the existing build() call.

**CPU path**: `WallModelCfg` exists in `include/models/wall_model.hpp` but not yet integrated into `cpu_rk3.cpp`; logs `[WARN] wmles not yet wired for CPU path`.

### NSCBC (extends existing BC dispatch)

| Key | Type | Default | Notes |
|---|---|---|---|
| `"bc": "NSCBC"` | — | — | Extends `parse_bc_str` lambda to return `NscbcBC{nscbc_p_inf}` |
| `"nscbc_p_inf"` | double | `1.0` | `NscbcBC::p_inf` (subsonic outflow target pressure) |

`NscbcBC` is already in `include/mesh/bc_types.hpp`; GPU bc integer encoding is 3.

### BN two-phase model (new keys → `BNEosParams`)

| Key | Type | Default | Maps to |
|---|---|---|---|
| `"bn_gamma1"` | double | `1.4` | `BNEosParams::gamma1` |
| `"bn_gamma2"` | double | `1.6` | `BNEosParams::gamma2` |
| `"bn_pinf1"` | double | `0.0` | `BNEosParams::pinf1` |
| `"bn_pinf2"` | double | `6e8` | `BNEosParams::pinf2` |
| `"bn_cfl"` | double | `0.4` | CFL for `bn_cfl_dt()` (BN acoustic speeds are stiffer) |

Active only when `"model": "bn"`.

---

## simulate.cpp dispatch structure

```
main() {
    // === Model selection ===
    model = cfg.str("model", "ns")
    use_gpu = cfg.b("gpu", false)

    // === Scheme selection ===
    sc.exec.recon = (scheme == "teno5a") ? TENO5A : WENO5Z

    // === Viscosity ===
    mu = cfg.d("mu", 0.0)
    sutherland = cfg.b("sutherland", false)

    // === SGS (existing block, unchanged) ===
    ...

    // === ACDI phase field ===
    sc.acdi.use_acdi  = cfg.b("acdi", false)
    sc.acdi.acdi_ceps = cfg.d("acdi_ceps", 0.0)
    sc.acdi.gamma_a   = cfg.d("acdi_gamma_a", 1.4)
    sc.acdi.gamma_b   = cfg.d("acdi_gamma_b", 1.4)
    sc.acdi.p_inf_a   = cfg.d("acdi_pinf_a", 0.0)
    sc.acdi.p_inf_b   = cfg.d("acdi_pinf_b", 0.0)

    // === Combustion / Arrhenius ===
    sc.physics.combustion_enabled  = cfg.b("combustion", false)
    sc.physics.arrhenius.A         = cfg.d("combustion_A",    1e4)
    sc.physics.arrhenius.T_act     = cfg.d("combustion_Tact", 10.0)
    sc.physics.arrhenius.q_heat    = cfg.d("combustion_Q",    10.0)
    sc.physics.arrhenius.n_sub     = cfg.i("combustion_nsub", 8)

    // === Radiation / P1 ===
    sc.physics.radiation_enabled    = cfg.b("radiation", false)
    sc.physics.rad_params.kappa     = cfg.d("radiation_kappa", 1.0)
    sc.physics.rad_params.a_rad     = cfg.d("radiation_arad",  1.0)

    // === WMLES ===
    sc.physics.wmles_enabled        = cfg.b("wmles", false)
    sc.physics.wall_model.use_ode   = (cfg.str("wmles_model","reichardt") == "ode")

    // === NSCBC: extends parse_bc_str ===
    "NSCBC" → NscbcBC{ cfg.d("nscbc_p_inf", 1.0) }

    // === BN EOS (read regardless; active only when model=="bn") ===
    BNEosParams bn_eos{ bn_gamma1, bn_gamma2, bn_pinf1, bn_pinf2 }

    // === Model dispatch ===
    if (model == "bn") {
        build_bn_ic(cfg)          // returns void(BNCellBlock&, ox, oy, oz, h)
        BNSolver bn_solver;
        bn_solver.bc = parse_bc_str(cfg.str("bc","Periodic"))
        bn_solver.init(domain_L, bn_ic, bn_eos, bn_solver.bc)
        warn if combustion/radiation/wmles/gpu set (not yet wired for BN)
        bn_solver.run(sc.time.t_end, sc.time.max_steps)
    } else {
        NSSolver solver;
        // (existing config wiring)
        // GPU path:
        if (use_gpu) {
            GpuGraphSolver gpu_sol;
            gpu_sol.set_gpu_sgs(...)       // if sgs active
            gpu_sol.set_gpu_dyn_sgs(...)   // if Dynamic
            gpu_sol.set_gpu_acdi(...)      // if acdi
            warn if combustion/radiation/wmles set but not yet wired to GpuGraphSolver
            solver.set_gpu_solver(&gpu_sol)
            solver.cfg.exec.use_gpu = true
        }
        solver.init(domain_L, ic)
        solver.run()
    }
}
```

---

## New ICs in `initial_conditions.hpp`

### `bn_sod_x`

Fills a `BNCellBlock` with a 1D x-aligned two-phase Sod initial condition.
Left half (x < domain_L/2): α₁ = `ic_bn_alpha1_l` (default 0.9), p = `ic_bn_p_l` (default 1.0).
Right half: α₁ = `ic_bn_alpha1_r` (default 0.1), p = `ic_bn_p_r` (default 0.1).
Both phases at rest (u=v=w=0), ρ₁=ρ₂=1. Uses `BNEosParams` from caller.

New flat keys: `"ic_bn_alpha1_l"`, `"ic_bn_alpha1_r"`, `"ic_bn_p_l"`, `"ic_bn_p_r"`.

Signature:
```cpp
std::function<void(BNCellBlock&, double ox, double oy, double oz, double h)>
build_bn_ic(const Config& cfg, const BNEosParams& eos);
```
Returns a factory dispatching on `cfg.str("ic","bn_sod_x")`:
- `"bn_sod_x"` → x-split Sod
- `"bn_uniform"` → uniform α₁ = 0.5, uniform pressure

### `reactive_blast`

Fills a `CellBlock` for NSSolver with a spherical hot kernel suitable for Arrhenius ignition.
Kernel (r < `ic_blast_r`, default 0.1·L): T_hot = `ic_blast_T_hot` (default 4.0), Y_fuel = 1.0.
Exterior: T = 1.0, Y_fuel = 0.0. Both at uniform pressure `ic_blast_p` (default 1.0).

New flat keys: `"ic_blast_r"`, `"ic_blast_T_hot"`, `"ic_blast_p"`.

Returns a `std::function<Prim(double,double,double)>` via `build_ic(cfg)` when `ic="reactive_blast"`.
The species scalar φ_s is a separate pass (not in `build_ic`; handled by the GPU source path).

---

## Runtime warnings (no abort)

When a feature flag is set but its path is not yet wired, simulate.cpp prints:

```
[WARN] simulate: combustion=true requires gpu:true — ignored on CPU path
[WARN] simulate: radiation=true requires gpu:true — ignored on CPU path
[WARN] simulate: wmles=true requires gpu:true — ignored on CPU path
[WARN] simulate: scheme=teno7a not yet instantiated — falling back to teno5a
[WARN] simulate: gpu=true requested for model=bn — not yet wired, falling back to CPU BN
```

---

## template.json additions

New sections appended after the existing `"sgs_prt"` block and before `"diag_interval"`,
using `//` comment headers for visual grouping (parser strips them):

```
// ── Model dispatch ─────────────────────────────────────────────────────────────
"model"              : "ns",      // "ns" | "bn"
"gpu"                : false,     // true → GpuGraphSolver path

// ── Scheme selection ──────────────────────────────────────────────────────────
"scheme"             : "weno5z",  // "weno5z" | "teno5a"

// ── Viscosity ─────────────────────────────────────────────────────────────────
"mu"                 : 0.0,       // dynamic viscosity (0 = inviscid)
"sutherland"         : false,     // Sutherland temperature law (GPU only)

// ── ACDI phase field ──────────────────────────────────────────────────────────
"acdi"               : false,
"acdi_ceps"          : 0.0,
"acdi_gamma_a"       : 1.4,
"acdi_gamma_b"       : 1.4,
"acdi_pinf_a"        : 0.0,
"acdi_pinf_b"        : 0.0,

// ── Combustion / Arrhenius ────────────────────────────────────────────────────
"combustion"         : false,
"combustion_A"       : 1e4,
"combustion_Tact"    : 10.0,
"combustion_Q"       : 10.0,
"combustion_nsub"    : 8,

// ── Radiation / P1 ───────────────────────────────────────────────────────────
"radiation"          : false,
"radiation_kappa"    : 1.0,
"radiation_arad"     : 1.0,

// ── WMLES ─────────────────────────────────────────────────────────────────────
"wmles"              : false,
"wmles_model"        : "reichardt",  // "reichardt" | "ode"

// ── NSCBC ─────────────────────────────────────────────────────────────────────
// Set "bc": "NSCBC" on any face key to activate; nscbc_p_inf sets target pressure.
"nscbc_p_inf"        : 1.0,

// ── BN two-phase model ────────────────────────────────────────────────────────
"bn_gamma1"          : 1.4,
"bn_gamma2"          : 1.6,
"bn_pinf1"           : 0.0,
"bn_pinf2"           : 6e8,
"bn_cfl"             : 0.4,
```

IC section gains BN and reactive-blast entries:
```
// ── IC: bn_sod_x / bn_uniform ────────────────────────────────────────────────
"ic_bn_alpha1_l"     : 0.9,
"ic_bn_alpha1_r"     : 0.1,
"ic_bn_p_l"          : 1.0,
"ic_bn_p_r"          : 0.1,

// ── IC: reactive_blast ────────────────────────────────────────────────────────
"ic_blast_r"         : 0.1,
"ic_blast_T_hot"     : 4.0,
"ic_blast_p"         : 1.0
```

---

## Limitations and non-goals

- TENO7A: not instantiated in `operators.cpp`; spec defers it; simulate.cpp falls back to teno5a with a warning.
- Combustion and radiation on the CPU NSSolver path: deferred. Params are stored in SolverConfig but the CPU RK3 loop ignores them (with a printed warning).
- WMLES on CPU path: deferred. WallModelCfg is stored but `cpu_rk3.cpp` does not call the wall-model ghost fill. Warning printed.
- BN + GPU: not yet wired; spec defers it with a warning.
- Multi-GPU BN: out of scope.
