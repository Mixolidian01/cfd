<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# if the simulation parameters are read by a JSON file, won't there be runtime branching ? what to do if i want the simulation parameters and specifications be on an input file, and known at compile time?

This is a fundamental tension in scientific computing software design. Let me give the complete answer.

***

## The Core Problem

A JSON file is read at runtime. Template parameters, `constexpr`, `concept` constraints, and `if constexpr` branches all resolve at **compile time**. These two requirements seem mutually exclusive — but they are not, if you architect the boundary between them correctly.

***

## Why Runtime Branching From JSON Is Actually the Right Default

First, a critical clarification: **not all runtime branching is bad**. The branching that kills GPU performance is **warp-divergent branching inside hot kernels** — threads in the same warp taking different paths. A `switch` statement in the solver initialisation, executed once on the CPU before the first time step, has zero performance cost.

The real question is: **which parameters need to be compile-time, and which can stay runtime?**

```
┌─────────────────────────────────────────────────────────────┐
│ COMPILE-TIME (must be template parameters / constexpr)      │
│                                                             │
│  • Axis count (2D vs 3D) — affects loop bounds, mdspan      │
│  • Ghost cell count NG — affects block allocation size      │
│  • Block size (NX, NY, NZ) — affects thread-block dims      │
│  • Flux scheme type (HLLC vs Roe vs Chandrashekar)          │
│  • Reconstruction order (WENO3 vs WENO5 vs WENO7)           │
│  • Single vs multiphase (affects ConservedVars layout)      │
│  • Viscous vs inviscid                                      │
├─────────────────────────────────────────────────────────────┤
│ RUNTIME (JSON fine, zero kernel overhead)                   │
│                                                             │
│  • Domain size (Lx, Ly, Lz)                                 │
│  • Physical constants (gamma, nu, rho_ref)                  │
│  • CFL number, dt_max                                       │
│  • AMR refinement thresholds                                │
│  • Output frequency, checkpoint interval                    │
│  • Initial condition parameters                             │
│  • Boundary condition state values (p_inlet, T_wall)        │
│  • Number of MPI ranks (determined at launch)               │
└─────────────────────────────────────────────────────────────┘
```

The physical constants (γ, ν) that are runtime values **do not cause warp divergence** — they are loaded into registers once per kernel launch and used uniformly by all threads. They belong in the functor's state (as shown in answer on functors), not as template parameters.

***

## Strategy 1 — Code Generation: JSON → C++ Header at Build Time

This is the cleanest approach and the one used by high-performance solvers like AMReX and OpenFOAM's build system. The JSON file is consumed **by the build system**, not by the solver binary.

### The Pipeline

```
simulation.json
      │
      ▼  (Python script, CMake custom command, or Jinja2 template)
generated_config.hpp    ← compile-time constants
      │
      ▼  (C++ compiler)
solver binary           ← all scheme choices baked in, zero runtime branching
```


### The JSON File

```json
{
  "compile_time": {
    "dimensions":        3,
    "block_size":        [16, 16, 16],
    "ghost_cells":       3,
    "flux_scheme":       "ChandrashekarFlux",
    "reconstruction":    "WENO5",
    "multiphase":        true,
    "viscous":           true,
    "equation_of_state": "IdealGas"
  },
  "runtime": {
    "domain":            [1.0, 1.0, 1.0],
    "gamma":             1.4,
    "nu":                1e-5,
    "cfl":               0.4,
    "t_end":             2.0,
    "amr_threshold":     0.1
  }
}
```


### The Code Generator (Python, runs at CMake configure time)

```python
# generate_config.py — called by CMake before compilation

import json, sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
ct  = cfg["compile_time"]

header = f"""
// AUTO-GENERATED — do not edit. Source: {sys.argv[1]}
#pragma once

inline constexpr int  NDIM        = {ct['dimensions']};
inline constexpr int  BLOCK_NX    = {ct['block_size'][0]};
inline constexpr int  BLOCK_NY    = {ct['block_size'][1]};
inline constexpr int  BLOCK_NZ    = {ct['block_size'][2]};
inline constexpr int  NG          = {ct['ghost_cells']};
inline constexpr bool MULTIPHASE  = {'true' if ct['multiphase'] else 'false'};
inline constexpr bool VISCOUS     = {'true' if ct['viscous'] else 'false'};

// Scheme type aliases — resolved at compile time
using FluxScheme      = {ct['flux_scheme']};
using ReconScheme     = {ct['reconstruction']};
using EOSType         = {ct['equation_of_state']};
"""

Path("generated_config.hpp").write_text(header)
```


### CMake Integration

```cmake
# CMakeLists.txt

set(SIM_JSON "${CMAKE_SOURCE_DIR}/simulation.json")
set(GEN_HDR  "${CMAKE_BINARY_DIR}/generated_config.hpp")

add_custom_command(
    OUTPUT  ${GEN_HDR}
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_config.py
                    ${SIM_JSON} ${GEN_HDR}
    DEPENDS ${SIM_JSON}
    COMMENT "Generating compile-time config from simulation.json"
)

add_custom_target(generate_config DEPENDS ${GEN_HDR})
add_dependencies(cfd_solver generate_config)
target_include_directories(cfd_solver PRIVATE ${CMAKE_BINARY_DIR})
```

**Result:** changing `"flux_scheme": "HLLC"` in the JSON triggers a recompile of the affected translation units only. The binary contains zero scheme-selection branches. The user still edits a human-readable JSON.

***

## Strategy 2 — `constexpr` Parsing at Compile Time (C++20)

If you want the JSON parsing itself to happen at compile time — no code generator, no Python — C++20 `constexpr` evaluation can do this for a **simple subset** of JSON (essentially a key-value config format).

### A `constexpr` Config Parser

```cpp
// constexpr_config.hpp
// Parses a compile-time string literal into config values
// Works for flat key-value integer/bool configs (not nested JSON)

#include <string_view>
#include <array>

struct CompileTimeConfig {
    int  block_nx    = 16;
    int  block_ny    = 16;
    int  block_nz    = 16;
    int  ghost_cells = 3;
    int  dimensions  = 3;
    bool multiphase  = false;
    bool viscous     = true;
};

consteval int parse_int_field(std::string_view src, std::string_view key) {
    auto pos = src.find(key);
    if (pos == std::string_view::npos) return -1;
    pos = src.find(':', pos) + 1;
    while (src[pos] == ' ') ++pos;
    int result = 0;
    while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9')
        result = result * 10 + (src[pos++] - '0');
    return result;
}

consteval bool parse_bool_field(std::string_view src, std::string_view key) {
    auto pos = src.find(key);
    if (pos == std::string_view::npos) return false;
    pos = src.find(':', pos) + 1;
    while (src[pos] == ' ') ++pos;
    return src.substr(pos, 4) == "true";
}

// The config string is embedded as a raw string literal
// Updated via __has_include or a generated header
consteval CompileTimeConfig parse_config(std::string_view src) {
    return {
        .block_nx    = parse_int_field(src,  "\"block_nx\""),
        .block_ny    = parse_int_field(src,  "\"block_ny\""),
        .block_nz    = parse_int_field(src,  "\"block_nz\""),
        .ghost_cells = parse_int_field(src,  "\"ghost_cells\""),
        .dimensions  = parse_int_field(src,  "\"dimensions\""),
        .multiphase  = parse_bool_field(src, "\"multiphase\""),
        .viscous     = parse_bool_field(src, "\"viscous\""),
    };
}
```

Usage in the solver:

```cpp
// solver.hpp

// The config string is injected at compile time
// (either via a generated header, or __has_include trick)
#include "sim_config_string.hpp"   // defines SIM_CONFIG_JSON_STRING

inline constexpr auto CFG = parse_config(SIM_CONFIG_JSON_STRING);

// Now use CFG as constexpr values:
inline constexpr int  BLOCK_NX   = CFG.block_nx;
inline constexpr int  NG         = CFG.ghost_cells;
inline constexpr bool MULTIPHASE = CFG.multiphase;

// Scheme selection from constexpr config
using FluxScheme = std::conditional_t<
    /* condition from CFG */ false,   // extend with consteval scheme selector
    ChandrashekarFlux,
    HLLCFlux
>;
```

**Limitation:** Full nested JSON parsing at `consteval` time is complex. Strategy 1 (code generator) is simpler and more maintainable for realistic configs. Strategy 2 is elegant for simple flat configs.

***

## Strategy 3 — `if constexpr` + Explicit Instantiation Matrix

If you want full runtime JSON flexibility **without recompiling** but still no warp-divergent kernel branching, the answer is an **explicit instantiation matrix**: pre-compile all meaningful combinations at build time, then select among them at runtime startup — once, before the first timestep.

```cpp
// instantiation_matrix.hpp
// All scheme combinations are compiled into the binary
// Selection happens once at startup

#include "weno5_flux_sweep.hpp"
#include "hllc_flux.hpp"
#include "chandrashekar_flux.hpp"

// Explicit instantiations — compiled into separate translation units
// Each is a fully specialised, branch-free kernel

// HLLC + WENO5 + single-phase
template struct WENO5FluxSweep<Axis::X, HLLCFlux,         IdealGas>;
template struct WENO5FluxSweep<Axis::Y, HLLCFlux,         IdealGas>;
template struct WENO5FluxSweep<Axis::Z, HLLCFlux,         IdealGas>;

// Chandrashekar + WENO5 + single-phase
template struct WENO5FluxSweep<Axis::X, ChandrashekarFlux, IdealGas>;
template struct WENO5FluxSweep<Axis::Y, ChandrashekarFlux, IdealGas>;
template struct WENO5FluxSweep<Axis::Z, ChandrashekarFlux, IdealGas>;

// Chandrashekar + WENO5 + multiphase (ACDI)
template struct WENO5FluxSweep<Axis::X, ChandrashekarFlux, MultiPhaseEOS>;
// ... etc.
```

The runtime dispatch — which variant to run — lives in a **factory function** called once at solver startup from the JSON parameters:

```cpp
// solver_factory.cpp

std::unique_ptr<ISolverKernel> make_solver(const RuntimeConfig& cfg) {

    // This if/else chain runs ONCE at startup — zero GPU overhead
    if      (cfg.flux == "Chandrashekar" && cfg.eos == "IdealGas")
        return std::make_unique<
            ConcreteSolver<ChandrashekarFlux, WENO5, IdealGas>>();

    else if (cfg.flux == "HLLC"          && cfg.eos == "IdealGas")
        return std::make_unique<
            ConcreteSolver<HLLCFlux, WENO5, IdealGas>>();

    else if (cfg.flux == "Chandrashekar" && cfg.eos == "MultiPhase")
        return std::make_unique<
            ConcreteSolver<ChandrashekarFlux, WENO5, MultiPhaseEOS>>();

    else
        throw std::invalid_argument("Unknown scheme combination: "
            + cfg.flux + " + " + cfg.eos);
}
```

The `ISolverKernel` interface uses the **CRTP pattern** (no virtual dispatch on GPU — the factory runs on CPU, the kernel runs on GPU via the already-resolved concrete type). The GPU kernel is branch-free because `ConcreteSolver<ChandrashekarFlux, WENO5, IdealGas>` is a fully specialised type.

***

## Recommended Architecture: Hybrid of Strategies 1 and 3

| Config type | Strategy | Mechanism |
| :-- | :-- | :-- |
| Block size, ghost cells, `NDIM` | **Strategy 1** (code gen) | Must be compile-time — affect `mdspan` extents, thread-block dims, shared memory allocation |
| Flux scheme, EOS, multiphase flag | **Strategy 3** (instantiation matrix) | Pre-compiled, runtime-selected once at startup — allows scheme switching without recompile |
| Physical constants (γ, ν, CFL) | **Runtime JSON** (plain) | Loaded into functor state — no kernel branching, uniform across all threads |
| Output parameters, AMR thresholds | **Runtime JSON** (plain) | CPU-only logic, zero GPU impact |

The complete flow:

```
simulation.json
       │
       ├──► compile_time section
       │         │
       │         ▼ (CMake + generate_config.py)
       │    generated_config.hpp  ──► constexpr BLOCK_NX, NG, NDIM
       │
       └──► runtime section
                 │
                 ▼ (solver startup, once)
            RuntimeConfig struct  ──► factory selects pre-compiled kernel
                 │
                 ▼ (hot loop, billions of calls)
            branch-free GPU kernel — zero JSON overhead
```


***

## The Result

The user edits one `simulation.json`. Compile-time parameters trigger a targeted recompile via CMake dependency tracking. Runtime parameters are loaded at startup with no kernel overhead. The GPU sees only fully specialised, branch-free kernels. The distinction between "compile-time" and "runtime" is encoded in the JSON structure itself — under `"compile_time"` vs `"runtime"` keys — making it explicit and self-documenting.

