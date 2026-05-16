# Refactor Session Configuration

This document is the single source of truth for starting a `to_refactor` branch session.
It contains: the adapted CLAUDE.md for that branch, the correct `.claude/settings.json`,
custom slash commands, and hook scripts — all corrected from the Perplexity drafts.
A final section explains what the refactoring unlocks for future development.

---

## 0. Before the Session: Branch Setup

```bash
git checkout to_debug            # start from the stable reference
git checkout -b to_refactor      # create the refactor branch
```

Then deploy the files described in sections 1–4 below.
The `to_debug` branch is kept untouched throughout; `to_refactor` diverges from it.

---

## 1. CLAUDE.md for `to_refactor`

*Replace the repo-root `CLAUDE.md` with this content on the `to_refactor` branch.*

```markdown
# CLAUDE.md — Refactor Branch

This file provides guidance to Claude Code (claude.ai/code) when working on the
`to_refactor` branch of this repository.

## Role

Act as an expert in: compressible CFD, multiphase flows, shock-capturing schemes,
turbulence modelling, C++20, CUDA/GPU programming, and modern HPC software design.
Numerical correctness and physical fidelity are non-negotiable.

## Branching Rules

- Working branch: `to_refactor` (diverges from `to_debug` at commit 2f16dae)
- `to_debug` is the correctness reference. Do NOT modify it.
- Every refactor phase must leave all gate tests green before committing.
- Gate command: `cmake --build build -t ba`

## Current Architecture (to_debug baseline)

The solver has a well-defined four-layer stack:

```
Layer 0  linalg.hpp/cpp          — Kahan BLAS-1, CG, multigrid
Layer 1  cell_block.hpp          — CellBlock (SoA, NB=8, NG=2, NB2=12, NCELL=1728)
         block_tree.hpp/cpp      — BlockTree octree, AMR prolong/restrict
         amr_operators.cpp       — fill_cf_ghosts (must stay in libblock)
Layer 2  operators.hpp/cpp       — HLLC-ES, WENO5-Z RHS, compute_rhs, tree_rhs
Layer 3  ns_solver.hpp/cpp       — SSP-RK3, regrid, flux correction
         gpu_graph.cuh/cu        — CUDA Graph RK3 (P8.6), positivity floor (P16.1)
         gpu_rhs.cuh/cu          — WENO5-Z GPU RHS
         gpu_ghost_fill.cuh/cu   — GPU ghost fill
```

Key constants (do not change without updating both CPU and GPU headers):
| NB=8 | NG=2 | NB2=12 | NCELL=1728 | NVAR=5 | GAMMA=1.4 |

## Target Architecture (to_refactor goal)

The refactor adds three orthogonal layers **on top of** the existing stack without
replacing it wholesale:

```
LAYER P — Physics functors  (include/physics/)
           Structs with operator(). __host__ __device__. template <Axis DIR>.
           Carry only physics state (γ, p_inf, μ…). No execution knowledge.

LAYER C — Concept contracts  (include/concepts.hpp)
           C++20 concept definitions. Applied at every template boundary.
           RiemannFlux, SPOperator, EquationOfState, BoundaryCondition.
           Property flags: is_entropy_stable, is_conservative, is_skew_symmetric.

LAYER E — Execution backend tags  (include/execution.hpp)
           CPUSerial, GPUCuda tags. Backend selected once at solver startup via
           factory. No physics or contract knowledge.
```

These three layers compose independently. A new flux scheme lives only in Layer P.
A new GPU backend lives only in Layer E. The mathematical guarantees live only in
Layer C and are independent of execution.

## Migration Phases (execute in order)

### R0 — Enable C++20
- `cmake -S . -B build -DCMAKE_CXX_STANDARD=20`
- Verify: nvcc 12.4+ with `--std=c++20`. Check `nvcc --version`.
- Zero functional change. All tests must still pass.

### R1 — Concept layer (additive, non-breaking)
- Create `include/concepts.hpp` with `RiemannFlux`, `EquationOfState`,
  `BoundaryCondition` concepts and compile-time property flags.
- Add `static_assert` checks at existing call sites in `operators.cpp` as
  verification only. No behavioral change.
- Gate: all t1–t26 pass.

### R2 — Physics functor extraction (refactor, not rewrite)
- Move `hllc_flux`, `hllc_es_flux_t`, `weno5_face_t` from `operators.cpp`
  into structs in `include/physics/`.
- Existing free functions become one-liner wrappers calling the functor.
- The concept from R1 is applied to each functor.
- Gate: all t1–t26 pass.

### R3 — BC variant dispatch (replaces if/else BC chains)
- Replace BC-type enum dispatch in ghost fill with
  `std::variant<PeriodicBC, WallBC, OpenBC, ContactAngleBC>`.
- `std::visit` replaces every BC if/else chain.
- Gate: all t1–t26 pass, contact angle T16a/T16b still pass.

### R4 — Backend tag dispatch (formalises existing CPU/GPU split)
- Create `include/execution.hpp` with `CPUSerial{}` and `GPUCuda{}` tags.
- Wrap existing CPU operator calls behind `CPUSerial` dispatch.
- Wrap existing GPU kernel calls behind `GPUCuda` dispatch.
- Factory: `make_solver(cfg)` reads scheme+backend from `SolverConfig` and
  returns a `std::unique_ptr<ISolver>` pointing to the correct pre-compiled type.
- Gate: all t1–t26 pass including GPU gates t19–t26.

### R5 — Instantiation matrix (Strategy 3 — enables runtime scheme selection)
- Pre-compile all supported (Flux × Recon × EOS) combinations in a dedicated
  translation unit `src/instantiation_matrix.cpp`.
- Factory dispatches to the correct specialisation once at startup.
- Eliminates all scheme-selection runtime branches from GPU kernels.
- Gate: all t1–t26 pass. Add new test verifying scheme selection from JSON.

### R6 — mdspan block access (optional, breaking — last)
- Replace `Q[v][idx]` raw access with `std::mdspan` views in `CellBlock`.
- Axis rotation for `template <Axis DIR>` kernels becomes a zero-copy
  layout policy instead of manual index arithmetic.
- This phase changes every block access in the codebase — do not start until
  R0–R5 are complete and stable.
- Gate: all t1–t26 pass. Check for regression in T08 convergence rate (≥1.8).

## Code Rules

### Mathematical / Numerical
1. Interior flux: entropy-conservative (Chandrashekar) or entropy-stable (HLLC-ES).
   Never plain Roe without entropy fix.
2. Convective operator: Pirozzoli (2010) split-form for the compressible case.
3. AMR C/F flux correction: undo_cf and accumulate_cf must use the same
   reconstruction as accumulate_face for exact Berger-Colella cancellation.
4. Positivity floor: ρ≥1e-12, p≥1e-12 after every RK3 stage on both CPU and GPU.
5. Regrid must run at the TOP of advance(), before zero_flux_registers.

### C++
6. No raw owning pointers — use `std::unique_ptr` or `GpuArray<T>`.
7. No axis-specific duplicate functions — one `template <Axis DIR>` only.
8. No scheme-selection `if` branches inside GPU kernels — dispatch at launch site.
9. No `virtual` in device-callable code — use CRTP or `std::variant`.
10. Concept constraints applied at every template boundary (kernel launch sites,
    factory). Never inside `__global__` kernels (CUDA limitation).

### CUDA
11. `cudaDeviceSynchronize()` forbidden in the advance loop.
12. Halo exchanges use `cudaMemcpyAsync` on the solver stream; TMA pipelines
    are a R6+ upgrade.
13. Kernel roofline target: ≥50% of peak memory bandwidth (Nsight Compute).

### Testing
14. Every new functor must have a `CPUSerial` unit test before any GPU kernel.
15. Sod shock tube must give bit-identical results for Axis::X, Axis::Y, Axis::Z.
16. T08 isentropic vortex convergence rate must remain ≥1.8 after every phase.
17. Mass conservation drift < 1e-12 over 20 steps (checked by t4 T04).

## Validation Gate Commands

```bash
cmake --build build -t ba       # all gates (t1–t26)
cmake --build build -t t3       # operators — T08 convergence rate
cmake --build build -t t4       # ns_solver — 28 sub-tests
cmake --build build -t t24      # CUDA graph (P8.6)
cmake --build build -t t25      # GPU vs CPU correctness (P9.1)
cmake --build build -t t26      # NSSolver GPU dispatch (P10-A3)
```

## Key References

- Pirozzoli (2010) — split-form compressible convective operator
- Chandrashekar (2013) — entropy-conservative flux
- Einfeldt et al. (1991) — positivity-preserving schemes
- Zhang & Shu (2010) — maximum-principle positivity floor
- Berger & Colella (1989) — AMR flux register correction
- Trias et al. (2014) — symmetry-preserving discretization
- Huang & Johnsen (2024) — consistent-conservative ACDI multiphase
- Del Rey Fernández et al. — SBP-SAT at AMR coarse-fine interfaces
```

---

## 2. `.claude/settings.json` for `to_refactor`

*Create this file at the repo root `.claude/settings.json` on the `to_refactor` branch.*

The existing `settings.local.json` (user-level permissions) stays unchanged.
This project-level file adds the refactor-specific permissions and hooks.

```json
{
  "permissions": {
    "allow": [
      "Bash(cmake*)",
      "Bash(git diff*)",
      "Bash(git log*)",
      "Bash(git show*)",
      "Bash(git status*)",
      "Bash(git add*)",
      "Bash(git commit*)",
      "Bash(git push origin to_refactor*)",
      "Bash(grep -rn*)",
      "Bash(find . -name*)",
      "Bash(nvcc --version)",
      "Bash(nvidia-smi*)"
    ],
    "deny": [
      "Bash(git push origin main*)",
      "Bash(git push origin to_debug*)",
      "Bash(git reset --hard*)",
      "Bash(git checkout to_debug -- *)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/pre-edit.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/post-edit.sh"
          }
        ]
      }
    ]
  }
}
```

---

## 3. Hook Scripts

*Create these at `.claude/hooks/` on the `to_refactor` branch.*
*Both scripts read tool input as JSON from stdin.*

### `.claude/hooks/pre-edit.sh`

```bash
#!/usr/bin/env bash
# PreToolUse hook: blocks edits to to_debug and warns on kernel edits without tests.
# Claude Code passes tool input as JSON on stdin.

input=$(cat)
file=$(python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('file_path') or d.get('path') or '')
" <<< "$input" 2>/dev/null)

# Block any path that could be on to_debug via worktree
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$branch" = "to_debug" ] || [ "$branch" = "main" ]; then
    echo "BLOCKED: edits are not permitted on branch '$branch'. Switch to to_refactor."
    exit 2
fi

# Warn if a GPU kernel is edited without a corresponding CPU unit test
if echo "$file" | grep -q "src/cuda/\|kernels/"; then
    scheme=$(basename "$file" .cu | sed 's/_kernel//')
    if [ ! -f "tests/unit/test_${scheme}.cpp" ] && \
       [ ! -f "tests/cuda/test_${scheme}.cu" ]; then
        echo "WARNING: no unit test found for '${scheme}'."
        echo "Write a CPUSerial unit test in tests/unit/test_${scheme}.cpp first (Rule 14)."
        # Non-fatal: prints warning but does not block (exit 0)
    fi
fi

exit 0
```

### `.claude/hooks/post-edit.sh`

```bash
#!/usr/bin/env bash
# PostToolUse hook: runs the fast operator gate (t3) after any edit to operators.cpp
# or include/physics/, to catch T08 convergence regressions immediately.

input=$(cat)
file=$(python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('file_path') or d.get('path') or '')
" <<< "$input" 2>/dev/null)

if echo "$file" | grep -qE "operators\.cpp|include/physics/|include/concepts\.hpp"; then
    echo "--- post-edit: running t3 (operator convergence gate) ---"
    cmake --build build -t t3 --quiet 2>&1 | tail -6
fi

exit 0
```

Make both executable after creation:
```bash
chmod +x .claude/hooks/pre-edit.sh .claude/hooks/post-edit.sh
```

---

## 4. Custom Slash Commands

*Create these files at `.claude/commands/` on the `to_refactor` branch.*

### `.claude/commands/phase.md`

```markdown
# /phase

Reports the status of each refactor phase and what is safe to start next.

## Usage
/phase

## Instructions

1. Check R0 (C++20): run `cmake -LA build | grep CXX_STANDARD` — must show 20.
2. Check R1 (concepts): search for `include/concepts.hpp`.
   If present, check it defines at least `RiemannFlux` and `EquationOfState`.
3. Check R2 (functors): search for `include/physics/`. If present, list structs found.
4. Check R3 (variant BC): grep `src/` for `std::variant` and `std::visit` in ghost fill.
5. Check R4 (backend tags): search for `include/execution.hpp` and `make_solver` factory.
6. Check R5 (instantiation matrix): search for `src/instantiation_matrix.cpp`.
7. Check R6 (mdspan): grep `cell_block.hpp` for `std::mdspan`.

Report each as: ✅ Done / 🔄 In progress / ❌ Not started.
State which phase is safe to start next and its entry condition.
```

### `.claude/commands/validate.md`

```markdown
# /validate

Runs the full validation gate suite and reports convergence rates.

## Usage
/validate [phase_number or test_target]

## Instructions

If no argument: run `cmake --build build -t ba` (all t1–t26 gates).

If a phase number is given (e.g. `/validate R2`):
- R0: verify `cmake -LA build | grep CMAKE_CXX_STANDARD` shows 20
- R1: `cmake --build build -t t3` — check T08 rate ≥ 1.8 (concept checks are compile-time)
- R2: `cmake --build build -t t3 && cmake --build build -t t4` — 15/15 and 28/28
- R3: `cmake --build build -t t4` — T16a/T16b (contact angle BC) must pass
- R4: `cmake --build build -t t25 && cmake --build build -t t26` — GPU parity
- R5: `cmake --build build -t t4` and new scheme-selection test if written
- R6: `cmake --build build -t t3` — T08 rate must remain ≥ 1.8

If a test target is given (e.g. `/validate t4`):
- Run `cmake --build build -t [target]` and show the full output.

Always report: pass/fail counts, T08 convergence rate, and any regression vs. baseline.
```

### `.claude/commands/migrate.md`

```markdown
# /migrate

Migrates one free function or module to the target three-layer architecture.

## Usage
/migrate [function_name or file] [layer: functor|concept|backend]

## Instructions

Given a target (e.g. `/migrate hllc_es_flux_t functor`):

### If layer = functor (Layer P)
1. Read the current implementation in `src/operators.cpp`.
2. Create `include/physics/[name].hpp` as a struct with:
   - `operator()` annotated `__host__ __device__`
   - `template <Axis DIR>` with `constexpr int IX=...` index rotation
   - Physics state as member variables only
   - The concept from `include/concepts.hpp` satisfied
3. Replace the original free function call sites with the functor `operator()`.
4. Run `/validate R2` to confirm no regression.

### If layer = concept (Layer C)
1. Read `include/concepts.hpp` (create it if absent).
2. Define the concept with: `requires` clause on `operator()` signature,
   and `static constexpr bool is_entropy_stable` compile-time property.
3. Add `static_assert(ConceptName<TargetType>)` at the first call site.
4. Run `/validate R1`.

### If layer = backend (Layer E)
1. Read `include/execution.hpp` (create if absent).
2. Add a tag struct (e.g. `struct CPUSerial{};`).
3. Write a `dispatch(CPUSerial, ...)` overload wrapping the existing CPU call.
4. Run `/validate R4`.

Do not touch the existing implementation until the new layer passes tests.
```

### `.claude/commands/review.md`

```markdown
# /review

Expert CFD + architecture review of a file or directory.

## Usage
/review [file or directory]

## Instructions

Analyse in this order:

### 1. Numerical correctness
- Is interior flux entropy-stable (HLLC-ES or Chandrashekar)? Flag if standard Roe.
- Is T08 convergence rate documented and ≥ 1.8?
- Does ghost fill use the same reconstruction as accumulate_face (BC consistency)?
- Is the positivity floor applied after every RK3 stage (CPU and GPU)?

### 2. Refactor compliance (target architecture)
- Are there `hllc_flux_x` / `_y` / `_z` axis duplicates? Flag as [HIGH].
- Are BC types dispatched via `if (bc_type == ...)` chains instead of `std::variant`? Flag.
- Are scheme-selection `if` branches inside `__global__` kernels? Flag as [CRITICAL].
- Are there `virtual` functions in device-callable structs? Flag as [HIGH].

### 3. C++ quality
- Raw owning pointers (`new`/`delete`)? Flag.
- Magic numbers (unnamed literals for stencil weights, NVAR, etc.)? Flag.
- Missing `constexpr` on pure compile-time constants? Flag.

### 4. CUDA correctness
- `cudaDeviceSynchronize()` in the advance loop? Flag as [CRITICAL].
- Synchronous `cudaMemcpy` in the hot path? Flag as [HIGH].

### Output format
[CRITICAL] — correctness or UB, blocks merge
[HIGH]     — architecture violation or performance regression
[MEDIUM]   — maintainability issue
[INFO]     — improvement suggestion

For each issue: file, line, problem, corrected snippet.
```

---

## 5. Why This Refactoring Enables Smoother Future Implementations

### 5.1 Adding a new flux scheme (e.g. Roe-EC, Ducros-modified Chandrashekar)

**Today (to_debug):** Requires editing `operators.cpp` in at least 4 places
(the free function, the `accumulate_face` dispatch, `undo_cf_one_face`, and
`cf_accum_one_face`), then verifying every AMR path uses the right variant.

**After R2 + R5 (functor + instantiation matrix):** Write one struct in
`include/physics/`, add two lines to `src/instantiation_matrix.cpp`, add one
`else-if` in the factory. The AMR paths automatically use the new type because
they are templated on the functor. Zero AMR code changes.

### 5.2 Adding a new boundary condition

**Today:** BC logic is embedded in ghost fill with `if (bc_type == ...)` chains
spread across `fill_same_level_ghost`, `fill_cf_ghosts`, and the GPU equivalent.

**After R3 (variant BC):** Write one struct satisfying `BoundaryCondition`,
add it to the `std::variant` type list. `std::visit` in the ghost fill
automatically dispatches to it on both CPU and GPU. One file, no other changes.

### 5.3 Porting to a new execution target (HIP/ROCm, SYCL, multi-GPU)

**Today:** The GPU path is deeply coupled to CUDA in `gpu_graph.cu`,
`gpu_rhs.cu`, `gpu_ghost_fill.cu`. Porting would require rewriting all three
in the new API with no clear boundary for what is physics and what is execution.

**After R4 (backend tag dispatch):** The physics functors (Layer P) are
`__host__ __device__` and execution-agnostic. A HIP backend is a new `GPUHip{}`
tag with HIP kernel wrappers. The physics layer is untouched. The factory
selects the right backend at startup. Estimated effort: the kernel wrappers only.

### 5.4 Adding a new EOS (stiffened gas variants, van der Waals, tabulated)

**Today:** EOS logic (`GAMMA`, `p_inf_a`, `p_inf_b`) is accessed via
`SolverConfig` fields threaded through function signatures. Each new EOS
adds parameters to `SolverConfig` and conditional branches in `prim_from_cons`.

**After R2 + R5:** Each EOS is a struct satisfying `EquationOfState`. The
instantiation matrix adds one row per EOS. `prim_from_cons` calls
`eos.pressure(rho, e)` — no conditionals.

### 5.5 Subcycling AMR (future roadmap item)

Sub-level time stepping requires advancing fine leaves at a smaller `dt` than
coarse leaves, with flux registers accumulating over multiple fine steps.
This is straightforward to express when the time advance is templated:

```cpp
template <RiemannFlux Flux, EquationOfState EOS, typename Backend>
void advance_level(BlockTree& tree, int level, double dt_fine, Backend);
```

**Today:** `NSSolver::advance()` is a monolithic function that iterates all
leaves at one `dt`. Threading level-aware subcycling through it requires
significant restructuring.

**After R4:** The backend tag dispatch means `advance_level<..., GPUCuda>`
and `advance_level<..., CPUSerial>` are separate overloads. The subcycling
logic lives in the time-loop controller, not tangled into the physics.

### 5.6 Neural SGS and ONNX inference (P4.6 on the roadmap)

A neural SGS model is a functor satisfying the same `SGSModel` concept as
`SmagorinskyModel`. It is selected via the instantiation matrix. No changes
to the time loop, RHS computation, or GPU graph.

### Summary table

| Future feature | Without refactor | After R2–R5 |
|---|---|---|
| New flux scheme | 4+ edit sites in operators.cpp | 1 struct + 2 matrix lines |
| New BC | 3+ if/else chains | 1 struct + 1 variant type |
| New EOS | SolverConfig param + branches | 1 struct + 1 matrix row |
| New GPU backend | Rewrite all .cu files | 1 tag + kernel wrappers |
| AMR subcycling | Restructure advance() | Template the time loop |
| Neural SGS | Ad-hoc integration | Concept-satisfying functor |

The refactor is a one-time investment. After R0–R5 every subsequent feature
follows the same three-step pattern: write a functor, satisfy its concept,
add it to the matrix. The gate suite (`ba`) validates correctness throughout.
