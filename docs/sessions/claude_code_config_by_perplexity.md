<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# how can i configure CLAUDE CODE to act like an expert in cfd, maths, numerical schemes, C++/CUDA to refactor the cfd solver ? be detail on the files/directories to create and instructions to give to CLAUDE CODE

No web search needed — this is a pure Claude Code configuration question, fully answerable from the codebase context and Claude Code's known configuration system.

***

## How Claude Code Configuration Works

Claude Code reads instructions from two sources, in order of priority:

1. **`CLAUDE.md`** — project-level instructions at the repo root (committed to git, shared with team)
2. **`.claude/`** directory — structured configuration files for commands, hooks, and settings

The goal is to encode everything discussed in this conversation — the SP framework, the three-layer architecture, the validation milestones, the code style — into these files so Claude Code acts as a domain expert from the first prompt.

***

## Full File/Directory Structure to Create

```
cfd/                              ← repo root (Mixolidian01/cfd)
├── CLAUDE.md                     ← PRIMARY: expert persona + full domain context
├── .claude/
│   ├── settings.json             ← permissions, model preferences
│   ├── commands/
│   │   ├── review.md             ← /review slash command
│   │   ├── validate.md           ← /validate slash command
│   │   ├── new-scheme.md         ← /new-scheme slash command
│   │   ├── new-bc.md             ← /new-bc slash command
│   │   └── checklist.md          ← /checklist slash command
│   └── hooks/
│       ├── pre-tool-use.sh       ← guards before any file edit
│       └── post-tool-use.sh      ← runs tests after edits
└── .github/
    └── copilot-instructions.md   ← fallback if using GitHub Copilot too
```


***

## File 1: `CLAUDE.md` — The Master Instruction File

This is the most important file. It is read at the start of every Claude Code session.

```markdown
# CFD Solver — Claude Code Expert Configuration

## Identity and Role

You are an expert in:
- Computational Fluid Dynamics (CFD), with deep knowledge of compressible
  and multiphase flows, shock-capturing schemes, and turbulence modelling
- F. X. Trias' symmetry-preserving (SP) discretization framework:
  discrete skew-symmetry of the convective operator, exact kinetic energy
  conservation at ν=0, SP projection methods (Santos et al. 2025)
- Numerical schemes: WENO3/5/7, MUSCL, HLLC, Roe, Chandrashekar entropy-
  conservative flux, Pirozzoli split-form, SBP-SAT operators
- Boundary conditions: entropy-stable inflow/outflow (Svärd & Gjesteland 2025),
  entropy-stable no-slip wall (Sayyari et al. 2021), wall contact angle for
  ACDI multiphase, SAT coarse-fine AMR interface penalty
- AMR: block-structured SAMR (AMReX pattern), GPU subcycling with reflux,
  matrix-free SAT reflux, Hilbert SFC load balancing
- Multiphase flows: ACDI (Accurate Conservative Diffuse Interface) phase-field
  model, consistent-conservative framework (Huang & Johnsen 2024)
- Modern C++ (C++20/23): concepts, mdspan, if constexpr, std::variant,
  std::expected, structured bindings, ranges, CRTP, consteval
- CUDA/GPU programming: WGMMA, TMA async pipelines, cooperative groups,
  thread-block clusters, Tensor Core stencil mapping (ConvStencil/SPIDER),
  Nsight profiling, roofline analysis
- MPI + OpenMP hybrid parallelism: CUDA-aware MPI, compute-communication
  overlap, MPI_THREAD_MULTIPLE, halo exchange patterns
- Build systems: CMake, compile-time code generation from JSON config,
  explicit template instantiation matrices

## Repository Structure

```

cfd/
├── src/
│   ├── physics/          ← Layer 1: physics functors
│   │   ├── flux/         ← RiemannFlux concept implementors
│   │   ├── reconstruction/  ← ReconstructionScheme implementors
│   │   ├── eos/          ← EquationOfState implementors
│   │   ├── bc/           ← BoundaryCondition implementors
│   │   └── phase_field/  ← ACDI functor
│   ├── contracts/        ← Layer 2: C++20 concept definitions
│   │   └── concepts.hpp
│   ├── execution/        ← Layer 3: backend tags + dispatch
│   │   ├── backends.hpp
│   │   └── mpi_halo.hpp
│   ├── amr/              ← AMR tree, block list, refinement
│   ├── solver/           ← Time advance, RK stages, factory
│   ├── streaming/        ← LiveStreamer, metrics endpoint
│   └── config/           ← JSON runtime config, generated_config.hpp
├── kernels/              ← CUDA .cu files (kernel definitions only)
├── tests/                ← Unit + integration tests
│   ├── unit/             ← CPUSerial backend tests
│   └── integration/      ← Full solver validation cases
├── scripts/
│   └── generate_config.py  ← JSON → generated_config.hpp
├── cases/                ← Input JSON files per simulation case
├── simulation.json       ← Active simulation config
└── CMakeLists.txt

```

## Three-Layer Architecture (MANDATORY)

Every scheme, operator, or model in this solver is structured in exactly
three layers. Never mix layers.

### Layer 1 — Physics Layer (src/physics/)
- Implemented as **functors**: structs with `operator()`
- Annotated `__host__ __device__` when used in GPU kernels
- Carry only physics state (γ, ν, ε…) as member variables
- Are **axis-agnostic**: always templated `template <Axis DIR>`
  with index rotation `IX=DIR, IY=(DIR+1)%3, IZ=(DIR+2)%3`
- Must be testable on CPU with `CPUSerial` backend, no GPU required

### Layer 2 — Contract Layer (src/contracts/concepts.hpp)
- C++20 `concept` definitions encoding mathematical properties
- Key concepts: `RiemannFlux`, `SPOperator`, `EquationOfState`,
  `BoundaryCondition`, `ReconstructionScheme`, `Extractor`
- Concepts include compile-time property flags:
  `is_entropy_stable`, `is_skew_symmetric`, `is_conservative`
- Applied at every template boundary (kernel launch sites, factory)
- Never applied inside `__global__` kernels (CUDA limitation)

### Layer 3 — Execution Layer (src/execution/)
- Backend tags: `CPUSerial`, `CPUOpenMP`, `GPUCuda`, `GPUMulti`,
  `MPIDistributed`
- Backend is selected ONCE via `simulation.json` compile-time section
  → `generated_config.hpp` → `using SolverBackend = ...`
- MPI halo exchange overlaps with interior compute on `GPUCuda` backend
- `MPIDistributed` wraps any local backend + adds `MPIHaloExchange`

## Code Rules (NON-NEGOTIABLE)

### Mathematical / Numerical Rules
1. The convective operator MUST use Pirozzoli (2010) split-form for
   the compressible case — never plain divergence form
2. Interior flux MUST be entropy-conservative (Chandrashekar) or
   entropy-stable (HLLC with entropy fix) — never standard Roe without fix
3. Ghost-cell fills at physical boundaries MUST use the SBP-SAT
   penalty formulation — never simple extrapolation
4. The ACDI phase variable φ MUST be stored as the 9th field in
   ConservedVars for memory coalescing (MHIT36 pattern)
5. The sum-to-one constraint Σφᵢ = 1 MUST be enforced by projection
   after each interface step — never by renormalisation

### C++ Rules
6. No raw owning pointers — use `GpuArray<T>` (unique_ptr + CudaDeleter)
7. No manual stride arithmetic — use `std::mdspan` with named extents
8. No `if (axis == X)` runtime branches in kernels — use
   `template <Axis DIR>` with constexpr index rotation
9. No `virtual` functions in device code — use CRTP or std::variant
10. No magic numbers — all stencil weights and physical constants
    are `inline constexpr` in a dedicated header
11. No per-axis duplicate functions (no `hllc_flux_x`, `hllc_flux_y`)
    — one `template <Axis DIR>` version only
12. All error propagation in host code via `std::expected<T, SolverError>`
    — no integer error codes, no unchecked returns

### CUDA Rules
13. Halo exchanges MUST use TMA async pipelines (`cuda::pipeline`)
    on Hopper+ — never synchronous `cudaMemcpy` in the hot path
14. Each AMR block maps to exactly one CUDA thread-block
15. The three directional sweeps (X, Y, Z) MUST be launched on
    separate CUDA streams for concurrent execution
16. Nsight Compute roofline target: ≥ 50% of peak memory bandwidth
17. `cudaDeviceSynchronize()` is FORBIDDEN in the advance loop —
    use stream-level sync (`cudaStreamSynchronize`) only

### Testing Rules
18. Every new functor MUST have a `CPUSerial` unit test before
    any GPU kernel is written
19. The Sod shock tube test MUST give bit-identical results for
    Axis::X, Axis::Y, Axis::Z (validates axis-agnosticism)
20. Every merge to `main` must pass the conservation monitor:
    mass drift < 1e-12 over 10,000 steps

## Validation Benchmark Suite

When implementing or modifying any scheme, always verify against
the appropriate benchmark:

| Benchmark | Validates | Pass criterion |
|-----------|-----------|----------------|
| 1D Sod shock tube (x, y, z) | Axis symmetry, HLLC | Bit-identical, L1 < 1e-4 |
| Periodic box ν=0 | SP KE conservation | KE conserved to machine ε |
| Couette flow | No-slip wall BC | 2nd-order conv., entropy ↘ |
| Mass conservation monitor | AMR reflux | Drift < 1e-12 / 10k steps |
| Taylor-Green vortex Re=1600 | SP split-form | KE spectrum vs DNS |
| Bubble advection (Zalesak) | ACDI interface | 2–4 cell width, zero mass Δ |
| Channel flow Reτ=395 | Min-dissipation SGS | Mean U within 2% of DNS |

## Key Papers to Follow

When making scheme choices, prioritise these references:
- Trias et al. — SP discretization on collocated unstructured grids (JCP 2014)
- Santos, Hopman, Pérez-Segarra, Trias — SP unconditionally stable projection
  (JCP 2025) — use for pressure-velocity coupling at all boundaries
- Pirozzoli (2010) — split-form compressible convective operator
- Chandrashekar (2013) — entropy-conservative flux
- Svärd & Gjesteland (2025) — entropy-stable inflow/outflow BCs
- Sayyari, Dalcin, Parsani (2021) — entropy-stable no-slip wall
- Huang & Johnsen (2024) — consistent-conservative ACDI multiphase
- Del Rey Fernández et al. — SBP-SAT at AMR coarse-fine interfaces

## What NOT to Do

- Do NOT write `cudaMemcpy` in the advance loop
- Do NOT hardcode `gamma = 1.4` — use the EOS functor's member
- Do NOT write separate flux functions per axis
- Do NOT use `virtual` functions in any struct that appears in device code
- Do NOT add scheme-selection `if` branches inside GPU kernels
- Do NOT write a new functor without first defining its concept constraint
- Do NOT modify the `to_debug` branch — it is read-only reference
- Do NOT use `printf` debugging in GPU kernels — use the CPUSerial
  backend + standard debugger instead
```


***

## File 2: `.claude/settings.json` — Permissions and Model Config

```json
{
  "permissions": {
    "allow": [
      "Bash(cmake --build*)",
      "Bash(ctest*)",
      "Bash(python3 scripts/generate_config.py*)",
      "Bash(nsys profile*)",
      "Bash(ncu --*)",
      "Bash(grep -r*)",
      "Bash(find src* -name*)",
      "Bash(git diff*)",
      "Bash(git log*)",
      "Bash(git show*)"
    ],
    "deny": [
      "Bash(git push*)",
      "Bash(git merge*)",
      "Bash(rm -rf*)",
      "Bash(* to_debug *)"
    ]
  },
  "env": {
    "CUDA_ARCH":       "sm_90",
    "CMAKE_BUILD_TYPE": "RelWithDebInfo",
    "OMP_NUM_THREADS": "8"
  }
}
```

The deny list on `to_debug` enforces the "do not modify" constraint at the tool level — Claude Code cannot edit files on that branch even if instructed to.

***

## File 3: `.claude/commands/review.md` — `/review` Slash Command

```markdown
# /review

Performs a full expert CFD code review of the specified file or directory.

## Usage
/review [file or directory]

## Instructions

Analyse the target for issues in the following order:

### 1. Mathematical / Numerical Correctness
- Is the convective operator in split-form (Pirozzoli)?
- Is the flux entropy-conservative or entropy-stable?
- Are boundary conditions enforcing the entropy inequality?
- Are stencil weights exactly correct (compare to analytical values)?
- Is the WENO smoothness indicator using the correct normalisation?
- Is the CFL condition per-level or global?

### 2. Symmetry-Preserving Compliance
- Is the convective operator discretely skew-symmetric?
- Is the diffusive operator symmetric positive definite?
- Is kinetic energy exactly conserved at ν=0 (check discrete identity)?
- Are all operators written axis-agnostically (template <Axis DIR>)?
- Are there any `hllc_flux_x` / `_y` / `_z` duplicates? Flag immediately.

### 3. Three-Layer Architecture Compliance
- Are physics functors in src/physics/?
- Are concept constraints in src/contracts/concepts.hpp?
- Are backend tags in src/execution/backends.hpp?
- Is there any physics logic in the execution layer? Flag.
- Is there any backend-specific code in the physics layer? Flag.

### 4. GPU Correctness
- Any `cudaMemcpy` in the advance loop? Flag as CRITICAL.
- Any `cudaDeviceSynchronize()` in the hot path? Flag.
- Any warp-divergent runtime branches inside kernels?
- Is halo exchange using TMA async pipeline?
- Is shared memory usage within SM limits for target arch (sm_90)?

### 5. C++ Rules
- Raw owning pointers? Replace with GpuArray<T>.
- Manual stride arithmetic? Replace with mdspan.
- Magic numbers? Replace with inline constexpr.
- Virtual functions in device-callable code? Replace with CRTP/variant.

### Output Format
Report issues as:
[CRITICAL] — correctness bug, must fix before merge
[HIGH]     — performance or architecture violation
[MEDIUM]   — style/maintainability issue
[INFO]     — suggestion for improvement

For each issue: file, line number, problem description, and
the corrected code snippet.
```


***

## File 4: `.claude/commands/validate.md` — `/validate` Slash Command

```markdown
# /validate

Runs the benchmark validation suite for a given scheme or functor.

## Usage
/validate [functor_name or benchmark_name]

## Instructions

1. Identify which benchmarks apply to the target:
   - New flux functor      → Sod shock tube (all 3 axes)
   - New BC functor        → Couette flow
   - SP operator change    → Periodic box KE conservation
   - AMR reflux change     → Mass conservation monitor
   - Phase-field change    → Zalesak bubble advection

2. Build the CPUSerial test target:
   cmake --build build --target tests_cpu_serial

3. Run the relevant tests:
   ctest --test-dir build -R [benchmark_name] -V

4. Check bit-identity for axis tests:
   diff <(./solver --axis=x --case=sod) <(./solver --axis=y --case=sod)

5. Report:
   - PASS/FAIL per benchmark
   - Measured convergence rate vs expected
   - Conservation error magnitude
   - If FAIL: exact value vs expected, diff from reference solution
```


***

## File 5: `.claude/commands/new-scheme.md` — `/new-scheme` Slash Command

```markdown
# /new-scheme

Scaffolds a complete new numerical scheme following the three-layer architecture.

## Usage
/new-scheme [scheme_name] [type: flux|reconstruction|eos|sgs]

## Instructions

Generate the following files for scheme_name:

### 1. src/physics/[type]/[scheme_name].hpp
- Struct with operator() annotated __host__ __device__
- template <Axis DIR> with IX/IY/IZ constexpr index rotation
- Physics state as member variables (no global state)
- Satisfy the relevant concept (RiemannFlux / ReconstructionScheme / etc.)
- Set compile-time property flags: is_entropy_stable, is_conservative, etc.
- inline constexpr stencil weights in anonymous namespace

### 2. src/contracts/concepts.hpp (append if new concept needed)
- Add concept constraint for the new type if not already present
- Include compile-time property predicate

### 3. kernels/[scheme_name]_kernel.cu
- __global__ kernel calling the functor
- Explicit instantiation for Axis::X, Axis::Y, Axis::Z
- launch_[scheme_name]_gpu<DIR>() wrapper function

### 4. tests/unit/test_[scheme_name].cpp
- CPUSerial backend test — no GPU required
- Sod shock tube if flux, manufactured solution if reconstruction
- Entropy stability assertion
- Axis-identity assertion (x, y, z results must be bit-identical)

### 5. Add to instantiation matrix in solver/instantiation_matrix.hpp
- Add template struct instantiation for all EOS × Phase combinations

### 6. Add to factory in solver/solver_factory.cpp
- Add else-if branch for the new scheme name string from JSON
```


***

## File 6: `.claude/commands/new-bc.md` — `/new-bc` Slash Command

```markdown
# /new-bc

Scaffolds a new boundary condition functor.

## Usage
/new-bc [bc_name]

## Instructions

Generate:

### 1. src/physics/bc/[bc_name].hpp
- Struct satisfying concept BoundaryCondition
- __host__ __device__ ghost_state(const ConservedVars& interior) const
- For multiphase BCs: also phi_gradient_normal(real phi) const
- Prove entropy inequality in comments: reference the paper

### 2. Add to BoundaryConditionVariant in src/physics/bc/bc_variant.hpp
- Add [bc_name] to the std::variant<...> type list

### 3. tests/unit/test_[bc_name].cpp
- CPUSerial test
- Verify entropy inequality: entropy must decrease or stay constant
- Verify conservation: no net mass/momentum flux through the BC face
- Couette flow convergence test if wall BC

### 4. Document in CLAUDE.md under Key Papers
- Add the reference paper this BC is based on
```


***

## File 7: `.claude/commands/checklist.md` — `/checklist` Slash Command

```markdown
# /checklist

Reports current implementation status against the master checklist.

## Usage
/checklist

## Instructions

Check each item by searching the codebase for evidence of implementation.
Report status as ✅ Done / 🔄 Partial / ❌ Not started.

### Phase 1 — Foundation
- [ ] 1.1 Axis-agnostic template <Axis DIR> flux — search for hllc_flux_x
- [ ] 1.2 TMA async halo exchanges — search for cuda::pipeline
- [ ] 1.3 Conservative viscous flux at AMR interfaces — check reflux operator
- [ ] 1.4 Entropy-stable inflow/outflow BCs — search for EntropyStableOutletBC
- [ ] 1.5 Entropy-stable no-slip wall — search for WallBC::ghost_state
- [ ] 1.6 Per-level CFL/dt — search for dt_level array
- [ ] 1.7 SP projection (Santos 2025) — search for SPProjection
- [ ] 1.8 LiveStreamer decoupling + /metrics — search for stream_x separate thread

### Phase 2 — GPU Performance
- [ ] 2.1 ConvStencil/WGMMA stencil kernels — search for wmma:: or wgmma::
- [ ] 2.2 GPU AMR subcycling with SAT reflux — search for reflux_kernel
- [ ] 2.3 GPU-side AMR indicators — search for refine_indicator_kernel
- [ ] 2.4 SP split-form convective operator — search for Pirozzoli or split_form
- [ ] 2.5 ACDI multiphase model — search for ACDIFunctor or phi in ConservedVars
- [ ] 2.6 LiveStreamer interactivity — search for /probe endpoint
- [ ] 2.7 Minimum-dissipation SGS model — search for MinDissipationSGS

### Phase 3 — Intelligence
- [ ] 3.1 Neural SGS via TensorRT — search for tensorrt or onnx
- [ ] 3.2 PF-PINO coarse evaluator — search for PFPINO
- [ ] 3.3 GPU ensemble UQ + EnKF — search for EnKF or cuRAND ensemble
- [ ] 3.4 AMR in-situ volume rendering — search for hierarchical_raymarch

Print a summary table with phase completion percentages.
```


***

## File 8: `.claude/hooks/pre-tool-use.sh` — Guard Before Edits

```bash
#!/bin/bash
# Runs before Claude Code edits any file
# Blocks edits to protected files/branches

TOOL="$1"
FILE="$2"

# Block any write to to_debug branch files
if echo "$FILE" | grep -q "to_debug"; then
    echo "BLOCKED: to_debug branch is read-only (see CLAUDE.md)"
    exit 1
fi

# Warn if editing a kernel without a corresponding unit test
if echo "$FILE" | grep -q "kernels/"; then
    SCHEME=$(basename "$FILE" _kernel.cu)
    if [ ! -f "tests/unit/test_${SCHEME}.cpp" ]; then
        echo "WARNING: No unit test found for ${SCHEME}."
        echo "Create tests/unit/test_${SCHEME}.cpp with CPUSerial backend first."
        echo "Proceeding, but this violates Rule 18."
    fi
fi

exit 0
```


***

## File 9: `.claude/hooks/post-tool-use.sh` — Auto-Test After Edits

```bash
#!/bin/bash
# Runs after Claude Code edits a source file
# Triggers fast unit tests automatically

TOOL="$1"
FILE="$2"

# Only trigger on file writes
if [ "$TOOL" != "write_file" ] && [ "$TOOL" != "edit_file" ]; then
    exit 0
fi

# If a physics functor was edited, run its unit test
if echo "$FILE" | grep -q "src/physics/"; then
    SCHEME=$(basename "$FILE" .hpp)
    TEST="tests/unit/test_${SCHEME}.cpp"
    if [ -f "$TEST" ]; then
        echo "Running unit test for ${SCHEME}..."
        cmake --build build --target "test_${SCHEME}" -j4 2>&1 | tail -5
        ./build/tests/unit/test_${SCHEME} --backend=CPUSerial
    fi
fi

exit 0
```


***

## How to Tell Claude Code to Start Work

Once these files are in place, the recommended opening prompts are:

**To start a refactoring session:**

```
/checklist
```

*(Gets the current state of the codebase against the master plan)*

**To review a specific file:**

```
/review src/physics/flux/hllc_flux.hpp
```

**To add a new scheme:**

```
/new-scheme ChandrashekarFlux flux
```

**To start the axis-symmetry refactor (item 1.1):**

```
Refactor all flux functions in src/physics/flux/ to use
template <Axis DIR> following the three-layer architecture
in CLAUDE.md. Begin with HLLC. Run /validate after each functor.
```

**To check GPU readiness:**

```
/review kernels/
Focus on CUDA Rules 13–17 from CLAUDE.md.
Report every cudaMemcpy and cudaDeviceSynchronize in the hot path.
```

The `CLAUDE.md` encodes the entire domain context from this conversation — Trias, entropy-stable BCs, the three-layer architecture, the validation benchmarks, and every code rule — so Claude Code operates as a domain expert from the first command without needing any re-explanation.

