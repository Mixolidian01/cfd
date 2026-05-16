# CFD Solver

High-order compressible CFD solver with GPU acceleration, adaptive mesh refinement, and multiphase support.

## Features

- **SSP-RK3** time integration with CFL-limited timestep
- **WENO5-Z** shock-capturing reconstruction, **HLLC-ES** entropy-stable flux
- **Octree AMR** with Berger-Colella flux correction at coarse/fine interfaces
- **GPU backend** via CUDA Graphs (full SSP-RK3 loop captured and replayed)
- **Smagorinsky SGS** turbulence model (operator-split, GPU-native)
- **MPI domain decomposition** with GPU halo exchange (D2H → MPI → H2D)
- **ACDI compressible multiphase** (Allaire 5-equation + stiffened-gas EOS)
- **SBP-SAT** penalty at AMR coarse/fine interfaces
- In-situ browser live feed, VTK and checkpoint I/O

## Build

```bash
cmake -S . -B build -DCMAKE_CXX_STANDARD=20
cmake --build build -j$(nproc)
```

Optional flags: `-DCMAKE_BUILD_TYPE=Debug`, `-DLZ4_LIBRARY=...`, `-DZFP_DIR=...`

## Run all gate tests

```bash
cmake --build build -t ba          # CPU gate suite (32 tests)
cmake --build build -t t24         # CUDA Graph re-capture
cmake --build build -t t25         # GPU vs CPU correctness
cmake --build build -t t26         # NSSolver GPU dispatch
cmake --build build -t t27         # Smagorinsky SGS kernel
cmake --build build -t t28         # MPI + GPU halo exchange
```

## Repository layout

```
apps/          Driver programs (simulate.cpp, JSON configs)
include/       Public headers (mesh, schemes, solver, cuda, physics, …)
src/           Implementations (mirrors include/ layout)
tests/         Gate tests (linalg, mesh, schemes, solver, cuda, mpi, bench)
docs/          Project documentation
  ROADMAP.md       Development roadmap
  TODO.md          Outstanding work items
  DOCUMENTATION.md Full technical reference (physics, numerics, API)
  to_develop.md    Configuration spec for the to_develop GPU branch
  dev/             Developer notes (bug rules, …)
  plans/           Implementation plans (per session)
  specs/           Design specifications
  sessions/        AI session logs (Perplexity, early Claude drafts)
```

## Documentation

See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for the full technical reference
covering physics, mathematical schemes, and the developer extension API.

## Branches

| Branch | Purpose |
|---|---|
| `to_debug`    | Correctness reference — do not modify |
| `to_refactor` | Architecture cleanup (R0–R6 complete) |
| `to_develop`  | State-of-the-art GPU development (see `docs/to_develop.md`) |
