# Known Limitations & Tech Debt

Deliberate deferrals, known approximations, and things that look like bugs but aren't.
Check this before treating something as a bug to fix.

---

## Numerical / Physics

### BVH sign determination uses pseudo-normal, not winding number
**File:** `include/cuda/gpu_bvh.cuh`, `bvh_sdf()`
**What:** Sign of the SDF is determined by dot-product of the displacement vector with the closest-triangle face normal. Robust for smooth convex bodies; can mis-sign near sharp edges or vertices of arbitrary (non-convex, non-watertight) STL meshes.
**Why deferred:** Generalised winding number requires an O(n log n) GPU BVH traversal that is significantly more complex. The pseudo-normal test is correct for the smooth geometries in the current test suite.
**Risk:** IBM will silently mis-classify cells near sharp concavities in complex STL files. No runtime warning is emitted.

### IBM SolidFill uses linear scan — O(n_leaves × NCELL)
**File:** `src/cuda/gpu_ibm.cu`, `GpuIbmList::build()`
**What:** Finding SOLID cells iterates every cell of every leaf on CPU. A TODO comment marks the location.
**Why deferred:** Acceptable for current problem sizes. For large AMR trees (>10k leaves) with many SOLID cells, a spatial hash indexed by cell-centre coordinate would reduce build time from O(n_leaves × NCELL) to O(n_solid).
**Risk:** `build()` latency increases linearly with leaf count on regrid.

### D0.5 shared-memory tiling reverted
**File:** `src/cuda/gpu_rhs.cu` — `k_rhs_conv_tiled` exists but `exec()` calls `k_rhs_conv`
**What:** The tiled kernel was implemented and verified correct but regressed latency 1.5× (3.80 ms vs 2.51 ms). Root cause: `k_rhs_conv` is latency-bound at 4.2% peak BW — L2 already serves the stencil re-reads; shmem addresses the wrong bottleneck.
**Why kept:** Reference for future D8-class optimisations. Do not re-enable unless profiling shows BW has become the bottleneck (e.g., after moving to H100 or increasing block count significantly).
**Current BW:** 4.2% of peak (RTX 3070 Laptop, 448 GB/s). Target ≥ 55% remains unmet for `k_rhs_conv`.

### Adjoint uses frozen TENO7-A weights
**Files:** `include/physics/adjoint_teno7.hpp`, `include/physics/adjoint_hllc.hpp`
**What:** `adjoint_rhs` differentiates through HLLC-ES and TENO7-A with **frozen smoothness-indicator weights** (weights treated as constants, not differentiated). This is a frozen-lambda / one-level adjoint — correct for shape optimisation and data assimilation where the flow sensitivity is the goal, but not exact for full second-order adjoints.
**Why:** Full differentiation through TENO7-A weight computation is significantly more complex and was not required for the D10 dot-product gate.
**Risk:** Gradient accuracy degrades near shocks where weight sensitivity is non-negligible.

### D7 WMLES gate is unit-only, not channel DNS
**File:** `tests/cuda/test_t35_wmles_gpu.cu`
**What:** t35 verifies Newton inversion accuracy, ghost-cell encoding, and Reichardt log-law behaviour. It does **not** run a full turbulent channel at Re_τ = 395 and compare against DNS data (the original CLAUDE.md D7 gate).
**Why:** Full channel LES requires a multi-block rectangular domain and ≥ 10k steps — impractical as an automated gate. The unit test is sufficient for code correctness; DNS validation is a separate integration test.

---

## Infrastructure

### CUDA-aware MPI inactive on WSL2/OpenMPI
**File:** `src/cuda/gpu_mpi_halo.cu`
**What:** `#ifdef MPIX_CUDA_AWARE_SUPPORT` is not defined in the WSL2 + OpenMPI build. The CPU-staging fallback is always active: GPU pack → D2H → MPI_Isend → MPI_Recv → H2D → GPU unpack. The CUDA-aware path (MPI_Isend directly on GPU buffer) is compiled and correct but untested.
**Why:** WSL2 OpenMPI does not expose `MPIX_CUDA_AWARE_SUPPORT`. Must be validated on a cluster with UCX + NVLink.
**Consequence:** t30 measures a 3× buffer reduction (face-only vs full-block) but not the full D2H/H2D elimination that the CUDA-aware path provides.

### t28 and t30 require MPI build
**CMakeLists:** `add_nvcc_mpi_gate`
**What:** These targets are only built when `HAVE_MPI` is set. They cannot run in single-process mode and are excluded from `ba`.
**Action needed to run:** `cmake -DHAVE_MPI=ON ..` and `mpirun -n 2 ./t28_gpu_mpi_halo`.

### Python bindings (D11) GPU path guarded
**File:** `src/python/cfd_module.cpp`
**What:** The pybind11 `advance()` binding calls the CPU `NSSolver` path. A size-mismatch guard prevents calling the GPU path directly from Python. JAX `custom_vjp` wires through the CPU adjoint.
**Why:** Zero-copy NumPy↔GPU transfer requires pinned memory registration that was not implemented in D11.
**Risk:** Python simulations run at CPU speed. GPU acceleration from Python requires a future D11+ extension.

### dev_log.md stops at D7
**File:** `docs/dev_log.md`
**What:** Detailed narrative entries exist for D0–D7 only. D9, D10, D11, G1–G6, IBM, rectangular domain, and simulate-sync are not in the log — only their commit messages and `docs/dev_phases.md` record them.
**Action:** Append entries to `dev_log.md` if post-mortem narrative is needed for any of these phases.

---

## Permanently Dropped

### D8 — H100 TMA + thread-block clusters
**Never implement.** Dropped by explicit user decision. Not a deferral — do not suggest or reference it.

---

## Key Formulas (quick reference)

```
Cell flat index:      k*NB2² + j*NB2 + i   (i fastest, NB2=12)
Interior range:       i,j,k ∈ [ilo=2, ihi=9]
Q device layout:      d_Q[v * NCELL + flat]   (v=0..4: ρ,ρu,ρv,ρw,E)
IBM image point:      I = G − 2·sdf·n_outward  (sdf<0 for SOLID → I into fluid)
IBM solid clamp:      sdf_eff = max(sdf, −1.5·h)  (prevents deep SOLID overshoot)
Trilinear weight:     8-cell stencil at image point I; weights from (1−fx)(1−fy)(1−fz) etc.
SSP-RK3 weights:      s1: α=1,β=1;  s2: α=0.75,β=0.25;  s3: α=1/3,β=2/3
```
