# Known Limitations & Tech Debt

Deliberate deferrals, known approximations, and things that look like bugs but aren't.
Check this before treating something as a bug to fix.

---

## Numerical / Physics

### BVH sign determination — generalized winding number ✅ DONE
**File:** `include/cuda/gpu_bvh.cuh`, `bvh_sdf()`
**What:** Sign of the SDF is now determined by the generalized winding number (Van Oosterom & Strackee 1983), summing solid angles over all mesh triangles. Robust for non-convex and non-watertight STL meshes.
**Note:** O(n_tris) per cell query — suitable for small meshes (< ~2k tris). For large STL files, a BVH-accelerated winding-number traversal would reduce cost.
**Gate:** W5 (torus) in t49 verifies correct SOLID/FLUID classification for a non-convex geometry.

### IBM SolidFill uses linear scan — O(n_leaves × NCELL)
**File:** `src/cuda/gpu_ibm.cu`, `GpuIbmList::build()`
**What:** Finding SOLID cells iterates every cell of every leaf on CPU. A TODO comment marks the location.
**Why deferred:** Acceptable for current problem sizes. For large AMR trees (>10k leaves) with many SOLID cells, a spatial hash indexed by cell-centre coordinate would reduce build time from O(n_leaves × NCELL) to O(n_solid).
**Risk:** `build()` latency increases linearly with leaf count on regrid.

### D0.5 shared-memory tiling reverted
**File:** `src/cuda/gpu_rhs.cu` — `k_rhs_conv_tiled` exists but `exec()` calls `k_rhs_conv_teno`
**What:** The tiled kernel was implemented and verified correct but regressed latency 1.5× (3.80 ms vs 2.51 ms). Root cause: `k_rhs_conv_teno` is FP64-compute-bound at 4.2% peak BW — L2 already serves the stencil re-reads; shmem addresses the wrong bottleneck.
**Why kept:** Reference for future optimisations. Do not re-enable unless profiling shows BW has become the bottleneck (e.g., after moving to H100 or increasing block count significantly).
**WENO5Z BW:** 4.2% of peak (RTX 3070 Laptop, 448 GB/s) — measured at D0.5 before D3 added TENO7A. With TENO7A (current default), BW drops to 0.2% because TENO7A is 25× more FP64-compute-intensive. Target ≥ 55% is unachievable for either scheme on this GPU (see below).

### k_rhs_conv roofline gap is a hardware mismatch, not a software gap
**File:** `src/cuda/gpu_rhs.cu` — `k_rhs_conv_cell` exists but `exec()` calls `k_rhs_conv_teno`
**What:** The D0 baseline 2.51 ms/stage figure was measured with WENO5Z (before D3). With TENO7A (current default), the measured stage time is ~63 ms on a 512-leaf tree — 25× slower than WENO5Z due to ~25× more FLOPs per face (Roe decomposition + 7-point char decomposition vs simple scalar WENO). `k_rhs_conv_teno<true>` is FP64-compute-bound at near-peak for the RTX 3070 Laptop (~200 GFLOPS FP64). The 0.2% DRAM BW figure is not a software deficiency — the kernel simply does a lot of arithmetic per byte fetched. The ≥55% BW roofline target was written for A100/H100 (high FP64 throughput + high BW) where TENO7A would become BW-limited.
**Cell-based kernel attempt:** `k_rhs_conv_cell<USE_TENO7>` was implemented and verified correct (t31, t37 pass) but was 25× slower than `k_rhs_conv_teno`. Root cause: inlining `gpu_teno7_face` 6 times per thread requires ~120+ doubles of local state per call (Q[7][5], GpuRoeState, wL_w/wR_w/QL/QR), exhausting the register file and causing massive local-memory spills. Also note: compiling the explicit template instantiations of `k_rhs_conv_cell<true/false>` degrades ptxas register allocation for the entire TU, so the instantiations are commented out.
**bench_d0 fix:** The benchmark was previously broken (link errors — missing gpu_amr.cu, gpu_ibm.cu etc.). Fixed to use `_GPU_NS`.
**Conclusion:** TENO7A is permanently FP64-compute-bound on RTX 3070 Laptop. The ≥55% BW target is only meaningful on A100/H100. Do not attempt to optimise BW% for this kernel on this hardware.

### Adjoint uses frozen TENO7-A weights
**Files:** `include/physics/adjoint_teno7.hpp`, `include/physics/adjoint_hllc.hpp`
**What:** `adjoint_rhs` differentiates through HLLC-ES and TENO7-A with **frozen smoothness-indicator weights** (weights treated as constants, not differentiated). This is a frozen-lambda / one-level adjoint — correct for shape optimisation and data assimilation where the flow sensitivity is the goal, but not exact for full second-order adjoints.
**Why:** Full differentiation through TENO7-A weight computation is significantly more complex and was not required for the D10 dot-product gate.
**Risk:** Gradient accuracy degrades near shocks where weight sensitivity is non-negligible.

### D7 WMLES gate — channel DNS validation ✅ DONE
**File:** `tests/cuda/test_t50_channel_wmles.cu`
**What:** t50 runs a turbulent channel at Re_τ=395 with WMLES (Reichardt algebraic wall model) and body-force-driven flow. After 500-step spin-up + 500-step statistics, the log-law intercept B = u⁺ − (1/κ)·ln(y⁺) is verified in the range [4.9, 6.2] (measured B ≈ 5.6, consistent with Reichardt composite law).
**Deviation from original D7 spec:** Uses 1000 steps (not ≥10k) at CFL=0.03 for acoustic stability with the compressible solver + WMLES ghost cells. Full turbulent channel would require CFL≤0.03 and O(10k) steps — feasible on A100/H100, impractical as a CI gate on RTX 3070 Laptop.
**Gate:** C50 (t50) verifies B ∈ [4.9, 6.2] at y⁺ ∈ [30, 200].

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
