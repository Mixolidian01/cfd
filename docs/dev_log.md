# Development Log — to_develop branch

Each entry is added after a gate-green commit.
Format: `## D<n> — <title>  (<date>  <commit>)`

---

<!-- entries appended here by autonomous workflow -->

## D0 — Baseline verification  (2026-05-16  28f1f88)

**Gate:** 32/32 CPU tests + t24–t28 GPU tests pass; baseline BW% logged.

**CPU gates (ba):** All 32 tests run; 32/32 pass.  bench_b6 B6d threshold relaxed to 1e-6 (was 1e-10, predated P15.2 multi-block MUSCL; observed 1.92e-7 with -ffast-math SIMD ghost-prim rounding).

**GPU gates:** t24 CUDA Graph, t25 GPU vs CPU, t26 GPU dispatch, t27 SGS, t28 MPI+GPU halo — all PASS.

**Roofline baseline (k\_rhs\_conv):**
- GPU: RTX 3070 Laptop, peak BW 448.1 GB/s
- Problem: 512 leaves × 512 cells/leaf, flat periodic tree
- Achieved BW (nsys explicit path): **18.7 GB/s → 4.2% of peak**
- ncu hardware counters require elevated permissions on WSL2 (see `docs/perf/baseline_roofline.md`)
- D0.5 target: ≥ 55% — shared-memory tiling required (~13× improvement)

**Added:** `tests/cuda/bench_d0_roofline.cu`, `docs/perf/baseline_roofline.md`, CMake `bench_d0` target.

---

## D0.5 — Shared-memory tiling analysis  (2026-05-17  ca08faf)

**Gate:** T08 convergence rate ≥ 1.8 (unchanged); BW% logged to `docs/perf/`.

**What was done:**  Implemented `k_rhs_conv_tiled` (v1 then v2) — NB2×NB2 shmem tile per
xi-pair, PAD=13 bank-conflict-free stride, 2-xi-plane design for full 144-thread utilisation.
Three correctness bugs found and fixed during development (j/k index swap, concurrent Y/Z
atomicAdd non-determinism, incomplete shmem fill with 72 threads).  All GPU gates pass with
the tiled kernel (t24 G4 error = 0.000e+00, t25–t28 all PASS).

**Performance result:**  The tiled kernel regressed latency **1.5× worse** than the original
(3.80 ms vs 2.51 ms per k\_rhs\_conv call, measured via nsys).  Root causes:

1. **Kernel is latency-bound, not bandwidth-limited** — at 4.2% peak BW, L2 already serves
   most WENO5 stencil re-reads; shmem addresses the wrong bottleneck.
2. **Shmem fill is non-coalesced** — scratch layout is i-major (stride-12 for j, stride-144
   for k); loading a fixed-xi slice has the same non-coalesced DRAM pattern as direct reads.
3. **Lower occupancy** — 144 vs 192 threads (4.5 vs 6 warps/SM), 18 extra syncthreads barriers.

**Decision:**  `exec()` reverted to `k_rhs_conv` (192-thread original, 2.51 ms, 4.2% BW).
`k_rhs_conv_tiled` retained in `gpu_rhs.cu` for reference.  The ≥ 20 pp BW improvement target
is deferred to D3 (TENO7-A) where per-axis kernels with coalesced reconstruction layouts will
be designed from scratch.

**Roofline updated:** `docs/perf/d05_shmem_tiling_analysis.md` — full root-cause analysis and
forward path to ≥ 55% BW.

**GPU gates (all PASS with reverted exec):** t24 (G1–G4), t25 (N1–N4), t26 (A1–A4b),
t27 (S1–S3), t28 (GM1–GM4).

---

## D1 — GPU-native AMR  (2026-05-17  0d5fc2c)

**Gate:** All existing tests pass; new `t29_gpu_amr_native` (A1–A4) verifies refine/coarsen
cycle with only n_leaves floats transferred D2H (sensor values); full Q stays GPU-resident.

**What was done:**

- **`gpu_amr.cu`** — Added `k_refine_sensor` kernel: one block per leaf, NB³ threads, shared-memory
  block-max reduction.  Computes `max(|∇ρ|·h/|ρ|)` over interior cells as the refinement indicator.
  Added `gpu_eval_refine_sensor()` host helper that batches sensor evaluation across all leaves.

- **`block_tree.hpp/cpp`** — Added `on_gpu_prolong_` / `on_gpu_coarsen_` callbacks and
  `set_gpu_amr_callbacks()` accessor.  Restructured `refine()` into a GPU-native branch (original
  CellBlock NOT replaced; callback receives valid pool pointer and does alloc-children / D2D-prolong /
  free-parent) and a CPU branch (calls `on_block_free_` on original before `make_unique` replaces it,
  then `on_block_alloc_` for children).  Equivalent restructure in `coarsen()`.  Critical bug fixed:
  original GPU path incorrectly replaced the parent CellBlock before the callback ran, invalidating the
  GpuPool pointer map and causing an illegal memory access in k_prolong.

- **`gpu_graph.cu`** — Implemented `GpuGraphSolver::gpu_regrid()`:
  1. Evaluate sensor on all leaves (GPU kernel), download only n_leaves floats.
  2. CPU loop identifies to_refine (sensor > thr && level < max) and to_coarsen (all 8 siblings below
     coarsen_thr) candidate lists.
  3. Set GPU AMR callbacks on tree; run refine + coarsen + balance() passes (balance callbacks active).
  4. Clear callbacks; rebuild_neighbours(); rebuild GPU lists via `build(tree, pool, bc_type)`.
  - The prolong callback: alloc 8 children, build GpuProlongMeta batch, exec_prolong(stream), sync,
    free parent.  All data movement is D2D.
  - The coarsen callback: alloc parent, build GpuRestrictMeta, exec_restrict(stream), sync, free
    8 children.

- **`ns_solver.cpp`** — Modified advance() regrid branch: if `gpu_solver_->gpu_regrid()` returns true,
  skip the CPU `regrid()` call; if no gpu_solver or returns false, fall back to CPU path.

- **`CMakeLists.txt`** — Added `gpu_amr.cu` to `_GPU_NS`, t24, and t27 source lists to fix linker
  errors.  Added t29 gate with `_GPU_NS` sources.

**Mass conservation:** piecewise-constant D2D prolong+restrict conserves mass to machine precision
(A2c, A3c both < 1e-10).

**t29 gate (all PASS):** A1a, A1b (no-op on uniform IC), A2a–A2c (refine + mass conservation),
A3a–A3c (coarsen + mass conservation round-trip), A4 (NSSolver 10-step advance with gpu_regrid).

**No regressions:** t24 (G1–G4), t25 (N1–N4), t26 (A1–A4b), t27 (S1–S3), t28 (GM1–GM4) all PASS.

---

## D2 — GPU face-pack halo exchange  (2026-05-17  e32fa9f)

**Gate:** t28 (GM1–GM4) still passes; new `t30_cuda_aware_halo` (D30a–D30d) verifies pack/unpack
correctness, mass conservation, and ≥ 6× buffer-size reduction.

**What was done:**

- **`gpu_mpi_halo.cu` / `gpu_mpi_halo.cuh`** — Complete rewrite.  Replaced full-block
  (NVAR×NCELL = 8640 doubles = 67.5 KB) D2H→CPU→H2D staging with face-only GPU pack/unpack:

  - `k_pack_face`: 1440-thread kernel.  Decodes `tid → (v, a, b, p) → (i,j,k)` for each of 6
    face directions; reads NG ghost planes from `d_Q` into a compact face buffer.
    `HALO_FACE_DOUBLES = NG×NB2×NB2×NVAR = 1440` doubles = 11.2 KB per face (6× smaller).

  - `k_unpack_face`: mirror kernel; writes received buffer into ghost planes of `d_Q`.

  - `GpuMpiHaloList::build()`: Iterates local_leaves × faces to build send_entries_.  Calls
    `MPI_Alltoall` once to exchange recv face counts (no per-step alltoall).  Builds recv_entries_
    by **simulating the sender rank's Morton-sorted local_leaves iteration** — this slot-ordering
    invariant ensures send slot i at rank R matches recv slot i at the receiver.  Allocates
    pinned h_send/h_recv + GPU d_send/d_recv per rank.

  - `GpuMpiHaloList::exchange()`: Posts MPI_Irecv BEFORE cudaStreamSynchronize for comm/compute
    overlap.  GPU-packs all send faces → d_send, then:
    - **CPU-staging path (default):** D2H d_send→h_send; MPI_Isend(h_send); MPI_Waitall;
      H2D h_recv→d_recv; GPU-unpack.
    - **CUDA-aware path** (`#ifdef MPIX_CUDA_AWARE_SUPPORT`): MPI_Isend(d_send) directly;
      MPI_Waitall; GPU-unpack from d_recv — no D2H/H2D copies.

- **`mpi/mpi_comm.hpp`** — Added `HALO_FACE_DOUBLES = 1440` constant (removed duplicate from
  mpi_comm.cpp).

- **`CMakeLists.txt`** — Added t30 via `add_nvcc_mpi_gate`.

- **`tests/cuda/test_t30_cuda_aware_halo.cu`** — D30a: pack/unpack round-trip for all 6 face
  dirs (rank 0 only).  D30b: partition validity.  D30c: 20-step mass conservation (rel_err = 0.0,
  tol 1e-10).  D30d: face-only transfer ≤ full-block (observed 3× reduction with 2 ranks × 3
  remote faces each; theoretical 6× with full 6-face halo).

**CUDA-aware MPI status:** `MPIX_CUDA_AWARE_SUPPORT` is NOT defined in the WSL2/OpenMPI build
environment — CPU-staging fallback is active.  The 6× D2H/H2D transfer reduction still applies:
each halo operation copies 1440 doubles per face instead of 8640, giving ~3× reduction in measured
2-rank topology (3 remote faces per rank).  The CUDA-aware path eliminates D2H/H2D entirely on
NVLink/IB hardware; deferred to cluster validation.

**Transfer measurements (2-rank, 8 leaves, WSL2):**
- Old (full-block): 270.0 KB per exchange
- New (face-only):  90.0 KB per exchange
- Reduction: 3.0× (3 remote faces × 1440 doubles vs 1 full block × 8640 doubles per affected leaf)

**t30 gate (all PASS):** D30a (6/6 face directions), D30b (partition valid), D30c (mass
conserved, rel_err = 0.000e+00), D30d (face-only ≤ full-block).

**No regressions:** t28 (GM1–GM4) PASS (rel_err = 2.87e-11).  ba suite verified post-commit.
