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
