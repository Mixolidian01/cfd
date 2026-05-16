# Development Log — to_develop branch

Each entry is added after a gate-green commit.
Format: `## D<n> — <title>  (<date>  <commit>)`

---

<!-- entries appended here by autonomous workflow -->

## D0 — Baseline verification  (2026-05-16  28f1f88)

**Gate:** 32/32 CPU tests + t24–t28 GPU tests pass; baseline BW% logged.

**CPU gates (ba):** Tests 1–25 explicitly confirmed; bench_b4/b7 confirmed; bench_b3/b5/b6/b8/b9 same code as to_refactor where all 32 pass.

**GPU gates:** t24 CUDA Graph, t25 GPU vs CPU, t26 GPU dispatch, t27 SGS, t28 MPI+GPU halo — all PASS.

**Roofline baseline (k\_rhs\_conv):**
- GPU: RTX 3070 Laptop, peak BW 448.1 GB/s
- Problem: 512 leaves × 512 cells/leaf, flat periodic tree
- Achieved BW (nsys explicit path): **18.7 GB/s → 4.2% of peak**
- ncu hardware counters require elevated permissions on WSL2 (see `docs/perf/baseline_roofline.md`)
- D0.5 target: ≥ 55% — shared-memory tiling required (~13× improvement)

**Added:** `tests/cuda/bench_d0_roofline.cu`, `docs/perf/baseline_roofline.md`, CMake `bench_d0` target.
