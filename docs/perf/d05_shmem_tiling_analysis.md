# D0.5 — Shared-Memory Tiling Analysis for k_rhs_conv

**Date:** 2026-05-17  
**Branch:** to_develop  
**Status:** Reverted — tiling regressed performance; root cause documented

---

## Objective

Load each NB2×NB2 i-plane (xi-slice) of the primitive-variable scratch array into shared
memory before the WENO5-Z sweep, targeting ≥ 55% of peak DRAM BW (448.1 GB/s) and
≥ 20 pp improvement over the 4.2% baseline.

## Implementation Summary

### k_rhs_conv_tiled (v2)

- **Block**: 144 = NB2² threads, split into two groups of 72 (xi_sel 0 or 1)
- **Shmem**: 2 xi-planes × 8 components × NB2 × PAD = 19968 B (PAD = NB2+1 = 13)
- **Padding**: stride-13 chosen so gcd(13, 32) = 1 → zero 4-byte bank conflicts on
  Z-stencil reads (was stride-12 → 4-way conflicts in v1)
- **Loop**: 6 xi-pairs; each pair loads → syncthreads → Y-faces → syncthreads →
  Z-faces → syncthreads
- **X-faces**: unchanged; read directly from global memory (stride-1, coalesced)
- **Correctness**: All gates pass — t24 (CUDA Graph, 0.000e+00 error), t25–t28 (GPU vs
  CPU correctness, SGS, MPI+GPU) — with the tiled kernel

### Bugs fixed during development

| Bug | Symptom | Fix |
|---|---|---|
| j/k index swap in `gpu_weno5_shmem<AXIS>` | G2-G4 failed (large error) | Corrected jkd formula per axis |
| Concurrent Y/Z atomicAdds in same xi-plane | G4 non-deterministic (3.4e-7 error vs graph replay) | Added `__syncthreads()` between Y and Z passes |
| Shmem fill covered only k=0..5 (72 threads, first iter only) | G1 failed (zero Q change) | Changed to stride-NF72 loop covering all 144 cells |

---

## Measured Performance

| Version | Kernel | Avg time (nsys) | Est. BW (GB/s) | % of peak |
|---|---|---|---|---|
| D0 baseline | `k_rhs_conv` (192 threads) | 2.51 ms | 18.7 | 4.2% |
| D0.5 tiled v2 | `k_rhs_conv_tiled` (144 threads) | 3.80 ms | ~12.4 | ~2.8% |
| **Reverted (exec)** | `k_rhs_conv` (192 threads) | 2.51 ms | 18.7 | 4.2% |

D0.5 tiled v2 regressed latency by **1.5×** with no BW improvement.

---

## Root Cause Analysis

### 1. The kernel is NOT bandwidth-limited

At 4.2% peak BW the kernel is **latency-bound / arithmetic-bound**, not DRAM-bandwidth-
bound. The working set per leaf is 9 comps × 1728 cells × 8 B = 124 KB. With 512 leaves
(46 SMs, ~11 concurrent blocks/SM), a significant fraction of WENO5 stencil re-reads is
served from **L2 cache** (4 MB) rather than DRAM. Shared memory cannot improve DRAM BW
when L2 is already capturing the data.

### 2. The shmem fill is itself non-coalesced

The scratch layout is **i-major** (SoA):
```
flat = i + NB2*j + NB2²*k     (NB2=12, so j stride=12, k stride=144)
```
Loading a full NB2×NB2 xi-plane (fixed i, all j,k) from this layout means:
- Consecutive threads access addresses at stride-12 doubles within each k row.
- A 32-thread warp spans addresses: `xi+0, xi+12, ..., xi+132, xi+144, ...`

This is identical non-coalescence to the original WENO5 stencil reads — no DRAM traffic
savings in the fill phase. The shmem load is as cache-line-inefficient as the direct reads
it replaces.

### 3. Added overhead outweighs stencil reuse

- **18 extra `__syncthreads()` barriers** per block (3 per xi-pair × 6 pairs)
- **Lower occupancy**: 144 threads (4.5 warps/SM) vs 192 threads (6 warps/SM); fewer
  warps to hide instruction latency → longer effective wall time
- **Double load**: 16 global reads per thread just for shmem fill vs 10 stencil reads
  for WENO5 in the original (most cached in L1/L2)

### 4. Summary

Shared-memory tiling is the right approach in principle (eliminates redundant DRAM reads)
but only when the kernel IS DRAM-bandwidth-limited. At 4.2% BW on a problem where L2
already captures most stencil reuse, shmem adds overhead with no bandwidth benefit.

---

## Path to ≥ 55% BW

To genuinely reach ≥ 55% peak BW for WENO-class kernels, the bottleneck must be shifted
from latency/arithmetic to bandwidth. This requires:

| Approach | Notes |
|---|---|
| **Per-axis kernels** (X, Y, Z separately) | Each kernel processes faces along one axis; thread layout chosen so xi varies across a warp → stride-1 reads in the normal direction. Eliminates transverse stride. 3× kernel launches per stage. |
| **Transposed scratch layout** | At RHS entry, run a transposing kernel to produce separate j-major and k-major scratch arrays. Y and Z WENO5 then read stride-1. Extra memory + 1 transpose kernel. |
| **TENO7-A (D3)** | Higher-order reconstruction naturally leads to a different thread-to-face assignment; can be designed coalesced from scratch. |
| **Persistent threads + software pipeline** | Keep blocks resident on SM across multiple RK stages; software-prefetch next block's data while computing current. Complex but can reach 40–60% BW on Volta/Ampere. |

The highest-ROI next step is **per-axis kernels** (two additional `__global__` functions
`k_rhs_conv_y` and `k_rhs_conv_z`, each with a thread layout matching their normal axis).
This is a natural prerequisite for TENO7-A (D3) and should be implemented there.

---

## Gate Status

| Requirement | Status |
|---|---|
| T08 convergence rate ≥ 1.8 (unchanged) | **PASS** (rate = 1.8; original kernel restored) |
| BW% logged to docs/perf/ | **DONE** (see above; 4.2% → 4.2%; no improvement) |
| exec() uses faster kernel | **DONE** (reverted to `k_rhs_conv` 192-thread original) |
| `k_rhs_conv_tiled` retained for reference | **YES** (retained in gpu_rhs.cu, not called) |

The ≥ 20 pp BW improvement target was **not achieved** with shared-memory tiling.
The analysis and forward path are documented above. D0.5 correctness work is complete;
performance optimisation is deferred to D3 (per-axis TENO7-A reconstruction).
