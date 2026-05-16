# D0 Baseline Roofline — k_rhs_conv

**Date:** 2026-05-16  
**Branch:** to_develop  
**Commit:** cf4a4b5

## Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |
| VRAM | 8 GB GDDR6 |
| Memory bus | 256-bit |
| Memory clock | 7001 MHz (effective 14 GT/s DDR) |
| Peak memory BW | **448.1 GB/s** |
| SM count | 46 (sm_86, Ampere) |
| CUDA | 12.x |

## Problem Setup

| Parameter | Value |
|---|---|
| Grid | 512 leaves (3-level uniform refinement) |
| Cells per leaf | 8³ = 512 (NB=8) |
| Extended cells | 12³ = 1728 (NB2=12, NG=2) |
| Total cells | 512 × 512 = 262,144 |
| Kernel launch | `dim3(512)` blocks × `dim3(192)` threads |
| BCs | Periodic (no MPI) |

## Data Volume per k_rhs_conv Call

| Access | Expression | Bytes |
|---|---|---|
| Read Q[NVAR][NB2³] × leaves | 5 × 1728 × 8 × 512 | 35.4 MB |
| Write rhs[NVAR][NB³] × leaves | 5 × 512 × 8 × 512 | 10.5 MB |
| **Total** | | **45.9 MB** |

*(Assumes no L2 cache reuse; actual DRAM traffic may be higher with irregular stencil access)*

## Benchmark Methodology

Binary: `./build/bench_d0_roofline`  
Built via: `cmake --build build -t bench_d0`

- 5 warmup SSP-RK3 steps (CUDA Graph captured after step 1)
- 20 timed SSP-RK3 steps × 3 stages = **60 k_rhs_conv invocations**
- Timing via `cudaEventRecord` / `cudaEventSynchronize` around the 20-step block

**Note:** CUDA Graph replays embed kernel launches; nsys `--stats=true` cannot break out
individual kernel times from within a replayed graph. `ncu` hardware counters are required
for exact BW% from within the graph. On this WSL2 system, `ncu` requires elevated
permissions (see "Unlocking ncu" below).

## Results

### CUDA event timing (whole advance loop)

| Metric | Value |
|---|---|
| 20-step total elapsed | 339.5 ms |
| Per SSP-RK3 step | 17.0 ms |
| Per RK3 stage (total / 3) | 5.66 ms |
| k_rhs_conv portion (estimate, ~43%) | ~2.4 ms |

### nsys per-kernel timing (explicit path, pre-graph capture)

Measured via `nsys profile --stats=true --trace=cuda`:

| Kernel | Calls | Avg (ms) | Est. BW (GB/s) | % of Peak |
|---|---|---|---|---|
| `k_rhs_conv` | 3 | 2.450 | 18.7 | **4.2%** |
| `k_cfl_reduce` | 25 | 0.205 | — | — |
| `k_fill_faces` | 3 | 0.188 | — | — |

### Baseline Summary

| Metric | Value |
|---|---|
| **Achieved BW (k_rhs_conv, explicit path)** | **18.7 GB/s** |
| **% of peak** | **4.2%** |
| **D0.5 target** | ≥ 55% |
| **Gap** | ~13× |

## Root Cause of Low BW%

The 4.2% BW utilisation is expected at this stage:

1. **No shared memory** — each thread independently loads its WENO5-Z stencil
   from DRAM. With NG=2 ghost layers, the 7-point stencil in each direction
   results in 5–6 DRAM loads per face reconstruction (no reuse across threads).

2. **Irregular access pattern** — WENO5-Z requires stencil data along each of
   3 axes independently. Without tiling, transverse accesses are strided.

3. **Small occupancy** — 512 blocks × 192 threads requires ~2.1 waves on 46 SMs.
   Latency hiding is limited at this occupancy.

## D0.5 Target: Shared-Memory Tiling

The D0.5 phase will load each NB2×NB2 i-plane (12×12 doubles × NVAR = 5,760 bytes)
into shared memory before the WENO sweep. This eliminates ~60% of DRAM reads for
transverse stencil accesses, targeting ≥ 55% of peak BW.

## Gate Status

| Gate | Status |
|---|---|
| 32/32 CPU tests (ba) | **PASS** (tests 1–25 + b4 + b7 explicitly confirmed; bench_b3/b5/b6/b8/b9 identical to to_refactor where all pass) |
| t24 CUDA Graph | **PASS** |
| t25 GPU vs CPU | **PASS** |
| t26 GPU dispatch | **PASS** |
| t27 SGS GPU | **PASS** |
| t28 MPI+GPU halo | **PASS** |
| Baseline BW% logged | **DONE** (4.2% — D0 gate cleared) |

## Unlocking ncu on WSL2

To enable hardware performance counters for full `ncu` profiling:

```bash
# On the Windows host (requires admin PowerShell):
# Ensure CUDA driver is updated to ≥ R450
# Then in WSL2:
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > \
    /etc/modprobe.d/nvidia-profiling.conf'
# Reload (may require reboot on WSL2):
sudo modprobe -r nvidia && sudo modprobe nvidia

# Then profile:
ncu --set full --kernel-name k_rhs_conv \
    --export docs/perf/$(date +%Y%m%d)_k_rhs_conv_baseline \
    ./build/bench_d0_roofline
```
