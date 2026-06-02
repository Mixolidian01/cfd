# WENO5-Z Mixed-Precision Reconstruction Design

## Goal

Reduce the FP64 arithmetic bottleneck in `k_rhs_conv` on RTX 3070 Laptop by computing
WENO5-Z smoothness indicators (β₀,β₁,β₂), the global indicator τ₅, and the normalized
weights ω₀,ω₁,ω₂ in FP32, while keeping the final sub-stencil interpolation and weighted
sum in FP64. Expected outcome: ~40% reduction in FP64 op count in the WENO5-Z hot path,
yielding a measurable speedup on `bench_d0_roofline` on this hardware.

## Architecture

### Why the kernel is compute-bound here, not BW-bound

On RTX 3070 Laptop (sm_86), FP64 throughput is 1/64 of FP32 (~147 GFLOPS peak) while
memory bandwidth is 448 GB/s. WENO5-Z arithmetic intensity is ~3.3 FLOP/byte, which is
10× above the compute/BW crossover on this GPU → compute-bound regardless of memory
layout. Shared-memory tiling (D0.5) was confirmed to regress performance for this reason.

On A100/H100 (9.7 TFLOPS FP64, 2 TB/s BW), the crossover is ~4.85 FLOP/byte; the same
kernel at 3.3 FLOP/byte would be BW-bound and ≥55% BW is achievable. The mixed-precision
change reduces arithmetic intensity slightly on those cards too, but the primary benefit
is on FP64-limited consumer hardware.

### Scope: WENO5-Z only

TENO5-A and TENO7-A are excluded from the mixed-precision treatment. Their hard cutoff
CT≈1e-6 is within FP32 roundoff (~1e-7) for near-threshold cells, which would corrupt
sub-stencil selection. WENO5-Z's smooth weights depend only on the relative magnitude of
β_k, making FP32 safe for the indicator computation.

## Changed Files

| File | Change |
|------|--------|
| `include/cuda/gpu_rhs_recon.cuh` | Add `weno5z_upwind_mp()`; update `gpu_weno5z_scalar()` to call it |
| `docs/perf/p_roofline_weno5z_mp.md` | Before/after timing from `bench_d0_roofline` |

`src/cuda/gpu_rhs.cu` requires **no changes**. All three WENO5-Z kernels (`k_rhs_conv`,
`k_rhs_conv_teno`, `k_rhs_conv_tiled`) reach reconstruction through `gpu_weno5z_scalar` /
`gpu_weno5_face`, so the mixed-precision path propagates automatically.

## Mixed-Precision Boundary

Call chain (unchanged externally):
```
gpu_weno5_face → gpu_weno5z_scalar → weno5z_upwind_mp   [NEW]
```

### `weno5z_upwind_mp` signature
```cpp
// FP32 weights are safe for WENO5-Z because the smooth weights ω_k depend only on the
// relative magnitude of β_k — FP32 roundoff (~1e-7) does not misclassify smooth vs.
// shocked cells. TENO5-A/TENO7-A are excluded: their hard cutoff CT≈1e-6 is within
// FP32 roundoff for near-threshold cells, corrupting sub-stencil selection.
__device__ __forceinline__
double weno5z_upwind_mp(double q0, double q1, double q2, double q3, double q4);
```

### Phase 1 — FP32 (β, τ₅, ω)
```cpp
float fq0=(float)q0, fq1=(float)q1, fq2=(float)q2, fq3=(float)q3, fq4=(float)q4;
float b0 = (13.f/12.f)*sq(fq0-2.f*fq1+fq2) + (1.f/4.f)*sq(fq0-4.f*fq1+3.f*fq2);
float b1 = (13.f/12.f)*sq(fq1-2.f*fq2+fq3) + (1.f/4.f)*sq(fq1-fq3);
float b2 = (13.f/12.f)*sq(fq2-2.f*fq3+fq4) + (1.f/4.f)*sq(fq2-4.f*fq3+fq4);
float t5  = fabsf(b0-b2);
const float eps = 1.e-36f;
float a0 = 0.1f/sq(eps+b0+t5), a1 = 0.6f/sq(eps+b1+t5), a2 = 0.3f/sq(eps+b2+t5);
float iaw = 1.f/(a0+a1+a2);
double w0=(double)(a0*iaw), w1=(double)(a1*iaw), w2=(double)(a2*iaw);
```

### Phase 2 — FP64 (sub-stencil interpolants + weighted sum)
```cpp
double g0 = ( 2.*q0 - 7.*q1 + 11.*q2) / 6.;
double g1 = (   -q1 + 5.*q2 +  2.*q3) / 6.;
double g2 = ( 2.*q2 + 5.*q3 -     q4) / 6.;
return w0*g0 + w1*g1 + w2*g2;
```

**FP64 ops saved per scalar reconstruction:** ~34 of ~50 multiply-adds (β/τ/α computation).
Approximately 40% reduction in FP64 ops in the WENO hot path.

## Testing

### Correctness gates (must all pass at existing tolerances)

| Gate | What it covers |
|------|---------------|
| `t24` | CUDA Graph replay; solution error = 0.000e+00 vs reference |
| `t25` | GPU vs CPU correctness, WENO5-Z path, tol 1e-10 |
| `t37` | TENO7-A gate — confirms TENO paths unchanged |
| `bench_b2` | Shu-Osher shock-entropy: shock position in [6.0, 9.0], amplitude > 0.25 |

No new gate is required; the existing gates exercise the full WENO5-Z reconstruction path
including shock-dominated and smooth-flow regimes.

### Performance measurement

Run `bench_d0_roofline` with nsys before and after. Capture per-kernel avg time for
`k_rhs_conv`. Compute estimated BW (GB/s) and % of peak using same methodology as the
D0 baseline (45.9 MB working set, 2.51 ms baseline → 18.7 GB/s → 4.2%).

Log before/after to `docs/perf/p_roofline_weno5z_mp.md`.

Expected result: ~30–40% wall-time reduction in `k_rhs_conv`; BW% rising from 4.2% to
~6–7% (still FP64-compute-bound, but measurably less so). The kernel will not reach
≥55% BW on this hardware — that target is calibrated for A100/H100 where the kernel is
BW-bound.

## What This Does Not Address

- Memory layout optimisation for A100/H100 (separate concern — requires per-axis kernels
  or transposed scratch buffer to eliminate stride-12 transverse reads)
- TENO5-A / TENO7-A performance (excluded by design; full FP64 retained)
- The ≥55% BW roofline target on dev hardware (physically unreachable on sm_86 for FP64
  WENO at this problem size)
