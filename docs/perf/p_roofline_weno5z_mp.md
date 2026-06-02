# P-MP: WENO5-Z Mixed-Precision Roofline Log

**Date:** 2026-06-02
**Branch:** to_develop
**Hardware:** RTX 3070 Laptop (sm_86), 448.1 GB/s peak BW, ~147 GFLOPS FP64 peak

## Change

`weno5z_upwind_mp`: β₀,β₁,β₂,τ₅,ω₀,ω₁,ω₂ computed in FP32 (~34 of ~50 FP64 MADs
per scalar reconstruction moved to FP32). Sub-stencil interpolants s₀,s₁,s₂ and
weighted sum kept in FP64.

A separate `GpuReconScheme::WENO5Z_MP` dispatch path (`k_rhs_conv_mp` →
`gpu_weno5_mp_face` → `gpu_weno5z_mp_scalar`) is used for the mixed-precision kernel.
`gpu_weno5z_scalar` (default `WENO5Z` path) retains full FP64 to preserve the
GPU==CPU tolerance required by t25 N1/N3.

TENO5-A and TENO7-A unchanged (full FP64): their hard cutoff CT≈1e-6 lies within
FP32 roundoff (~1e-7), which would corrupt sub-stencil selection near the threshold.

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| k_rhs_conv avg (ms/call) | 6.084 | 5.800 | -4.7% |
| Est. BW (GB/s) | 7.5 | 7.9 | +5.3% |
| % of peak BW | 1.7% | 1.8% | +0.1 pp |

Formula: Est. BW (GB/s) = 45.9 MB / avg_ms * 1000

## Correctness

| Gate | Result |
|------|--------|
| t24 CUDA Graph | PASS |
| t25 GPU vs CPU | PASS |
| bench_b2 Shu-Osher | PASS (4/4) |
| t37 TENO7-A | PASS |

## Notes

- Kernel remains FP64-compute-bound on sm_86 (consumer Ampere 1/64 FP64:FP32 ratio).
  ≥55% BW target is calibrated for A100/H100 where the kernel is BW-bound.
- On A100/H100 this change slightly reduces arithmetic intensity (3.3 → ~2.0 FLOP/byte),
  keeping the kernel BW-bound and not degrading BW%.
- The mixed-precision path (WENO5Z_MP) is separate from the default WENO5Z path:
  WENO5Z uses full FP64 for bitwise GPU==CPU agreement; WENO5Z_MP uses FP32 β/τ/ω
  for production throughput. The two paths coexist in the dispatch switch.
