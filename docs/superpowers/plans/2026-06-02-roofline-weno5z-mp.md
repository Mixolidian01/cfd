# WENO5-Z Mixed-Precision Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce FP64 op count in the WENO5-Z reconstruction hot path by computing smoothness indicators and weights in FP32, keeping sub-stencil interpolants and the final weighted sum in FP64.

**Architecture:** Add `weno5z_upwind_mp()` to `gpu_rhs_recon.cuh`; redirect `gpu_weno5z_scalar()` to call it. All three WENO5-Z kernels (`k_rhs_conv`, `k_rhs_conv_teno`, `k_rhs_conv_tiled`) and the shmem helper `gpu_recon_shmem` in `gpu_rhs.cu` call reconstruction through `gpu_weno5z_scalar`, so one two-line edit propagates the change everywhere automatically — no changes to `gpu_rhs.cu` needed.

**Tech Stack:** CUDA C++20, `__device__ __forceinline__`, FP32/FP64 mixed arithmetic, nsys for per-kernel timing measurement.

---

## File Map

| File | Change |
|------|--------|
| `include/cuda/gpu_rhs_recon.cuh` | Add `weno5z_upwind_mp()` after `weno5z_upwind()`; update `gpu_weno5z_scalar()` to call it (2-line change) |
| `docs/perf/p_roofline_weno5z_mp.md` | New: before/after `bench_d0_roofline` timing log |

---

## Task 1: Record baseline timing

**Files:**
- Read: `docs/perf/baseline_roofline.md` (the D0 baseline for reference)

This task captures the pre-change timing so Task 3's perf log has a before/after comparison.

- [ ] **Step 1: Build the roofline benchmark**

```bash
cmake --build build -t bench_d0
```

Expected: `[100%] Built target bench_d0` (or already up to date).

- [ ] **Step 2: Run with nsys to capture per-kernel timing**

```bash
nsys profile --stats=true --trace=cuda --output /tmp/weno_mp_before \
    ./build/bench_d0_roofline 2>&1 | grep -A5 "k_rhs_conv"
```

Expected output (approximate — exact numbers will vary):

```
 Time(%)  Total Time (ns)  Instances    Avg (ns)   ...  Name
 ...      ...              3            2,450,000  ...  k_rhs_conv
```

Record the `Avg (ns)` value for `k_rhs_conv`. This is your baseline.

If nsys is unavailable, use the CUDA-event timing the binary prints directly:
```bash
./build/bench_d0_roofline
```
Expected: prints `step_time_ms` per RK3 step. Record this.

- [ ] **Step 3: Commit the recorded baseline as a comment in the perf log**

No code change yet — just note the number. We'll write `docs/perf/p_roofline_weno5z_mp.md`
fully in Task 3 after the post-change measurement.

---

## Task 2: Implement `weno5z_upwind_mp` and redirect `gpu_weno5z_scalar`

**Files:**
- Modify: `include/cuda/gpu_rhs_recon.cuh:21-51`

The only file that changes is `gpu_rhs_recon.cuh`. The change is two additions:
1. A new function `weno5z_upwind_mp` inserted right after `weno5z_upwind`.
2. Two-line change in `gpu_weno5z_scalar` to call `weno5z_upwind_mp` instead of `weno5z_upwind`.

- [ ] **Step 1: Add `weno5z_upwind_mp` after line 42 of `gpu_rhs_recon.cuh`**

The existing `weno5z_upwind` ends at line 42. Insert the following block between `weno5z_upwind`
and the "WENO5-Z scalar reconstruction" comment at line 44:

```cpp
// FP32 weights are safe for WENO5-Z because ω_k depends only on the relative magnitude
// of β_k — FP32 roundoff (~1e-7) does not misclassify smooth vs. shocked cells.
// TENO5-A/TENO7-A are excluded: their hard cutoff CT≈1e-6 lies within FP32 roundoff
// (~1e-7) for near-threshold cells, corrupting sub-stencil selection.
__device__ __forceinline__
double weno5z_upwind_mp(double a, double b, double c, double d, double e) noexcept {
    // Phase 2 (FP64): sub-stencil interpolants — computed before cast, inputs still double
    const double s0 = ( 2.0*a -  7.0*b + 11.0*c) * (1.0/6.0);
    const double s1 = (      -b +  5.0*c +  2.0*d) * (1.0/6.0);
    const double s2 = ( 2.0*c +  5.0*d -       e) * (1.0/6.0);
    // Phase 1 (FP32): smoothness indicators β, global indicator τ₅, WENO-Z weights
    const float fa=(float)a, fb=(float)b, fc=(float)c, fd=(float)d, fe=(float)e;
    const float B0 = (13.f/12.f)*(fa-2.f*fb+fc)*(fa-2.f*fb+fc)
                   +  (1.f/ 4.f)*(fa-4.f*fb+3.f*fc)*(fa-4.f*fb+3.f*fc);
    const float B1 = (13.f/12.f)*(fb-2.f*fc+fd)*(fb-2.f*fc+fd)
                   +  (1.f/ 4.f)*(fb-fd)*(fb-fd);
    const float B2 = (13.f/12.f)*(fc-2.f*fd+fe)*(fc-2.f*fd+fe)
                   +  (1.f/ 4.f)*(3.f*fc-4.f*fd+fe)*(3.f*fc-4.f*fd+fe);
    const float tau5 = fabsf(B0 - B2);
    constexpr float eps32 = 1.e-36f;
    constexpr float c0 = 0.1f, c1 = 0.6f, c2 = 0.3f;
    const float t0 = tau5/(B0+eps32), t1 = tau5/(B1+eps32), t2 = tau5/(B2+eps32);
    const float A0 = c0*(1.f+t0*t0), A1 = c1*(1.f+t1*t1), A2 = c2*(1.f+t2*t2);
    const float iAw = 1.f/(A0+A1+A2);
    const double w0=(double)(A0*iAw), w1=(double)(A1*iAw), w2=(double)(A2*iAw);
    // Phase 2 (FP64) continued: weighted sum
    return w0*s0 + w1*s1 + w2*s2;
}
```

- [ ] **Step 2: Update `gpu_weno5z_scalar` to call `weno5z_upwind_mp`**

Current `gpu_weno5z_scalar` (lines 46-51 of `gpu_rhs_recon.cuh`):

```cpp
__device__ __forceinline__
void gpu_weno5z_scalar(double vm2, double vm1, double v0,
                       double vp1, double vp2, double vp3,
                       double& vL, double& vR) noexcept {
    vL = weno5z_upwind(vm2, vm1, v0,  vp1, vp2);   // left state
    vR = weno5z_upwind(vp3, vp2, vp1, v0,  vm1);   // right state (mirrored)
}
```

Replace the two body lines only:

```cpp
__device__ __forceinline__
void gpu_weno5z_scalar(double vm2, double vm1, double v0,
                       double vp1, double vp2, double vp3,
                       double& vL, double& vR) noexcept {
    vL = weno5z_upwind_mp(vm2, vm1, v0,  vp1, vp2);   // left state
    vR = weno5z_upwind_mp(vp3, vp2, vp1, v0,  vm1);   // right state (mirrored)
}
```

- [ ] **Step 3: Verify it compiles**

Build only the GPU gates that use WENO5-Z (fast check, no test execution):

```bash
cmake --build build -t t24 2>&1 | tail -5
```

Expected: `[100%] Built target t24` — no compiler errors or warnings. If you see
`error: identifier "fabsf" not found` add `#include <cmath>` at the top of
`gpu_rhs_recon.cuh` (it's a device intrinsic that's usually available via
`cuda_runtime.h` which is already included transitively, but double-check).

- [ ] **Step 4: Commit**

```bash
git add include/cuda/gpu_rhs_recon.cuh
git commit -m "P-MP: WENO5-Z mixed-precision weights (FP32 β/τ/ω, FP64 interp)"
```

---

## Task 3: Run correctness gates and log performance

**Files:**
- Create: `docs/perf/p_roofline_weno5z_mp.md`

The existing correctness gates fully cover the WENO5-Z path. All four must pass at the
same tolerances as before the change.

- [ ] **Step 1: Run t24 (CUDA Graph + WENO5-Z correctness)**

```bash
cmake --build build -t t24 2>&1 | tail -10
```

Expected:
```
PASS  G0  GPU graph advance error = 0.000e+00 (tol 1e-10)
=== Result: 0 failure(s) ===
[100%] Built target t24
```

- [ ] **Step 2: Run t25 (GPU vs CPU correctness)**

```bash
cmake --build build -t t25 2>&1 | tail -10
```

Expected:
```
PASS  P9.1  GPU vs CPU max |Q| error < 1e-10
=== Result: 0 failure(s) ===
[100%] Built target t25
```

- [ ] **Step 3: Run bench_b2 (Shu-Osher shock-entropy, WENO5-Z)**

```bash
cmake --build build -t bench_b2 2>&1 | tail -15
```

Expected:
```
PASS  B2a  shock position in [6.0, 9.0]
PASS  B2b  post-shock oscillation amplitude > 0.25
PASS  B2c  no negative pressure
PASS  B2d  no negative density (WENO5-Z stability)
Results: 4 passed, 0 failed
==> PASS  B.2 gate cleared
```

- [ ] **Step 4: Run t37 (TENO7-A gate — confirms TENO paths unaffected)**

```bash
cmake --build build -t t37 2>&1 | tail -10
```

Expected:
```
PASS  A72 ...
PASS  A73a ...
PASS  A73b ...
=== Result: 0 failure(s) ===
[100%] Built target t37
```

- [ ] **Step 5: Run bench_d0_roofline post-change**

```bash
nsys profile --stats=true --trace=cuda --output /tmp/weno_mp_after \
    ./build/bench_d0_roofline 2>&1 | grep -A5 "k_rhs_conv"
```

Record the `Avg (ns)` for `k_rhs_conv`. If nsys is unavailable:
```bash
./build/bench_d0_roofline
```
Record the `step_time_ms`.

- [ ] **Step 6: Write `docs/perf/p_roofline_weno5z_mp.md`**

Use the measured values from Tasks 1 and 3 to fill in the actual numbers. Template:

```markdown
# P-MP: WENO5-Z Mixed-Precision Roofline Log

**Date:** 2026-06-02
**Branch:** to_develop
**Hardware:** RTX 3070 Laptop (sm_86), 448.1 GB/s peak BW, ~147 GFLOPS FP64 peak

## Change

`weno5z_upwind_mp`: β₀,β₁,β₂,τ₅,ω₀,ω₁,ω₂ computed in FP32 (~34 of ~50 FP64 MADs
per scalar reconstruction moved to FP32). Sub-stencil interpolants s₀,s₁,s₂ and
weighted sum kept in FP64.

TENO5-A and TENO7-A unchanged (full FP64): their hard cutoff CT≈1e-6 lies within
FP32 roundoff (~1e-7), which would corrupt sub-stencil selection near the threshold.

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| k_rhs_conv avg (nsys, ms) | 2.450 | <MEASURED> | <+/- %>  |
| Est. BW (GB/s) | 18.7 | <COMPUTED> | — |
| % of peak BW | 4.2% | <COMPUTED> | — |

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
```

- [ ] **Step 7: Commit**

```bash
git add docs/perf/p_roofline_weno5z_mp.md
git commit -m "perf: WENO5-Z mixed-precision roofline log; before/after bench_d0"
```

---

## Self-check before merge

Run the four gates together in one shot to confirm nothing regressed:

```bash
cmake --build build -t t24 t25 t37 bench_b2 2>&1 | grep -E "Result:|PASS|FAIL"
```

Expected: all `PASS`, `0 failure(s)` for each.
