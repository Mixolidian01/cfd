# /review

Expert CFD + architecture review of a file or directory.

## Usage
/review [file or directory]

## Instructions

Analyse in this order:

### 1. Numerical correctness
- Is interior flux entropy-stable (HLLC-ES or Chandrashekar)? Flag if standard Roe.
- Is T08 convergence rate documented and ≥ 1.8?
- Does ghost fill use the same reconstruction as accumulate_face (BC consistency)?
- Is the positivity floor applied after every RK3 stage (CPU and GPU)?

### 2. Refactor compliance (target architecture)
- Are there `hllc_flux_x` / `_y` / `_z` axis duplicates? Flag as [HIGH].
- Are BC types dispatched via `if (bc_type == ...)` chains instead of `std::variant`? Flag.
- Are scheme-selection `if` branches inside `__global__` kernels? Flag as [CRITICAL].
- Are there `virtual` functions in device-callable structs? Flag as [HIGH].

### 3. C++ quality
- Raw owning pointers (`new`/`delete`)? Flag.
- Magic numbers (unnamed literals for stencil weights, NVAR, etc.)? Flag.
- Missing `constexpr` on pure compile-time constants? Flag.

### 4. CUDA correctness
- `cudaDeviceSynchronize()` in the advance loop? Flag as [CRITICAL].
- Synchronous `cudaMemcpy` in the hot path? Flag as [HIGH].

### Output format
[CRITICAL] — correctness or UB, blocks merge
[HIGH]     — architecture violation or performance regression
[MEDIUM]   — maintainability issue
[INFO]     — improvement suggestion

For each issue: file, line, problem, corrected snippet.
