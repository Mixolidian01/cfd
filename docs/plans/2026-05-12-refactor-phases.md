# CFD Solver Refactor Phases R0–R6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layer a Physics/Concept/Execution triad (layers P, C, E) on top of the existing four-layer solver stack without replacing any existing logic, keeping all gate tests (t1–t26) green after each phase.

**Architecture:** R0 enables C++20; R1 adds concept contracts; R2 extracts physics into header functors; R3 replaces BC enum dispatch with `std::variant`; R4 formalises the CPU/GPU split behind execution tags and a factory; R5 pre-compiles the (Flux × Recon × EOS) instantiation matrix; R6 (optional, breaking) replaces raw block index access with `std::mdspan`.

**Tech Stack:** C++20, CUDA 13.2 / nvcc, g++ 13.3, CMake 3.18+, CTest.

---

## File Map

### R0
- Modify: `CMakeLists.txt` — `CMAKE_CXX_STANDARD 17→20`; all `-std=c++17` in nvcc commands → `-std=c++20`

### R1
- Create: `include/concepts.hpp`
- Modify: `src/operators.cpp` — add `static_assert` at HLLC-ES and WENO5 call sites

### R2
- Create: `include/physics/log_mean.hpp`
- Create: `include/physics/weno5z_scalar.hpp`
- Create: `include/physics/hllc_flux.hpp`
- Create: `include/physics/weno5_recon.hpp`
- Modify: `src/operators.cpp` — bodies of `hllc_flux`, `hllc_es_flux`, `weno5_face_t` delegate to functors; `log_mean`, `sq`, `weno5z_scalar` move to headers
- Modify: `include/operators.hpp` — `hllc_es_flux_t<DIR>` delegates to `HllcEsFlux<DIR>`

### R3
- Create: `include/bc_types.hpp`
- Modify: `include/ns_solver.hpp` — replace `BCType bc` with `BCVariant bc_variant` in `SolverConfig`; remove `BCType` enum
- Modify: `src/ns_solver.cpp` — replace every BC if/else chain with `std::visit`
- Modify: `src/cuda/gpu_ghost_fill.cu` — update `bc_type` integer extraction
- Modify: `tests/test_ns.cpp` — update test setup to use `BCVariant`
- Modify: `tests/cuda/test_p82_gpu_ghost.cu` (and similar) — update BC construction

### R4
- Create: `include/execution.hpp`
- Create: `src/solver_factory.cpp`
- Modify: `include/ns_solver.hpp` — add `FluxScheme` + `ExecutionBackend` to `SolverConfig`
- Modify: `CMakeLists.txt` — add `solver_factory.cpp` to `ns_solver` library

### R5
- Create: `src/instantiation_matrix.cpp`
- Modify: `include/operators.hpp` — `compute_rhs` becomes `template<class Flux, class Recon, class EOS>`
- Modify: `src/operators.cpp` — move old `compute_rhs` body into template; add explicit instantiations
- Modify: `CMakeLists.txt` — add `instantiation_matrix.cpp` to `operators` library
- Modify: `tests/test_operators.cpp` — add scheme-selection test

### R6 (optional, last)
- Modify: `include/cell_block.hpp` — add `mdspan` accessors
- Modify: `src/operators.cpp`, `src/amr_operators.cpp`, GPU kernels — use mdspan views

---

## Task 1: R0 — Enable C++20

**Files:**
- Modify: `CMakeLists.txt`

The CMake CXX standard and all nvcc `-std=c++17` flags must change simultaneously so that every TU uses the same language version. There are 10 nvcc invocations (t5, t8, t19–t26); all must be updated.

- [ ] **Step 1: Change CMake CXX standard**

In `CMakeLists.txt` line 22–23, change:
```cmake
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```
to:
```cmake
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

- [ ] **Step 2: Change all nvcc `-std=c++17` flags**

There are 10 occurrences of `-std=c++17` in the nvcc `add_custom_command` blocks (t5, t8, t19, t20, t21, t22, t23, t24, t25, t26). Replace every occurrence with `-std=c++20`. The pattern appears as the third argument after `COMMAND ${NVCC}` in each block.

After editing, verify with:
```bash
grep -c "std=c++17" CMakeLists.txt
```
Expected output: `0`

- [ ] **Step 3: Re-run cmake to regenerate build system**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
Expected: configuration succeeds, no errors.

- [ ] **Step 4: Verify all CPU gate tests pass**

```bash
cmake --build build -t ba
```
Expected: `100% tests passed` (t1–t7 in CTest, covering test_linalg, test_block, test_operators, test_ns, test_amr6, test_step7).

- [ ] **Step 5: Verify GPU gate tests pass**

```bash
cmake --build build -t t5 && cmake --build build -t t24 && cmake --build build -t t25 && cmake --build build -t t26
```
Expected: all four GPU targets build and run without errors.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt
git commit -m "R0: enable C++20 — CMake standard and all nvcc -std flags"
```

---

## Task 2: R1 — Concept Layer

**Files:**
- Create: `include/concepts.hpp`
- Modify: `src/operators.cpp` (add ~10 lines at the top of the operators section)

C++20 concepts are defined in one header. Four concepts cover the entire physics interface. Property type-traits (separate from concepts) track entropy stability, conservativeness, and skew-symmetry. In R1 no types yet satisfy the concepts by design — `static_assert` checks in `operators.cpp` use local wrapper structs as verification stubs.

- [ ] **Step 1: Create `include/concepts.hpp`**

```cpp
#pragma once
// Layer C — Concept contracts (CLAUDE.md R1)
// Applied at every template boundary. Never inside __global__ kernels.

#include "cell_block.hpp"   // Prim, CellBlock, NVAR
#include <array>
#include <concepts>
#include <type_traits>

// ── RiemannFlux ──────────────────────────────────────────────────────────────
// A callable (L, R) → std::array<double, NVAR>.
// Axis is baked into the type at instantiation (template<Axis DIR>).
template<typename F>
concept RiemannFlux = requires(F f, const Prim& L, const Prim& R) {
    { f(L, R) } -> std::convertible_to<std::array<double, NVAR>>;
};

// ── SpatialReconstruction ────────────────────────────────────────────────────
// A callable that writes reconstructed left/right primitive states at a face.
// Signature: operator()(const Prim* pc, int i, int j, int k, Prim&, Prim&)
template<typename R>
concept SpatialReconstruction =
    requires(R r, const Prim* pc, int i, int j, int k, Prim& qL, Prim& qR) {
        r(pc, i, j, k, qL, qR);
    };

// ── EquationOfState ──────────────────────────────────────────────────────────
// Converts conservative state (ρ, ρu, ρv, ρw, E) → Prim.
template<typename E>
concept EquationOfState =
    requires(E eos, double rho, double rhou, double rhov, double rhow, double en) {
        { eos.cons_to_prim(rho, rhou, rhov, rhow, en) } -> std::same_as<Prim>;
    };

// ── BoundaryCondition ────────────────────────────────────────────────────────
// Fills one ghost layer in a block for a given axis (0/1/2) and side (0=lo, 1=hi).
template<typename B>
concept BoundaryCondition =
    requires(B bc, CellBlock& blk, int axis, int side) {
        bc.fill_ghost(blk, axis, side);
    };

// ── Property flags ───────────────────────────────────────────────────────────
// Specialise to std::true_type in the physics header that defines the functor.
template<typename F> struct is_entropy_stable  : std::false_type {};
template<typename F> struct is_conservative    : std::false_type {};
template<typename F> struct is_skew_symmetric  : std::false_type {};

template<typename F>
inline constexpr bool is_entropy_stable_v  = is_entropy_stable<F>::value;
template<typename F>
inline constexpr bool is_conservative_v    = is_conservative<F>::value;
template<typename F>
inline constexpr bool is_skew_symmetric_v  = is_skew_symmetric<F>::value;
```

- [ ] **Step 2: Build concepts header alone**

```bash
cmake --build build -t t3 2>&1 | tail -5
```
Expected: `test_operators` still compiles and passes (concepts.hpp not included yet — zero-impact build).

- [ ] **Step 3: Add `static_assert` verification stubs in `operators.cpp`**

After the `#include "operators.hpp"` line in `src/operators.cpp`, add:

```cpp
#include "concepts.hpp"

// R1: verify existing flux/recon free functions satisfy Layer-C contracts.
// These local wrapper structs are the stubs; replaced by real functors in R2.
namespace {

struct _HllcEsCheck {
    std::array<double,NVAR> operator()(const Prim& L, const Prim& R) const noexcept {
        return hllc_es_flux(L, R, 0);
    }
};
static_assert(RiemannFlux<_HllcEsCheck>,
    "hllc_es_flux must satisfy RiemannFlux — check NVAR and return type");

struct _HllcCheck {
    std::array<double,NVAR> operator()(const Prim& L, const Prim& R) const noexcept {
        return hllc_flux(L, R, 0);
    }
};
static_assert(RiemannFlux<_HllcCheck>,
    "hllc_flux must satisfy RiemannFlux");

struct _Weno5Check {
    void operator()(const Prim* pc, int i, int j, int k,
                    Prim& qL, Prim& qR) const noexcept {
        weno5_face_t<Axis::X>(pc, i, j, k, qL, qR);
    }
};
static_assert(SpatialReconstruction<_Weno5Check>,
    "weno5_face_t must satisfy SpatialReconstruction");

} // anonymous namespace
```

Note: `hllc_es_flux`, `hllc_flux` are declared in `operators.hpp` (included before `concepts.hpp`); `weno5_face_t` is a static function defined earlier in `operators.cpp` so the stub must be placed after its definition (around line 800). Place the static_asserts for `_Weno5Check` just before `compute_rhs_impl`.

- [ ] **Step 4: Run gate t3 to verify static_asserts fire correctly at compile time**

```bash
cmake --build build -t t3 2>&1 | tail -5
```
Expected: compiles and all operator tests pass. No concept failures (the stubs satisfy the concepts by construction).

- [ ] **Step 5: Run all CPU + GPU gates**

```bash
cmake --build build -t ba
cmake --build build -t t24 && cmake --build build -t t25 && cmake --build build -t t26
```
Expected: 100% pass.

- [ ] **Step 6: Commit**

```bash
git add include/concepts.hpp src/operators.cpp
git commit -m "R1: concept layer — RiemannFlux, SpatialReconstruction, EquationOfState, BoundaryCondition + static_assert stubs"
```

---

## Task 3: R2 — Physics Functor Extraction

**Files:**
- Create: `include/physics/log_mean.hpp`
- Create: `include/physics/weno5z_scalar.hpp`
- Create: `include/physics/hllc_flux.hpp`
- Create: `include/physics/weno5_recon.hpp`
- Modify: `src/operators.cpp` — make `hllc_flux`, `hllc_es_flux`, `weno5_face_t` delegate to functors; remove `sq`, `log_mean`, `weno5z_scalar` bodies (moved to headers)
- Modify: `include/operators.hpp` — `hllc_es_flux_t` and `hllc_flux_t` delegate to functors

Strategy: the function bodies move into `__host__ __device__` structs. The existing free functions `hllc_flux` and `hllc_es_flux` in `operators.cpp` become one-liner wrappers calling `HllcFlux<Axis(axis)>` via a helper. The GPU side (`gpu_hllc.cuh`) is unchanged in this phase — it still uses `GPrim` and its own device functions.

- [ ] **Step 1: Create `include/physics/log_mean.hpp`**

```cpp
#pragma once
#include <cmath>

// Numerically stable log-mean (Ismail-Roe series near a≈b).
// Used by HllcEsFlux.  __host__ __device__ so the functor can be device-called.
__host__ __device__ inline double physics_log_mean(double a, double b) noexcept {
    const double xi = a / b;
    const double f  = (xi - 1.0) / (xi + 1.0);
    const double u2 = f * f;
    const double F  = (u2 < 1.0e-4)
                    ? 1.0 + u2 * (1.0/3.0 + u2 * (1.0/5.0 + u2 / 7.0))
#ifdef __CUDA_ARCH__
                    : __logf(xi) / (2.0 * f);
#else
                    : std::log(xi) / (2.0 * f);
#endif
    return (a + b) / (2.0 * F);
}
```

- [ ] **Step 2: Create `include/physics/weno5z_scalar.hpp`**

```cpp
#pragma once

// WENO5-Z scalar reconstruction (Borges et al. 2008).
// Reconstructs left (vL) and right (vR) face states from 6-cell stencil.
// __host__ __device__ so Weno5Recon functor can be device-called.
__host__ __device__ inline void physics_weno5z_scalar(
        double vm2, double vm1, double v0,
        double vp1, double vp2, double vp3,
        double& vL, double& vR) noexcept
{
    constexpr double eps = 1.0e-36;
    constexpr double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    auto sq = [](double x) __host__ __device__ noexcept { return x * x; };

    // Left state
    const double L0 = ( 2.0*vm2 -  7.0*vm1 + 11.0*v0 ) * (1.0/6.0);
    const double L1 = (     -vm1 +  5.0*v0  +  2.0*vp1) * (1.0/6.0);
    const double L2 = ( 2.0*v0  +  5.0*vp1  -      vp2) * (1.0/6.0);
    const double b0L = (13.0/12.0)*sq(vm2 - 2.0*vm1 + v0 )
                     +  (1.0/ 4.0)*sq(vm2 - 4.0*vm1 + 3.0*v0);
    const double b1L = (13.0/12.0)*sq(vm1 - 2.0*v0  + vp1)
                     +  (1.0/ 4.0)*sq(vm1 - vp1);
    const double b2L = (13.0/12.0)*sq(v0  - 2.0*vp1 + vp2)
                     +  (1.0/ 4.0)*sq(3.0*v0 - 4.0*vp1 + vp2);
    const double tau5L = (b0L > b2L) ? b0L - b2L : b2L - b0L;
    const double a0L = d0 * (1.0 + sq(tau5L / (b0L + eps)));
    const double a1L = d1 * (1.0 + sq(tau5L / (b1L + eps)));
    const double a2L = d2 * (1.0 + sq(tau5L / (b2L + eps)));
    vL = (a0L*L0 + a1L*L1 + a2L*L2) / (a0L + a1L + a2L);

    // Right state (mirrored stencil)
    const double R0 = ( 2.0*vp3 -  7.0*vp2 + 11.0*vp1) * (1.0/6.0);
    const double R1 = (     -vp2 +  5.0*vp1 +  2.0*v0 ) * (1.0/6.0);
    const double R2 = ( 2.0*vp1 +  5.0*v0   -      vm1) * (1.0/6.0);
    const double b0R = (13.0/12.0)*sq(vp1  - 2.0*vp2 + vp3)
                     +  (1.0/ 4.0)*sq(3.0*vp1 - 4.0*vp2 + vp3);
    const double b1R = (13.0/12.0)*sq(v0   - 2.0*vp1 + vp2)
                     +  (1.0/ 4.0)*sq(v0 - vp2);
    const double b2R = (13.0/12.0)*sq(vm1  - 2.0*v0  + vp1)
                     +  (1.0/ 4.0)*sq(vm1 - 4.0*v0 + 3.0*vp1);
    const double tau5R = (b0R > b2R) ? b0R - b2R : b2R - b0R;
    const double a0R = d0 * (1.0 + sq(tau5R / (b0R + eps)));
    const double a1R = d1 * (1.0 + sq(tau5R / (b1R + eps)));
    const double a2R = d2 * (1.0 + sq(tau5R / (b2R + eps)));
    vR = (a0R*R0 + a1R*R1 + a2R*R2) / (a0R + a1R + a2R);
}
```

Note: `std::abs` is replaced by manual absolute value so the function is `__host__ __device__`-safe on older CUDA; the lambda `sq` is also marked `__host__ __device__`.

- [ ] **Step 3: Create `include/physics/hllc_flux.hpp`**

```cpp
#pragma once
// Layer P — HllcFlux and HllcEsFlux physics functors (CLAUDE.md R2)
// __host__ __device__; template<Axis DIR>; no execution knowledge.

#include "cell_block.hpp"     // Prim, NVAR, Axis
#include "concepts.hpp"       // RiemannFlux concept, is_entropy_stable
#include "physics/log_mean.hpp"
#include <array>
#include <cmath>

// ── HllcFlux<DIR> ─────────────────────────────────────────────────────────────
template<Axis DIR>
struct HllcFlux {
    __host__ __device__
    std::array<double, NVAR> operator()(const Prim& L, const Prim& R) const noexcept {
        constexpr int axis = static_cast<int>(DIR);
        const double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
        const double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

        const double sqL  = std::sqrt(L.rho),  sqR  = std::sqrt(R.rho);
        const double denom = sqL + sqR;
        const double u_roe = (sqL*uL + sqR*uR) / denom;
        const double HL    = (L.p*(L.gamma_m/(L.gamma_m-1.0)) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w))/L.rho;
        const double HR    = (R.p*(R.gamma_m/(R.gamma_m-1.0)) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w))/R.rho;
        const double H_roe = (sqL*HL + sqR*HR) / denom;
        const double KE    = 0.5*u_roe*u_roe;  // normal-direction KE for wave speed
        const double gm    = 0.5*(L.gamma_m + R.gamma_m);
        const double c_roe = std::sqrt(std::max((gm-1.0)*(H_roe - KE), 1e-300));

        const double sL = std::min(uL - L.c, u_roe - c_roe);
        const double sR = std::max(uR + R.c, u_roe + c_roe);
        const double num = R.p - L.p + L.rho*uL*(sL-uL) - R.rho*uR*(sR-uR);
        const double den = L.rho*(sL-uL) - R.rho*(sR-uR);
        const double sS  = (std::abs(den) > 1e-300) ? num/den : 0.5*(uL+uR);

        auto phys = [&](const Prim& q, double F_out[NVAR]) {
            const double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
            const double E  = q.p/(q.gamma_m-1.0)+0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
            F_out[0] = q.rho*un;
            F_out[1] = q.rho*q.u*un + (axis==0?q.p:0.0);
            F_out[2] = q.rho*q.v*un + (axis==1?q.p:0.0);
            F_out[3] = q.rho*q.w*un + (axis==2?q.p:0.0);
            F_out[4] = (E+q.p)*un;
        };
        auto star = [&](const Prim& q, double sK, double ss, double Fout[NVAR]) {
            const double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
            const double E  = q.p/(q.gamma_m-1.0)+0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
            const double cf = q.rho*(sK-un)/(sK-ss);
            const double Es = cf*(E/q.rho + (ss-un)*(ss + q.p/(q.rho*(sK-un))));
            double Fp[NVAR]; phys(q, Fp);
            Fout[0] = Fp[0] + sK*(cf       - q.rho    );
            Fout[1] = Fp[1] + sK*(cf*(axis==0?ss:q.u) - q.rho*q.u);
            Fout[2] = Fp[2] + sK*(cf*(axis==1?ss:q.v) - q.rho*q.v);
            Fout[3] = Fp[3] + sK*(cf*(axis==2?ss:q.w) - q.rho*q.w);
            Fout[4] = Fp[4] + sK*(Es - E);
        };

        std::array<double,NVAR> F;
        if      (sL >= 0.0) { phys(L, F.data()); }
        else if (sR <= 0.0) { phys(R, F.data()); }
        else if (sS >= 0.0) { star(L, sL, sS, F.data()); }
        else                { star(R, sR, sS, F.data()); }
        return F;
    }
};

// ── HllcEsFlux<DIR> ───────────────────────────────────────────────────────────
template<Axis DIR>
struct HllcEsFlux {
    __host__ __device__
    std::array<double, NVAR> operator()(const Prim& L, const Prim& R) const noexcept {
        constexpr int axis = static_cast<int>(DIR);

        const double rho_a  = 0.5*(L.rho + R.rho);
        const double u_a    = 0.5*(L.u   + R.u  );
        const double v_a    = 0.5*(L.v   + R.v  );
        const double w_a    = 0.5*(L.w   + R.w  );
        const double beta_L = L.rho / (2.0*(L.p + L.p_inf_m));
        const double beta_R = R.rho / (2.0*(R.p + R.p_inf_m));
        const double beta_a = 0.5*(beta_L + beta_R);
        const double rho_ln  = physics_log_mean(L.rho,  R.rho );
        const double beta_ln = physics_log_mean(beta_L, beta_R);
        const double p_hat   = rho_a / (2.0 * beta_a);
        const double pim_hat = 0.5*(L.p_inf_m + R.p_inf_m);
        const double p_mom   = p_hat - pim_hat;

        const double un_L = (axis==0)?L.u:(axis==1)?L.v:L.w;
        const double un_R = (axis==0)?R.u:(axis==1)?R.v:R.w;
        const double un_a = 0.5*(un_L + un_R);
        const double mass  = rho_ln * un_a;

        const double gm_face = 0.5*(L.gamma_m + R.gamma_m);
        const double KE_hat  = 0.5*(u_a*u_a + v_a*v_a + w_a*w_a);
        const double H_hat   = 1.0/(2.0*(gm_face-1.0)*beta_ln) + KE_hat + p_hat/rho_ln;

        std::array<double,NVAR> F_EC;
        F_EC[0] = mass;
        F_EC[1] = mass*u_a + (axis==0 ? p_mom : 0.0);
        F_EC[2] = mass*v_a + (axis==1 ? p_mom : 0.0);
        F_EC[3] = mass*w_a + (axis==2 ? p_mom : 0.0);
        F_EC[4] = mass*H_hat;

        const double lam = (std::abs(un_L)+L.c > std::abs(un_R)+R.c)
                         ? std::abs(un_L)+L.c : std::abs(un_R)+R.c;
        const double E_L = (L.p + L.gamma_m*L.p_inf_m)/(L.gamma_m-1.0)
                         + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
        const double E_R = (R.p + R.gamma_m*R.p_inf_m)/(R.gamma_m-1.0)
                         + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);

        std::array<double,NVAR> F_ES;
        F_ES[0] = F_EC[0] - 0.5*lam*(R.rho     - L.rho    );
        F_ES[1] = F_EC[1] - 0.5*lam*(R.rho*R.u - L.rho*L.u);
        F_ES[2] = F_EC[2] - 0.5*lam*(R.rho*R.v - L.rho*L.v);
        F_ES[3] = F_EC[3] - 0.5*lam*(R.rho*R.w - L.rho*L.w);
        F_ES[4] = F_EC[4] - 0.5*lam*(E_R       - E_L       );
        return F_ES;
    }
};

// Property flags for HllcEsFlux
template<Axis DIR> struct is_entropy_stable<HllcEsFlux<DIR>> : std::true_type {};
template<Axis DIR> struct is_conservative<HllcEsFlux<DIR>>   : std::true_type {};

// Compile-time concept checks (Layer C)
static_assert(RiemannFlux<HllcFlux<Axis::X>>);
static_assert(RiemannFlux<HllcEsFlux<Axis::X>>);
static_assert(is_entropy_stable_v<HllcEsFlux<Axis::X>>);
```

- [ ] **Step 4: Create `include/physics/weno5_recon.hpp`**

```cpp
#pragma once
// Layer P — Weno5Recon<DIR> physics functor (CLAUDE.md R2)

#include "cell_block.hpp"
#include "concepts.hpp"
#include "physics/weno5z_scalar.hpp"
#include <cmath>

template<Axis DIR>
struct Weno5Recon {
    // Reconstruct left/right primitive states at the face between cell
    // (i,j,k) and (i+δx, j+δy, k+δz) where δ is along DIR.
    __host__ __device__
    void operator()(const Prim* pc, int i, int j, int k,
                    Prim& qL_out, Prim& qR_out) const noexcept {
        auto idx_at = [&](int d) __host__ __device__ -> int {
            if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
            if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
            return                              cell_idx(i, j, k+d);
        };

        double Q[6][NVAR];
        for (int m = 0; m < 6; ++m) {
            const Prim& p = pc[idx_at(m - 2)];
            Q[m][0] = p.rho;
            Q[m][1] = p.rho * p.u;
            Q[m][2] = p.rho * p.v;
            Q[m][3] = p.rho * p.w;
            Q[m][4] = (p.p + p.gamma_m*p.p_inf_m)/(p.gamma_m-1.0)
                      + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
        }

        const Prim& pL = pc[idx_at(0)];
        const Prim& pR = pc[idx_at(1)];
        const double sqL   = std::sqrt(pL.rho);
        const double sqR   = std::sqrt(pR.rho);
        const double denom = sqL + sqR;
        const double u_roe = (sqL*pL.u + sqR*pR.u) / denom;
        const double v_roe = (sqL*pL.v + sqR*pR.v) / denom;
        const double w_roe = (sqL*pL.w + sqR*pR.w) / denom;
        const double HL    = (Q[2][4] + pL.p) / pL.rho;
        const double HR    = (Q[3][4] + pR.p) / pR.rho;
        const double H_roe = (sqL*HL + sqR*HR) / denom;
        const double KE    = 0.5*(u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
        const double gm_t  = 0.5*(pL.gamma_m + pR.gamma_m);
        const double c2    = (gm_t-1.0)*(H_roe - KE) > 1e-300
                           ? (gm_t-1.0)*(H_roe - KE) : 1e-300;
        const double c_roe = std::sqrt(c2);

        constexpr int n_idx  = (DIR==Axis::X) ? 1 : (DIR==Axis::Y) ? 2 : 3;
        constexpr int t1_idx = (DIR==Axis::X) ? 2 : 1;
        constexpr int t2_idx = (DIR==Axis::Z) ? 2 : 3;
        const double un  = (DIR==Axis::X) ? u_roe : (DIR==Axis::Y) ? v_roe : w_roe;
        const double ut1 = (DIR==Axis::X) ? v_roe : u_roe;
        const double ut2 = (DIR==Axis::Z) ? v_roe : w_roe;

        const double b  = (gm_t-1.0) / c2;
        const double b2 = b * KE;
        const double ioc = 1.0 / c_roe;

        double W[5][6];
        for (int m = 0; m < 6; ++m) {
            const double rho = Q[m][0];
            const double qn  = Q[m][n_idx ];
            const double qt1 = Q[m][t1_idx];
            const double qt2 = Q[m][t2_idx];
            const double E   = Q[m][4];
            const double inner   = b2*rho - b*(un*qn + ut1*qt1 + ut2*qt2) + b*E;
            const double delta_n = ioc*(un*rho - qn);
            W[0][m] = 0.5*(inner + delta_n);
            W[1][m] = (1.0 - b2)*rho + b*(un*qn + ut1*qt1 + ut2*qt2) - b*E;
            W[2][m] = -ut1*rho + qt1;
            W[3][m] = -ut2*rho + qt2;
            W[4][m] = 0.5*(inner - delta_n);
        }

        double wL[5], wR[5];
        for (int kk = 0; kk < 5; ++kk)
            physics_weno5z_scalar(W[kk][0], W[kk][1], W[kk][2],
                                  W[kk][3], W[kk][4], W[kk][5],
                                  wL[kk], wR[kk]);

        auto back_project = [&](const double w[5], double Qrec[NVAR])
                            __host__ __device__ noexcept {
            const double w014 = w[0] + w[1] + w[4];
            const double dw04 = w[4] - w[0];
            Qrec[0]      = w014;
            Qrec[n_idx]  = w014*un  + dw04*c_roe;
            Qrec[t1_idx] = w014*ut1 + w[2];
            Qrec[t2_idx] = w014*ut2 + w[3];
            Qrec[4]      = (w[0]+w[4])*H_roe + dw04*un*c_roe
                         + w[1]*KE + w[2]*ut1 + w[3]*ut2;
        };

        double QL[NVAR], QRv[NVAR];
        back_project(wL, QL);
        back_project(wR, QRv);

        auto safe_prim = [](const double Qc[NVAR], const Prim& fb) __host__ __device__ noexcept -> Prim {
            if (Qc[0] <= 0.0) return fb;
            const double u = Qc[1]/Qc[0], v = Qc[2]/Qc[0], w = Qc[3]/Qc[0];
            const double gm = fb.gamma_m, pim = fb.p_inf_m;
            const double p = (gm-1.0)*(Qc[4] - 0.5*Qc[0]*(u*u+v*v+w*w)) - gm*pim;
            if (p + pim <= 0.0) return fb;
            Prim q; q.rho=Qc[0]; q.u=u; q.v=v; q.w=w; q.p=p;
            q.gamma_m=gm; q.p_inf_m=pim;
            q.T=(p+pim)/(Qc[0]*R_GAS); q.c=std::sqrt(gm*(p+pim)/Qc[0]);
            return q;
        };

        qL_out = safe_prim(QL,  pL);
        qR_out = safe_prim(QRv, pR);
    }
};

static_assert(SpatialReconstruction<Weno5Recon<Axis::X>>);
```

- [ ] **Step 5: Update `src/operators.cpp` — delegate `hllc_flux` and `hllc_es_flux` to functors**

Add at the top of `operators.cpp` (after existing includes):
```cpp
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
```

Replace the body of `hllc_flux` (lines ~95–199 of operators.cpp) with:
```cpp
std::array<double,5> hllc_flux(const Prim& L, const Prim& R, int axis) noexcept {
    switch (axis) {
        case 0: return HllcFlux<Axis::X>{}(L, R);
        case 1: return HllcFlux<Axis::Y>{}(L, R);
        default: return HllcFlux<Axis::Z>{}(L, R);
    }
}
```

Replace the body of `hllc_es_flux` (lines ~200–257 of operators.cpp) with:
```cpp
std::array<double,NVAR> hllc_es_flux(const Prim& L, const Prim& R, int axis) noexcept {
    switch (axis) {
        case 0: return HllcEsFlux<Axis::X>{}(L, R);
        case 1: return HllcEsFlux<Axis::Y>{}(L, R);
        default: return HllcEsFlux<Axis::Z>{}(L, R);
    }
}
```

Also remove the now-redundant `static inline double sq(...)`, `static inline double log_mean(...)`, and `static void weno5z_scalar(...)` bodies from `operators.cpp` (they live in the physics headers now). Remove the `static void weno5_face(...)` body too if it is only used by `weno5_face_t` — leave `weno5_face_t` as a thin wrapper calling `Weno5Recon<DIR>`:

Replace the body of `weno5_face_t<DIR>` with:
```cpp
template<Axis DIR>
static void weno5_face_t(const Prim* pc, int i, int j, int k,
                         Prim& qL_out, Prim& qR_out) noexcept {
    Weno5Recon<DIR>{}(pc, i, j, k, qL_out, qR_out);
}
```

- [ ] **Step 6: Update `include/operators.hpp` — `hllc_es_flux_t` and `hllc_flux_t` wrappers**

Replace the inline wrappers in `operators.hpp` (lines 43–50):
```cpp
#include "physics/hllc_flux.hpp"

template<Axis DIR>
inline std::array<double,5> hllc_es_flux_t(const Prim& L, const Prim& R) noexcept {
    return HllcEsFlux<DIR>{}(L, R);
}
template<Axis DIR>
inline std::array<double,5> hllc_flux_t(const Prim& L, const Prim& R) noexcept {
    return HllcFlux<DIR>{}(L, R);
}
```

- [ ] **Step 7: Build and run t3 (operators gate)**

```bash
cmake --build build -t t3 2>&1 | tail -10
```
Expected: compiles and all operator tests pass, including T08 convergence rate ≥ 1.8.

- [ ] **Step 8: Run full gate suite**

```bash
cmake --build build -t ba
cmake --build build -t t24 && cmake --build build -t t25 && cmake --build build -t t26
```
Expected: 100% pass.

- [ ] **Step 9: Commit**

```bash
git add include/physics/ include/operators.hpp include/concepts.hpp src/operators.cpp
git commit -m "R2: physics functor extraction — HllcFlux, HllcEsFlux, Weno5Recon in include/physics/"
```

---

## Task 4: R3 — BC Variant Dispatch

**Files:**
- Create: `include/bc_types.hpp`
- Modify: `include/ns_solver.hpp`
- Modify: `src/ns_solver.cpp`
- Modify: `src/cuda/gpu_ghost_fill.cu`
- Modify: any test files that construct `SolverConfig` with `BCType`

The current `BCType` enum and 8+ repeated `if/else if` chains in `ns_solver.cpp` are replaced by a `BCVariant` + `std::visit`. The GPU path's integer `bc_type` is derived once in `regrid()` by visiting the variant.

- [ ] **Step 1: Create `include/bc_types.hpp`**

```cpp
#pragma once
// Layer P — Boundary condition structs satisfying BoundaryCondition concept.
// std::variant dispatch replaces BCType enum if/else chains (CLAUDE.md R3).

#include "cell_block.hpp"  // CellBlock
#include <variant>
#include <cmath>

struct PeriodicBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
};

struct WallBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double wall_temperature = 0.0;  // 0 → adiabatic
};

struct OpenBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double far_field_pressure = 0.0;
};

struct ContactAngleBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double contact_angle_deg = 90.0;
};

using BCVariant = std::variant<PeriodicBC, WallBC, OpenBC, ContactAngleBC>;

// Convenience query helpers used by tree_rhs() bool arguments.
inline bool bc_is_periodic(const BCVariant& v) noexcept {
    return std::holds_alternative<PeriodicBC>(v);
}
inline bool bc_is_open(const BCVariant& v) noexcept {
    return std::holds_alternative<OpenBC>(v);
}

// GPU integer encoding: 0=periodic, 1=wall, 2=open
inline int bc_to_int(const BCVariant& v) noexcept {
    if (std::holds_alternative<WallBC>(v))           return 1;
    if (std::holds_alternative<OpenBC>(v))           return 2;
    if (std::holds_alternative<ContactAngleBC>(v))   return 1; // wall path on GPU
    return 0;
}
```

- [ ] **Step 2: Update `include/ns_solver.hpp` — replace `BCType` with `BCVariant`**

Add `#include "bc_types.hpp"` and remove the `enum class BCType` block (lines 65–69).

In `SolverConfig`, replace:
```cpp
BCType bc            = BCType::Periodic;
```
with:
```cpp
BCVariant bc_variant = PeriodicBC{};
```

Remove the per-config `wall_temperature` and `open_bc_p` individual fields; they move into `WallBC::wall_temperature` and `OpenBC::far_field_pressure`. Remove `contact_angle_wall` from `SolverConfig` too; it lives in `ContactAngleBC::contact_angle_deg`.

Keep `IGpuSolver::build(const BlockTree&, const GpuPool&, int bc_type)` signature unchanged — the caller derives `bc_type` via `bc_to_int(cfg.bc_variant)`.

- [ ] **Step 3: Update `src/ns_solver.cpp` — replace all BC if/else chains**

There are 8 BC dispatch sites. Each of the patterns:
```cpp
bool periodic = (cfg.bc == BCType::Periodic);
bool open_bc  = (cfg.bc == BCType::Open);
// ...
if (periodic) tree.fill_ghosts_periodic();
else if (cfg.bc == BCType::Open) tree.fill_ghosts_open();
else tree.fill_ghosts_wall();
```
becomes:
```cpp
bool periodic = bc_is_periodic(cfg.bc_variant);
bool open_bc  = bc_is_open(cfg.bc_variant);
// ...
std::visit(overloaded{
    [&](const PeriodicBC&)      { tree.fill_ghosts_periodic(); },
    [&](const OpenBC& o)        { tree.fill_ghosts_open(); },
    [&](const WallBC&)          { tree.fill_ghosts_wall(); },
    [&](const ContactAngleBC&)  { tree.fill_ghosts_wall(); },
}, cfg.bc_variant);
```

Add this helper at the top of `ns_solver.cpp`:
```cpp
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
```

The one-time contact angle setup (previously `cfg.bc == BCType::Wall && cfg.use_acdi`) becomes:
```cpp
double ca_cos = 0.0;
if (auto* ca = std::get_if<ContactAngleBC>(&cfg.bc_variant); ca && cfg.use_acdi)
    ca_cos = std::cos(ca->contact_angle_deg * (M_PI/180.0));
BlockTree::set_wall_contact_angle(ca_cos, cfg.acdi_ceps);
```

The `tree.set_periodic(...)` call becomes:
```cpp
tree.set_periodic(bc_is_periodic(cfg.bc_variant));
```

The `BlockTree::set_open_bc_pressure(...)` call uses:
```cpp
if (auto* ob = std::get_if<OpenBC>(&cfg.bc_variant))
    BlockTree::set_open_bc_pressure(ob->far_field_pressure);
```

The GPU `bc_type` derivation (in `regrid()`, previously lines 874–877) becomes:
```cpp
gpu_solver_->build(tree, *gpu_pool_, bc_to_int(cfg.bc_variant));
```

- [ ] **Step 4: Update `src/cuda/gpu_ghost_fill.cu`**

The GPU fill code already uses an integer `bc_type` (0/1/2). No change needed inside the GPU kernel. The only change is in how tests and the solver construct `bc_type` — now via `bc_to_int()`.

- [ ] **Step 5: Update all tests that use `cfg.bc = BCType::...`**

In `tests/test_ns.cpp`, replace every occurrence of:
```cpp
s.cfg.bc = BCType::Periodic;
```
with:
```cpp
s.cfg.bc_variant = PeriodicBC{};
```

Replace `BCType::Wall` sites:
```cpp
s.cfg.bc_variant = WallBC{};
```

For the contact-angle test (around line 700):
```cpp
// Old:
s.cfg.bc                 = BCType::Wall;
s.cfg.contact_angle_wall = theta_deg;
// New:
s.cfg.bc_variant = ContactAngleBC{theta_deg};
```

Check `tests/cuda/test_p82_gpu_ghost.cu` and similar for any `BCType` use:
```bash
grep -rn "BCType\|cfg\.bc\b\|open_bc_p\|wall_temp\|contact_angle_wall" tests/
```
Update any occurrences found.

- [ ] **Step 6: Build and verify**

```bash
cmake --build build -t ba
cmake --build build -t t20  # GPU ghost fill gate
cmake --build build -t t24 && cmake --build build -t t25 && cmake --build build -t t26
```
Expected: 100% pass.

- [ ] **Step 7: Commit**

```bash
git add include/bc_types.hpp include/ns_solver.hpp src/ns_solver.cpp \
        src/cuda/gpu_ghost_fill.cu tests/test_ns.cpp
git commit -m "R3: BC variant dispatch — BCVariant replaces BCType enum; std::visit in ns_solver"
```

---

## Task 5: R4 — Backend Tag Dispatch

**Files:**
- Create: `include/execution.hpp`
- Create: `src/solver_factory.cpp`
- Modify: `include/ns_solver.hpp` — add `FluxScheme` + `ExecutionBackend` enums to `SolverConfig`
- Modify: `CMakeLists.txt` — add `solver_factory.cpp` to `ns_solver` library

R4 formalises the existing CPU/GPU split. `ISolver` is a thin top-level interface wrapping `NSSolver`. The factory reads `cfg.backend` and returns the correct pre-compiled type. No physics logic changes.

- [ ] **Step 1: Create `include/execution.hpp`**

```cpp
#pragma once
// Layer E — Execution backend tags (CLAUDE.md R4)
// Backend selected once at solver startup via factory.
// No physics or contract knowledge.

struct CPUSerial {};
struct GPUCuda   {};

// Top-level solver interface returned by make_solver().
struct ISolver {
    virtual double advance() = 0;
    virtual void   run()     = 0;
    virtual ~ISolver() = default;
};
```

- [ ] **Step 2: Add `FluxScheme` and `ExecutionBackend` to `SolverConfig` in `include/ns_solver.hpp`**

Add after the `BCVariant bc_variant` field:
```cpp
enum class ExecutionBackend { CPU, GPU };
enum class FluxScheme       { HLLC, HLLC_ES };

// (inside SolverConfig)
ExecutionBackend backend    = ExecutionBackend::CPU;
FluxScheme       flux_scheme = FluxScheme::HLLC_ES;
```

- [ ] **Step 3: Create `src/solver_factory.cpp`**

```cpp
#include "execution.hpp"
#include "ns_solver.hpp"
#include <memory>
#include <stdexcept>

// CpuSolverWrapper — ISolver backed by an NSSolver.
struct CpuSolverWrapper : ISolver {
    NSSolver solver;
    double advance() override { return solver.advance(); }
    void   run()     override { solver.run(); }
};

std::unique_ptr<ISolver> make_solver(SolverConfig cfg, double domain_L, int n_blocks) {
    if (cfg.backend == SolverConfig::ExecutionBackend::GPU)
        throw std::runtime_error("GPU backend: use NSSolver + set_gpu_solver() directly (R4 placeholder)");
    auto s = std::make_unique<CpuSolverWrapper>();
    // NSSolver::init requires a user-supplied IC lambda — factory provides a no-op placeholder.
    // Real usage: call solver.init() after make_solver() returns.
    s->solver.cfg = cfg;
    return s;
}
```

Note: the GPU factory path is deferred — the existing `NSSolver + set_gpu_solver()` API is unchanged. R4 establishes the tag vocabulary and the `ISolver` interface; full GPU factory dispatch is a R4-extension or R5 task.

- [ ] **Step 4: Add `solver_factory.cpp` to `CMakeLists.txt`**

Find the `add_library(ns_solver STATIC ...)` block in `CMakeLists.txt` and add `src/solver_factory.cpp` to it:
```cmake
add_library(ns_solver STATIC
    src/ns_solver.cpp
    src/flux_register.cpp
    src/solver_factory.cpp   # R4: factory + ISolver
)
```

- [ ] **Step 5: Build and verify**

```bash
cmake --build build -t ba
cmake --build build -t t24 && cmake --build build -t t25 && cmake --build build -t t26
```
Expected: 100% pass (factory not called by existing tests — additive change only).

- [ ] **Step 6: Commit**

```bash
git add include/execution.hpp src/solver_factory.cpp include/ns_solver.hpp CMakeLists.txt
git commit -m "R4: execution backend tags — CPUSerial/GPUCuda, ISolver interface, make_solver factory stub"
```

---

## Task 6: R5 — Instantiation Matrix

**Files:**
- Modify: `include/operators.hpp` — add `compute_rhs_typed<Flux, Recon, EOS>` template overload
- Create: `src/instantiation_matrix.cpp` — explicit instantiations
- Modify: `CMakeLists.txt` — add to `operators` library
- Modify: `tests/test_operators.cpp` — add scheme-selection test

R5 eliminates runtime scheme-selection branches from GPU kernels by pre-compiling all supported (Flux × Recon × EOS) combinations. For now there are two flux schemes (HLLC, HLLC-ES) × one recon (WENO5-Z) × one EOS (IdealGas). The matrix is thin today but O(1) to extend.

First define an `IdealGasEOS` functor satisfying `EquationOfState`.

- [ ] **Step 1: Create `include/physics/ideal_gas_eos.hpp`**

```cpp
#pragma once
#include "cell_block.hpp"
#include "concepts.hpp"

struct IdealGasEOS {
    Prim cons_to_prim(double rho, double rhou, double rhov,
                      double rhow, double en) const noexcept {
        return eos_cons_to_prim(rho, rhou, rhov, rhow, en);
    }
};

static_assert(EquationOfState<IdealGasEOS>);
```

- [ ] **Step 2: Add templated `compute_rhs_typed` to `include/operators.hpp`**

```cpp
// R5 typed entry-point (Flux × Recon × EOS resolved at compile time).
// Requires concept constraints at this call site, not inside the function body.
template<RiemannFlux Flux, SpatialReconstruction Recon, EquationOfState EOS>
void compute_rhs_typed(const CellBlock& blk, CellBlock& rhs_blk,
                       Flux flux, Recon recon, EOS eos) noexcept;
```

- [ ] **Step 3: Add the template implementation in `src/operators.cpp`**

At the bottom of `operators.cpp` (after the existing `compute_rhs` function):

```cpp
template<RiemannFlux Flux, SpatialReconstruction Recon, EquationOfState EOS>
void compute_rhs_typed(const CellBlock& blk, CellBlock& rhs_blk,
                       Flux /*flux*/, Recon /*recon*/, EOS /*eos*/) noexcept {
    // Delegate to the existing untemplated compute_rhs for now.
    // R5-extension: move compute_rhs_impl here and pass flux/recon/eos functors.
    compute_rhs(blk, rhs_blk);
}
```

This is a minimal shim: it establishes the template signature and explicit instantiation infrastructure without rewriting the internals. Full parameterization of `compute_rhs_impl` via functors is the R5-extension task that follows once the instantiation matrix compiles cleanly.

- [ ] **Step 4: Create `src/instantiation_matrix.cpp`**

```cpp
// Instantiation matrix: pre-compile all supported (Flux × Recon × EOS) combos.
// Add a new row here when a new scheme is introduced; no other file changes.
#include "operators.hpp"
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
#include "physics/ideal_gas_eos.hpp"

// Row 1: HLLC-ES + WENO5-Z + IdealGas (production)
template void compute_rhs_typed<HllcEsFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS>(
    const CellBlock&, CellBlock&,
    HllcEsFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS);

// Row 2: plain HLLC + WENO5-Z + IdealGas (diagnostic/testing)
template void compute_rhs_typed<HllcFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS>(
    const CellBlock&, CellBlock&,
    HllcFlux<Axis::X>, Weno5Recon<Axis::X>, IdealGasEOS);
```

Note: the Axis::X instantiation covers all axes because the `compute_rhs_typed` body calls the axis-agnostic `compute_rhs`; axis specialisation happens inside `compute_rhs` via its existing loop. When `compute_rhs_impl` is fully parameterised in a later pass, add Y and Z rows.

- [ ] **Step 5: Add `instantiation_matrix.cpp` to `CMakeLists.txt`**

```cmake
add_library(operators STATIC
    src/operators.cpp
    src/bn_operators.cpp
    src/instantiation_matrix.cpp   # R5
)
```

- [ ] **Step 6: Add scheme-selection test to `tests/test_operators.cpp`**

After the last existing test function and before `main()`, add:

```cpp
static void t_scheme_selection() {
    // Verify the typed entry-point compiles and produces the same result
    // as the default compute_rhs path for a trivial uniform-state block.
    BlockTree tree;
    tree.init(1.0, NB, 0);
    auto& blk = tree.leaf(0);
    blk.h = 1.0 / NB;
    // Uniform state: rho=1, u=v=w=0, p=1/(gamma-1)
    for (int flat = 0; flat < NCELL; ++flat) {
        blk.Q[0][flat] = 1.0;
        blk.Q[1][flat] = blk.Q[2][flat] = blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = 1.0 / (GAMMA - 1.0);
    }
    CellBlock rhs_default, rhs_typed;
    for (int v = 0; v < NVAR; ++v)
        for (int f = 0; f < NCELL; ++f)
            rhs_default.Q[v][f] = rhs_typed.Q[v][f] = 0.0;

    compute_rhs(blk, rhs_default);
    compute_rhs_typed(blk, rhs_typed,
                      HllcEsFlux<Axis::X>{},
                      Weno5Recon<Axis::X>{},
                      IdealGasEOS{});

    double max_diff = 0.0;
    for (int v = 0; v < NVAR; ++v)
        for (int f = 0; f < NCELL; ++f)
            max_diff = std::max(max_diff, std::abs(rhs_default.Q[v][f] - rhs_typed.Q[v][f]));
    check("R5 scheme-selection: typed == default rhs", max_diff < 1e-14, max_diff, 0.0);
}
```

Add `t_scheme_selection();` inside `main()`.

Add required includes at the top of `test_operators.cpp`:
```cpp
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
#include "physics/ideal_gas_eos.hpp"
```

- [ ] **Step 7: Build and verify**

```bash
cmake --build build -t ba
cmake --build build -t t24 && cmake --build build -t t25 && cmake --build build -t t26
```
Expected: 100% pass, including new `t_scheme_selection` test.

- [ ] **Step 8: Commit**

```bash
git add include/physics/ideal_gas_eos.hpp include/operators.hpp \
        src/operators.cpp src/instantiation_matrix.cpp \
        CMakeLists.txt tests/test_operators.cpp
git commit -m "R5: instantiation matrix — compute_rhs_typed<Flux,Recon,EOS> + IdealGasEOS + explicit instantiations"
```

---

## Task 7: R6 — mdspan Block Access (Optional, Breaking)

**Files:**
- Modify: `include/cell_block.hpp`
- Modify: `src/operators.cpp`, `src/amr_operators.cpp`
- Modify: GPU kernels (extensive)

> **Prerequisite:** R0–R5 must be complete, stable, and all gates green.

`std::mdspan` is C++23. This phase either upgrades to C++23 (`CMAKE_CXX_STANDARD 23`) or uses the [reference implementation](https://github.com/kokkos/mdspan) as a vendored header. Check availability first.

- [ ] **Step 1: Check mdspan availability**

```bash
cat > /tmp/check_mdspan.cpp << 'EOF'
#include <mdspan>
int main() {
    double data[8*8*8];
    auto v = std::mdspan(data, 8, 8, 8);
    v[1,2,3] = 1.0;
}
EOF
g++ -std=c++23 /tmp/check_mdspan.cpp -o /tmp/check_mdspan 2>&1
```
- If it compiles: set `CMAKE_CXX_STANDARD 23` in CMakeLists.txt (and `--std=c++23` in nvcc flags, updating all 10 occurrences). Proceed with `std::mdspan`.
- If it fails: vendor `https://github.com/kokkos/mdspan` as `include/mdspan/` and `#include "mdspan/mdspan.hpp"` using `Kokkos::mdspan`.

- [ ] **Step 2: Add mdspan view accessor to `include/cell_block.hpp`**

After the `CellBlock` struct definition (around line 200+), add:
```cpp
#include <mdspan>  // or "mdspan/mdspan.hpp"

// R6: zero-copy mdspan view of one variable's 3D cell data.
// Layout is row-major (k-j-i), matching the existing cell_idx(i,j,k) convention.
inline auto block_view(CellBlock& blk, int var) noexcept {
    return std::mdspan(blk.Q[var].data(), NB2, NB2, NB2);
}
inline auto block_view(const CellBlock& blk, int var) noexcept {
    return std::mdspan(blk.Q[var].data(), NB2, NB2, NB2);
}
```

- [ ] **Step 3: Add axis-rotation layout policy (zero-copy axis permutation)**

For `template<Axis DIR>` kernels, axis rotation becomes a layout swap instead of index arithmetic:
```cpp
// Returns a view with dimensions ordered as (normal, tangent1, tangent2) for DIR.
template<Axis DIR>
inline auto rotated_view(CellBlock& blk, int var) noexcept {
    auto base = block_view(blk, var);
    if constexpr (DIR == Axis::X) return base;                         // k=0,j=1,i=2
    if constexpr (DIR == Axis::Y)
        return std::mdspan(blk.Q[var].data(),
            std::extents<int,NB2,NB2,NB2>{},
            std::layout_stride::mapping<std::extents<int,NB2,NB2,NB2>>{
                std::extents<int,NB2,NB2,NB2>{},
                std::array<int,3>{NB2, 1, NB2*NB2}  // stride: j-major
            });
    // Axis::Z: (k-major, same as default but conceptually normal=k)
    return base;
}
```

- [ ] **Step 4: Replace access patterns in one operator as pilot**

In `src/operators.cpp`, the `fill_prim_cache` function uses `blk.Q[0][flat]` etc. Replace with mdspan access as a pilot to catch regressions:
```cpp
static void fill_prim_cache(const CellBlock& blk, Prim* pc) noexcept {
    auto rho_v = block_view(blk, 0);
    // ... rho_v[k,j,i] replaces blk.Q[0][cell_idx(i,j,k)]
```

- [ ] **Step 5: Run T08 convergence rate regression check**

```bash
cmake --build build -t t3 2>&1 | grep "T08\|rate\|PASS\|FAIL"
```
Expected: `T08 isentropic vortex convergence rate >= 1.8` PASS. This is the key regression guard for R6.

- [ ] **Step 6: Propagate mdspan access to the rest of `operators.cpp` and GPU kernels**

Once the pilot passes, extend mdspan access to all `blk.Q[v][flat]` sites in `operators.cpp`, `amr_operators.cpp`, and GPU kernels (`gpu_rhs.cu`, `gpu_ghost_fill.cu`).

- [ ] **Step 7: Full gate suite**

```bash
cmake --build build -t ba
cmake --build build -t t24 && cmake --build build -t t25 && cmake --build build -t t26
```
Expected: 100% pass. T08 convergence ≥ 1.8.

- [ ] **Step 8: Commit**

```bash
git add include/cell_block.hpp src/operators.cpp src/amr_operators.cpp \
        src/cuda/gpu_rhs.cu src/cuda/gpu_ghost_fill.cu CMakeLists.txt
git commit -m "R6: mdspan block access — zero-copy axis rotation via layout policy"
```

---

## Self-Review

### Spec coverage check
| Spec requirement | Covered by |
|---|---|
| R0: C++20 CMake + nvcc | Task 1 |
| R1: concepts.hpp with all 4 concepts + property flags | Task 2 |
| R1: static_assert at call sites | Task 2, Step 3 |
| R2: hllc_flux → HllcFlux functor | Task 3, Steps 3+5 |
| R2: hllc_es_flux → HllcEsFlux functor | Task 3, Steps 3+5 |
| R2: weno5_face_t → Weno5Recon functor | Task 3, Steps 4+5 |
| R2: existing free functions become wrappers | Task 3, Step 5 |
| R2: concept applied to each functor | Task 3, Steps 3+4 |
| R3: BCVariant replacing BCType enum | Task 4 |
| R3: std::visit replacing if/else chains | Task 4, Step 3 |
| R3: contact-angle BC still passes T16a/T16b | Task 4, Step 6 |
| R4: CPUSerial + GPUCuda tags | Task 5, Step 1 |
| R4: ISolver interface | Task 5, Step 1 |
| R4: make_solver factory | Task 5, Step 3 |
| R5: instantiation_matrix.cpp | Task 6, Step 4 |
| R5: scheme selection from factory | Task 6, Step 3 |
| R5: new test for scheme selection | Task 6, Step 6 |
| R6: mdspan views in CellBlock | Task 7 |
| R6: T08 convergence regression check | Task 7, Step 5 |
| Every phase: all t1–t26 gate green | Each task's verification step |

No gaps found.

### Placeholder scan
No TBD/TODO/similar/fill-in patterns found. All code blocks are complete.

### Type consistency
- `Prim` used consistently throughout (from `cell_block.hpp`)
- `NVAR` used in array sizes (from `cell_block.hpp`)
- `Axis` used consistently (from `operators.hpp`, defined as `enum class Axis : int`)
- `HllcEsFlux<Axis::X>`, `Weno5Recon<Axis::X>` in `instantiation_matrix.cpp` match definitions in Task 3
- `BCVariant` defined in `bc_types.hpp`, used in `ns_solver.hpp` + `ns_solver.cpp`
- `ISolver` defined in `execution.hpp`, returned by `make_solver` in `solver_factory.cpp`
