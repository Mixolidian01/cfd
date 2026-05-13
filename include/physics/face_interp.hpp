#pragma once
#include <cmath>
#include "axis.hpp"
#include "log_mean.hpp"

// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

// ── Mean policies (Layer P) ───────────────────────────────────────────────────
// Stateless binary functors: Mean(a, b) where a = f(cell), b = f(neighbor).
// All are trivially copyable → safe as constexpr members in GPU kernels.

struct ArithmeticMean {
    __host__ __device__
    double operator()(double a, double b) const noexcept { return 0.5*(a + b); }
};

struct GeometricMean {
    // Requires a, b > 0.
    __host__ __device__
    double operator()(double a, double b) const noexcept { return std::sqrt(a * b); }
};

struct LogMean {
    // Numerically stable log-mean (Ismail-Roe series); degenerates to arithmetic
    // when a≈b. Required by the Chandrashekar entropy-conservative flux for ρ_ln, β_ln.
    __host__ __device__
    double operator()(double a, double b) const noexcept {
        return physics_log_mean(a, b);
    }
};

struct HarmonicMean {
    // Requires a, b > 0. Appropriate for diffusivities in heterogeneous media.
    __host__ __device__
    double operator()(double a, double b) const noexcept { return 2.0*a*b / (a + b); }
};

// ── FaceInterp<DIR, Mean> (Layer P) ──────────────────────────────────────────
// Face-value interpolation functor. operator()(f, i, j, k) returns Mean applied
// to the two cell values bracketing face DIR+½:
//   X: Mean(f(i,j,k), f(i+1,j,k))
//   Y: Mean(f(i,j,k), f(i,j+1,k))
//   Z: Mean(f(i,j,k), f(i,j,k+1))
//
// Convention: FaceInterp<DIR>{}(f, i, j, k) → face DIR+½.
//             FaceInterp<DIR>{}(f, i-1, j, k) → face DIR-½.

template<Axis DIR, typename Mean = ArithmeticMean>
struct FaceInterp {
    template<typename Field>
    __host__ __device__
    double operator()(Field f, int i, int j, int k) const noexcept {
        constexpr Mean m{};
        if constexpr (DIR == Axis::X) return m(f(i,j,k), f(i+1,j,k));
        else if constexpr (DIR == Axis::Y) return m(f(i,j,k), f(i,j+1,k));
        else                               return m(f(i,j,k), f(i,j,k+1));
    }
};
