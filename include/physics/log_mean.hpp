#pragma once
#include <cmath>

// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

// Numerically stable log-mean (Ismail-Roe series near a≈b).
// Used by HllcEsFlux. __host__ __device__ so functor can be device-called.
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
