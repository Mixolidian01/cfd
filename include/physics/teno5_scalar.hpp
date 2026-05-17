#pragma once

// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

// TENO5-A scalar reconstruction (Fu, Hu, Adams 2016/2019).
// Same 6-point stencil as WENO5-Z; replaces WENO nonlinear weights with a
// hard cutoff: γₖ = χₖ/Σχ where χₖ = (1 + τ₅/(βₖ+ε))^q.
// Smooth stencils (large χ) are included; non-smooth (small χ → small γ) are
// excluded if γₖ < C_T.  Active stencils receive their exact optimal weight.
// In fully smooth regions: all γₖ ≈ 1/3 >> C_T → all included → exactly the
// 5th-order optimal polynomial (smaller constant than WENO5-Z).
// __host__ __device__ so Teno5Recon and GPU device functions can use this.
__host__ __device__ inline void physics_teno5_scalar(
        double vm2, double vm1, double v0,
        double vp1, double vp2, double vp3,
        double& vL, double& vR) noexcept
{
    // Disable FMA contraction so βk, τ5, χk are identical on CPU (GCC) and GPU (ptxas).
    // __CUDA_ARCH__ is defined only during device compilation; device FMA is controlled by
    // --fmad=false (in _NVCC_BASE). For host code (both GCC standalone and NVCC host pass)
    // we use the C standard pragma to enforce strict multiply-add rounding.
#ifndef __CUDA_ARCH__
#pragma STDC FP_CONTRACT OFF
#endif
    constexpr double eps  = 1.0e-36;
    constexpr double CT   = 1.0e-5;   // cutoff threshold (q=6 integer exponent)
    constexpr double d0   = 0.1, d1 = 0.6, d2 = 0.3;

    auto sq  = [](double x) noexcept -> double { return x * x; };

    // One-sided TENO5-A reconstruction (left state); right = mirrored call.
    // Inputs: a=vm2..e=vp2 are the five-cell stencil for the left state.
    auto one_sided = [&](double a, double b, double c, double d, double e) noexcept
        -> double
    {
        // Sub-stencil polynomials (identical to WENO5).
        const double s0 = ( 2.0*a -  7.0*b + 11.0*c) * (1.0/6.0);
        const double s1 = (     -b +  5.0*c +  2.0*d) * (1.0/6.0);
        const double s2 = ( 2.0*c +  5.0*d -      e) * (1.0/6.0);

        // Smoothness indicators (identical to WENO5-Z).
        const double b0 = (13.0/12.0)*sq(a - 2.0*b + c) + (1.0/4.0)*sq(a - 4.0*b + 3.0*c);
        const double b1 = (13.0/12.0)*sq(b - 2.0*c + d) + (1.0/4.0)*sq(b - d);
        const double b2 = (13.0/12.0)*sq(c - 2.0*d + e) + (1.0/4.0)*sq(3.0*c - 4.0*d + e);

        const double tau5 = (b0 > b2) ? b0 - b2 : b2 - b0;

        // χₖ = (1 + τ₅/(βₖ+ε))^q — large for smooth substencils, ≈1 for non-smooth.
        auto chi6 = [&](double bk) noexcept -> double {
            double r = 1.0 + tau5 / (bk + eps);
            r *= r; r *= r; r *= r;  // r^6 via squaring (3 muls)
            return r;
        };
        const double c0 = chi6(b0), c1 = chi6(b1), c2 = chi6(b2);
        const double csum_inv = 1.0 / (c0 + c1 + c2 + eps);

        // γₖ = χₖ / Σχ; include substencil if γₖ ≥ C_T.
        const double w0 = (c0 * csum_inv >= CT) ? d0 : 0.0;
        const double w1 = (c1 * csum_inv >= CT) ? d1 : 0.0;
        const double w2 = (c2 * csum_inv >= CT) ? d2 : 0.0;
        const double wsum = w0 + w1 + w2;

        if (wsum > 0.0)
            return (w0*s0 + w1*s1 + w2*s2) / wsum;

        // ENO fallback: all substencils excluded (very strong discontinuity).
        if (b0 <= b1 && b0 <= b2) return s0;
        if (b1 <= b2)              return s1;
        return s2;
    };

    vL = one_sided(vm2, vm1, v0,  vp1, vp2);          // left  state
    vR = one_sided(vp3, vp2, vp1, v0,  vm1);          // right state (mirrored)
}
