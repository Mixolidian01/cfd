#pragma once
// TENO7-A scalar reconstruction (Fu, Hu, Adams 2016/2019).
// 7-point stencil; 4 sub-stencils of 4 points each (cubic polynomials).
// Smoothness indicators: Balsara & Shu (2000) WENO7, divided by 240.
// Global indicator: τ₇ = |β₀ − β₃| (outer sub-stencil pair).
// Cutoff: χₖ = (1+τ₇/(βₖ+ε))^6, include substencil k if γₖ = χₖ/Σχ ≥ C_T.
// Optimal weights: d₀=1/35, d₁=12/35, d₂=18/35, d₃=4/35.
// Smooth limit: (−3,25,−101,407,70,34,−12)/420 ← 7th-order combination.
// Falls back to TENO5-A (5-point) at faces where d=±3 is outside [0,NB2).

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

__host__ __device__ inline void physics_teno7_scalar(
        double vm3, double vm2, double vm1, double v0,
        double vp1, double vp2, double vp3,
        double& vL, double& vR) noexcept
{
#if !defined(__CUDA_ARCH__) && (defined(__NVCC__) || defined(__clang__))
#pragma STDC FP_CONTRACT OFF
#endif
    constexpr double eps  = 1.0e-36;
    constexpr double CT   = 1.0e-6;
    constexpr double d0   = 1.0/35.0, d1 = 12.0/35.0, d2 = 18.0/35.0, d3 = 4.0/35.0;
    constexpr double i12  = 1.0/12.0;
    constexpr double i240 = 1.0/240.0;

    auto sq = [](double x) noexcept -> double { return x * x; };

    // one_sided: left-state reconstruction at face between positions d=0 and d=1.
    //   Left  call: (a,b,c,d,e,f,g) = (vm3,vm2,vm1,v0,vp1,vp2,vp3)
    //   Right call: (a,b,c,d,e,f,g) = (vp3,vp2,vp1,v0,vm1,vm2,vm3)  ← mirrored
    auto one_sided = [&](double a, double b, double c, double d,
                         double e, double f, double g) noexcept -> double {
        // Sub-stencil polynomials at face (Shu 1998 Table 2.2, k=4):
        const double s0 = i12*(-3.0*a + 13.0*b - 23.0*c + 25.0*d);   // {a,b,c,d}
        const double s1 = i12*( 1.0*b -  5.0*c + 13.0*d +  3.0*e);   // {b,c,d,e}
        const double s2 = i12*(-1.0*c +  7.0*d +  7.0*e -  1.0*f);   // {c,d,e,f}
        const double s3 = i12*(25.0*d - 23.0*e + 13.0*f -  3.0*g);   // {d,e,f,g}

        // Balsara-Shu (2000) WENO7 smoothness indicators (normalised by /240):
        const double b0 = i240*(547.0*sq(a) - 3882.0*a*b + 4642.0*a*c - 1854.0*a*d
                               + 7043.0*sq(b) - 17246.0*b*c + 7042.0*b*d
                               + 11003.0*sq(c) - 9402.0*c*d + 2107.0*sq(d));
        const double b1 = i240*(267.0*sq(b) - 1642.0*b*c + 1602.0*b*d - 494.0*b*e
                               + 2843.0*sq(c) - 5966.0*c*d + 1922.0*c*e
                               + 3443.0*sq(d) - 2522.0*d*e + 547.0*sq(e));
        const double b2 = i240*(267.0*sq(c) - 1642.0*c*d + 1602.0*c*e - 494.0*c*f
                               + 2843.0*sq(d) - 5966.0*d*e + 1922.0*d*f
                               + 3443.0*sq(e) - 2522.0*e*f + 547.0*sq(f));
        const double b3 = i240*(2107.0*sq(d) - 9402.0*d*e + 7042.0*d*f - 1854.0*d*g
                               + 11003.0*sq(e) - 17246.0*e*f + 4642.0*e*g
                               + 7043.0*sq(f) - 3882.0*f*g + 547.0*sq(g));

        // τ₇ = |β₀ − β₃|; χₖ = (1 + τ₇/(βₖ+ε))^6 via three squarings
        const double tau7 = (b0 > b3) ? b0 - b3 : b3 - b0;
        auto chi6 = [&](double bk) noexcept -> double {
            double r = 1.0 + tau7 / (bk + eps);
            r *= r; r *= r; r *= r;
            return r;
        };
        const double c0 = chi6(b0), c1 = chi6(b1), c2 = chi6(b2), c3 = chi6(b3);
        const double ci = 1.0 / (c0 + c1 + c2 + c3 + eps);

        // γₖ = χₖ/Σχ; include substencil if γₖ ≥ C_T
        const double w0 = (c0*ci >= CT) ? d0 : 0.0;
        const double w1 = (c1*ci >= CT) ? d1 : 0.0;
        const double w2 = (c2*ci >= CT) ? d2 : 0.0;
        const double w3 = (c3*ci >= CT) ? d3 : 0.0;
        const double ws = w0 + w1 + w2 + w3;

        if (ws > 0.0)
            return (w0*s0 + w1*s1 + w2*s2 + w3*s3) / ws;

        // ENO fallback: minimum-β sub-stencil
        if (b0 <= b1 && b0 <= b2 && b0 <= b3) return s0;
        if (b1 <= b2 && b1 <= b3)              return s1;
        if (b2 <= b3)                           return s2;
        return s3;
    };

    vL = one_sided(vm3, vm2, vm1, v0,  vp1, vp2, vp3);   // left state
    vR = one_sided(vp3, vp2, vp1, v0,  vm1, vm2, vm3);   // right state (mirrored)
}
