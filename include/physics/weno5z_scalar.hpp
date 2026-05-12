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

// WENO5-Z scalar reconstruction (Borges et al. 2008).
// __host__ __device__ so Weno5Recon functor can be device-called.
__host__ __device__ inline void physics_weno5z_scalar(
        double vm2, double vm1, double v0,
        double vp1, double vp2, double vp3,
        double& vL, double& vR) noexcept
{
    constexpr double eps = 1.0e-36;
    constexpr double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    auto sq = [](double x) noexcept -> double { return x * x; };

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
    // tau5 uses ternary for __host__ __device__ safety (no std::abs)
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
