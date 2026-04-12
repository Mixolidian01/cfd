#pragma once
// gpu_hllc.cuh — HLLC Riemann flux, device-inline
//
// Called once per interior face per block per RK stage.
// All inputs/outputs are in registers.  No shared memory, no global loads.
// axis: 0=x, 1=y, 2=z.

#include "gpu_block.cuh"

__host__ __device__ __forceinline__
void gpu_hllc_flux(const GPrim& L, const GPrim& R, int axis,
                   double F[GPU_NVAR]) noexcept
{
    // Normal velocity
    double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
    double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

    // Wave speed estimates (Einfeldt)
    double sL = fmin(uL - L.c, uR - R.c);
    double sR = fmax(uL + L.c, uR + R.c);

    // Contact wave speed S*
    double num = R.p - L.p + L.rho*uL*(sL-uL) - R.rho*uR*(sR-uR);
    double den = L.rho*(sL-uL) - R.rho*(sR-uR);
    double sS  = (fabs(den) > 1e-300) ? num/den : 0.5*(uL+uR);

    // Physical flux for state q
    auto phys = [&](const GPrim& q, double F_out[GPU_NVAR]) {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GPU_GAMMA-1.0)
                  + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        F_out[0] = q.rho*un;
        F_out[1] = q.rho*q.u*un + (axis==0?q.p:0.0);
        F_out[2] = q.rho*q.v*un + (axis==1?q.p:0.0);
        F_out[3] = q.rho*q.w*un + (axis==2?q.p:0.0);
        F_out[4] = (E+q.p)*un;
    };

    // HLLC star flux for side K (sK, sS)
    auto star = [&](const GPrim& q, double sK, double ss,
                    double Fout[GPU_NVAR]) {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GPU_GAMMA-1.0)
                  + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        double cf  = q.rho*(sK-un)/(sK-ss);
        double rho_s  = cf;
        double rhou_s = cf * (axis==0 ? ss : q.u);
        double rhov_s = cf * (axis==1 ? ss : q.v);
        double rhow_s = cf * (axis==2 ? ss : q.w);
        double E_s    = cf * (E/q.rho + (ss-un)*(ss + q.p/(q.rho*(sK-un))));
        double Fp[GPU_NVAR];
        phys(q, Fp);
        Fout[0] = Fp[0] + sK*(rho_s  - q.rho);
        Fout[1] = Fp[1] + sK*(rhou_s - q.rho*q.u);
        Fout[2] = Fp[2] + sK*(rhov_s - q.rho*q.v);
        Fout[3] = Fp[3] + sK*(rhow_s - q.rho*q.w);
        Fout[4] = Fp[4] + sK*(E_s    - E);
    };

    if      (sL >= 0.0)    phys(L, F);
    else if (sR <= 0.0)    phys(R, F);
    else if (sS >= 0.0)    star(L, sL, sS, F);
    else                   star(R, sR, sS, F);
}
