#pragma once
// gpu_hllc.cuh — HLLC, HLLC-ES, and KEP Riemann fluxes, device-inline
//
// Called once per interior face per block per RK stage.
// All inputs/outputs are in registers.  No shared memory, no global loads.
// axis: 0=x, 1=y, 2=z.

#include "gpu_block.cuh"

// ── Log-mean (Ismail & Roe 2009, numerically stable for xi near 1) ────────────
__device__ __host__ __forceinline__
double gpu_log_mean(double a, double b) noexcept {
    const double xi = a / b;
    const double f  = (xi - 1.0) / (xi + 1.0);
    const double u2 = f * f;
    const double F  = (u2 < 1.0e-4)
                    ? 1.0 + u2 * (1.0/3.0 + u2 * (1.0/5.0 + u2 / 7.0))
                    : log(xi) / (2.0 * f);
    return (a + b) / (2.0 * F);
}

// ── Entropy-stable HLLC-ES flux (Chandrashekar 2013) ─────────────────────────
__device__ __host__ __forceinline__
void gpu_hllc_es_flux(const GPrim& L, const GPrim& R, int axis,
                      double F[GPU_NVAR]) noexcept {
    const double rho_a  = 0.5*(L.rho + R.rho);
    const double u_a    = 0.5*(L.u   + R.u  );
    const double v_a    = 0.5*(L.v   + R.v  );
    const double w_a    = 0.5*(L.w   + R.w  );
    const double beta_L = L.rho / (2.0*L.p);
    const double beta_R = R.rho / (2.0*R.p);
    const double beta_a = 0.5*(beta_L + beta_R);
    const double rho_ln  = gpu_log_mean(L.rho,  R.rho );
    const double beta_ln = gpu_log_mean(beta_L, beta_R);
    const double p_hat   = rho_a / (2.0 * beta_a);
    const double un_L = (axis==0)?L.u:(axis==1)?L.v:L.w;
    const double un_R = (axis==0)?R.u:(axis==1)?R.v:R.w;
    const double un_a = 0.5*(un_L + un_R);
    const double mass = rho_ln * un_a;
    const double KE_hat = 0.5*(u_a*u_a + v_a*v_a + w_a*w_a);
    const double H_hat  = 1.0/(2.0*(GPU_GAMMA-1.0)*beta_ln) + KE_hat + p_hat/rho_ln;
    F[0] = mass;
    F[1] = mass*u_a + (axis==0 ? p_hat : 0.0);
    F[2] = mass*v_a + (axis==1 ? p_hat : 0.0);
    F[3] = mass*w_a + (axis==2 ? p_hat : 0.0);
    F[4] = mass*H_hat;
    const double lam = fmax(fabs(un_L)+L.c, fabs(un_R)+R.c);
    const double E_L = L.p/(GPU_GAMMA-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R = R.p/(GPU_GAMMA-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    F[0] -= 0.5*lam*(R.rho     - L.rho    );
    F[1] -= 0.5*lam*(R.rho*R.u - L.rho*L.u);
    F[2] -= 0.5*lam*(R.rho*R.v - L.rho*L.v);
    F[3] -= 0.5*lam*(R.rho*R.w - L.rho*L.w);
    F[4] -= 0.5*lam*(E_R       - E_L      );
}

// ── KE-preserving flux (Pirozzoli 2011) ──────────────────────────────────────
__device__ __host__ __forceinline__
void gpu_kep_flux(const GPrim& L, const GPrim& R, int axis,
                  double F[GPU_NVAR]) noexcept {
    const double rho_a = 0.5*(L.rho + R.rho);
    const double u_a   = 0.5*(L.u   + R.u  );
    const double v_a   = 0.5*(L.v   + R.v  );
    const double w_a   = 0.5*(L.w   + R.w  );
    const double p_a   = 0.5*(L.p   + R.p  );
    const double E_L   = L.p/(GPU_GAMMA-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R   = R.p/(GPU_GAMMA-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    const double H_a   = 0.5*((E_L+L.p)/L.rho + (E_R+R.p)/R.rho);
    const double un_a  = (axis==0)?u_a:(axis==1)?v_a:w_a;
    const double mass  = rho_a * un_a;
    F[0] = mass;
    F[1] = mass*u_a + (axis==0 ? p_a : 0.0);
    F[2] = mass*v_a + (axis==1 ? p_a : 0.0);
    F[3] = mass*w_a + (axis==2 ? p_a : 0.0);
    F[4] = mass*H_a;
}

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
