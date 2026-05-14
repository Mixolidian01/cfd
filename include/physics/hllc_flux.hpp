#pragma once
// Layer P — HllcFlux and HllcEsFlux physics functors (CLAUDE.md R2)

// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif
// __host__ __device__; template<Axis DIR>; no execution knowledge.

#include "mesh/cell_block.hpp"   // Prim, NVAR
#include "schemes/concepts.hpp"     // RiemannFlux concept, is_entropy_stable
#include "mesh/axis.hpp"         // Axis
#include "physics/log_mean.hpp"
#include <array>
#include <cmath>

// ── HllcFlux<DIR> ─────────────────────────────────────────────────────────────
// Standard HLLC Riemann solver (Toro 2009 §10.4, Batten et al. 1997).
// Roe-average wave speed estimates; entropy-consistent near sonic points.
template<Axis DIR>
struct HllcFlux {
    __host__ __device__
    std::array<double, NVAR> operator()(const Prim& L, const Prim& R) const noexcept {
        constexpr int axis = static_cast<int>(DIR);
        const double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
        const double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

        // Roe-averaged state
        const double sqL  = std::sqrt(L.rho);
        const double sqR  = std::sqrt(R.rho);
        const double isq  = 1.0 / (sqL + sqR);
        // P14.1: use mixture γ for enthalpy H = c²/(γ-1) + ½|u|²
        const double HL   = L.c*L.c/(L.gamma_m-1.0) + 0.5*(L.u*L.u + L.v*L.v + L.w*L.w);
        const double HR   = R.c*R.c/(R.gamma_m-1.0) + 0.5*(R.u*R.u + R.v*R.v + R.w*R.w);
        const double uh   = (sqL*L.u + sqR*R.u)*isq;
        const double vh   = (sqL*L.v + sqR*R.v)*isq;
        const double wh   = (sqL*L.w + sqR*R.w)*isq;
        const double Hh   = (sqL*HL  + sqR*HR )*isq;
        const double gm_face = 0.5*(L.gamma_m + R.gamma_m);
        const double c2h  = (gm_face-1.0)*(Hh - 0.5*(uh*uh + vh*vh + wh*wh));
        const double ch   = (c2h > 0.0) ? std::sqrt(c2h) : 0.5*(L.c + R.c);
        const double u_h  = (axis==0) ? uh : (axis==1) ? vh : wh;

        const double sL   = std::min(uL - L.c, u_h - ch);
        const double sR   = std::max(uR + R.c, u_h + ch);

        const double numer = R.p - L.p + L.rho*uL*(sL - uL) - R.rho*uR*(sR - uR);
        const double denom = L.rho*(sL - uL) - R.rho*(sR - uR);
        const double sStar = (std::abs(denom) > 1e-300) ? numer/denom : 0.5*(uL + uR);

        // Physical flux helper (no nested __host__ __device__ lambda needed)
        auto phys_flux = [&](const Prim& q) -> std::array<double,NVAR> {
            const double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
            // P14.1: stiffened gas E = (p + γ·p∞)/(γ-1) + KE
            const double E  = (q.p + q.gamma_m*q.p_inf_m)/(q.gamma_m-1.0)
                            + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
            return {q.rho*un,
                    q.rho*q.u*un + (axis==0?q.p:0.0),
                    q.rho*q.v*un + (axis==1?q.p:0.0),
                    q.rho*q.w*un + (axis==2?q.p:0.0),
                    (E+q.p)*un};
        };

        auto star_flux = [&](const Prim& q, double sK, double sS)
            -> std::array<double,NVAR>
        {
            const double un    = (axis==0)?q.u:(axis==1)?q.v:q.w;
            const double E     = (q.p + q.gamma_m*q.p_inf_m)/(q.gamma_m-1.0)
                               + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);  // P14.1
            const double coeff = q.rho * (sK - un) / (sK - sS);
            const double rho_s  = coeff;
            const double rhou_s = coeff * (axis==0 ? sS : q.u);
            const double rhov_s = coeff * (axis==1 ? sS : q.v);
            const double rhow_s = coeff * (axis==2 ? sS : q.w);
            const double E_s    = coeff * (E/q.rho + (sS - un)*(sS + q.p/(q.rho*(sK-un))));

            auto F = phys_flux(q);
            const double rho  = q.rho;
            const double rhou = rho*q.u, rhov = rho*q.v, rhow = rho*q.w;
            return {F[0] + sK*(rho_s  - rho ),
                    F[1] + sK*(rhou_s - rhou),
                    F[2] + sK*(rhov_s - rhov),
                    F[3] + sK*(rhow_s - rhow),
                    F[4] + sK*(E_s    - E   )};
        };

        if      (sL >= 0.0)     return phys_flux(L);
        else if (sR <= 0.0)     return phys_flux(R);
        else if (sStar >= 0.0)  return star_flux(L, sL, sStar);
        else                    return star_flux(R, sR, sStar);
    }
};

// ── HllcEsFlux<DIR> ───────────────────────────────────────────────────────────
// Entropy-stable HLLC-ES flux (Chandrashekar 2013).
// EC base: log-mean ρ_ln, β_ln where β=ρ/(2(p+p∞)); LF scalar dissipation.
template<Axis DIR>
struct HllcEsFlux {
    __host__ __device__
    std::array<double, NVAR> operator()(const Prim& L, const Prim& R) const noexcept {
        constexpr int axis = static_cast<int>(DIR);

        // Arithmetic means
        const double rho_a  = 0.5*(L.rho + R.rho);
        const double u_a    = 0.5*(L.u   + R.u  );
        const double v_a    = 0.5*(L.v   + R.v  );
        const double w_a    = 0.5*(L.w   + R.w  );
        // P14.1c SG EOS: β = ρ/(2(p+p∞)); ideal gas: p∞=0 → β = ρ/(2p)
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

        // LF scalar dissipation
        const double lam = std::max(std::abs(un_L)+L.c, std::abs(un_R)+R.c);
        // P14.1: stiffened-gas E
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

// ── Property flags ────────────────────────────────────────────────────────────
template<Axis DIR> struct is_conservative<HllcFlux<DIR>>     : std::true_type {};
template<Axis DIR> struct is_entropy_stable<HllcEsFlux<DIR>> : std::true_type {};
template<Axis DIR> struct is_conservative<HllcEsFlux<DIR>>   : std::true_type {};

// ── Compile-time concept checks (Layer C) ─────────────────────────────────────
static_assert(RiemannFlux<HllcFlux<Axis::X>>);
static_assert(RiemannFlux<HllcEsFlux<Axis::X>>);
static_assert(is_entropy_stable_v<HllcEsFlux<Axis::X>>);
