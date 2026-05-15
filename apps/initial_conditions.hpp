#pragma once
// apps-internal header — include after Config is defined (see simulate.cpp).
// Provides fill_leaves() and build_ic() for the simulation entry point.

#include "solver/ns_solver.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

// Re-fill all leaf blocks with IC (needed after manual refinement).
inline void fill_leaves(NSSolver& s,
                        const std::function<Prim(double, double, double)>& ic)
{
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int ii = 0; ii < NB2; ++ii) {
            double x = blk.ox + (ii - NG + 0.5) * blk.h;
            double y = blk.oy + (j  - NG + 0.5) * blk.h;
            double z = blk.oz + (k  - NG + 0.5) * blk.h;
            Prim   p = ic(x, y, z);
            int    idx = cell_idx(ii, j, k);
            blk.Q[0][idx] = p.rho;
            blk.Q[1][idx] = p.rho * p.u;
            blk.Q[2][idx] = p.rho * p.v;
            blk.Q[3][idx] = p.rho * p.w;
            blk.Q[4][idx] = p.p / (GAMMA - 1.0)
                           + 0.5 * p.rho * (p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
}

// IC factory — Config must be visible at the include site.
inline std::function<Prim(double, double, double)>
build_ic(const Config& cfg)
{
    std::string name = cfg.str("ic", "uniform");

    if (name == "uniform") {
        double rho = cfg.d("ic_rho", 1.0);
        double p   = cfg.d("ic_p",   1.0);
        double u   = cfg.d("ic_u",   0.0);
        double v   = cfg.d("ic_v",   0.0);
        double w   = cfg.d("ic_w",   0.0);
        return [=](double, double, double) -> Prim {
            Prim q{};
            q.rho = rho; q.u = u; q.v = v; q.w = w; q.p = p;
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "sod") {
        double rho_l = cfg.d("ic_rho_l", 1.0);
        double p_l   = cfg.d("ic_p_l",   1.0);
        double rho_r = cfg.d("ic_rho_r", 0.125);
        double p_r   = cfg.d("ic_p_r",   0.1);
        double x0    = cfg.d("ic_x0",    0.5);
        return [=](double x, double, double) -> Prim {
            Prim q{};
            bool left = (x < x0);
            q.rho = left ? rho_l : rho_r;
            q.p   = left ? p_l   : p_r;
            q.u = 0.0; q.v = 0.0; q.w = 0.0;
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "taylor_green") {
        double ma   = cfg.d("ic_ma",   0.1);
        double v0   = cfg.d("ic_v0",   1.0);
        double rho0 = cfg.d("ic_rho0", 1.0);
        double p0   = rho0 * v0 * v0 / (GAMMA * ma * ma);
        return [=](double x, double y, double z) -> Prim {
            Prim q{};
            q.rho = rho0;
            q.u   =  v0 * std::sin(x) * std::cos(y) * std::cos(z);
            q.v   = -v0 * std::cos(x) * std::sin(y) * std::cos(z);
            q.w   =  0.0;
            q.p   = p0 + rho0*v0*v0/16.0
                       * (std::cos(2.0*x) + std::cos(2.0*y))
                       * (std::cos(2.0*z) + 2.0);
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "kelvin_helmholtz") {
        double du    = cfg.d("ic_du",    1.0);
        double eps   = cfg.d("ic_eps",   0.01);
        double delta = cfg.d("ic_delta", 0.025);
        double p0    = cfg.d("ic_p0",    2.5);
        double L     = cfg.d("domain_L", 1.0);
        return [=](double x, double y, double) -> Prim {
            Prim q{};
            bool dense = (y >= 0.25*L && y <= 0.75*L);
            double sign = dense ? 1.0 : -1.0;
            double d1 = std::tanh((y - 0.25*L) / delta);
            double d2 = std::tanh((0.75*L - y) / delta);
            double taper = 0.5 * (1.0 + d1) * 0.5 * (1.0 + d2);
            q.rho = dense ? 2.0 : 1.0;
            q.u   = sign * 0.5 * du;
            q.v   = eps * du * std::sin(2.0 * M_PI * x / L) * taper;
            q.w   = 0.0;
            q.p   = p0;
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "isentropic_vortex") {
        double mach = cfg.d("ic_mach", 0.3);
        double rc   = cfg.d("ic_rc",   0.1);
        double L    = cfg.d("domain_L", 1.0);
        double p_inf   = 1.0 / GAMMA;
        double rho_inf = 1.0;
        double u_inf   = mach;
        double beta    = mach;
        double xc = 0.5*L, yc = 0.5*L;
        return [=](double x, double y, double) -> Prim {
            double dx = x - xc, dy = y - yc;
            double r2 = (dx*dx + dy*dy) / (rc*rc);
            double f  = beta * beta * (GAMMA - 1.0) / (8.0 * M_PI * M_PI * GAMMA);
            double fac = std::exp(1.0 - r2);
            double T_ratio = 1.0 - f * fac;
            double rho = rho_inf * std::pow(T_ratio, 1.0/(GAMMA-1.0));
            double p   = p_inf   * std::pow(T_ratio, GAMMA/(GAMMA-1.0));
            double du  = -beta / (2.0 * M_PI * rc) * dy * std::exp(0.5*(1.0 - r2));
            double dv  =  beta / (2.0 * M_PI * rc) * dx * std::exp(0.5*(1.0 - r2));
            Prim q{};
            q.rho = rho; q.p = p;
            q.u = u_inf + du; q.v = dv; q.w = 0.0;
            q.T = q.p / (q.rho * R_GAS);
            q.c = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    fprintf(stderr, "simulate: unknown ic '%s'. "
            "Valid: uniform sod taylor_green kelvin_helmholtz isentropic_vortex\n",
            name.c_str());
    exit(1);
}
