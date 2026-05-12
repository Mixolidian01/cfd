// B.6 benchmark — Kelvin-Helmholtz instability (2D, density-stratified)
//
// Validates the hybrid WENO5/KEP convective scheme and the Ducros sensor
// on a classic shear-flow instability with roll-up.
//
// Reference: Chandrashekar (2013) Commun. Comput. Phys. 14(5), 1252–1286
//            Pirozzoli (2011) J. Comput. Phys. 230(9), 3270–3271
//
// Setup (2D in xy-plane, periodic domain [0,1]²×[0,1]):
//   Dense layer  (0.25 ≤ y ≤ 0.75): ρ=2, u=+ΔU/2
//   Light layers (y<0.25, y>0.75):   ρ=1, u=−ΔU/2
//   ΔU = 1.0,  ε=0.01 (transverse perturbation), p=2.5 (uniform)
//   Smooth interface: tanh profile with δ=0.025 to avoid Gibbs artefacts
//   Level-2 uniform refinement → 32³ cells, h≈0.0313
//   BCType::Periodic, CFL=0.4, t_end=2.0
//
// Physical analysis:
//   Ma_conv = ΔU/2 / c  where c=sqrt(γ p/ρ_mean)=sqrt(1.4×2.5/1.5)≈1.528 → Ma≈0.33
//   Linear growth rate (sharp-interface, stratified):
//     σ = k * sqrt(ρ₁ρ₂)/(ρ₁+ρ₂) * ΔU  for k=2π (one full wavelength in x)
//     With ρ₁=2, ρ₂=1: σ = 2π * sqrt(2)/3 * 1 ≈ 2.96
//   Roll-up expected at t ≈ 1–2
//
// Gate criteria:
//   B6a  Transverse KE grew ≥ 20× from initial    (exponential linear growth)
//   B6b  Time of peak E_v ∈ [0.3, 5.0]            (roll-up timing)
//   B6c  No negative pressure                       (scheme robustness)
//   B6d  Mass conservation |Δm|/m₀ < 1e-10         (periodic domain)

#include "../include/ns_solver.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool ok, double got = -1, double thr = -1)
{
    if (ok) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.3e  thr %.3e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

static void fill_leaves(NSSolver& s,
    const std::function<Prim(double,double,double)>& ic)
{
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = blk.ox + (i - NG + 0.5) * blk.h;
            double y = blk.oy + (j - NG + 0.5) * blk.h;
            double z = blk.oz + (k - NG + 0.5) * blk.h;
            Prim p   = ic(x, y, z);
            int  idx = cell_idx(i, j, k);
            blk.Q[0][idx] = p.rho;
            blk.Q[1][idx] = p.rho * p.u;
            blk.Q[2][idx] = p.rho * p.v;
            blk.Q[3][idx] = p.rho * p.w;
            blk.Q[4][idx] = p.p / (GAMMA - 1.0)
                           + 0.5 * p.rho * (p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
static void b6_kelvin_helmholtz()
{
    printf("\n-- B6  Kelvin-Helmholtz instability (Chandrashekar 2013, Pirozzoli 2011) --\n");

    const double DU    = 1.0;      // velocity jump
    const double eps   = 0.01;     // perturbation amplitude
    const double delta = 0.025;    // shear layer thickness (tanh smoothing)
    const double p0    = 2.5;      // background pressure
    const double L     = 1.0;

    // Smooth density/velocity interface via tanh profile
    auto kh_ic = [&](double x, double y, double /*z*/) -> Prim {
        // Two tanh interfaces at y=0.25 and y=0.75
        const double s1 = std::tanh((y - 0.25) / delta);
        const double s2 = std::tanh((y - 0.75) / delta);
        // u transitions from -ΔU/2 (bottom) → +ΔU/2 (middle) → -ΔU/2 (top)
        const double u  = 0.5*DU*(s1 - s2 - 1.0);
        // ρ transitions from 1 (bottom) → 2 (middle) → 1 (top)
        const double rho = 1.0 + 0.5*(s1 - s2);
        // Transverse perturbation: localized near each interface
        const double v  = eps * std::sin(2.0*M_PI*x)
                        * (std::exp(-((y-0.25)*(y-0.25))/(2.0*delta*delta))
                         + std::exp(-((y-0.75)*(y-0.75))/(2.0*delta*delta)));
        Prim q;
        q.rho = rho; q.u = u; q.v = v; q.w = 0.0; q.p = p0;
        q.T   = q.p / (q.rho * R_GAS);
        q.c   = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    };

    NSSolver s;
    s.cfg.bc_variant = PeriodicBC{};
    s.cfg.cfl             = 0.4;
    s.cfg.regrid_interval = 0;
    s.cfg.max_level       = 0;
    s.cfg.verbose         = false;
    s.init(L, kh_ic);

    // Level-2 uniform refinement → 32³ cells in [0,1]³, h≈0.03125
    for (int lvl = 0; lvl < 2; ++lvl) {
        auto lv = s.tree.leaf_indices();
        for (int li : lv) s.tree.refine(li);
        s.tree.rebuild_neighbours();
    }
    fill_leaves(s, kh_ic);
    s.alloc_scratch();

    // ── Initial mass and transverse KE ───────────────────────────────────────
    double mass0  = 0.0;
    double Ev_ini = 0.0;
    for (int li : s.tree.leaf_indices()) {
        const auto& blk = *s.tree.nodes[li].block;
        const double h3 = blk.h * blk.h * blk.h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int idx = cell_idx(i,j,k);
            const double rho  = blk.Q[0][idx];
            const double rhov = blk.Q[2][idx];
            mass0  += rho * h3;
            Ev_ini += 0.5 * rhov * rhov / rho * h3;
        }
    }

    // ── Run to t_end=4, recording diagnostics every 20 steps ─────────────────
    double t_peak = 0.0;
    double Ev_peak = 0.0;
    double p_min_global = 1e30;
    std::vector<std::pair<double,double>> diag_hist;  // (t, Ev)

    while (s.t < 4.0 && s.step < 200000) {
        double dt = s.advance();
        (void)dt;

        if (s.step % 20 == 0 || s.t >= 4.0) {
            double Ev = 0.0;
            double pm = 1e30;
            for (int li : s.tree.leaf_indices()) {
                const auto& blk = *s.tree.nodes[li].block;
                const double h3  = blk.h * blk.h * blk.h;
                for (int k = NG; k < NG+NB; ++k)
                for (int j = NG; j < NG+NB; ++j)
                for (int i = NG; i < NG+NB; ++i) {
                    const int idx = cell_idx(i,j,k);
                    const double rho  = blk.Q[0][idx];
                    const double rhov = blk.Q[2][idx];
                    const double rhou = blk.Q[1][idx];
                    const double rhow = blk.Q[3][idx];
                    const double E    = blk.Q[4][idx];
                    const double KE   = 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho;
                    const double p    = (GAMMA - 1.0) * (E - KE);
                    Ev += 0.5 * rhov * rhov / rho * h3;
                    pm  = std::min(pm, p);
                }
            }
            diag_hist.push_back({s.t, Ev});
            if (Ev > Ev_peak) { Ev_peak = Ev; t_peak = s.t; }
            p_min_global = std::min(p_min_global, pm);
        }
    }

    // ── Final mass ────────────────────────────────────────────────────────────
    double mass1 = 0.0;
    for (int li : s.tree.leaf_indices()) {
        const auto& blk = *s.tree.nodes[li].block;
        const double h3 = blk.h * blk.h * blk.h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            mass1 += blk.Q[0][cell_idx(i,j,k)] * h3;
    }
    double mass_err = std::fabs(mass1 - mass0) / std::fabs(mass0);

    const double Ev_ratio = Ev_peak / std::max(Ev_ini, 1e-300);

    double h_cell = s.tree.nodes[s.tree.leaf_indices()[0]].block->h;
    printf("   t_final = %.4f  steps = %d  leaves = %d  h = %.6f\n",
           s.t, s.step, (int)s.tree.leaf_indices().size(), h_cell);
    printf("   E_v(ini) = %.4e  E_v(peak) = %.4e  ratio = %.2f\n",
           Ev_ini, Ev_peak, Ev_ratio);
    printf("   t_peak = %.4f  p_min = %.4e  mass_err = %.2e\n",
           t_peak, p_min_global, mass_err);

    check("B6a  transverse KE grew ≥ 20× (exponential linear instability growth)",
          Ev_ratio >= 20.0, Ev_ratio, 20.0);
    check("B6b  roll-up time t_peak ∈ [0.3, 5.0]",
          t_peak >= 0.3 && t_peak <= 5.0, t_peak, 1.5);
    check("B6c  no negative pressure (scheme robustness)",
          p_min_global > 0.0, p_min_global, 0.0);
    check("B6d  mass conservation < 1e-10",
          mass_err < 1e-10, mass_err, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.6 gate: Kelvin-Helmholtz instability (2D, density-stratified) ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d\n", NB, NG, NB2, NCELL);
    b6_kelvin_helmholtz();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.6 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
