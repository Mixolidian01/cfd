// B.3 benchmark — Taylor-Green Vortex (3D, periodic, ILES)
//
// Canonical benchmark for 3D turbulence development and kinetic energy
// dissipation via the energy cascade.  The smooth sinusoidal IC develops
// vortex sheets and a turbulent energy cascade; the kinetic energy (KE)
// dissipation rate peaks at t* ≈ 9 for Re=1600 (Brachet et al. 1983).
//
// Reference: Brachet et al. (1983) J. Fluid Mech. 130, 411–452;
//            HiOCFD4 (2016) test case C3.3.
//
// Setup (standard HiOCFD4 IC, non-dimensional with V₀=ρ₀=1, L=1):
//   Domain: [0, 2π]³, BCType::Periodic
//   u =  sin x · cos y · cos z
//   v = −cos x · sin y · cos z
//   w = 0
//   p = p₀ + (1/16)(cos 2x + cos 2y)(cos 2z + 2),  p₀ = 1/(γ Ma²) ≈ 71.43
//   Ma = 0.1  →  T₀ = p₀/(ρ₀Rgas) ≈ 0.249 K  →  c₀ = V₀/Ma = 10 m/s
//   Sutherland at T₀ ≈ 0.25K gives μ ≈ 0 → effectively inviscid (ILES).
//   Level-3 uniform refinement → 64³ cells, h = 2π/64 ≈ 0.098.
//   Run to t* = 18 (t measured in L/V₀ = 1 units).
//
// Exact initial KE = π³ ≈ 31.006  (analytic integral of ρ|u|²/2 over [0,2π]³).
//
// Gate criteria (ILES at 32³; quantitative DNS comparison requires 128³):
//   B3a  KE(0) within 5% of π³ (cell-quadrature accuracy)
//   B3b  KE dissipation rate peak at t* ∈ [3, 16]
//   B3c  KE(t*=18)/KE(0) < 0.98  (energy cascade dissipates KE)
//   B3d  No negative density at t* = 18
//   B3e  No negative pressure at t* = 18

#include "../include/ns_solver.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

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
static void b3_tgv()
{
    printf("\n-- B3  Taylor-Green Vortex (3D periodic ILES, HiOCFD4 C3.3) --\n");

    const double V0   = 1.0;
    const double rho0 = 1.0;
    const double Ma   = 0.1;
    const double p0   = rho0 * V0*V0 / (GAMMA * Ma*Ma);  // ≈ 71.429
    const double L_code = 2.0 * M_PI;

    auto tgv_ic = [&](double xc, double yc, double zc) -> Prim {
        Prim q;
        q.rho = rho0;
        q.u   =  V0 * std::sin(xc) * std::cos(yc) * std::cos(zc);
        q.v   = -V0 * std::cos(xc) * std::sin(yc) * std::cos(zc);
        q.w   =  0.0;
        q.p   = p0 + rho0*V0*V0/16.0
                   * (std::cos(2.0*xc) + std::cos(2.0*yc))
                   * (std::cos(2.0*zc) + 2.0);
        q.T   = q.p / (q.rho * R_GAS);
        q.c   = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    };

    NSSolver s;
    s.cfg.bc              = BCType::Periodic;
    s.cfg.cfl             = 0.4;
    s.cfg.regrid_interval = 0;
    s.cfg.max_level       = 0;
    s.cfg.verbose         = false;
    s.init(L_code, tgv_ic);

    // Level-3 refinement → 64³ cells (512 blocks), h ≈ 0.098
    for (int lvl = 0; lvl < 3; ++lvl) {
        auto lv = s.tree.leaf_indices();
        for (int li : lv) s.tree.refine(li);
        s.tree.rebuild_neighbours();
    }
    fill_leaves(s, tgv_ic);
    s.alloc_scratch();

    // Initial KE (before any time stepping)
    const double ke_exact = M_PI * M_PI * M_PI;  // π³ ≈ 31.006
    double ke0 = s.compute_diag().kinetic_energy;

    // Run to t* = 18 with frequent diagnostics to track KE(t*)
    s.cfg.t_end         = 18.0;
    s.cfg.max_steps     = 200000;
    s.cfg.diag_interval = 20;
    s.run();

    // ── Find peak KE dissipation rate from history ────────────────────────────
    double eps_max = 0.0;
    double t_peak  = -1.0;
    for (int ii = 1; ii < (int)s.history.size(); ++ii) {
        double dt_h = s.history[ii].t - s.history[ii-1].t;
        if (dt_h < 1e-14) continue;
        double eps = -(s.history[ii].kinetic_energy - s.history[ii-1].kinetic_energy) / dt_h;
        if (eps > eps_max) {
            eps_max = eps;
            t_peak  = 0.5 * (s.history[ii].t + s.history[ii-1].t);
        }
    }

    double ke_final = s.history.empty() ? ke0 : s.history.back().kinetic_energy;
    double ke_ratio = ke_final / ke0;

    // ── Min density and pressure over all leaf interior cells ─────────────────
    double rho_min = 1e30, p_min_global = 1e30;
    for (int li : s.tree.leaf_indices()) {
        const auto& blk = *s.tree.nodes[li].block;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            int    idx = cell_idx(i,j,k);
            double rho = blk.Q[0][idx];
            double KE_cell = 0.5*(blk.Q[1][idx]*blk.Q[1][idx]
                                 +blk.Q[2][idx]*blk.Q[2][idx]
                                 +blk.Q[3][idx]*blk.Q[3][idx]) / rho;
            double pg  = (GAMMA-1.0) * (blk.Q[4][idx] - KE_cell);
            rho_min       = std::min(rho_min,       rho);
            p_min_global  = std::min(p_min_global,  pg);
        }
    }

    double ke0_rel_err = std::fabs(ke0 - ke_exact) / ke_exact;

    printf("   t_final = %.2f  leaves = %d  h = %.5f\n",
           s.t, (int)s.tree.leaf_indices().size(),
           s.tree.nodes[s.tree.leaf_indices()[0]].block->h);
    printf("   ke0 = %.4f  ke_exact(π³) = %.4f  rel_err = %.2f%%\n",
           ke0, ke_exact, 100.0*ke0_rel_err);
    printf("   t*_peak = %.2f  eps_max = %.4e\n", t_peak, eps_max);
    printf("   ke_ratio(t*=18) = %.4f  rho_min = %.4f  p_min = %.4f\n",
           ke_ratio, rho_min, p_min_global);

    check("B3a  initial KE within 5% of π³",
          ke0_rel_err < 0.05, ke0_rel_err, 0.05);
    check("B3b  KE dissipation peak at t* ∈ [3, 16]",
          t_peak >= 3.0 && t_peak <= 16.0, t_peak, 9.0);
    check("B3c  KE(18)/KE(0) < 0.98  (energy cascade dissipates KE)",
          ke_ratio < 0.98, ke_ratio, 0.98);
    check("B3d  no negative density",
          rho_min > 0.0, rho_min, 0.0);
    check("B3e  no negative pressure",
          p_min_global > 0.0, p_min_global, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.3 gate: Taylor-Green Vortex (3D periodic ILES) ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d\n", NB, NG, NB2, NCELL);
    b3_tgv();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.3 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
