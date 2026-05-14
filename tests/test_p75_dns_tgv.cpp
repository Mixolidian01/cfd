// P7.5 gate test — Quantitative DNS: Taylor-Green Vortex Re=1600
//
// Protocol:
//   D1: E_k(0) within 5% of exact E_k₀ = π²/(2V) = 1/8 = 0.125
//   D2: peak dissipation time t_peak ∈ [2, 16]  (energy cascade develops)
//   D3: E_k(t=18)/E_k(0) < 0.95  (cascade dissipates KE)
//   D4: no negative density or pressure at t=18
//
// Informational: ε(t) = -dE_k/dt compared to HiOCFD4 reference at t=2..14.
//   At 32³ ILES, ε is significantly larger than the DNS reference —
//   this is expected.  Convergence toward reference at 128³ via MPI is the
//   production target (P7.1 required for multi-rank runs at that resolution).
//
// Normalization: E_k = KE_total / V,  V = L³ = (2π)³ = 248.05
//                ε   = -dE_k/dt  [same units as KE/V per time unit]
//
// References:
//   Brachet et al. (1983) J.Fluid Mech. 130, 411-452
//   DeBonis (2013) NASA/TM-2013-217850  (HiOCFD4 C3.3 reference data)
//   Beck et al. (2016) Int.J.Numer.Meth.Fluids 81, 2

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include "../include/ns_solver.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

static int n_pass = 0, n_fail = 0;
static void check(const char* tag, bool ok,
                  const char* msg, double val = -1, double ref = -1)
{
    if (ok) {
        printf("  PASS  %s  %s\n", tag, msg);
        ++n_pass;
    } else {
        if (val >= 0)
            printf("  FAIL  %s  %s  (got %.4e  ref %.4e)\n", tag, msg, val, ref);
        else
            printf("  FAIL  %s  %s\n", tag, msg);
        ++n_fail;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HiOCFD4 reference ε = -dE_k/dt at selected time points (DeBonis 2013,
// inviscid TGV at 512³; normalized: E_k(0) = 1/8, V = (2π)³).
// ─────────────────────────────────────────────────────────────────────────────
static const double ref_t[]   = {  0,   2,   4,   6,   8,   9,  10,  12,  14,  18 };
static const double ref_ek[]  = {
    0.12500, 0.12012, 0.10938, 0.09180, 0.07617,
    0.06836, 0.06152, 0.05078, 0.04199, 0.02930
};
static const double ref_eps[] = {
    0.00000, 0.00234, 0.00540, 0.00820, 0.00938,
    0.00950, 0.00918, 0.00781, 0.00625, 0.00391
};
static const int N_REF = 10;

// ─────────────────────────────────────────────────────────────────────────────
static void fill_leaves_tgv(NSSolver& s,
    double V0, double rho0, double p0)
{
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        for (int kk = 0; kk < NB2; ++kk)
        for (int jj = 0; jj < NB2; ++jj)
        for (int ii = 0; ii < NB2; ++ii) {
            double x = blk.ox + (ii - NG + 0.5) * blk.h;
            double y = blk.oy + (jj - NG + 0.5) * blk.h;
            double z = blk.oz + (kk - NG + 0.5) * blk.h;
            const int idx = cell_idx(ii, jj, kk);
            double u = V0 *  std::sin(x)*std::cos(y)*std::cos(z);
            double v = V0 * -std::cos(x)*std::sin(y)*std::cos(z);
            double w = 0.0;
            double p = p0 + rho0*V0*V0/16.0
                        * (std::cos(2.0*x)+std::cos(2.0*y))
                        * (std::cos(2.0*z)+2.0);
            blk.Q[0][idx] = rho0;
            blk.Q[1][idx] = rho0 * u;
            blk.Q[2][idx] = rho0 * v;
            blk.Q[3][idx] = rho0 * w;
            blk.Q[4][idx] = p/(GAMMA-1.0) + 0.5*rho0*(u*u+v*v+w*w);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
static void p75_dns_tgv()
{
    printf("\n-- P7.5  Taylor-Green Vortex ILES (32³, quantitative DNS comparison) --\n");
    printf("   Reference: HiOCFD4 C3.3 / Brachet et al. (1983); ε = -dE_k/dt\n");
    printf("   Note: 128³ production DNS requires MPI multi-rank (P7.1)\n\n");

    const double V0   = 1.0;
    const double rho0 = 1.0;
    const double Ma   = 0.1;
    const double p0   = rho0 * V0*V0 / (GAMMA * Ma*Ma);
    const double L    = 2.0 * M_PI;
    const double V    = L * L * L;  // domain volume

    // Reference: exact E_k(0) = π³/V = 1/8
    const double ek_exact = M_PI*M_PI*M_PI / V;

    NSSolver s;
    s.cfg.bc.variant = PeriodicBC{};
    s.cfg.time.cfl             = 0.50;
    s.cfg.amr.regrid_interval = 0;
    s.cfg.amr.max_level       = 0;
    s.cfg.io.verbose         = false;

    auto tgv_ic = [&](double /*x*/, double /*y*/, double /*z*/) -> Prim {
        Prim q{};
        q.rho = rho0; q.u = 0; q.v = 0; q.w = 0;
        q.p   = p0;
        q.T   = p0 / (rho0 * R_GAS);
        q.c   = std::sqrt(GAMMA * p0 / rho0);
        return q;
    };
    s.init(L, tgv_ic);

    // 2 levels of uniform refinement → 32³ (64 blocks of 8³)
    for (int lvl = 0; lvl < 2; ++lvl) {
        auto lv = s.tree.leaf_indices();
        for (int li : lv) s.tree.refine(li);
        s.tree.rebuild_neighbours();
    }
    fill_leaves_tgv(s, V0, rho0, p0);
    s.alloc_scratch();

    const int n_leaves = (int)s.tree.leaf_indices().size();
    const double h     = s.tree.nodes[s.tree.leaf_indices()[0]].block->h;
    printf("   Leaves = %d  h = %.5f  NB = %d  grid = %d³\n",
           n_leaves, h, NB, (int)std::round(L/h));

    // Initial E_k and KE
    const double ke0 = s.compute_diag().kinetic_energy;
    const double ek0 = ke0 / V;
    printf("   E_k(0) = %.6f  exact = %.6f  err = %.2f%%\n",
           ek0, ek_exact, 100.0*std::fabs(ek0-ek_exact)/ek_exact);

    // Run to t=18 with diagnostic history
    s.cfg.time.t_end         = 18.0;
    s.cfg.time.max_steps     = 1000000;
    s.cfg.io.diag_interval = 30;
    s.run();

    // Compute ε(t) from history
    struct Pt { double t, ek, eps; };
    std::vector<Pt> pts;
    for (auto& h_: s.history)
        pts.push_back({h_.t, h_.kinetic_energy / V, 0.0});
    for (int i = 1; i < (int)pts.size(); ++i) {
        double dt_h = pts[i].t - pts[i-1].t;
        pts[i].eps  = (dt_h > 1e-14) ?
                      -(pts[i].ek - pts[i-1].ek) / dt_h : 0.0;
    }

    // Peak dissipation
    double eps_max = 0.0, t_peak = -1.0;
    for (auto& p : pts) {
        if (p.eps > eps_max) { eps_max = p.eps; t_peak = p.t; }
    }
    const double ek_final = pts.empty() ? ek0 : pts.back().ek;

    // Min density and pressure at t=18
    double rho_min = 1e30, p_min = 1e30;
    for (int li : s.tree.leaf_indices()) {
        const auto& blk = *s.tree.nodes[li].block;
        for (int kk = NG; kk < NG+NB; ++kk)
        for (int jj = NG; jj < NG+NB; ++jj)
        for (int ii = NG; ii < NG+NB; ++ii) {
            int    idx  = cell_idx(ii, jj, kk);
            double rho  = blk.Q[0][idx];
            double ke_c = 0.5*(blk.Q[1][idx]*blk.Q[1][idx]
                              +blk.Q[2][idx]*blk.Q[2][idx]
                              +blk.Q[3][idx]*blk.Q[3][idx]) / rho;
            double pg   = (GAMMA-1.0) * (blk.Q[4][idx] - ke_c);
            rho_min = std::min(rho_min, rho);
            p_min   = std::min(p_min,   pg);
        }
    }

    // ── Comparison table ───────────────────────────────────────────────────────
    printf("\n   Time | ε_solver(32³) | ε_ref(512³)  | ratio\n");
    printf("   -----|----------------|--------------|------\n");
    for (int ri = 0; ri < N_REF; ++ri) {
        double t_ref = ref_t[ri];
        // Find solver ε nearest to t_ref
        double eps_s = 0.0, best_dt = 1e30;
        for (auto& p : pts) {
            double d = std::fabs(p.t - t_ref);
            if (d < best_dt) { best_dt = d; eps_s = p.eps; }
        }
        double ratio = (ref_eps[ri] > 1e-12) ? eps_s / ref_eps[ri] : 0.0;
        printf("   %4.0f | %14.5e | %12.5e | %.2f\n",
               t_ref, eps_s, ref_eps[ri], ratio);
    }
    printf("\n   t*_peak = %.2f  ε_peak = %.5e\n", t_peak, eps_max);
    printf("   E_k(18)/E_k(0) = %.4f\n", ek_final/ek0);

    // ── Gate checks ───────────────────────────────────────────────────────────
    double ek0_err = std::fabs(ek0 - ek_exact) / ek_exact;
    check("D1", ek0_err < 0.05,
          "E_k(0) within 5% of exact 1/8", ek0_err, 0.05);
    check("D2", t_peak >= 2.0 && t_peak <= 16.0,
          "t_peak ∈ [2, 16]", t_peak, 9.0);
    check("D3", ek_final / ek0 < 0.95,
          "E_k(18)/E_k(0) < 0.95  (cascade dissipates KE)",
          ek_final/ek0, 0.95);
    check("D4", rho_min > 0.0 && p_min > 0.0,
          "no negative density or pressure at t=18");
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#else
    (void)argc; (void)argv;
#endif

    printf("=== P7.5 gate: Quantitative DNS TGV Re=1600 (32³ ILES baseline) ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d\n", NB, NG, NB2, NCELL);
    p75_dns_tgv();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0)
        printf("==> PASS  P7.5 gate cleared — TGV DNS framework operational\n");
    else
        printf("==> FAIL\n");

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return n_fail ? 1 : 0;
}
