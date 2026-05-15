// test_bn_amr.cpp — BNSolver AMR gate tests
//
// BN09: refine(0) — prolong is mass-conservative to machine precision.
// BN10: coarsen(0) — restrict is mass-conservative to machine precision.
// BN11: 2-level AMR SSP-RK3 — mass drift < 1e-12 over 10 steps.
//       (7 level-1 + 8 level-2 leaves, flux-register correction active)

#include "models/bn_solver.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool ok, double got = -1, double thr = -1) {
    if (ok) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0) printf("  FAIL  %s  (got %.3e  thr %.3e)\n", name, got, thr);
        else          printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

static BNEosParams make_eos() {
    BNEosParams p;
    p.gamma1 = 1.4;  p.pinf1 = 0.0;
    p.gamma2 = 4.4;  p.pinf2 = 6e8;
    return p;
}

static void uniform_ic(BNCellBlock& blk,
                        double /*ox*/, double /*oy*/, double /*oz*/, double /*h*/) {
    const double rho1 = 1000.0, rho2 = 1.0;
    const double alpha1 = 0.99, alpha2 = 1.0 - alpha1;
    const double u = 0.1, v = 0.0, w = 0.0;
    const double p1 = 1e5, p2 = 1e5;
    const BNEosParams eos = make_eos();
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        int f = cell_idx(i, j, k);
        blk.Q[0][f] = alpha1 * rho1;
        blk.Q[1][f] = alpha2 * rho2;
        blk.Q[2][f] = (alpha1*rho1 + alpha2*rho2) * u;
        blk.Q[3][f] = (alpha1*rho1 + alpha2*rho2) * v;
        blk.Q[4][f] = (alpha1*rho1 + alpha2*rho2) * w;
        double e1 = p1 / ((eos.gamma1 - 1.0) * rho1);
        double e2 = (p2 + eos.gamma2 * eos.pinf2) / ((eos.gamma2 - 1.0) * rho2);
        double ke = 0.5 * (u*u + v*v + w*w);
        blk.Q[5][f] = alpha1*rho1*(e1 + ke) + alpha2*rho2*(e2 + ke);
        blk.Q[6][f] = alpha1;
    }
}

static double total_mass1(const BNSolver& s) {
    double m = 0.0;
    const auto& leaves = s.tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        double h3 = s.Q[ii].h * s.Q[ii].h * s.Q[ii].h;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i)
            m += s.Q[ii].Q[0][cell_idx(i,j,k)] * h3;
    }
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// BN09: refine(0) — prolong conserves total phase-1 mass
// ─────────────────────────────────────────────────────────────────────────────
static void bn09_refine_mass() {
    printf("\n-- BN09  refine(0) mass conservation --\n");

    BNSolver s;
    s.init(1.0, uniform_ic, make_eos(), PeriodicBC{});
    // Single leaf at slot 0

    double m0 = total_mass1(s);
    int nl0 = (int)s.tree.leaf_indices().size();

    s.refine(0);  // 1 leaf → 8 leaves

    int nl1 = (int)s.tree.leaf_indices().size();
    double m1 = total_mass1(s);

    check("BN09 NL before == 1", nl0 == 1);
    check("BN09 NL after == 8", nl1 == 8);
    double err = std::abs(m1 - m0) / std::abs(m0);
    check("BN09 prolong mass conserved < 1e-14", err < 1e-14, err, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// BN10: coarsen(0) — restrict conserves total phase-1 mass
// ─────────────────────────────────────────────────────────────────────────────
static void bn10_coarsen_mass() {
    printf("\n-- BN10  coarsen(0) mass conservation --\n");

    BNSolver s;
    s.init(1.0, uniform_ic, make_eos(), PeriodicBC{});
    s.refine(0);  // 8 level-1 leaves

    double m0 = total_mass1(s);
    int nl0 = (int)s.tree.leaf_indices().size();

    // Root node (index 0) has 8 leaf children — coarsen them back
    s.coarsen(0);

    int nl1 = (int)s.tree.leaf_indices().size();
    double m1 = total_mass1(s);

    check("BN10 NL before == 8", nl0 == 8);
    check("BN10 NL after == 1", nl1 == 1);
    double err = std::abs(m1 - m0) / std::abs(m0);
    check("BN10 restrict mass conserved < 1e-14", err < 1e-14, err, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// BN11: 2-level AMR (7 level-1 + 8 level-2), 10 RK3 steps, mass drift < 1e-12
// ─────────────────────────────────────────────────────────────────────────────
static void bn11_amr_advance_mass() {
    printf("\n-- BN11  2-level AMR advance mass conservation (10 steps) --\n");

    // Build: root → 8 level-1 (via tree refine), then refine leaf-slot-0 → 8 level-2.
    BNSolver s;
    s.bc = PeriodicBC{};
    s.tree.set_periodic(true);
    s.tree.init(1.0);
    s.tree.refine(0);           // 8 level-1 leaves
    s.tree.rebuild_neighbours();
    s.eos = make_eos();
    s.t = 0.0; s.step = 0;
    s.alloc_scratch();

    const auto& leaves = s.tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii)
        uniform_ic(s.Q[ii], s.Q[ii].ox, s.Q[ii].oy, s.Q[ii].oz, s.Q[ii].h);

    // Refine slot 0 → 8 level-2 grandchildren (15 leaves total)
    s.refine(0);
    check("BN11 NL == 15", (int)s.tree.leaf_indices().size() == 15);

    // Diagnostic: peek at interior cell (ilo,ilo,ilo) in each leaf
    {
        const auto& lv = s.tree.leaf_indices();
        int f0 = cell_idx(ilo(), ilo(), ilo());
        BNEosParams eos = make_eos();
        for (int ii = 0; ii < (int)lv.size(); ++ii) {
            const auto& nd = s.tree.nodes[lv[ii]];
            Prim2Phase p0 = bn_cons_to_prim(
                s.Q[ii].Q[0][f0], s.Q[ii].Q[1][f0],
                s.Q[ii].Q[2][f0], s.Q[ii].Q[3][f0], s.Q[ii].Q[4][f0],
                s.Q[ii].Q[5][f0], s.Q[ii].Q[6][f0], eos);
            double sp = std::abs(p0.u) + std::abs(p0.v) + std::abs(p0.w) + p0.c_mix;
            printf("     diag slot %2d  level=%d  h=%.5f  a1r1=%.3e  E=%.3e  alpha1=%.3f  u=%.3f  c_mix=%.3f  sp=%.3e\n",
                   ii, nd.level, s.Q[ii].h,
                   s.Q[ii].Q[0][f0], s.Q[ii].Q[5][f0], s.Q[ii].Q[6][f0],
                   p0.u, p0.c_mix, sp);
        }
    }

    double m0 = total_mass1(s);
    for (int i = 0; i < 10; ++i) {
        double dt_used = s.advance();
        if (i < 3) printf("     step %d  dt=%.4e  t=%.4e\n", i, dt_used, s.t);
    }
    double m1 = total_mass1(s);

    double err = std::abs(m1 - m0) / std::abs(m0);
    check("BN11 2-level mass conserved < 1e-12", err < 1e-12, err, 1e-12);
    printf("     t = %.4e   steps = %d   leaves = %d\n",
           s.t, s.step, (int)s.tree.leaf_indices().size());
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== BNSolver AMR gate tests (BN09–BN11) ===\n");

    bn09_refine_mass();
    bn10_coarsen_mass();
    bn11_amr_advance_mass();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0)
        printf("==> FAIL  BNSolver AMR gate NOT cleared\n");
    else
        printf("==> PASS  BNSolver AMR gate cleared\n");
    return (n_fail == 0) ? 0 : 1;
}
