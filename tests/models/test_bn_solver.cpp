// test_bn_solver.cpp — BNSolver gate tests
//
// BN05: Single-block SSP-RK3 — total mass conserved < 1e-12 over 10 steps.
//       (uses PeriodicBC; single block wraps to itself)
// BN06: Multi-block ghost fill — 2x2x2 periodic domain, ghost values match
//       neighbour interior after bn_fill_ghosts_tree().
// BN07: Multi-block mass conservation — 2x2x2 domain, 10 steps < 1e-12 drift.

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

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static BNEosParams make_water_air_eos() {
    BNEosParams p;
    p.gamma1 = 1.4;  p.pinf1 = 0.0;    // phase 1: air (ideal gas)
    p.gamma2 = 4.4;  p.pinf2 = 6e8;    // phase 2: water (stiffened gas)
    return p;
}

// Uniform IC: pure phase 1 (water), low velocity, high pressure.
static void uniform_ic(BNCellBlock& blk,
                        double /*ox*/, double /*oy*/, double /*oz*/, double /*h*/) {
    const double rho1 = 1000.0, rho2 = 1.0;
    const double alpha1 = 0.99, alpha2 = 1.0 - alpha1;
    const double u = 0.1, v = 0.0, w = 0.0;
    const double p1 = 1e5, p2 = 1e5;
    const BNEosParams eos = make_water_air_eos();
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

// Compute total phase-1 mass over all leaves in a BNSolver.
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
// BN05: single-block periodic, 10 steps, mass conserved.
// ─────────────────────────────────────────────────────────────────────────────
static void bn05_single_block_mass() {
    printf("\n-- BN05  Single-block SSP-RK3 mass conservation --\n");

    BNSolver s;
    s.init(1.0, uniform_ic, make_water_air_eos(), PeriodicBC{});

    double m0 = total_mass1(s);
    for (int i = 0; i < 10; ++i) s.advance();
    double m1 = total_mass1(s);

    double err = std::abs(m1 - m0) / std::abs(m0);
    check("BN05 phase-1 mass conserved < 1e-12", err < 1e-12, err, 1e-12);
    printf("     t = %.4e   dt = %.4e   steps = %d\n", s.t, s.t / s.step, s.step);
}

// ─────────────────────────────────────────────────────────────────────────────
// BN06: 2x2x2 multi-block ghost fill correctness.
// ─────────────────────────────────────────────────────────────────────────────
static void bn06_multi_block_ghost_fill() {
    printf("\n-- BN06  Multi-block ghost fill (2x2x2 periodic) --\n");

    BNSolver s;
    s.bc = PeriodicBC{};
    s.tree.set_periodic(true);
    s.tree.init(1.0);
    s.tree.refine(0);          // 1 block → 8 blocks (2x2x2)
    s.tree.rebuild_neighbours();
    s.eos = make_water_air_eos();
    s.t = 0.0; s.step = 0;
    s.alloc_scratch();

    // Fill each block with a unique constant (block index) in all variables.
    const auto& leaves = s.tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        for (int v = 0; v < NVAR_BN; ++v)
            s.Q[ii].Q[v].assign(NCELL, (double)(ii + 1));
    }

    bn_fill_ghosts_tree(s.tree, s.Q, s.bc);

    // After ghost fill, each ghost layer should contain the value of the
    // x-plus neighbor.  Check block 0 (oct 0 = ---): its XPLUS neighbor is
    // oct 1 (x-high), whose interior value is 2.0.
    int li0 = leaves[0];  // oct 0
    int li1 = -1;
    for (int ii = 1; ii < (int)leaves.size(); ++ii) {
        if (s.tree.nodes[leaves[ii]].level == 1) {
            if (s.tree.nodes[leaves[0]].neighbours[XPLUS] == leaves[ii]) {
                li1 = ii; break;
            }
        }
    }
    (void)li0;

    bool xplus_ok = true;
    const int blk0_idx = 0;
    double nb_val = -1.0;
    int xplus_nb = s.tree.nodes[leaves[blk0_idx]].neighbours[XPLUS];
    if (xplus_nb >= 0) {
        // Find slot of the XPLUS neighbor.
        for (int ii = 0; ii < (int)leaves.size(); ++ii) {
            if (leaves[ii] == xplus_nb) { nb_val = (double)(ii + 1); break; }
        }
    }

    if (nb_val > 0) {
        // Check ghost cells on XPLUS face of block 0 (i = NB+NG, NB+NG+1).
        const BNCellBlock& blk = s.Q[blk0_idx];
        int jc = NG + NB/2, kc = NG + NB/2;
        for (int g = 0; g < NG && xplus_ok; ++g) {
            int f = cell_idx(NB + NG + g, jc, kc);
            if (std::abs(blk.Q[0][f] - nb_val) > 1e-14) xplus_ok = false;
        }
    } else {
        xplus_ok = false;  // couldn't find neighbor
    }
    check("BN06 XPLUS ghost == neighbor interior", xplus_ok);
    (void)li1;
}

// ─────────────────────────────────────────────────────────────────────────────
// BN07: 2x2x2 multi-block mass conservation over 10 steps.
// ─────────────────────────────────────────────────────────────────────────────
static void bn07_multi_block_mass() {
    printf("\n-- BN07  Multi-block mass conservation (2x2x2, 10 steps) --\n");

    BNSolver s;
    s.bc = PeriodicBC{};
    s.tree.set_periodic(true);
    s.tree.init(1.0);
    s.tree.refine(0);
    s.tree.rebuild_neighbours();
    s.eos = make_water_air_eos();
    s.t = 0.0; s.step = 0;
    s.alloc_scratch();

    const auto& leaves = s.tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii)
        uniform_ic(s.Q[ii], s.Q[ii].ox, s.Q[ii].oy, s.Q[ii].oz, s.Q[ii].h);

    double m0 = total_mass1(s);
    for (int i = 0; i < 10; ++i) s.advance();
    double m1 = total_mass1(s);

    double err = std::abs(m1 - m0) / std::abs(m0);
    check("BN07 multi-block phase-1 mass conserved < 1e-12", err < 1e-12, err, 1e-12);
    printf("     t = %.4e   steps = %d\n", s.t, s.step);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== BNSolver gate tests (BN05–BN07) ===\n");

    bn05_single_block_mass();
    bn06_multi_block_ghost_fill();
    bn07_multi_block_mass();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0)
        printf("==> FAIL  BNSolver gate NOT cleared\n");
    else
        printf("==> PASS  BNSolver gate cleared\n");
    return (n_fail == 0) ? 0 : 1;
}
