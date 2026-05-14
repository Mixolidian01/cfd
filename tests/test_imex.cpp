// test_imex.cpp — R9-F1: IMEX integration test
//
// Gate criteria:
//   T01  IMEX energy conserved < 1e-8 relative over 10 steps (acoustic pulse, u=0)
//   T02  IMEX mass   conserved < 1e-10 relative over 10 steps
//   T03  No NaN/Inf in any conserved field after 10 steps
//   T04  Explicit RK3 (use_imex=false) same IC → energy conserved < 1e-9
//
// Physical justification:
//   T01/T02: Acoustic pulse with zero background velocity.  The IMEX Helmholtz
//   correction solves (I − α∇²)u = u^* for each velocity component.  With u=0
//   exactly, ke_old = ke_new = 0 → no energy change from the implicit step.
//   Total energy is then conserved to the same floating-point round-off as the
//   explicit RK3 flux telescoping.  Mass is conserved exactly because Q[0] is
//   never touched by the Helmholtz step (code guarantee in advance_imex()).
//   T03: Verifies the positivity floor and Helmholtz solve do not produce
//   non-finite values on a smooth IC without shocks.
//   T04: Cross-checks that the explicit path also conserves energy for the same
//   IC, confirming the IC itself is consistent.

#include "ns_solver.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

static int n_pass = 0, n_fail = 0;

static void check(const char* name, bool cond, double got = -1.0, double thr = -1.0)
{
    if (cond) {
        printf("  PASS  %s\n", name);
        ++n_pass;
    } else {
        if (got >= 0.0)
            printf("  FAIL  %s  (got %.6e  threshold %.6e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Sum Q[0] (mass) and Q[4] (total energy) over all interior cells.
static void sum_mass_energy(const NSSolver& s, double& mass, double& energy)
{
    mass = 0.0; energy = 0.0;
    for (int li : s.tree.leaf_indices()) {
        if (!s.tree.nodes[li].has_block()) continue;
        const CellBlock& blk = *s.tree.nodes[li].block;
        const double dv = blk.h * blk.h * blk.h;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i, j, k);
            mass   += blk.Q[0][f] * dv;
            energy += blk.Q[4][f] * dv;
        }
    }
}

// Return true if every conserved variable in every interior cell is finite.
static bool all_finite(const NSSolver& s)
{
    for (int li : s.tree.leaf_indices()) {
        if (!s.tree.nodes[li].has_block()) continue;
        const CellBlock& blk = *s.tree.nodes[li].block;
        for (int v = 0; v < NVAR; ++v)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const double val = blk.Q[v][cell_idx(i, j, k)];
            if (!std::isfinite(val)) return false;
        }
    }
    return true;
}

// Build an NSSolver initialised with an acoustic-pulse IC (u=v=w=0).
// rho = rho_bg + A * sin(2π x / L),  p isentropic,  velocity = 0.
static NSSolver make_acoustic_solver(bool use_imex)
{
    NSSolver s;
    s.cfg.time.cfl           = 0.3;
    s.cfg.time.max_steps     = 10;
    s.cfg.time.t_end         = 1e30;
    s.cfg.bc.variant    = PeriodicBC{};
    s.cfg.io.verbose       = false;
    s.cfg.io.diag_interval = 10;
    s.cfg.physics.use_imex      = use_imex;
    s.cfg.physics.mg_levels     = 3;

    const double pi     = std::acos(-1.0);
    const double rho_bg = 1.225;
    const double p_bg   = 101325.0;
    const double A      = 0.01 * rho_bg;   // small-amplitude: A/rho_bg = 1%

    s.init(1.0, [&](double x, double /*y*/, double /*z*/) {
        Prim q;
        q.rho = rho_bg + A * std::sin(2.0 * pi * x);
        q.u   = 0.0; q.v = 0.0; q.w = 0.0;
        // Isentropic: p/p_bg = (rho/rho_bg)^gamma
        q.p   = p_bg * std::pow(q.rho / rho_bg, GAMMA);
        q.T   = q.p / (q.rho * R_GAS);
        q.c   = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    });

    return s;
}

// =============================================================================
// T01 + T02 + T03  IMEX: energy, mass, finite-values over 10 steps
// =============================================================================
static void t01_imex_energy_conservation()
{
    NSSolver s = make_acoustic_solver(/*use_imex=*/true);

    double m0, e0;
    sum_mass_energy(s, m0, e0);

    for (int step = 0; step < 10; ++step) s.advance();

    double m1, e1;
    sum_mass_energy(s, m1, e1);

    const double energy_err = std::abs(e1 - e0) / (std::abs(e0) + 1e-300);
    const double mass_err   = std::abs(m1 - m0) / (std::abs(m0) + 1e-300);

    check("T01 IMEX energy conserved < 1e-8 over 10 steps", energy_err < 1e-8,
          energy_err, 1e-8);
    check("T02 IMEX mass   conserved < 1e-10 over 10 steps", mass_err < 1e-10,
          mass_err, 1e-10);
    check("T03 no NaN/Inf in any field after 10 IMEX steps", all_finite(s));
}

// =============================================================================
// T04  Explicit RK3 (use_imex=false), same IC → energy conserved < 1e-9
// =============================================================================
static void t04_explicit_energy_conservation()
{
    NSSolver s = make_acoustic_solver(/*use_imex=*/false);

    double m0, e0;
    sum_mass_energy(s, m0, e0);

    for (int step = 0; step < 10; ++step) s.advance();

    double m1, e1;
    sum_mass_energy(s, m1, e1);

    const double energy_err = std::abs(e1 - e0) / (std::abs(e0) + 1e-300);
    check("T04 explicit RK3 energy conserved < 1e-9 over 10 steps",
          energy_err < 1e-9, energy_err, 1e-9);
}

// =============================================================================
// main
// =============================================================================
int main()
{
    printf("=== R9-F1: IMEX integration test ===\n");
    printf("    Gate: energy < 1e-8, mass < 1e-10, no NaN/Inf (IMEX);\n");
    printf("          explicit RK3 energy < 1e-9 (same IC)\n\n");

    t01_imex_energy_conservation();
    t04_explicit_energy_conservation();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0)
        printf("==> FAIL  test_imex (R9-F1 gate NOT cleared)\n");
    else
        printf("==> PASS  R9-F1 gate cleared\n");
    return (n_fail == 0) ? 0 : 1;
}
