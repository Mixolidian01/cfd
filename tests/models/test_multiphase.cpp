// P4.4 gate — Allaire 5-equation two-phase model
//
// BN01 — Global conservation (partial densities + mixture momentum + energy)
//         on a 1D periodic slab after 10 Euler steps.  α₁ is NOT conserved
//         (non-conservative advection); total ρ = α₁ρ₁ + α₂ρ₂ IS conserved.
//
// BN02 — Freestream preservation: uniform state → zero RHS for all 7 variables.
//
// BN03 — Bound preservation: α₁ ∈ [0,1] after 20 Euler steps on a sharp
//         interface (α₁ transitions from 0.02 to 0.98 over one cell).
//
// BN04 — Single-phase limit: α₁ = 1, pinf₁ = 0 → BN flux for vars {0,2,3,4,5}
//         matches standard HLLC flux to 1e-13 on a random state pair.
//
// Pass criteria: all labelled PASS/FAIL; exit 0 iff all pass.

#include "models/bn_model.hpp"
#include "mesh/cell_block.hpp"
#include "schemes/operators.hpp"   // for hllc_flux (BN04 comparison)
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

// ── Utilities ─────────────────────────────────────────────────────────────────
static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool cond, double got=-1, double thr=-1) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.3e  threshold %.3e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// Scalar Euler step Q ← Q + dt * RHS (interior only).
static void euler_step(BNCellBlock& blk, const BNEosParams& eos, double dt) {
    BNCellBlock rhs;
    rhs.h = blk.h;
    bn_fill_ghosts_periodic(blk);
    compute_rhs_bn(blk, rhs, eos);
    for (int v = 0; v < NVAR_BN; ++v)
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int f = cell_idx(i,j,k);
        blk.Q[v][f] += dt * rhs.Q[v][f];
    }
}

// Sum of Q[v] over interior cells.
static double interior_sum(const BNCellBlock& blk, int v) {
    double s = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        s += blk.Q[v][cell_idx(i,j,k)];
    return s;
}

// =============================================================================
// BN01 — Conservation
// =============================================================================
static void test_BN01() {
    printf("\n-- BN01  Global conservation --\n");

    // Slab interface in x: left half phase 1 dominant, right half phase 2.
    // Uniform pressure and velocity for smooth initial data.
    BNEosParams eos;  // defaults: γ₁=1.4, p∞₁=0, γ₂=4.4, p∞₂=6e8
    const double h   = 1.0 / NB;
    const double p0  = 1.0e5;   // [Pa]
    const double u0  = 50.0;    // [m/s]
    const double rho1_ref = 1.2;
    const double rho2_ref = 900.0;

    BNCellBlock blk(0.0, 0.0, 0.0, h);
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int f = cell_idx(i,j,k);
        const double x = blk.xc(i);
        // Smooth transition in x: tanh profile width ~ 2*h
        const double a1 = 0.5 + 0.4 * std::tanh(4.0 * (x - 0.5));  // ∈ [0.1, 0.9]
        const double a2 = 1.0 - a1;
        // Partial densities
        blk.Q[0][f] = a1 * rho1_ref;
        blk.Q[1][f] = a2 * rho2_ref;
        // Momentum (uniform velocity u0 in x only)
        const double rho = blk.Q[0][f] + blk.Q[1][f];
        blk.Q[2][f] = rho * u0;
        blk.Q[3][f] = 0.0;
        blk.Q[4][f] = 0.0;
        // Energy from pressure equilibrium
        const double KE   = 0.5 * rho * u0 * u0;
        const double rho_e = a1*(p0 + eos.gamma1*eos.pinf1)/(eos.gamma1-1.0)
                           + a2*(p0 + eos.gamma2*eos.pinf2)/(eos.gamma2-1.0);
        blk.Q[5][f] = KE + rho_e;
        blk.Q[6][f] = a1;
    }

    // Record initial integrals
    const double m1_0 = interior_sum(blk, 0);
    const double m2_0 = interior_sum(blk, 1);
    const double px_0 = interior_sum(blk, 2);
    const double E_0  = interior_sum(blk, 5);

    // Advance 10 steps (CFL ≈ 0.3)
    const double cfl = 0.3;
    for (int step = 0; step < 10; ++step) {
        const double dt = bn_cfl_dt(blk, cfl, eos);
        euler_step(blk, eos, dt);
    }

    const double tol = 1.0e-9;
    const double dm1 = std::abs(interior_sum(blk,0) - m1_0) / (std::abs(m1_0) + 1.0);
    const double dm2 = std::abs(interior_sum(blk,1) - m2_0) / (std::abs(m2_0) + 1.0);
    const double dpx = std::abs(interior_sum(blk,2) - px_0) / (std::abs(px_0) + 1.0);
    const double dE  = std::abs(interior_sum(blk,5) - E_0 ) / (std::abs(E_0 ) + 1.0);

    check("BN01a α₁ρ₁ conservation", dm1 < tol, dm1, tol);
    check("BN01b α₂ρ₂ conservation", dm2 < tol, dm2, tol);
    check("BN01c ρu  conservation",  dpx < tol, dpx, tol);
    check("BN01d E   conservation",  dE  < tol, dE,  tol);
}

// =============================================================================
// BN02 — Freestream preservation
// =============================================================================
static void test_BN02() {
    printf("\n-- BN02  Freestream preservation --\n");

    BNEosParams eos;
    const double h   = 0.1;
    const double a1  = 0.7;
    const double a2  = 0.3;
    const double rho1= 1.2,  rho2 = 900.0;
    const double u0  = 50.0, p0   = 1.0e5;

    BNCellBlock blk(0.0, 0.0, 0.0, h);
    const double rho = a1*rho1 + a2*rho2;
    const double KE  = 0.5*rho*u0*u0;
    const double rho_e = a1*(p0 + eos.gamma1*eos.pinf1)/(eos.gamma1-1.0)
                       + a2*(p0 + eos.gamma2*eos.pinf2)/(eos.gamma2-1.0);
    const double E0  = KE + rho_e;

    for (int flat = 0; flat < NCELL; ++flat) {
        blk.Q[0][flat] = a1 * rho1;
        blk.Q[1][flat] = a2 * rho2;
        blk.Q[2][flat] = rho * u0;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = 0.0;
        blk.Q[5][flat] = E0;
        blk.Q[6][flat] = a1;
    }

    bn_fill_ghosts_periodic(blk);
    BNCellBlock rhs;
    rhs.h = h;
    compute_rhs_bn(blk, rhs, eos);

    double max_rhs = 0.0;
    for (int v = 0; v < NVAR_BN; ++v)
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        max_rhs = std::max(max_rhs, std::abs(rhs.Q[v][cell_idx(i,j,k)]));

    // Normalise by a pressure-like scale: p0/h
    const double scale = p0 / h;
    const double rel   = max_rhs / scale;
    const double tol   = 1.0e-11;
    check("BN02  uniform state → zero RHS", rel < tol, rel, tol);
}

// =============================================================================
// BN03 — Bound preservation  α₁ ∈ [0, 1]
// =============================================================================
static void test_BN03() {
    printf("\n-- BN03  Bound preservation (α₁ ∈ [0,1]) --\n");

    BNEosParams eos;
    const double h = 1.0 / NB;
    const double p0 = 1.0e5, u0 = 100.0;
    const double rho1 = 1.2, rho2 = 900.0;

    BNCellBlock blk(0.0, 0.0, 0.0, h);
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int f = cell_idx(i,j,k);
        const double x = blk.xc(i);
        // Sharp interface: α₁ near 0 on left, near 1 on right
        const double a1 = (x < 0.5) ? 0.02 : 0.98;
        const double a2 = 1.0 - a1;
        blk.Q[0][f] = a1 * rho1;
        blk.Q[1][f] = a2 * rho2;
        const double rho = blk.Q[0][f] + blk.Q[1][f];
        blk.Q[2][f] = rho * u0;
        blk.Q[3][f] = 0.0;
        blk.Q[4][f] = 0.0;
        const double KE   = 0.5*rho*u0*u0;
        const double rho_e = a1*(p0 + eos.gamma1*eos.pinf1)/(eos.gamma1-1.0)
                           + a2*(p0 + eos.gamma2*eos.pinf2)/(eos.gamma2-1.0);
        blk.Q[5][f] = KE + rho_e;
        blk.Q[6][f] = a1;
    }

    for (int step = 0; step < 20; ++step) {
        const double dt = bn_cfl_dt(blk, 0.3, eos);
        euler_step(blk, eos, dt);
    }

    double a1_min = 1.0, a1_max = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const double a1 = blk.Q[6][cell_idx(i,j,k)];
        a1_min = std::min(a1_min, a1);
        a1_max = std::max(a1_max, a1);
    }

    // Allow a small numerical overshoot (first-order upwind should have none,
    // but floating-point accumulation may shift the extrema by ~eps).
    const double tol = 1.0e-12;
    check("BN03a α₁ ≥ 0", a1_min >= -tol, (a1_min < 0 ? -a1_min : 0.0), tol);
    check("BN03b α₁ ≤ 1", a1_max <=  1.0 + tol, (a1_max > 1 ? a1_max-1.0 : 0.0), tol);
    printf("         (α₁ range: [%.6f, %.6f])\n", a1_min, a1_max);
}

// =============================================================================
// BN04 — Single-phase limit  (α₁ = 1, pinf₁ = 0 → matches 5-var HLLC)
// =============================================================================
static void test_BN04() {
    printf("\n-- BN04  Single-phase limit --\n");

    BNEosParams eos;
    eos.gamma1 = GAMMA;
    eos.pinf1  = 0.0;
    // Phase 2 parameters irrelevant (α₂=0), but must not cause NaN.
    eos.gamma2 = 1.4;
    eos.pinf2  = 0.0;

    // Sod left state (phase 1 = ideal gas matching single-phase solver)
    const double rhoL = 1.0, uL = 0.0, pL = 1.0e5;
    const double rhoR = 0.125, uR = 0.0, pR = 1.0e4;

    // Build Prim2Phase with α₁=1
    auto make_prim_bn = [&](double rho, double u, double p) -> Prim2Phase {
        Prim2Phase q;
        q.alpha1 = 1.0;
        q.rho1   = rho;
        q.rho2   = 0.0;
        q.rho    = rho;
        q.u = u; q.v = 0.0; q.w = 0.0;
        q.p = p;
        const double c2 = eos.gamma1 * p / rho;
        q.c_mix = std::sqrt(c2);
        return q;
    };

    const Prim2Phase BL = make_prim_bn(rhoL, uL, pL);
    const Prim2Phase BR = make_prim_bn(rhoR, uR, pR);

    // BN flux (axis = 0)
    const BNFaceFlux bn_res = hllc_bn_flux(BL, BR, 0, eos);

    // Standard single-phase HLLC flux (declared in operators.hpp)
    // Build Prim structs
    Prim pL_ref, pR_ref;
    pL_ref.rho = rhoL; pL_ref.u = uL; pL_ref.v = 0; pL_ref.w = 0; pL_ref.p = pL;
    pL_ref.T   = pL / (rhoL * R_GAS);
    pL_ref.c   = std::sqrt(GAMMA * pL / rhoL);
    pR_ref.rho = rhoR; pR_ref.u = uR; pR_ref.v = 0; pR_ref.w = 0; pR_ref.p = pR;
    pR_ref.T   = pR / (rhoR * R_GAS);
    pR_ref.c   = std::sqrt(GAMMA * pR / rhoR);

    const auto hllc_ref = hllc_flux(pL_ref, pR_ref, 0);

    // Mapping: BN[0]=α₁ρ₁=ρ, BN[2..5]=momentum+energy; compare to hllc_ref[0..4]
    const double tol = 1.0e-13;
    const double err_rho = std::abs(bn_res.F[0] - hllc_ref[0]);
    const double err_rhou= std::abs(bn_res.F[2] - hllc_ref[1]);
    const double err_rhov= std::abs(bn_res.F[3] - hllc_ref[2]);
    const double err_rhow= std::abs(bn_res.F[4] - hllc_ref[3]);
    const double err_E   = std::abs(bn_res.F[5] - hllc_ref[4]);
    const double max_err = std::max({err_rho, err_rhou, err_rhov, err_rhow, err_E});

    check("BN04  α₁=1 BN flux == HLLC flux", max_err < tol, max_err, tol);
    printf("         (F_BN[0,2-5] vs F_HLLC[0-4]  max_err=%.3e)\n", max_err);
}

// =============================================================================
int main() {
    printf("=== P4.4 gate: Allaire 5-equation two-phase model ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d  NVAR_BN=%d\n",
           NB, NG, NB2, NCELL, NVAR_BN);

    test_BN01();
    test_BN02();
    test_BN03();
    test_BN04();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0)
        printf("==> PASS  P4.4 gate cleared\n");
    else
        printf("==> FAIL  P4.4 gate NOT cleared\n");
    return (n_fail == 0) ? 0 : 1;
}
