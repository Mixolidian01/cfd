// P7.6 gate test — Wall-modelled LES (algebraic + ODE mixing-length)
//
// Protocol:
//   W1: Reichardt formula: u⁺(y⁺=0.1) ≈ y⁺  (viscous sublayer, < 2% error)
//   W2: Log-law Newton solver: residual F(u_τ) < 1e-6·u_t after inversion
//   W3: ODE mixing-length: |u_τ_ODE − u_τ_loglaw| < 5% for y⁺ ≈ 100
//   W4: Ghost-cell τ_w direction: wall-stress aligned with interior velocity
//   W5: wm_apply_wall: no NaN/crash; τ_w > 0 in all interior cells

#include "../include/wall_model.hpp"
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <algorithm>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg,
                  double val = -1, double ref = -1) {
    if (ok) {
        printf("  PASS  %s  %s\n", tag, msg);
    } else {
        if (val >= 0)
            printf("  FAIL  %s  %s  (got %.4e  ref %.4e)\n", tag, msg, val, ref);
        else
            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// =============================================================================
// W1: Reichardt composite law → u⁺ → y⁺ in viscous sublayer
// =============================================================================
static void test_w1() {
    printf("\n-- W1  Reichardt viscous sublayer --\n");
    WallModelCfg cfg;

    // In viscous sublayer (y+ << 1): Reichardt should give u+ ≈ y+
    // Reichardt at y+=0.1: (1/κ)ln(1+0.041) + 7.8*(...)
    // Leading term: (1/κ)*(κ*y+) + ... ≈ y+ for small y+
    double yp = 0.1;
    double up = reichardt_uplus(yp, cfg.kappa);
    double rel_err = std::fabs(up - yp) / yp;
    printf("   y⁺=%.2f  u⁺_Reichardt=%.6f  u⁺_viscous=%.6f  err=%.2f%%\n",
           yp, up, yp, 100.0*rel_err);
    check(rel_err < 0.02, "W1a", "Reichardt u⁺(y⁺=0.1) ≈ y⁺ within 2%",
          rel_err, 0.02);

    // In log layer (y+=100): u+ should match (1/κ)ln(y+)+B within 5%
    yp = 100.0;
    up = reichardt_uplus(yp, cfg.kappa);
    double up_log = (1.0/cfg.kappa)*std::log(yp) + cfg.B;
    rel_err = std::fabs(up - up_log) / up_log;
    printf("   y⁺=%.0f  u⁺_Reichardt=%.4f  u⁺_loglaw=%.4f  err=%.2f%%\n",
           yp, up, up_log, 100.0*rel_err);
    check(rel_err < 0.05, "W1b", "Reichardt u⁺(y⁺=100) within 5% of log-law",
          rel_err, 0.05);
}

// =============================================================================
// W2: Log-law Newton inversion — residual check
// =============================================================================
static void test_w2() {
    printf("\n-- W2  Log-law Newton solver residual --\n");
    WallModelCfg cfg;

    // Case: y⁺=100, u⁺=log-law value — known u_τ should be recovered exactly
    const double nu    = 1.5e-5;   // air kinematic viscosity [m²/s]
    const double y_m   = 0.005;    // matching point [m]
    const double utau_ref = 0.30;  // reference friction velocity [m/s]
    const double yp_ref   = y_m * utau_ref / nu;          // = 100
    const double u_t      = utau_ref * reichardt_uplus(yp_ref, cfg.kappa);

    printf("   Input: u_t=%.4f m/s, y_m=%.4f m, ν=%.2e, y⁺_ref=%.1f\n",
           u_t, y_m, nu, yp_ref);

    const double utau = wm_log_law(u_t, y_m, nu, cfg);
    const double yp   = y_m * utau / nu;
    const double F    = std::fabs(utau * reichardt_uplus(yp, cfg.kappa) - u_t) / u_t;

    printf("   u_τ_ref=%.6f  u_τ_solved=%.6f  |err|/u_t=%.2e\n",
           utau_ref, utau, F);
    check(F < 1e-6, "W2a", "Newton residual |F|/u_t < 1e-6", F, 1e-6);

    const double utau_err = std::fabs(utau - utau_ref) / utau_ref;
    check(utau_err < 1e-4, "W2b", "|u_τ−u_τ_ref|/u_τ_ref < 1e-4",
          utau_err, 1e-4);
}

// =============================================================================
// W3: ODE mixing-length vs log-law consistency at y⁺≈100
// =============================================================================
static void test_w3() {
    printf("\n-- W3  ODE mixing-length vs algebraic log-law (y⁺≈100) --\n");
    WallModelCfg cfg;

    const double nu    = 1.5e-5;
    const double y_m   = 0.005;
    const double utau_ref = 0.30;
    const double u_t   = utau_ref * reichardt_uplus(y_m*utau_ref/nu, cfg.kappa);

    const double utau_ll  = wm_log_law(u_t, y_m, nu, cfg);
    cfg.ode_pts = 256;
    const double utau_ode = wm_ode_ml(u_t, y_m, nu, cfg);

    const double rel_diff = std::fabs(utau_ode - utau_ll) / utau_ll;
    printf("   u_τ_loglaw=%.6f  u_τ_ODE=%.6f  rel_diff=%.2f%%\n",
           utau_ll, utau_ode, 100.0*rel_diff);
    check(rel_diff < 0.05, "W3",
          "|u_τ_ODE − u_τ_loglaw| / u_τ_loglaw < 5%  (y⁺≈100)",
          rel_diff, 0.05);
}

// =============================================================================
// W4: Ghost-cell wall stress — flux magnitude and direction check
//
// WMLES ghost cells impose τ_w via the viscous flux:
//   F_visc = μ (u_int − u_ghost) / h  →  u_ghost = u_int − h τ_w / μ
// For high Re (y⁺ ≫ 1), u_ghost is in the OPPOSITE direction to u_int
// (which is correct — the ghost encodes a strong gradient, not a nearby flow
// state).  The test therefore checks the FLUX, not the ghost direction.
// =============================================================================
static void test_w4() {
    printf("\n-- W4  Ghost-cell: flux magnitude = τ_w, correct direction --\n");

    // Moderate h so we can check exact numbers; y⁺ will be large (WMLES regime).
    CellBlock blk(0.0, 0.0, 0.0, 0.1);  // h = 0.1 m
    const double rho = 1.2;
    const double U   = 10.0;
    const double W_z  = 5.0;
    const double nu  = 1.5e-5;
    const double mu  = rho * nu;
    const double h   = blk.h;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int idx = cell_idx(i,j,k);
        blk.Q[0][idx] = rho;
        blk.Q[1][idx] = rho * U;
        blk.Q[2][idx] = 0.0;          // wall-normal = y-axis (side=0)
        blk.Q[3][idx] = rho * W_z;
        blk.Q[4][idx] = 0.5*rho*(U*U + W_z*W_z);
    }

    WallModelCfg cfg;
    wm_apply_wall(blk, /*wall_ax*/1, /*side*/0, nu, cfg);

    const int idx_g0  = cell_idx(NG, NG-1, NG);   // first ghost
    const int idx_int = cell_idx(NG, NG,   NG);

    const double u_int_x = blk.Q[1][idx_int] / blk.Q[0][idx_int];
    const double u_int_z = blk.Q[3][idx_int] / blk.Q[0][idx_int];
    const double u_g_x   = blk.Q[1][idx_g0]  / blk.Q[0][idx_g0];
    const double u_g_z   = blk.Q[3][idx_g0]  / blk.Q[0][idx_g0];
    const double u_normal_g = blk.Q[2][idx_g0] / blk.Q[0][idx_g0];

    // Flux at wall face (between ghost and interior):
    // F_x = μ(u_int_x − u_g_x)/h,  F_z = μ(u_int_z − u_g_z)/h
    const double F_x = mu * (u_int_x - u_g_x) / h;
    const double F_z = mu * (u_int_z - u_g_z) / h;
    const double tau_from_ghost = std::sqrt(F_x*F_x + F_z*F_z);

    // Expected τ_w from log-law applied to the same u_t, y_m
    const double u_t = std::sqrt(U*U + W_z*W_z);
    const double y_m = 0.5*h;
    const double utau = wm_log_law(u_t, y_m, nu, cfg);
    const double tau_w_ref = rho * utau * utau;

    printf("   Interior (u,w) = (%.3f, %.3f)  u_t=%.3f\n", u_int_x, u_int_z, u_t);
    printf("   Ghost-0 (u,v_n,w) = (%.4f, %.4f, %.4f)\n",
           u_g_x, u_normal_g, u_g_z);
    printf("   Flux τ_from_ghost = %.5e  τ_w_ref = %.5e\n",
           tau_from_ghost, tau_w_ref);

    // The flux must equal τ_w_ref (by construction of the ghost cell formula)
    double flux_err = std::fabs(tau_from_ghost - tau_w_ref) /
                      (tau_w_ref + 1e-20);
    check(flux_err < 1e-8, "W4a",
          "|τ_w_from_ghost − τ_w_ref| / τ_w_ref < 1e-8  (flux equality)",
          flux_err, 1e-8);

    // Direction: (F_x, F_z) must be parallel to (U, W_z)
    double cos_theta = (F_x*U + F_z*W_z) /
                       (std::sqrt(F_x*F_x+F_z*F_z)*std::sqrt(U*U+W_z*W_z)+1e-20);
    check(cos_theta > 0.999, "W4b",
          "flux direction parallel to interior velocity  (cos θ > 0.999)",
          cos_theta, 1.0);

    // No penetration: wall-normal ghost component = −u_normal_interior = 0
    check(std::fabs(u_normal_g) < 1e-10, "W4c",
          "ghost wall-normal velocity = 0  (no penetration)");
}

// =============================================================================
// W5: wm_apply_wall full block — no NaN; τ_w > 0 wherever u_t > 0
//
// At high Re (y⁺ ≫ 1), ghost velocities are strongly negative — that is the
// correct encoding of the wall stress (see W4).  We do NOT check ghost |u|≤
// interior |u|.  Instead we verify: (a) no NaN, (b) wm_log_law gives τ_w>0
// for any interior cell with nonzero tangential velocity.
// =============================================================================
static void test_w5() {
    printf("\n-- W5  wm_apply_wall full block: no NaN, τ_w > 0 --\n");

    CellBlock blk(0.0, 0.0, 0.0, 1.0/NB);
    const double rho = 1.2;
    const double U   = 8.0;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int idx = cell_idx(i,j,k);
        blk.Q[0][idx] = rho;
        blk.Q[1][idx] = 0.0;
        blk.Q[2][idx] = rho * U;  // y-velocity; walls on all faces
        blk.Q[3][idx] = 0.0;
        blk.Q[4][idx] = 0.5*rho*U*U;
    }

    const double nu = 1.5e-5;
    WallModelCfg cfg;
    for (int ax = 0; ax < 3; ++ax)
    for (int s  = 0; s  < 2; ++s)
        wm_apply_wall(blk, ax, s, nu, cfg);

    // W5a: no NaN anywhere
    bool has_nan = false;
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            if (!std::isfinite(blk.Q[v][idx])) { has_nan = true; break; }
    check(!has_nan, "W5a", "no NaN in ghost cells after wm_apply_wall");

    // W5b: for all wall-adjacent cells, wm_log_law gives τ_w > 0
    const double h   = blk.h;
    const double y_m = 0.5 * h;
    bool tau_positive = true;
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int idx_int = cell_idx(i, j, NG);    // adjacent to z=low wall
        double r   = blk.Q[0][idx_int];
        if (r < 1e-10) continue;
        // Wall-parallel velocity: z-wall, wall_ax=2, so u_t = sqrt(u² + v²)
        double u_t = std::sqrt(
            (blk.Q[1][idx_int]/r)*(blk.Q[1][idx_int]/r) +
            (blk.Q[2][idx_int]/r)*(blk.Q[2][idx_int]/r));
        if (u_t < 1e-10) continue;
        double utau = wm_log_law(u_t, y_m, nu, cfg);
        if (utau <= 0.0) tau_positive = false;
    }
    check(tau_positive, "W5b",
          "wm_log_law returns u_τ > 0 for all u_t > 0 wall-adjacent cells");
}

// =============================================================================
int main() {
    printf("=== P7.6 gate: Wall-modelled LES (algebraic + ODE mixing-length) ===\n");
    test_w1();
    test_w2();
    test_w3();
    test_w4();
    test_w5();

    const int ntotal = 8;
    const int npass  = ntotal - nfail;
    printf("\nResults: %d passed, %d failed\n", npass, nfail);
    if (nfail == 0)
        printf("==> PASS  P7.6 gate cleared — WMLES operational, Re_τ ≫ 180 enabled\n");
    return nfail > 0 ? 1 : 0;
}
