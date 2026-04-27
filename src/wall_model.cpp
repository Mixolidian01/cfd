// wall_model.cpp — WMLES algebraic and ODE models (P7.6)
#include "../include/wall_model.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

// =============================================================================
// wm_log_law: Newton iteration to invert Reichardt formula
//   F(u_τ) = u_τ · R(y_m·u_τ/ν) − u_t = 0
// =============================================================================
double wm_log_law(double u_t, double y_m, double nu,
                  const WallModelCfg& cfg) noexcept {
    if (u_t <= WM_UTAU_MIN) return 0.0;

    // Initial guess: assume log-law region, u+ ≈ (1/κ)ln(y+)+B
    // u_t/u_τ = (1/κ)ln(y_m·u_τ/ν)+B → solve approximately
    double utau = u_t * cfg.kappa / (std::log(u_t * y_m / nu + 1.0) + cfg.kappa * cfg.B);
    utau = std::max(utau, WM_UTAU_MIN);

    for (int iter = 0; iter < WM_MAX_ITER; ++iter) {
        const double yp = y_m * utau / nu;
        const double up = reichardt_uplus(yp, cfg.kappa);
        const double F  = utau * up - u_t;
        if (std::fabs(F) < cfg.tol * u_t + 1e-15) break;

        // Jacobian via finite differences
        const double eps  = 1e-6 * utau + 1e-20;
        const double yp2  = y_m * (utau + eps) / nu;
        const double up2  = reichardt_uplus(yp2, cfg.kappa);
        const double dF   = ((utau + eps)*up2 - utau*up) / eps;
        if (std::fabs(dF) < 1e-20) break;

        utau = std::max(utau - F / dF, WM_UTAU_MIN);
    }
    return utau;
}

// =============================================================================
// wm_ode_ml: equilibrium TBLE with van Driest damping
//   ν_t = (κ y D)² |du/dy|,  D = 1−exp(−y⁺/A⁺)
//   Equilibrium: (ν + ν_t) du/dy = u_τ²  (wall stress drives the layer)
//   → du/dy = u_τ²/(ν + ν_t)
//   → u_m = ∫₀^{y_m} u_τ²/(ν + ν_t(y)) dy  (integral depends on u_τ via y⁺)
//
// Picard iteration:
//   1. Start with u_τ from algebraic log-law.
//   2. Compute ν_t(y) profile, integrate to get u_pred.
//   3. Update u_τ: u_τ ← u_τ · sqrt(u_t / u_pred)  (energy scaling).
//   4. Repeat until converged.
// =============================================================================
double wm_ode_ml(double u_t, double y_m, double nu,
                 const WallModelCfg& cfg) noexcept {
    if (u_t <= WM_UTAU_MIN) return 0.0;

    // Start with algebraic guess
    double utau = wm_log_law(u_t, y_m, nu, cfg);
    if (utau <= WM_UTAU_MIN) return 0.0;

    const int N = cfg.ode_pts;
    const double dy = y_m / N;

    for (int iter = 0; iter < WM_MAX_ITER; ++iter) {
        // Integrate u_pred = ∫₀^{y_m} utau²/(ν + ν_t) dy  (trapezoidal rule)
        double u_pred = 0.0;
        double f_prev = 1.0;  // f = u_τ²/(ν+ν_t) / u_τ² = ν/(ν+ν_t) at y=0 → 1
        for (int i = 1; i <= N; ++i) {
            const double y   = i * dy;
            const double yp  = y * utau / nu;
            const double D   = 1.0 - std::exp(-yp / cfg.A_plus);
            // ν_t in explicit form: ν_t = (κ y D)² / (ν + ν_t) * u_τ²
            // Approximate: use ν_t ≈ (κ y D)² * utau² / ν for outer step
            const double lm  = cfg.kappa * y * D;
            // Self-consistent ν_t = lm² * |du/dy| = lm² * utau²/(ν+ν_t)
            // Solve: ν_t = lm²·utau²/(ν+ν_t) → ν_t² + ν·ν_t − lm²·utau² = 0
            const double disc = nu*nu + 4.0*lm*lm*utau*utau;
            const double nut  = 0.5*(-nu + std::sqrt(disc));
            const double f_cur = utau*utau / (nu + nut);
            u_pred += 0.5*(f_prev + f_cur) * dy;
            f_prev = f_cur;
        }

        const double ratio = u_t / (u_pred + 1e-300);
        const double utau_new = utau * std::sqrt(ratio);
        if (std::fabs(utau_new - utau) < cfg.tol * utau + 1e-15) {
            utau = utau_new;
            break;
        }
        // Damped update to prevent oscillation
        utau = 0.5*utau + 0.5*utau_new;
        utau = std::max(utau, WM_UTAU_MIN);
    }
    return utau;
}

// =============================================================================
// wm_apply_ghost: fill ghost cells for one wall-adjacent interior cell
// =============================================================================
double wm_apply_ghost(CellBlock& blk, int ci, int cj, int ck,
                      int wall_ax, int side, double nu,
                      const WallModelCfg& cfg) noexcept {
    assert(ci >= NG && ci < NG+NB);
    assert(cj >= NG && cj < NG+NB);
    assert(ck >= NG && ck < NG+NB);
    assert(wall_ax >= 0 && wall_ax < 3);
    assert(side == 0 || side == 1);

    const int idx_int = cell_idx(ci, cj, ck);
    const double rho  = blk.Q[0][idx_int];
    const double h    = blk.h;

    if (rho < 1e-10) return 0.0;

    // Interior cell distance from wall: 1st cell = h/2
    const double y_m = 0.5 * h;

    // Extract wall-parallel velocity components
    const double ru = blk.Q[1][idx_int];
    const double rv = blk.Q[2][idx_int];
    const double rw = blk.Q[3][idx_int];
    double u[3] = { ru/rho, rv/rho, rw/rho };

    // Zero out wall-normal component → wall-parallel magnitude
    double u_n = u[wall_ax];
    u[wall_ax] = 0.0;
    const double u_t = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

    // Compute friction velocity and wall stress
    const double utau = wm_friction_vel(u_t, y_m, nu, cfg);
    const double tau_w = rho * utau * utau;   // [Pa]
    const double mu    = rho * nu;

    // Ghost layer indices: for side=0 (low wall), ghost is at lower indices.
    // Interior cell: (ci,cj,ck), ghost gl=0 is one step toward wall,
    // ghost gl=1 is two steps toward wall.
    for (int gl = 0; gl < NG; ++gl) {
        int gi = ci, gj = cj, gk = ck;
        int step = gl + 1;  // how many steps from interior toward wall
        if (side == 0) {
            if (wall_ax == 0) gi = ci - step;
            else if (wall_ax == 1) gj = cj - step;
            else gk = ck - step;
        } else {
            if (wall_ax == 0) gi = ci + step;
            else if (wall_ax == 1) gj = cj + step;
            else gk = ck + step;
        }

        const int idx_g = cell_idx(gi, gj, gk);

        // Ghost density: copy from interior
        blk.Q[0][idx_g] = rho;

        // Wall-parallel ghost velocity: set so derivative = τ_w/μ
        // du/dn|_wall ≈ (u_int - u_ghost) / (step * h) = τ_w / μ
        // → u_ghost = u_int - step * h * τ_w / μ
        const double scale = (u_t > WM_UTAU_MIN)
                             ? 1.0 - (step * h * tau_w / mu) / u_t
                             : 0.0;
        double u_g[3] = {
            u[0] * scale,
            u[1] * scale,
            u[2] * scale
        };
        // Wall-normal: no penetration (image method)
        u_g[wall_ax] = -u_n;

        blk.Q[1][idx_g] = rho * u_g[0];
        blk.Q[2][idx_g] = rho * u_g[1];
        blk.Q[3][idx_g] = rho * u_g[2];

        // Energy: copy from interior (adiabatic wall approx)
        blk.Q[4][idx_g] = blk.Q[4][idx_int];
    }

    return tau_w;
}

// =============================================================================
// wm_apply_wall: apply WMLES to a full wall face
// =============================================================================
void wm_apply_wall(CellBlock& blk, int wall_ax, int side, double nu,
                   const WallModelCfg& cfg) noexcept {
    // Iterate over all interior cells in the wall-adjacent layer
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        // Only process cells adjacent to the wall face
        bool is_wall_adj = false;
        if (wall_ax == 0) is_wall_adj = (side == 0) ? (i == NG) : (i == NG+NB-1);
        if (wall_ax == 1) is_wall_adj = (side == 0) ? (j == NG) : (j == NG+NB-1);
        if (wall_ax == 2) is_wall_adj = (side == 0) ? (k == NG) : (k == NG+NB-1);
        if (!is_wall_adj) continue;
        wm_apply_ghost(blk, i, j, k, wall_ax, side, nu, cfg);
    }
}
