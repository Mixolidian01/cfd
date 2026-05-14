// wall_model.hpp — Wall-modelled LES (WMLES) interface (P7.6)
//
// Two models:
//   1. Algebraic log-law (Reichardt 1951 composite formula):
//        u⁺ = (1/κ)ln(1+κy⁺) + 7.8[1−e^{−y⁺/11} − (y⁺/11)e^{−y⁺/3}]
//      Continuous from viscous sublayer (u⁺→y⁺) to log layer.
//      Inverted via Newton iteration to obtain u_τ from (u_t, y_m, ν).
//
//   2. ODE mixing-length (TBLE, equilibrium):
//        d/dy[(ν + ν_t) du/dy] = 0,  ν_t = (κ y D)² |du/dy|
//        D = 1 − e^{−y⁺/A⁺}    (van Driest damping)
//      Solved by Picard iteration on the integral u_m = ∫₀^{y_m} u_τ²/(ν+ν_t) dy.
//
// Ghost-cell application (NG ghost layers, low wall at index side=0):
//   Ghost layer gl=0 (y = −h/2): u_ghost = u_interior − h·τ_w/μ
//   Ghost layer gl=1 (y = −3h/2): linear extrapolation outward
//   No-penetration: wall-normal velocity component reversed.
//
// References:
//   Larsson et al. (2016) Annual Research Briefs, CTR
//   Cabot & Moin (1999) Flow Turbulence Combust. 63:269–291
//   Reichardt (1951) Z.Angew.Math.Mech. 31:208–219

#pragma once
#include "mesh/cell_block.hpp"
#include <cmath>
#include <algorithm>

static constexpr int  WM_MAX_ITER  = 60;
static constexpr double WM_UTAU_MIN = 1e-14;

struct WallModelCfg {
    double kappa   = 0.41;  // von Karman constant
    double B       = 5.2;   // log-law constant  (Coles-Fernholz)
    double A_plus  = 26.0;  // van Driest inner-layer constant
    double tol     = 1e-8;  // Newton/Picard tolerance
    bool   use_ode = false; // true → ODE mixing-length; false → algebraic Reichardt
    int    ode_pts = 128;   // integration points for ODE model
};

// =============================================================================
// Composite Reichardt wall-law  u⁺(y⁺)
// =============================================================================
inline double reichardt_uplus(double yp, double kappa) noexcept {
    return (1.0/kappa)*std::log(1.0 + kappa*yp)
         + 7.8*(1.0 - std::exp(-yp/11.0) - (yp/11.0)*std::exp(-yp/3.0));
}

// =============================================================================
// Algebraic log-law model
//   Solves:  u_t = u_τ · R(y_m·u_τ/ν)   for  u_τ ≥ 0.
//   Uses Newton iteration with finite-difference Jacobian.
//   u_t > 0 required; returns 0 if u_t == 0.
// =============================================================================
double wm_log_law(double u_t, double y_m, double nu,
                  const WallModelCfg& cfg = {}) noexcept;

// =============================================================================
// ODE mixing-length model (equilibrium TBLE)
//   Integrates the equilibrium channel profile with van Driest damping.
//   More accurate than log-law for non-equilibrium boundary layers.
// =============================================================================
double wm_ode_ml(double u_t, double y_m, double nu,
                 const WallModelCfg& cfg = {}) noexcept;

// =============================================================================
// Dispatch: returns u_τ from either model based on cfg.use_ode.
// =============================================================================
inline double wm_friction_vel(double u_t, double y_m, double nu,
                               const WallModelCfg& cfg) noexcept {
    return cfg.use_ode ? wm_ode_ml(u_t, y_m, nu, cfg)
                       : wm_log_law(u_t, y_m, nu, cfg);
}

// =============================================================================
// Apply WMLES ghost cells for one wall-adjacent cell.
//   ci, cj, ck : interior cell index (NG ≤ index < NG+NB)
//   wall_ax    : axis perpendicular to wall  (0=x, 1=y, 2=z)
//   side       : 0 = low-index wall, 1 = high-index wall
//   nu         : kinematic viscosity  [m²/s]
// Fills both ghost layers (gl = 0 and 1 for NG=2).
// Returns τ_w = ρ·u_τ²  [Pa].
// =============================================================================
double wm_apply_ghost(CellBlock& blk, int ci, int cj, int ck,
                      int wall_ax, int side, double nu,
                      const WallModelCfg& cfg = {}) noexcept;

// Convenience: apply WMLES to all wall faces in a block given axis and side.
// nu is assumed uniform (Sutherland already applied upstream if needed).
void wm_apply_wall(CellBlock& blk, int wall_ax, int side, double nu,
                   const WallModelCfg& cfg = {}) noexcept;
