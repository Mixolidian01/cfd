// DESIGN.md reference: Layer 2 — Discrete Operators
//
// Fix log:
//   B1  : viscous energy equation: dE/dt|_visc includes u*ax + v*ay + w*az
//         (u·div(τ) work term was missing).
//   P2.2: convective_rhs refactored to face-centred loop.
//         Each of 3*(NB+1)*NB^2 = 1728 face fluxes computed once
//         vs 3*2*NB^3 = 3072 in the original cell loop — 43.7% fewer HLLC
//         evaluations per block.
//         Sign convention: F is the flux leaving the left cell / entering the
//         right cell.  rhs[left] -= ih*F;  rhs[right] += ih*F.
//   P2.3: Primitive variable cache pc[NCELL] pre-computed ONCE in compute_rhs
//         and shared by both convective and viscous operators.
//         Reduces EOS inversions from ~3584 (convective) + ~6656 (viscous)
//         to NCELL=1000 per compute_rhs call.  Enables SIMD on inner i-loop.
//   B5  : viscous_rhs_impl restructured to face-averaged µ.
//         Conservative divergence form div(τ) with µ_face = ½(µ_L + µ_R)
//         replaces the constant-µ expansion µ·(Δu + ⅓∇div u).
//         A separate µ cache (µ_arr[NCELL]) is built once in compute_rhs
//         alongside the prim cache, adding NCELL Sutherland calls (1000)
//         at negligible cost relative to the gradient evaluation.
//         Energy: cell-centred tau:S and κ·ΔT are unchanged (O(h²)).

#include "operators.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>
#ifdef _OPENMP
#  include <omp.h>
#endif

// =============================================================================
// HLLC Riemann flux
// =============================================================================
// Reference: Toro (2009) §10.4, Batten et al. (1997).
// axis: 0=x, 1=y, 2=z  — selects which velocity component is normal.
//
// Sign convention: flux is in the +axis direction.
// The five components of the returned flux correspond to:
//   [0] mass flux        rho * u_n
//   [1] x-momentum flux  rho*u*u_n + p*nx  (nx=1 for axis=0, else 0)
//   [2] y-momentum flux
//   [3] z-momentum flux
//   [4] energy flux      (E+p)*u_n

std::array<double,5> hllc_flux(const Prim& L, const Prim& R, int axis) noexcept
{
    double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
    double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

    double sL = std::min(uL - L.c, uR - R.c);
    double sR = std::max(uL + L.c, uR + R.c);

    double numer = R.p - L.p + L.rho*uL*(sL - uL) - R.rho*uR*(sR - uR);
    double denom = L.rho*(sL - uL) - R.rho*(sR - uR);
    double sStar = (std::abs(denom) > 1e-300) ? numer / denom : 0.5*(uL + uR);

    auto phys_flux = [&](const Prim& q) -> std::array<double,5> {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GAMMA-1.0) + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        return {q.rho*un,
                q.rho*q.u*un + (axis==0?q.p:0.0),
                q.rho*q.v*un + (axis==1?q.p:0.0),
                q.rho*q.w*un + (axis==2?q.p:0.0),
                (E+q.p)*un};
    };

    auto star_flux = [&](const Prim& q, double sK, double sS)
        -> std::array<double,5>
    {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GAMMA-1.0) + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        double coeff = q.rho * (sK - un) / (sK - sS);
        double rho_s  = coeff;
        double rhou_s = coeff * (axis==0 ? sS : q.u);
        double rhov_s = coeff * (axis==1 ? sS : q.v);
        double rhow_s = coeff * (axis==2 ? sS : q.w);
        double E_s    = coeff * (E/q.rho + (sS - un)*(sS + q.p/(q.rho*(sK-un))));

        auto F = phys_flux(q);
        double rho  = q.rho;
        double rhou = rho*q.u, rhov = rho*q.v, rhow = rho*q.w;
        return {F[0] + sK*(rho_s  - rho ),
                F[1] + sK*(rhou_s - rhou),
                F[2] + sK*(rhov_s - rhov),
                F[3] + sK*(rhow_s - rhow),
                F[4] + sK*(E_s    - E   )};
    };

    if      (sL >= 0.0) return phys_flux(L);
    else if (sR <= 0.0) return phys_flux(R);
    else if (sStar >= 0.0) return star_flux(L, sL, sStar);
    else                   return star_flux(R, sR, sStar);
}

// =============================================================================
// P2.3 — Primitive variable and viscosity caches
// =============================================================================
// Pre-compute all NCELL primitive states (and Sutherland µ) once per
// compute_rhs call.  Interior + ghost cells filled → stencil = pure lookup.
static void fill_prim_cache(const CellBlock& blk, Prim* pc) noexcept {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i)
        pc[cell_idx(i,j,k)] = blk.prim(i,j,k);
}

// B5: µ cache — one Sutherland call per cell, shared across viscous operator.
static void fill_mu_cache(const Prim* pc, double* mu_arr) noexcept {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const int idx = cell_idx(i,j,k);
        mu_arr[idx] = sutherland(pc[idx].T);
    }
}

// =============================================================================
// P2.2 — Convective RHS: face-centred loop
// =============================================================================
// Iterates over faces rather than cells.  For axis d, there are
// (NB+1)*NB*NB interior+boundary faces; each evaluated exactly once.
//
// Flux sign: F is positive leaving the left cell:
//   rhs[left]  -= ih * F   (divergence: flux out → subtract)
//   rhs[right] += ih * F   (divergence: flux in  → add)
// This telescopes exactly to the standard finite-volume divergence.
static void convective_rhs_impl(const Prim* pc, CellBlock& rhs, double h) noexcept
{
    const double ih = 1.0 / h;

    // X-direction: face between (i,j,k) and (i+1,j,k), i ∈ [ilo-1, ihi]
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo()-1; i <= ihi(); ++i) {
        const auto F = hllc_flux(pc[cell_idx(i,  j,k)],
                                  pc[cell_idx(i+1,j,k)], 0);
        if (i >= ilo())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,  j,k)] -= ih * F[v];
        if (i+1 <= ihi())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i+1,j,k)] += ih * F[v];
    }

    // Y-direction: face between (i,j,k) and (i,j+1,k), j ∈ [ilo-1, ihi]
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo()-1; j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const auto F = hllc_flux(pc[cell_idx(i,j,  k)],
                                  pc[cell_idx(i,j+1,k)], 1);
        if (j >= ilo())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j,  k)] -= ih * F[v];
        if (j+1 <= ihi())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j+1,k)] += ih * F[v];
    }

    // Z-direction: face between (i,j,k) and (i,j,k+1), k ∈ [ilo-1, ihi]
    for (int k = ilo()-1; k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const auto F = hllc_flux(pc[cell_idx(i,j,k  )],
                                  pc[cell_idx(i,j,k+1)], 2);
        if (k >= ilo())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j,k  )] -= ih * F[v];
        if (k+1 <= ihi())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j,k+1)] += ih * F[v];
    }
}

// =============================================================================
// B5 — Viscous RHS: face-averaged µ, conservative divergence form
// =============================================================================
// Replaces the old constant-µ expansion  a_i = µ·(Δu_i + ⅓ ∂(div u)/∂x_i).
// That form is only equivalent to div(τ) when µ is spatially constant; it
// misses the ∇µ·(∂u/∂x + ...) term at AMR coarse-fine interfaces.
//
// New form: for each cell (i,j,k), compute the Conservative divergence
//
//   a_x = h⁻¹ [(τ_xx|_{i+½} - τ_xx|_{i-½})
//              + (τ_yx|_{j+½} - τ_yx|_{j-½})
//              + (τ_zx|_{k+½} - τ_zx|_{k-½})]
//
// where τ_ab|_{face} = µ_face · (∂u_a/∂x_b + ∂u_b/∂x_a - δ_ab⅔ div u)|_{face}
// and µ_face = ½(µ_left + µ_right)  (arithmetic average).
//
// Velocity gradients at face i+½:
//   normal:    (u[i+1] - u[i]) / h           (1st-order forward, 2nd-order at face)
//   tangential: average of centred differences on both sides of the face, e.g.
//               du/dy|_{i+½} = ((u[i+1,j+1]-u[i+1,j-1]) + (u[i,j+1]-u[i,j-1])) / 4h
//
// Energy: cell-centred τ:S and κ·Δ(T) are unchanged (O(h²), conserved on
// periodic domains to machine precision as confirmed by T04/S07).
// =============================================================================
static constexpr double PR = 0.72;
static constexpr double CP = GAMMA * R_GAS / (GAMMA - 1.0);

static void viscous_rhs_impl(const Prim* pc, const double* mu_arr,
                              CellBlock& rhs, double h) noexcept
{
    const double ih      = 1.0 / h;
    const double ih2     = ih * ih;
    const double ih_half = 0.5 * ih;   // 1/(2h): face tangential gradients

    // Velocity and µ accessors from caches
    auto U  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].u; };
    auto V  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].v; };
    auto W  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].w; };
    auto Tf = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].T; };
    auto MU = [&](int ii,int jj,int kk){ return mu_arr[cell_idx(ii,jj,kk)]; };

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {

        // ── Face-averaged µ at the 6 faces ──────────────────────────────────
        double mu_xp = 0.5*(MU(i,j,k) + MU(i+1,j,  k  ));
        double mu_xm = 0.5*(MU(i,j,k) + MU(i-1,j,  k  ));
        double mu_yp = 0.5*(MU(i,j,k) + MU(i,  j+1,k  ));
        double mu_ym = 0.5*(MU(i,j,k) + MU(i,  j-1,k  ));
        double mu_zp = 0.5*(MU(i,j,k) + MU(i,  j,  k+1));
        double mu_zm = 0.5*(MU(i,j,k) + MU(i,  j,  k-1));

        // ── Velocity gradients at x-faces (i±½) ─────────────────────────────
        // x+½
        double dudx_xp = ih*(U(i+1,j,k)-U(i,j,k));
        double dvdx_xp = ih*(V(i+1,j,k)-V(i,j,k));
        double dwdx_xp = ih*(W(i+1,j,k)-W(i,j,k));
        double dudy_xp = ih_half*(U(i+1,j+1,k)-U(i+1,j-1,k) + U(i,j+1,k)-U(i,j-1,k));
        double dudz_xp = ih_half*(U(i+1,j,k+1)-U(i+1,j,k-1) + U(i,j,k+1)-U(i,j,k-1));
        double dvdy_xp = ih_half*(V(i+1,j+1,k)-V(i+1,j-1,k) + V(i,j+1,k)-V(i,j-1,k));
        double dwdz_xp = ih_half*(W(i+1,j,k+1)-W(i+1,j,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_xp = dudx_xp + dvdy_xp + dwdz_xp;
        // x-½
        double dudx_xm = ih*(U(i,j,k)-U(i-1,j,k));
        double dvdx_xm = ih*(V(i,j,k)-V(i-1,j,k));
        double dwdx_xm = ih*(W(i,j,k)-W(i-1,j,k));
        double dudy_xm = ih_half*(U(i-1,j+1,k)-U(i-1,j-1,k) + U(i,j+1,k)-U(i,j-1,k));
        double dudz_xm = ih_half*(U(i-1,j,k+1)-U(i-1,j,k-1) + U(i,j,k+1)-U(i,j,k-1));
        double dvdy_xm = ih_half*(V(i-1,j+1,k)-V(i-1,j-1,k) + V(i,j+1,k)-V(i,j-1,k));
        double dwdz_xm = ih_half*(W(i-1,j,k+1)-W(i-1,j,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_xm = dudx_xm + dvdy_xm + dwdz_xm;

        // ── Velocity gradients at y-faces (j±½) ─────────────────────────────
        // j+½
        double dudy_yp = ih*(U(i,j+1,k)-U(i,j,k));
        double dvdx_yp = ih_half*(V(i+1,j+1,k)-V(i-1,j+1,k) + V(i+1,j,k)-V(i-1,j,k));
        double dvdy_yp = ih*(V(i,j+1,k)-V(i,j,k));
        double dvdz_yp = ih_half*(V(i,j+1,k+1)-V(i,j+1,k-1) + V(i,j,k+1)-V(i,j,k-1));
        double dwdy_yp = ih*(W(i,j+1,k)-W(i,j,k));
        double dudx_yp = ih_half*(U(i+1,j+1,k)-U(i-1,j+1,k) + U(i+1,j,k)-U(i-1,j,k));
        double dwdz_yp = ih_half*(W(i,j+1,k+1)-W(i,j+1,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_yp = dudx_yp + dvdy_yp + dwdz_yp;
        // j-½
        double dudy_ym = ih*(U(i,j,k)-U(i,j-1,k));
        double dvdx_ym = ih_half*(V(i+1,j-1,k)-V(i-1,j-1,k) + V(i+1,j,k)-V(i-1,j,k));
        double dvdy_ym = ih*(V(i,j,k)-V(i,j-1,k));
        double dvdz_ym = ih_half*(V(i,j-1,k+1)-V(i,j-1,k-1) + V(i,j,k+1)-V(i,j,k-1));
        double dwdy_ym = ih*(W(i,j,k)-W(i,j-1,k));
        double dudx_ym = ih_half*(U(i+1,j-1,k)-U(i-1,j-1,k) + U(i+1,j,k)-U(i-1,j,k));
        double dwdz_ym = ih_half*(W(i,j-1,k+1)-W(i,j-1,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_ym = dudx_ym + dvdy_ym + dwdz_ym;

        // ── Velocity gradients at z-faces (k±½) ─────────────────────────────
        // k+½
        double dudz_zp = ih*(U(i,j,k+1)-U(i,j,k));
        double dvdz_zp = ih*(V(i,j,k+1)-V(i,j,k));
        double dwdz_zp = ih*(W(i,j,k+1)-W(i,j,k));
        double dwdx_zp = ih_half*(W(i+1,j,k+1)-W(i-1,j,k+1) + W(i+1,j,k)-W(i-1,j,k));
        double dwdy_zp = ih_half*(W(i,j+1,k+1)-W(i,j-1,k+1) + W(i,j+1,k)-W(i,j-1,k));
        double dudx_zp = ih_half*(U(i+1,j,k+1)-U(i-1,j,k+1) + U(i+1,j,k)-U(i-1,j,k));
        double dvdy_zp = ih_half*(V(i,j+1,k+1)-V(i,j-1,k+1) + V(i,j+1,k)-V(i,j-1,k));
        double divu_zp = dudx_zp + dvdy_zp + dwdz_zp;
        // k-½
        double dudz_zm = ih*(U(i,j,k)-U(i,j,k-1));
        double dvdz_zm = ih*(V(i,j,k)-V(i,j,k-1));
        double dwdz_zm = ih*(W(i,j,k)-W(i,j,k-1));
        double dwdx_zm = ih_half*(W(i+1,j,k-1)-W(i-1,j,k-1) + W(i+1,j,k)-W(i-1,j,k));
        double dwdy_zm = ih_half*(W(i,j+1,k-1)-W(i,j-1,k-1) + W(i,j+1,k)-W(i,j-1,k));
        double dudx_zm = ih_half*(U(i+1,j,k-1)-U(i-1,j,k-1) + U(i+1,j,k)-U(i-1,j,k));
        double dvdy_zm = ih_half*(V(i,j+1,k-1)-V(i,j-1,k-1) + V(i,j+1,k)-V(i,j-1,k));
        double divu_zm = dudx_zm + dvdy_zm + dwdz_zm;

        // ── Face stresses τ_ab|_{face} = µ_face·(∂u_a/∂x_b + ∂u_b/∂x_a - δ·⅔ div u)
        // x-faces: components needed for ax(τ_xx), ay(τ_xy), az(τ_xz)
        double txx_xp = mu_xp*(2.0*dudx_xp - (2.0/3.0)*divu_xp);
        double txy_xp = mu_xp*(dudy_xp + dvdx_xp);
        double txz_xp = mu_xp*(dudz_xp + dwdx_xp);
        double txx_xm = mu_xm*(2.0*dudx_xm - (2.0/3.0)*divu_xm);
        double txy_xm = mu_xm*(dudy_xm + dvdx_xm);
        double txz_xm = mu_xm*(dudz_xm + dwdx_xm);

        // y-faces: τ_yx(=τ_xy), τ_yy, τ_yz
        double tyx_yp = mu_yp*(dudy_yp + dvdx_yp);
        double tyy_yp = mu_yp*(2.0*dvdy_yp - (2.0/3.0)*divu_yp);
        double tyz_yp = mu_yp*(dvdz_yp + dwdy_yp);
        double tyx_ym = mu_ym*(dudy_ym + dvdx_ym);
        double tyy_ym = mu_ym*(2.0*dvdy_ym - (2.0/3.0)*divu_ym);
        double tyz_ym = mu_ym*(dvdz_ym + dwdy_ym);

        // z-faces: τ_zx(=τ_xz), τ_zy(=τ_yz), τ_zz
        double tzx_zp = mu_zp*(dudz_zp + dwdx_zp);
        double tzy_zp = mu_zp*(dvdz_zp + dwdy_zp);
        double tzz_zp = mu_zp*(2.0*dwdz_zp - (2.0/3.0)*divu_zp);
        double tzx_zm = mu_zm*(dudz_zm + dwdx_zm);
        double tzy_zm = mu_zm*(dvdz_zm + dwdy_zm);
        double tzz_zm = mu_zm*(2.0*dwdz_zm - (2.0/3.0)*divu_zm);

        // ── Conservative momentum divergences ────────────────────────────────
        double ax = ih*((txx_xp-txx_xm) + (tyx_yp-tyx_ym) + (tzx_zp-tzx_zm));
        double ay = ih*((txy_xp-txy_xm) + (tyy_yp-tyy_ym) + (tzy_zp-tzy_zm));
        double az = ih*((txz_xp-txz_xm) + (tyz_yp-tyz_ym) + (tzz_zp-tzz_zm));

        // ── Energy: cell-centred τ:S and κ·Δ(T) ─────────────────────────────
        // Cell-centred gradients (central differences, O(h²))
        double dudx_c = ih_half*(U(i+1,j,k)-U(i-1,j,k));
        double dudy_c = ih_half*(U(i,j+1,k)-U(i,j-1,k));
        double dudz_c = ih_half*(U(i,j,k+1)-U(i,j,k-1));
        double dvdx_c = ih_half*(V(i+1,j,k)-V(i-1,j,k));
        double dvdy_c = ih_half*(V(i,j+1,k)-V(i,j-1,k));
        double dvdz_c = ih_half*(V(i,j,k+1)-V(i,j,k-1));
        double dwdx_c = ih_half*(W(i+1,j,k)-W(i-1,j,k));
        double dwdy_c = ih_half*(W(i,j+1,k)-W(i,j-1,k));
        double dwdz_c = ih_half*(W(i,j,k+1)-W(i,j,k-1));
        double divu_c = dudx_c + dvdy_c + dwdz_c;

        double mu_c   = mu_arr[cell_idx(i,j,k)];
        double txx_c  = mu_c*(2.0*dudx_c - (2.0/3.0)*divu_c);
        double tyy_c  = mu_c*(2.0*dvdy_c - (2.0/3.0)*divu_c);
        double tzz_c  = mu_c*(2.0*dwdz_c - (2.0/3.0)*divu_c);
        double txy_c  = mu_c*(dudy_c + dvdx_c);
        double txz_c  = mu_c*(dudz_c + dwdx_c);
        double tyz_c  = mu_c*(dvdz_c + dwdy_c);
        double visc_pw = txx_c*dudx_c + tyy_c*dvdy_c + tzz_c*dwdz_c
                       + txy_c*(dudy_c+dvdx_c) + txz_c*(dudz_c+dwdx_c)
                       + tyz_c*(dvdz_c+dwdy_c);

        const Prim& c = pc[cell_idx(i,j,k)];
        double kappa  = mu_c * CP / PR;
        double lap_T  = ih2*(Tf(i+1,j,k)-2*c.T+Tf(i-1,j,k)
                           + Tf(i,j+1,k)-2*c.T+Tf(i,j-1,k)
                           + Tf(i,j,k+1)-2*c.T+Tf(i,j,k-1));
        double heat   = kappa * lap_T;

        int idx = cell_idx(i,j,k);
        rhs.Q[1][idx] += ax;
        rhs.Q[2][idx] += ay;
        rhs.Q[3][idx] += az;
        rhs.Q[4][idx] += c.u*ax + c.v*ay + c.w*az + heat + visc_pw;
    }
}

// =============================================================================
// Public wrappers — preserve API for standalone test calls
// =============================================================================
void convective_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    static thread_local std::array<Prim, NCELL> pc;
    fill_prim_cache(blk, pc.data());
    convective_rhs_impl(pc.data(), rhs_blk, blk.h);
}

void viscous_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> mu_arr;
    fill_prim_cache(blk, pc.data());
    fill_mu_cache(pc.data(), mu_arr.data());
    viscous_rhs_impl(pc.data(), mu_arr.data(), rhs_blk, blk.h);
}

// =============================================================================
// Full RHS — P2.3 + B5: build prim and µ caches once, shared by both operators
// =============================================================================
void compute_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> mu_arr;
    fill_prim_cache(blk, pc.data());
    fill_mu_cache(pc.data(), mu_arr.data());

    for (int v = 0; v < NVAR; ++v)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i)
            rhs_blk.Q[v][cell_idx(i,j,k)] = 0.0;

    convective_rhs_impl(pc.data(),             rhs_blk, blk.h);
    viscous_rhs_impl   (pc.data(), mu_arr.data(), rhs_blk, blk.h);
}

// =============================================================================
// Berger-Colella reflux: undo_cf_face_flux
// ─────────────────────────────────────────
// Undoes the coarse HLLC contribution at each CF face so that
// apply_flux_correction() can add the correct fine-side average.
// Net effect after correction:
//   dQ/dt += (1/h) * [F_fine_avg - F_coarse_own]
// =============================================================================
static void undo_cf_face_flux(const BlockTree& tree, int node_idx,
                               CellBlock& rhs) noexcept
{
    const auto& nd  = tree.nodes[node_idx];
    const auto& blk = *nd.block;
    const double ih = 1.0 / blk.h;

    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

    for (int d = 0; d < NFACES; ++d) {
        const int ni = nd.neighbours[d];
        if (ni < 0 || !tree.nodes[ni].has_block()) continue;
        if (tree.nodes[ni].level <= nd.level) continue;

        const int axis  = face_axis[d];
        const int delta = face_delta[d];
        const int bound = (delta > 0) ? ihi() : ilo();

        for (int b = ilo(); b <= ihi(); ++b)
        for (int a = ilo(); a <= ihi(); ++a) {
            int ci, cj, ck, gi, gj, gk;
            if (axis == 0) {
                ci=bound; cj=a;     ck=b;
                gi=bound+delta; gj=a; gk=b;
            } else if (axis == 1) {
                ci=a;     cj=bound; ck=b;
                gi=a; gj=bound+delta; gk=b;
            } else {
                ci=a;     cj=b;     ck=bound;
                gi=a; gj=b; gk=bound+delta;
            }

            Prim interior = blk.prim(ci, cj, ck);
            Prim ghost    = blk.prim(gi, gj, gk);

            std::array<double,5> F;
            if (delta > 0)
                F = hllc_flux(interior, ghost, axis);
            else
                F = hllc_flux(ghost, interior, axis);

            const double sign = (delta > 0) ? +1.0 : -1.0;
            const int idx = cell_idx(ci, cj, ck);
            for (int v = 0; v < NVAR; ++v)
                rhs.Q[v][idx] += sign * ih * F[v];
        }
    }
}

// =============================================================================
// accumulate_cf_fine_fluxes
// ─────────────────────────
// A05-fix4: stage_weight is the SSP-RK3 quadrature weight for this stage:
//   stage 1: w = 1/6,  stage 2: w = 1/6,  stage 3: w = 2/3
// Registers are zeroed ONCE before stage 1 (in advance()).
// Each call accumulates w * F_fine; apply_flux_correction(dt) then applies
// the time-averaged fine flux, giving exact Berger-Colella conservation
// across all three RK3 sub-steps.
//
// Register layout (must match apply_flux_correction in block_tree.cpp):
//   reg[v*NB*NB + jc*NB + ic]
//   axis=0 (x-face, YZ plane): jc = coarse_y_idx (0..NB-1), ic = coarse_z_idx
//   axis=1 (y-face, XZ plane): jc = coarse_z_idx,           ic = coarse_x_idx
//   axis=2 (z-face, XY plane): jc = coarse_y_idx,           ic = coarse_x_idx
//
// A05-fix7: correct fine-to-coarse index mapping for axis=1 and axis=2.
//
// The face loop runs `a` over the first transverse direction and `b` over the
// second.  The physical direction these represent depends on the face axis:
//   axis=0 (x-face): a→y (j), b→z (k)  → jc uses a_local, ic uses b_local ✓
//   axis=1 (y-face): a→x (i), b→z (k)  → jc (→z) must use b_local,
//                                          ic (→x) must use a_local
//   axis=2 (z-face): a→x (i), b→y (j)  → jc (→y) must use b_local,
//                                          ic (→x) must use a_local
//
// The previous code (A05-fix6) used a_local for jc and b_local for ic
// uniformly across all axes.  This was correct only for axis=0; for axis=1
// and axis=2 the a/b roles are swapped, sending fine fluxes from non-diagonal
// octants to the wrong coarse register slots and producing a systematic mass
// leak whenever y- or z-face CF interfaces exist.
// =============================================================================
static void accumulate_cf_fine_fluxes(BlockTree& tree,
                                       double stage_weight) noexcept
{
    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};
    static constexpr int HALF = NB / 2;

    for (int li : tree.leaf_indices()) {
        const auto& nd  = tree.nodes[li];
        const auto& blk = *nd.block;

        // Octant of this fine leaf relative to its parent.
        // Needed to compute the quadrant offset on the coarse face.
        // oct ∈ {0..7}: bit0=ix, bit1=iy, bit2=iz.
        const int parent_idx = nd.parent;
        const int oct = (parent_idx >= 0)
                        ? li - tree.nodes[parent_idx].first_child
                        : 0;
        const int o_ix = oct_ix(oct);  // 0 or 1
        const int o_iy = oct_iy(oct);
        const int o_iz = oct_iz(oct);

        for (int d = 0; d < NFACES; ++d) {
            const int ni = nd.neighbours[d];
            if (ni < 0 || !tree.nodes[ni].has_block()) continue;
            if (tree.nodes[ni].level >= nd.level) continue;

            const int axis  = face_axis[d];
            const int delta = face_delta[d];
            const int bound = (delta > 0) ? ihi() : ilo();

            // Transverse octant offsets for this face axis.
            // off1 is the octant component in the jc direction (1st register index).
            // off2 is the octant component in the ic direction (2nd register index).
            //   axis=0: jc→y, ic→z  → off1=o_iy, off2=o_iz
            //   axis=1: jc→z, ic→x  → off1=o_iz, off2=o_ix
            //   axis=2: jc→y, ic→x  → off1=o_iy, off2=o_ix
            int off1, off2;
            if      (axis == 0) { off1 = o_iy; off2 = o_iz; }
            else if (axis == 1) { off1 = o_iz; off2 = o_ix; }
            else                { off1 = o_iy; off2 = o_ix; }

            // Accumulate into a NB×NB register of COARSE-cell averaged fluxes.
            // Each fine block contributes to one (NB/2)×(NB/2) quadrant.
            // Four fine cells per coarse slot → each contributes 1/4 (applied
            // by area_ratio=0.25 inside accumulate_fine_flux).
            std::vector<double> face_flux(NVAR * NB * NB, 0.0);

            for (int b = ilo(); b <= ihi(); ++b)
            for (int a = ilo(); a <= ihi(); ++a) {
                int ci, cj, ck, gi, gj, gk;
                if (axis == 0) {
                    ci=bound; cj=a;     ck=b;
                    gi=bound+delta; gj=a; gk=b;
                } else if (axis == 1) {
                    ci=a;     cj=bound; ck=b;
                    gi=a; gj=bound+delta; gk=b;
                } else {
                    ci=a;     cj=b;     ck=bound;
                    gi=a; gj=b; gk=bound+delta;
                }

                Prim interior = blk.prim(ci, cj, ck);
                Prim ghost    = blk.prim(gi, gj, gk);

                std::array<double,5> F;
                if (delta > 0) F = hllc_flux(interior, ghost, axis);
                else           F = hllc_flux(ghost, interior, axis);

                // Map fine cell (a,b) to the coarse-cell register index.
                // The mapping depends on which physical direction a and b
                // represent for this face axis (A05-fix7):
                //   axis=0: a→y→jc, b→z→ic  → jc uses a_local, ic uses b_local
                //   axis=1: a→x→ic, b→z→jc  → jc uses b_local, ic uses a_local
                //   axis=2: a→x→ic, b→y→jc  → jc uses b_local, ic uses a_local
                const int a_local = a - ilo();
                const int b_local = b - ilo();
                int jc, ic;
                if (axis == 0) {
                    jc = off1 * HALF + a_local / 2;
                    ic = off2 * HALF + b_local / 2;
                } else {
                    // axis=1 and axis=2: a→x, b→(z or y); jc indexes the
                    // non-x transverse direction, so it must use b_local.
                    jc = off1 * HALF + b_local / 2;
                    ic = off2 * HALF + a_local / 2;
                }

                // Accumulate (+=): 4 fine cells contribute to the same coarse
                // slot; area_ratio=0.25 in accumulate_fine_flux completes the
                // area-weighted average.
                for (int v = 0; v < NVAR; ++v)
                    face_flux[v*NB*NB + jc*NB + ic] += stage_weight * F[v];
            }
            tree.accumulate_fine_flux(li, static_cast<FaceDir>(d), face_flux);
        }
    }
}

// =============================================================================
// tree_rhs
// ─────────
// A05-fix4: stage_weight is the SSP-RK3 quadrature weight (1/6, 1/6, 2/3).
// The caller (advance()) must zero flux registers once before stage 1 and
// must NOT zero them between stages.  apply_flux_correction(dt) is called
// after stage 3.
//
// P2.4: leaf loop parallelised with OpenMP.
//   - Ghost fill runs BEFORE the parallel region: it touches shared neighbour
//     data and is not thread-safe to parallelise.
//   - compute_rhs and undo_cf_face_flux are fully independent per leaf: each
//     reads from its own block (+ already-filled ghost cells, read-only) and
//     writes only to its own rhs_blocks[li] slot.
//   - static thread_local prim caches in compute_rhs are safe: each thread
//     gets its own copy.
//   - accumulate_cf_fine_fluxes writes to shared flux registers; it runs
//     serially after the parallel loop.
//   - Leaves are sorted by Morton code before the loop so spatially adjacent
//     blocks land on the same CPU socket / L3 cache slice.
// =============================================================================
void tree_rhs(BlockTree& tree,
              std::vector<CellBlock>& rhs_blocks,
              bool periodic,
              double stage_weight) noexcept
{
    // ── 1. Ghost fill — serial (shared neighbour reads) ──────────────────────
    if (periodic)
        tree.fill_ghosts_periodic();
    else
        tree.fill_ghosts_wall();

    const auto& leaves = tree.leaf_indices();
    const int n_leaves = (int)leaves.size();
    assert((int)rhs_blocks.size() == n_leaves);

    // ── 2. Build Morton-sorted slot order for L3 cache locality ──────────────
    // order[oi] = slot index li such that leaves are visited in Morton order.
    // Must be a plain local (not thread_local): the parallel loop reads from it
    // on all threads; a thread_local copy would be uninitialized on workers.
    std::vector<int> order(n_leaves);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tree.nodes[leaves[a]].morton < tree.nodes[leaves[b]].morton;
    });

    // ── 3. Per-leaf RHS — parallel (fully independent per slot) ──────────────
#pragma omp parallel for schedule(dynamic,4)
    for (int oi = 0; oi < n_leaves; ++oi) {
        const int li       = order[oi];
        const int node_idx = leaves[li];
        compute_rhs(*tree.nodes[node_idx].block, rhs_blocks[li]);
        undo_cf_face_flux(tree, node_idx, rhs_blocks[li]);
    }

    // ── 4. Flux register accumulation — serial (shared CF register writes) ───
    accumulate_cf_fine_fluxes(tree, stage_weight);
}

// =============================================================================
// CFL time step
// =============================================================================
double tree_cfl_dt(const BlockTree& tree, double cfl) noexcept
{
    double dt = 1e300;
    for (auto li : tree.leaf_indices())
        dt = std::min(dt, tree.nodes[li].block->cfl_dt(cfl));
    return dt;
}
