// P4.4 — Inviscid RHS for Allaire 5-equation two-phase model.
//
// Implements the face-centered convective RHS for BNCellBlock.
// The loop structure mirrors operators.cpp §convective_rhs_impl:
//   for each axis, iterate over all faces (including ghost↔interior faces)
//   spanning [ilo()-1 .. ihi()] in the normal direction and
//   [ilo() .. ihi()] in tangential directions.
//   rhs[L] -= (1/h) * F[v],  rhs[R] += (1/h) * F[v]   for v=0..5 (conservative).
//
// α₁ non-conservative update (Allaire 2002 §3, eq. 3.10):
//   For face f between cells L and R, contact speed sStar_f:
//     rhs[L][6] += (1/h) * sStar_f * (α₁_L − α₁^upwind_f)
//     rhs[R][6] += (1/h) * sStar_f * (α₁^upwind_f − α₁_R)
//   where α₁^upwind = α₁_L if sStar_f ≥ 0, else α₁_R.
//
// Ghost cells must be filled before this function is called.

#include "models/bn_model.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

// ── Prim cache ─────────────────────────────────────────────────────────────────
static void fill_prim_cache_bn(const BNCellBlock& blk, Prim2Phase* pc,
                                const BNEosParams& eos) noexcept
{
    for (int flat = 0; flat < NCELL; ++flat) {
        pc[flat] = bn_cons_to_prim(
            blk.Q[0][flat], blk.Q[1][flat],
            blk.Q[2][flat], blk.Q[3][flat], blk.Q[4][flat],
            blk.Q[5][flat], blk.Q[6][flat],
            eos);
    }
}

// ── Convective RHS (face-centred, all 3 axes) ─────────────────────────────────
static void bn_convective_rhs_impl(const BNCellBlock& blk,
                                   const Prim2Phase* pc,
                                   BNCellBlock& rhs,
                                   const BNEosParams& eos) noexcept
{
    const double ih = 1.0 / blk.h;

    // ── X-direction ──────────────────────────────────────────────────────────
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo()-1; i <= ihi(); ++i) {
        const int fL = cell_idx(i,   j, k);
        const int fR = cell_idx(i+1, j, k);
        const BNFaceFlux res = hllc_bn_flux(pc[fL], pc[fR], 0, eos);

        if (i >= ilo()) {
            for (int v = 0; v < 6; ++v) rhs.Q[v][fL] -= ih * res.F[v];
            // α₁ non-conservative: right face of cell L
            const double a1L   = blk.Q[6][fL];
            const double a1up  = (res.s_star >= 0.0) ? a1L : blk.Q[6][fR];
            rhs.Q[6][fL] += ih * res.s_star * (a1L - a1up);
        }
        if (i+1 <= ihi()) {
            for (int v = 0; v < 6; ++v) rhs.Q[v][fR] += ih * res.F[v];
            // α₁ non-conservative: left face of cell R
            const double a1R  = blk.Q[6][fR];
            const double a1up = (res.s_star >= 0.0) ? blk.Q[6][fL] : a1R;
            rhs.Q[6][fR] += ih * res.s_star * (a1up - a1R);
        }
    }

    // ── Y-direction ──────────────────────────────────────────────────────────
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo()-1; j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int fL = cell_idx(i, j,   k);
        const int fR = cell_idx(i, j+1, k);
        const BNFaceFlux res = hllc_bn_flux(pc[fL], pc[fR], 1, eos);

        if (j >= ilo()) {
            for (int v = 0; v < 6; ++v) rhs.Q[v][fL] -= ih * res.F[v];
            const double a1L  = blk.Q[6][fL];
            const double a1up = (res.s_star >= 0.0) ? a1L : blk.Q[6][fR];
            rhs.Q[6][fL] += ih * res.s_star * (a1L - a1up);
        }
        if (j+1 <= ihi()) {
            for (int v = 0; v < 6; ++v) rhs.Q[v][fR] += ih * res.F[v];
            const double a1R  = blk.Q[6][fR];
            const double a1up = (res.s_star >= 0.0) ? blk.Q[6][fL] : a1R;
            rhs.Q[6][fR] += ih * res.s_star * (a1up - a1R);
        }
    }

    // ── Z-direction ──────────────────────────────────────────────────────────
    for (int k = ilo()-1; k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int fL = cell_idx(i, j, k  );
        const int fR = cell_idx(i, j, k+1);
        const BNFaceFlux res = hllc_bn_flux(pc[fL], pc[fR], 2, eos);

        if (k >= ilo()) {
            for (int v = 0; v < 6; ++v) rhs.Q[v][fL] -= ih * res.F[v];
            const double a1L  = blk.Q[6][fL];
            const double a1up = (res.s_star >= 0.0) ? a1L : blk.Q[6][fR];
            rhs.Q[6][fL] += ih * res.s_star * (a1L - a1up);
        }
        if (k+1 <= ihi()) {
            for (int v = 0; v < 6; ++v) rhs.Q[v][fR] += ih * res.F[v];
            const double a1R  = blk.Q[6][fR];
            const double a1up = (res.s_star >= 0.0) ? blk.Q[6][fL] : a1R;
            rhs.Q[6][fR] += ih * res.s_star * (a1up - a1R);
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────
void compute_rhs_bn(const BNCellBlock& blk, BNCellBlock& rhs_blk,
                    const BNEosParams& eos) noexcept
{
    // Zero RHS interior (ghost cells remain 0 from BNCellBlock default).
    for (int v = 0; v < NVAR_BN; ++v)
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        rhs_blk.Q[v][cell_idx(i,j,k)] = 0.0;

    Prim2Phase pc[NCELL];
    fill_prim_cache_bn(blk, pc, eos);
    bn_convective_rhs_impl(blk, pc, rhs_blk, eos);
}

// ── CFL time step ─────────────────────────────────────────────────────────────
double bn_cfl_dt(const BNCellBlock& blk, double cfl, const BNEosParams& eos) noexcept
{
    double max_speed = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int f = cell_idx(i, j, k);
        const Prim2Phase q = bn_cons_to_prim(
            blk.Q[0][f], blk.Q[1][f],
            blk.Q[2][f], blk.Q[3][f], blk.Q[4][f],
            blk.Q[5][f], blk.Q[6][f],
            eos);
        const double sp = std::abs(q.u) + std::abs(q.v) + std::abs(q.w) + q.c_mix;
        if (sp > max_speed) max_speed = sp;
    }
    return (max_speed > 0.0) ? cfl * blk.h / max_speed : 1.0e10;
}
