// =============================================================================
// adjoint_rhs.cpp — D10: block-level adjoint of convective RHS
// =============================================================================
// Implements L*(Q)·λ where L is the convective RHS (accumulate_face loop).
//
// Strategy: frozen-weight TENO7-A + frozen-lam HLLC-ES.
//   For interior faces (not touching a ghost layer):
//     - Run teno7_recon_fwd<DIR> to capture frozen reconstruction state
//     - Compute frozen lam = max(|unL|+cL, |unR|+cR) from reconstructed states
//     - Run adjoint_hllces_flux<DIR> then teno7_recon_adj<DIR>
//   For boundary faces (one or both cells in ghost zone, has_nbr=0):
//     - PCM (pL=pc[Li], pR=pc[Ri]) + HLLC-ES with frozen lam
//
// Note on KEP branch: The forward accumulate_face uses KEP for smooth flow
// (theta < 1e-8). The dot-product test (t38) is designed with a non-uniform
// IC (Sod-like discontinuity) where theta >= 1e-8 at interior faces, so the
// forward always takes the TENO7+HLLC-ES path. This adjoint does NOT implement
// the KEP adjoint — callers must ensure the IC causes theta >= kep_threshold
// everywhere for adjoint–forward consistency.
// =============================================================================

#include "schemes/adjoint_rhs.hpp"
#include "physics/adjoint_teno7.hpp"   // teno7_recon_fwd, teno7_recon_adj
#include "physics/adjoint_hllc.hpp"    // adjoint_hllces_flux, acc_adj_cons_to_prim

#include <cmath>
#include <algorithm>
#include <cstring>

// ── adjoint_accumulate_face<DIR> ─────────────────────────────────────────────
// Adjoint of one face contribution in accumulate_face<DIR>.
//
// Forward:
//   rhs.axis_view<DIR>(v)(n,   a, b) -= ih * F[v]   if n   >= ilo()
//   rhs.axis_view<DIR>(v)(n+1, a, b) += ih * F[v]   if n+1 <= ihi()
//
// Adjoint (seed on rhs → accumulate into prim adjoint l_pc):
//   l_F[v] = -ih * lambda_rhs.axis_view<DIR>(v)(n,   a, b)   [if Li is interior]
//            +ih * lambda_rhs.axis_view<DIR>(v)(n+1, a, b)   [if Ri is interior]
//
// Then propagate l_F → l_pc via the appropriate path:
//   Interior face: TENO7-A reconstruction + HLLC-ES
//   Boundary face: PCM + HLLC-ES
//
// l_pc[flat][NVAR]: prim adjoint accumulator (in-place accumulate).
template<Axis DIR>
static void adjoint_accumulate_face(
        const Prim* pc,
        const CellBlock& lambda_rhs,
        double l_pc[][NVAR],
        double ih,
        int n, int a, int b,
        uint8_t /*has_nbr*/) noexcept
{
    const int Li = cell_idx_axis<DIR>(n,   a, b);
    const int Ri = cell_idx_axis<DIR>(n+1, a, b);

    const bool is_left_bnd  = (n   < ilo());
    const bool is_right_bnd = (n+1 > ihi());
    const bool is_bnd = is_left_bnd || is_right_bnd;

    // ── Assemble flux seed l_F from lambda_rhs ────────────────────────────────
    // Forward: rhs[Li] -= ih*F,  rhs[Ri] += ih*F
    // Adjoint: l_F = -ih * lrhs[Li]   (if Li is in interior)
    //              + ih * lrhs[Ri]    (if Ri is in interior)
    double l_F[NVAR] = {};
    if (n >= ilo()) {   // Li is an interior cell
        for (int v = 0; v < NVAR; ++v)
            l_F[v] -= ih * lambda_rhs.axis_view<DIR>(v)(n,   a, b);
    }
    if (n+1 <= ihi()) { // Ri is an interior cell
        for (int v = 0; v < NVAR; ++v)
            l_F[v] += ih * lambda_rhs.axis_view<DIR>(v)(n+1, a, b);
    }

    // ── Propagate l_F → l_pc ─────────────────────────────────────────────────
    if (!is_bnd) {
        // Interior face: TENO7-A + HLLC-ES (frozen weights, frozen lam)
        int xi, yi, zi;
        face_to_ijk<DIR>(n, a, b, xi, yi, zi);

        Prim qL, qR;
        Teno7CharFwd cf;
        teno7_recon_fwd<DIR>(pc, xi, yi, zi, qL, qR, cf);

        // Compute frozen spectral radius from reconstructed states
        constexpr int axis = static_cast<int>(DIR);
        const double unL = (axis==0)?qL.u:(axis==1)?qL.v:qL.w;
        const double unR = (axis==0)?qR.u:(axis==1)?qR.v:qR.w;
        const double lam_frozen = std::max(std::abs(unL)+qL.c,
                                           std::abs(unR)+qR.c);

        double l_pL[NVAR] = {};
        double l_pR[NVAR] = {};
        adjoint_hllces_flux<DIR>(qL, qR, l_F, lam_frozen, l_pL, l_pR);

        // Convert prim adjoint → cons adjoint seeds for TENO7 adjoint
        double l_qL_cons[NVAR] = {};
        double l_qR_cons[NVAR] = {};
        acc_adj_cons_to_prim(qL, l_pL, l_qL_cons);
        acc_adj_cons_to_prim(qR, l_pR, l_qR_cons);

        teno7_recon_adj<DIR>(pc, xi, yi, zi, cf, l_qL_cons, l_qR_cons, l_pc);
    } else {
        // Boundary face: PCM (no reconstruction) + HLLC-ES with frozen lam
        const Prim& pL = pc[Li];
        const Prim& pR = pc[Ri];

        constexpr int axis = static_cast<int>(DIR);
        const double unL = (axis==0)?pL.u:(axis==1)?pL.v:pL.w;
        const double unR = (axis==0)?pR.u:(axis==1)?pR.v:pR.w;
        const double lam_frozen = std::max(std::abs(unL)+pL.c,
                                           std::abs(unR)+pR.c);

        double l_pL[NVAR] = {};
        double l_pR[NVAR] = {};
        adjoint_hllces_flux<DIR>(pL, pR, l_F, lam_frozen, l_pL, l_pR);

        for (int v = 0; v < NVAR; ++v) l_pc[Li][v] += l_pL[v];
        for (int v = 0; v < NVAR; ++v) l_pc[Ri][v] += l_pR[v];
    }
}

// =============================================================================
// adjoint_rhs — public entry point
// =============================================================================
void adjoint_rhs(const CellBlock& Q_blk,
                 const CellBlock& lambda_rhs,
                 CellBlock&       lambda_Q,
                 uint8_t          has_nbr) noexcept
{
    // ── Step 1: Build flat primitive array ───────────────────────────────────
    Prim pc[NCELL];
    for (int k = 0; k < NB2; ++k) {
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i)
            pc[cell_idx(i,j,k)] = Q_blk.prim(i,j,k);
    }

    const double ih = 1.0 / Q_blk.h;

    // ── Step 2: Prim adjoint accumulator (zero-init) ─────────────────────────
    // NCELL=1728, NVAR=5 → 1728*5*8 = 69120 bytes (~67 KB).
    // Acceptable for a CPU test/adjoint function (not called in the inner loop).
    double l_pc[NCELL][NVAR] = {};

    // ── Step 3: Adjoint of face loops (same bounds as convective_rhs_impl) ───

    // X: n=i (normal), a=j, b=k
    for (int k = ilo(); k <= ihi(); ++k) {
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo()-1; i <= ihi(); ++i)
            adjoint_accumulate_face<Axis::X>(pc, lambda_rhs, l_pc, ih, i, j, k, has_nbr);
    }
    // Y: n=j (normal), a=i, b=k
    for (int k = ilo(); k <= ihi(); ++k) {
        for (int j = ilo()-1; j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i)
            adjoint_accumulate_face<Axis::Y>(pc, lambda_rhs, l_pc, ih, j, i, k, has_nbr);
    }
    // Z: n=k (normal), a=i, b=j
    for (int k = ilo()-1; k <= ihi(); ++k) {
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i)
            adjoint_accumulate_face<Axis::Z>(pc, lambda_rhs, l_pc, ih, k, i, j, has_nbr);
    }

    // ── Step 4: Convert prim adjoint → cons adjoint, accumulate into lambda_Q ─
    // Only accumulate for interior cells (ghost cells are not independent vars).
    for (int k = ilo(); k <= ihi(); ++k) {
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i, j, k);
            double lq[NVAR] = {};
            acc_adj_cons_to_prim(pc[f], l_pc[f], lq);
            for (int v = 0; v < NVAR; ++v)
                lambda_Q.Q[v][f] += lq[v];
        }
    }
}
