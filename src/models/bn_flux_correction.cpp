// bn_flux_correction.cpp — Berger-Colella C/F flux correction for BNSolver.
//
// At coarse/fine boundaries the coarse block's RHS is computed using ghost
// cells filled with averages of fine-cell values (a coarse-ghost estimate).
// The corrections here replace that ghost-based flux with the area-averaged
// fine fluxes (Berger & Colella 1989).
//
// Only the 6 conservative BN variables Q[0..5] are corrected.  Q[6] = α₁ is
// non-conservative (upwind-split advection) and needs no flux register.
//
// Per-step protocol in BNSolver::advance():
//   bn_zero_regs()                               — zero registers
//   For each RK stage (sw = 1/6, 1/6, 2/3):
//     compute_rhs_bn(Qs, rhs, eos)               — standard block-local RHS
//     bn_accumulate_cf_correction_fluxes(...)    — accumulate (F_fine−F_coarse) × sw
//   bn_apply_flux_correction(tree, Q, regs, dt) — apply after RK3
//
// Crucially, the coarse block's RHS is NOT modified during the RK stages.
// The coarse block evolves with its ghost-based flux, keeping intermediate
// RK states physical.  The correction replaces the ghost flux post-RK3.

#include "models/bn_solver.hpp"
#include "mesh/block_tree.hpp"
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>

// ─── helpers ──────────────────────────────────────────────────────────────────

// Build (level << 32 | morton) → leaf-slot map for fast neighbour lookup.
static void build_lm_map(const BlockTree& tree,
                          std::unordered_map<uint64_t,int>& lm_map) {
    const auto& leaves = tree.leaf_indices();
    lm_map.clear();
    lm_map.reserve(leaves.size() * 2);
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        const auto& nd = tree.nodes[leaves[ii]];
        lm_map[((uint64_t)nd.level << 32) | nd.morton] = ii;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// bn_accumulate_cf_correction_fluxes
// ─────────────────────────────────────────────────────────────────────────────
// For each FINE leaf whose neighbour on face d is COARSER:
//   Accumulate sw * (F_fine − F_coarse_ghost) into regs[coarse_slot][d^1].
//
// F_fine:         HLLC flux at the fine block's C/F face.
// F_coarse_ghost: HLLC flux that compute_rhs_bn used on the coarse block at
//                 the same interface (computed from the coarse block's ghost
//                 and interior cells).
//
// The coarse RHS is NOT modified here; the difference is applied post-RK3 by
// bn_apply_flux_correction, which adds sign*(dt/h)*0.25*reg to the coarse
// boundary cells.  Each coarse face cell (jc,ic) receives 4 fine contributions
// (2×2 fine cells); the 0.25 factor in the apply step is the area ratio.
// ─────────────────────────────────────────────────────────────────────────────
void bn_accumulate_cf_correction_fluxes(
        const BlockTree& tree,
        const std::vector<BNCellBlock>& blocks,
        std::vector<std::array<std::vector<double>, NFACES>>& regs,
        double stage_weight,
        const BNEosParams& eos) noexcept {
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();

    std::unordered_map<uint64_t,int> lm_map;
    build_lm_map(tree, lm_map);

    static constexpr int face_axis[NFACES]  = {0, 0, 1, 1, 2, 2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};
    static constexpr int HALF = NB / 2;

    for (int ii = 0; ii < NL; ++ii) {
        const auto& nd  = tree.nodes[leaves[ii]];
        const BNCellBlock& blk = blocks[ii];  // fine block

        // Octant of this fine block relative to its parent.
        int oct = 0;
        if (nd.parent >= 0) {
            int fc = tree.nodes[nd.parent].first_child;
            oct = (fc >= 0) ? (leaves[ii] - fc) : 0;
        }
        const int o_ix = oct_ix(oct);
        const int o_iy = oct_iy(oct);
        const int o_iz = oct_iz(oct);

        for (int d = 0; d < NFACES; ++d) {
            int ni_node = nd.neighbours[d];
            if (ni_node < 0) continue;
            if (!tree.nodes[ni_node].is_leaf()) continue;
            if (tree.nodes[ni_node].level >= nd.level) continue;  // only C/F fine side

            // Coarse-neighbour slot.
            const auto& cn = tree.nodes[ni_node];
            auto it = lm_map.find(((uint64_t)cn.level << 32) | cn.morton);
            if (it == lm_map.end()) continue;
            int coarse_slot = it->second;
            const BNCellBlock& cblk = blocks[coarse_slot];

            const int axis   = face_axis[d];
            const int delta  = face_delta[d];
            const int bound  = (delta > 0) ? ihi() : ilo();

            // Coarse face is d^1: opposite delta, opposite bound.
            const int delta_c = -delta;
            const int bound_c = (delta_c > 0) ? ihi() : ilo();

            // Offset of this fine block in the coarse face plane.
            int off1, off2;
            if      (axis == 0) { off1 = o_iy; off2 = o_iz; }
            else if (axis == 1) { off1 = o_iz; off2 = o_ix; }
            else                { off1 = o_iy; off2 = o_ix; }

            // Register on the coarse block's face toward the fine block.
            auto& reg = regs[coarse_slot][d ^ 1];
            const int reg_sz = NVAR_BN_CONS * NB * NB;
            if ((int)reg.size() != reg_sz) reg.assign(reg_sz, 0.0);

            for (int b = ilo(); b <= ihi(); ++b)
            for (int a = ilo(); a <= ihi(); ++a) {
                // Fine face cell indices.
                int ci, cj, ck, gi, gj, gk;
                if (axis == 0) {
                    ci=bound; cj=a; ck=b; gi=bound+delta; gj=a; gk=b;
                } else if (axis == 1) {
                    ci=a; cj=bound; ck=b; gi=a; gj=bound+delta; gk=b;
                } else {
                    ci=a; cj=b; ck=bound; gi=a; gj=b; gk=bound+delta;
                }

                const int fi_int = cell_idx(ci, cj, ck);
                const int fi_gst = cell_idx(gi, gj, gk);

                auto prim_fine = [&](int flat) {
                    return bn_cons_to_prim(blk.Q[0][flat], blk.Q[1][flat],
                                          blk.Q[2][flat], blk.Q[3][flat],
                                          blk.Q[4][flat], blk.Q[5][flat],
                                          blk.Q[6][flat], eos);
                };

                const Prim2Phase pL_f = (delta > 0) ? prim_fine(fi_int) : prim_fine(fi_gst);
                const Prim2Phase pR_f = (delta > 0) ? prim_fine(fi_gst) : prim_fine(fi_int);
                const BNFaceFlux F_fine = hllc_bn_flux(pL_f, pR_f, axis, eos);

                // Coarse face position (jc, ic) in the register.
                const int a_local = a - ilo();
                const int b_local = b - ilo();
                int jc, ic;
                if (axis == 0) {
                    jc = off1 * HALF + a_local / 2;
                    ic = off2 * HALF + b_local / 2;
                } else {
                    jc = off1 * HALF + b_local / 2;
                    ic = off2 * HALF + a_local / 2;
                }

                // Coarse cell at (jc, ic) — interior and ghost for coarse face d^1.
                int ci_c, cj_c, ck_c, gi_c, gj_c, gk_c;
                if (axis == 0) {
                    ci_c=bound_c;   cj_c=ilo()+jc; ck_c=ilo()+ic;
                    gi_c=ci_c+delta_c; gj_c=cj_c;    gk_c=ck_c;
                } else if (axis == 1) {
                    ci_c=ilo()+ic; cj_c=bound_c;   ck_c=ilo()+jc;
                    gi_c=ci_c;     gj_c=cj_c+delta_c; gk_c=ck_c;
                } else {
                    ci_c=ilo()+ic; cj_c=ilo()+jc; ck_c=bound_c;
                    gi_c=ci_c;     gj_c=cj_c;     gk_c=ck_c+delta_c;
                }
                const int coarse_int = cell_idx(ci_c, cj_c, ck_c);
                const int coarse_gst = cell_idx(gi_c, gj_c, gk_c);

                auto prim_coarse = [&](int flat) {
                    return bn_cons_to_prim(cblk.Q[0][flat], cblk.Q[1][flat],
                                          cblk.Q[2][flat], cblk.Q[3][flat],
                                          cblk.Q[4][flat], cblk.Q[5][flat],
                                          cblk.Q[6][flat], eos);
                };

                // Coarse ghost-based flux (same L/R convention as compute_rhs_bn).
                const Prim2Phase pL_c = (delta_c > 0) ? prim_coarse(coarse_int) : prim_coarse(coarse_gst);
                const Prim2Phase pR_c = (delta_c > 0) ? prim_coarse(coarse_gst) : prim_coarse(coarse_int);
                const BNFaceFlux F_coarse = hllc_bn_flux(pL_c, pR_c, axis, eos);

                for (int v = 0; v < NVAR_BN_CONS; ++v)
                    reg[v*NB*NB + jc*NB + ic] +=
                        stage_weight * (F_fine.F[v] - F_coarse.F[v]);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// bn_apply_flux_correction
// ─────────────────────────────────────────────────────────────────────────────
// For each COARSE leaf with accumulated register on face d:
//   Apply  Q[coarse_boundary_cell][v] += sign * (dt/h) * 0.25 * reg[jc,ic]
//
// 0.25 = area ratio (4 fine faces per coarse face cell).
// Sign: -face (flux enters) → ADD; +face (flux leaves) → SUBTRACT.
// ─────────────────────────────────────────────────────────────────────────────
void bn_apply_flux_correction(
        const BlockTree& tree,
        std::vector<BNCellBlock>& Q,
        const std::vector<std::array<std::vector<double>, NFACES>>& regs,
        double dt) noexcept {
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();

    static constexpr int face_axis[NFACES] = {0, 0, 1, 1, 2, 2};
    static constexpr int face_side[NFACES] = {0, 1, 0, 1, 0, 1};

    for (int ii = 0; ii < NL; ++ii) {
        const auto& nd  = tree.nodes[leaves[ii]];
        const double h_c = Q[ii].h;

        for (int d = 0; d < NFACES; ++d) {
            const auto& reg = regs[ii][d];
            if (reg.empty()) continue;

            int ni_node = nd.neighbours[d];
            if (ni_node < 0 || !tree.nodes[ni_node].is_leaf()) continue;
            if (tree.nodes[ni_node].level <= nd.level) continue;  // coarse leaf only

            const int axis = face_axis[d];
            const int side = face_side[d];
            const double sign = (side == 1) ? -1.0 : +1.0;
            const int    g    = (side == 0) ? ilo() : ihi();
            const double fac  = sign * (dt / h_c) * 0.25;

            for (int v = 0; v < NVAR_BN_CONS; ++v)
            for (int jc = 0; jc < NB; ++jc)
            for (int ic = 0; ic < NB; ++ic) {
                double corr = fac * reg[v*NB*NB + jc*NB + ic];
                int ci, cj, ck;
                if      (axis == 0) { ci = g;        cj = ilo()+jc; ck = ilo()+ic; }
                else if (axis == 1) { ci = ilo()+ic; cj = g;        ck = ilo()+jc; }
                else                { ci = ilo()+ic; cj = ilo()+jc; ck = g;        }
                Q[ii].Q[v][cell_idx(ci,cj,ck)] += corr;
            }
        }
    }
}
