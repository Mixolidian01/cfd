// lts_advance.cpp — NSSolver Local Time Stepping (Berger-Oliger)
//
// Contains: NSSolver::copy_tree_to_stage_level
//           NSSolver::copy_stage_to_tree_level
//           NSSolver::lts_rk3_level
//           NSSolver::advance_lts
//
// These are member functions of NSSolver; declarations remain in ns_solver.hpp.
// Extracted from ns_solver.cpp as part of R9-B maintainability refactor.
//
// Reference: Berger & Colella (1989), §3; Berger & Oliger (1984).
#include "../include/sgs.hpp"
#include "../include/amr_operators.hpp"
#include "../include/ns_solver.hpp"
#include "../include/operators.hpp"
#include "../include/linalg.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

// =============================================================================
// Level-filtered copy helpers for LTS (only touch leaves at `level`).
// =============================================================================
void NSSolver::copy_tree_to_stage_level(std::vector<CellBlock>& stage, int level) {
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        stage[ii] = *tree.nodes[leaves[ii]].block;
    }
}

void NSSolver::copy_stage_to_tree_level(const std::vector<CellBlock>& stage, int level) {
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        *tree.nodes[leaves[ii]].block = stage[ii];
    }
}

// =============================================================================
// P4.1 — lts_rk3_level
// Runs one full SSP-RK3 step for leaves at `level`.
//
// sub_weight controls the Berger-Colella flux accumulation weight passed to
// tree_rhs (which multiplies stage_weight = sub_weight × {1/6, 1/6, 2/3}).
// For fine sub-steps: sub_weight = 1/r so that after r sub-steps the total
// weight equals 1, giving the correct time-averaged fine flux in the register.
// For the coarse step: sub_weight = 1.0 (standard SSP-RK3 weights).
//
// Ghost fill inside tree_rhs is always global, so C/F ghosts are filled from
// the CURRENT coarse solution (frozen-coarse approximation for fine sub-steps).
// =============================================================================
void NSSolver::lts_rk3_level(int level, double dt, double sub_weight, bool coarse_mode) {
    bool periodic = bc_is_periodic(cfg.bc_variant);
    bool open_bc  = bc_is_open(cfg.bc_variant);
    const DucrosConfig ducros{ cfg.ducros_p_threshold, cfg.ducros_blend_width };
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();

    // Save Q^n for leaves at this level into Qn_ slots.
    copy_tree_to_stage_level(Qn_, level);

    // Stage 1: Q^(1) = Q^n + dt * L(Q^n)
    tree_rhs(tree, rhs_, periodic, sub_weight / 6.0, level, coarse_mode, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        for (int v = 0; v < NVAR; ++v)
        for (int t = 0; t < CellBlock::NTILE; ++t) {
            double*       qs = Qs_[ii].Q[v].tile_ptr(t);
            const double* qn = Qn_[ii].Q[v].tile_ptr(t);
            const double* r  = rhs_[ii].Q[v].tile_ptr(t);
            #pragma omp simd simdlen(8)
            for (int lane = 0; lane < CellBlock::W; ++lane)
                qs[lane] = qn[lane] + dt * r[lane];
        }
    }
    copy_stage_to_tree_level(Qs_, level);

    // Stage 2: Q^(2) = 3/4*Q^n + 1/4*(Q^(1) + dt*L(Q^(1)))
    tree_rhs(tree, rhs_, periodic, sub_weight / 6.0, level, coarse_mode, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        for (int v = 0; v < NVAR; ++v)
        for (int t = 0; t < CellBlock::NTILE; ++t) {
            double*       qs = Qs_[ii].Q[v].tile_ptr(t);
            const double* qn = Qn_[ii].Q[v].tile_ptr(t);
            const double* r  = rhs_[ii].Q[v].tile_ptr(t);
            #pragma omp simd simdlen(8)
            for (int lane = 0; lane < CellBlock::W; ++lane)
                qs[lane] = (3.0/4.0)*qn[lane] + (1.0/4.0)*(qs[lane] + dt*r[lane]);
        }
    }
    copy_stage_to_tree_level(Qs_, level);

    // Stage 3: Q^{n+1} = 1/3*Q^n + 2/3*(Q^(2) + dt*L(Q^(2)))
    tree_rhs(tree, rhs_, periodic, sub_weight * (2.0/3.0), level, coarse_mode, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        for (int v = 0; v < NVAR; ++v)
        for (int t = 0; t < CellBlock::NTILE; ++t) {
            double*       qs = Qs_[ii].Q[v].tile_ptr(t);
            const double* qn = Qn_[ii].Q[v].tile_ptr(t);
            const double* r  = rhs_[ii].Q[v].tile_ptr(t);
            #pragma omp simd simdlen(8)
            for (int lane = 0; lane < CellBlock::W; ++lane)
                qs[lane] = (1.0/3.0)*qn[lane] + (2.0/3.0)*(qs[lane] + dt*r[lane]);
        }
    }
    copy_stage_to_tree_level(Qs_, level);
}

// =============================================================================
// P4.1 — advance_lts: Berger-Oliger local time stepping
//
// Algorithm for a 2-level tree (levels L_min=0, L_max):
//   1. Zero flux registers (accumulate fine flux over all sub-steps).
//   2. Fine level: r sub-steps of dt_f = dt_c/r with sub_weight = 1/r.
//      Ghost fill uses coarse cells at t^n (frozen-coarse approximation).
//   3. Coarse level: 1 step of dt_c with sub_weight = 1.
//      undo_cf_face_flux removes the coarse C/F flux from coarse RHS;
//      it is replaced by the time-averaged fine flux accumulated in step 2.
//   4. apply_flux_correction(dt_c): adds fine-level correction to coarse cells.
//
// Flux conservation proof:
//   After r fine sub-steps each with stage_weight = (1/r)*{1/6,1/6,2/3}:
//     reg = area_ratio * (1/r) * r * <F_fine_avg> = area_ratio * <F_fine_avg>
//   apply_flux_correction(dt_c) applies sign*(dt_c/h_c)*reg = correct correction.
//
// Reference: Berger & Colella (1989), §3; Berger & Oliger (1984).
// =============================================================================
double NSSolver::advance_lts() {
    // Rule 006: regrid on Q^n BEFORE RK3.
    if (cfg.regrid_interval > 0 && step > 0 && step % cfg.regrid_interval == 0)
        regrid();  // regrid() calls alloc_scratch() if topology changes.

    // Determine actual coarsest/finest levels that hold leaves.
    const int L_min = tree.min_leaf_level();
    const int L_max = tree.max_leaf_level();
    // Sanity guard: should not be dispatched when single-level.
    if (L_max <= L_min) return advance();

    const int r = cfg.lts_ratio;

    // CFL time steps: fine level drives dt_f; coarse dt_c = r * dt_f.
    // Also bound dt_c by the coarse CFL to prevent instability if the coarse
    // level has faster waves than fine (e.g., after a refinement change).
    double dt_f_cfl = level_cfl_dt(tree, L_max, cfg.cfl);
    double dt_c_cfl = level_cfl_dt(tree, L_min, cfg.cfl);
    double dt_c = std::min(dt_c_cfl, static_cast<double>(r) * dt_f_cfl);
    double dt_f = dt_c / r;

    // Zero flux registers once; fine sub-steps will accumulate into them.
    tree.zero_flux_registers();

    // ── Step 1: r fine sub-steps ─────────────────────────────────────────────
    // coarse_mode=true: C/F coarse ghosts use zero-gradient for BOTH fine and
    // coarse steps.  This makes viscous face flux exactly zero at all C/F faces,
    // so the convective-only flux register achieves exact energy conservation.
    for (int k = 0; k < r; ++k)
        lts_rk3_level(L_max, dt_f, 1.0 / r, /*coarse_mode=*/true);

    // ── Step 2: coarse advance ───────────────────────────────────────────────
    // coarse_mode=true: C/F coarse ghosts use zero-gradient (ghost=interior).
    // This makes viscous flux exactly zero at C/F faces, so the Berger-Colella
    // reflux correction restores full conservation for both mass and energy.
    lts_rk3_level(L_min, dt_c, 1.0, /*coarse_mode=*/true);

    // ── Step 3: Berger-Colella correction ────────────────────────────────────
    tree.apply_flux_correction(dt_c);

    last_dt_ = dt_c;
    t    += dt_c;
    step += 1;

    return dt_c;
}
