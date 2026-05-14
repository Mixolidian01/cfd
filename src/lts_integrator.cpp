// lts_integrator.cpp — LtsIntegrator: Berger-Oliger local time stepping (P4.1).
//
// Extracted from src/lts_advance.cpp (NSSolver member functions) as part of
// P10-A2.  step() corresponds to NSSolver::advance_lts(); rk3_level() to
// NSSolver::lts_rk3_level().
//
// step() returns dt_c WITHOUT updating solver.t / solver.step / solver.last_dt_;
// those are updated by NSSolver::advance() after step() returns, consistent
// with CpuRk3Integrator::step().
//
// Reference: Berger & Colella (1989), §3; Berger & Oliger (1984).
#include "../include/lts_integrator.hpp"
#include "../include/operators.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

// =============================================================================
// Level-filtered copy helpers — only touch leaves at `level`.
// =============================================================================
void LtsIntegrator::copy_tree_to_stage_level(std::vector<CellBlock>& stage, int level) {
    const auto& leaves = solver.tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        if (solver.tree.nodes[leaves[ii]].level != level) continue;
        stage[ii] = *solver.tree.nodes[leaves[ii]].block;
    }
}

void LtsIntegrator::copy_stage_to_tree_level(const std::vector<CellBlock>& stage, int level) {
    const auto& leaves = solver.tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        if (solver.tree.nodes[leaves[ii]].level != level) continue;
        *solver.tree.nodes[leaves[ii]].block = stage[ii];
    }
}

// =============================================================================
// P4.1 — rk3_level: one full SSP-RK3 step for leaves at `level`.
//
// sub_weight controls the Berger-Colella flux accumulation weight passed to
// tree_rhs (which multiplies stage_weight = sub_weight × {1/6, 1/6, 2/3}).
// For fine sub-steps: sub_weight = 1/r so that after r sub-steps the total
// weight equals 1.  For the coarse step: sub_weight = 1.0.
// =============================================================================
void LtsIntegrator::rk3_level(BlockTree& tree, int level, double dt,
                               double sub_weight, bool coarse_mode) {
    bool periodic = bc_is_periodic(solver.cfg.bc_variant);
    bool open_bc  = bc_is_open(solver.cfg.bc_variant);
    const DucrosConfig ducros{ solver.cfg.ducros_p_threshold,
                                solver.cfg.ducros_blend_width };
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();

    copy_tree_to_stage_level(solver.Qn_, level);

    // Stage 1: Q^(1) = Q^n + dt * L(Q^n)
    tree_rhs(tree, solver.rhs_, periodic, sub_weight / 6.0, level, coarse_mode, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        for (int v = 0; v < NVAR; ++v)
        for (int t = 0; t < CellBlock::NTILE; ++t) {
            double*       qs = solver.Qs_[ii].Q[v].tile_ptr(t);
            const double* qn = solver.Qn_[ii].Q[v].tile_ptr(t);
            const double* r  = solver.rhs_[ii].Q[v].tile_ptr(t);
            #pragma omp simd simdlen(8)
            for (int lane = 0; lane < CellBlock::W; ++lane)
                qs[lane] = qn[lane] + dt * r[lane];
        }
    }
    copy_stage_to_tree_level(solver.Qs_, level);

    // Stage 2: Q^(2) = 3/4*Q^n + 1/4*(Q^(1) + dt*L(Q^(1)))
    tree_rhs(tree, solver.rhs_, periodic, sub_weight / 6.0, level, coarse_mode, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        for (int v = 0; v < NVAR; ++v)
        for (int t = 0; t < CellBlock::NTILE; ++t) {
            double*       qs = solver.Qs_[ii].Q[v].tile_ptr(t);
            const double* qn = solver.Qn_[ii].Q[v].tile_ptr(t);
            const double* r  = solver.rhs_[ii].Q[v].tile_ptr(t);
            #pragma omp simd simdlen(8)
            for (int lane = 0; lane < CellBlock::W; ++lane)
                qs[lane] = (3.0/4.0)*qn[lane] + (1.0/4.0)*(qs[lane] + dt*r[lane]);
        }
    }
    copy_stage_to_tree_level(solver.Qs_, level);

    // Stage 3: Q^{n+1} = 1/3*Q^n + 2/3*(Q^(2) + dt*L(Q^(2)))
    tree_rhs(tree, solver.rhs_, periodic, sub_weight * (2.0/3.0), level, coarse_mode, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].level != level) continue;
        for (int v = 0; v < NVAR; ++v)
        for (int t = 0; t < CellBlock::NTILE; ++t) {
            double*       qs = solver.Qs_[ii].Q[v].tile_ptr(t);
            const double* qn = solver.Qn_[ii].Q[v].tile_ptr(t);
            const double* r  = solver.rhs_[ii].Q[v].tile_ptr(t);
            #pragma omp simd simdlen(8)
            for (int lane = 0; lane < CellBlock::W; ++lane)
                qs[lane] = (1.0/3.0)*qn[lane] + (2.0/3.0)*(qs[lane] + dt*r[lane]);
        }
    }
    copy_stage_to_tree_level(solver.Qs_, level);
}

// =============================================================================
// P4.1 — step: Berger-Oliger LTS entry point.
//
// Returns dt_c.  advance() updates solver.t / solver.step / solver.last_dt_.
//
// Regrid runs here (not in advance()) because the LTS path is an early exit
// in advance() that bypasses advance()'s own regrid block.
// =============================================================================
double LtsIntegrator::step(BlockTree& tree, double cfl) {
    if (solver.cfg.regrid_interval > 0 && solver.step > 0 &&
        solver.step % solver.cfg.regrid_interval == 0)
        solver.regrid();

    const int L_min = tree.min_leaf_level();
    const int L_max = tree.max_leaf_level();
    assert(L_max > L_min);  // advance() dispatches here only when max_leaf_level > 0

    const int r = solver.cfg.lts_ratio;

    double dt_f_cfl = level_cfl_dt(tree, L_max, cfl);
    double dt_c_cfl = level_cfl_dt(tree, L_min, cfl);
    double dt_c = std::min(dt_c_cfl, static_cast<double>(r) * dt_f_cfl);
    double dt_f = dt_c / r;

    tree.zero_flux_registers();

    for (int k = 0; k < r; ++k)
        rk3_level(tree, L_max, dt_f, 1.0 / r, /*coarse_mode=*/true);

    rk3_level(tree, L_min, dt_c, 1.0, /*coarse_mode=*/true);

    tree.apply_flux_correction(dt_c);

    return dt_c;
}
