// P10-A2: CpuRk3Integrator — CPU flat-tree SSP-RK3 implementation.
//
// Extracted from NSSolver::advance() so that the time-integration kernel
// lives in its own translation unit, separate from diagnostics and dispatch.

#include "solver/cpu_rk3.hpp"
#include "schemes/operators.hpp"
#include "mpi/mpi_comm.hpp"
#include <algorithm>
#include <cmath>

// P11.3: Zhang-Shu positivity floor — ρ ≥ ε, p ≥ ε (interior cells only).
static constexpr double EPS_POS = 1e-12;

static void apply_positivity_floor(std::vector<CellBlock>& stage) noexcept {
#pragma omp parallel for
    for (int bi = 0; bi < (int)stage.size(); ++bi) {
        auto& blk = stage[bi];
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i, j, k);
            double rho_v = blk.rho(f);
            if (rho_v < EPS_POS) { blk.rho(f) = rho_v = EPS_POS; }
            const double ke = 0.5 * (blk.rhou(f)*blk.rhou(f)
                                   + blk.rhov(f)*blk.rhov(f)
                                   + blk.rhow(f)*blk.rhow(f)) / rho_v;
            if ((GAMMA - 1.0) * (blk.E(f) - ke) < EPS_POS)
                blk.E(f) = ke + EPS_POS / (GAMMA - 1.0);
        }
    }
}

double CpuRk3Integrator::step(BlockTree& tree, double cfl) {
    const SolverConfig& cfg = solver.cfg;

    const bool periodic = bc_is_periodic(cfg.bc.variant);
    const bool open_bc  = bc_is_open(cfg.bc.variant);
    const DucrosConfig ducros{ cfg.numerics.ducros_p_threshold, cfg.numerics.ducros_blend_width };

    // P14.1c: activate stiffened-gas mixture EOS when ACDI is on and fluids differ.
    {
        const bool sg = cfg.acdi.use_acdi &&
                        (cfg.acdi.gamma_a != cfg.acdi.gamma_b ||
                         cfg.acdi.p_inf_a != 0.0 || cfg.acdi.p_inf_b != 0.0);
        CellBlock::set_sg_eos(sg, cfg.acdi.gamma_a, cfg.acdi.gamma_b, cfg.acdi.p_inf_a, cfg.acdi.p_inf_b);
    }

    double dt = tree_cfl_dt(tree, cfl);
    dt = mpi_allreduce_min(dt, solver.mpi_);

    solver.save_Qn();

    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();

    tree.zero_flux_registers();

    const bool use_sat  = (cfg.numerics.sat_tau > 0.0) && (tree.max_leaf_level() > 0);
    const bool use_acdi = cfg.acdi.use_acdi;

    // P14.1: phi SSP-RK3 helper — fills phi rhs and applies one stage update.
    // alpha < 0 → stage 1 (ps = pn + dt*L).
    // alpha ≥ 0 → stages 2/3: ps = α*pn + (1-α)*(ps_prev + dt*L).
    auto phi_stage = [&](double alpha) {
        if (!use_acdi) return;
        for (int ii = 0; ii < NL; ++ii) {
            if (!tree.nodes[leaves[ii]].has_block()) continue;
            const CellBlock& blk = *tree.nodes[leaves[ii]].block;
            std::fill(solver.rhs_[ii].phi_data_, solver.rhs_[ii].phi_data_ + NCELL, 0.0);
            phi_rhs(blk, solver.rhs_[ii]);
            if (cfg.acdi.acdi_ceps > 0.0)
                phi_compression_rhs(blk, solver.rhs_[ii], cfg.acdi.acdi_ceps);
            const double* pn = solver.Qn_[ii].phi_data_;
            double*       ps = solver.Qs_[ii].phi_data_;
            const double* pr = solver.rhs_[ii].phi_data_;
            if (alpha < 0.0) {
                for (int f = 0; f < NCELL; ++f)
                    ps[f] = pn[f] + dt * pr[f];
            } else {
                const double beta = 1.0 - alpha;
                for (int f = 0; f < NCELL; ++f)
                    ps[f] = alpha * pn[f] + beta * (ps[f] + dt * pr[f]);
            }
        }
    };

    // Stage 1: Q^(1) = Q^n + dt*L(Q^n)   [RK weight 1/6]
    mpi_exchange_halos(tree, solver.mpi_);
    tree_rhs(tree, solver.rhs_, periodic, 1.0/6.0, -1, false, open_bc, ducros);
    if (use_sat) tree_sat_penalty(tree, solver.rhs_, cfg.numerics.sat_tau);
#pragma omp parallel for
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int t  = 0; t  < CellBlock::NTILE; ++t) {
        double*       qs = solver.Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = solver.Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = solver.rhs_[ii].Q[v].tile_ptr(t);
        #pragma omp simd simdlen(8)
        for (int lane = 0; lane < CellBlock::W; ++lane)
            qs[lane] = qn[lane] + dt * r[lane];
    }
    phi_stage(-1.0);
    apply_positivity_floor(solver.Qs_);
    solver.copy_stage_to_tree(solver.Qs_);

    // Stage 2: Q^(2) = 3/4*Q^n + 1/4*(Q^(1) + dt*L(Q^(1)))   [RK weight 1/6]
    mpi_exchange_halos(tree, solver.mpi_);
    tree_rhs(tree, solver.rhs_, periodic, 1.0/6.0, -1, false, open_bc, ducros);
    if (use_sat) tree_sat_penalty(tree, solver.rhs_, cfg.numerics.sat_tau);
#pragma omp parallel for
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int t  = 0; t  < CellBlock::NTILE; ++t) {
        double*       qs = solver.Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = solver.Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = solver.rhs_[ii].Q[v].tile_ptr(t);
        #pragma omp simd simdlen(8)
        for (int lane = 0; lane < CellBlock::W; ++lane)
            qs[lane] = (3.0/4.0)*qn[lane] + (1.0/4.0)*(qs[lane] + dt*r[lane]);
    }
    phi_stage(3.0/4.0);
    apply_positivity_floor(solver.Qs_);
    solver.copy_stage_to_tree(solver.Qs_);

    // Stage 3: Q^(n+1) = 1/3*Q^n + 2/3*(Q^(2) + dt*L(Q^(2)))   [RK weight 2/3]
    mpi_exchange_halos(tree, solver.mpi_);
    tree_rhs(tree, solver.rhs_, periodic, 2.0/3.0, -1, false, open_bc, ducros);
    if (use_sat) tree_sat_penalty(tree, solver.rhs_, cfg.numerics.sat_tau);
#pragma omp parallel for
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int t  = 0; t  < CellBlock::NTILE; ++t) {
        double*       qs = solver.Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = solver.Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = solver.rhs_[ii].Q[v].tile_ptr(t);
        #pragma omp simd simdlen(8)
        for (int lane = 0; lane < CellBlock::W; ++lane)
            qs[lane] = (1.0/3.0)*qn[lane] + (2.0/3.0)*(qs[lane] + dt*r[lane]);
    }
    phi_stage(1.0/3.0);
    apply_positivity_floor(solver.Qs_);
    solver.copy_stage_to_tree(solver.Qs_);

    tree.apply_flux_correction(dt);
    return dt;
}
