// imex_advance.cpp — NSSolver IMEX-Euler operator splitting
//
// Contains: NSSolver::advance_imex
//
// This is a member function of NSSolver; declaration remains in ns_solver.hpp.
// Extracted from ns_solver.cpp as part of R9-B maintainability refactor.
//
// Operator split:
//   Step 1 — Explicit SSP-RK3: Q* ← RK3(Q^n)  (convection + explicit viscous)
//   Step 2 — Implicit viscous Helmholtz correction per leaf block:
//     For each velocity component u_i:
//       (I − α·∇²) u_i^{n+1} = u_i^*   where  α = dt·μ_avg / ρ_avg
//     This re-solves viscosity implicitly, allowing dt larger than the viscous
//     CFL limit  dt < h²/(2ν).  Density Q[0] is never modified →
//     mass conservation is exact to floating-point round-off.
//   Step 3 — Energy update: IE is unchanged (T implicit would require a full
//     energy Helmholtz); only the kinetic energy change is applied:
//       E^{n+1} = E* − KE* + KE^{n+1}
//
// Reference: Kennedy & Carpenter (2003), §3, IMEX-ARK(2,3,3)-L scheme.
// For α > 0 the Helmholtz operator (I − α∇²) is SPD → V-cycle converges
// geometrically for any α/h² (M-matrix property).
#include "models/sgs.hpp"
#include "mesh/amr_operators.hpp"
#include "solver/ns_solver.hpp"
#include "schemes/operators.hpp"
#include "linalg/linalg.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

// R3: overloaded helper for std::visit dispatch (C++17 deduction-guide, C++20 compatible).
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

static constexpr double EPS_POS = 1e-12;

static void apply_positivity_floor(std::vector<CellBlock>& stage) noexcept {
    for (auto& blk : stage) {
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i, j, k);
            double rho  = blk.Q[0][f];
            double rhou = blk.Q[1][f];
            double rhov = blk.Q[2][f];
            double rhow = blk.Q[3][f];
            double E    = blk.Q[4][f];
            if (rho < EPS_POS) { blk.Q[0][f] = rho = EPS_POS; }
            const double ke = 0.5*(rhou*rhou + rhov*rhov + rhow*rhow) / rho;
            if ((GAMMA - 1.0)*(E - ke) < EPS_POS)
                blk.Q[4][f] = ke + EPS_POS / (GAMMA - 1.0);
        }
    }
}

// =============================================================================
// NSSolver::advance_imex — P3.5 IMEX-Euler operator splitting
// =============================================================================
double NSSolver::advance_imex() {
    bool periodic = bc_is_periodic(cfg.bc.variant);
    bool open_bc  = bc_is_open(cfg.bc.variant);
    const DucrosConfig ducros{ cfg.numerics.ducros_p_threshold, cfg.numerics.ducros_blend_width };

    // ── Step 1: same regrid + SSP-RK3 as advance() ───────────────────────
    if (cfg.amr.regrid_interval > 0 && step > 0 &&
        step % cfg.amr.regrid_interval == 0)
        regrid();

    double dt = tree_cfl_dt(tree, cfg.time.cfl);
    save_Qn();

    const auto& leaves = tree.leaf_indices();
    int NL = (int)leaves.size();

    tree.zero_flux_registers();

    // Stage 1: Q^(1) = Q^n + dt·L(Q^n)   [RK weight 1/6]
    tree_rhs(tree, rhs_, periodic, 1.0/6.0, -1, false, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int t  = 0; t  < CellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        #pragma omp simd simdlen(8)
        for (int lane = 0; lane < CellBlock::W; ++lane)
            qs[lane] = qn[lane] + dt * r[lane];
    }
    apply_positivity_floor(Qs_);
    copy_stage_to_tree(Qs_);

    // Stage 2: Q^(2) = 3/4·Q^n + 1/4·(Q^(1) + dt·L(Q^(1)))   [RK weight 1/6]
    tree_rhs(tree, rhs_, periodic, 1.0/6.0, -1, false, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int t  = 0; t  < CellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        #pragma omp simd simdlen(8)
        for (int lane = 0; lane < CellBlock::W; ++lane)
            qs[lane] = (3.0/4.0)*qn[lane] + (1.0/4.0)*(qs[lane] + dt*r[lane]);
    }
    apply_positivity_floor(Qs_);
    copy_stage_to_tree(Qs_);

    // Stage 3: Q^{n+1} = 1/3·Q^n + 2/3·(Q^(2) + dt·L(Q^(2)))  [RK weight 2/3]
    tree_rhs(tree, rhs_, periodic, 2.0/3.0, -1, false, open_bc, ducros);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int t  = 0; t  < CellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        #pragma omp simd simdlen(8)
        for (int lane = 0; lane < CellBlock::W; ++lane)
            qs[lane] = (1.0/3.0)*qn[lane] + (2.0/3.0)*(qs[lane] + dt*r[lane]);
    }
    apply_positivity_floor(Qs_);
    copy_stage_to_tree(Qs_);

    tree.apply_flux_correction(dt);

    last_dt_ = dt;
    t    += dt;
    step += 1;

    // ── Step 2: implicit viscous Helmholtz correction per block ──────────
    // Refresh ghosts so per-block velocity stencils see Q^{n+1} values.
    std::visit(overloaded{
        [&](const PeriodicBC&)      { tree.fill_ghosts_periodic(); },
        [&](const OpenBC&)          { tree.fill_ghosts_open(); },
        [&](const WallBC&)          { tree.fill_ghosts_wall(); },
        [&](const ContactAngleBC&)  { tree.fill_ghosts_wall(); },
    }, cfg.bc.variant);

    // Per-call NB³ scratch (thread_local avoids repeated heap allocation).
    static thread_local std::vector<double> uf(NB*NB*NB);
    static thread_local std::vector<double> vf(NB*NB*NB);
    static thread_local std::vector<double> wf(NB*NB*NB);
    static thread_local std::vector<double> rhs_f(NB*NB*NB);

    for (int li : tree.leaf_indices()) {
        CellBlock& blk = *tree.nodes[li].block;
        const double h = blk.h;

        // Block-averaged density and dynamic viscosity (Sutherland).
        double rho_avg = 0.0, mu_avg = 0.0;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            Prim q = blk.prim(i, j, k);
            rho_avg += q.rho;
            mu_avg  += sutherland(q.T);
        }
        const int Nint = NB*NB*NB;
        rho_avg /= Nint;
        mu_avg  /= Nint;

        // α = dt · ν = dt · μ/ρ   [m²]
        const double alpha = dt * mu_avg / rho_avg;
        if (alpha <= 0.0) continue;

        MGSolver mg;
        mg.build(NB, h, cfg.physics.mg_levels);

        // Extract all three post-RK3 velocity components (ghost-free NB³).
        // Flat index: (i-NG)*NB² + (j-NG)*NB + (k-NG)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            int flat  = (i-NG)*NB*NB + (j-NG)*NB + (k-NG);
            int cidx  = cell_idx(i,j,k);
            double rho_c = blk.Q[0][cidx];
            uf[flat] = blk.Q[1][cidx] / rho_c;
            vf[flat] = blk.Q[2][cidx] / rho_c;
            wf[flat] = blk.Q[3][cidx] / rho_c;
        }

        // Solve (I − α∇²) u_i^{new} = u_i^*  for u, v, w independently.
        // Initial guess = RHS = u_i^* (post-RK3 velocity).
        rhs_f = uf;  mg.solve_helmholtz(uf, rhs_f, alpha);
        rhs_f = vf;  mg.solve_helmholtz(vf, rhs_f, alpha);
        rhs_f = wf;  mg.solve_helmholtz(wf, rhs_f, alpha);

        // Write back momenta and update energy atomically per cell.
        // IE (internal energy) = E − KE is unchanged; only KE changes.
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            int flat  = (i-NG)*NB*NB + (j-NG)*NB + (k-NG);
            int cidx  = cell_idx(i,j,k);
            double rho_c = blk.Q[0][cidx];
            // Pre-Helmholtz velocities (still in blk.Q before overwrite).
            double u_old = blk.Q[1][cidx] / rho_c;
            double v_old = blk.Q[2][cidx] / rho_c;
            double w_old = blk.Q[3][cidx] / rho_c;
            double ke_old = 0.5 * rho_c * (u_old*u_old + v_old*v_old + w_old*w_old);
            // Post-Helmholtz velocities.
            double ke_new = 0.5 * rho_c * (uf[flat]*uf[flat]
                                          + vf[flat]*vf[flat]
                                          + wf[flat]*wf[flat]);
            blk.Q[1][cidx] = rho_c * uf[flat];
            blk.Q[2][cidx] = rho_c * vf[flat];
            blk.Q[3][cidx] = rho_c * wf[flat];
            blk.Q[4][cidx] += (ke_new - ke_old);  // IE unchanged, KE updated
        }
    }

    // SGS operator split (same as advance()).
    if (cfg.physics.sgs) {
        std::visit(overloaded{
            [&](const PeriodicBC&)      { tree.fill_ghosts_periodic(); },
            [&](const OpenBC&)          { tree.fill_ghosts_open(); },
            [&](const WallBC&)          { tree.fill_ghosts_wall(); },
            [&](const ContactAngleBC&)  { tree.fill_ghosts_wall(); },
        }, cfg.bc.variant);
        for (int li : tree.leaf_indices())
            cfg.physics.sgs->apply(*tree.nodes[li].block, tree.nodes[li].block->h, dt);
    }

    return dt;
}
