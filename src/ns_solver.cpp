// DESIGN.md reference: Layer 3 — Time Loop
// Fix log:
//   sig-match    : all signatures match ns_solver.hpp exactly
//   cfg-members  : regrid_interval / max_level / sgs / verbose_json live in cfg
//   alias        : RK3 stages 2/3 no longer alias Qs_ as both src and dst
//   h-field      : node.block->h used everywhere
//   P0.6         : static ke_prev replaced by member ke_prev_
//   P1.5         : regrid() full Berger-Colella protocol
//   S07/S08/S03-fix : fill_ghosts refreshed before SGS apply() so stencil
//                     reads Q^{n+1} ghosts, not stale Q^{(2)} from Stage 3
//   A05-fix4     : flux registers zeroed once before RK3; each stage passes
//                  its SSP-RK3 quadrature weight (1/6, 1/6, 2/3) to tree_rhs
//                  so apply_flux_correction(dt) uses the time-averaged flux
//   P1.5-fix     : regrid() now calls rebuild_neighbours()+fill_ghosts after
//                  Pass 1 (refine) and before Pass 2 (coarsen) so that
//                  should_coarsen() sees initialised ghost cells on new fine
//                  blocks and cannot spuriously coarsen freshly refined leaves.
//   A05-fix5     : regrid() moved to TOP of advance() (on Q^n) so that
//                  coarsen()->restrict_to_parent() can never overwrite cells
//                  that were already corrected by apply_flux_correction().
//                  Protocol: regrid(Q^n) → zero_regs → RK3 → apply_correction.
#include "../include/sgs.hpp"
#include "../include/amr_operators.hpp"
#include "../include/ns_solver.hpp"
#include "../include/cpu_rk3.hpp"
#include "../include/lts_integrator.hpp"
#include "../include/operators.hpp"
#include "../include/linalg.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

// R3: overloaded helper for std::visit dispatch (C++17 deduction-guide, C++20 compatible).
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// =============================================================================
// Helpers
// =============================================================================
static double block_mass(const CellBlock& b)         { return b.total_mass(); }
static double block_momentum_x(const CellBlock& b)   { return b.total_momentum_x(); }
static double block_total_energy(const CellBlock& b) { return b.total_energy(); }

static double block_kinetic_energy(const CellBlock& b) {
    double s = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        Prim q = b.prim(i,j,k);
        s += 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
    }
    return s * b.h*b.h*b.h;
}

// =============================================================================
// SolverConfig::validate
// =============================================================================
void SolverConfig::validate() const {
    if (time.cfl <= 0.0 || time.cfl > 1.0)
        throw std::invalid_argument("SolverConfig: cfl must be in (0, 1]");
    if (time.t_end <= 0.0)
        throw std::invalid_argument("SolverConfig: t_end must be > 0");
    if (time.max_steps <= 0)
        throw std::invalid_argument("SolverConfig: max_steps must be > 0");
    if (amr.max_level < 0 || amr.max_level > 6)
        throw std::invalid_argument("SolverConfig: max_level must be in [0, 6]");
    if (amr.lts_ratio != 1 && amr.lts_ratio != 2 && amr.lts_ratio != 4)
        throw std::invalid_argument("SolverConfig: lts_ratio must be 1, 2, or 4");
    if (acdi.acdi_ceps < 0.0)
        throw std::invalid_argument("SolverConfig: acdi_ceps must be >= 0");
    if (numerics.sat_tau < 0.0)
        throw std::invalid_argument("SolverConfig: sat_tau must be >= 0");
    if (numerics.ducros_p_threshold < 0.0 || numerics.ducros_p_threshold > 1.0)
        throw std::invalid_argument("SolverConfig: ducros_p_threshold must be in [0, 1]");
    if (numerics.ducros_blend_width < 0.0)
        throw std::invalid_argument("SolverConfig: ducros_blend_width must be >= 0");
    // Note: ducros_blend_width == 0.0 is coerced to 1e-30 in fill_ducros_cache,
    // producing a step function rather than a ramp — set >= 1e-30 for a smooth blend.
    // wall_T == 0.0 is the adiabatic sentinel; any positive value is an
    // isothermal temperature.  Negative values are always invalid.
    if (bc.wall_T < 0.0)
        throw std::invalid_argument("SolverConfig: wall_T must be >= 0 (0 = adiabatic)");
    if (acdi.gamma_a <= 1.0)
        throw std::invalid_argument("SolverConfig: gamma_a must be > 1");
    if (acdi.gamma_b <= 1.0)
        throw std::invalid_argument("SolverConfig: gamma_b must be > 1");
    if (acdi.p_inf_a < 0.0)
        throw std::invalid_argument("SolverConfig: p_inf_a must be >= 0");
    if (acdi.p_inf_b < 0.0)
        throw std::invalid_argument("SolverConfig: p_inf_b must be >= 0");
    if (physics.mg_levels < 1 || physics.mg_levels > 8)
        throw std::invalid_argument("SolverConfig: mg_levels must be in [1, 8]");
}

// =============================================================================
// NSSolver::init
// =============================================================================
void NSSolver::init(double domain_L,
                    const std::function<Prim(double,double,double)>& ic,
                    const std::function<double(double,double,double)>* phi_ic) {
    cfg.validate();
    // P4.1-fix: set periodic flag BEFORE tree.init() so that every subsequent
    // rebuild_neighbours() call (triggered by refine/balance) wraps domain-boundary
    // C/F faces into the periodic neighbour table.
    tree.set_periodic(bc_is_periodic(cfg.bc.variant));
    tree.init(domain_L);
    t = 0.0; step = 0;
    history.clear();
    ke_prev_ = -1.0;

    auto& blk = *tree.nodes[0].block;
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = blk.ox + (i - NG + 0.5) * blk.h;
        double y = blk.oy + (j - NG + 0.5) * blk.h;
        double z = blk.oz + (k - NG + 0.5) * blk.h;
        Prim p = ic(x, y, z);
        int idx = cell_idx(i,j,k);
        // P14.1c: use eos_prim_to_cons so gamma_m/p_inf_m from IC are respected
        eos_prim_to_cons(p, blk.Q[0][idx], blk.Q[1][idx],
                            blk.Q[2][idx], blk.Q[3][idx], blk.Q[4][idx]);
        // P14.1: initialise phase field (interior + ghost; ghost overwritten at first fill)
        if (phi_ic && cfg.acdi.use_acdi)
            blk.phi_data_[idx] = (*phi_ic)(x, y, z);
    }

    alloc_scratch();
    // P10-A2: create the CPU integrators.  GPU solver (if any) is injected by the
    // application TU after init() returns — see set_gpu_solver().
    integrator_     = std::make_unique<CpuRk3Integrator>(*this);
    lts_integrator_ = std::make_unique<LtsIntegrator>(*this);
    // P8.1: GPU pool wiring (alloc+upload of IC, tree callbacks) is performed
    // by the application TU after init() returns — see gpu_pool.hpp.
    // NSSolver only stores the gpu_pool_ pointer; direct CUDA calls are in .cu TUs.
}

// =============================================================================
// copy helpers
// =============================================================================
void NSSolver::copy_tree_to_stage(std::vector<CellBlock>& stage) {
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        if (!tree.nodes[leaves[ii]].has_block()) continue;  // P7.1: remote leaf
        stage[ii] = *tree.nodes[leaves[ii]].block;
    }
}

void NSSolver::copy_stage_to_tree(const std::vector<CellBlock>& stage) {
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        if (!tree.nodes[leaves[ii]].has_block()) continue;  // P7.1: remote leaf
        *tree.nodes[leaves[ii]].block = stage[ii];
    }
}

void NSSolver::save_Qn() { copy_tree_to_stage(Qn_); }

// =============================================================================
// NSSolver::advance — alias-free SSP-RK3 with weighted Berger-Colella reflux
//
// SSP-RK3 (Shu-Osher) expands as:
//   Q^{n+1} = Q^n + (1/6)*dt*L^{(1)} + (1/6)*dt*L^{(2)} + (2/3)*dt*L^{(3)}\n//
// The Berger-Colella flux correction uses the same time-averaged fine flux:
//   1. Zero flux registers ONCE before stage 1.
//   2. Pass stage weight w_s to tree_rhs (w = 1/6, 1/6, 2/3).
//      accumulate_cf_fine_fluxes adds w_s * F_fine to the register.
//   3. apply_flux_correction(dt) once after stage 3.
// Register holds sum_s(w_s * F_fine_s) = time-averaged fine flux. ✓
//
// Regrid order (A05-fix5):
//   regrid() is called at the TOP of advance() on Q^n, BEFORE any RK3 stage.
//   This guarantees that the tree topology is fixed for the entire
//   zero_regs → RK3 → apply_correction sequence.  In particular, coarsen()
//   → restrict_to_parent() can never overwrite cells already written by
//   apply_flux_correction(), eliminating the ~2.69e-8 mass leak.
// =============================================================================
double NSSolver::advance() {
    // Populate BCRuntimeConfig before any ghost-fill (including in advance_imex/lts regrid).
    tree.bc_cfg.wall_T = cfg.bc.wall_T;
    if (auto* ob = std::get_if<OpenBC>(&cfg.bc.variant))
        tree.bc_cfg.open_bc_p = ob->far_field_pressure;
    else
        tree.bc_cfg.open_bc_p = 0.0;
    {
        double ca_cos = 0.0;
        if (auto* ca = std::get_if<ContactAngleBC>(&cfg.bc.variant); ca && cfg.acdi.use_acdi)
            ca_cos = std::cos(ca->contact_angle_deg * (M_PI / 180.0));
        tree.bc_cfg.wall_ca_cos  = ca_cos;
        tree.bc_cfg.wall_ca_ceps = cfg.acdi.acdi_ceps;
    }

    // P3.5: IMEX path.
    if (cfg.physics.use_imex) return advance_imex();
    // P4.1: LTS path — only if tree has more than one level.
    if (cfg.amr.use_lts && tree.max_leaf_level() > 0) {
        const double dt = lts_integrator_->step(tree, cfg.time.cfl);
        last_dt_ = dt;
        t    += dt;
        step += 1;
        return dt;
    }

    // A05-fix5: regrid on Q^n BEFORE the RK3 cycle so that the tree
    // topology is immutable during zero_regs → stages → apply_correction.
    // step > 0 guard: skip at step 0 (initial IC, no dynamics yet).
    if (cfg.amr.regrid_interval > 0 && step > 0 &&
        step % cfg.amr.regrid_interval == 0)
        regrid();

    // P11.8: GPU path — flat tree only.
    // When AMR is active (max_leaf_level > 0), fall back to CPU path which handles
    // Berger-Colella flux registers correctly.  Re-upload Q before the next GPU step
    // because the CPU path modifies CellBlock::Q without going through the GPU pool.
    if (gpu_solver_) {
        if (gpu_q_stale_) {
            gpu_solver_->upload_q();
            gpu_q_stale_ = false;
        }
        if (tree.max_leaf_level() == 0) {
            // Flat tree: full GPU path (no flux correction needed).
            // NOTE: gpu_rhs.cu hardcodes ducros p_threshold=0.1 and blend_width=0.1;
            // cfg.numerics.ducros_p_threshold/blend_width have no effect on this path (R9-D).
            const double dt = gpu_solver_->advance(tree, cfg.time.cfl);
            gpu_solver_->download_q(tree);
            last_dt_ = dt;
            t    += dt;
            step += 1;
            return dt;
        }
        // AMR active: fall through to CPU path; mark GPU Q stale for next flat step.
        gpu_q_stale_ = true;
    }

    // P10-A2: CPU flat-tree SSP-RK3 via CpuRk3Integrator.
    const double dt = integrator_->step(tree, cfg.time.cfl);

    last_dt_ = dt;
    t    += dt;
    step += 1;

    // S07/S08/S03-fix: refresh ghost cells before SGS operator-split.
    // After copy_stage_to_tree() the interior holds Q^{n+1} but ghosts
    // still carry Q^{(2)} from Stage 3's fill_ghosts call inside tree_rhs().
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

    // P12.1/P12.3/P12.4: compute per-step diagnostics.
    if (cfg.io.verbose_json || streamer_) {
        MetricsSnapshot ms;
        ms.step = step;
        ms.t    = t;
        ms.dt   = dt;
        const auto& lvs = tree.leaf_indices();
        ms.n_leaves = static_cast<int>(lvs.size());
        ms.rho_min = 1e300;
        ms.rho_max = 0.0;
        ms.gpu_active = (gpu_solver_ != nullptr);
        double px = 0.0, py = 0.0, pz = 0.0, etot = 0.0;
        for (int li : lvs) {
            auto& blk = *tree.nodes[li].block;
            const double h3 = blk.h * blk.h * blk.h;
            const int lv = tree.nodes[li].level;
            if (lv < 8) ms.leaves_per_level[lv]++;
            ms.mass += blk.total_mass();
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j)
            for (int i = NG; i < NG+NB; ++i) {
                Prim q = blk.prim(i,j,k);
                ms.ke += 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w) * h3;
                if (q.rho < ms.rho_min) ms.rho_min = q.rho;
                if (q.rho > ms.rho_max) ms.rho_max = q.rho;
                px   += blk.rhou(i,j,k) * h3;
                py   += blk.rhov(i,j,k) * h3;
                pz   += blk.rhow(i,j,k) * h3;
                etot += blk.E   (i,j,k) * h3;
            }
        }
        ms.cfl = dt / tree_cfl_dt(tree, 1.0);
        // P12.3: conservation baselines (set on first call; relative errors thereafter)
        const double mtm = std::sqrt(px*px + py*py + pz*pz);
        if (mass0_ < 0.0) {
            mass0_   = ms.mass;
            mtm0_    = mtm;
            energy0_ = etot;
        }
        ms.mass_error     = std::abs(ms.mass - mass0_)     / (std::abs(mass0_)   + 1e-300);
        ms.momentum_error = std::abs(mtm - mtm0_)          / (std::abs(mtm0_)    + 1e-300);
        ms.energy_error   = std::abs(etot  - energy0_)     / (std::abs(energy0_) + 1e-300);
        if (cfg.io.verbose_json) {
            std::fprintf(stdout,
                "{\"step\":%d,\"t\":%.6e,\"dt\":%.6e,\"cfl\":%.4f,"
                "\"ke\":%.6e,\"mass\":%.6e,\"leaves\":%d}\n",
                step, t, dt, ms.cfl, ms.ke, ms.mass, ms.n_leaves);
            std::fflush(stdout);
        }
        if (streamer_) streamer_->push_metrics(ms);
    }

    // Phase 6: push completed step to browser live feed (no-op when null).
    if (streamer_) streamer_->snapshot(tree, step, t);

    return dt;
}

// advance_imex — moved to src/imex_advance.cpp (R9-B)

// =============================================================================
// run
// =============================================================================
void NSSolver::run() {
    if (cfg.io.verbose)
        printf("%-8s %-12s %-12s %-14s %-14s\n",
               "step","t","dt","mass","KE");

    while (t < cfg.time.t_end && step < cfg.time.max_steps) {
        double dt = advance();
        if (step % cfg.io.diag_interval == 0 || t >= cfg.time.t_end) {
            auto d = compute_diag();
            history.push_back(d);
            if (cfg.io.verbose) print_diag(d);
        }
        (void)dt;
    }
}

// =============================================================================
// Diagnostics
// =============================================================================
StepDiag NSSolver::compute_diag() const {
    StepDiag d;
    d.step = step; d.t = t; d.dt = last_dt_;
    d.mass = 0; d.momentum_x = 0; d.kinetic_energy = 0; d.total_energy = 0;
    for (int li : tree.leaf_indices()) {
        if (!tree.nodes[li].has_block()) continue;  // remote leaf on this rank
        const auto& b = *tree.nodes[li].block;
        d.mass           += block_mass(b);
        d.momentum_x     += block_momentum_x(b);
        d.kinetic_energy += block_kinetic_energy(b);
        d.total_energy   += block_total_energy(b);
    }
    // P7.1: reduce across all MPI ranks
    if (mpi_) {
        d.mass           = mpi_allreduce_sum(d.mass,           *mpi_);
        d.momentum_x     = mpi_allreduce_sum(d.momentum_x,     *mpi_);
        d.kinetic_energy = mpi_allreduce_sum(d.kinetic_energy, *mpi_);
        d.total_energy   = mpi_allreduce_sum(d.total_energy,   *mpi_);
    }
    return d;
}

void NSSolver::print_diag(const StepDiag& d) const {
    printf("%-8d %-12.6e %-12.6e %-14.8e %-14.8e\n",
           d.step, d.t, d.dt, d.mass, d.kinetic_energy);
}

// advance_lts / lts_rk3_level / copy_{tree,stage}_to_{stage,tree}_level
// — extracted to LtsIntegrator in src/lts_integrator.cpp (P10-A2)

// =============================================================================
// alloc_scratch
// =============================================================================
void NSSolver::alloc_scratch() {
    const auto& leaves = tree.leaf_indices();
    int n = (int)leaves.size();
    if (n == scratch_leaf_count_) return;

    rhs_.clear(); Qn_.clear(); Qs_.clear();
    rhs_.reserve(n); Qn_.reserve(n); Qs_.reserve(n);
    for (int li : leaves) {
        const auto& blk = *tree.nodes[li].block;
        rhs_.emplace_back(blk.ox, blk.oy, blk.oz, blk.h);
        Qn_.emplace_back( blk.ox, blk.oy, blk.oz, blk.h);
        Qs_.emplace_back( blk.ox, blk.oy, blk.oz, blk.h);
    }
    scratch_leaf_count_ = n;
}

// =============================================================================
// NSSolver::regrid — P1.5 full Berger-Colella protocol
// =============================================================================
void NSSolver::regrid() {
    bool topology_changed = false;

    // ── Pass 1: refinement ────────────────────────────────────────────────
    {
        std::vector<int> to_refine;
        for (int li : tree.leaf_indices()) {
            auto& node = tree.nodes[li];
            if (node.level >= cfg.amr.max_level) continue;
            if (should_refine(*node.block, node.block->h))
                to_refine.push_back(li);
        }
        for (int li : to_refine) {
            if (!tree.nodes[li].is_leaf()) continue;
            tree.refine(li);
            topology_changed = true;
        }
    }

    // P1.5-fix: rebuild neighbours and fill ghosts after Pass 1 so that
    // the new fine blocks have initialised ghost cells before Pass 2
    // calls should_coarsen().  Without this, should_coarsen() reads zero
    // (or uninitialised) ghost data on freshly refined leaves and may
    // spuriously coarsen them immediately in the same regrid cycle.
    if (topology_changed) {
        tree.rebuild_neighbours();
        std::visit(overloaded{
            [&](const PeriodicBC&)      { tree.fill_ghosts_periodic(); },
            [&](const OpenBC&)          { tree.fill_ghosts_open(); },
            [&](const WallBC&)          { tree.fill_ghosts_wall(); },
            [&](const ContactAngleBC&)  { tree.fill_ghosts_wall(); },
        }, cfg.bc.variant);
    }

    // ── Pass 2: coarsening ────────────────────────────────────────────────
    {
        std::vector<int> to_coarsen;
        for (int li : tree.leaf_indices()) {
            int p = tree.nodes[li].parent;
            if (p < 0) continue;
            int fc = tree.nodes[p].first_child;
            if (fc < 0) continue;
            bool all_leaf   = true;
            bool all_coarse = true;
            for (int oct = 0; oct < 8; ++oct) {
                int ci = fc + oct;
                if (!tree.nodes[ci].is_leaf())          { all_leaf = false; break; }
                if (!should_coarsen(*tree.nodes[ci].block,
                                   tree.nodes[ci].block->h)) { all_coarse = false; }
            }
            if (all_leaf && all_coarse) {
                if (std::find(to_coarsen.begin(), to_coarsen.end(), p)
                    == to_coarsen.end())
                    to_coarsen.push_back(p);
            }
        }
        for (int p : to_coarsen) {
            int fc = tree.nodes[p].first_child;
            if (fc < 0) continue;
            bool all_leaf = true;
            for (int oct = 0; oct < 8; ++oct)
                if (!tree.nodes[fc+oct].is_leaf()) { all_leaf = false; break; }
            if (!all_leaf) continue;
            tree.coarsen(p);
            topology_changed = true;
        }
    }

    if (!topology_changed) return;

    tree.balance();
    tree.rebuild_neighbours();
    std::visit(overloaded{
        [&](const PeriodicBC&)      { tree.fill_ghosts_periodic(); },
        [&](const OpenBC&)          { tree.fill_ghosts_open(); },
        [&](const WallBC&)          { tree.fill_ghosts_wall(); },
        [&](const ContactAngleBC&)  { tree.fill_ghosts_wall(); },
    }, cfg.bc.variant);

    scratch_leaf_count_ = -1;
    alloc_scratch();

    // A3: rebuild GPU lists after topology change (new d_Q pointers; stale
    // CUDA graphs from the previous build would reference freed memory).
    if (gpu_solver_)
        gpu_solver_->build(tree, *gpu_pool_, bc_to_int(cfg.bc.variant));
}
