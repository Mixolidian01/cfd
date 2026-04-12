// DESIGN.md reference: Layer 3 — Time Loop
// Fix log:
//   sig-match    : all signatures match ns_solver.hpp exactly
//   cfg-members  : regrid_interval / max_level / sgs / verbose_json live in cfg
//   alias        : RK3 stages 2/3 no longer alias Qs_ as both src and dst
//   h-field      : node.block->h used everywhere
//   P0.6         : static ke_prev replaced by member ke_prev_
//   P1.5         : regrid() full Berger-Colella protocol
#include "../include/sgs.hpp"
#include "../include/amr_operators.hpp"
#include "../include/ns_solver.hpp"
#include "../include/operators.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

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
// NSSolver::init
// =============================================================================
void NSSolver::init(double domain_L,
                    const std::function<Prim(double,double,double)>& ic) {
    tree.init(domain_L);
    t = 0.0; step = 0;
    history.clear();
    ke_prev_ = -1.0;   // FIX P0.6: reset sentinel on every init()

    auto& blk = *tree.nodes[0].block;
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = blk.ox + (i - NG + 0.5) * blk.h;
        double y = blk.oy + (j - NG + 0.5) * blk.h;
        double z = blk.oz + (k - NG + 0.5) * blk.h;
        Prim p = ic(x, y, z);
        int idx = cell_idx(i,j,k);
        blk.Q[0][idx] = p.rho;
        blk.Q[1][idx] = p.rho * p.u;
        blk.Q[2][idx] = p.rho * p.v;
        blk.Q[3][idx] = p.rho * p.w;
        blk.Q[4][idx] = p.rho * 0.5*(p.u*p.u + p.v*p.v + p.w*p.w)
                       + p.p / (GAMMA - 1.0);
    }

    alloc_scratch();
}

// =============================================================================
// copy helpers
// =============================================================================
void NSSolver::copy_tree_to_stage(std::vector<CellBlock>& stage) {
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        const auto& src = *tree.nodes[leaves[ii]].block;
        auto&       dst = stage[ii];
        for (int v = 0; v < NVAR; ++v) dst.Q[v] = src.Q[v];
    }
}

void NSSolver::copy_stage_to_tree(const std::vector<CellBlock>& stage) {
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        auto&       dst = *tree.nodes[leaves[ii]].block;
        const auto& src = stage[ii];
        for (int v = 0; v < NVAR; ++v) dst.Q[v] = src.Q[v];
    }
}

void NSSolver::save_Qn() { copy_tree_to_stage(Qn_); }

// =============================================================================
// NSSolver::advance — alias-free SSP-RK3
// =============================================================================
double NSSolver::advance() {
    bool periodic = (cfg.bc == BCType::Periodic);
    double dt = tree_cfl_dt(tree, cfg.cfl);

    save_Qn();

    const auto& leaves = tree.leaf_indices();
    int NL = (int)leaves.size();

    // Stage 1: Q^(1) = Q^n + dt*L(Q^n)
    tree_rhs(tree, rhs_, periodic);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int k  = ilo(); k <= ihi(); ++k)
    for (int j  = ilo(); j <= ihi(); ++j)
    for (int i  = ilo(); i <= ihi(); ++i) {
        int idx = cell_idx(i,j,k);
        Qs_[ii].Q[v][idx] = Qn_[ii].Q[v][idx] + dt * rhs_[ii].Q[v][idx];
    }
    copy_stage_to_tree(Qs_);

    // Stage 2: Q^(2) = 3/4*Q^n + 1/4*(Q^(1) + dt*L(Q^(1)))
    tree_rhs(tree, rhs_, periodic);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int k  = ilo(); k <= ihi(); ++k)
    for (int j  = ilo(); j <= ihi(); ++j)
    for (int i  = ilo(); i <= ihi(); ++i) {
        int idx = cell_idx(i,j,k);
        Qs_[ii].Q[v][idx] =
            (3.0/4.0) * Qn_[ii].Q[v][idx] +
            (1.0/4.0) * (Qs_[ii].Q[v][idx] + dt * rhs_[ii].Q[v][idx]);
    }
    copy_stage_to_tree(Qs_);

    // Stage 3: Q^(n+1) = 1/3*Q^n + 2/3*(Q^(2) + dt*L(Q^(2)))
    tree_rhs(tree, rhs_, periodic);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR; ++v)
    for (int k  = ilo(); k <= ihi(); ++k)
    for (int j  = ilo(); j <= ihi(); ++j)
    for (int i  = ilo(); i <= ihi(); ++i) {
        int idx = cell_idx(i,j,k);
        Qs_[ii].Q[v][idx] =
            (1.0/3.0) * Qn_[ii].Q[v][idx] +
            (2.0/3.0) * (Qs_[ii].Q[v][idx] + dt * rhs_[ii].Q[v][idx]);
    }
    copy_stage_to_tree(Qs_);

    // P1.4: apply flux correction after each full RK3 step
    tree.apply_flux_correction(dt);
    tree.zero_flux_registers();

    t    += dt;
    step += 1;

    // Regrid
    if (cfg.regrid_interval > 0 && step % cfg.regrid_interval == 0)
        regrid();

    // SGS operator-split
    if (cfg.sgs) {
        for (int li : tree.leaf_indices())
            cfg.sgs->apply(*tree.nodes[li].block, tree.nodes[li].block->h, dt);
    }

    // JSON diagnostics
    if (cfg.verbose_json) {
        double ke = 0.0;
        for (int li : tree.leaf_indices()) {
            auto& blk = *tree.nodes[li].block;
            double h3 = blk.h * blk.h * blk.h;
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j)
            for (int i = NG; i < NG+NB; ++i) {
                Prim q = blk.prim(i,j,k);
                ke += 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w) * h3;
            }
        }
        double residual = (ke_prev_ >= 0.0 && ke_prev_ > 0.0)
                          ? std::fabs(ke - ke_prev_) / ke_prev_
                          : 0.0;
        ke_prev_ = ke;
        double cfl_actual = dt / tree_cfl_dt(tree, 1.0);
        std::fprintf(stdout,
            "{\"step\":%d,\"t\":%.6e,\"dt\":%.6e,\"KE\":%.6e,"
            "\"residual\":%.6e,\"cfl\":%.4f}\n",
            step, t, dt, ke, residual, cfl_actual);
        std::fflush(stdout);
    }

    return dt;
}

// =============================================================================
// run
// =============================================================================
void NSSolver::run() {
    if (cfg.verbose)
        printf("%-8s %-12s %-12s %-14s %-14s\n",
               "step","t","dt","mass","KE");

    while (t < cfg.t_end && step < cfg.max_steps) {
        double dt = advance();
        if (step % cfg.diag_interval == 0 || t >= cfg.t_end) {
            auto d = compute_diag();
            history.push_back(d);
            if (cfg.verbose) print_diag(d);
        }
        (void)dt;
    }
}

// =============================================================================
// Diagnostics
// =============================================================================
StepDiag NSSolver::compute_diag() const {
    StepDiag d;
    d.step = step; d.t = t; d.dt = 0.0;
    d.mass = 0; d.momentum_x = 0; d.kinetic_energy = 0; d.total_energy = 0;
    for (int li : tree.leaf_indices()) {
        const auto& b = *tree.nodes[li].block;
        d.mass           += block_mass(b);
        d.momentum_x     += block_momentum_x(b);
        d.kinetic_energy += block_kinetic_energy(b);
        d.total_energy   += block_total_energy(b);
    }
    return d;
}

void NSSolver::print_diag(const StepDiag& d) const {
    printf("%-8d %-12.6e %-12.6e %-14.8e %-14.8e\n",
           d.step, d.t, d.dt, d.mass, d.kinetic_energy);
}

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
        double h = tree.nodes[li].block->h;
        rhs_.emplace_back(0.0, 0.0, 0.0, h);
        Qn_.emplace_back( 0.0, 0.0, 0.0, h);
        Qs_.emplace_back( 0.0, 0.0, 0.0, h);
    }
    scratch_leaf_count_ = n;
}

// =============================================================================
// NSSolver::regrid — P1.5 full Berger-Colella protocol
// =============================================================================
void NSSolver::regrid() {
    bool topology_changed = false;
    bool periodic = (cfg.bc == BCType::Periodic);

    // ── Pass 1: refinement ────────────────────────────────────────────────
    {
        // Snapshot current leaves before topology changes
        std::vector<int> to_refine;
        for (int li : tree.leaf_indices()) {
            auto& node = tree.nodes[li];
            if (node.level >= cfg.max_level) continue;
            if (should_refine(*node.block, node.block->h))
                to_refine.push_back(li);
        }
        for (int li : to_refine) {
            if (!tree.nodes[li].is_leaf()) continue;  // already refined by balance
            tree.refine(li);
            topology_changed = true;
        }
    }

    // ── Pass 2: coarsening (sibling groups where ALL 8 should_coarsen) ────
    {
        // Build set of candidate parent nodes whose 8 children are all leaves
        // and all pass should_coarsen.
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
                // Avoid duplicates
                if (std::find(to_coarsen.begin(), to_coarsen.end(), p)
                    == to_coarsen.end())
                    to_coarsen.push_back(p);
            }
        }
        for (int p : to_coarsen) {
            // Re-check: topology may have changed in refinement pass
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

    // ── 2:1 balance ───────────────────────────────────────────────────────
    tree.balance();

    // ── Rebuild neighbours + ghost fill ───────────────────────────────────
    tree.rebuild_neighbours();
    if (periodic)
        tree.fill_ghosts_periodic();
    else
        tree.fill_ghosts_wall();

    // ── Resize scratch arrays ─────────────────────────────────────────────
    scratch_leaf_count_ = -1;  // force realloc
    alloc_scratch();
}
