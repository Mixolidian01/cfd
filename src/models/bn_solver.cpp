// bn_solver.cpp — BNSolver: BlockTree-backed two-phase Baer-Nunziato solver.
//
// SSP-RK3 (Shu-Osher 1988) time integration for the BN two-phase model.
// Ghost fill: bn_fill_ghosts_tree() — same-level periodic/wall only.
// Prolong/restrict: piecewise-constant (volume-average), same as CellBlock.

#include "models/bn_solver.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// bn_prolong: piecewise-constant prolongation from parent to 8 children.
// ─────────────────────────────────────────────────────────────────────────────
void bn_prolong(const BNCellBlock& parent, BNCellBlock children[8]) noexcept {
    for (int oct = 0; oct < 8; ++oct) {
        int ix = oct_ix(oct), iy = oct_iy(oct), iz = oct_iz(oct);
        BNCellBlock& child = children[oct];
        // Each child interior cell maps to one parent interior cell.
        // Child (i,j,k) ∈ [NG, NG+NB) → parent (NG + ix*NB/2 + (i-NG)/2, ...).
        // NB must be even (NB=8 ✓).
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            int pi = NG + ix*(NB/2) + (i - NG)/2;
            int pj = NG + iy*(NB/2) + (j - NG)/2;
            int pk = NG + iz*(NB/2) + (k - NG)/2;
            int pf = cell_idx(pi, pj, pk);
            int cf = cell_idx(i,  j,  k);
            for (int v = 0; v < NVAR_BN; ++v)
                child.Q[v][cf] = parent.Q[v][pf];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// bn_restrict: volume-average 8 children into parent.
// ─────────────────────────────────────────────────────────────────────────────
void bn_restrict(BNCellBlock& parent, const BNCellBlock children[8]) noexcept {
    // Zero interior.
    for (int v = 0; v < NVAR_BN; ++v)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i)
            parent.Q[v][cell_idx(i,j,k)] = 0.0;

    for (int oct = 0; oct < 8; ++oct) {
        int ix = oct_ix(oct), iy = oct_iy(oct), iz = oct_iz(oct);
        const BNCellBlock& child = children[oct];
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            int pi = NG + ix*(NB/2) + (i - NG)/2;
            int pj = NG + iy*(NB/2) + (j - NG)/2;
            int pk = NG + iz*(NB/2) + (k - NG)/2;
            int pf = cell_idx(pi, pj, pk);
            int cf = cell_idx(i,  j,  k);
            for (int v = 0; v < NVAR_BN; ++v)
                parent.Q[v][pf] += 0.125 * child.Q[v][cf];  // average over 8 children
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BNSolver internals
// ─────────────────────────────────────────────────────────────────────────────
void BNSolver::alloc_scratch() {
    const int n = (int)tree.leaf_indices().size();
    const auto& leaves = tree.leaf_indices();
    Q.clear();   Q.reserve(n);
    rhs_.clear(); rhs_.reserve(n);
    Qn_.clear(); Qn_.reserve(n);
    Qs_.clear(); Qs_.reserve(n);
    for (int li : leaves) {
        const auto& nd = tree.nodes[li];
        double ox = nd.ox, oy = nd.oy, oz = nd.oz;
        double h  = tree.domain_L() / (NB * (1u << nd.level));
        Q.emplace_back(ox, oy, oz, h);
        rhs_.emplace_back(ox, oy, oz, h);
        Qn_.emplace_back(ox, oy, oz, h);
        Qs_.emplace_back(ox, oy, oz, h);
    }
}

void BNSolver::sync_tree_to_q() {
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        const auto& nd = tree.nodes[leaves[ii]];
        if (!nd.has_block()) continue;
        // Copy from the block stub into Q (block holds geometry only after init;
        // actual state lives in Q).  No-op if block data not used.
    }
}

void BNSolver::sync_q_to_tree() {
    // BNSolver does not store Q inside BlockNode (no BNCellBlock in BlockNode).
    // This is a no-op; tree topology is managed via BlockTree independently.
}

void BNSolver::init(double L,
                    const std::function<void(BNCellBlock&, double, double, double, double)>& ic,
                    BNEosParams eos_params,
                    BCVariant bc_variant) {
    eos = eos_params;
    bc  = bc_variant;
    tree.set_periodic(std::holds_alternative<PeriodicBC>(bc));
    tree.init(L);
    t = 0.0; step = 0;
    alloc_scratch();
    // Apply IC to each leaf.
    const auto& leaves = tree.leaf_indices();
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        const auto& nd = tree.nodes[leaves[ii]];
        ic(Q[ii], nd.ox, nd.oy, nd.oz, Q[ii].h);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CFL time step over all leaves.
// ─────────────────────────────────────────────────────────────────────────────
static double bn_tree_cfl_dt(const BlockTree& tree,
                              const std::vector<BNCellBlock>& Q,
                              double cfl,
                              const BNEosParams& eos) noexcept {
    const auto& leaves = tree.leaf_indices();
    double dt = 1e30;
    for (int ii = 0; ii < (int)leaves.size(); ++ii)
        dt = std::min(dt, bn_cfl_dt(Q[ii], cfl, eos));
    return dt;
}

// ─────────────────────────────────────────────────────────────────────────────
// BNSolver::advance — SSP-RK3 step.
// ─────────────────────────────────────────────────────────────────────────────
double BNSolver::advance() {
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();
    assert(NL == (int)Q.size());

    const double cfl = 0.4;
    double dt = bn_tree_cfl_dt(tree, Q, cfl, eos);

    // Save Q^n.
    Qn_ = Q;

    // Helper: fill ghosts, compute RHS, accumulate stage.
    auto rhs_call = [&]() {
        bn_fill_ghosts_tree(tree, Qs_, bc);
        for (int ii = 0; ii < NL; ++ii) {
            for (int v = 0; v < NVAR_BN; ++v)
                rhs_[ii].Q[v].assign(NCELL, 0.0);
            compute_rhs_bn(Qs_[ii], rhs_[ii], eos);
        }
    };

    // Stage 1: Q^(1) = Q^n + dt*L(Q^n)
    Qs_ = Qn_;
    rhs_call();
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR_BN; ++v)
    for (int t  = 0; t  < BNCellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        for (int lane = 0; lane < BNCellBlock::W; ++lane)
            qs[lane] = qn[lane] + dt * r[lane];
    }

    // Stage 2: Q^(2) = 3/4*Q^n + 1/4*(Q^(1) + dt*L(Q^(1)))
    rhs_call();
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR_BN; ++v)
    for (int t  = 0; t  < BNCellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        for (int lane = 0; lane < BNCellBlock::W; ++lane)
            qs[lane] = (3.0/4.0)*qn[lane] + (1.0/4.0)*(qs[lane] + dt*r[lane]);
    }

    // Stage 3: Q^{n+1} = 1/3*Q^n + 2/3*(Q^(2) + dt*L(Q^(2)))
    rhs_call();
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR_BN; ++v)
    for (int t  = 0; t  < BNCellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        for (int lane = 0; lane < BNCellBlock::W; ++lane)
            qs[lane] = (1.0/3.0)*qn[lane] + (2.0/3.0)*(qs[lane] + dt*r[lane]);
    }

    Q = Qs_;
    this->t += dt;
    this->step += 1;
    return dt;
}

void BNSolver::run(double t_end, int max_steps) {
    while (t < t_end && step < max_steps)
        advance();
}
