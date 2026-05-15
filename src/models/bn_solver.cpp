// bn_solver.cpp — BNSolver: BlockTree-backed two-phase Baer-Nunziato solver.
//
// SSP-RK3 (Shu-Osher 1988) time integration for the BN two-phase model.
// Ghost fill: bn_fill_ghosts_tree() — same-level periodic/wall only.
// Prolong/restrict: piecewise-constant (volume-average), same as CellBlock.
// C/F flux correction: Berger-Colella reflux via
//   bn_accumulate_cf_correction_fluxes / bn_apply_flux_correction.

#include "models/bn_solver.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <unordered_map>

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
    const auto& leaves = tree.leaf_indices();
    const int n = (int)leaves.size();
    Q.clear();   Q.reserve(n);
    rhs_.clear(); rhs_.reserve(n);
    Qn_.clear(); Qn_.reserve(n);
    Qs_.clear(); Qs_.reserve(n);
    regs_.clear(); regs_.resize(n);
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

void BNSolver::bn_zero_regs() noexcept {
    for (auto& slot_regs : regs_)
        for (auto& fr : slot_regs)
            std::fill(fr.begin(), fr.end(), 0.0);
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

// ─────────────────────────────────────────────────────────────────────────────
// BNSolver::refine — prolong leaf at slot → 8 children; rebuild all arrays.
// ─────────────────────────────────────────────────────────────────────────────
void BNSolver::refine(int slot) {
    const auto& old_leaves = tree.leaf_indices();
    assert(slot >= 0 && slot < (int)old_leaves.size());
    int ni = old_leaves[slot];

    // Save parent BN state; build (level,morton) → old-slot map for other leaves.
    BNCellBlock parent_blk = Q[slot];
    std::unordered_map<uint64_t, int> old_map;
    old_map.reserve(old_leaves.size() * 2);
    for (int ii = 0; ii < (int)old_leaves.size(); ++ii) {
        if (ii == slot) continue;
        const auto& nd = tree.nodes[old_leaves[ii]];
        old_map[((uint64_t)nd.level << 32) | nd.morton] = ii;
    }

    // Tree topology change — allocates 8 child CellBlocks (needed by leaf_indices).
    tree.refine(ni);

    int fc = tree.nodes[ni].first_child;

    // Prolong BN data into 8 child BNCellBlocks.
    BNCellBlock children[8];
    for (int oct = 0; oct < 8; ++oct) {
        int ci = fc + oct;
        double hc = parent_blk.h * 0.5;
        children[oct] = BNCellBlock(tree.nodes[ci].ox, tree.nodes[ci].oy,
                                     tree.nodes[ci].oz, hc);
    }
    bn_prolong(parent_blk, children);

    // Rebuild parallel arrays to match new leaf layout.
    const auto& new_leaves = tree.leaf_indices();
    const int NL = (int)new_leaves.size();
    std::vector<BNCellBlock> nQ, nR, nQn, nQs;
    nQ.reserve(NL); nR.reserve(NL); nQn.reserve(NL); nQs.reserve(NL);
    std::vector<std::array<std::vector<double>, NFACES>> nRegs(NL);

    for (int li : new_leaves) {
        const auto& nd = tree.nodes[li];
        double h = tree.domain_L() / (NB * (1u << nd.level));

        if (nd.parent == ni) {
            int oct = li - fc;
            nQ.push_back(children[oct]);
        } else {
            uint64_t key = ((uint64_t)nd.level << 32) | nd.morton;
            auto it = old_map.find(key);
            if (it != old_map.end())
                nQ.push_back(Q[it->second]);
            else
                nQ.emplace_back(nd.ox, nd.oy, nd.oz, h);
        }
        nR.emplace_back(nd.ox, nd.oy, nd.oz, h);
        nQn.emplace_back(nd.ox, nd.oy, nd.oz, h);
        nQs.emplace_back(nd.ox, nd.oy, nd.oz, h);
    }

    Q    = std::move(nQ);
    rhs_ = std::move(nR);
    Qn_  = std::move(nQn);
    Qs_  = std::move(nQs);
    regs_ = std::move(nRegs);
    tree.rebuild_neighbours();
}

// ─────────────────────────────────────────────────────────────────────────────
// BNSolver::coarsen — restrict 8 leaf children of parent_node; rebuild arrays.
// ─────────────────────────────────────────────────────────────────────────────
void BNSolver::coarsen(int parent_node) {
    int fc = tree.nodes[parent_node].first_child;
    assert(fc >= 0);
    for (int oct = 0; oct < 8; ++oct)
        assert(tree.nodes[fc + oct].is_leaf());

    // Build (level,morton) → old-slot map.
    const auto& old_leaves = tree.leaf_indices();
    std::unordered_map<uint64_t, int> old_map;
    old_map.reserve(old_leaves.size() * 2);
    for (int ii = 0; ii < (int)old_leaves.size(); ++ii) {
        const auto& nd = tree.nodes[old_leaves[ii]];
        old_map[((uint64_t)nd.level << 32) | nd.morton] = ii;
    }

    // Collect 8 child BNCellBlocks in octant order (fc+0 .. fc+7).
    BNCellBlock children[8];
    for (int oct = 0; oct < 8; ++oct) {
        int ci = fc + oct;
        uint64_t key = ((uint64_t)tree.nodes[ci].level << 32) | tree.nodes[ci].morton;
        auto it = old_map.find(key);
        assert(it != old_map.end());
        children[oct] = Q[it->second];
    }

    // Create parent BNCellBlock via volume-average restriction.
    const auto& pnd = tree.nodes[parent_node];
    double h_par = children[0].h * 2.0;
    BNCellBlock parent_blk(pnd.ox, pnd.oy, pnd.oz, h_par);
    bn_restrict(parent_blk, children);

    // Tree topology change.
    tree.coarsen(parent_node);

    // Rebuild parallel arrays.
    const auto& new_leaves = tree.leaf_indices();
    const int NL = (int)new_leaves.size();
    std::vector<BNCellBlock> nQ, nR, nQn, nQs;
    nQ.reserve(NL); nR.reserve(NL); nQn.reserve(NL); nQs.reserve(NL);
    std::vector<std::array<std::vector<double>, NFACES>> nRegs(NL);

    for (int li : new_leaves) {
        const auto& nd = tree.nodes[li];
        double h = tree.domain_L() / (NB * (1u << nd.level));

        if (li == parent_node) {
            nQ.push_back(parent_blk);
        } else {
            uint64_t key = ((uint64_t)nd.level << 32) | nd.morton;
            auto it = old_map.find(key);
            if (it != old_map.end())
                nQ.push_back(Q[it->second]);
            else
                nQ.emplace_back(nd.ox, nd.oy, nd.oz, h);
        }
        nR.emplace_back(nd.ox, nd.oy, nd.oz, h);
        nQn.emplace_back(nd.ox, nd.oy, nd.oz, h);
        nQs.emplace_back(nd.ox, nd.oy, nd.oz, h);
    }

    Q    = std::move(nQ);
    rhs_ = std::move(nR);
    Qn_  = std::move(nQn);
    Qs_  = std::move(nQs);
    regs_ = std::move(nRegs);
    tree.rebuild_neighbours();
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

    Qn_ = Q;

    const bool has_cf = (tree.max_leaf_level() > 0);
    if (has_cf) {
        bn_zero_regs();
        // Ensure registers are sized: NVAR_BN_CONS * NB * NB per face per slot.
        const int reg_sz = NVAR_BN_CONS * NB * NB;
        for (int ii = 0; ii < NL; ++ii)
            for (int d = 0; d < NFACES; ++d)
                if ((int)regs_[ii][d].size() != reg_sz)
                    regs_[ii][d].assign(reg_sz, 0.0);
    }

    // Helper: fill ghosts, compute RHS, apply C/F flux correction step.
    // stage_weight: SSP-RK3 quadrature weights (1/6, 1/6, 2/3).
    auto rhs_call = [&](double sw) {
        bn_fill_ghosts_tree(tree, Qs_, bc);
        for (int ii = 0; ii < NL; ++ii) {
            for (int v = 0; v < NVAR_BN; ++v)
                rhs_[ii].Q[v].assign(NCELL, 0.0);
            compute_rhs_bn(Qs_[ii], rhs_[ii], eos);
        }
        if (has_cf)
            bn_accumulate_cf_correction_fluxes(tree, Qs_, regs_, sw, eos);
    };

    // Stage 1: Q^(1) = Q^n + dt*L(Q^n)   [RK weight 1/6]
    Qs_ = Qn_;
    rhs_call(1.0 / 6.0);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR_BN; ++v)
    for (int t  = 0; t  < BNCellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        for (int lane = 0; lane < BNCellBlock::W; ++lane)
            qs[lane] = qn[lane] + dt * r[lane];
    }

    // Stage 2: Q^(2) = 3/4*Q^n + 1/4*(Q^(1) + dt*L(Q^(1)))   [RK weight 1/6]
    rhs_call(1.0 / 6.0);
    for (int ii = 0; ii < NL; ++ii)
    for (int v  = 0; v  < NVAR_BN; ++v)
    for (int t  = 0; t  < BNCellBlock::NTILE; ++t) {
        double*       qs = Qs_[ii].Q[v].tile_ptr(t);
        const double* qn = Qn_[ii].Q[v].tile_ptr(t);
        const double* r  = rhs_[ii].Q[v].tile_ptr(t);
        for (int lane = 0; lane < BNCellBlock::W; ++lane)
            qs[lane] = (3.0/4.0)*qn[lane] + (1.0/4.0)*(qs[lane] + dt*r[lane]);
    }

    // Stage 3: Q^{n+1} = 1/3*Q^n + 2/3*(Q^(2) + dt*L(Q^(2)))   [RK weight 2/3]
    rhs_call(2.0 / 3.0);
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
    if (has_cf)
        bn_apply_flux_correction(tree, Q, regs_, dt);

    this->t += dt;
    this->step += 1;
    return dt;
}

void BNSolver::run(double t_end, int max_steps) {
    while (t < t_end && step < max_steps)
        advance();
}
