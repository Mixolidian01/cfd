#pragma once
// BNSolver — BlockTree-backed two-phase Baer-Nunziato solver.
//
// Architecture: external parallel vector pattern (mirrors NSSolver).
//   tree        — AMR topology (BlockTree); leaf order drives all arrays.
//   Q/rhs_/Qn_/Qs_ — one BNCellBlock per leaf slot (indexed by leaf_indices()[ii]).
//
// Ghost fill: bn_fill_ghosts_tree() handles same-level, coarse-fine, and
//   domain-boundary (periodic/wall) BCs.
//   C/F fine←coarse: piecewise-constant (0th-order) from coarse interior cell.
//   C/F coarse←fine: 2×2 cell average over fine interior (conservative).
//
// Time integration: SSP-RK3 (Shu-Osher 1988).

#include "models/bn_model.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/bc_types.hpp"
#include <functional>
#include <vector>

// Ghost fill: fills BNCellBlock ghost layers from BlockTree neighbor table.
// Supports same-level neighbors (periodic wrap via tree.periodic_bc_) and
// zero-gradient wall fill.
void bn_fill_ghosts_tree(BlockTree& tree,
                         std::vector<BNCellBlock>& blocks,
                         const BCVariant& bc) noexcept;

// Prolong parent BNCellBlock Q into 8 child BNCellBlocks (piecewise-constant).
// Used by BNSolver::refine to initialise new leaves.
void bn_prolong(const BNCellBlock& parent, BNCellBlock children[8]) noexcept;

// Restrict 8 child BNCellBlocks into parent (volume-average).
// Used by BNSolver::coarsen.
void bn_restrict(BNCellBlock& parent, const BNCellBlock children[8]) noexcept;

// ── Berger-Colella C/F flux correction ───────────────────────────────────────
// Conservative variable count (Q[0..5]; Q[6]=α₁ is non-conservative).
static constexpr int NVAR_BN_CONS = 6;

// For each FINE leaf adjacent to a coarser leaf: accumulate
// sw*(F_fine − F_coarse_ghost) into regs[coarse_slot][d^1], weighted by
// stage_weight (SSP-RK3 quadrature: 1/6, 1/6, 2/3).
// The coarse RHS is NOT modified; correction is applied post-RK3.
void bn_accumulate_cf_correction_fluxes(
        const BlockTree& tree,
        const std::vector<BNCellBlock>& blocks,
        std::vector<std::array<std::vector<double>, NFACES>>& regs,
        double stage_weight,
        const BNEosParams& eos) noexcept;

// Apply the accumulated fine-flux correction to Q (called once after RK3).
void bn_apply_flux_correction(
        const BlockTree& tree,
        std::vector<BNCellBlock>& Q,
        const std::vector<std::array<std::vector<double>, NFACES>>& regs,
        double dt) noexcept;

struct BNSolver {
    BlockTree   tree;
    BNEosParams eos;
    BCVariant   bc = PeriodicBC{};

    std::vector<BNCellBlock> Q;     // current state (leaf-indexed)
    std::vector<BNCellBlock> rhs_;  // RHS scratch
    std::vector<BNCellBlock> Qn_;   // Q^n backup
    std::vector<BNCellBlock> Qs_;   // RK stage

    // Berger-Colella flux registers: regs_[slot][face] holds NVAR_BN_CONS*NB*NB
    // accumulated (stage-weighted) fine-face fluxes for each coarse leaf.
    std::vector<std::array<std::vector<double>, NFACES>> regs_;

    double t    = 0.0;
    int    step = 0;

    // Initialise: domain [0,L]^3, single-block root.
    // ic(x,y,z) fills one BNCellBlock; called for every leaf after init.
    void init(double L,
              const std::function<void(BNCellBlock&, double, double, double, double)>& ic,
              BNEosParams eos_params,
              BCVariant bc_variant = PeriodicBC{});

    // Advance one SSP-RK3 step.  Returns dt used.
    double advance();

    // Run until t >= t_end or step >= max_steps.
    void run(double t_end, int max_steps = 1000000);

    // Resize scratch arrays to match current leaf count.
    void alloc_scratch();

    // ── Runtime AMR ──────────────────────────────────────────────────────────
    // refine(slot): prolong leaf at slot → 8 children; rebuilds Q/rhs_/Qn_/Qs_.
    // slot is the position in tree.leaf_indices() of the leaf to refine.
    void refine(int slot);

    // coarsen(parent_node): restrict 8 leaf children of parent_node → parent;
    // rebuilds Q/rhs_/Qn_/Qs_.  parent_node is a BlockTree node index.
    void coarsen(int parent_node);

    // Copy all leaf blocks into Q.
    void sync_tree_to_q();
    // Copy Q back into leaf blocks.
    void sync_q_to_tree();

private:
    void bn_zero_regs() noexcept;
};
