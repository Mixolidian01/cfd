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

struct BNSolver {
    BlockTree   tree;
    BNEosParams eos;
    BCVariant   bc = PeriodicBC{};

    std::vector<BNCellBlock> Q;     // current state (leaf-indexed)
    std::vector<BNCellBlock> rhs_;  // RHS scratch
    std::vector<BNCellBlock> Qn_;   // Q^n backup
    std::vector<BNCellBlock> Qs_;   // RK stage

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

    // Copy all leaf blocks into Q.
    void sync_tree_to_q();
    // Copy Q back into leaf blocks.
    void sync_q_to_tree();
};
