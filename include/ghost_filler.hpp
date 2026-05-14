#pragma once
// R9-C: GhostFiller — consolidates ghost-fill dispatch for a BlockTree.
//
// Design rationale:
//   fill_ghosts_periodic/wall/open are BlockTree member functions that access
//   private internals (nodes, leaf_cache_, mpi_, static file-scope helpers).
//   Moving those bodies here would require friendship or broad accessor exposure,
//   adding more coupling than it removes.
//
//   Instead, GhostFiller owns the BCVariant → fill_ghosts_* dispatch logic that
//   is currently duplicated in NSSolver (and would accumulate in every new
//   caller).  The three BlockTree methods are the implementation; this struct
//   is the single call site.
//
// Usage:
//   GhostFiller::fill_all(tree, cfg.bc_variant);           // normal step
//   GhostFiller::fill_all(tree, cfg.bc_variant, true);     // LTS coarse step

#include "block_tree.hpp"
#include "bc_types.hpp"

struct GhostFiller {
    // Dispatch to the correct BlockTree::fill_ghosts_* method based on the
    // BCVariant carried in the solver config.
    //
    // cf_zero_grad=false (default): fine C/F ghosts from coarse cell averages.
    // cf_zero_grad=true  (LTS):     coarse C/F ghosts use zero-gradient
    //                               extrapolation so viscous flux at C/F is zero.
    static void fill_all(BlockTree& tree, const BCVariant& bc, bool cf_zero_grad = false);
};
