// R9-C: GhostFiller implementation — BCVariant → BlockTree::fill_ghosts_* dispatch.
//
// Compiled as part of libblock (alongside amr_operators.cpp) so that
// fill_cf_ghosts ODR rules are respected — block_tree.cpp calls fill_cf_ghosts
// directly and must not be compiled again in libns_solver.
#include "../include/ghost_filler.hpp"

void GhostFiller::fill_all(BlockTree& tree, const BCVariant& bc, bool cf_zero_grad) {
    std::visit([&](const auto& variant) {
        using T = std::decay_t<decltype(variant)>;
        if constexpr (std::is_same_v<T, PeriodicBC>) {
            tree.fill_ghosts_periodic(cf_zero_grad);
        } else if constexpr (std::is_same_v<T, WallBC>) {
            tree.fill_ghosts_wall(cf_zero_grad);
        } else if constexpr (std::is_same_v<T, OpenBC>) {
            tree.fill_ghosts_open(cf_zero_grad);
        } else if constexpr (std::is_same_v<T, ContactAngleBC>) {
            // ContactAngleBC uses the wall path (bc_cfg.wall_ca_cos/ceps already set)
            tree.fill_ghosts_wall(cf_zero_grad);
        }
    }, bc);
}
