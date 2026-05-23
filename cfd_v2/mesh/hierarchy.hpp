#pragma once

#include <vector>

#include "cfd_v2/mesh/patch.hpp"

namespace cfd_v2::mesh {

// Level: collection of patches at a single refinement level.
//
// Invariants (to be enforced by the implementation):
//   - All patches on a level share the same (hx,hy,hz) and level index.
//   - Patches tile the domain at this level without overlap; gaps may exist
//     if finer levels cover some regions.
struct Level {
    int level_index{0};
    std::vector<Patch<>> patches;

    [[nodiscard]] std::size_t size() const noexcept { return patches.size(); }
};

// Hierarchy: vector of levels plus minimal metadata for AMR.
//
// Later phases will add:
//   - neighbor connectivity (same-level patch neighbors),
//   - parent/child relationships between levels,
//   - flux registers for Berger–Colella reflux,
//   - SAT penalty data at coarse/fine interfaces.
//
// For Phase 1 we focus on correct storage and basic iteration.
struct Hierarchy {
    std::vector<Level> levels;

    [[nodiscard]] bool empty() const noexcept { return levels.empty(); }
    [[nodiscard]] std::size_t n_levels() const noexcept { return levels.size(); }

    Level&       level(int idx)       noexcept { return levels.at(static_cast<std::size_t>(idx)); }
    const Level& level(int idx) const noexcept { return levels.at(static_cast<std::size_t>(idx)); }

    // Convenience helpers for the root level; these assume level 0 exists.
    [[nodiscard]] Level&       root()       { return levels.front(); }
    [[nodiscard]] const Level& root() const { return levels.front(); }
};

} // namespace cfd_v2::mesh
