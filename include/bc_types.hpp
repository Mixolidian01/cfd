#pragma once
// Layer P — Boundary condition structs satisfying BoundaryCondition concept.
// std::variant dispatch replaces BCType enum if/else chains (CLAUDE.md R3).

#include "cell_block.hpp"  // CellBlock
#include "concepts.hpp"    // BoundaryCondition
#include <variant>
#include <cmath>

struct PeriodicBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
};

struct WallBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double wall_temperature = 0.0;  // 0 → adiabatic
};

struct OpenBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double far_field_pressure = 0.0;
};

struct ContactAngleBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double contact_angle_deg = 90.0;
};

using BCVariant = std::variant<PeriodicBC, WallBC, OpenBC, ContactAngleBC>;

// Convenience query helpers
inline bool bc_is_periodic(const BCVariant& v) noexcept {
    return std::holds_alternative<PeriodicBC>(v);
}
inline bool bc_is_open(const BCVariant& v) noexcept {
    return std::holds_alternative<OpenBC>(v);
}

// GPU integer encoding: 0=periodic, 1=wall, 2=open
// ContactAngleBC uses GPU wall path (1); Phase 14.2 sets contact angle via set_wall_contact_angle().
inline int bc_to_int(const BCVariant& v) noexcept {
    if (std::holds_alternative<WallBC>(v))           return 1;
    if (std::holds_alternative<OpenBC>(v))           return 2;
    if (std::holds_alternative<ContactAngleBC>(v))   return 1;
    return 0;
}

// ── Layer C concept checks ────────────────────────────────────────────────────
static_assert(BoundaryCondition<PeriodicBC>);
static_assert(BoundaryCondition<WallBC>);
static_assert(BoundaryCondition<OpenBC>);
static_assert(BoundaryCondition<ContactAngleBC>);
