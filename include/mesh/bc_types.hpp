#pragma once
// Layer P — Boundary condition structs satisfying BoundaryCondition concept.
// std::variant dispatch replaces BCType enum if/else chains (CLAUDE.md R3).

#include "mesh/cell_block.hpp"  // CellBlock
#include "schemes/concepts.hpp"    // BoundaryCondition
#include <variant>
#include <cmath>

struct PeriodicBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
};

struct WallBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double wall_temperature = 0.0;  // 0 → adiabatic
};

// Slip (inviscid) wall: only the wall-normal momentum is reflected.
// Use this for inviscid simulations to avoid tangential velocity discontinuities.
struct SlipWallBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
};

struct OpenBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double far_field_pressure = 0.0;
};

struct ContactAngleBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double contact_angle_deg = 90.0;
};

struct NscbcBC {
    void fill_ghost(CellBlock&, int, int) const noexcept {}
    double p_inf = 1.0;  // far-field static pressure (subsonic outflow)
};

using BCVariant = std::variant<PeriodicBC, WallBC, SlipWallBC, OpenBC, ContactAngleBC, NscbcBC>;

// Convenience query helpers
inline bool bc_is_periodic(const BCVariant& v) noexcept {
    return std::holds_alternative<PeriodicBC>(v);
}
inline bool bc_is_open(const BCVariant& v) noexcept {
    return std::holds_alternative<OpenBC>(v);
}
inline bool bc_is_nscbc(const BCVariant& v) noexcept {
    return std::holds_alternative<NscbcBC>(v);
}

// GPU integer encoding: 0=periodic, 1=wall (no-slip), 2=open, 3=nscbc, 4=slip-wall
// ContactAngleBC uses GPU wall path (1); Phase 14.2 sets contact angle via BlockTree::bc_cfg.
inline int bc_to_int(const BCVariant& v) noexcept {
    if (std::holds_alternative<WallBC>(v))           return 1;
    if (std::holds_alternative<OpenBC>(v))           return 2;
    if (std::holds_alternative<NscbcBC>(v))          return 3;
    if (std::holds_alternative<ContactAngleBC>(v))   return 1;
    if (std::holds_alternative<SlipWallBC>(v))       return 4;
    return 0;
}

// ── Layer C concept checks ────────────────────────────────────────────────────
static_assert(BoundaryCondition<PeriodicBC>);
static_assert(BoundaryCondition<WallBC>);
static_assert(BoundaryCondition<SlipWallBC>);
static_assert(BoundaryCondition<OpenBC>);
static_assert(BoundaryCondition<ContactAngleBC>);
static_assert(BoundaryCondition<NscbcBC>);
