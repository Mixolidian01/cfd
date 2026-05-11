#pragma once
// Layer C — Concept contracts (CLAUDE.md R1)
// Applied at every template boundary. Never inside __global__ kernels.

#include "cell_block.hpp"   // Prim, CellBlock, NVAR
#include <array>
#include <concepts>
#include <type_traits>

// ── RiemannFlux ──────────────────────────────────────────────────────────────
// A callable (L, R) → std::array<double, NVAR>.
// Axis is baked into the type at instantiation (template<Axis DIR>).
template<typename F>
concept RiemannFlux = requires(F f, const Prim& L, const Prim& R) {
    { f(L, R) } -> std::convertible_to<std::array<double, NVAR>>;
};

// ── SpatialReconstruction ────────────────────────────────────────────────────
// A callable that writes reconstructed left/right primitive states at a face.
// Signature: operator()(const Prim* pc, int i, int j, int k, Prim&, Prim&)
template<typename R>
concept SpatialReconstruction =
    requires(R r, const Prim* pc, int i, int j, int k, Prim& qL, Prim& qR) {
        r(pc, i, j, k, qL, qR);
    };

// ── EquationOfState ──────────────────────────────────────────────────────────
// Converts conservative state (ρ, ρu, ρv, ρw, E) → Prim.
template<typename E>
concept EquationOfState =
    requires(E eos, double rho, double rhou, double rhov, double rhow, double en) {
        { eos.cons_to_prim(rho, rhou, rhov, rhow, en) } -> std::same_as<Prim>;
    };

// ── BoundaryCondition ────────────────────────────────────────────────────────
// Fills one ghost layer in a block for a given axis (0/1/2) and side (0=lo, 1=hi).
template<typename B>
concept BoundaryCondition =
    requires(B bc, CellBlock& blk, int axis, int side) {
        bc.fill_ghost(blk, axis, side);
    };

// ── Property flags ───────────────────────────────────────────────────────────
// Specialise to std::true_type in the physics header that defines the functor.
template<typename F> struct is_entropy_stable  : std::false_type {};
template<typename F> struct is_conservative    : std::false_type {};
template<typename F> struct is_skew_symmetric  : std::false_type {};

template<typename F>
inline constexpr bool is_entropy_stable_v  = is_entropy_stable<F>::value;
template<typename F>
inline constexpr bool is_conservative_v    = is_conservative<F>::value;
template<typename F>
inline constexpr bool is_skew_symmetric_v  = is_skew_symmetric<F>::value;
