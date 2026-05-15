#pragma once
// DESIGN.md reference: Layer 2 — Discrete Operators
//
// This layer knows about CellBlocks and BlockTrees but not about time integration.
// Every function is matrix-free: it reads Q, writes a flux or RHS, nothing else.
//
// Operator inventory:
//   hllc_flux_x/y/z   — HLLC Riemann flux at one face (compressible, entropy-stable)
//   convective_rhs     — divergence of convective fluxes for one block
//   viscous_rhs        — divergence of viscous stress tensor for one block
//   rhs                — convective_rhs + viscous_rhs  (full inviscid+viscous RHS)
//
// Sign convention (DESIGN.md):
//   dQ/dt = rhs(Q)   where  rhs = -(1/h)[F_{i+1/2} - F_{i-1/2} + G + H]  + viscous
//   The negative sign is inside rhs — callers just do Q += dt * rhs.
//
// Ghost cells MUST be filled before calling any rhs function.
// Primitive variables are computed on-the-fly inside each function (no storage).

#include "mesh/block_tree.hpp"
#include "mesh/bc_types.hpp"
#include "mesh/axis.hpp"
#include "schemes/concepts.hpp"
#include "physics/hllc_flux.hpp"
#include "physics/diff_ops.hpp"
#include "physics/face_interp.hpp"
#include <array>

// ── P13.1: compile-time axis tag ─────────────────────────────────────────────
// Enables template<Axis DIR> flux specialisations; eliminates 3-way if/else
// axis dispatch and asymmetric-bug surface. Underlying value matches int axis
// convention (0=X, 1=Y, 2=Z) for backward compat with existing callers.
// NOTE: Axis is now defined in axis.hpp; included above.

// ── Single-face HLLC flux (5 conserved variables) ────────────────────────────
// Returns F_hllc at the interface between state L (left) and R (right).
// Both states are given as primitive variables (rho,u,v,w,p).
std::array<double,5> hllc_flux(const Prim& L, const Prim& R, int axis) noexcept;

// ── P13.1/R2 — compile-time-axis variants (DIR known at compile time) ─────────
// R2: delegates directly to HllcFlux<DIR> / HllcEsFlux<DIR> physics functors
// (include/physics/hllc_flux.hpp) — no runtime axis dispatch overhead.
// Usage: auto f = hllc_es_flux_t<Axis::X>(L, R);
template<Axis DIR>
inline std::array<double,5> hllc_es_flux_t(const Prim& L, const Prim& R) noexcept {
    return HllcEsFlux<DIR>{}(L, R);
}
template<Axis DIR>
inline std::array<double,5> hllc_flux_t(const Prim& L, const Prim& R) noexcept {
    return HllcFlux<DIR>{}(L, R);
}

// ── P3.2 — KE-consistent (Pirozzoli 2011) flux — shared by convective_rhs.cpp
//           and operators.cpp (accumulate_face_typed). Inline header to avoid
//           duplication across the two translation units.
// P13.2: FDKEC mass flux (Subbareddy & Candler 2009).
// P14.1: stiffened-gas E for H computation.
template<Axis DIR>
inline std::array<double,NVAR> kep_flux_t(const Prim& L, const Prim& R) noexcept {
    const double u_a   = 0.5*(L.u   + R.u  );
    const double v_a   = 0.5*(L.v   + R.v  );
    const double w_a   = 0.5*(L.w   + R.w  );
    const double p_a   = 0.5*(L.p   + R.p  );
    const double E_L   = (L.p + L.gamma_m*L.p_inf_m)/(L.gamma_m-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R   = (R.p + R.gamma_m*R.p_inf_m)/(R.gamma_m-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    const double H_a   = 0.5*((E_L+L.p)/L.rho + (E_R+R.p)/R.rho);
    const double un_L  = (DIR==Axis::X) ? L.u : (DIR==Axis::Y) ? L.v : L.w;
    const double un_R  = (DIR==Axis::X) ? R.u : (DIR==Axis::Y) ? R.v : R.w;
    const double mass  = 0.5*(L.rho*un_L + R.rho*un_R);
    std::array<double,NVAR> F;
    F[0] = mass;
    F[1] = mass*u_a + (DIR==Axis::X ? p_a : 0.0);
    F[2] = mass*v_a + (DIR==Axis::Y ? p_a : 0.0);
    F[3] = mass*w_a + (DIR==Axis::Z ? p_a : 0.0);
    F[4] = mass*H_a;
    return F;
}

// ── Per-block convective RHS  dQ/dt|_conv = -(1/h)(dF/dx + dG/dy + dH/dz) ──
// Ghost cells must be filled.  rhs is added to (not overwritten).
void convective_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept;

// ── Per-block viscous RHS  dQ/dt|_visc = (1/h)(d tau_ij/dx_j + d q_i/dx_i) ─
// Uses 2nd-order central differences for velocity gradients.
// mu from Sutherland, Pr = 0.72, kappa = mu*cp/Pr.
void viscous_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept;

// ── Ducros sensor configuration (P11.6) ──────────────────────────────────────
// Passed as a value to compute_rhs / tree_rhs; constructed from SolverConfig
// fields (ducros_p_threshold, ducros_blend_width) by the caller.
struct DucrosConfig {
    double p_threshold = 0.1;   // |Δp|/p below this → phi_p = 0
    double blend_width = 0.1;   // linear blend width above threshold
};

// ── Full RHS: convective + viscous ───────────────────────────────────────────
// rhs_blk.Q is zeroed then filled with convective + viscous contributions.
void compute_rhs(const CellBlock& blk, CellBlock& rhs_blk,
                 const DucrosConfig& ducros = DucrosConfig{}) noexcept;

// ── Tree-level RHS (loops over all leaves) ───────────────────────────────────
// Fills ghost cells, then calls compute_rhs for every leaf block.
// rhs_blocks must have same size and order as tree.leaf_indices().
//
// A05-fix4: stage_weight is the SSP-RK3 quadrature weight for this sub-step:
//   stage 1: 1/6,  stage 2: 1/6,  stage 3: 2/3
// The caller must zero flux registers once before stage 1 and must NOT zero
// them between stages.  apply_flux_correction(dt) is called after stage 3.
// stage_weight: SSP-RK3 quadrature weight (1/6, 1/6, 2/3) for Berger-Colella flux.
// level_filter: when >= 0, only leaves at that refinement level participate in
//   compute_rhs/undo_cf/accumulate_fine steps.  Pass -1 for all levels (default).
//   Ghost fill is always global (all leaves) regardless of the filter.
// cf_coarse_zero_grad: when true, coarse C/F ghost cells are filled with zero-gradient
//   extrapolation (ghost = interior) rather than fine cell averages.  Used by the
//   LTS coarse step so that viscous flux at C/F interfaces is exactly zero, allowing
//   total-energy conservation after Berger-Colella correction.
void tree_rhs(BlockTree& tree,
              std::vector<CellBlock>& rhs_blocks,
              const BCVariant& bc,
              double stage_weight        = 1.0,
              int    level_filter        = -1,
              bool   cf_coarse_zero_grad = false,
              const DucrosConfig& ducros = DucrosConfig{}) noexcept;

// R10-T6: Typed tree-level RHS — same signature as tree_rhs but routes through
// compute_rhs_typed<Flux,Recon,EOS> for compile-time scheme resolution.
template<template<Axis> class Flux, template<Axis> class Recon, class EOS>
    requires RiemannFlux<Flux<Axis::X>>
          && SpatialReconstruction<Recon<Axis::X>>
          && EquationOfState<EOS>
void tree_rhs_typed(BlockTree& tree,
                    std::vector<CellBlock>& rhs_blocks,
                    const BCVariant& bc,
                    double stage_weight        = 1.0,
                    int    level_filter        = -1,
                    bool   cf_coarse_zero_grad = false,
                    const DucrosConfig& ducros = DucrosConfig{},
                    EOS    eos                 = EOS{}) noexcept;

// ── R5: typed entry-point (Flux × Recon × EOS resolved at compile time) ──────
// Template template parameters so the loop over Axis::X/Y/Z inside
// operators.cpp can instantiate Flux<Axis::X>, Flux<Axis::Y>, Flux<Axis::Z>
// from a single template argument.  Concept constraints applied here via
// requires (Layer C, CLAUDE.md R5).  Body lives in operators.cpp; explicit
// instantiations at the bottom of that TU satisfy the ODR.
template<template<Axis> class Flux, template<Axis> class Recon, class EOS>
    requires RiemannFlux<Flux<Axis::X>>
          && SpatialReconstruction<Recon<Axis::X>>
          && EquationOfState<EOS>
void compute_rhs_typed(const CellBlock& blk, CellBlock& rhs_blk,
                       const DucrosConfig& ducros = DucrosConfig{}) noexcept;

// ── P14.1: ACDI phase-field advection RHS ────────────────────────────────────
// Conservative 1st-order upwind: ∂φ/∂t = -∇·(φu).
// Ghost phi must be filled (via fill_ghosts_periodic or equivalent) before call.
// Adds (does not zero) the advective flux divergence to rhs_blk.phi_data_.
void phi_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept;

// ── P14.1b: ACDI interface-compression source ─────────────────────────────────
// Adds ε·∇·(∇φ − φ(1−φ)·n̂) to rhs_blk.phi_data_ where ε = ceps·h.
// n̂ = ∇φ / |∇φ| is the interface unit normal (regularised by eps_sq=1e-10/h²).
// Requires NG ≥ 2 (both gradient and divergence use 1-cell stencils from ghost layer).
// Ghost phi must be filled at least 2 layers deep before call.
void phi_compression_rhs(const CellBlock& blk, CellBlock& rhs_blk,
                          double ceps) noexcept;

// ── CFL time step ─────────────────────────────────────────────────────────────
// tree_cfl_dt: global minimum over all leaves.
// level_cfl_dt: minimum over leaves at a specific refinement level (P4.1 LTS).
double tree_cfl_dt(const BlockTree& tree, double cfl) noexcept;
double level_cfl_dt(const BlockTree& tree, int level, double cfl) noexcept;

// ── P13.5: SBP-SAT interface penalty at AMR C/F boundaries ───────────────────
// Adds σ·(Q_coarse_ghost − Q_fine_interior)/h_f to fine boundary cells and
// −σ·avg_patch/h_c to the corresponding coarse boundary cell (conservative).
// Ghost cells must be filled BEFORE calling this.
// tau: penalty coefficient (>= 0.5 for semi-discrete energy stability).
// Typical usage: after each tree_rhs() stage in the RK3 loop.
void tree_sat_penalty(BlockTree& tree,
                      std::vector<CellBlock>& rhs_blocks,
                      double tau = 0.5) noexcept;
