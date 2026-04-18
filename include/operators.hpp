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

#include "block_tree.hpp"
#include <array>

// ── Single-face HLLC flux (5 conserved variables) ────────────────────────────
// Returns F_hllc at the interface between state L (left) and R (right).
// Both states are given as primitive variables (rho,u,v,w,p).
std::array<double,5> hllc_flux(const Prim& L, const Prim& R, int axis) noexcept;

// ── P3.3 — Entropy-stable HLLC-ES flux (Chandrashekar 2013) ─────────────────
// Entropy-conservative (EC) base using log-mean ρ_ln, β_ln (β=ρ/2p) +
// Lax-Friedrichs scalar dissipation λ_max/2 * ΔQ.
// Satisfies the entropy inequality ∂η/∂t + ∂F_η/∂x ≤ 0 pointwise.
std::array<double,5> hllc_es_flux(const Prim& L, const Prim& R, int axis) noexcept;

// ── Per-block convective RHS  dQ/dt|_conv = -(1/h)(dF/dx + dG/dy + dH/dz) ──
// Ghost cells must be filled.  rhs is added to (not overwritten).
void convective_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept;

// ── Per-block viscous RHS  dQ/dt|_visc = (1/h)(d tau_ij/dx_j + d q_i/dx_i) ─
// Uses 2nd-order central differences for velocity gradients.
// mu from Sutherland, Pr = 0.72, kappa = mu*cp/Pr.
void viscous_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept;

// ── Full RHS: convective + viscous ───────────────────────────────────────────
// rhs_blk.Q is zeroed then filled with convective + viscous contributions.
void compute_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept;

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
              bool periodic,
              double stage_weight        = 1.0,
              int    level_filter        = -1,
              bool   cf_coarse_zero_grad = false,
              bool   open_bc             = false) noexcept;

// ── CFL time step ─────────────────────────────────────────────────────────────
// tree_cfl_dt: global minimum over all leaves.
// level_cfl_dt: minimum over leaves at a specific refinement level (P4.1 LTS).
double tree_cfl_dt(const BlockTree& tree, double cfl) noexcept;
double level_cfl_dt(const BlockTree& tree, int level, double cfl) noexcept;
