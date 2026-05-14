#pragma once
// sgs.hpp — Sub-Grid Scale (SGS) model interface
//
// DESIGN CONSTRAINT (Step 7 gate): this module must be a pure plug-in.
// It may not modify any file in Layers 0–3:
//   Layer 0: linalg, cell_block, block_tree
//   Layer 1: operators (HLLC, viscous RHS)
//   Layer 2: amr_operators, flux_register
//   Layer 3: ns_solver (advance, init, run)
//
// The SGS model is injected via a callback stored in SolverConfig:
//   cfg.sgs = std::make_shared<SmagorinskyModel>(Cs, Pr_t);
// NSSolver::advance() calls cfg.sgs->apply(*blk, h) on each leaf block
// AFTER the RK3 update and BEFORE ghost fill for the next step.
// This is the standard operator-split approach for LES SGS models.

#include "mesh/cell_block.hpp"
#include <memory>
#include <cmath>

// ── Base interface ────────────────────────────────────────────────────────────
struct SGSModel {
    virtual ~SGSModel() = default;
    // Apply SGS diffusion to block in-place.
    // h  = cell width (uniform within block)
    // dt = current time step
    virtual void apply(CellBlock& blk, double h, double dt) const = 0;
    virtual const char* name() const = 0;
};

// ── Null model (no SGS, default) ─────────────────────────────────────────────
struct NullSGS : SGSModel {
    void apply(CellBlock&, double, double) const override {}
    const char* name() const override { return "none"; }
};

// ── Smagorinsky–Lilly (1963/1966) ────────────────────────────────────────────
//
// Eddy viscosity:  ν_t = (Cs·Δ)² · |S̄|
//   Cs  = Smagorinsky constant (default 0.16)
//   Δ   = filter width = h  (one cell width)
//   |S̄| = sqrt(2 · S_ij · S_ij)   (resolved strain-rate magnitude)
//   S_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)
//
// SGS heat flux:   κ_t = ρ·ν_t·Cp / Pr_t
//   Pr_t = turbulent Prandtl number (default 0.9)
//
// Implementation: explicit forward-Euler diffusion added to conserved
// variables after the RK3 step (operator splitting, 1st-order in time).
// For stability: dt · ν_t / h² < 0.5  (checked but not enforced here;
// the CFL limiter in advance() already accounts for molecular viscosity,
// so the SGS contribution is a secondary correction).

class SmagorinskyModel : public SGSModel {
public:
    double Cs;    // Smagorinsky constant
    double Pr_t;  // turbulent Prandtl number

    explicit SmagorinskyModel(double Cs_ = 0.16, double Pr_t_ = 0.9)
        : Cs(Cs_), Pr_t(Pr_t_) {}

    const char* name() const override { return "Smagorinsky"; }

    void apply(CellBlock& blk, double h, double dt) const override;

private:
    // Strain-rate magnitude |S̄| at cell (i,j,k) using central differences
    static double strain_rate(const CellBlock& b, double h_inv,
                              int i, int j, int k);
};

// ── Dynamic Smagorinsky (Germano 1991, Lilly 1992) ───────────────────────────
//
// The Smagorinsky constant Cs² is computed locally via the Germano identity
// with a 3×3×3 box test filter at scale 2Δ.  Requires NG ≥ 2 (P2.1 gate).
//
// Algorithm per block:
//   1. Compute grid-scale S_ij and |S̄| for cells [1..NB2-2] (full stencil).
//   2. Apply test filter (1/27 box average over 3×3×3) to:
//        - velocities  ũ_i
//        - products    (u_i u_j)~  (resolved stress)
//        - products    (|S̄| S_ij)~  (model stress at grid scale)
//   3. Compute test-scale strain S̃_ij from ũ via central differences.
//   4. Germano / Lilly LS over block:
//        L_ij = (u_i u_j)~ - ũ_i ũ_j
//        M_ij = 2Δ²(4|S̃|S̃_ij - (|S̄|S_ij)~)
//        Cs²  = max(0, Σ_block L:M / (Σ_block M:M + ε))
//   5. mu_t = ρ · Cs² · Δ² · |S̄|  →  same face-centred div(τ) as Smagorinsky.
//
// Backscatter is suppressed (Cs² clipped at 0).
// References: Germano et al. (1991) Phys. Fluids A 3:1760;
//             Lilly (1992) Phys. Fluids A 4:633.
class DynamicSmagorinskyModel : public SGSModel {
public:
    double Pr_t;  // turbulent Prandtl number (default 0.9)

    explicit DynamicSmagorinskyModel(double Pr_t_ = 0.9) : Pr_t(Pr_t_) {}

    const char* name() const override { return "DynamicSmagorinsky"; }

    void apply(CellBlock& blk, double h, double dt) const override;
};
