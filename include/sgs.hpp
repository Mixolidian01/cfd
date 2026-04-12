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

#include "cell_block.hpp"
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
