#pragma once
// DESIGN.md reference: Layer 3 — Time Loop
//
// NSSolver owns the BlockTree and drives the SSP-RK3 time integration.
// It knows nothing about linear algebra (Layer 0) directly.
//
// SSP-RK3 (Shu-Osher):
//   Q^(1) = Q^n + dt * L(Q^n)
//   Q^(2) = 3/4 * Q^n + 1/4 * (Q^(1) + dt * L(Q^(1)))
//   Q^(n+1) = 1/3 * Q^n + 2/3 * (Q^(2) + dt * L(Q^(2)))
//
// Each stage: fill ghosts → compute rhs → update Q.
// dt is recomputed from CFL at the start of each step.

#include <memory>
#include "sgs.hpp"
#include "operators.hpp"
#include <functional>
#include <string>

// ── Diagnostics written every `diag_interval` steps ───────────────────────────────────
struct StepDiag {
    int    step;
    double t;
    double dt;
    double mass;        // total mass (should be conserved)
    double momentum_x;  // total x-momentum
    double kinetic_energy;
    double total_energy;
};

// ── Boundary condition selector ───────────────────────────────────────────────────────────────
enum class BCType { Periodic, Wall };

// ── Solver configuration ─────────────────────────────────────────────────────────────────────────────
struct SolverConfig {
    double cfl           = 0.8;
    double t_end         = 1.0;
    int    max_steps     = 1000000;
    int    diag_interval = 10;
    BCType bc            = BCType::Periodic;
    bool   verbose       = true;
    int    regrid_interval = 0;
    int    max_level       = 2;
    std::shared_ptr<SGSModel> sgs = nullptr;
    bool verbose_json = false;
};

// ── NSSolver ──────────────────────────────────────────────────────────────────────────────────────
struct NSSolver {
    BlockTree    tree;
    SolverConfig cfg;
    double       t    = 0.0;
    int          step = 0;

    std::vector<StepDiag> history;

    void init(double domain_L,
              const std::function<Prim(double,double,double)>& ic);
    void run();
    double advance();
    void regrid();

    StepDiag compute_diag() const;
    void     print_diag(const StepDiag& d) const;

    int  scratch_leaf_count_ = -1;  ///< FIX P5: tracks last alloc size
    void alloc_scratch();

private:
    std::vector<CellBlock> rhs_;
    std::vector<CellBlock> Qn_;
    std::vector<CellBlock> Qs_;

    // FIX P0.6: was a static local inside advance(); promoted to member so
    // that multiple NSSolver instances do not share state, and so that
    // reset on init() is explicit. Sentinel -1.0 → first step residual=0.
    double ke_prev_ = -1.0;

    void save_Qn();
    void copy_stage_to_tree(const std::vector<CellBlock>& stage);
    void copy_tree_to_stage(std::vector<CellBlock>& stage);
};
