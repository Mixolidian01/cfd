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

    // P3.5: IMEX-ARK implicit viscous solve
    bool use_imex  = false;
    int  mg_levels = 3;

    // P4.1: Berger-Oliger local time stepping.
    // When use_lts == true and the tree has more than one refinement level,
    // advance() dispatches to advance_lts().  The fine level takes lts_ratio
    // sub-steps of dt_c/lts_ratio per coarse step.  Two levels are supported;
    // deeper trees recursively apply the same principle.
    bool use_lts   = false;
    int  lts_ratio = 2;   // refinement ratio (must match tree's geometric ratio)
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
    double ke_prev_  = -1.0;
    double last_dt_  = 0.0;   // B4: populated by advance(), exposed via compute_diag()

    void save_Qn();
    void copy_stage_to_tree(const std::vector<CellBlock>& stage);
    void copy_tree_to_stage(std::vector<CellBlock>& stage);

    // Level-filtered copy helpers for LTS (only touch leaves at `level`).
    void copy_tree_to_stage_level(std::vector<CellBlock>& stage, int level);
    void copy_stage_to_tree_level(const std::vector<CellBlock>& stage, int level);

    // P3.5: IMEX advance — implicit viscous Helmholtz correction after RK3.
    double advance_imex();

    // P4.1: Berger-Oliger LTS.
    // lts_rk3_level runs one full SSP-RK3 step for leaves at `level`.
    //   sub_weight: 1/r for fine levels (r sub-steps), 1.0 for the coarse level.
    //   Flux accumulation weight = sub_weight × {1/6, 1/6, 2/3} per stage.
    // advance_lts is the entry point dispatched from advance() when use_lts=true.
    // coarse_mode=true: C/F coarse ghosts use zero-gradient fill (LTS coarse step only).
    void   lts_rk3_level(int level, double dt, double sub_weight, bool coarse_mode = false);
    double advance_lts();
};
