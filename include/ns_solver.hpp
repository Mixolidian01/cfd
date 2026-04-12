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

// ── Diagnostics written every `diag_interval` steps ──────────────────────────
struct StepDiag {
    int    step;
    double t;
    double dt;
    double mass;        // total mass (should be conserved)
    double momentum_x;  // total x-momentum
    double kinetic_energy;
    double total_energy;
};

// ── Boundary condition selector ───────────────────────────────────────────────
enum class BCType { Periodic, Wall };

// ── Solver configuration ──────────────────────────────────────────────────────
struct SolverConfig {
    double cfl          = 0.8;
    double t_end        = 1.0;
    int    max_steps    = 1000000;
    int    diag_interval = 10;        // print diagnostics every N steps
    BCType bc           = BCType::Periodic;
    bool   verbose      = true;
    int    regrid_interval = 0;   // 0 = disabled; call regrid() every N steps
    int    max_level       = 2;   // max AMR refinement depth
    std::shared_ptr<SGSModel> sgs = nullptr;
    bool verbose_json = false;   // emit one JSON line per step to stdout
};

// ── NSSolver ──────────────────────────────────────────────────────────────────
struct NSSolver {
    BlockTree    tree;
    SolverConfig cfg;
    double       t    = 0.0;
    int          step = 0;

    std::vector<StepDiag> history;   // all recorded diagnostics

    // ── Initialise from a user-supplied IC function ───────────────────────────
    // ic(x, y, z) returns a Prim at that location.
    // Fills ALL cells (interior + ghost) so that ghost fill on step 0 is clean.
    void init(double domain_L,
              const std::function<Prim(double,double,double)>& ic);

    // ── Run until t >= t_end or step >= max_steps ────────────────────────────
    void run();

    // ── Single SSP-RK3 step ───────────────────────────────────────────────────
    // Returns the dt used.
    double advance();
    void regrid(); 

    // ── Diagnostics ──────────────────────────────────────────────────────────
    StepDiag compute_diag() const;
    void      print_diag(const StepDiag& d) const;
    
    int  scratch_leaf_count_ = -1;  ///< FIX P5: tracks last alloc size
    void alloc_scratch();

private:
    // Scratch storage: one RHS block per leaf (reused across stages)
    std::vector<CellBlock> rhs_;
    // Q^n storage: one block per leaf (for SSP-RK3 multi-stage)
    std::vector<CellBlock> Qn_;
    // Q^s storage: one block per leaf (intermediate stage)
    std::vector<CellBlock> Qs_;

    
    void save_Qn();
    void copy_stage_to_tree(const std::vector<CellBlock>& stage);
    void copy_tree_to_stage(std::vector<CellBlock>& stage);
};
