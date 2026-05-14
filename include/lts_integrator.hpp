#pragma once
// P10-A2: LtsIntegrator — Berger-Oliger local time stepping (P4.1).
//
// Extracted from NSSolver so that the LTS time-integration kernel lives in
// its own translation unit alongside CpuRk3Integrator.
//
// Design mirrors CpuRk3Integrator: holds a reference to NSSolver and accesses
// private members (rhs_, Qn_, Qs_) via `friend struct LtsIntegrator` declared
// in ns_solver.hpp.  The level-filtered copy helpers (copy_tree_to_stage_level,
// copy_stage_to_tree_level) are private methods of this class rather than of
// NSSolver — they are only ever called from lts_rk3_level_.
//
// step() does NOT update solver.t / solver.step / solver.last_dt_; those are
// updated by NSSolver::advance() after step() returns, consistent with
// CpuRk3Integrator::step().

#include "ns_solver.hpp"
#include <vector>

struct LtsIntegrator : TimeIntegrator {
    NSSolver& solver;

    explicit LtsIntegrator(NSSolver& s) noexcept : solver(s) {}

    // Execute one Berger-Oliger LTS step.  Returns the coarse dt used.
    double step(BlockTree& tree, double cfl) override;

private:
    // One full SSP-RK3 step for leaves at `level` only.
    void rk3_level(BlockTree& tree, int level, double dt,
                   double sub_weight, bool coarse_mode);

    // Level-filtered stage ↔ tree copy helpers.
    void copy_tree_to_stage_level(std::vector<CellBlock>& stage, int level);
    void copy_stage_to_tree_level(const std::vector<CellBlock>& stage, int level);
};
