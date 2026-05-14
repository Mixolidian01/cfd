#pragma once
// P10-A2: CpuRk3Integrator — CPU flat-tree SSP-RK3 implementation.
//
// Owns the three Shu-Osher stages, positivity floor, and Berger-Colella
// flux correction.  SGS operator-split and per-step diagnostics remain in
// NSSolver::advance() to keep this TU focused on the pure time-integration
// kernel.
//
// Design note: friend access to NSSolver private members (rhs_, Qn_, Qs_,
// save_Qn()) is granted via `friend struct CpuRk3Integrator` in ns_solver.hpp.
// All other dependencies (cfg, mpi_, streamer_, tree) are public.

#include "ns_solver.hpp"

struct CpuRk3Integrator : TimeIntegrator {
    NSSolver& solver;

    explicit CpuRk3Integrator(NSSolver& s) noexcept : solver(s) {}

    // Execute one full SSP-RK3 step on the CPU flat-tree path.
    // Mutates tree (ghost fill, flux registers, Q data).
    // Returns the CFL-limited dt used for this step.
    double step(BlockTree& tree, double cfl) override;
};
