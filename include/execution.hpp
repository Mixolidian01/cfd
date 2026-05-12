#pragma once
// Layer E — Execution backend tags (CLAUDE.md R4)
// Backend selected once at solver startup via factory.
// No physics or contract knowledge.

#include "ns_solver.hpp"
#include <memory>

struct CPUSerial {};
struct GPUCuda   {};

// Top-level solver interface returned by make_solver().
// ISolver is host-only; virtual is intentional and correct here (CLAUDE.md R4).
struct ISolver {
    virtual void   init(double domain_L,
                        const std::function<Prim(double,double,double)>& ic,
                        const std::function<double(double,double,double)>* phi_ic = nullptr) = 0;
    virtual double advance() = 0;
    virtual void   run()     = 0;
    virtual ~ISolver() = default;
};

// Factory: reads cfg.backend and returns the correct ISolver implementation.
// Currently: CPU path returns CpuSolverWrapper.
// GPU path deferred (use NSSolver + set_gpu_solver() directly for now).
std::unique_ptr<ISolver> make_solver(SolverConfig cfg, double domain_L, int n_blocks);
