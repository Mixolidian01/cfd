#pragma once
// Layer E — Execution backend tags (CLAUDE.md R4)
// Backend selected once at solver startup via factory.
// No physics or contract knowledge.

#include "solver/ns_solver.hpp"
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

// Factory: reads cfg.exec.backend and returns the correct ISolver.
// CPU path: always available (CpuSolverWrapper).
// GPU path: available only when a GPU factory has been registered via
//   register_gpu_solver_factory().  This is done at static-init time by
//   a CUDA TU (e.g. src/cuda/solver_factory_gpu.cu) linked into the binary.
//   If no factory is registered and GPU is requested, make_solver throws.
std::unique_ptr<ISolver> make_solver(SolverConfig cfg, double domain_L, int n_blocks);

// Plugin hook for the GPU factory.  A CUDA TU calls this once at static init.
using GpuSolverFactory = std::unique_ptr<ISolver>(*)(SolverConfig, double, int);
void register_gpu_solver_factory(GpuSolverFactory f) noexcept;
