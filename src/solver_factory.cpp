#include "execution.hpp"
#include "ns_solver.hpp"
#include <memory>
#include <stdexcept>

// CpuSolverWrapper — ISolver backed by an NSSolver.
struct CpuSolverWrapper : ISolver {
    NSSolver solver;
    void   init(double domain_L,
                const std::function<Prim(double,double,double)>& ic,
                const std::function<double(double,double,double)>* phi_ic) override {
        solver.init(domain_L, ic, phi_ic);
    }
    double advance() override { return solver.advance(); }
    void   run()     override { solver.run(); }
};

std::unique_ptr<ISolver> make_solver(SolverConfig cfg,
                                     double /*domain_L*/, int /*n_blocks*/) {
    if (cfg.backend == SolverConfig::ExecutionBackend::GPU)
        throw std::runtime_error("GPU backend: use NSSolver + set_gpu_solver() directly (R4 placeholder)");

    auto s = std::make_unique<CpuSolverWrapper>();
    s->solver.cfg = cfg;
    return s;
}
