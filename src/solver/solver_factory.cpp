#include "schemes/execution.hpp"
#include "solver/ns_solver.hpp"
#include <memory>
#include <stdexcept>

// ── CPU wrapper ───────────────────────────────────────────────────────────────
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

// ── GPU plugin slot ───────────────────────────────────────────────────────────
// Filled at static-init time by a linked CUDA TU (e.g. solver_factory_gpu.cu).
// Remains nullptr in CPU-only builds → make_solver throws on GPU request.
static GpuSolverFactory s_gpu_factory = nullptr;

void register_gpu_solver_factory(GpuSolverFactory f) noexcept {
    s_gpu_factory = f;
}

// ── Factory ───────────────────────────────────────────────────────────────────
std::unique_ptr<ISolver> make_solver(SolverConfig cfg, double domain_L, int n_blocks) {
    if (cfg.exec.backend == SolverConfig::ExecutionBackend::GPU) {
        if (!s_gpu_factory)
            throw std::runtime_error(
                "make_solver: GPU backend requested but no GPU factory registered.\n"
                "Link solver_factory_gpu.cu (CUDA TU) into your binary, or use\n"
                "NSSolver + set_gpu_solver() directly.");
        return s_gpu_factory(cfg, domain_L, n_blocks);
    }
    auto s = std::make_unique<CpuSolverWrapper>();
    s->solver.cfg = cfg;
    return s;
}
