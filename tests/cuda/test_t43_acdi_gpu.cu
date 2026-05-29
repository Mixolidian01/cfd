// tests/cuda/test_t43_acdi_gpu.cu
// Gate t43: phi mass conservation ≤ 1e-10 over 20 steps via GPU ACDI path.
// Periodic box, two-fluid bubble (phi=1 sphere, phi=0 exterior).
#include "solver/ns_solver.hpp"
#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_acdi.cuh"
#include <cmath>
#include <cstdio>
#include <cassert>

static double sphere_phi(double x, double y, double z) {
    double r2 = (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5);
    return 0.5*(1.0 - std::tanh((std::sqrt(r2)-0.2)/0.05));
}

int main() {
    NSSolver solver;
    solver.cfg.time.cfl      = 0.3;
    solver.cfg.acdi.use_acdi  = true;
    solver.cfg.acdi.acdi_ceps = 0.2;
    solver.cfg.exec.use_gpu   = true;
    solver.cfg.io.verbose     = false;
    solver.cfg.amr.regrid_interval = 0;

    std::function<Prim(double,double,double)> ic =
        [](double, double, double) -> Prim { return {1.0, 0.1, 0.0, 0.0, 1.0}; };
    std::function<double(double,double,double)> phi_fn =
        [](double x, double y, double z) { return sphere_phi(x, y, z); };
    solver.init(1.0, 1.0, 1.0, ic, &phi_fn);

    double phi0 = 0.0;
    for (int li : solver.tree.leaf_indices()) {
        auto& blk = *solver.tree.nodes[li].block;
        for (int k=2;k<10;++k) for (int j=2;j<10;++j) for (int i=2;i<10;++i)
            phi0 += blk.phi(i,j,k) * blk.h * blk.hy * blk.hz;
    }

    for (int s = 0; s < 20; ++s) solver.advance();

    double phi1 = 0.0;
    for (int li : solver.tree.leaf_indices()) {
        auto& blk = *solver.tree.nodes[li].block;
        for (int k=2;k<10;++k) for (int j=2;j<10;++j) for (int i=2;i<10;++i)
            phi1 += blk.phi(i,j,k) * blk.h * blk.hy * blk.hz;
    }

    const double err = std::fabs(phi1 - phi0) / (phi0 + 1e-30);
    std::printf("phi conservation error: %.3e\n", err);
    assert(err < 1e-10 && "GPU ACDI phi not conserved");
    std::puts("PASS");
}
