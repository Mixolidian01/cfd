// Gate t51 — Metrics & Monitoring system (M1-M4 sub-gates added incrementally)
#include "metrics/metrics_bus.hpp"
#include "metrics/field_dumper.hpp"
#include "solver/ns_solver.hpp"
#include "cuda/gpu_metrics.cuh"
#include "cuda/gpu_rhs.cuh"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_check.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_m1_residual() {
    // Allocate a fake RHS pool: n_leaves=2, NVAR=5, NCELL=1728.
    constexpr int NL = 2;
    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * NL * sizeof(double);
    double* d_rhs;
    CUDA_CHECK(cudaMalloc(&d_rhs, rhs_bytes));

    // Fill var 0 with 1.0, var 1-4 with 2.0 everywhere
    std::vector<double> h_rhs(GPU_NVAR * GPU_NCELL * NL, 0.0);
    for (int li = 0; li < NL; ++li)
        for (int v = 0; v < GPU_NVAR; ++v)
            for (int ci = 0; ci < GPU_NCELL; ++ci)
                h_rhs[(size_t)(li * GPU_NVAR + v) * GPU_NCELL + ci] = (v == 0) ? 1.0 : 2.0;
    CUDA_CHECK(cudaMemcpy(d_rhs, h_rhs.data(), rhs_bytes, cudaMemcpyHostToDevice));

    // Build fake GpuLeafRhsMeta array pointing into d_rhs
    std::vector<GpuLeafRhsMeta> h_metas(NL);
    for (int li = 0; li < NL; ++li) {
        h_metas[li] = {};
        h_metas[li].d_RHS = d_rhs + (size_t)li * GPU_NVAR * GPU_NCELL;
        h_metas[li].hx = h_metas[li].hy = h_metas[li].hz = 0.125;
    }
    GpuLeafRhsMeta* d_metas;
    CUDA_CHECK(cudaMalloc(&d_metas, NL * sizeof(GpuLeafRhsMeta)));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas.data(), NL * sizeof(GpuLeafRhsMeta),
                          cudaMemcpyHostToDevice));

    // Allocate pinned output buffer (n_leaves * NVAR partial sums)
    double* h_out;
    CUDA_CHECK(cudaMallocHost(&h_out, NL * GPU_NVAR * sizeof(double)));

    // Launch k_residual_norm
    k_residual_norm<<<dim3(NL, GPU_NVAR), 64>>>(d_metas, h_out, NL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU fold: sum over n_leaves, divide by n_leaves*N_INT, take sqrt
    constexpr int N_INT = GPU_NB * GPU_NB * GPU_NB;
    for (int v = 0; v < GPU_NVAR; ++v) {
        double sum = 0.0;
        for (int li = 0; li < NL; ++li) sum += h_out[li * GPU_NVAR + v];
        double l2 = sqrt(sum / (NL * N_INT));
        double expected = (v == 0) ? 1.0 : 2.0;
        assert(fabs(l2 - expected) < 1e-10 && "M1: L2 norm mismatch");
    }

    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_metas));
    CUDA_CHECK(cudaFreeHost(h_out));
    printf("M1 PASS\n");
}

int main() {
    // M0: config structs compile and have correct defaults
    SurfaceConfig sc;
    assert(sc.rho_ref == 0.0);
    ProbeConfig pc;
    assert(pc.type == ProbeConfig::Type::POINT);
    assert(pc.n_slabs == 32);
    MetricsConfig mc;
    assert(mc.global_interval == 10);
    assert(mc.residual_interval == 0);

    // CsvWriter smoke-test (writes to /dev/null)
    {
        CsvWriter w("/dev/null", "a,b,c");
        w.append(1, 2.0, "x");
    }
    printf("M0 PASS\n");

    test_m1_residual();
    return 0;
}
