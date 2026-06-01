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

static void test_m2_surface() {
    // Build 6 ghost entries with ±x, ±y, ±z face normals and uniform pressure P0=1.
    // With no shear (u_wall=0, u_I=0), net force = sum(-p*n*A) = 0 by symmetry.
    constexpr int N = 6;
    constexpr double P0 = 1.0;
    constexpr float H  = 0.25f;

    // Fake Q array: rho=1, rhou=rhov=rhow=0, E=P0/(gamma-1)
    constexpr double E0 = P0 / (1.4 - 1.0);
    std::vector<double> h_Q(GPU_NVAR * GPU_NCELL, 0.0);
    for (int i = 0; i < GPU_NCELL; ++i) {
        h_Q[0 * GPU_NCELL + i] = 1.0;
        h_Q[4 * GPU_NCELL + i] = E0;
    }
    double* d_Q;
    CUDA_CHECK(cudaMalloc(&d_Q, GPU_NVAR * GPU_NCELL * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), GPU_NVAR * GPU_NCELL * sizeof(double),
                          cudaMemcpyHostToDevice));

    float normals[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    std::vector<GpuSurfaceEntry> h_entries(N);
    for (int i = 0; i < N; ++i) {
        h_entries[i] = {};
        h_entries[i].ghost_ptr = d_Q;
        h_entries[i].stencil[0] = d_Q;
        h_entries[i].w[0] = 1.0f;
        h_entries[i].nx = normals[i][0];
        h_entries[i].ny = normals[i][1];
        h_entries[i].nz = normals[i][2];
        h_entries[i].d  = 0.5f;
        h_entries[i].h  = H;
        // cx=cy=cz=0: moment arm is zero, so moments also zero
        h_entries[i].u_wall = 0.f; h_entries[i].v_wall = 0.f; h_entries[i].w_wall = 0.f;
    }
    GpuSurfaceEntry* d_entries;
    CUDA_CHECK(cudaMalloc(&d_entries, N * sizeof(GpuSurfaceEntry)));
    CUDA_CHECK(cudaMemcpy(d_entries, h_entries.data(), N * sizeof(GpuSurfaceEntry),
                          cudaMemcpyHostToDevice));

    double* d_acc;
    CUDA_CHECK(cudaMalloc(&d_acc, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_acc, 0, 6 * sizeof(double)));

    k_surface_forces<<<1, 256>>>(d_entries, N, d_acc, 1.8e-5f, 0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_acc[6];
    CUDA_CHECK(cudaMemcpy(h_acc, d_acc, 6 * sizeof(double), cudaMemcpyDeviceToHost));

    double A_sphere = (double)N * H * H;
    double force_mag = sqrt(h_acc[0]*h_acc[0] + h_acc[1]*h_acc[1] + h_acc[2]*h_acc[2]);
    assert(force_mag < 0.01 * P0 * A_sphere && "M2: net force not near zero");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_entries));
    CUDA_CHECK(cudaFree(d_acc));
    printf("M2 PASS\n");
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
    test_m2_surface();
    return 0;
}
