// P4.3 gate test — TBC / DSMEM ghost exchange
//
// Creates two adjacent CellBlocks with distinct interior data, uploads them
// to the GPU, runs gpu_ghost_exchange_x/y/z, downloads, and checks that
// each block's ghost cells match the adjacent block's interior face exactly.
//
// On sm_90+ (H100/B200): the TBC path is exercised.
// On sm_86 (RTX 3070):   the global-memory fallback runs instead;
//   the gate reports PASS with a "TBC fallback" note — correctness is the
//   same; the bandwidth advantage only materialises on sm_90+ hardware.
//
// Gate: GC01–GC03 each with L∞ < 1e-14.

#include "cuda/gpu_ghost_cluster.cuh"
#include "cuda/gpu_constants.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(e)); exit(1);} } while(0)

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool cond, double got = -1, double thr = -1) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.6e  threshold %.6e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// Fill host buffer for block b with pattern: Q[v][b][flat] = (b+1)*1000 + v*100 + flat
static void fill_host(std::vector<double>& h, int b, int n_blocks) {
    for (int v = 0; v < GPU_NVAR; ++v) {
        double* Qv = h.data() + (size_t)v * n_blocks * GPU_NCELL
                               + (size_t)b * GPU_NCELL;
        for (int flat = 0; flat < GPU_NCELL; ++flat)
            Qv[flat] = (b + 1) * 1000.0 + v * 100.0 + flat;
    }
}

static void test_axis(int axis, const char* name, const char* tc_name) {
    const int n_blocks = 2;
    const int b0 = 0, b1 = 1;

    // Allocate host buffer
    std::vector<double> h_Q((size_t)GPU_NVAR * n_blocks * GPU_NCELL, 0.0);
    fill_host(h_Q, b0, n_blocks);
    fill_host(h_Q, b1, n_blocks);

    // Upload to device
    double* d_Q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, h_Q.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Pair descriptor
    int h_pairs[2] = {b0, b1};
    int* d_pairs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pairs, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pairs, h_pairs, 2 * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Run ghost exchange
    if (axis == 0) gpu_ghost_exchange_x(d_Q, d_pairs, 1, n_blocks);
    if (axis == 1) gpu_ghost_exchange_y(d_Q, d_pairs, 1, n_blocks);
    if (axis == 2) gpu_ghost_exchange_z(d_Q, d_pairs, 1, n_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download result
    std::vector<double> h_out(h_Q.size());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_Q, h_out.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Verify: b0's right ghost layers should equal b1's left interior face
    double max_err = 0.0;
    for (int v = 0; v < GPU_NVAR; ++v) {
        const double* Q0 = h_out.data() + (size_t)v * n_blocks * GPU_NCELL
                                        + (size_t)b0 * GPU_NCELL;
        const double* Q1_orig = h_Q.data() + (size_t)v * n_blocks * GPU_NCELL
                                           + (size_t)b1 * GPU_NCELL;

        for (int a2 = 0; a2 < GPU_NB2; ++a2)
        for (int a1 = 0; a1 < GPU_NB2; ++a1)
        for (int g  = 0; g  < GPU_NG;   ++g) {
            int i, j, k, i_peer;
            // b0 right ghost: index depends on axis
            if (axis == 0) {
                i = GPU_NB + GPU_NG + g;  j = a1;  k = a2;
                i_peer = GPU_NG + g;      // b1 left interior
                double expected = Q1_orig[gpu_cell_idx(i_peer, j, k)];
                double got_val  = Q0[gpu_cell_idx(i, j, k)];
                max_err = fmax(max_err, fabs(got_val - expected));
            } else if (axis == 1) {
                i = a1;  j = GPU_NB + GPU_NG + g;  k = a2;
                int j_peer = GPU_NG + g;
                double expected = Q1_orig[gpu_cell_idx(i, j_peer, k)];
                double got_val  = Q0[gpu_cell_idx(i, j, k)];
                max_err = fmax(max_err, fabs(got_val - expected));
            } else {
                i = a1;  j = a2;  k = GPU_NB + GPU_NG + g;
                int k_peer = GPU_NG + g;
                double expected = Q1_orig[gpu_cell_idx(i, j, k_peer)];
                double got_val  = Q0[gpu_cell_idx(i, j, k)];
                max_err = fmax(max_err, fabs(got_val - expected));
            }
        }
    }

    char label[128];
    snprintf(label, sizeof(label), "%s b0 right ghost == b1 left face", name);
    check(label, max_err < 1e-14, max_err, 1e-14);

    // Also verify b1 left ghost equals b0 right interior face
    max_err = 0.0;
    for (int v = 0; v < GPU_NVAR; ++v) {
        const double* Q1 = h_out.data() + (size_t)v * n_blocks * GPU_NCELL
                                        + (size_t)b1 * GPU_NCELL;
        const double* Q0_orig = h_Q.data() + (size_t)v * n_blocks * GPU_NCELL
                                           + (size_t)b0 * GPU_NCELL;

        for (int a2 = 0; a2 < GPU_NB2; ++a2)
        for (int a1 = 0; a1 < GPU_NB2; ++a1)
        for (int g  = 0; g  < GPU_NG;   ++g) {
            if (axis == 0) {
                int i = GPU_NG - 1 - g;
                int i_peer = GPU_NB + g;  // b0 right interior
                double expected = Q0_orig[gpu_cell_idx(i_peer, a1, a2)];
                double got_val  = Q1[gpu_cell_idx(i, a1, a2)];
                max_err = fmax(max_err, fabs(got_val - expected));
            } else if (axis == 1) {
                int j = GPU_NG - 1 - g;
                int j_peer = GPU_NB + g;
                double expected = Q0_orig[gpu_cell_idx(a1, j_peer, a2)];
                double got_val  = Q1[gpu_cell_idx(a1, j, a2)];
                max_err = fmax(max_err, fabs(got_val - expected));
            } else {
                int k = GPU_NG - 1 - g;
                int k_peer = GPU_NB + g;
                double expected = Q0_orig[gpu_cell_idx(a1, a2, k_peer)];
                double got_val  = Q1[gpu_cell_idx(a1, a2, k)];
                max_err = fmax(max_err, fabs(got_val - expected));
            }
        }
    }
    char label2[128];
    snprintf(label2, sizeof(label2), "%s b1 left ghost == b0 right face", name);
    check(label2, max_err < 1e-14, max_err, 1e-14);

    cudaFree(d_Q);
    cudaFree(d_pairs);
}

int main() {
    printf("=== P4.3 gate: TBC / DSMEM ghost exchange ===\n");
    bool tbc = gpu_tbc_available();
    printf("  Hardware: sm_%d%d  TBC: %s\n\n",
           [](){ int m; cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor, 0); return m; }(),
           [](){ int m; cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMinor, 0); return m; }(),
           tbc ? "YES (H100+ path active)" : "NO  (global-memory fallback)");

    test_axis(0, "GC01 x-ghost-exchange", "GC01");
    test_axis(1, "GC02 y-ghost-exchange", "GC02");
    test_axis(2, "GC03 z-ghost-exchange", "GC03");

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) {
        if (tbc)
            printf("==> PASS  P4.3 gate cleared (TBC path: H100+ active)\n");
        else
            printf("==> PASS  P4.3 gate cleared (fallback path: sm_86, TBC activates on H100+)\n");
    } else {
        printf("==> FAIL  P4.3 gate NOT cleared\n");
    }
    return (n_fail == 0) ? 0 : 1;
}
