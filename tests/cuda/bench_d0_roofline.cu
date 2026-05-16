// D0 roofline baseline benchmark for k_rhs_conv
//
// Creates a 512-leaf flat periodic tree (3 levels of uniform refinement),
// warms up for 5 steps, then times 20 SSP-RK3 steps with CUDA events.
//
// Outputs: ms/call, achieved BW (GB/s), % of peak BW.
// Peak BW queried from CUDA device attributes at runtime.
//
// Usage:
//   ./build/bench_d0_roofline
//
// Note: ncu hardware-counter BW% requires elevated permissions on WSL2.
//   To unlock:
//     echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
//     sudo tee /etc/modprobe.d/nvidia-profiling.conf
//   Then: sudo modprobe -r nvidia && sudo modprobe nvidia
//   Then: ncu --set full --kernel-name k_rhs_conv ./build/bench_d0_roofline

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_check.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "schemes/operators.hpp"
#include <cstdio>
#include <cuda_runtime.h>

static GpuPool pool;

static void fill_uniform(BlockTree& tree) {
    const double rho = 1.225, u = 34.72, p = 101325.0;
    const double e = p / ((GAMMA - 1.0) * rho);
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        for (int idx = 0; idx < NB2*NB2*NB2; ++idx) {
            blk->Q[0][idx] = rho;
            blk->Q[1][idx] = rho * u;
            blk->Q[2][idx] = 0.0;
            blk->Q[3][idx] = 0.0;
            blk->Q[4][idx] = rho * (e + 0.5 * u * u);
        }
        if (!pool.has_device(blk)) pool.alloc(blk);
        pool.upload(blk);
    }
}

int main() {
    // Peak BW from device attributes (memoryClockRate in kHz, bus in bits)
    int mem_clock_khz = 0, bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz,  cudaDevAttrMemoryClockRate,   0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_bw_gbs = (double)mem_clock_khz * 1e3  // Hz
                       * 2.0                           // DDR
                       * (double)bus_width_bits / 8.0  // bytes/cycle
                       / 1e9;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== D0 k_rhs_conv Roofline Baseline ===\n");
    printf("GPU:          %s\n", prop.name);
    printf("Peak mem BW:  %.1f GB/s  (mem_clock=%d kHz  bus=%d bit)\n",
           peak_bw_gbs, mem_clock_khz, bus_width_bits);
    printf("\n");

    // Build 512-leaf flat tree: 3 uniform refinement levels (8^3 leaves)
    // Needs ~1000+ leaves for adequate GPU occupancy (46 SMs × 22+ blocks/SM)
    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic(true);
    // Level 0→1: 1 root → 8 leaves
    tree.refine(0);
    // Level 1→2: 8 → 64 leaves
    {
        std::vector<int> lvl1(tree.leaf_indices().begin(), tree.leaf_indices().end());
        for (int idx : lvl1) tree.refine(idx);
    }
    // Level 2→3: 64 → 512 leaves
    {
        std::vector<int> lvl2(tree.leaf_indices().begin(), tree.leaf_indices().end());
        for (int idx : lvl2) tree.refine(idx);
    }
    int n_leaves = (int)tree.leaf_indices().size();
    printf("Grid:         %d leaves  (NB=%d  NG=%d  NB2=%d)\n",
           n_leaves, NB, NG, NB2);

    fill_uniform(tree);

    // Build GPU solver (periodic BC, no MPI)
    GpuGraphSolver solver;
    solver.build(tree, pool);

    // Bytes per k_rhs_conv call:
    //   Read  Q[NVAR][NB2^3]:  5 * 1728 * 8 bytes per leaf
    //   Write rhs[NVAR][NB^3]: 5 *  512 * 8 bytes per leaf
    const double bytes_per_leaf = (double)NVAR * (NB2*NB2*NB2 + NB*NB*NB) * sizeof(double);
    const double bytes_per_call = (double)n_leaves * bytes_per_leaf;
    printf("Data/call:    %.2f MB  (read+write, no cache reuse assumed)\n",
           bytes_per_call / 1e6);
    printf("\n");

    // Warm up: 5 advance steps (discarded)
    for (int i = 0; i < 5; ++i) solver.advance(tree, 0.4);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run: 20 SSP-RK3 steps → 60 k_rhs_conv invocations
    const int N_STEPS = 20;
    const int N_RHS_PER_STEP = 3;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < N_STEPS; ++i) solver.advance(tree, 0.4);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev0, ev1));

    const int n_calls = N_STEPS * N_RHS_PER_STEP;
    const double ms_per_call  = elapsed_ms / n_calls;
    const double achieved_bw  = bytes_per_call / (ms_per_call * 1e-3) / 1e9;
    const double pct_of_peak  = achieved_bw / peak_bw_gbs * 100.0;

    printf("=== Results ===\n");
    printf("Steps timed:      %d  (%d k_rhs_conv calls)\n", N_STEPS, n_calls);
    printf("Total elapsed:    %.2f ms\n", elapsed_ms);
    printf("ms / k_rhs_conv:  %.3f ms\n", ms_per_call);
    printf("Achieved BW:      %.1f GB/s\n", achieved_bw);
    printf("Pct of peak:      %.1f%%\n", pct_of_peak);

    if (pct_of_peak >= 55.0) {
        printf("\nSTATUS: PASS (>= 55%% — D0.5 target already met)\n");
    } else {
        printf("\nSTATUS: BASELINE (%.1f%% < 55%% — D0.5 shared-mem tiling required)\n",
               pct_of_peak);
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return 0;
}
