// bench_cpu_vs_gpu_tgv — CPU vs GPU throughput on the TGV benchmark IC
//
// Runs N_WARM warm-up steps then N_TIMED timed steps on both paths:
//   CPU: NSSolver (single rank, SSP-RK3 + WENO5-Z)
//   GPU: GpuGraphSolver (CUDA Graph SSP-RK3 + TENO5-A)
//
// Grid: 3-level uniform refinement of [0, 2π]³ → 512 leaves (64³ = 262 144 cells)
// IC:   Taylor-Green vortex (bench_b3), Ma=0.1, CFL=0.40
//
// Output: ms/step, MCELL/s, and GPU/CPU speedup ratio.
//         Projection table for 1–16 MPI ranks (ideal linear scaling).
//
// Build: cmake --build build -t bench_tgv
// Run:   ./build/bench_tgv

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_check.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "solver/ns_solver.hpp"
#include "schemes/operators.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

static constexpr int    N_WARM  = 5;
static constexpr int    N_TIMED = 50;
static constexpr double CFL     = 0.40;
static constexpr double MA      = 0.10;
static constexpr double V0      = 1.0;
static constexpr double RHO0    = 1.0;
static constexpr double P0_COEF = RHO0 * V0 * V0 / (GAMMA * MA * MA);  // ≈71.43

static GpuPool pool;

// ── TGV IC (bench_b3 / HiOCFD4 C3.3) ────────────────────────────────────────
static Prim tgv_ic(double x, double y, double z) {
    const double p0 = P0_COEF;
    Prim q{};
    q.rho =  RHO0;
    q.u   =  V0 * std::sin(x) * std::cos(y) * std::cos(z);
    q.v   = -V0 * std::cos(x) * std::sin(y) * std::cos(z);
    q.w   =  0.0;
    q.p   = p0 + RHO0 * V0 * V0 / 16.0
               * (std::cos(2.0*x) + std::cos(2.0*y))
               * (std::cos(2.0*z) + 2.0);
    q.T   = q.p / (q.rho * R_GAS);
    q.c   = std::sqrt(GAMMA * q.p / q.rho);
    return q;
}

// ── Fill all leaves on a tree with tgv_ic ────────────────────────────────────
static void fill_leaves(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock& blk = *tree.nodes[li].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = blk.ox + (i - NG + 0.5) * blk.h;
            double y = blk.oy + (j - NG + 0.5) * blk.h;
            double z = blk.oz + (k - NG + 0.5) * blk.h;
            Prim p   = tgv_ic(x, y, z);
            int  idx = cell_idx(i, j, k);
            blk.Q[0][idx] = p.rho;
            blk.Q[1][idx] = p.rho * p.u;
            blk.Q[2][idx] = p.rho * p.v;
            blk.Q[3][idx] = p.rho * p.w;
            blk.Q[4][idx] = p.p / (GAMMA - 1.0)
                           + 0.5 * p.rho * (p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
}

// ── Build 3-level uniform periodic tree → 512 leaves (64³) ──────────────────
static void build_tgv_tree(BlockTree& tree) {
    const double L = 2.0 * M_PI;
    tree.init(L);
    tree.set_periodic(true);
    for (int lvl = 0; lvl < 3; ++lvl) {
        std::vector<int> cur(tree.leaf_indices().begin(), tree.leaf_indices().end());
        for (int idx : cur) tree.refine(idx);
    }
    tree.rebuild_neighbours();
    fill_leaves(tree);
}

// ── CPU bench ────────────────────────────────────────────────────────────────
static double bench_cpu() {
    printf("── CPU path (single rank, SSP-RK3 + WENO5-Z) ──\n");

    NSSolver s;
    s.cfg.bc.variant           = PeriodicBC{};
    s.cfg.time.cfl             = CFL;
    s.cfg.amr.regrid_interval  = 0;
    s.cfg.amr.max_level        = 0;
    s.cfg.io.verbose           = false;
    s.init(2.0 * M_PI, tgv_ic);

    // 3-level uniform refinement → 512 leaves
    for (int lvl = 0; lvl < 3; ++lvl) {
        std::vector<int> cur(s.tree.leaf_indices().begin(), s.tree.leaf_indices().end());
        for (int li : cur) s.tree.refine(li);
        s.tree.rebuild_neighbours();
    }
    fill_leaves(s.tree);
    s.alloc_scratch();

    // Warm-up
    for (int i = 0; i < N_WARM; ++i) s.advance();

    // Timed run
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N_TIMED; ++i) s.advance();
    auto t1 = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_per_step = elapsed_ms / N_TIMED;
    const int    n_leaves    = (int)s.tree.leaf_indices().size();
    const int    n_cells     = n_leaves * NB * NB * NB;
    const double mcell_per_s = (double)n_cells / ms_per_step / 1.0e3;

    printf("  leaves=%d  cells=%d  h=%.5f\n",
           n_leaves, n_cells, s.tree.nodes[s.tree.leaf_indices()[0]].block->h);
    printf("  %d-step elapsed:  %.2f s\n", N_TIMED, elapsed_ms * 1e-3);
    printf("  ms / step:       %.2f\n",   ms_per_step);
    printf("  MCELL / s:       %.2f\n",   mcell_per_s);
    return ms_per_step;
}

// ── GPU bench ────────────────────────────────────────────────────────────────
static double bench_gpu() {
    printf("── GPU path (TENO5-A, CUDA Graph SSP-RK3) ──\n");

    BlockTree tree;
    build_tgv_tree(tree);

    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!pool.has_device(blk)) pool.alloc(blk);
        pool.upload(blk);
    }

    GpuGraphSolver solver;
    // TENO5-A is the production default (set in GpuRhsList); no override needed.
    solver.build(tree, pool);

    // Warm-up (first advance() triggers CUDA Graph capture)
    for (int i = 0; i < N_WARM; ++i) solver.advance(tree, CFL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run via CUDA events (includes full advance: ghost fill + CFL + 3-stage RK3)
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < N_TIMED; ++i) solver.advance(tree, CFL);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float elapsed_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    const double elapsed_ms  = (double)elapsed_ms_f;
    const double ms_per_step = elapsed_ms / N_TIMED;
    const int    n_leaves    = (int)tree.leaf_indices().size();
    const int    n_cells     = n_leaves * NB * NB * NB;
    const double mcell_per_s = (double)n_cells / ms_per_step / 1.0e3;

    printf("  leaves=%d  cells=%d  h=%.5f\n",
           n_leaves, n_cells, tree.nodes[tree.leaf_indices()[0]].block->h);
    printf("  %d-step elapsed:  %.3f s\n", N_TIMED, elapsed_ms * 1e-3);
    printf("  ms / step:       %.3f\n",   ms_per_step);
    printf("  MCELL / s:       %.2f\n",   mcell_per_s);

    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
    return ms_per_step;
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== TGV CPU vs GPU Throughput  (%d warm-up + %d timed steps, CFL=%.2f) ===\n",
           N_WARM, N_TIMED, CFL);
    printf("GPU: %s\n\n", prop.name);

    const double ms_cpu = bench_cpu();
    printf("\n");
    const double ms_gpu = bench_gpu();
    printf("\n");

    // ── Speedup summary ──────────────────────────────────────────────────────
    const double speedup = ms_cpu / ms_gpu;
    printf("── Comparison ──\n");
    printf("  CPU ms/step:  %.2f\n",   ms_cpu);
    printf("  GPU ms/step:  %.3f\n",   ms_gpu);
    printf("  GPU speedup over CPU (1 rank):  %.1f×\n\n", speedup);

    // ── MPI scaling projection (ideal linear scaling) ─────────────────────
    printf("── MPI scaling projection (ideal linear, halo overhead not modelled) ──\n");
    printf("  %-8s  %-16s  %-12s\n", "Ranks", "CPU ms/step", "GPU speedup");
    static const int ranks[] = {1, 2, 4, 8, 16, 32};
    for (int r : ranks) {
        const double ms_r = ms_cpu / r;
        printf("  %-8d  %-16.2f  %.1f×\n", r, ms_r, ms_r / ms_gpu);
    }
    printf("  (real MPI efficiency < 100%% at 512/%d leaves/rank;\n"
           "   halo exchange + load imbalance shrink the gap)\n", ranks[5]);

    return 0;
}
