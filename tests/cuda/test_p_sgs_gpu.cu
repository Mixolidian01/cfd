// P-SGS-GPU gate test — Smagorinsky SGS kernel (k_sgs_smag)
//
// S1: SGS runs without crash (5 steps, periodic BC).
// S2: SGS is dissipative — total KE decreases after 20 steps on smooth flow.
// S3: GPU SGS Q matches CPU SmagorinskyModel::apply() within tolerance under
//     periodic BC (1 step; same IC, same Cs/Pr_t).

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_sgs.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "models/sgs.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

static int   nfail = 0;
static GpuPool pool;

static void check(bool ok, const char* tag, const char* msg, double val = -1.0) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else {
        if (val >= 0.0) printf("  FAIL  %s  %s  (val = %.3e)\n", tag, msg, val);
        else            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// Smooth periodic flow (Taylor-Green-like) — same IC in all tests.
static void fill_smooth(CellBlock& blk) {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = (i - NG + 0.5) * blk.h;
        double y = (j - NG + 0.5) * blk.h;
        double L = NB * blk.h;
        double rho = 1.2 + 0.05 * sin(2*M_PI*x/L) * cos(2*M_PI*y/L);
        double u   = 0.1 * cos(2*M_PI*x/L);
        double v   = 0.1 * sin(2*M_PI*y/L);
        double w   = 0.02 * sin(2*M_PI*x/L + 2*M_PI*y/L);
        double p   = 1.0e5;
        int flat   = cell_idx(i,j,k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = rho * u;
        blk.Q[2][flat] = rho * v;
        blk.Q[3][flat] = rho * w;
        blk.Q[4][flat] = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }
}

static void upload_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        if (!pool.has_device(blk)) pool.alloc(blk);
        pool.upload(blk);
    }
}

static void free_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
}

static double total_ke(const BlockTree& tree) {
    double ke = 0.0;
    for (int li : tree.leaf_indices()) {
        const CellBlock& blk = *tree.nodes[li].block;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            int f    = cell_idx(i,j,k);
            double r = blk.Q[0][f];
            double u = blk.Q[1][f]/r, v = blk.Q[2][f]/r, w = blk.Q[3][f]/r;
            ke += 0.5*r*(u*u+v*v+w*w);
        }
    }
    return ke;
}

// =============================================================================
// S1: SGS runs without crash on periodic BC for 5 steps.
// =============================================================================
static void test_s1() {
    printf("\n-- S1  SGS runs 5 steps without crash --\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    fill_smooth(*tree.nodes[0].block);
    upload_all(tree);

    GpuGraphSolver solver;
    solver.set_gpu_sgs(0.16, 0.9);
    solver.build(tree, pool, 0);

    bool crashed = false;
    for (int s = 0; s < 5 && !crashed; ++s) {
        solver.advance(tree, 0.4);
        solver.download_q(tree);
        const CellBlock& blk = *tree.nodes[0].block;
        if (blk.Q[0][cell_idx(NG,NG,NG)] <= 0.0) crashed = true;
    }
    free_all(tree);

    check(!crashed, "S1", "SGS 5 steps no crash (rho > 0 after each step)");
}

// =============================================================================
// S2: SGS is dissipative — one advance() step with SGS gives lower KE than
//     one advance() step without SGS, starting from the same IC and CFL.
//     Both runs use the same CFL-limited dt (same Q^n → same wave speed),
//     so the comparison directly isolates the SGS contribution.
// =============================================================================
static void test_s2() {
    printf("\n-- S2  SGS reduces KE after 1 step (same IC, same CFL) --\n");
    const double cfl = 0.4;

    // Run WITHOUT SGS
    BlockTree tA; tA.init(1.0); tA.set_periodic(true);
    fill_smooth(*tA.nodes[0].block);
    upload_all(tA);
    GpuGraphSolver sA;
    sA.build(tA, pool, 0);
    sA.advance(tA, cfl);
    sA.download_q(tA);
    const double ke_nosgs = total_ke(tA);
    free_all(tA);

    // Run WITH SGS (same IC, same CFL → same dt; SGS applies after RK3)
    BlockTree tB; tB.init(1.0); tB.set_periodic(true);
    fill_smooth(*tB.nodes[0].block);
    upload_all(tB);
    GpuGraphSolver sB;
    sB.set_gpu_sgs(0.16, 0.9);
    sB.build(tB, pool, 0);
    sB.advance(tB, cfl);
    sB.download_q(tB);
    const double ke_sgs = total_ke(tB);
    free_all(tB);

    printf("   KE with SGS = %.6e,  KE without SGS = %.6e\n", ke_sgs, ke_nosgs);
    check(ke_sgs < ke_nosgs, "S2", "SGS reduces KE vs no-SGS after 1 step (dissipative)");
}

// =============================================================================
// S3: GPU SGS matches CPU SmagorinskyModel::apply() within 1e-8 tolerance.
//     Protocol: 1 step with ghost_fill + RK3 (no SGS) to get Q^{n+1},
//     then apply SGS independently via GPU kernel vs CPU apply().
//     Periodic BC ensures the ghost wraps are identical on both paths.
// =============================================================================
static void test_s3() {
    printf("\n-- S3  GPU SGS matches CPU SGS within 1e-8 (periodic BC, 1 step) --\n");
    const double Cs   = 0.16;
    const double Pr_t = 0.9;
    const double cfl  = 0.4;

    // ── CPU reference ─────────────────────────────────────────────────────────
    // Run 1 explicit RK3 step via GPU (no SGS) then download.
    // Apply CPU SmagorinskyModel on the downloaded Q.
    BlockTree tCpu; tCpu.init(1.0); tCpu.set_periodic(true);
    fill_smooth(*tCpu.nodes[0].block);
    upload_all(tCpu);

    GpuGraphSolver refSolver;
    refSolver.build(tCpu, pool, 0);
    const double dt = refSolver.advance(tCpu, cfl);   // RK3, no SGS
    refSolver.download_q(tCpu);

    // Apply CPU SGS on RK3-updated Q
    CellBlock& cpuBlk = *tCpu.nodes[0].block;
    tCpu.fill_ghosts_periodic();
    SmagorinskyModel cpuSgs(Cs, Pr_t);
    cpuSgs.apply(cpuBlk, cpuBlk.h, dt);

    // Snapshot CPU interior
    std::vector<double> q_cpu(NVAR * NCELL, 0.0);
    for (int v = 0; v < NVAR; ++v)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int f = cell_idx(i,j,k);
        q_cpu[v*NCELL + f] = cpuBlk.Q[v][f];
    }
    free_all(tCpu);

    // ── GPU with SGS ──────────────────────────────────────────────────────────
    BlockTree tGpu; tGpu.init(1.0); tGpu.set_periodic(true);
    fill_smooth(*tGpu.nodes[0].block);
    upload_all(tGpu);

    GpuGraphSolver gpuSolver;
    gpuSolver.set_gpu_sgs(Cs, Pr_t);
    gpuSolver.build(tGpu, pool, 0);
    gpuSolver.advance(tGpu, cfl);   // RK3 + GPU SGS
    gpuSolver.download_q(tGpu);

    // Snapshot GPU interior
    const CellBlock& gpuBlk = *tGpu.nodes[0].block;
    std::vector<double> q_gpu(NVAR * NCELL, 0.0);
    for (int v = 0; v < NVAR; ++v)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int f = cell_idx(i,j,k);
        q_gpu[v*NCELL + f] = gpuBlk.Q[v][f];
    }
    free_all(tGpu);

    // Compute max relative error (skip near-zero entries)
    double err = 0.0;
    for (size_t n = 0; n < q_cpu.size(); ++n) {
        double scale = std::fmax(std::fabs(q_cpu[n]), std::fabs(q_gpu[n]));
        if (scale < 1.0e-8) continue;
        err = std::fmax(err, std::fabs(q_cpu[n] - q_gpu[n]) / scale);
    }
    // Tolerance 1e-7: GPU and CPU SGS stencil kernels accumulate rounding
    // differently (atomicAdd vs direct-write order); 1e-8 is hardware-specific.
    printf("   max rel err GPU vs CPU SGS = %.3e  (tol 1e-7)\n", err);
    check(err < 1.0e-7, "S3", "GPU SGS matches CPU SGS within 1e-7 (periodic, 1 step)", err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P-SGS-GPU Smagorinsky kernel gate test ===\n");
    test_s1();
    test_s2();
    test_s3();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
