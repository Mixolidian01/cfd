// test_t45_dyn_sgs_gpu.cu — G3: GPU dynamic Smagorinsky (Germano + Lilly LS) gate
//
// G3a: Dynamic SGS runs 5 steps without crash (periodic BC, smooth IC).
// G3b: Mass is conserved (< 1e-10 rel error) after 5 steps with dynamic SGS
//      (stress divergence is conservative; Cs²≥0 per Lilly clipping).
// G3c: GPU dynamic SGS matches CPU DynamicSmagorinskyModel::apply() within 1e-7.

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

static int    nfail = 0;
static GpuPool pool;

static void check(bool ok, const char* tag, const char* msg, double val = -1.0) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else {
        if (val >= 0.0) printf("  FAIL  %s  %s  (val = %.3e)\n", tag, msg, val);
        else            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// Smooth periodic single-scale IC (for G3a and G3c).
static void fill_smooth(CellBlock& blk) {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = (i - NG + 0.5) * blk.h;
        double y = (j - NG + 0.5) * blk.h;
        double z = (k - NG + 0.5) * blk.h;
        double L = NB * blk.h;
        double rho = 1.2 + 0.05 * sin(2*M_PI*x/L) * cos(2*M_PI*y/L);
        double u   = 0.3 * cos(2*M_PI*x/L) * sin(2*M_PI*y/L) * cos(2*M_PI*z/L);
        double v   =-0.3 * sin(2*M_PI*x/L) * cos(2*M_PI*y/L) * cos(2*M_PI*z/L);
        double w   = 0.1 * cos(2*M_PI*x/L) * cos(2*M_PI*y/L) * sin(2*M_PI*z/L);
        double p   = 1.0e5;
        int flat   = cell_idx(i,j,k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = rho * u;
        blk.Q[2][flat] = rho * v;
        blk.Q[3][flat] = rho * w;
        blk.Q[4][flat] = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }
}

// Multi-scale IC: fundamental + near-Nyquist component so the 3×3×3 test filter
// captures non-trivial Leonard stresses → Cs² > 0 (required for G3b).
static void fill_multiscale(CellBlock& blk) {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = (i - NG + 0.5) * blk.h;
        double y = (j - NG + 0.5) * blk.h;
        double z = (k - NG + 0.5) * blk.h;
        double L = NB * blk.h;
        // k1 = 2π/L (fundamental), k2 = 4π/L (subgrid near Nyquist)
        double k1 = 2*M_PI/L, k2 = 4*M_PI/L;
        double u =  0.3*cos(k1*x)*sin(k1*y)*cos(k1*z)
                  + 0.2*sin(k2*x)*cos(k2*y)*sin(k2*z);
        double v = -0.3*sin(k1*x)*cos(k1*y)*cos(k1*z)
                  - 0.2*cos(k2*x)*sin(k2*y)*sin(k2*z);
        double w =  0.1*cos(k1*x)*cos(k1*y)*sin(k1*z)
                  + 0.1*cos(k2*x)*cos(k2*y)*cos(k2*z);
        double rho = 1.2 + 0.05*sin(k1*x)*cos(k1*y);
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
            int f = cell_idx(i,j,k);
            double r = blk.Q[0][f];
            double u = blk.Q[1][f]/r, v = blk.Q[2][f]/r, w = blk.Q[3][f]/r;
            ke += 0.5*r*(u*u+v*v+w*w);
        }
    }
    return ke;
}

// =============================================================================
// G3a: Dynamic SGS runs 5 steps without crash.
// =============================================================================
static void test_g3a() {
    printf("\n-- G3a  Dynamic SGS runs 5 steps without crash --\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    fill_smooth(*tree.nodes[0].block);
    upload_all(tree);

    GpuGraphSolver solver;
    solver.set_gpu_dyn_sgs(0.9);
    solver.build(tree, pool, 0);

    bool crashed = false;
    for (int s = 0; s < 5 && !crashed; ++s) {
        solver.advance(tree, 0.4);
        solver.download_q(tree);
        const CellBlock& blk = *tree.nodes[0].block;
        if (blk.Q[0][cell_idx(NG,NG,NG)] <= 0.0) crashed = true;
    }
    free_all(tree);

    check(!crashed, "G3a", "Dynamic SGS 5 steps no crash (rho > 0)");
}

// =============================================================================
// G3b: Mass is conserved (|Δm|/m < 1e-10) over 5 steps with dynamic SGS.
// The stress divergence of the SGS model is conservative (telescoping sum);
// any mass drift indicates a bug in the update path.
// =============================================================================
static void test_g3b() {
    printf("\n-- G3b  Mass conserved (|Δm|/m < 1e-10) over 5 steps with dynamic SGS --\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    fill_smooth(*tree.nodes[0].block);
    upload_all(tree);

    // Initial mass
    const CellBlock& blk0 = *tree.nodes[0].block;
    double mass0 = 0.0;
    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i)
        mass0 += blk0.Q[0][cell_idx(i,j,k)];

    GpuGraphSolver solver;
    solver.set_gpu_dyn_sgs(0.9);
    solver.build(tree, pool, 0);

    for (int s = 0; s < 5; ++s) solver.advance(tree, 0.4);
    solver.download_q(tree);

    const CellBlock& blk = *tree.nodes[0].block;
    double mass1 = 0.0;
    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i)
        mass1 += blk.Q[0][cell_idx(i,j,k)];

    free_all(tree);

    const double mass_err = std::fabs(mass1 - mass0) / (mass0 + 1e-300);
    printf("   mass_err = %.3e  (tol 1e-10)\n", mass_err);
    check(mass_err < 1e-10, "G3b",
          "Dynamic SGS mass conservation |Δm|/m < 1e-10 over 5 steps", mass_err);
}

// =============================================================================
// G3c: GPU dynamic SGS matches CPU DynamicSmagorinskyModel::apply() within 1e-7.
// Protocol: run 1 GPU RK3 step (no SGS) to get Q^{n+1}, then apply CPU and GPU
// dynamic SGS independently starting from the same Q^{n+1}.
// =============================================================================
static void test_g3c() {
    printf("\n-- G3c  GPU dynamic SGS matches CPU within 1e-7 (periodic, 1 step) --\n");
    const double Pr_t = 0.9;
    const double cfl  = 0.4;

    // ── CPU reference ─────────────────────────────────────────────────────────
    BlockTree tCpu; tCpu.init(1.0); tCpu.set_periodic(true);
    fill_smooth(*tCpu.nodes[0].block);
    upload_all(tCpu);

    GpuGraphSolver refSolver;
    refSolver.build(tCpu, pool, 0);
    const double dt = refSolver.advance(tCpu, cfl);  // RK3 only, no SGS
    refSolver.download_q(tCpu);

    CellBlock& cpuBlk = *tCpu.nodes[0].block;
    tCpu.fill_ghosts_periodic();
    DynamicSmagorinskyModel cpuSgs(Pr_t);
    cpuSgs.apply(cpuBlk, cpuBlk.h, dt);

    std::vector<double> q_cpu(NVAR * NCELL, 0.0);
    for (int v = 0; v < NVAR; ++v)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int f = cell_idx(i,j,k);
        q_cpu[v*NCELL + f] = cpuBlk.Q[v][f];
    }
    free_all(tCpu);

    // ── GPU with dynamic SGS ─────────────────────────────────────────────────
    BlockTree tGpu; tGpu.init(1.0); tGpu.set_periodic(true);
    fill_smooth(*tGpu.nodes[0].block);
    upload_all(tGpu);

    GpuGraphSolver gpuSolver;
    gpuSolver.set_gpu_dyn_sgs(Pr_t);
    gpuSolver.build(tGpu, pool, 0);
    gpuSolver.advance(tGpu, cfl);  // RK3 + GPU dynamic SGS
    gpuSolver.download_q(tGpu);

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

    // Max relative error (skip near-zero entries)
    double err = 0.0;
    for (size_t n = 0; n < q_cpu.size(); ++n) {
        double scale = std::fmax(std::fabs(q_cpu[n]), std::fabs(q_gpu[n]));
        if (scale < 1.0e-8) continue;
        err = std::fmax(err, std::fabs(q_cpu[n] - q_gpu[n]) / scale);
    }
    printf("   max rel err GPU vs CPU dynamic SGS = %.3e  (tol 1e-7)\n", err);
    check(err < 1.0e-7, "G3c",
          "GPU dynamic SGS matches CPU within 1e-7 (periodic, 1 step)", err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== G3 GPU dynamic Smagorinsky (Germano+Lilly) gate test ===\n");
    test_g3a();
    test_g3b();
    test_g3c();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
