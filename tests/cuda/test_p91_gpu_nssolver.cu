// P9.1 gate test — GPU GpuGraphSolver vs CPU NSSolver on flat periodic tree
//
// Validates that the GPU solver (GpuGraphSolver) produces the same conserved
// state Q as the CPU NSSolver for a flat (no AMR) periodic tree.  Both run
// the same Sod shock-tube IC; the GPU uses WENO5-Z + HLLC-ES exactly as the
// CPU path does, so results should be bitwise-close within machine epsilon.
//
// Protocol:
//   N1: 4-step GPU Q == 4-step CPU Q  (max_phys_err < 1e-8)
//   N2: GPU dt matches CPU dt per step (max relative error < 1e-10)
//   N3: 20-step GPU Q == 20-step CPU Q (max_phys_err < 1e-8)
//   N4: GPU mass conserved to 1e-8 over 20 steps (same tolerance as CPU)

#include "../../include/cuda/gpu_graph.cuh"
#include "../../include/cuda/gpu_check.cuh"
#include "../../include/gpu_pool.hpp"
#include "../../include/block_tree.hpp"
#include "../../include/cell_block.hpp"
#include "../../include/ns_solver.hpp"
#include "../../include/operators.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

static int nfail = 0;

static void check(bool ok, const char* tag, const char* msg, double val = -1.0) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else {
        if (val >= 0.0) printf("  FAIL  %s  %s  (val = %.3e)\n", tag, msg, val);
        else            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// ── Sod shock-tube IC (1D in x, uniform in y/z) ──────────────────────────────
static Prim sod_ic(double x, double /*y*/, double /*z*/) {
    Prim p{};
    bool left = x < 0.5;
    p.rho = left ? 1.0 : 0.125;
    p.u   = 0.0; p.v = 0.0; p.w = 0.0;
    p.p   = left ? 1.0e5 : 1.0e4;
    return p;
}

// ── Per-block helpers ─────────────────────────────────────────────────────────
static GpuPool pool;

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

// Save interior Q to flat buffer
using QSnap = std::vector<double>;

static QSnap snap(const BlockTree& tree) {
    const auto& lv = tree.leaf_indices();
    QSnap q(lv.size() * NVAR * NCELL, 0.0);
    for (int li = 0; li < (int)lv.size(); ++li) {
        const CellBlock& blk = *tree.nodes[lv[li]].block;
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            int flat = cell_idx(i,j,k);
            q[(size_t)li*(NVAR*NCELL) + v*NCELL + flat] = blk.Q[v][flat];
        }
    }
    return q;
}

static double max_phys_err(const QSnap& a, const QSnap& b,
                           double abs_floor = 1.0e-8) {
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double scale = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
        if (scale < abs_floor) continue;
        double d = std::fabs(a[i] - b[i]);
        err = std::fmax(err, d / scale);
    }
    return err;
}

static double total_mass(const BlockTree& tree) {
    double m = 0.0;
    for (int li : tree.leaf_indices()) {
        const CellBlock& blk = *tree.nodes[li].block;
        const double h3 = blk.h * blk.h * blk.h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            m += blk.Q[0][cell_idx(i,j,k)] * h3;
    }
    return m;
}

// =============================================================================
// run_cpu: N steps via NSSolver (CPU SSP-RK3)
// =============================================================================
static QSnap run_cpu(int nstep, double cfl, std::vector<double>* dts = nullptr) {
    NSSolver solver;
    SolverConfig cfg;
    cfg.time.cfl            = cfl;
    cfg.bc.variant = PeriodicBC{};
    cfg.amr.regrid_interval = 0;
    cfg.io.verbose        = false;
    solver.cfg = cfg;
    solver.init(1.0, sod_ic);

    if (dts) dts->resize(nstep);
    for (int s = 0; s < nstep; ++s) {
        double dt = solver.advance();
        if (dts) (*dts)[s] = dt;
    }
    return snap(solver.tree);
}

// =============================================================================
// run_gpu: N steps via GpuGraphSolver
// =============================================================================
static QSnap run_gpu(int nstep, double cfl, std::vector<double>* dts = nullptr) {
    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    // IC
    {
        CellBlock& blk = *tree.nodes[0].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = (i - NG + 0.5) * blk.h;
            Prim p = sod_ic(x, 0.0, 0.0);
            int flat = cell_idx(i,j,k);
            blk.Q[0][flat] = p.rho;
            blk.Q[1][flat] = p.rho * p.u;
            blk.Q[2][flat] = p.rho * p.v;
            blk.Q[3][flat] = p.rho * p.w;
            blk.Q[4][flat] = p.p / (GAMMA - 1.0) + 0.5*p.rho*(p.u*p.u+p.v*p.v+p.w*p.w);
        }
    }
    upload_all(tree);

    GpuGraphSolver solver;
    solver.build(tree, pool);

    if (dts) dts->resize(nstep);
    for (int s = 0; s < nstep; ++s) {
        double dt = solver.advance(tree, cfl);
        if (dts) (*dts)[s] = dt;
    }
    solver.download_q(tree);
    auto q = snap(tree);
    free_all(tree);
    return q;
}

// =============================================================================
// N1 + N2: 4-step GPU == CPU (Q and dt)
// =============================================================================
static void test_n1_n2() {
    printf("\n-- N1/N2  4-step GPU Q == CPU Q and dt  (tol 1e-8 / 1e-10) --\n");
    const double cfl   = 0.4;
    const int    NSTEP = 4;

    std::vector<double> dt_cpu, dt_gpu;
    const QSnap qcpu = run_cpu(NSTEP, cfl, &dt_cpu);
    const QSnap qgpu = run_gpu(NSTEP, cfl, &dt_gpu);

    const double qerr = max_phys_err(qgpu, qcpu);
    printf("   max phys_err (GPU vs CPU, 4 steps) = %.3e  (tol 1e-8)\n", qerr);
    check(qerr < 1.0e-8, "N1", "GPU Q == CPU Q (4 steps, tol 1e-8)", qerr);

    double dt_err = 0.0;
    for (int s = 0; s < NSTEP; ++s) {
        double d = std::fabs(dt_gpu[s] - dt_cpu[s]);
        double scale = std::fmax(dt_cpu[s], 1.0e-16);
        dt_err = std::fmax(dt_err, d / scale);
    }
    printf("   max rel dt error (GPU vs CPU, 4 steps) = %.3e  (tol 1e-10)\n", dt_err);
    check(dt_err < 1.0e-10, "N2", "GPU dt == CPU dt (4 steps, rel tol 1e-10)", dt_err);
}

// =============================================================================
// N3: 20-step GPU == CPU (Q)
// =============================================================================
static void test_n3() {
    printf("\n-- N3  20-step GPU Q == CPU Q  (tol 1e-8) --\n");
    const double cfl   = 0.3;
    const int    NSTEP = 20;

    const QSnap qcpu = run_cpu(NSTEP, cfl);
    const QSnap qgpu = run_gpu(NSTEP, cfl);

    const double err = max_phys_err(qgpu, qcpu);
    printf("   max phys_err (GPU vs CPU, 20 steps) = %.3e  (tol 1e-8)\n", err);
    check(err < 1.0e-8, "N3", "GPU Q == CPU Q (20 steps, tol 1e-8)", err);
}

// =============================================================================
// N4: GPU mass conserved over 20 steps
// =============================================================================
static void test_n4() {
    printf("\n-- N4  GPU mass conserved over 20 steps  (tol 1e-8) --\n");
    const double cfl   = 0.3;
    const int    NSTEP = 20;

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    {
        CellBlock& blk = *tree.nodes[0].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = (i - NG + 0.5) * blk.h;
            Prim p = sod_ic(x, 0.0, 0.0);
            int flat = cell_idx(i,j,k);
            blk.Q[0][flat] = p.rho;
            blk.Q[1][flat] = p.rho * p.u;
            blk.Q[2][flat] = p.rho * p.v;
            blk.Q[3][flat] = p.rho * p.w;
            blk.Q[4][flat] = p.p / (GAMMA - 1.0) + 0.5*p.rho*(p.u*p.u+p.v*p.v+p.w*p.w);
        }
    }
    upload_all(tree);
    const double m0 = total_mass(tree);

    GpuGraphSolver solver;
    solver.build(tree, pool);
    for (int s = 0; s < NSTEP; ++s)
        solver.advance(tree, cfl);
    solver.download_q(tree);

    const double mf = total_mass(tree);
    const double rel = std::fabs(mf - m0) / std::fabs(m0);
    printf("   mass rel error over 20 GPU steps = %.3e  (tol 1e-8)\n", rel);
    check(rel < 1.0e-8, "N4", "GPU mass conserved over 20 steps (tol 1e-8)", rel);

    free_all(tree);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P9.1 GPU GpuGraphSolver vs CPU NSSolver gate test ===\n");
    test_n1_n2();
    test_n3();
    test_n4();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
