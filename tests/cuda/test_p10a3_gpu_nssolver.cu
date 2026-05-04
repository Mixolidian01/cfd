// P10-A3 gate test — NSSolver::advance() GPU dispatch via set_gpu_solver()
//
// Validates that NSSolver with an injected GpuGraphSolver (via set_gpu_solver())
// produces the same Q and dt as the CPU NSSolver path.
//
// Protocol:
//   A1: NSSolver GPU dispatch Q == CPU NSSolver Q (20 steps, tol 1e-8)
//   A2: NSSolver GPU dispatch dt == CPU dt (relative tol 1e-10)
//   A3: NSSolver GPU mass conservation over 20 steps (tol 1e-8)

#include "../../include/cuda/gpu_graph.cuh"
#include "../../include/cuda/gpu_check.cuh"
#include "../../include/gpu_pool.hpp"
#include "../../include/block_tree.hpp"
#include "../../include/cell_block.hpp"
#include "../../include/ns_solver.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
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

static Prim flat_ic(double, double, double) {
    return {1.0, 0.1, 0.05, 0.0, 1.0e5};
}

// Q snapshot: interior cells only, all leaves, all vars (for comparison)
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
            int idx  = li * NVAR * NCELL + v * NCELL + flat;
            q[idx]   = blk.Q[v][flat];
        }
    }
    return q;
}

// Relative error with absolute floor: skip cells where both values are below
// 1e-8 (avoids false positives when a variable is analytically zero but has
// machine-epsilon noise). Matches P9.1's max_phys_err convention.
static double max_rel_err(const QSnap& a, const QSnap& b,
                          double abs_floor = 1.0e-8) {
    assert(a.size() == b.size());
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double scale = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
        if (scale < abs_floor) continue;
        err = std::fmax(err, std::fabs(a[i] - b[i]) / scale);
    }
    return err;
}

static double total_mass(const BlockTree& tree) {
    double m = 0.0;
    for (int li : tree.leaf_indices()) {
        const CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            m += blk->Q[0][cell_idx(i,j,k)] * (blk->h * blk->h * blk->h);
    }
    return m;
}

static SolverConfig make_cfg(double cfl) {
    SolverConfig cfg;
    cfg.cfl             = cfl;
    cfg.bc              = BCType::Periodic;
    cfg.regrid_interval = 0;
    cfg.verbose         = false;
    return cfg;
}

// =============================================================================
// A1/A2: GPU-dispatched NSSolver Q and dt == CPU NSSolver
// =============================================================================
static void test_a1_a2() {
    printf("\n-- A1/A2  NSSolver GPU dispatch Q and dt == CPU NSSolver (20 steps) --\n");
    const int    nstep = 20;
    const double cfl   = 0.4;

    // ── CPU path ─────────────────────────────────────────────────────────────
    NSSolver cpu_solver;
    cpu_solver.cfg = make_cfg(cfl);
    cpu_solver.init(1.0, flat_ic);

    std::vector<double> cpu_dts;
    for (int s = 0; s < nstep; ++s)
        cpu_dts.push_back(cpu_solver.advance());
    const QSnap cpu_q = snap(cpu_solver.tree);

    // ── GPU-dispatched NSSolver path ─────────────────────────────────────────
    NSSolver gpu_solver;
    gpu_solver.cfg = make_cfg(cfl);
    gpu_solver.cfg.use_gpu = true;
    gpu_solver.init(1.0, flat_ic);

    GpuPool pool;
    for (int li : gpu_solver.tree.leaf_indices()) {
        CellBlock* blk = gpu_solver.tree.nodes[li].block.get();
        if (!blk) continue;
        pool.alloc(blk);
        pool.upload(blk);
    }

    GpuGraphSolver graph_solver;
    gpu_solver.set_gpu_pool(&pool);
    graph_solver.build(gpu_solver.tree, pool, /*bc_type=*/0);
    gpu_solver.set_gpu_solver(&graph_solver);

    std::vector<double> gpu_dts;
    for (int s = 0; s < nstep; ++s)
        gpu_dts.push_back(gpu_solver.advance());
    const QSnap gpu_q = snap(gpu_solver.tree);

    for (int li : gpu_solver.tree.leaf_indices()) {
        CellBlock* blk = gpu_solver.tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }

    // ── Compare ──────────────────────────────────────────────────────────────
    const double phys_err = max_rel_err(cpu_q, gpu_q);
    printf("   max phys_err (GPU dispatch vs CPU, %d steps) = %.3e  (tol 1e-8)\n",
           nstep, phys_err);
    check(phys_err < 1.0e-8, "A1",
          "NSSolver GPU dispatch Q == CPU Q (20 steps, tol 1e-8)", phys_err);

    double max_dt_err = 0.0;
    for (int s = 0; s < nstep; ++s)
        max_dt_err = std::fmax(max_dt_err,
            std::fabs(cpu_dts[s] - gpu_dts[s]) / cpu_dts[s]);
    printf("   max rel dt error (%d steps) = %.3e  (tol 1e-10)\n",
           nstep, max_dt_err);
    check(max_dt_err < 1.0e-10, "A2",
          "NSSolver GPU dispatch dt == CPU dt (tol 1e-10)", max_dt_err);
}

// =============================================================================
// A3: mass conservation over 20 GPU-dispatched steps
// =============================================================================
static void test_a3() {
    printf("\n-- A3  NSSolver GPU dispatch mass conservation (20 steps) --\n");
    const int    nstep = 20;
    const double cfl   = 0.4;

    NSSolver solver;
    solver.cfg = make_cfg(cfl);
    solver.cfg.use_gpu = true;
    solver.init(1.0, flat_ic);

    GpuPool pool;
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (!blk) continue;
        pool.alloc(blk);
        pool.upload(blk);
    }

    GpuGraphSolver graph_solver;
    solver.set_gpu_pool(&pool);
    graph_solver.build(solver.tree, pool, 0);
    solver.set_gpu_solver(&graph_solver);

    const double m0 = total_mass(solver.tree);
    for (int s = 0; s < nstep; ++s) solver.advance();
    const double m1 = total_mass(solver.tree);

    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }

    const double mass_err = std::fabs(m1 - m0) / m0;
    printf("   mass rel error over %d GPU dispatch steps = %.3e  (tol 1e-8)\n",
           nstep, mass_err);
    check(mass_err < 1.0e-8, "A3",
          "NSSolver GPU dispatch mass conserved (20 steps, tol 1e-8)", mass_err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P10-A3 NSSolver GPU dispatch gate test ===\n");
    test_a1_a2();
    test_a3();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
