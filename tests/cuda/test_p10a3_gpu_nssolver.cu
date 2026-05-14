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
    cfg.time.cfl             = cfl;
    cfg.bc.variant = PeriodicBC{};
    cfg.amr.regrid_interval = 0;
    cfg.io.verbose         = false;
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
    gpu_solver.cfg.exec.use_gpu = true;
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
    solver.cfg.exec.use_gpu = true;
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
// P11.8 A4: GPU solver with AMR falls back to CPU path; mass conserved
// =============================================================================
// IC: density step (high gradient triggers should_refine).
// GPU solver with regrid_interval=1 should use CPU path when AMR levels > 0.
// Mass conservation across CPU-fallback + potential GPU steps validates P11.8.
static Prim amr_ic(double x, double, double) {
    return {(x < 0.5) ? 1.0 : 2.0, 0.0, 0.0, 0.0, 1.0e5};
}

static void test_a4() {
    printf("\n-- A4  P11.8 AMR fallback: GPU solver uses CPU path when AMR active --\n");
    const int    nstep = 10;
    const double cfl   = 0.4;

    // ── Pure CPU solver with AMR (reference) ─────────────────────────────────
    NSSolver cpu_solver;
    cpu_solver.cfg = make_cfg(cfl);
    cpu_solver.cfg.amr.regrid_interval = 1;
    cpu_solver.cfg.amr.max_level       = 1;
    cpu_solver.init(1.0, amr_ic);

    const double m0_cpu = total_mass(cpu_solver.tree);
    for (int s = 0; s < nstep; ++s) cpu_solver.advance();
    const double m1_cpu = total_mass(cpu_solver.tree);

    // ── GPU-dispatched solver with AMR ────────────────────────────────────────
    NSSolver gpu_solver;
    gpu_solver.cfg = make_cfg(cfl);
    gpu_solver.cfg.amr.regrid_interval = 1;
    gpu_solver.cfg.amr.max_level       = 1;
    gpu_solver.cfg.exec.use_gpu         = true;
    gpu_solver.init(1.0, amr_ic);

    GpuPool pool;
    // Wire GPU pool callbacks so refine/coarsen manage GPU memory correctly.
    gpu_solver.tree.set_gpu_callbacks(
        [&pool](CellBlock* blk) { pool.alloc(blk); pool.upload(blk); },
        [&pool](CellBlock* blk) { pool.free(blk); }
    );
    // Allocate initial blocks.
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

    const double m0_gpu = total_mass(gpu_solver.tree);
    for (int s = 0; s < nstep; ++s) gpu_solver.advance();
    const double m1_gpu = total_mass(gpu_solver.tree);

    // Cleanup live blocks in the pool.
    for (int li : gpu_solver.tree.leaf_indices()) {
        CellBlock* blk = gpu_solver.tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
    // Remove callbacks to avoid dangling references after pool goes out of scope.
    gpu_solver.tree.set_gpu_callbacks(nullptr, nullptr);

    const double cpu_mass_err = std::fabs(m1_cpu - m0_cpu) / m0_cpu;
    const double gpu_mass_err = std::fabs(m1_gpu - m0_gpu) / m0_gpu;
    printf("   CPU-AMR mass rel err = %.3e  (tol 1e-8)\n", cpu_mass_err);
    printf("   GPU-AMR mass rel err = %.3e  (tol 1e-8)\n", gpu_mass_err);
    check(cpu_mass_err < 1.0e-8, "A4a",
          "CPU AMR mass conserved (10 steps, tol 1e-8)", cpu_mass_err);
    check(gpu_mass_err < 1.0e-8, "A4b",
          "GPU+AMR fallback mass conserved (10 steps, tol 1e-8)", gpu_mass_err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P10-A3 / P11.8 NSSolver GPU dispatch gate test ===\n");
    test_a1_a2();
    test_a3();
    test_a4();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
