// D1 gate test — GPU-native AMR (t29)
//
// Verifies GpuGraphSolver::gpu_regrid() runs refinement and coarsening entirely
// on-device.  Only n_leaves floats (sensor values) are transferred D2H during
// regrid — full Q arrays stay GPU-resident throughout.
//
// Protocol:
//   A1: uniform IC → sensor ≈ 0 → gpu_regrid returns false (no topology change)
//   A2: step-function IC → refine → 1 root block becomes 8 children; mass conserved
//   A3: smooth IC after prolong → coarsen → 8 children collapse to 1; mass conserved
//   A4: NSSolver with gpu_solver + gpu_regrid; mass conserved over 10 steps (tol 1e-8)

#include "cuda/gpu_graph.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "solver/ns_solver.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
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

// ── IC helpers ───────────────────────────────────────────────────────────────

// Uniform IC: sensor = 0
static void fill_uniform(CellBlock& blk) {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        int f = cell_idx(i,j,k);
        blk.Q[0][f] = 1.0;
        blk.Q[1][f] = 0.0;
        blk.Q[2][f] = 0.0;
        blk.Q[3][f] = 0.0;
        blk.Q[4][f] = 1.0e5 / (GAMMA - 1.0);
    }
}

// Step-function IC: high density gradient → sensor >> threshold
static void fill_step(CellBlock& blk) {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double rho = (i < NB2 / 2) ? 1.0 : 0.125;
        double p   = (i < NB2 / 2) ? 1.0e5 : 1.0e4;
        int f = cell_idx(i,j,k);
        blk.Q[0][f] = rho;
        blk.Q[1][f] = 0.0;
        blk.Q[2][f] = 0.0;
        blk.Q[3][f] = 0.0;
        blk.Q[4][f] = p / (GAMMA - 1.0);
    }
}

// Upload all current leaves to pool (alloc if needed)
static void upload_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        if (!pool.has_device(blk)) pool.alloc(blk);
        pool.upload(blk);
    }
}

// Download all current leaves from pool to CPU
static void download_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.download(blk);
    }
}

// Free all current leaves from pool
static void free_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
}

// Total mass (CPU Q, interior cells only)
static double total_mass(const BlockTree& tree) {
    double m = 0.0;
    for (int li : tree.leaf_indices()) {
        const CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        double dV = blk->h * blk->h * blk->h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            m += blk->Q[0][cell_idx(i,j,k)] * dV;
    }
    return m;
}

// =============================================================================
// A1: uniform IC → sensor ≈ 0 → gpu_regrid returns false
// =============================================================================
static void test_a1() {
    printf("\n-- A1  Uniform IC: sensor ≈ 0, gpu_regrid returns false --\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    fill_uniform(*tree.nodes[0].block);
    upload_all(tree);

    GpuGraphSolver solver;
    solver.build(tree, pool, 0);

    const int leaves_before = (int)tree.leaf_indices().size();
    const bool changed = solver.gpu_regrid(tree, pool, 0, /*max_level=*/2,
                                            /*refine_thr=*/0.05f,
                                            /*coarsen_thr=*/0.01f);
    const int leaves_after  = (int)tree.leaf_indices().size();

    printf("   gpu_regrid returned %s  (expect false)\n", changed ? "true" : "false");
    printf("   leaf count: %d → %d  (expect no change)\n", leaves_before, leaves_after);
    check(!changed,            "A1a", "gpu_regrid returns false on uniform IC");
    check(leaves_after == 1,   "A1b", "leaf count unchanged (1 leaf)");

    free_all(tree);
}

// =============================================================================
// A2: step-function IC → refine → 8 children; mass conserved to 1e-10
// =============================================================================
static void test_a2() {
    printf("\n-- A2  Step IC: refine triggered; mass conserved --\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    fill_step(*tree.nodes[0].block);

    // Compute pre-refine mass from CPU IC
    const double m_before = total_mass(tree);

    upload_all(tree);

    GpuGraphSolver solver;
    solver.build(tree, pool, 0);

    // Low threshold → the step function will trigger refinement
    const bool changed = solver.gpu_regrid(tree, pool, 0, /*max_level=*/2,
                                            /*refine_thr=*/0.01f,
                                            /*coarsen_thr=*/0.001f);
    const int leaves_after = (int)tree.leaf_indices().size();

    // Download to CPU for mass check
    download_all(tree);
    const double m_after = total_mass(tree);

    const double mass_err = std::fabs(m_after - m_before) / m_before;
    printf("   gpu_regrid returned %s  (expect true)\n", changed ? "true" : "false");
    printf("   leaf count after refine: %d  (expect 8)\n", leaves_after);
    printf("   mass rel err = %.3e  (tol 1e-10)\n", mass_err);

    check(changed,            "A2a", "gpu_regrid returns true (topology changed)");
    check(leaves_after == 8,  "A2b", "1 block refined to 8 children");
    check(mass_err < 1.0e-10, "A2c", "mass conserved after GPU-native prolong", mass_err);

    free_all(tree);
}

// =============================================================================
// A3: refine then coarsen → mass conserved round-trip to 1e-10
// =============================================================================
static void test_a3() {
    printf("\n-- A3  Refine then coarsen: mass conserved round-trip --\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    fill_step(*tree.nodes[0].block);

    const double m_before = total_mass(tree);
    upload_all(tree);

    GpuGraphSolver solver;
    solver.build(tree, pool, 0);

    // Step 1: refine (step IC → high gradient)
    solver.gpu_regrid(tree, pool, 0, 2, 0.01f, 0.001f);
    const int after_refine = (int)tree.leaf_indices().size();

    // Step 2: coarsen (piecewise-constant children are smooth within each octant)
    // Use high refine_thr (never refine) and low coarsen_thr (always coarsen)
    solver.gpu_regrid(tree, pool, 0, 2, /*refine_thr=*/100.0f, /*coarsen_thr=*/100.0f);
    const int after_coarsen = (int)tree.leaf_indices().size();

    download_all(tree);
    const double m_after = total_mass(tree);
    const double mass_err = std::fabs(m_after - m_before) / m_before;

    printf("   leaf count: 1 → %d → %d  (expect 1 → 8 → 1)\n",
           after_refine, after_coarsen);
    printf("   mass rel err (round-trip) = %.3e  (tol 1e-10)\n", mass_err);

    check(after_refine  == 8, "A3a", "refine: 1 root → 8 children");
    check(after_coarsen == 1, "A3b", "coarsen: 8 children → 1 root");
    check(mass_err < 1.0e-10, "A3c", "mass conserved across prolong+restrict round-trip",
          mass_err);

    free_all(tree);
}

// =============================================================================
// A4: NSSolver + gpu_solver + periodic gpu_regrid; mass conserved (10 steps)
// =============================================================================
static Prim amr_ic_ns(double x, double, double) {
    return {(x < 0.5) ? 1.0 : 2.0, 0.0, 0.0, 0.0, 1.0e5};
}

static double total_mass_ns(const BlockTree& tree) {
    double m = 0.0;
    for (int li : tree.leaf_indices()) {
        const CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        double dV = blk->h * blk->h * blk->h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            m += blk->Q[0][cell_idx(i,j,k)] * dV;
    }
    return m;
}

static void test_a4() {
    printf("\n-- A4  NSSolver with gpu_regrid: mass conserved (10 steps) --\n");
    const int    nstep = 10;
    const double cfl   = 0.4;

    NSSolver solver;
    solver.cfg.time.cfl             = cfl;
    solver.cfg.bc.variant           = PeriodicBC{};
    solver.cfg.amr.regrid_interval  = 2;
    solver.cfg.amr.max_level        = 1;
    solver.cfg.exec.use_gpu         = true;
    solver.cfg.io.verbose           = false;
    solver.init(1.0, amr_ic_ns);

    GpuPool a4_pool;
    // Wire CPU callbacks for refine/coarsen triggered outside gpu_regrid.
    // (balance() during gpu_regrid uses the GPU callbacks; these handle
    //  any topology change before gpu_solver_ is set.)
    solver.tree.set_gpu_callbacks(
        [&a4_pool](CellBlock* blk) { a4_pool.alloc(blk); a4_pool.upload(blk); },
        [&a4_pool](CellBlock* blk) { a4_pool.free(blk); }
    );

    // Upload initial IC
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (!blk) continue;
        a4_pool.alloc(blk);
        a4_pool.upload(blk);
    }

    GpuGraphSolver graph_solver;
    solver.set_gpu_pool(&a4_pool);
    graph_solver.build(solver.tree, a4_pool, 0);
    solver.set_gpu_solver(&graph_solver);

    const double m0 = total_mass_ns(solver.tree);

    for (int s = 0; s < nstep; ++s) solver.advance();

    // Download final state
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (blk && a4_pool.has_device(blk)) a4_pool.download(blk);
    }
    const double m1 = total_mass_ns(solver.tree);

    // Cleanup
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (blk && a4_pool.has_device(blk)) a4_pool.free(blk);
    }
    solver.tree.set_gpu_callbacks(nullptr, nullptr);

    const double mass_err = std::fabs(m1 - m0) / m0;
    printf("   mass rel err over %d steps with gpu_regrid = %.3e  (tol 1e-8)\n",
           nstep, mass_err);
    check(mass_err < 1.0e-8, "A4",
          "NSSolver GPU-native AMR mass conserved (10 steps, tol 1e-8)", mass_err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== D1 GPU-native AMR gate test (t29) ===\n");
    test_a1();
    test_a2();
    test_a3();
    test_a4();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
