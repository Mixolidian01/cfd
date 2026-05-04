// P8.6 gate test — CUDA Graph re-capture on regrid
//
// Protocol:
//   G1: 1 explicit advance() == reference explicit (first advance is always explicit)
//   G2: 4-step graph replay gives same Q as 4-step explicit-only run
//   G3: rebuild() invalidates graph; first step after rebuild is explicit;
//       second step is replay; result matches explicit reference
//   G4: 20 graph-replay steps on periodic smooth flow: KE conserved < 1%

#include "../../include/cuda/gpu_graph.cuh"
#include "../../include/gpu_pool.hpp"
#include "../../include/block_tree.hpp"
#include "../../include/cell_block.hpp"
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

// ── Helpers ──────────────────────────────────────────────────────────────────

static void fill_sod(CellBlock& blk) {
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        double x = (i - NG + 0.5) * blk.h;
        bool left = x < 0.5 * NB * blk.h;
        double rho = left ? 1.0 : 0.125;
        double p   = left ? 1.0e5 : 1.0e4;
        int flat   = cell_idx(i,j,k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = 0.0;
        blk.Q[2][flat] = 0.0;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = p / (GAMMA - 1.0);
    }
    // Fill ghosts with same L/R states (simple)
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        bool ghost = i < NG || i >= NG+NB ||
                     j < NG || j >= NG+NB ||
                     k < NG || k >= NG+NB;
        if (!ghost) continue;
        double x = (i - NG + 0.5) * blk.h;
        bool left = x < 0.5 * NB * blk.h;
        double rho = left ? 1.0 : 0.125;
        double p   = left ? 1.0e5 : 1.0e4;
        int flat   = cell_idx(i,j,k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = 0.0;
        blk.Q[2][flat] = 0.0;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = p / (GAMMA - 1.0);
    }
}

// Smooth periodic flow suitable for G4
static void fill_smooth(CellBlock& blk) {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = (i - NG + 0.5) * blk.h;
        double y = (j - NG + 0.5) * blk.h;
        double L = NB * blk.h;
        double rho = 1.0 + 0.1 * sin(2*M_PI*x/L) * cos(2*M_PI*y/L);
        double u   = 0.05 * cos(2*M_PI*x/L);
        double v   = 0.05 * sin(2*M_PI*y/L);
        double p   = 1.0e5;
        int flat   = cell_idx(i,j,k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = rho * u;
        blk.Q[2][flat] = rho * v;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v);
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

// Save interior Q to a flat buffer [leaf][v*NCELL + flat]
using QSnapshot = std::vector<double>;

static QSnapshot snap(const BlockTree& tree) {
    const auto& lv = tree.leaf_indices();
    QSnapshot q(lv.size() * NVAR * NCELL, 0.0);
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

static double max_rel_err(const QSnapshot& a, const QSnapshot& b) {
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double ref = b[i];
        double d = std::fabs(a[i] - ref);
        if (ref != 0.0) err = std::fmax(err, d / std::fabs(ref));
        else            err = std::fmax(err, d);
    }
    return err;
}

// Mixed absolute+relative error: skips cells where both values are below the
// physical floor (avoids false alarms from near-zero momentum components).
// scale: typical field magnitude for denominator (e.g. max|Q| or 1.0 for rho).
static double max_phys_err(const QSnapshot& a, const QSnapshot& b,
                           double abs_floor = 1.0e-8) {
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double scale = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
        if (scale < abs_floor) continue;  // both near-zero: skip round-off noise
        double d = std::fabs(a[i] - b[i]);
        err = std::fmax(err, d / scale);
    }
    return err;
}

// =============================================================================
// G1: single advance() (first step = explicit) gives sensible result
// =============================================================================
static void test_g1() {
    printf("\n-- G1  Single advance() (explicit step) produces non-trivial update --\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    fill_sod(*tree.nodes[0].block);
    upload_all(tree);

    GpuGraphSolver solver;
    solver.build(tree, pool);

    const QSnapshot before = snap(tree);

    solver.advance(tree, 0.5);
    solver.download_q(tree);

    const QSnapshot after = snap(tree);
    free_all(tree);

    // Verify Q changed (the step actually ran)
    double change = 0.0;
    for (size_t i = 0; i < before.size(); ++i)
        change = std::fmax(change, std::fabs(after[i] - before[i]));

    printf("   max Q change after 1 step = %.3e  (expect > 0)\n", change);
    check(change > 0.0, "G1", "Q changed after 1 explicit step");
}

// =============================================================================
// G2: 4 graph-replay steps == 4 explicit-only steps (bitwise identical Q)
//
// SolverA: rebuild before each step → forces explicit every step
// SolverB: build once → step 1 explicit+capture, steps 2-4 replay
// =============================================================================

// Helper: run nstep steps with explicit-only mode (build before each)
static QSnapshot run_explicit(int nstep, double cfl) {
    BlockTree t; t.init(1.0); t.set_periodic(true);
    fill_sod(*t.nodes[0].block);
    upload_all(t);
    GpuGraphSolver s;
    for (int i = 0; i < nstep; ++i) { s.build(t, pool); s.advance(t, cfl); }
    s.download_q(t);
    auto q = snap(t);
    free_all(t);
    return q;
}

// Helper: run nstep steps with graph replay (build once)
static QSnapshot run_graph(int nstep, double cfl) {
    BlockTree t; t.init(1.0); t.set_periodic(true);
    fill_sod(*t.nodes[0].block);
    upload_all(t);
    GpuGraphSolver s;
    s.build(t, pool);
    for (int i = 0; i < nstep; ++i) s.advance(t, cfl);
    s.download_q(t);
    auto q = snap(t);
    free_all(t);
    return q;
}

static void test_g2() {
    printf("\n-- G2  4-step graph replay == 4-step explicit (Q match within 1e-8) --\n");
    const double cfl = 0.4;
    const int NSTEP = 4;

    // Both trees alive simultaneously → separate device slots → separate streams
    BlockTree treeA; treeA.init(1.0); treeA.set_periodic(true);
    fill_sod(*treeA.nodes[0].block);
    upload_all(treeA);

    BlockTree treeB; treeB.init(1.0); treeB.set_periodic(true);
    fill_sod(*treeB.nodes[0].block);
    upload_all(treeB);

    // SolverA: explicit every step (build before each advance)
    GpuGraphSolver solverA;
    for (int s = 0; s < NSTEP; ++s) {
        solverA.build(treeA, pool);
        solverA.advance(treeA, cfl);
    }
    solverA.download_q(treeA);
    const QSnapshot qa = snap(treeA);
    free_all(treeA);

    // SolverB: build once → step 1 explicit+capture, steps 2-4 replay
    GpuGraphSolver solverB;
    solverB.build(treeB, pool);
    for (int s = 0; s < NSTEP; ++s)
        solverB.advance(treeB, cfl);
    solverB.download_q(treeB);
    const QSnapshot qb = snap(treeB);
    free_all(treeB);

    // max_phys_err skips near-zero momentum cells (|val| < 1e-8) so that
    // sub-machine-epsilon atomicAdd noise doesn't inflate relative error.
    const double err = max_phys_err(qb, qa);
    printf("   max phys_err (graph vs explicit) = %.3e  (tol 1e-8)\n", err);
    check(err < 1.0e-8, "G2", "graph replay Q == explicit Q (4 steps, tol 1e-8)", err);
}

// =============================================================================
// G3: rebuild invalidates graph; post-rebuild steps are correct
//
// Run 2 steps (step 1: explicit+capture; step 2: replay).
// Rebuild. Run 2 more steps (step 3: explicit+capture; step 4: replay).
// Compare against all-explicit reference (SolverRef).
// =============================================================================
static void test_g3() {
    printf("\n-- G3  Post-rebuild graph re-capture correct (4 steps with mid-rebuild) --\n");
    const double cfl = 0.4;

    // ── Reference: 4 explicit steps ──────────────────────────────────────────
    BlockTree treeRef; treeRef.init(1.0); treeRef.set_periodic(true);
    fill_sod(*treeRef.nodes[0].block);
    upload_all(treeRef);
    GpuGraphSolver ref;
    for (int s = 0; s < 4; ++s) {
        ref.build(treeRef, pool);
        ref.advance(treeRef, cfl);
    }
    ref.download_q(treeRef);
    const QSnapshot qref = snap(treeRef);
    free_all(treeRef);

    // ── Test: 2 steps, rebuild, 2 more steps ─────────────────────────────────
    BlockTree treeT; treeT.init(1.0); treeT.set_periodic(true);
    fill_sod(*treeT.nodes[0].block);
    upload_all(treeT);
    GpuGraphSolver solver;
    solver.build(treeT, pool);
    solver.advance(treeT, cfl);   // explicit + capture
    solver.advance(treeT, cfl);   // replay
    solver.build(treeT, pool);          // simulate regrid → invalidate + rebuild
    solver.advance(treeT, cfl);   // explicit + re-capture
    solver.advance(treeT, cfl);   // replay
    solver.download_q(treeT);
    const QSnapshot qt = snap(treeT);
    free_all(treeT);

    const double err = max_phys_err(qt, qref);
    printf("   max phys_err (mid-rebuild vs all-explicit) = %.3e  (tol 1e-8)\n", err);
    check(err < 1.0e-8, "G3",
          "post-rebuild graph re-capture gives correct Q (4 steps, tol 1e-8)", err);
}

// =============================================================================
// G4: 20 graph-replay steps on smooth periodic flow match 20 explicit steps
//
// SolverA: rebuild before every step (always explicit — reference)
// SolverB: build once (step 1 explicit+capture, steps 2-20 graph replay)
// Gate: max_phys_err(SolverB, SolverA) < 1e-8
// =============================================================================
static void test_g4() {
    printf("\n-- G4  20 replay steps, smooth periodic: graph Q == explicit Q (tol 1e-8) --\n");
    const double cfl = 0.3;
    const int NSTEP = 20;

    // Both trees alive simultaneously → separate device slots
    BlockTree treeA; treeA.init(1.0); treeA.set_periodic(true);
    fill_smooth(*treeA.nodes[0].block);
    upload_all(treeA);

    BlockTree treeB; treeB.init(1.0); treeB.set_periodic(true);
    fill_smooth(*treeB.nodes[0].block);
    upload_all(treeB);

    // SolverA: always explicit (build before each step)
    GpuGraphSolver solverA;
    for (int s = 0; s < NSTEP; ++s) {
        solverA.build(treeA, pool);
        solverA.advance(treeA, cfl);
    }
    solverA.download_q(treeA);
    const QSnapshot qa = snap(treeA);
    free_all(treeA);

    // SolverB: graph replay after first step
    GpuGraphSolver solverB;
    solverB.build(treeB, pool);
    for (int s = 0; s < NSTEP; ++s)
        solverB.advance(treeB, cfl);
    solverB.download_q(treeB);
    const QSnapshot qb = snap(treeB);
    free_all(treeB);

    const double err = max_phys_err(qb, qa);
    printf("   max phys_err (replay vs explicit, 20 steps) = %.3e  (tol 1e-8)\n", err);
    check(err < 1.0e-8, "G4",
          "20 graph-replay steps == 20 explicit steps on smooth periodic (tol 1e-8)", err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P8.6 CUDA Graph re-capture on regrid gate test ===\n");
    test_g1();
    test_g2();
    test_g3();
    test_g4();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
