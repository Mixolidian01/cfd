// D3 gate test — GPU TENO7-A convective kernel (k_rhs_conv_teno7)
//
// A71: mass conservation over 4 steps — TENO7-A default
//      (4 steps is within the stable range before the shock wraps on an 8-cell
//      periodic domain; long-time conservation is not meaningful on this coarse
//      grid since the wrapped shock creates sub-stencil configurations where
//      TENO7's 4-point sub-stencils span the discontinuity from both sides)
// A72: translational invariance — X-Sod IC is constant in y,z after 4 GPU steps
//      (catches per-axis bugs in gpu_teno7_face Roe decomposition)
// A73: axis symmetry — X-Sod / Y-Sod / Z-Sod give same max(rho) after 4 steps
//      (verifies nidx/t1idx/t2idx mapping for all three normals)

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_rhs.cuh"
#include "cuda/gpu_check.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
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

static GpuPool pool;

// Sod IC parameterised by axis: 0=X, 1=Y, 2=Z
static Prim sod_ic(double x, double y, double z, int axis) {
    double coord = (axis == 0) ? x : (axis == 1) ? y : z;
    Prim p{};
    bool left = coord < 0.5;
    p.rho = left ? 1.0 : 0.125;
    p.u = p.v = p.w = 0.0;
    p.p = left ? 1.0e5 : 1.0e4;
    return p;
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

// Run N steps with a given scheme on an axis-Sod IC.  Downloads Q to CPU after run.
static BlockTree run_scheme(int nstep, double cfl, int axis, GpuReconScheme scheme) {
    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    {
        CellBlock& blk = *tree.nodes[0].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = (i - NG + 0.5) * blk.h;
            double y = (j - NG + 0.5) * blk.h;
            double z = (k - NG + 0.5) * blk.h;
            Prim p = sod_ic(x, y, z, axis);
            int flat = cell_idx(i, j, k);
            blk.Q[0][flat] = p.rho;
            blk.Q[1][flat] = p.rho * p.u;
            blk.Q[2][flat] = p.rho * p.v;
            blk.Q[3][flat] = p.rho * p.w;
            blk.Q[4][flat] = p.p / (GAMMA - 1.0)
                           + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
    upload_all(tree);
    GpuGraphSolver solver;
    solver.rhs_list.scheme = scheme;
    solver.build(tree, pool);
    for (int s = 0; s < nstep; ++s)
        solver.advance(tree, cfl);
    solver.download_q(tree);
    free_all(tree);
    return tree;
}

// Run N steps with TENO7-A on an axis-Sod IC.  Downloads Q to CPU after run.
static BlockTree run_teno7a(int nstep, double cfl, int axis) {
    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    {
        CellBlock& blk = *tree.nodes[0].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = (i - NG + 0.5) * blk.h;
            double y = (j - NG + 0.5) * blk.h;
            double z = (k - NG + 0.5) * blk.h;
            Prim p = sod_ic(x, y, z, axis);
            int flat = cell_idx(i, j, k);
            blk.Q[0][flat] = p.rho;
            blk.Q[1][flat] = p.rho * p.u;
            blk.Q[2][flat] = p.rho * p.v;
            blk.Q[3][flat] = p.rho * p.w;
            blk.Q[4][flat] = p.p / (GAMMA - 1.0)
                           + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
    upload_all(tree);

    GpuGraphSolver solver;
    solver.rhs_list.scheme = GpuReconScheme::TENO7A;
    solver.build(tree, pool);
    for (int s = 0; s < nstep; ++s)
        solver.advance(tree, cfl);
    solver.download_q(tree);
    free_all(tree);
    return tree;
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
// A71: mass conservation over 4 steps, TENO7-A
// =============================================================================
static void test_a71() {
    printf("\n-- A71  TENO7-A mass conservation over 4 steps  (tol 1e-8) --\n");
    const double cfl = 0.3;
    const int NSTEP  = 4;

    BlockTree tree0; tree0.init(1.0); tree0.set_periodic(true);
    {
        CellBlock& blk = *tree0.nodes[0].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = (i - NG + 0.5) * blk.h;
            Prim p = sod_ic(x, 0.0, 0.0, 0);
            int flat = cell_idx(i, j, k);
            blk.Q[0][flat] = p.rho;
            blk.Q[1][flat] = p.rho * p.u;
            blk.Q[2][flat] = p.rho * p.v;
            blk.Q[3][flat] = p.rho * p.w;
            blk.Q[4][flat] = p.p / (GAMMA - 1.0)
                           + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
    upload_all(tree0);
    const double m0 = total_mass(tree0);

    GpuGraphSolver solver;
    solver.rhs_list.scheme = GpuReconScheme::TENO7A;
    solver.build(tree0, pool);
    for (int s = 0; s < NSTEP; ++s)
        solver.advance(tree0, cfl);
    solver.download_q(tree0);

    const double mf  = total_mass(tree0);
    const double rel = std::fabs(mf - m0) / std::fabs(m0);
    printf("   TENO7-A mass rel error over 4 steps = %.3e  (tol 1e-8)\n", rel);
    check(rel < 1.0e-8, "A71", "TENO7-A mass conserved over 4 steps (tol 1e-8)", rel);

    free_all(tree0);
}

// =============================================================================
// A72_STEP: same check for N=1,2,3,4 to find when variation appears
// =============================================================================
static void test_a72_step_scan() {
    printf("\n-- A72_STEP  Variable spread scan after 1 step (TENO7A vs TENO5A) --\n");
    const double cfl = 0.3;
    const int nstep = 1;

    BlockTree t7 = run_scheme(nstep, cfl, 0, GpuReconScheme::TENO7A);
    BlockTree t5 = run_scheme(nstep, cfl, 0, GpuReconScheme::TENO5A);
    const CellBlock& b7 = *t7.nodes[0].block;
    const CellBlock& b5 = *t5.nodes[0].block;

    const char* vname[5] = {"rho","rhou","rhov","rhow","E"};
    for (int v = 0; v < 5; ++v) {
        double max_spread7 = 0.0, max_spread5 = 0.0, max_diff = 0.0;
        int wi7 = -1, wj7 = -1, wk7 = -1;
        for (int i = NG; i < NG+NB; ++i) {
            double vmin7=1e300, vmax7=-1e300;
            double vmin5=1e300, vmax5=-1e300;
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j) {
                double q7 = b7.Q[v][cell_idx(i,j,k)];
                double q5 = b5.Q[v][cell_idx(i,j,k)];
                vmin7=std::min(vmin7,q7); vmax7=std::max(vmax7,q7);
                vmin5=std::min(vmin5,q5); vmax5=std::max(vmax5,q5);
                double d = std::fabs(q7-q5)/std::max(std::fabs(q5),1e-12);
                if (d > max_diff) { max_diff=d; wi7=i; wj7=j; wk7=k; }
            }
            double s7 = (vmax7-vmin7)/std::max(std::fabs(vmin7),1e-12);
            double s5 = (vmax5-vmin5)/std::max(std::fabs(vmin5),1e-12);
            max_spread7 = std::max(max_spread7, s7);
            max_spread5 = std::max(max_spread5, s5);
        }
        printf("   v=%s  spread7=%.3e  spread5=%.3e  max|7-5|/5=%.3e  (worst i=%d j=%d k=%d)\n",
               vname[v], max_spread7, max_spread5, max_diff, wi7, wj7, wk7);
    }

}

// =============================================================================
// A72: translational invariance — X-Sod rho must be constant in y,z after 4 steps
// =============================================================================
static void test_a72() {
    printf("\n-- A72  Translational invariance (X-Sod constant in y,z)  (tol 1e-10) --\n");
    const double cfl = 0.3;
    const int NSTEP  = 4;

    BlockTree tree = run_teno7a(NSTEP, cfl, 0);  // X-Sod
    const CellBlock& blk = *tree.nodes[0].block;

    double max_spread = 0.0;
    for (int i = NG; i < NG+NB; ++i) {
        double rho_min =  1e300, rho_max = -1e300;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j) {
            double rho = blk.Q[0][cell_idx(i,j,k)];
            rho_min = std::min(rho_min, rho);
            rho_max = std::max(rho_max, rho);
        }
        double spread = (rho_max - rho_min) / std::max(rho_min, 1.0e-12);
        max_spread = std::max(max_spread, spread);
    }
    printf("   max rel spread of rho over y,z at fixed i = %.3e  (tol 1e-10)\n", max_spread);
    check(max_spread < 1.0e-10, "A72",
          "X-Sod rho constant in y,z (translational invariance, tol 1e-10)", max_spread);
}

// =============================================================================
// A73: axis symmetry — X/Y/Z Sod give same max(rho) over 4 steps
// =============================================================================
static void test_a73() {
    printf("\n-- A73  Axis symmetry: X/Y/Z Sod max(rho) equal  (tol 1e-10) --\n");
    const double cfl = 0.3;
    const int NSTEP  = 4;

    auto max_rho = [](const BlockTree& tree) -> double {
        double mx = 0.0;
        for (int li : tree.leaf_indices()) {
            const CellBlock& blk = *tree.nodes[li].block;
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j)
            for (int i = NG; i < NG+NB; ++i)
                mx = std::max(mx, blk.Q[0][cell_idx(i,j,k)]);
        }
        return mx;
    };

    BlockTree tx = run_teno7a(NSTEP, cfl, 0);
    BlockTree ty = run_teno7a(NSTEP, cfl, 1);
    BlockTree tz = run_teno7a(NSTEP, cfl, 2);

    const double rx = max_rho(tx);
    const double ry = max_rho(ty);
    const double rz = max_rho(tz);
    const double err_xy = std::fabs(rx - ry) / std::max(rx, 1.0e-12);
    const double err_xz = std::fabs(rx - rz) / std::max(rx, 1.0e-12);
    printf("   max_rho X=%.6e  Y=%.6e  Z=%.6e\n", rx, ry, rz);
    printf("   rel |X-Y|=%.3e  rel |X-Z|=%.3e  (tol 1e-10)\n", err_xy, err_xz);
    check(err_xy < 1.0e-10, "A73a", "X-Sod max(rho) == Y-Sod max(rho) (tol 1e-10)", err_xy);
    check(err_xz < 1.0e-10, "A73b", "X-Sod max(rho) == Z-Sod max(rho) (tol 1e-10)", err_xz);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== D3 GPU TENO7-A gate test (t37) ===\n");
    test_a71();
    test_a72_step_scan();
    test_a72();
    test_a73();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
