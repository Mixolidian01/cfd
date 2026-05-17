// D3 gate test — GPU TENO5-A convective kernel (k_rhs_conv_teno)
//
// A31: mass conservation over 20 steps — TENO5-A default (no WENO5-Z override)
// A32: translational invariance — X-Sod IC is constant in y,z after 4 GPU steps
//      (catches per-axis bugs in gpu_teno5_face Roe decomposition)
// A33: axis symmetry — X-Sod / Y-Sod / Z-Sod give same max(rho) after 4 steps
//      (verifies nidx/t1idx/t2idx mapping is correct for all three normals)

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

// Run N steps with TENO5-A (default — no scheme override) on an axis-Sod IC.
// Downloads Q to CPU after run.
static BlockTree run_teno5a(int nstep, double cfl, int axis) {
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
    // TENO5-A is the default — do NOT set scheme = WENO5Z.
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
// A31: mass conservation over 20 steps, TENO5-A default
// =============================================================================
static void test_a31() {
    printf("\n-- A31  TENO5-A mass conservation over 20 steps  (tol 1e-8) --\n");
    const double cfl = 0.3;
    const int NSTEP  = 20;

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
    // TENO5-A default — no override.
    solver.build(tree0, pool);
    for (int s = 0; s < NSTEP; ++s)
        solver.advance(tree0, cfl);
    solver.download_q(tree0);

    const double mf  = total_mass(tree0);
    const double rel = std::fabs(mf - m0) / std::fabs(m0);
    printf("   TENO5-A mass rel error over 20 steps = %.3e  (tol 1e-8)\n", rel);
    check(rel < 1.0e-8, "A31", "TENO5-A mass conserved over 20 steps (tol 1e-8)", rel);

    free_all(tree0);
}

// =============================================================================
// A32: translational invariance — X-Sod rho must be constant in y,z after 4 steps
// =============================================================================
static void test_a32() {
    printf("\n-- A32  Translational invariance (X-Sod constant in y,z)  (tol 1e-10) --\n");
    const double cfl = 0.3;
    const int NSTEP  = 4;

    BlockTree tree = run_teno5a(NSTEP, cfl, 0);  // X-Sod
    const CellBlock& blk = *tree.nodes[0].block;

    // For each interior i, rho[i][j][k] must be equal for all (j,k).
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
    check(max_spread < 1.0e-10, "A32",
          "X-Sod rho constant in y,z (translational invariance, tol 1e-10)", max_spread);
}

// =============================================================================
// A33: axis symmetry — X/Y/Z Sod give same max(rho) over 4 steps
// =============================================================================
static void test_a33() {
    printf("\n-- A33  Axis symmetry: X/Y/Z Sod max(rho) equal  (tol 1e-10) --\n");
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

    BlockTree tx = run_teno5a(NSTEP, cfl, 0);
    BlockTree ty = run_teno5a(NSTEP, cfl, 1);
    BlockTree tz = run_teno5a(NSTEP, cfl, 2);

    const double rx = max_rho(tx);
    const double ry = max_rho(ty);
    const double rz = max_rho(tz);
    const double err_xy = std::fabs(rx - ry) / std::max(rx, 1.0e-12);
    const double err_xz = std::fabs(rx - rz) / std::max(rx, 1.0e-12);
    printf("   max_rho X=%.6e  Y=%.6e  Z=%.6e\n", rx, ry, rz);
    printf("   rel |X-Y|=%.3e  rel |X-Z|=%.3e  (tol 1e-10)\n", err_xy, err_xz);
    check(err_xy < 1.0e-10, "A33a", "X-Sod max(rho) == Y-Sod max(rho) (tol 1e-10)", err_xy);
    check(err_xz < 1.0e-10, "A33b", "X-Sod max(rho) == Z-Sod max(rho) (tol 1e-10)", err_xz);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== D3 GPU TENO5-A gate test (t31) ===\n");
    test_a31();
    test_a32();
    test_a33();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
