// P8.5 gate test — GPU CFL warp-shuffle tree reduction
//
// Protocol:
//   C1: single block, GPU dt matches CPU cfl_dt (< 1e-12 relative error)
//   C2: 8-leaf tree, GPU global min matches CPU global min
//   C3: d_dt stays on device (pointer-dt pattern: read back without exec())
//   C4: exec() twice in a row returns consistent dt (reset logic correct)

#include "../../include/cuda/gpu_cfl.cuh"
#include "../../include/gpu_pool.hpp"
#include "../../include/block_tree.hpp"
#include "../../include/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

static int nfail = 0;
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

// CPU CFL dt for one block (from prim variables already in Q)
static double cpu_cfl_dt(const CellBlock& blk, double cfl) {
    double dt = 1.0e300;
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int flat = cell_idx(i,j,k);
        double rho  = blk.Q[0][flat];
        double rhou = blk.Q[1][flat];
        double rhov = blk.Q[2][flat];
        double rhow = blk.Q[3][flat];
        double E    = blk.Q[4][flat];
        double inv_rho = 1.0 / rho;
        double u = rhou*inv_rho, v = rhov*inv_rho, w = rhow*inv_rho;
        double p = (GAMMA-1.0)*(E - 0.5*rho*(u*u+v*v+w*w));
        double c = std::sqrt(GAMMA*p*inv_rho);
        double sp = std::fmax(std::fabs(u)+c, std::fmax(std::fabs(v)+c, std::fabs(w)+c));
        if (sp > 0.0) dt = std::fmin(dt, cfl * blk.h / sp);
    }
    return dt;
}

static void fill_sod(CellBlock& blk, double seed) {
    // Left state: high pressure; right state: low pressure
    const double h = blk.h;
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = (i - NG + 0.5) * h;
        bool left = (x < seed * 0.5 * NB * h);
        double rho = left ? 1.0 : 0.125;
        double p   = left ? 1.0e5 : 1.0e4;
        double E   = p / (GAMMA - 1.0);
        int flat   = cell_idx(i,j,k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = 0.0;
        blk.Q[2][flat] = 0.0;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = E;
    }
}

static void upload_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        if (!blk->d_Q) pool.alloc(blk);
        pool.upload(blk);
    }
}

static void free_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && blk->d_Q) pool.free(blk);
    }
}

// =============================================================================
// C1: single block, GPU dt == CPU dt
// =============================================================================
static void test_c1() {
    printf("\n-- C1  Single block, GPU vs CPU CFL dt --\n");
    const double cfl = 0.5;

    BlockTree tree; tree.init(1.0);
    CellBlock& blk = *tree.nodes[0].block;
    fill_sod(blk, 1.0);

    const double cpu_dt = cpu_cfl_dt(blk, cfl);

    upload_all(tree);
    GpuCflList cfl_list; cfl_list.build(tree);
    const double gpu_dt = cfl_list.exec(cfl);
    free_all(tree);

    const double rel_err = std::fabs(gpu_dt - cpu_dt) / cpu_dt;
    printf("   cpu_dt = %.6e  gpu_dt = %.6e  rel_err = %.3e  (tol 1e-12)\n",
           cpu_dt, gpu_dt, rel_err);
    check(rel_err < 1.0e-12, "C1", "GPU dt == CPU dt (single block)", rel_err);
}

// =============================================================================
// C2: 8-leaf tree, GPU global min == CPU global min
// =============================================================================
static void test_c2() {
    printf("\n-- C2  8-leaf tree, GPU global min == CPU global min --\n");
    const double cfl = 0.5;

    BlockTree tree; tree.init(1.0);
    tree.set_periodic(true);
    tree.refine(0);  // 8 level-1 leaves

    const auto& leaves = tree.leaf_indices();
    assert(leaves.size() == 8);

    double cpu_global_dt = 1.0e300;
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        CellBlock& blk = *tree.nodes[leaves[ii]].block;
        fill_sod(blk, 0.5 + 0.1*ii);
        cpu_global_dt = std::fmin(cpu_global_dt, cpu_cfl_dt(blk, cfl));
    }

    upload_all(tree);
    GpuCflList cfl_list; cfl_list.build(tree);
    const double gpu_dt = cfl_list.exec(cfl);
    free_all(tree);

    const double rel_err = std::fabs(gpu_dt - cpu_global_dt) / cpu_global_dt;
    printf("   cpu_dt = %.6e  gpu_dt = %.6e  rel_err = %.3e  (tol 1e-12)\n",
           cpu_global_dt, gpu_dt, rel_err);
    check(rel_err < 1.0e-12, "C2", "GPU global min dt == CPU min (8 leaves)", rel_err);
}

// =============================================================================
// C3: d_dt device pointer readable without calling exec() again
// =============================================================================
static void test_c3() {
    printf("\n-- C3  d_dt device pointer readable after exec() --\n");
    const double cfl = 0.4;

    BlockTree tree; tree.init(1.0);
    CellBlock& blk = *tree.nodes[0].block;
    fill_sod(blk, 1.0);
    upload_all(tree);

    GpuCflList cfl_list; cfl_list.build(tree);
    double dt_from_exec = cfl_list.exec(cfl);

    // Read d_dt directly from device (no second exec())
    double dt_from_ptr = 0.0;
    cudaMemcpy(&dt_from_ptr, cfl_list.d_dt, sizeof(double),
               cudaMemcpyDeviceToHost);
    free_all(tree);

    printf("   exec() = %.6e  d_dt = %.6e\n", dt_from_exec, dt_from_ptr);
    check(dt_from_exec == dt_from_ptr, "C3",
          "d_dt device pointer holds same value as exec() return");
}

// =============================================================================
// C4: exec() twice gives the same result (reset logic correct)
// =============================================================================
static void test_c4() {
    printf("\n-- C4  exec() twice → consistent dt (reset verified) --\n");
    const double cfl = 0.5;

    BlockTree tree; tree.init(1.0);
    CellBlock& blk = *tree.nodes[0].block;
    fill_sod(blk, 1.0);
    upload_all(tree);

    GpuCflList cfl_list; cfl_list.build(tree);
    const double dt1 = cfl_list.exec(cfl);
    const double dt2 = cfl_list.exec(cfl);
    free_all(tree);

    printf("   dt1 = %.6e  dt2 = %.6e\n", dt1, dt2);
    check(dt1 == dt2, "C4", "exec() twice gives identical dt (reset logic OK)");
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P8.5 GPU CFL warp-shuffle reduction gate test ===\n");
    test_c1();
    test_c2();
    test_c3();
    test_c4();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
