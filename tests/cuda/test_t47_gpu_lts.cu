// test_t47_gpu_lts.cu — G5: GPU Berger-Oliger LTS integrator gate
//
// T47a: GPU LTS runs 10 steps on a 2-level periodic tree without crash.
// T47b: Mass is conserved (|Δm|/m < 1e-10) over 10 LTS steps (periodic).
// T47c: dt_f ≤ dt_c / r and dt_c > 0 (CFL constraints respected).

#include "cuda/gpu_lts.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>
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

// Smooth 2-level tree: root block refined → 8 fine children + 0 coarse
// (root is NOT a leaf after refinement).  For a 2-level tree, the root becomes
// an internal node with 8 children as fine leaves.  Coarse leaf = ?
//
// Actually, BlockTree::refine(0) replaces the root leaf with 8 child leaves,
// so ALL leaves are at level 1.  To get a mixed level tree we need to:
//   1. refine(0): root → 8 children at level 1
//   2. refine one child: gives 8 grandchildren at level 2
// Now: L_min=1 (7 coarse leaves), L_max=2 (8 fine leaves).
static void make_2level_tree(BlockTree& tree) {
    tree.init(1.0);
    tree.set_periodic(true);
    // Refine root → 8 leaves at level 1
    tree.refine(0);
    // Refine first child → 8 leaves at level 2 (fine)
    // The children of the root are nodes 1..8
    const auto& lvl1 = tree.leaf_indices();
    if (!lvl1.empty()) tree.refine(lvl1[0]);
}

// Global-coordinate fill: use domain size L=1 so the function is C∞ across C/F
// interfaces regardless of block level.  L = NB*h varies per level, so using
// per-block L would create a ~0.04 density jump at coarse/fine boundaries.
static void fill_smooth(CellBlock& blk, double ox, double oy, double oz) {
    constexpr double L = 1.0;
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = ox + (i - NG + 0.5) * blk.h;
        double y = oy + (j - NG + 0.5) * blk.h;
        double z = oz + (k - NG + 0.5) * blk.h;
        double rho = 1.2 + 0.05*sin(2*M_PI*x/L)*cos(2*M_PI*y/L);
        double u   = 0.3*cos(2*M_PI*x/L)*sin(2*M_PI*y/L)*cos(2*M_PI*z/L);
        double v   =-0.3*sin(2*M_PI*x/L)*cos(2*M_PI*y/L)*cos(2*M_PI*z/L);
        double w   = 0.1*cos(2*M_PI*x/L)*cos(2*M_PI*y/L)*sin(2*M_PI*z/L);
        double p   = 1.0e5;
        int f = cell_idx(i,j,k);
        blk.Q[0][f] = rho;
        blk.Q[1][f] = rho * u;
        blk.Q[2][f] = rho * v;
        blk.Q[3][f] = rho * w;
        blk.Q[4][f] = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
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

static void download_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk || !pool.has_device(blk)) continue;
        pool.download(blk);
    }
}

static void free_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
}

// Physical mass: sum(rho * h^3) per interior cell — the conserved quantity for AMR.
// Unweighted digital sum is NOT conserved when fine and coarse cells have different h.
static double total_mass(const BlockTree& tree) {
    double mass = 0.0;
    for (int li : tree.leaf_indices()) {
        const CellBlock& blk = *tree.nodes[li].block;
        const double dV = blk.h * blk.h * blk.h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            mass += blk.Q[0][cell_idx(i,j,k)] * dV;
    }
    return mass;
}

// =============================================================================
// T47a: GPU LTS runs 10 steps without crash (rho > 0 throughout).
// =============================================================================
static void test_t47a() {
    printf("\n-- T47a  GPU LTS 10 steps no crash (rho > 0) --\n");

    BlockTree tree;
    make_2level_tree(tree);
    const int L_min = tree.min_leaf_level();
    const int L_max = tree.max_leaf_level();
    printf("   2-level tree: L_min=%d  L_max=%d  n_leaves=%d\n",
           L_min, L_max, (int)tree.leaf_indices().size());

    for (int li : tree.leaf_indices())
        fill_smooth(*tree.nodes[li].block,
                    tree.nodes[li].ox, tree.nodes[li].oy, tree.nodes[li].oz);
    upload_all(tree);
    tree.fill_ghosts_periodic();

    GpuLtsIntegrator lts;
    lts.build(tree, pool, /*bc_type=*/0, /*r=*/2);
    printf("   n_fine=%d  n_coarse=%d\n", lts.n_fine, lts.n_coarse);

    bool crashed = false;
    for (int s = 0; s < 10 && !crashed; ++s) {
        lts.step(0.4);
        download_all(tree);
        double rho_min_fine = 1e300, rho_min_coarse = 1e300;
        double rho_max_fine = -1e300, rho_max_coarse = -1e300;
        for (int li : tree.leaf_indices()) {
            const CellBlock& blk = *tree.nodes[li].block;
            const int lvl = tree.nodes[li].level;
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j)
            for (int i = NG; i < NG+NB; ++i) {
                double r = blk.Q[0][cell_idx(i,j,k)];
                if (lvl == L_max) { rho_min_fine   = std::min(rho_min_fine,   r); rho_max_fine   = std::max(rho_max_fine,   r); }
                else              { rho_min_coarse = std::min(rho_min_coarse, r); rho_max_coarse = std::max(rho_max_coarse, r); }
            }
            if (!std::isfinite(blk.Q[0][cell_idx(NG,NG,NG)]) ||
                blk.Q[0][cell_idx(NG,NG,NG)] <= 0.0) crashed = true;
        }
        printf("   step=%d  rho_fine=[%.4e,%.4e]  rho_coarse=[%.4e,%.4e]%s\n",
               s, rho_min_fine, rho_max_fine, rho_min_coarse, rho_max_coarse,
               crashed ? " CRASH" : "");
    }
    free_all(tree);

    check(!crashed, "T47a", "GPU LTS 10 steps no crash (rho > 0)");
}

// =============================================================================
// T47b: Mass conservation |Δm|/m < 1e-10 over 10 LTS steps (periodic).
// =============================================================================
static void test_t47b() {
    printf("\n-- T47b  Mass conserved |Δm|/m < 1e-10 over 10 LTS steps --\n");

    BlockTree tree;
    make_2level_tree(tree);
    for (int li : tree.leaf_indices())
        fill_smooth(*tree.nodes[li].block,
                    tree.nodes[li].ox, tree.nodes[li].oy, tree.nodes[li].oz);
    upload_all(tree);

    const double mass0 = total_mass(tree);

    GpuLtsIntegrator lts;
    lts.build(tree, pool, /*bc_type=*/0, /*r=*/2);

    for (int s = 0; s < 10; ++s) lts.step(0.4);
    download_all(tree);
    const double mass1 = total_mass(tree);
    const double err   = std::fabs(mass1 - mass0) / (mass0 + 1e-300);
    printf("   mass_err = %.3e  (tol 1e-10)\n", err);
    free_all(tree);

    check(err < 1e-10, "T47b", "GPU LTS mass conservation |Δm|/m < 1e-10", err);
}

// =============================================================================
// T47c: dt_f ≤ dt_c / r (fine CFL is more restrictive than coarse / ratio).
// =============================================================================
static void test_t47c() {
    printf("\n-- T47c  dt_f ≤ dt_c/r — CFL ratio constraint respected --\n");

    BlockTree tree;
    make_2level_tree(tree);
    for (int li : tree.leaf_indices())
        fill_smooth(*tree.nodes[li].block,
                    tree.nodes[li].ox, tree.nodes[li].oy, tree.nodes[li].oz);
    upload_all(tree);

    GpuLtsIntegrator lts;
    lts.build(tree, pool, /*bc_type=*/0, /*r=*/2);

    // Check CFL: dt_fine from fine list, dt_coarse from coarse list
    const double cfl    = 0.4;
    const double dt_f   = lts.cfl_fine.exec(cfl, nullptr);
    const double dt_c   = lts.cfl_coarse.exec(cfl, nullptr);
    const double dt_c_r = dt_c / (double)lts.lts_r;

    printf("   dt_f = %.6e  dt_c = %.6e  dt_c/r = %.6e\n",
           dt_f, dt_c, dt_c_r);
    // Fine cells are 2x smaller → dt_f ≤ dt_c / 2 (within 10% tolerance for rounding)
    const bool ok = (dt_f > 0.0 && dt_c > 0.0 && dt_f <= dt_c_r * 1.1);
    free_all(tree);

    check(ok, "T47c", "GPU LTS dt_f ≤ dt_c/r — CFL ratio constraint (±10%)");
}

// =============================================================================
int main() {
    printf("=== G5 GPU Berger-Oliger LTS integrator gate test ===\n");
    test_t47a();
    test_t47b();
    test_t47c();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
