// test_t46_wmles_ode_gpu.cu — G4: GPU ODE mixing-length wall model gate
//
// T46a: d_wm_ode_ml (host-callable) matches CPU wm_ode_ml within 1e-8
//       at 7 y+ values spanning viscous sublayer to log region.
// T46b: GPU ghost cells with use_ode=true match CPU wm_apply_ghost
//       (use_ode=true) within 1e-7 relative error.
// T46c: GPU WMLES ODE exec_apply called 5× on a wall block — no NaN/crash
//       and rho remains positive.

#include "cuda/gpu_wmles.cuh"
#include "models/wall_model.hpp"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
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

static void free_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
}

// =============================================================================
// T46a: d_wm_ode_ml (host-callable) matches CPU wm_ode_ml within 1e-8
// =============================================================================
static void test_t46a() {
    printf("\n-- T46a  d_wm_ode_ml matches CPU wm_ode_ml within 1e-8 --\n");

    WallModelCfg cfg;
    cfg.use_ode = true;
    const double nu       = 1.0 / 395.0;
    const double utau_ref = 1.0;

    const double yp_tests[] = { 1.0, 5.0, 11.0, 30.0, 100.0, 200.0, 395.0 };
    const int N = (int)(sizeof(yp_tests) / sizeof(double));

    double max_rel = 0.0;
    bool all_ok = true;
    for (int n = 0; n < N; ++n) {
        const double yp  = yp_tests[n];
        const double y_m = yp * nu / utau_ref;
        const double u_t = utau_ref * d_reichardt_uplus(yp, cfg.kappa);

        const double utau_gpu = d_wm_ode_ml(u_t, y_m, nu, cfg.kappa, cfg.A_plus, cfg.tol);
        const double utau_cpu = wm_ode_ml(u_t, y_m, nu, cfg);

        const double rel = (utau_cpu > 1e-30)
                           ? std::fabs(utau_gpu - utau_cpu) / utau_cpu
                           : 0.0;
        max_rel = std::max(max_rel, rel);
        if (rel >= 1e-8) all_ok = false;
        printf("   y+=%.1f: gpu=%.10f  cpu=%.10f  rel=%.2e\n",
               yp, utau_gpu, utau_cpu, rel);
    }
    check(all_ok, "T46a",
          "d_wm_ode_ml matches CPU wm_ode_ml within 1e-8 at 7 y+ values", max_rel);
}

// =============================================================================
// T46b: GPU ghost cells (use_ode=true) match CPU wm_apply_ghost within 1e-7
// =============================================================================
static void test_t46b() {
    printf("\n-- T46b  GPU ghost cells (ODE) match CPU wm_apply_ghost within 1e-7 --\n");

    WallModelCfg cfg;
    cfg.use_ode = true;
    const double nu  = 1.0 / 395.0;
    const double rho = 1.0;

    BlockTree tree; tree.init(1.0); tree.set_periodic(false);
    CellBlock& blk = *tree.nodes[0].block;
    const double h   = blk.h;
    const double y_m = 0.5 * h;

    const double yp_ref = y_m / nu;
    const double u_t    = d_reichardt_uplus(yp_ref, cfg.kappa);

    for (int flat = 0; flat < NCELL; ++flat) {
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = 0.0;
        blk.Q[2][flat] = 0.0;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = 1.0e5 / (GAMMA - 1.0);
    }
    for (int k = NG; k < NG+NB; ++k)
    for (int i = NG; i < NG+NB; ++i)
        blk.Q[1][cell_idx(i, NG, k)] = rho * u_t;

    // CPU reference: apply ghost cells with ODE model.
    CellBlock blk_cpu = blk;
    wm_apply_wall(blk_cpu, /*wall_ax=*/1, /*side=*/0, nu, cfg);

    // GPU: upload, run ODE WMLES kernel, download.
    pool.alloc(&blk); pool.upload(&blk);
    GpuWmlesList wml;
    wml.build_from_tree(tree, pool, /*wall_ax=*/1, /*side=*/0);
    wml.exec_apply(nu, cfg, nullptr);
    cudaDeviceSynchronize();
    pool.download(&blk);

    // Compare all NG ghost layers.
    double max_rel = 0.0;
    bool all_ok = true;
    for (int gl = 1; gl <= NG; ++gl) {
        const int jg = NG - gl;
        for (int k = NG; k < NG+NB; ++k)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i, jg, k);
            for (int v = 0; v < NVAR; ++v) {
                const double qc = blk_cpu.Q[v][f];
                const double qg = blk.Q[v][f];
                const double sc = std::fmax(std::fabs(qc), std::fabs(qg));
                if (sc < 1.0e-10) continue;
                const double rel = std::fabs(qg - qc) / sc;
                if (rel > max_rel) max_rel = rel;
                if (rel >= 1e-7)   all_ok = false;
            }
        }
    }
    printf("   max rel err ghost cells (ODE) = %.3e  (tol 1e-7)\n", max_rel);
    check(all_ok, "T46b",
          "GPU ghost cells (ODE) match CPU wm_apply_ghost within 1e-7", max_rel);
    free_all(tree);
}

// =============================================================================
// T46c: exec_apply (ODE) called 5× — rho stays positive, no NaN/inf.
// =============================================================================
static void test_t46c() {
    printf("\n-- T46c  GPU WMLES ODE 5x exec_apply — rho > 0, no NaN --\n");

    WallModelCfg cfg;
    cfg.use_ode = true;
    const double nu  = 1.0 / 395.0;
    const double rho = 1.2;

    BlockTree tree; tree.init(1.0); tree.set_periodic(false);
    CellBlock& blk = *tree.nodes[0].block;
    const double h   = blk.h;
    const double u_t = d_reichardt_uplus(0.5 * h / nu, cfg.kappa);

    for (int flat = 0; flat < NCELL; ++flat) {
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = rho * u_t;
        blk.Q[2][flat] = 0.0;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = 1.0e5 / (GAMMA - 1.0) + 0.5 * rho * u_t * u_t;
    }

    pool.alloc(&blk); pool.upload(&blk);

    GpuWmlesList wml;
    wml.build_from_tree(tree, pool, /*wall_ax=*/1, /*side=*/0);

    bool crashed = false;
    for (int s = 0; s < 5 && !crashed; ++s) {
        wml.exec_apply(nu, cfg, nullptr);
        cudaDeviceSynchronize();
        pool.download(&blk);
        const double rho_sample = blk.Q[0][cell_idx(NG, NG, NG)];
        if (rho_sample <= 0.0 || std::isnan(rho_sample)) crashed = true;
    }
    free_all(tree);

    check(!crashed, "T46c", "GPU WMLES ODE 5x exec_apply rho > 0 and finite");
}

// =============================================================================
int main() {
    printf("=== G4 GPU ODE mixing-length wall model gate test ===\n");
    test_t46a();
    test_t46b();
    test_t46c();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
