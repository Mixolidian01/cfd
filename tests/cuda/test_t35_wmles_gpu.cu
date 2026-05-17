// D7 gate — GPU-resident WMLES algebraic Reichardt wall model.
//
// W61: d_wm_log_law (GPU Newton, host-called) matches CPU wm_log_law within 1e-8
//      at 7 y+ values spanning viscous sublayer to log region.
// W62: GPU ghost cells encode τ_w = ρu_τ² correctly (Newton tol 1e-7).
// W63: u⁺ = u_t/u_τ self-consistent with Reichardt(y⁺) within 0.1%.
// W64: Reichardt log-law intercept B = u⁺ − (1/κ)ln(y⁺) ∈ [5.0, 6.5]
//      at y⁺ ∈ [50, 395] (Reichardt formula asymptotes to B ≈ 5.6 for κ=0.41).

#include "cuda/gpu_wmles.cuh"
#include "models/wall_model.hpp"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
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

static void free_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
}

// ── W61: d_wm_log_law (host-callable __host__ __device__) vs CPU wm_log_law ──
static void test_w61() {
    printf("\n-- W61  d_wm_log_law matches CPU wm_log_law within 1e-8 --\n");

    WallModelCfg cfg;                    // κ=0.41, B=5.2, tol=1e-8
    const double nu      = 1.0 / 395.0;
    const double utau_ref = 1.0;

    // Test at y+ spanning viscous sublayer through log region.
    const double yp_tests[] = { 1.0, 5.0, 11.0, 30.0, 100.0, 200.0, 395.0 };
    const int N = (int)(sizeof(yp_tests) / sizeof(double));

    double max_rel = 0.0;
    bool all_ok = true;
    for (int n = 0; n < N; ++n) {
        const double yp  = yp_tests[n];
        const double y_m = yp * nu / utau_ref;
        const double u_t = utau_ref * d_reichardt_uplus(yp, cfg.kappa);

        const double utau_gpu = d_wm_log_law(u_t, y_m, nu, cfg.kappa, cfg.B, cfg.tol);
        const double utau_cpu = wm_log_law(u_t, y_m, nu, cfg);

        const double rel = (utau_cpu > 1e-30)
                           ? std::fabs(utau_gpu - utau_cpu) / utau_cpu
                           : 0.0;
        max_rel = std::max(max_rel, rel);
        if (rel >= 1e-8) all_ok = false;
        printf("   y+=%.1f: gpu=%.10f  cpu=%.10f  rel=%.2e\n",
               yp, utau_gpu, utau_cpu, rel);
    }
    check(all_ok, "W61",
          "d_wm_log_law matches CPU wm_log_law within 1e-8 at 7 y+ values", max_rel);
}

// ── W62 + W63: GPU ghost cells ─────────────────────────────────────────────────
static void test_w62_w63() {
    printf("\n-- W62  GPU ghost cells encode τ_w = ρu_τ² within Newton tol --\n");
    printf("-- W63  u⁺ self-consistent with Reichardt within 0.1%% --\n");

    WallModelCfg cfg;
    const double nu  = 1.0 / 395.0;
    const double rho = 1.0;

    BlockTree tree; tree.init(1.0); tree.set_periodic(false);
    CellBlock& blk = *tree.nodes[0].block;
    const double h   = blk.h;         // 1/NB = 0.125
    const double y_m = 0.5 * h;

    // Reference: utau=1 → y+ = y_m/ν; u_t = u+ at that y+.
    const double yp_ref = y_m / nu;    // ≈ 24.7
    const double u_t    = d_reichardt_uplus(yp_ref, cfg.kappa);  // utau=1 → u_t=u+

    // Initialise block; set bottom interior layer (j=NG) with tangential u_x=u_t.
    for (int flat = 0; flat < NCELL; ++flat) {
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = 0.0;
        blk.Q[2][flat] = 0.0;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = 1.0;
    }
    for (int k = NG; k < NG+NB; ++k)
    for (int i = NG; i < NG+NB; ++i)
        blk.Q[1][cell_idx(i, NG, k)] = rho * u_t;

    pool.alloc(&blk); pool.upload(&blk);

    GpuWmlesList wml;
    wml.build_from_tree(tree, pool, /*wall_ax=*/1, /*side=*/0);
    wml.exec_apply(nu, cfg, nullptr);
    cudaDeviceSynchronize();
    pool.download(&blk);

    // CPU reference.
    const double utau_cpu = wm_log_law(u_t, y_m, nu, cfg);
    const double tau_w_cpu = rho * utau_cpu * utau_cpu;
    const double mu        = rho * nu;

    // Sample centre cell (i=NG+NB/2, k=NG+NB/2).
    const int ic = NG + NB/2, kc = NG + NB/2;

    // W62: ghost at j=NG-1 (step=1, gl=0).
    const int idx_g1 = cell_idx(ic, NG - 1, kc);
    const double rho_g  = blk.Q[0][idx_g1];
    const double ug_x   = blk.Q[1][idx_g1] / rho_g;
    // τ_w_gpu = μ·(u_t − u_ghost)/h  (step=1 ghost at y=−h/2 from wall)
    const double tau_w_gpu = mu * (u_t - ug_x) / h;
    const double rel62 = std::fabs(tau_w_gpu - tau_w_cpu)
                         / (tau_w_cpu + 1e-30);
    printf("   W62: tau_w_gpu=%.8f  tau_w_cpu=%.8f  rel=%.3e\n",
           tau_w_gpu, tau_w_cpu, rel62);
    check(rel62 < 1e-7, "W62",
          "Ghost-cell τ_w = ρu_τ² matches CPU reference within 1e-7", rel62);

    // W63: u+ self-consistency.
    const double utau_w63 = std::sqrt(tau_w_gpu / rho);
    const double yp_meas  = y_m * utau_w63 / nu;
    const double up_meas  = u_t / utau_w63;
    const double up_reich = d_reichardt_uplus(yp_meas, cfg.kappa);
    const double rel63    = std::fabs(up_meas - up_reich) / (up_reich + 1e-30);
    printf("   W63: u+=%.6f  Reichardt(y+=%.4f)=%.6f  rel=%.3e\n",
           up_meas, yp_meas, up_reich, rel63);
    check(rel63 < 0.001, "W63",
          "u+ = u_t/u_τ self-consistent with Reichardt(y+) within 0.1%%", rel63);

    free_all(tree);
}

// ── W64: log-law intercept B over log region ──────────────────────────────────
// Reichardt formula with κ=0.41 asymptotes to B ≈ (1/κ)ln(κ) + 7.8 ≈ 5.6
// in the log region.  Gate: B ∈ [5.0, 6.5] for y⁺ ∈ [50, 395].
static void test_w64() {
    printf("\n-- W64  Reichardt log-law intercept B ∈ [5.0, 6.5] at y⁺ ∈ [50, 395] --\n");

    WallModelCfg cfg;
    const double nu       = 1.0 / 395.0;
    const double utau_ref = 1.0;

    const int yp_arr[] = { 50, 75, 100, 150, 200, 300, 395 };
    const int N = (int)(sizeof(yp_arr) / sizeof(int));

    double min_B = 1e30, max_B = -1e30;
    bool all_ok = true;
    for (int n = 0; n < N; ++n) {
        const double yp  = (double)yp_arr[n];
        const double y_m = yp * nu / utau_ref;
        const double up  = d_reichardt_uplus(yp, cfg.kappa);
        const double u_t = utau_ref * up;

        const double utau_meas = d_wm_log_law(u_t, y_m, nu, cfg.kappa, cfg.B, cfg.tol);
        const double yp_meas   = y_m * utau_meas / nu;
        const double up_meas   = u_t / utau_meas;
        const double B_meas    = up_meas - (1.0 / cfg.kappa) * std::log(yp_meas);

        min_B = std::min(min_B, B_meas);
        max_B = std::max(max_B, B_meas);
        const bool ok = (B_meas >= 5.0 && B_meas <= 6.5);
        if (!ok) all_ok = false;
        printf("   y+=%3d: u+=%.4f  B=%.4f  %s\n",
               (int)yp, up_meas, B_meas, ok ? "ok" : "FAIL");
    }
    printf("   B range: [%.4f, %.4f]  (gate [5.0, 6.5])\n", min_B, max_B);
    check(all_ok, "W64",
          "Reichardt log-law intercept B ∈ [5.0, 6.5] for y⁺ ∈ [50, 395]");
}

// =============================================================================
int main() {
    printf("=== D7 GPU WMLES algebraic wall model gate test ===\n");
    test_w61();
    test_w62_w63();
    test_w64();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
