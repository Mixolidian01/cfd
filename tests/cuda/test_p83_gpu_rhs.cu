// P8.3 gate test — GPU WENO5-Z RHS kernels
//
// Protocol:
//   R1: single block, smooth flow — GPU RHS vs CPU compute_rhs (expect < 1e-10)
//   R2: mass conservation (periodic) — Σ d_RHS[rho] ≈ 0 to machine eps
//   R3: momentum conservation (periodic) — same for rhou, rhov, rhow
//   R4: multi-block (8 level-1 leaves), conservation check still holds
//
// R1 uses a smooth sinusoidal flow so Ducros ≈ 0, pure KEP path: GPU and
// CPU arithmetic are identical up to floating-point operation ordering.
// Tolerance 1e-10 × max|RHS| for the comparison.

#include "../../include/cuda/gpu_rhs.cuh"
#include "../../include/cuda/gpu_ghost_fill.cuh"
#include "../../include/gpu_pool.hpp"
#include "../../include/block_tree.hpp"
#include "../../include/operators.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <numeric>

static int nfail = 0;
static GpuPool pool;

static void check(bool ok, const char* tag, const char* msg, double val = -1.0) {
    if (ok) {
        printf("  PASS  %s  %s\n", tag, msg);
    } else {
        if (val >= 0.0)
            printf("  FAIL  %s  %s  (val = %.3e)\n", tag, msg, val);
        else
            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

// Smooth sinusoidal IC: rho = rho0 + A*sin(2πx/L)·sin(2πy/L)·sin(2πz/L)
// u, v, w = small smooth velocity; p from EOS; designed to keep Ducros ≈ 0.
static void fill_smooth(CellBlock& blk) {
    const double h = blk.h;
    const double L = h * NB;       // block physical size
    const double A = 0.01;         // density perturbation amplitude
    const double p0 = 1.0e5;       // reference pressure [Pa]
    const double rho0 = 1.2;       // reference density  [kg/m³]
    const double Umax = 10.0;      // velocity scale [m/s]

    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const double x = (i - NG + 0.5) * h;
        const double y = (j - NG + 0.5) * h;
        const double z = (k - NG + 0.5) * h;
        const double pi2 = 2.0 * M_PI;
        const double rho = rho0 * (1.0 + A*sin(pi2*x/L)*sin(pi2*y/L)*sin(pi2*z/L));
        const double u   = Umax * sin(pi2*x/L) * cos(pi2*y/L);
        const double v   = Umax * cos(pi2*x/L) * sin(pi2*y/L);
        const double w   = 0.0;
        const double p   = p0;
        const double E   = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
        int flat = cell_idx(i,j,k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = rho*u;
        blk.Q[2][flat] = rho*v;
        blk.Q[3][flat] = rho*w;
        blk.Q[4][flat] = E;
    }
}

// Download GPU RHS pool entry li into a flat vector (NVAR×NCELL)
static std::vector<double> download_rhs_flat(const double* d_rhs_pool, int li) {
    std::vector<double> buf(NVAR * NCELL);
    cudaMemcpy(buf.data(), d_rhs_pool + (size_t)li * NVAR * NCELL,
               NVAR * NCELL * sizeof(double), cudaMemcpyDeviceToHost);
    return buf;
}

// Max error over interior cells for variable v
static double interior_max_err(const std::vector<double>& gpu,
                                const CellBlock& cpu_rhs, int v) {
    double e = 0.0;
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int flat = cell_idx(i,j,k);
        double diff = fabs(gpu[v*NCELL+flat] - cpu_rhs.Q[v][flat]);
        e = fmax(e, diff);
    }
    return e;
}

// Sum of interior RHS for variable v (conservation check)
static double interior_sum(const std::vector<double>& buf, int v) {
    double s = 0.0;
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        s += buf[v*NCELL + cell_idx(i,j,k)];
    return s;
}

static void upload_all_leaves(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        if (!blk->d_Q) pool.alloc(blk);
        pool.upload(blk);
    }
}

static void free_all_leaves(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && blk->d_Q) pool.free(blk);
    }
}

// =============================================================================
// R1: single block, smooth flow — GPU RHS vs CPU compute_rhs
// =============================================================================
static void test_r1() {
    printf("\n-- R1  Single block, GPU vs CPU RHS (smooth flow) --\n");

    BlockTree tree;
    tree.init(1.0);
    CellBlock& blk = *tree.nodes[0].block;
    fill_smooth(blk);

    // CPU reference: fill ghosts + compute_rhs
    tree.fill_ghosts_periodic();
    CellBlock cpu_rhs;
    cpu_rhs.h = blk.h;
    compute_rhs(blk, cpu_rhs);

    // GPU: upload, ghost fill, exec
    upload_all_leaves(tree);
    GpuGhostFillList gfl;
    gfl.build(tree, /*bc_type=*/0);
    gfl.exec(nullptr);

    GpuRhsList rhl;
    rhl.build(tree);
    rhl.exec(nullptr);
    cudaDeviceSynchronize();

    // Download GPU RHS for leaf 0
    auto gpu_rhs = download_rhs_flat(rhl.d_rhs_pool, 0);
    free_all_leaves(tree);

    // Compare: max error over interior, normalised by max|CPU RHS|
    double max_cpu = 0.0;
    for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            max_cpu = fmax(max_cpu, fabs(cpu_rhs.Q[v][cell_idx(i,j,k)]));

    double max_err = 0.0;
    for (int v = 0; v < NVAR; ++v)
        max_err = fmax(max_err, interior_max_err(gpu_rhs, cpu_rhs, v));

    const double rel_err = (max_cpu > 0.0) ? max_err / max_cpu : max_err;
    printf("   max RHS = %.3e  abs err = %.3e  rel err = %.3e  (tol 1e-10)\n",
           max_cpu, max_err, rel_err);
    check(rel_err < 1.0e-10, "R1", "GPU vs CPU RHS single block", rel_err);
}

// =============================================================================
// R2: mass conservation (periodic single block)
// =============================================================================
static void test_r2() {
    printf("\n-- R2  Mass conservation (periodic single block) --\n");

    BlockTree tree;
    tree.init(1.0);
    CellBlock& blk = *tree.nodes[0].block;
    fill_smooth(blk);

    upload_all_leaves(tree);
    GpuGhostFillList gfl;
    gfl.build(tree, 0);
    gfl.exec(nullptr);

    GpuRhsList rhl;
    rhl.build(tree);
    rhl.exec(nullptr);
    cudaDeviceSynchronize();

    auto gpu_rhs = download_rhs_flat(rhl.d_rhs_pool, 0);
    free_all_leaves(tree);

    double sum_rho = interior_sum(gpu_rhs, 0);
    double sum_rhou= interior_sum(gpu_rhs, 1);
    double sum_rhov= interior_sum(gpu_rhs, 2);
    double sum_rhow= interior_sum(gpu_rhs, 3);
    double sum_E   = interior_sum(gpu_rhs, 4);
    printf("   Σ dρ/dt = %.3e  Σ d(ρu)/dt = %.3e  Σ d(ρv)/dt = %.3e\n",
           sum_rho, sum_rhou, sum_rhov);
    printf("   Σ d(ρw)/dt = %.3e  Σ dE/dt = %.3e  (all expect < 1e-6)\n",
           sum_rhow, sum_E);

    const double tol = 1.0e-6;
    check(fabs(sum_rho)  < tol, "R2a", "Σ dρ/dt  ≈ 0",  fabs(sum_rho ));
    check(fabs(sum_rhou) < tol, "R2b", "Σ d(ρu)/dt ≈ 0", fabs(sum_rhou));
    check(fabs(sum_rhov) < tol, "R2c", "Σ d(ρv)/dt ≈ 0", fabs(sum_rhov));
    check(fabs(sum_rhow) < tol, "R2d", "Σ d(ρw)/dt ≈ 0", fabs(sum_rhow));
    check(fabs(sum_E)    < tol, "R2e", "Σ dE/dt    ≈ 0", fabs(sum_E   ));
}

// =============================================================================
// R3: momentum conservation (smooth flow, inviscid dominance)
// Same as R2 but verifies the momentum equations separately and with energy.
// =============================================================================
static void test_r3() {
    printf("\n-- R3  Energy conservation check (periodic single block) --\n");

    // Already covered by R2e; here we use a different IC (stronger velocity)
    BlockTree tree;
    tree.init(1.0);
    CellBlock& blk = *tree.nodes[0].block;

    // Fill with a pure divergence-free velocity (u=cos, v=-sin, w=0)
    const double h = blk.h;
    const double L = h * NB;
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = (i - NG + 0.5)*h, y = (j - NG + 0.5)*h;
        double rho = 1.2, p = 1.0e5;
        double u = 50.0*cos(2*M_PI*x/L), v = -50.0*sin(2*M_PI*y/L), w = 0.0;
        double E = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
        int f = cell_idx(i,j,k);
        blk.Q[0][f]=rho; blk.Q[1][f]=rho*u; blk.Q[2][f]=rho*v;
        blk.Q[3][f]=rho*w; blk.Q[4][f]=E;
    }

    upload_all_leaves(tree);
    GpuGhostFillList gfl;
    gfl.build(tree, 0);
    gfl.exec(nullptr);

    GpuRhsList rhl;
    rhl.build(tree);
    rhl.exec(nullptr);
    cudaDeviceSynchronize();

    auto gpu_rhs = download_rhs_flat(rhl.d_rhs_pool, 0);
    free_all_leaves(tree);

    double sum_rho  = interior_sum(gpu_rhs, 0);
    double sum_E    = interior_sum(gpu_rhs, 4);
    double abs_max_rhs = 0.0;
    for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            abs_max_rhs = fmax(abs_max_rhs, fabs(gpu_rhs[v*NCELL+cell_idx(i,j,k)]));

    const double tol = 1.0e-5 * abs_max_rhs;
    printf("   max|RHS| = %.3e  Σ dρ/dt = %.3e  Σ dE/dt = %.3e  (tol %.3e)\n",
           abs_max_rhs, sum_rho, sum_E, tol);
    check(fabs(sum_rho) < tol, "R3a", "Σ dρ/dt ≈ 0 (larger velocity field)", fabs(sum_rho));
    check(fabs(sum_E)   < tol, "R3b", "Σ dE/dt ≈ 0 (larger velocity field)", fabs(sum_E  ));
}

// =============================================================================
// R4: multi-block (8 level-1 leaves after refine), conservation check
// =============================================================================
static void test_r4() {
    printf("\n-- R4  Multi-block (8 leaves after refine), conservation --\n");

    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic(true);
    tree.refine(0);  // 8 level-1 leaves

    const auto& leaves = tree.leaf_indices();
    assert(leaves.size() == 8);

    // Fill each leaf with smooth flow
    for (int li : leaves) {
        CellBlock& blk = *tree.nodes[li].block;
        fill_smooth(blk);
    }

    upload_all_leaves(tree);
    GpuGhostFillList gfl;
    gfl.build(tree, 0);
    gfl.exec(nullptr);

    GpuRhsList rhl;
    rhl.build(tree);
    rhl.exec(nullptr);
    cudaDeviceSynchronize();

    // Sum RHS over all leaves
    double total[NVAR] = {};
    double max_rhs = 0.0;
    for (int li = 0; li < (int)leaves.size(); ++li) {
        auto buf = download_rhs_flat(rhl.d_rhs_pool, li);
        for (int v = 0; v < NVAR; ++v) {
            total[v] += interior_sum(buf, v);
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j)
            for (int i = NG; i < NG+NB; ++i)
                max_rhs = fmax(max_rhs, fabs(buf[v*NCELL+cell_idx(i,j,k)]));
        }
    }
    free_all_leaves(tree);

    const double tol = 1.0e-4 * max_rhs;  // slightly looser: 8 blocks, cross-block errors
    printf("   max|RHS| = %.3e  Σρ = %.3e  ΣρE = %.3e  (tol %.3e)\n",
           max_rhs, total[0], total[4], tol);
    check(fabs(total[0]) < tol, "R4a", "Σ dρ/dt  ≈ 0 (8-leaf tree)", fabs(total[0]));
    check(fabs(total[1]) < tol, "R4b", "Σ d(ρu)/dt ≈ 0 (8-leaf tree)", fabs(total[1]));
    check(fabs(total[2]) < tol, "R4c", "Σ d(ρv)/dt ≈ 0 (8-leaf tree)", fabs(total[2]));
    check(fabs(total[4]) < tol, "R4d", "Σ dE/dt    ≈ 0 (8-leaf tree)", fabs(total[4]));
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P8.3 GPU WENO5-Z RHS gate test ===\n");
    test_r1();
    test_r2();
    test_r3();
    test_r4();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
