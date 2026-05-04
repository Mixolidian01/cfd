// P8.4 gate test — GPU AMR prolongation and restriction kernels
//
// Protocol:
//   A1: prolong one octant — GPU interior matches CPU piecewise-constant (bitwise)
//   A2: restrict 8 children — GPU coarse matches CPU volume-weighted average
//   A3: prolong all 8 octants then restrict → identity (mass conservation)
//   A4: all 8 octants prolong independently correct

#include "../../include/cuda/gpu_amr.cuh"
#include "../../include/gpu_pool.hpp"
#include "../../include/cell_block.hpp"
#include "../../include/amr_operators.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>
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

static void fill_block(CellBlock& blk, double seed) {
    for (int v = 0; v < NVAR; ++v)
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        double x = i * 1.7 + j * 2.3 + k * 0.9 + v * 0.5 + seed;
        blk.Q[v][cell_idx(i, j, k)] = sin(x) + 3.0;   // always positive
    }
}

static void zero_interior(CellBlock& blk) {
    for (int v = 0; v < NVAR; ++v)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        blk.Q[v][cell_idx(i, j, k)] = 0.0;
}

// Maximum relative error over interior cells (all variables)
static double interior_rel_err(const CellBlock& a, const CellBlock& b) {
    double err = 0.0;
    for (int v = 0; v < NVAR; ++v)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        double ref = b.Q[v][cell_idx(i, j, k)];
        double diff = std::fabs(a.Q[v][cell_idx(i, j, k)] - ref);
        if (ref != 0.0) err = std::fmax(err, diff / std::fabs(ref));
        else            err = std::fmax(err, diff);
    }
    return err;
}

// =============================================================================
// A1: single octant prolong — GPU vs CPU
// =============================================================================
static void test_a1() {
    printf("\n-- A1  Single octant prolong: GPU vs CPU --\n");

    CellBlock coarse;  coarse.h = 1.0;
    fill_block(coarse, 1.0);

    // CPU reference
    CellBlock cpu_fine;  cpu_fine.h = 0.5;
    prolong_conservative(coarse, cpu_fine, /*oct=*/3);

    // GPU
    pool.alloc(&coarse);
    pool.upload(&coarse);

    CellBlock gpu_fine;  gpu_fine.h = 0.5;
    zero_interior(gpu_fine);
    pool.alloc(&gpu_fine);
    pool.upload(&gpu_fine);   // upload zeros (ghost layers)

    GpuProlongMeta meta;
    meta.d_coarse_Q = pool.d_Q(&coarse);
    meta.d_fine_Q   = pool.d_Q(&gpu_fine);
    meta.oct        = 3;
    meta._pad       = 0;

    GpuAmrList amr;
    amr.build_prolong({meta});
    amr.exec_prolong();

    pool.download(&gpu_fine);
    pool.free(&coarse);
    pool.free(&gpu_fine);

    const double err = interior_rel_err(gpu_fine, cpu_fine);
    printf("   max interior rel_err = %.3e  (expect 0 — bitwise identical)\n", err);
    check(err == 0.0, "A1", "GPU prolong interior == CPU piecewise-constant (oct=3)", err);
}

// =============================================================================
// A2: restrict 8 children — GPU vs CPU
// =============================================================================
static void test_a2() {
    printf("\n-- A2  Restrict 8 children: GPU vs CPU --\n");

    CellBlock children_cpu[8];
    const CellBlock* child_ptrs[8];

    for (int o = 0; o < 8; ++o) {
        children_cpu[o].h = 0.5;
        fill_block(children_cpu[o], 1.0 + 0.13 * o);
        child_ptrs[o] = &children_cpu[o];
    }

    // CPU reference
    CellBlock cpu_coarse;  cpu_coarse.h = 1.0;
    restrict_conservative(cpu_coarse, child_ptrs);

    // GPU: upload all children
    for (int o = 0; o < 8; ++o) {
        pool.alloc(&children_cpu[o]);
        pool.upload(&children_cpu[o]);
    }

    CellBlock gpu_coarse;  gpu_coarse.h = 1.0;
    zero_interior(gpu_coarse);
    pool.alloc(&gpu_coarse);
    pool.upload(&gpu_coarse);

    GpuRestrictMeta meta;
    meta.d_coarse_Q = pool.d_Q(&gpu_coarse);
    for (int o = 0; o < 8; ++o) meta.d_children_Q[o] = pool.d_Q(&children_cpu[o]);

    GpuAmrList amr;
    amr.build_restrict({meta});
    amr.exec_restrict();

    pool.download(&gpu_coarse);
    for (int o = 0; o < 8; ++o) pool.free(&children_cpu[o]);
    pool.free(&gpu_coarse);

    const double err = interior_rel_err(gpu_coarse, cpu_coarse);
    printf("   max interior rel_err = %.3e  (expect 0 — exact averaging)\n", err);
    check(err == 0.0, "A2", "GPU restrict interior == CPU volume-weighted average", err);
}

// =============================================================================
// A3: prolong all 8 octants → restrict → identity
// =============================================================================
static void test_a3() {
    printf("\n-- A3  Prolong×8 then restrict = identity (mass conservation) --\n");

    CellBlock coarse;  coarse.h = 1.0;
    fill_block(coarse, 2.5);
    pool.alloc(&coarse);
    pool.upload(&coarse);

    CellBlock children[8];
    std::vector<GpuProlongMeta> p_ops(8);
    for (int o = 0; o < 8; ++o) {
        children[o].h = 0.5;
        zero_interior(children[o]);
        pool.alloc(&children[o]);
        pool.upload(&children[o]);
        p_ops[o].d_coarse_Q = pool.d_Q(&coarse);
        p_ops[o].d_fine_Q   = pool.d_Q(&children[o]);
        p_ops[o].oct        = o;
        p_ops[o]._pad       = 0;
    }

    GpuAmrList amr;
    amr.build_prolong(p_ops);
    amr.exec_prolong();

    CellBlock result;  result.h = 1.0;
    zero_interior(result);
    pool.alloc(&result);
    pool.upload(&result);

    GpuRestrictMeta r_meta;
    r_meta.d_coarse_Q = pool.d_Q(&result);
    for (int o = 0; o < 8; ++o) r_meta.d_children_Q[o] = pool.d_Q(&children[o]);

    amr.build_restrict({r_meta});
    amr.exec_restrict();

    pool.download(&result);
    pool.free(&coarse);
    for (int o = 0; o < 8; ++o) pool.free(&children[o]);
    pool.free(&result);

    const double err = interior_rel_err(result, coarse);
    printf("   max interior rel_err = %.3e  (tol 1e-14)\n", err);
    check(err < 1.0e-14, "A3", "prolong×8 → restrict = identity (mass conserved)", err);
}

// =============================================================================
// A4: all 8 octants prolong independently match CPU
// =============================================================================
static void test_a4() {
    printf("\n-- A4  All 8 octants prolong match CPU --\n");
    const double cfl = 0.5;
    (void)cfl;

    CellBlock coarse;  coarse.h = 1.0;
    fill_block(coarse, 3.7);
    pool.alloc(&coarse);
    pool.upload(&coarse);

    int worst_oct = -1;
    double worst_err = 0.0;
    bool all_pass = true;

    std::vector<GpuProlongMeta> p_ops(8);
    CellBlock children[8];
    for (int o = 0; o < 8; ++o) {
        children[o].h = 0.5;
        zero_interior(children[o]);
        pool.alloc(&children[o]);
        pool.upload(&children[o]);
        p_ops[o].d_coarse_Q = pool.d_Q(&coarse);
        p_ops[o].d_fine_Q   = pool.d_Q(&children[o]);
        p_ops[o].oct        = o;
        p_ops[o]._pad       = 0;
    }

    // Launch all 8 prolongs in one batch
    GpuAmrList amr;
    amr.build_prolong(p_ops);
    amr.exec_prolong();

    for (int o = 0; o < 8; ++o) pool.download(&children[o]);

    // CPU references
    for (int o = 0; o < 8; ++o) {
        CellBlock cpu_fine;  cpu_fine.h = 0.5;
        prolong_conservative(coarse, cpu_fine, o);
        const double err = interior_rel_err(children[o], cpu_fine);
        if (err > worst_err) { worst_err = err; worst_oct = o; }
        if (err != 0.0) all_pass = false;
    }

    pool.free(&coarse);
    for (int o = 0; o < 8; ++o) pool.free(&children[o]);

    printf("   worst err = %.3e at oct=%d  (expect 0 — bitwise identical)\n",
           worst_err, worst_oct);
    check(all_pass, "A4", "all 8 octants GPU prolong == CPU (bitwise)", worst_err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P8.4 GPU AMR prolongation / restriction gate test ===\n");
    test_a1();
    test_a2();
    test_a3();
    test_a4();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
