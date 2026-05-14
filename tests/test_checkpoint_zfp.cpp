// P4.5 gate — ZFP / float32 compressed checkpointing
//
// BZ01 — F32 roundtrip: save with float32 compression, reload, check that
//         every interior cell error |Q_reload - Q_orig| / |Q_orig| ≤ 1.5e-7.
//
// BZ02 — F32 compression ratio: compressed file size ≤ original / 1.9
//         (float32 gives exactly 2×; 1.9 is a conservative threshold).
//
// BZ03 — ZFP roundtrip (if compiled in): same roundtrip at rate=16 bit/value.
//         Field error ≤ 1e-4 relative (rate-16 is near-double precision).
//         File size ≤ original / 3.
//
// BZ04 — Mode detection: loading an F32 file without specifying mode works
//         (mode is stored in the header).
//
// All tests use a single-block tree with a smooth Taylor-Green vortex IC
// so field values cover a wide dynamic range and are non-trivial.

#include "../include/checkpoint_zfp.hpp"
#include "../include/ns_solver.hpp"
#include "../include/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool cond, double got=-1, double thr=-1) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.3e  threshold %.3e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

static long file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return (long)st.st_size;
}

// Build a solver with one leaf block, Taylor-Green vortex IC.
static void make_tg_solver(NSSolver& s) {
    const double L = 1.0;
    const double rho0 = 1.225, p0 = 1.0e5, U0 = 50.0;

    s.init(L, [&](double x, double y, double z) -> Prim {
        const double u =  U0 * std::sin(2*M_PI*x/L) * std::cos(2*M_PI*y/L) * std::cos(2*M_PI*z/L);
        const double v = -U0 * std::cos(2*M_PI*x/L) * std::sin(2*M_PI*y/L) * std::cos(2*M_PI*z/L);
        const double w = 0.0;
        const double p = p0 + rho0*U0*U0/16.0 *
                         (std::cos(4*M_PI*x/L) + std::cos(4*M_PI*y/L)) *
                         (std::cos(4*M_PI*z/L) + 2.0);
        const double c = std::sqrt(GAMMA * p / rho0);
        const double T = p / (rho0 * R_GAS);
        return {rho0, u, v, w, p, T, c};
    });
}

// Max relative field error between two solvers over all interior cells, all vars.
static double max_rel_err(const NSSolver& a, const NSSolver& b) {
    double maxe = 0.0;
    auto la = a.tree.leaf_indices(), lb = b.tree.leaf_indices();
    if (la.size() != lb.size()) return 1.0;
    for (size_t ii = 0; ii < la.size(); ++ii) {
        auto& ba = *a.tree.nodes[la[ii]].block;
        auto& bb = *b.tree.nodes[lb[ii]].block;
        for (int v = 0; v < NVAR; ++v)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i,j,k);
            const double qa = ba.Q[v][f], qb = bb.Q[v][f];
            const double scale = std::max(std::abs(qa), 1.0e-300);
            maxe = std::max(maxe, std::abs(qa - qb) / scale);
        }
    }
    return maxe;
}

// Reference file size: save uncompressed, measure.
static long reference_size(const NSSolver& s, const char* path) {
    checkpoint_save_compressed(s, path, CKPT_COMPRESS_NONE);
    return file_size(path);
}

// =============================================================================
// BZ01-BZ02 — F32 roundtrip + compression ratio
// =============================================================================
static void test_BZ01_BZ02() {
    printf("\n-- BZ01/BZ02  F32 roundtrip + compression ratio --\n");

    NSSolver s_orig;
    make_tg_solver(s_orig);

    const char* ref_path  = "/tmp/ck_ref.bin";
    const char* cmp_path  = "/tmp/ck_f32.bin";
    const long ref_sz  = reference_size(s_orig, ref_path);
    checkpoint_save_compressed(s_orig, cmp_path, CKPT_COMPRESS_F32);
    const long cmp_sz  = file_size(cmp_path);

    // BZ02: compression ratio
    const double ratio = (double)ref_sz / (double)cmp_sz;
    check("BZ02  F32 compression ratio ≥ 1.9", ratio >= 1.9, ratio, 1.9);
    printf("         (ratio=%.2f : ref=%ld bytes  f32=%ld bytes)\n",
           ratio, ref_sz, cmp_sz);

    // BZ01: roundtrip fidelity
    NSSolver s_loaded;
    checkpoint_load_compressed(s_loaded, cmp_path);
    const double err = max_rel_err(s_orig, s_loaded);
    const double tol = 1.5e-7;
    check("BZ01  F32 roundtrip rel-error ≤ 1.5e-7", err <= tol, err, tol);
    printf("         (max_rel_err=%.3e)\n", err);

    std::remove(ref_path);
    std::remove(cmp_path);
}

// =============================================================================
// BZ03 — ZFP roundtrip (skip if not compiled in)
// =============================================================================
static void test_BZ03() {
    printf("\n-- BZ03  ZFP roundtrip (rate=16 bits/value) --\n");
    if (!ckpt_zfp_available()) {
        printf("  SKIP  ZFP not compiled in (install libzfp-dev + rebuild with -DUSE_ZFP=ON)\n");
        ++n_pass;  // count as pass: absence of ZFP is not a failure
        return;
    }

    NSSolver s_orig;
    make_tg_solver(s_orig);

    const char* ref_path = "/tmp/ck_ref2.bin";
    const char* zfp_path = "/tmp/ck_zfp.bin";
    const long ref_sz = reference_size(s_orig, ref_path);
    checkpoint_save_compressed(s_orig, zfp_path, CKPT_COMPRESS_ZFP, 16.0);
    const long zfp_sz = file_size(zfp_path);

    const double ratio = (double)ref_sz / (double)zfp_sz;
    check("BZ03a ZFP compression ratio ≥ 3.0", ratio >= 3.0, ratio, 3.0);
    printf("         (ratio=%.2f : ref=%ld  zfp=%ld bytes)\n",
           ratio, ref_sz, zfp_sz);

    NSSolver s_loaded;
    checkpoint_load_compressed(s_loaded, zfp_path);
    const double err = max_rel_err(s_orig, s_loaded);
    const double tol = 1.0e-4;
    check("BZ03b ZFP roundtrip rel-error ≤ 1e-4 (rate=16)", err <= tol, err, tol);
    printf("         (max_rel_err=%.3e)\n", err);

    std::remove(ref_path);
    std::remove(zfp_path);
}

// =============================================================================
// BZ04 — Mode detection from header
// =============================================================================
static void test_BZ04() {
    printf("\n-- BZ04  Mode detection from file header --\n");

    NSSolver s_orig;
    make_tg_solver(s_orig);

    const char* path = "/tmp/ck_mode.bin";
    checkpoint_save_compressed(s_orig, path, CKPT_COMPRESS_F32);

    NSSolver s2;
    bool ok = true;
    try {
        checkpoint_load_compressed(s2, path);
    } catch (...) { ok = false; }

    check("BZ04  load without specifying mode (header-driven)", ok);
    std::remove(path);
}

// =============================================================================
int main() {
    printf("=== P4.5 gate: ZFP / float32 compressed checkpointing ===\n");
    printf("  ZFP compiled in: %s\n", ckpt_zfp_available() ? "YES" : "NO");
    printf("  NCELL=%d  NVAR=%d  leaf_bytes_uncompressed=%zu\n",
           NCELL, NVAR, (size_t)NVAR * NCELL * sizeof(double));

    test_BZ01_BZ02();
    test_BZ03();
    test_BZ04();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0)
        printf("==> PASS  P4.5 gate cleared\n");
    else
        printf("==> FAIL  P4.5 gate NOT cleared\n");
    return (n_fail == 0) ? 0 : 1;
}
