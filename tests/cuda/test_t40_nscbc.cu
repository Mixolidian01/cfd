// D9 gate — NSCBC outflow/inflow BC
//
// Tests the fill_nscbc() GPU kernel directly by checking ghost cell values
// after one ghost fill pass — no scheme dynamics, no dissipation bias.
//
// N41: Isentropic acoustic IC at interior boundary cells (v_n > 0 → outflow).
//      NSCBC ghost cells (j=10,11 above YPLUS) must equal background state
//      to within 1% of the perturbation amplitude A.
// N42: Same IC, zero-gradient BC (OpenBC).
//      Ghost cells copy interior → ghost pressure equals perturbed interior;
//      deviation from background = A.  Confirms test discriminates.

#include "cuda/gpu_ghost_fill.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/bc_types.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static int nfail = 0;

static void check(const char* tag, const char* msg, bool ok, double val, double tol) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else { printf("  FAIL  %s  %s  (val=%.3e  tol=%.3e)\n", tag, msg, val, tol); ++nfail; }
}

static GpuPool pool;

// Set every cell (interior + ghost) to an isentropic acoustic state.
// Perturbation dp at background (rho=1, p=1, c=sqrt(GAMMA)).
// v_y = dp/(rho0*c0): right-traveling (+y) acoustic wave.
static void set_acoustic_state(CellBlock& blk, double dp) {
    const double c0  = std::sqrt(GAMMA);
    const double p   = 1.0 + dp;
    const double rho = std::pow(p, 1.0 / GAMMA);
    const double v_y = dp / (1.0 * c0);
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        int flat = cell_idx(i, j, k);
        blk.Q[0][flat] = rho;
        blk.Q[1][flat] = 0.0;
        blk.Q[2][flat] = rho * v_y;
        blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = p / (GAMMA - 1.0) + 0.5 * rho * v_y * v_y;
    }
}

// Pressure from conserved variables at flat index.
static double pressure_at(const CellBlock& blk, int flat) {
    double rho  = blk.Q[0][flat];
    double rhou = blk.Q[1][flat];
    double rhov = blk.Q[2][flat];
    double rhow = blk.Q[3][flat];
    double E    = blk.Q[4][flat];
    double ke   = 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
    return (GAMMA - 1.0) * (E - ke);
}

// Build a single-block tree, fill it with the acoustic IC, run one ghost fill,
// download, and return the max |p_ghost - 1.0| over the YPLUS ghost layer (j=NB+NG).
static double run_ghost_check(const FaceBCArray& face_bcs, double dp) {
    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic_axes(true, false, true);

    CellBlock* blk = tree.nodes[0].block.get();
    set_acoustic_state(*blk, dp);

    pool.alloc(blk);
    pool.upload(blk);

    GpuGhostFillList gfl;
    gfl.build(tree, pool, face_bcs);
    gfl.exec();                   // fills ghost cells on null stream

    pool.download(blk);           // cudaMemcpy — waits for kernel

    // Check both YPLUS ghost rows: j = NB+NG = 10 and j = NB+NG+1 = 11.
    double maxe = 0.0;
    for (int jg = NB + NG; jg < NB2; ++jg) {
        for (int k = ilo(); k <= ihi(); ++k)
        for (int i = ilo(); i <= ihi(); ++i) {
            double p = pressure_at(*blk, cell_idx(i, jg, k));
            maxe = std::max(maxe, std::abs(p - 1.0));
        }
    }

    pool.free(blk);
    return maxe;
}

// ── N41: NSCBC — ghost cell pressure snaps to background ──────────────────────
static void test_n41() {
    printf("\n-- N41  NSCBC: YPLUS ghost pressure = background (< 1%% of A) --\n");

    const double A = 0.05;   // 5% perturbation; v_n = A/c > 0 → NSCBC path

    FaceBCArray face_bcs;
    face_bcs[XMINUS] = PeriodicBC{};
    face_bcs[XPLUS]  = PeriodicBC{};
    face_bcs[YMINUS] = NscbcBC{1.0};
    face_bcs[YPLUS]  = NscbcBC{1.0};
    face_bcs[ZMINUS] = PeriodicBC{};
    face_bcs[ZPLUS]  = PeriodicBC{};

    double maxe = run_ghost_check(face_bcs, A);
    printf("   N41: max|p_ghost-1| / A = %.3e  threshold = 0.01\n", maxe / A);
    check("N41", "NSCBC: ghost pressure = background to within 1% of A",
          maxe / A < 0.01, maxe / A, 0.01);
}

// ── N42: Zero-gradient — ghost cell pressure equals perturbed interior ─────────
static void test_n42() {
    printf("\n-- N42  Zero-grad: YPLUS ghost pressure = perturbed interior --\n");

    const double A = 0.05;

    FaceBCArray face_bcs_zg;
    face_bcs_zg[XMINUS] = PeriodicBC{};
    face_bcs_zg[XPLUS]  = PeriodicBC{};
    face_bcs_zg[YMINUS] = OpenBC{};
    face_bcs_zg[YPLUS]  = OpenBC{};
    face_bcs_zg[ZMINUS] = PeriodicBC{};
    face_bcs_zg[ZPLUS]  = PeriodicBC{};

    double maxe = run_ghost_check(face_bcs_zg, A);
    printf("   N42: max|p_ghost-1| / A = %.3e  threshold > 0.9\n", maxe / A);
    // Zero-grad copies interior to ghost → ghost pressure = 1+A → deviation = A.
    check("N42", "zero-grad: ghost pressure = perturbed interior (> 90% of A)",
          maxe / A > 0.9, maxe / A, 0.9);
}

int main() {
    printf("=== D9: NSCBC outflow BC gate (t40) ===\n");
    test_n41();
    test_n42();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail ? 1 : 0;
}
