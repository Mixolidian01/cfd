// D6 gate — GPU P1 radiation transport (diffusion limit).
//
// M51: Steady-state 1D exponential decay — CPU reference.
//      Solve −D∇²G + κG = 0 with Dirichlet BCs; check G = exp(−x/λ) within 2%.
// M52: Same solve on GPU via GpuP1List.  Checks identical result to M51.
// M53: Energy coupling — single step from warm matter (T=1) + G=0;
//      verify Q[4] changes by κ(aT⁴ − G)·dt per cell to machine precision.

#include "cuda/gpu_p1.cuh"
#include "physics/p1_radiation.hpp"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
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

static void free_all(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
}

// ── Exact analytical solution ─────────────────────────────────────────────────
// −D∇²G + κG = 0  with G(0)=G_s, G(L)=G_s·exp(−√(κ/D)·L).
// Analytical: G(x) = G_s · exp(−x/λ)  where λ = √(D/κ).
// In 3D but uniform in y,z, the 3D Laplacian reduces to ∂²/∂x².
static double G_exact(double x, double G_s, double kappa, double D) {
    double lambda = std::sqrt(D / kappa);
    return G_s * std::exp(-x / lambda);
}

// ── M51: CPU reference (1D, Jacobi iteration) ─────────────────────────────────
// We test the PHYSICS, not the solver. Use a fine-enough grid that the
// analytical BC values are exact. For the gate, NB=8 cells in [0,1]:
//   h = 1/8 = 0.125,  D = 1/3,  κ = 1  →  λ = 1/√3 ≈ 0.577
//   Discretisation error: O(h²·1/(3λ⁴)·λ²) ≈ 0.4% < 2%.
static void test_m51() {
    printf("\n-- M51  CPU 1D steady-state radiation (analytical comparison) --\n");

    const double kappa = 1.0, D = 1.0 / 3.0, G_s = 1.0;
    const int    N     = NB;          // 8 interior cells
    const double L     = 1.0;
    const double h     = L / N;
    const double h2    = h * h;

    // Cell-centre positions.
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = (i + 0.5) * h;

    // Dirichlet BCs: face at x=0 has G=G_s; face at x=L has G=G_exact(L).
    double G_left  = G_s;
    double G_right = G_exact(L, G_s, kappa, D);

    // Tridiagonal solve for −D∇²G + κG = 0 with ghost-cell Dirichlet BCs.
    // Ghost at i=-1: G_{-1} = 2*G_left − G[0]  →  stencil at i=0 becomes:
    //   (3D/h²+κ)G[0] − (D/h²)G[1] = 2D*G_left/h²
    // Interior: (2D/h²+κ)G[i] − (D/h²)(G[i-1]+G[i+1]) = 0
    // Right BC: (3D/h²+κ)G[N-1] − (D/h²)G[N-2] = 2D*G_right/h²
    //
    // Thomas algorithm: a_i*x[i-1] + b_i*x[i] + c_i*x[i+1] = d_i
    //   a_i = c_i = −oa,  b_i = b[i],  d_i = rhs[i]
    //   oa = D/h² > 0
    const double oa    = D / h2;
    const double b_int = 2.0 * oa + kappa;
    const double b_bnd = 3.0 * oa + kappa;
    std::vector<double> G(N, 0.0);
    std::vector<double> rhs(N, 0.0);
    rhs[0]   = 2.0 * oa * G_left;
    rhs[N-1] = 2.0 * oa * G_right;

    // Forward sweep: eliminate lower diagonal.
    // w[i] = effective diagonal,  y[i] = effective RHS.
    std::vector<double> w(N), y(N);
    w[0] = b_bnd; y[0] = rhs[0];
    for (int i = 1; i < N; ++i) {
        double bi  = (i == N-1) ? b_bnd : b_int;
        double m   = -oa / w[i-1];      // multiplier (lower elim; a_i=-oa)
        w[i] = bi - m * (-oa);           // c_{i-1} = -oa
        y[i] = rhs[i] - m * y[i-1];
    }
    // Back substitution.
    G[N-1] = y[N-1] / w[N-1];
    for (int i = N-2; i >= 0; --i)
        G[i] = (y[i] - (-oa) * G[i+1]) / w[i];  // c_i=-oa

    // Compare to analytical.
    double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double G_an  = G_exact(x[i], G_s, kappa, D);
        double rel   = std::fabs(G[i] - G_an) / G_an;
        max_rel = std::max(max_rel, rel);
    }
    printf("   M51: D=%.4f κ=%.1f λ=%.4f h=%.4f\n",
           D, kappa, std::sqrt(D/kappa), h);
    printf("   M51: G[0]=%.6f  G_exact=%.6f  max_rel_err=%.3e\n",
           G[0], G_exact(x[0], G_s, kappa, D), max_rel);
    check(max_rel < 0.02, "M51",
          "1D steady-state G matches exp(−x/λ) within 2%", max_rel);
}

// ── M52: GPU CG solve ──────────────────────────────────────────────────────────
static void test_m52() {
    printf("\n-- M52  GPU P1 CG solve (Marshak penetration depth within 2%%) --\n");

    const double kappa = 1.0, G_s = 1.0;
    const double L     = 1.0;
    const double h     = L / NB;   // 0.125

    RadiationParams params;
    params.kappa   = kappa;
    params.a_rad   = 1.0;
    params.c_light = 1.0;
    params.R_spec  = 1.0;
    const double D = p1_diffusion(params);  // 1/3

    double G_left  = G_s;
    double G_right = G_exact(L, G_s, kappa, D);

    // Uniform CJ-state fluid (no chemistry): T=0, so RHS = κaT⁴ = 0.
    BlockTree tree; tree.init(L); tree.set_periodic(false);
    CellBlock& blk = *tree.nodes[0].block;
    for (int flat = 0; flat < NCELL; ++flat) {
        blk.Q[0][flat] = 1.0;
        blk.Q[1][flat] = 0.0; blk.Q[2][flat] = 0.0; blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = 0.0;  // T = 0 → no emission source
    }
    pool.alloc(&blk); pool.upload(&blk);

    GpuP1List p1;
    p1.build(tree, pool, /*bc_type=*/1);
    p1.set_bc(G_left, G_right);

    // Initial guess G=0 (zero from cudaMemset in build).
    // exec_cg solves to tol=1e-10 (tight).
    auto cg_res = p1.exec_cg(params, h, 200, 1e-10, nullptr);
    cudaDeviceSynchronize();

    printf("   M52: CG converged in %d iters, rel_res = %.2e\n",
           cg_res.max_iters, cg_res.max_rel_res);

    // Download and compare.
    std::vector<double> hG;
    p1.download_g(hG);

    double max_rel = 0.0;
    for (int i = 0; i < NB; ++i) {
        // G array is NB³ in compact order: idx = i + j*NB + k*NB²
        // For uniform j,k, all y,z values are the same (problem is 1D).
        // Take j=0, k=0.
        int lin = i;  // i + 0*NB + 0*NB²
        double x_c    = (i + 0.5) * h;
        double G_an   = G_exact(x_c, G_s, kappa, D);
        double G_num  = hG[lin];
        double rel    = std::fabs(G_num - G_an) / G_an;
        max_rel = std::max(max_rel, rel);
    }
    printf("   M52: Penetration depth λ = %.4f;  max_rel_err = %.3e\n",
           std::sqrt(D/kappa), max_rel);
    // CG tol=1e-10 << discretisation error ≈ 0.4%; gate at 2%.
    check(max_rel < 0.02, "M52",
          "GPU CG G matches exp(−x/λ) within 2% (Marshak penetration depth)", max_rel);
    check(cg_res.max_rel_res < 1e-8, "M52b",
          "CG residual ‖r‖/‖b‖ < 1e-8", cg_res.max_rel_res);

    free_all(tree);
}

// ── M53: Energy coupling ───────────────────────────────────────────────────────
static void test_m53() {
    printf("\n-- M53  Energy coupling: ΔE = κ(aT⁴ − G)·dt --\n");

    const double rho1  = 1.0;
    const double p1_   = 1.0;   // pressure
    const double dt    = 0.01;

    RadiationParams params;
    params.kappa   = 1.0;
    params.a_rad   = 1.0;
    params.c_light = 1.0;
    params.R_spec  = 1.0;

    // Set T=1 in all cells: e_int = T*R/(γ-1) = 1/(1*0.4) = 2.5 → E = ρ*e = 2.5
    double e_int = p1_ / ((GAMMA - 1.0) * rho1);  // 2.5
    double E0    = rho1 * e_int;
    double T0    = (GAMMA - 1.0) * e_int / params.R_spec;  // = p/ρ = 1.0
    double G_val = 0.3;  // G < G_eq → radiation cold, matter emits → matter cools

    double G_eq  = p1_emission(T0, params);   // c*a*T^4 = 1.0
    // ΔE = κ(G − G_eq)·dt  (negative: matter loses energy to radiation)
    double dE_expected = params.kappa * (G_val - G_eq) * dt;  // 1*(0.3-1.0)*0.01 = -0.007

    printf("   M53: T=%.1f G=%.3f G_eq=%.3f  expected ΔE=%.8f\n",
           T0, G_val, G_eq, dE_expected);

    BlockTree tree; tree.init(1.0); tree.set_periodic(true);
    CellBlock& blk = *tree.nodes[0].block;
    for (int flat = 0; flat < NCELL; ++flat) {
        blk.Q[0][flat] = rho1;
        blk.Q[1][flat] = 0.0; blk.Q[2][flat] = 0.0; blk.Q[3][flat] = 0.0;
        blk.Q[4][flat] = E0;
    }
    pool.alloc(&blk); pool.upload(&blk);

    GpuP1List p1;
    p1.build(tree, pool, 0);

    // Upload constant G=G_val to all interior cells.
    std::vector<double> hG_init(NB*NB*NB, G_val);
    p1.upload_g(hG_init);
    p1.exec_ghost_g(nullptr);

    // Apply energy coupling.
    p1.exec_couple(dt, params, nullptr);
    cudaDeviceSynchronize();

    // Download Q and check ΔE.
    pool.download(&blk);
    double E_final_sum = 0.0, E_init_sum = 0.0;
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int f = cell_idx(i, j, k);
        E_final_sum += blk.Q[4][f];
        E_init_sum  += E0;
    }
    double dE_measured = (E_final_sum - E_init_sum) / (NB * NB * NB);
    double rel_err = std::fabs(dE_measured - dE_expected) /
                     (std::fabs(dE_expected) + 1e-30);
    printf("   M53: dE_measured = %.8f  dE_expected = %.8f  rel_err = %.3e\n",
           dE_measured, dE_expected, rel_err);
    check(rel_err < 1e-10, "M53",
          "Energy coupling ΔE = κ(aT⁴−G)·dt to machine precision", rel_err);

    free_all(tree);
}

// =============================================================================
int main() {
    printf("=== D6 P1 radiation GPU gate test ===\n");
    test_m51();
    test_m52();
    test_m53();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
