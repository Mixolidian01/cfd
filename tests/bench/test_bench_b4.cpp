// B.4 benchmark — Water-air shock tube (1D, Allaire 5-equation)
//
// Validates the HLLC-BN flux and non-conservative α₁ advection.
// The key property of the Allaire (2002) model is that it produces NO
// spurious pressure oscillations at a material interface, unlike naive
// mixing of phase pressures (which generates p-spikes proportional to the
// pressure ratio and the interface stiffness).
//
// Reference: Saurel & Abgrall (1999) J. Comput. Phys. 150, 425–467;
//            Allaire, Clerc & Kokh (2002) J. Comput. Phys. 181, 577–616.
//
// Setup (1D, x ∈ [0, 1] m):
//   Phase 1 = air:   γ₁=1.4, p∞₁=0      (ideal gas)
//   Phase 2 = water: γ₂=4.4, p∞₂=6×10⁸  (stiffened-gas)
//   Interface at x = 0.5 m
//   Left  (x < 0.5): mostly water (α₁=1e-6, ρ₂=1000 kg/m³, p=1e7 Pa, u=0)
//   Right (x ≥ 0.5): mostly air   (α₁=1-1e-6, ρ₁=1 kg/m³, p=1e5 Pa, u=0)
//   10 BNCellBlocks in x → 80 interior cells, h = 0.0125 m
//   Forward Euler, CFL = 0.3, N = 100 steps
//
// Gate criteria:
//   B4a  α₁ ∈ [0, 1] for all cells (bound preservation)
//   B4b  No spurious pressure oscillations at the water-air interface
//        (< 2 pressure local extrema in the ±5-cell neighbourhood of the interface)
//   B4c  α₁ρ₁ ≥ 0 and α₂ρ₂ ≥ 0  (no negative partial densities)
//   B4d  p > 0 everywhere

#include "models/bn_model.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool ok, double got = -1, double thr = -1)
{
    if (ok) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.3e  thr %.3e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// ── Ghost fill helpers ────────────────────────────────────────────────────────
// Overwrite the x-direction ghost cells of each block with data from the
// adjacent block (or zero-gradient at domain boundaries).
// bn_fill_ghosts_periodic is called first on each block to handle y,z correctly;
// then this function corrects the x ghosts.
static void fill_x_ghosts_1d(std::vector<BNCellBlock>& blks)
{
    const int N = (int)blks.size();
    for (int b = 0; b < N; ++b) {
        for (int g = 0; g < NG; ++g) {
            for (int v = 0; v < NVAR_BN; ++v)
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j) {
                // Left ghost of block b: comes from right interior of block b-1
                // ghost i=g ← interior i=NB+g of block b-1
                if (b > 0)
                    blks[b].Q[v][cell_idx(g, j, k)] =
                        blks[b-1].Q[v][cell_idx(NB+g, j, k)];
                else  // zero-gradient at left boundary
                    blks[b].Q[v][cell_idx(g, j, k)] =
                        blks[b].Q[v][cell_idx(NG, j, k)];

                // Right ghost of block b: comes from left interior of block b+1
                // ghost i=NB+NG+g ← interior i=NG+g of block b+1
                if (b < N-1)
                    blks[b].Q[v][cell_idx(NB+NG+g, j, k)] =
                        blks[b+1].Q[v][cell_idx(NG+g, j, k)];
                else  // zero-gradient at right boundary
                    blks[b].Q[v][cell_idx(NB+NG+g, j, k)] =
                        blks[b].Q[v][cell_idx(NB+NG-1, j, k)];
            }
        }
    }
}

static void fill_ghosts_1d(std::vector<BNCellBlock>& blks)
{
    // 1. Fill y,z ghosts (periodic within each block = zero-gradient since 1D uniform)
    for (auto& blk : blks) bn_fill_ghosts_periodic(blk);
    // 2. Overwrite x ghosts with correct inter-block data
    fill_x_ghosts_1d(blks);
}

// ── Initial condition ─────────────────────────────────────────────────────────
static void set_ic(std::vector<BNCellBlock>& blks, const BNEosParams& eos,
                   double x_iface,
                   double a1L, double rho1L, double rho2L, double pL,
                   double a1R, double rho1R, double rho2R, double pR)
{
    for (auto& blk : blks)
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const double xc = blk.ox + (i - NG + 0.5) * blk.h;
        const bool  left = (xc < x_iface);

        const double a1   = left ? a1L   : a1R;
        const double a2   = 1.0 - a1;
        const double rho1 = left ? rho1L : rho1R;
        const double rho2 = left ? rho2L : rho2R;
        const double p    = left ? pL    : pR;

        const int f = cell_idx(i,j,k);
        blk.Q[0][f] = a1 * rho1;   // α₁ρ₁
        blk.Q[1][f] = a2 * rho2;   // α₂ρ₂
        blk.Q[2][f] = 0.0;         // ρu = 0
        blk.Q[3][f] = 0.0;
        blk.Q[4][f] = 0.0;
        const double rho_e = a1*(p + eos.gamma1*eos.pinf1)/(eos.gamma1-1.0)
                           + a2*(p + eos.gamma2*eos.pinf2)/(eos.gamma2-1.0);
        blk.Q[5][f] = rho_e;       // KE=0 since u=0
        blk.Q[6][f] = a1;          // α₁
    }
}

// ─────────────────────────────────────────────────────────────────────────────
static void b4_water_air_shock()
{
    printf("\n-- B4  Water-air shock tube (Saurel & Abgrall 1999, Allaire 2002) --\n");

    // EOS: phase 1 = air, phase 2 = water (Le Métayer 2004 params)
    BNEosParams eos;   // γ₁=1.4, p∞₁=0, γ₂=4.4, p∞₂=6e8

    // Shock tube IC
    const double x_iface = 0.5;
    const double pL      = 1.0e7;    // 100 bar (water side)
    const double pR      = 1.0e5;    // 1 bar   (air side)
    const double rho2L   = 1000.0;   // water density (left)
    const double rho1R   = 1.0;      // air density   (right)
    const double a1L     = 1.0e-6;   // α₁ left  (mostly water)
    const double a1R     = 1.0 - 1.0e-6;  // α₁ right (mostly air)

    // 1D chain of NX blocks in x
    const int    NX     = 10;
    const double L_dom  = 1.0;           // domain [0, 1] m
    const double h_cell = L_dom / (NX * NB);  // 0.0125 m

    std::vector<BNCellBlock> blks(NX);
    for (int b = 0; b < NX; ++b) {
        blks[b].h  = h_cell;
        blks[b].ox = b * NB * h_cell;
        blks[b].oy = 0.0;
        blks[b].oz = 0.0;
    }
    set_ic(blks, eos, x_iface,
           a1L, 1.0, rho2L, pL,
           a1R, rho1R, 1000.0, pR);

    // Forward Euler time loop (CFL = 0.3)
    const int N_STEPS = 100;
    const double CFL  = 0.3;
    double t = 0.0;

    for (int step = 0; step < N_STEPS; ++step) {
        fill_ghosts_1d(blks);
        // Global CFL dt
        double dt = 1.0e30;
        for (auto& blk : blks) dt = std::min(dt, bn_cfl_dt(blk, CFL, eos));

        // RHS
        std::vector<BNCellBlock> rhs(NX);
        for (int b = 0; b < NX; ++b) {
            rhs[b].h = h_cell;
            compute_rhs_bn(blks[b], rhs[b], eos);
        }
        // Update interior cells
        for (int b = 0; b < NX; ++b)
        for (int v = 0; v < NVAR_BN; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i,j,k);
            blks[b].Q[v][f] += dt * rhs[b].Q[v][f];
        }
        t += dt;
    }

    // ── Extract 1D centre-line profile ───────────────────────────────────────
    const int jmid = NG + NB/2;
    const int kmid = NG + NB/2;
    std::vector<double> x_p, a1_p, p_p, a1r1_p, a2r2_p;
    for (int b = 0; b < NX; ++b) {
        const auto& blk = blks[b];
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i, jmid, kmid);
            Prim2Phase q = bn_cons_to_prim(
                blk.Q[0][f], blk.Q[1][f],
                blk.Q[2][f], blk.Q[3][f], blk.Q[4][f],
                blk.Q[5][f], blk.Q[6][f], eos);
            x_p.push_back(blk.xc(i));
            a1_p.push_back(q.alpha1);
            p_p.push_back(q.p);
            a1r1_p.push_back(blk.Q[0][f]);
            a2r2_p.push_back(blk.Q[1][f]);
        }
    }

    // ── Find interface (maximum |Δα₁|) ───────────────────────────────────────
    int iface_idx = 1;
    double max_da = 0.0;
    for (int i = 1; i < (int)a1_p.size(); ++i) {
        double da = std::abs(a1_p[i] - a1_p[i-1]);
        if (da > max_da) { max_da = da; iface_idx = i; }
    }

    // ── B4a: α₁ ∈ [0,1] ──────────────────────────────────────────────────────
    double a1_min = *std::min_element(a1_p.begin(), a1_p.end());
    double a1_max = *std::max_element(a1_p.begin(), a1_p.end());

    // ── B4b: No spurious pressure oscillations near the interface ─────────────
    // Count local extrema (oscillations) in ±5-cell window around interface.
    const int WIN = 5;
    const int lo = std::max(1,     iface_idx - WIN);
    const int hi = std::min((int)p_p.size()-2, iface_idx + WIN);
    int n_osc = 0;
    for (int i = lo; i <= hi; ++i) {
        bool lmax = (p_p[i] > p_p[i-1]) && (p_p[i] > p_p[i+1]);
        bool lmin = (p_p[i] < p_p[i-1]) && (p_p[i] < p_p[i+1]);
        if (lmax || lmin) ++n_osc;
    }

    // ── B4c: No negative partial densities ────────────────────────────────────
    double a1r1_min = *std::min_element(a1r1_p.begin(), a1r1_p.end());
    double a2r2_min = *std::min_element(a2r2_p.begin(), a2r2_p.end());

    // ── B4d: Pressure positive everywhere ────────────────────────────────────
    double p_min = *std::min_element(p_p.begin(), p_p.end());

    printf("   t_final = %.3e s  h = %.4f m  NX=%d\n", t, h_cell, NX);
    printf("   α₁ ∈ [%.4e, %.4e]  interface cell = %d (|Δα₁|=%.3f)\n",
           a1_min, a1_max, iface_idx, max_da);
    printf("   pressure oscillations near interface: %d\n", n_osc);
    printf("   α₁ρ₁_min = %.3e  α₂ρ₂_min = %.3e  p_min = %.3e\n",
           a1r1_min, a2r2_min, p_min);

    const double tol_alpha = 1.0e-12;
    check("B4a  α₁ ∈ [0,1]",
          a1_min >= -tol_alpha && a1_max <= 1.0+tol_alpha,
          std::max(-a1_min, a1_max-1.0), tol_alpha);
    check("B4b  no spurious pressure oscillations at interface (< 2 extrema)",
          n_osc < 2, (double)n_osc, 2.0);
    check("B4c  α₁ρ₁ ≥ 0  and  α₂ρ₂ ≥ 0",
          a1r1_min >= 0.0 && a2r2_min >= 0.0,
          std::min(a1r1_min, a2r2_min), 0.0);
    check("B4d  p > 0 everywhere",
          p_min > 0.0, p_min, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.4 gate: Water-air shock tube (Allaire 5-equation) ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d  NVAR_BN=%d\n",
           NB, NG, NB2, NCELL, NVAR_BN);
    b4_water_air_shock();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.4 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
