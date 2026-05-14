// P7.2 gate test — 5th-order Lagrange C/F ghost fill accuracy
//
// Protocol:
//   Fill a coarse block with f(s) = 1 + 0.5·sin(2π·s/L) varying only along
//   the face-normal axis.  Call fill_cf_ghosts, then check each fine ghost
//   cell against the exact function at the fine centroid.
//
// Error analysis (NB=8, h_coarse=1/8, 5-point Lagrange):
//   |E| ≤ f^(5)(ξ)/5! · ∏|x*-xₖ| ≤ (2π)^5·0.5/120 · 9.71·h_coarse^5
//       ≈ 1937/120 · 9.71/32768 ≈ 4.8e-3.
//   Piecewise-constant error (0th order): O(h_coarse) ≈ 0.09.
//   Tolerance 1e-2 sits between these: passes iff error is well above 0th-order.
//
// Gates:
//   W1–W6: axis/side ghost L∞ error < 1e-2 (vs exact normal-axis sin)
//   W7: mass conserved through 20-step regrid with WENO5 ghost fill < 1e-10

#include "mesh/amr_operators.hpp"
#include "solver/ns_solver.hpp"
#include <cmath>
#include <cstdio>
#include <cassert>

static constexpr double PI = 3.14159265358979323846;
static constexpr double L  = 1.0;

// 1D test function along the face-normal axis only.
// fill_cf_ghosts uses nearest-neighbor in transverse directions, so the coarse
// block must be filled with a function constant in transverse to isolate
// 5th-order normal-direction accuracy (matching the protocol comment above).
// Single-frequency sin keeps the 5th-derivative bound (2π)^5·0.5 manageable.
static double f_normal(double s) noexcept {
    return 1.0 + 0.5 * std::sin(2.0 * PI * s / L);
}

// Fill a CellBlock with the 1D normal-axis test function (axis selects coord).
static void fill_block(CellBlock& blk, int axis) {
    for (int k = NG; k < NG + NB; ++k)
    for (int j = NG; j < NG + NB; ++j)
    for (int i = NG; i < NG + NB; ++i) {
        double cx = blk.ox + (i - NG + 0.5) * blk.h;
        double cy = blk.oy + (j - NG + 0.5) * blk.h;
        double cz = blk.oz + (k - NG + 0.5) * blk.h;
        double s  = (axis == 0) ? cx : (axis == 1) ? cy : cz;
        double v  = f_normal(s);
        for (int vv = 0; vv < NVAR; ++vv)
            blk.Q[vv][cell_idx(i, j, k)] = v;
    }
}

static int run_cf_accuracy_test(int axis, int side, double tol) {
    // h_coarse = L / NB so that the interior spans [0,L] in the relevant axis
    const double h_coarse = L / NB;
    const double h_fine   = h_coarse * 0.5;

    // Coarse block origin: interior starts at 0 in relevant axis
    CellBlock coarse(0.0, 0.0, 0.0, h_coarse);
    fill_block(coarse, axis);

    // Fine block occupies octant 0 (ix=iy=iz=0) of a parent whose origin matches
    // the coarse block (so coarse.ox + NB*h_coarse = parent.ox = fine.ox for axis=0,side=0)
    // For side=0, coarse is to the left: coarse right interior edge = fine left edge
    //   → fine.ox = coarse.ox + NB * h_coarse
    // For side=1, coarse is to the right: fine right edge = coarse left edge
    //   → fine.ox = coarse.ox - NB * h_fine (so fine right = coarse left = coarse.ox)

    double fox = 0.0, foy = 0.0, foz = 0.0;
    int child_octant = 0;  // ix=iy=iz=0

    if (axis == 0) {
        if (side == 0) fox = coarse.ox + NB * h_coarse;   // fine is to the right of coarse
        else           fox = coarse.ox - NB * h_fine;      // fine is to the left of coarse
    } else if (axis == 1) {
        if (side == 0) foy = coarse.oy + NB * h_coarse;
        else           foy = coarse.oy - NB * h_fine;
    } else {
        if (side == 0) foz = coarse.oz + NB * h_coarse;
        else           foz = coarse.oz - NB * h_fine;
    }

    CellBlock fine(fox, foy, foz, h_fine);
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            fine.Q[v][idx] = 0.0;

    fill_cf_ghosts(fine, coarse, child_octant, axis, side);

    // Check ghost cells in the face-normal direction
    double max_err = 0.0;
    for (int gl = 0; gl < NG; ++gl) {
        int gf_i, gf_j, gf_k;
        if      (axis == 0) { gf_i = (side==0) ? (NG-1-gl) : (NB2-NG+gl); gf_j = NG; gf_k = NG; }
        else if (axis == 1) { gf_j = (side==0) ? (NG-1-gl) : (NB2-NG+gl); gf_i = NG; gf_k = NG; }
        else                { gf_k = (side==0) ? (NG-1-gl) : (NB2-NG+gl); gf_i = NG; gf_j = NG; }

        // Exact position of the fine ghost cell center (normal axis only)
        double cx = fine.ox + (gf_i - NG + 0.5) * h_fine;
        double cy = fine.oy + (gf_j - NG + 0.5) * h_fine;
        double cz = fine.oz + (gf_k - NG + 0.5) * h_fine;
        double s  = (axis == 0) ? cx : (axis == 1) ? cy : cz;

        double exact = f_normal(s);
        for (int v = 0; v < NVAR; ++v) {
            double got = fine.Q[v][cell_idx(gf_i, gf_j, gf_k)];
            double err = std::abs(got - exact);
            if (err > max_err) max_err = err;
        }
    }

    const char* tag = (axis==0 && side==0) ? "W1" :
                      (axis==0 && side==1) ? "W2" :
                      (axis==1 && side==0) ? "W3" :
                      (axis==1 && side==1) ? "W4" :
                      (axis==2 && side==0) ? "W5" : "W6";
    if (max_err < tol) {
        printf("  PASS  %s axis=%d side=%d ghost L∞ error=%.3e < %.0e\n",
               tag, axis, side, max_err, tol);
        return 0;
    } else {
        printf("  FAIL  %s axis=%d side=%d ghost L∞ error=%.3e >= %.0e\n",
               tag, axis, side, max_err, tol);
        return 1;
    }
}

int main() {
    printf("=== P7.2 gate: WENO5 5th-order C/F ghost fill ===\n");

    // Tolerance 1e-2 sits between 5th-order error (~5e-3) and
    // piecewise-constant error (~9e-2).  Passes iff interpolation is
    // significantly better than 0th-order.
    const double tol = 1e-2;

    int nfail = 0;
    nfail += run_cf_accuracy_test(0, 0, tol);
    nfail += run_cf_accuracy_test(0, 1, tol);
    nfail += run_cf_accuracy_test(1, 0, tol);
    nfail += run_cf_accuracy_test(1, 1, tol);
    nfail += run_cf_accuracy_test(2, 0, tol);
    nfail += run_cf_accuracy_test(2, 1, tol);

    // W7: mass conservation through 20-step regrid (reuse A05 setup)
    {
        auto sod = [](double x, double, double) -> Prim {
            Prim q{};
            const double rhoL=1.0, pL=1.0;
            const double rhoR=0.125, pR=0.1;
            if (x < 0.5) {
                q.rho=rhoL; q.p=pL; q.T=pL/(rhoL*R_GAS); q.c=std::sqrt(GAMMA*pL/rhoL);
            } else {
                q.rho=rhoR; q.p=pR; q.T=pR/(rhoR*R_GAS); q.c=std::sqrt(GAMMA*pR/rhoR);
            }
            return q;
        };

        SolverConfig cfg;
        cfg.time.cfl = 0.4;
        cfg.time.t_end = 1e30;
        cfg.time.max_steps = 20;
        cfg.bc.variant = PeriodicBC{};
        cfg.io.verbose = false;
        cfg.amr.regrid_interval = 5;
        cfg.amr.max_level = 2;

        NSSolver solver;
        solver.cfg = cfg;
        solver.init(1.0, sod);

        StepDiag d0 = solver.compute_diag();
        for (int s = 0; s < 20; ++s) solver.advance();
        StepDiag df = solver.compute_diag();

        double rel = std::abs(df.mass - d0.mass) / (std::abs(d0.mass) + 1e-300);
        if (rel < 1e-10) {
            printf("  PASS  W7 mass conserved through 20-step regrid rel_err=%.2e < 1e-10\n", rel);
        } else {
            printf("  FAIL  W7 mass not conserved rel_err=%.2e\n", rel);
            ++nfail;
        }
    }

    const int ntotal = 7;
    const int npass  = ntotal - nfail;
    printf("\nResults: %d passed, %d failed\n", npass, nfail);
    if (nfail == 0)
        printf("==> PASS  P7.2 gate cleared — 5th-order C/F ghost fill active\n");
    return nfail > 0 ? 1 : 0;
}
