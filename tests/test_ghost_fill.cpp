// R9-C: Unit test for fill_cf_ghosts — 5th-order Lagrange polynomial exactness.
//
// A p-th order interpolation scheme reproduces polynomials of degree < p exactly.
// The 5th-order Lagrange scheme in fill_cf_ghosts must therefore reproduce
// polynomials of degree ≤ 4 with error at the level of floating-point rounding
// (≤ 1e-12 for double precision with these coefficient magnitudes).
//
// Test:
//   Fill a coarse CellBlock with f(x,y,z) = A + B·s (linear in normal coord s,
//   constant in transverse).  Call fill_cf_ghosts for all 6 axis/side
//   combinations.  Assert that every fine ghost cell matches f exactly to < 1e-12.
//
// This is a targeted, self-contained test that does NOT require NSSolver or
// any solver infrastructure — only CellBlock and amr_operators (both in libblock).

#include "../include/amr_operators.hpp"
#include "../include/ghost_filler.hpp"
#include "../include/bc_types.hpp"
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstring>

static int n_pass = 0, n_fail = 0;

static void check(const char* name, bool cond, double got = -1.0, double thr = -1.0) {
    if (cond) {
        printf("  PASS  %s\n", name);
        ++n_pass;
    } else {
        if (got >= 0.0)
            printf("  FAIL  %s  (max_err=%.3e  threshold=%.1e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T-CF-P: fill_cf_ghosts reproduces a degree-1 polynomial to machine precision.
//
// For each (axis, side):
//   - coarse block spans [0, NB*h_c] in the normal direction (h_c = 1/NB)
//   - fine  block has h_f = h_c/2, octant 0 (ix=iy=iz=0)
//   - field: Q[v][·] = A + B * s,  s = centroid in normal axis
//   - A=1.5, B=2.7 (arbitrary non-zero constants)
//
// Because the 5-point Lagrange stencil is exact for polynomials of degree ≤ 4,
// the interpolated ghost values must equal the exact polynomial to FP rounding.
// ─────────────────────────────────────────────────────────────────────────────
static void test_cf_polynomial(int axis, int side) {
    const double A = 1.5, B = 2.7;

    const double h_coarse = 1.0 / NB;
    const double h_fine   = h_coarse * 0.5;

    // Coarse block: interior in [0, h_c*NB] along normal axis
    CellBlock coarse(0.0, 0.0, 0.0, h_coarse);

    // Fill coarse interior with A + B * centroid_normal
    for (int k = NG; k < NG + NB; ++k)
    for (int j = NG; j < NG + NB; ++j)
    for (int i = NG; i < NG + NB; ++i) {
        double cx = coarse.ox + (i - NG + 0.5) * h_coarse;
        double cy = coarse.oy + (j - NG + 0.5) * h_coarse;
        double cz = coarse.oz + (k - NG + 0.5) * h_coarse;
        double s  = (axis == 0) ? cx : (axis == 1) ? cy : cz;
        double val = A + B * s;
        for (int v = 0; v < NVAR; ++v)
            coarse.Q[v][cell_idx(i, j, k)] = val;
    }

    // Fine block origin: octant 0, to the right (side=0) or left (side=1) of coarse
    double fox = 0.0, foy = 0.0, foz = 0.0;
    if (axis == 0) {
        fox = (side == 0) ? coarse.ox + NB * h_coarse : coarse.ox - NB * h_fine;
    } else if (axis == 1) {
        foy = (side == 0) ? coarse.oy + NB * h_coarse : coarse.oy - NB * h_fine;
    } else {
        foz = (side == 0) ? coarse.oz + NB * h_coarse : coarse.oz - NB * h_fine;
    }

    CellBlock fine(fox, foy, foz, h_fine);
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            fine.Q[v][idx] = 0.0;

    const int child_octant = 0;   // ix=iy=iz=0
    fill_cf_ghosts(fine, coarse, child_octant, axis, side);

    // Inspect ghost cells in the normal direction at all transverse positions
    double max_err = 0.0;
    for (int gl = 0; gl < NG; ++gl) {
        for (int a = 0; a < NB2; ++a)
        for (int b = 0; b < NB2; ++b) {
            int gf_i, gf_j, gf_k;
            if (axis == 0) {
                gf_i = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
                gf_j = a; gf_k = b;
            } else if (axis == 1) {
                gf_j = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
                gf_i = a; gf_k = b;
            } else {
                gf_k = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
                gf_i = a; gf_j = b;
            }

            // Exact centroid in normal axis
            double cx = fine.ox + (gf_i - NG + 0.5) * h_fine;
            double cy = fine.oy + (gf_j - NG + 0.5) * h_fine;
            double cz = fine.oz + (gf_k - NG + 0.5) * h_fine;
            double s  = (axis == 0) ? cx : (axis == 1) ? cy : cz;
            double exact = A + B * s;

            for (int v = 0; v < NVAR; ++v) {
                double got = fine.Q[v][cell_idx(gf_i, gf_j, gf_k)];
                double err = std::abs(got - exact);
                if (err > max_err) max_err = err;
            }
        }
    }

    const char* tags[3][2] = {
        {"CF-P1 axis=0 side=0", "CF-P2 axis=0 side=1"},
        {"CF-P3 axis=1 side=0", "CF-P4 axis=1 side=1"},
        {"CF-P5 axis=2 side=0", "CF-P6 axis=2 side=1"},
    };
    const double tol = 1e-12;
    check(tags[axis][side], max_err < tol, max_err, tol);
}

// ─────────────────────────────────────────────────────────────────────────────
// T-GF-1: GhostFiller::fill_all with PeriodicBC calls fill_ghosts_periodic.
//
// Construct a 1-block tree (root), fill interior with rho=1.5.
// After fill_all(PeriodicBC{}), x-minus ghost should equal interior (periodic wrap
// with a single block wraps to itself).
// ─────────────────────────────────────────────────────────────────────────────
static void test_ghost_filler_periodic_dispatch() {
    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic(true);
    tree.rebuild_neighbours();

    auto& blk = *tree.nodes[0].block;
    const double rho0 = 1.5;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        blk.rho(i, j, k) = rho0;

    BCVariant bc = PeriodicBC{};
    GhostFiller::fill_all(tree, bc);

    // Single-block periodic: ghost should equal interior (wraps to self)
    double ghost = blk.rho(0, ilo(), ilo());   // x-minus ghost
    check("GF-1 PeriodicBC dispatch: ghost == interior",
          std::abs(ghost - rho0) < 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// T-GF-2: GhostFiller::fill_all with WallBC calls fill_ghosts_wall (rhou negated).
// ─────────────────────────────────────────────────────────────────────────────
static void test_ghost_filler_wall_dispatch() {
    BlockTree tree;
    tree.init(1.0);

    auto& blk = *tree.nodes[0].block;
    // Set a simple conserved state
    const double rho0 = 1.2, rhou0 = 60.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        blk.rho (i, j, k) = rho0;
        blk.rhou(i, j, k) = rhou0;
        blk.rhov(i, j, k) = 0.0;
        blk.rhow(i, j, k) = 0.0;
        blk.E   (i, j, k) = 1e5;
    }

    BCVariant bc = WallBC{};
    GhostFiller::fill_all(tree, bc);

    // x-minus wall ghost: rhou must be negated (no-slip)
    double ghost_rhou = blk.rhou(0, ilo(), ilo());
    check("GF-2 WallBC dispatch: ghost rhou == -interior rhou",
          std::abs(ghost_rhou + rhou0) < 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== R9-C gate: fill_cf_ghosts polynomial exactness + GhostFiller dispatch ===\n\n");

    // Polynomial exactness: all 6 axis/side combinations
    for (int axis = 0; axis < 3; ++axis)
        for (int side = 0; side < 2; ++side)
            test_cf_polynomial(axis, side);

    printf("\n");

    // GhostFiller dispatch smoke tests
    test_ghost_filler_periodic_dispatch();
    test_ghost_filler_wall_dispatch();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0)
        printf("==> PASS  R9-C gate cleared\n");
    return (n_fail > 0) ? 1 : 0;
}
