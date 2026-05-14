// amr_operators.cpp — AMR prolongation / restriction / ghost-fill / sensors
// FIX B2: missing closing braces added to prolong_conservative,
//         restrict_conservative, and fill_cf_ghosts
// FIX B3 (original): restrict_conservative fine-cell base index corrected.
// FIX P0.1: restored *2 factor — fi = NG + (lf_i % half) * 2.
//           Without it both halves of a coarse axis resolve to fine indices
//           0‥3 instead of 0‥3 and 4‥7, breaking volume conservation.
// FIX C3: should_refine uses centred 2nd-order gradient
#include "../include/amr_operators.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

// Octant bit layout: bit0=x-high, bit1=y-high, bit2=z-high
static inline int oct_ix(int o) noexcept { return  o       & 1; }
static inline int oct_iy(int o) noexcept { return (o >> 1) & 1; }
static inline int oct_iz(int o) noexcept { return (o >> 2) & 1; }
static inline int oct_from_xyz(int ix, int iy, int iz) noexcept {
    return ix | (iy << 1) | (iz << 2);
}

// =============================================================================
// prolong_conservative: piecewise-constant parent → one child
// =============================================================================
void prolong_conservative(const CellBlock& coarse, CellBlock& fine, int child_octant) {
    int ix = oct_ix(child_octant);
    int iy = oct_iy(child_octant);
    int iz = oct_iz(child_octant);

    for (int v = 0; v < NVAR; ++v)
    for (int kf = NG; kf < NG + NB; ++kf)
    for (int jf = NG; jf < NG + NB; ++jf)
    for (int inf_ = NG; inf_ < NG + NB; ++inf_) {
        int ic = NG + ix * (NB / 2) + (inf_ - NG) / 2;
        int jc = NG + iy * (NB / 2) + (jf  - NG) / 2;
        int kc = NG + iz * (NB / 2) + (kf  - NG) / 2;
        fine.Q[v][cell_idx(inf_, jf, kf)] = coarse.Q[v][cell_idx(ic, jc, kc)];
    }   // FIX B2: this closing brace was missing
}

// =============================================================================
// restrict_conservative: 8 children → parent (volume-weighted average)
// =============================================================================
void restrict_conservative(CellBlock& coarse, const CellBlock* children[8]) {
    for (int v = 0; v < NVAR; ++v)
    for (int kc = NG; kc < NG + NB; ++kc)
    for (int jc = NG; jc < NG + NB; ++jc)
    for (int ic = NG; ic < NG + NB; ++ic) {
        int lf_i = ic - NG, lf_j = jc - NG, lf_k = kc - NG;
        int half  = NB / 2;
        int child = oct_from_xyz(lf_i / half, lf_j / half, lf_k / half);

        // FIX P0.1: base fine-cell index is NG + (offset within octant) * 2.
        // The *2 maps the coarse-level offset (0..half-1) to the fine-level
        // starting cell (0, 2, 4, … within that octant's 2×2×2 footprint).
        // The di/dj/dk loop below then covers the 2×2×2 fine cells.
        int fi = NG + (lf_i % half) * 2;
        int fj = NG + (lf_j % half) * 2;
        int fk = NG + (lf_k % half) * 2;

        double sum = 0.0;
        for (int dk = 0; dk < 2; ++dk)
        for (int dj = 0; dj < 2; ++dj)
        for (int di = 0; di < 2; ++di)
            sum += children[child]->Q[v][cell_idx(fi + di, fj + dj, fk + dk)];
        coarse.Q[v][cell_idx(ic, jc, kc)] = sum * 0.125;
    }   // FIX B2: this closing brace was missing
}

// =============================================================================
// fill_cf_ghosts — 5th-order Lagrange (P7.2) in the normal direction.
//
// Physics: fine ghost cells at the C/F interface must hold values consistent
// with the coarse solution at the fine ghost centroid to ensure O(h^5) accuracy
// near refinement boundaries (McCorquodale & Colella 2011).
//
// Geometry (h_fine = h_coarse/2, NG ghost layers):
//   side=0: fine ghost gl=0 centroid at ic+1/4, gl=1 at ic-1/4
//     where ic = NG+NB-1 (last interior coarse cell), stencil ic-4..ic.
//   side=1: fine ghost gl=0 centroid at i0-1/4, gl=1 at i0+1/4
//     where i0 = NG (first interior coarse cell), stencil i0..i0+4 (mirror).
//
// Two coefficient sets (nodes at {-4,-3,-2,-1,0}, stencil ends at ic):
//   Lp: target +1/4 h_coarse  [585,-3060,6630,-7956,9945]/6144  (gl=0, side=0)
//   Lm: target -1/4 h_coarse  [-231,1260,-2970,4620,3465]/6144  (gl=1, side=0)
// side=1 uses the same arrays reversed: Lp[4-k] for gl=0, Lm[4-k] for gl=1.
//
// Transverse directions: nearest coarse cell (piecewise-constant).
// Requires NB >= 5 so both 5-cell stencils fit inside the coarse interior.
// =============================================================================
static_assert(NB >= 5, "fill_cf_ghosts: need NB >= 5 for the 5-cell stencil");

void fill_cf_ghosts(CellBlock& fine, const CellBlock& coarse,
                    int child_octant, int axis, int side) {
    const int ix = oct_ix(child_octant);
    const int iy = oct_iy(child_octant);
    const int iz = oct_iz(child_octant);

    // Transverse index: maps NB2-wide fine range to coarse half-cell offset.
    auto local = [](int a) noexcept {
        return (a >= NG && a < NG + NB) ? (a - NG) : (a < NG ? 0 : NB - 1);
    };

    // 5-cell Lagrange, nodes {-4,-3,-2,-1,0} in units of h_coarse:
    //   Lp: target at +1/4 (ghost just outside coarse interior, gl=0 side=0)
    //   Lm: target at -1/4 (ghost one step inside stencil,    gl=1 side=0)
    static constexpr double INV = 1.0 / 6144.0;
    static constexpr double Lp[5] = {
         585*INV, -3060*INV,  6630*INV, -7956*INV,  9945*INV
    };
    static constexpr double Lm[5] = {
        -231*INV,  1260*INV, -2970*INV,  4620*INV,  3465*INV
    };

    for (int gl = 0; gl < NG; ++gl) {
        // side=0: gl=0→Lp (target +1/4), gl=1→Lm (target -1/4)
        // side=1: gl=0→Lp reversed  ,    gl=1→Lm reversed
        const double* Lc = (gl == 0) ? Lp : Lm;

    for (int v  = 0; v  < NVAR; ++v)
    for (int a  = 0; a  < NB2; ++a)
    for (int b  = 0; b  < NB2; ++b) {
        int gf_i, gf_j, gf_k;
        double val = 0.0;

        if (axis == 0) {
            gf_i = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_j = a; gf_k = b;
            const int cj = NG + iy*(NB/2) + local(a)/2;
            const int ck = NG + iz*(NB/2) + local(b)/2;
            if (side == 0) {
                const int i0 = NG + NB - 1;
                for (int k = 0; k < 5; ++k)
                    val += Lc[k] * coarse.Q[v][cell_idx(i0-4+k, cj, ck)];
            } else {
                const int i0 = NG;
                for (int k = 0; k < 5; ++k)
                    val += Lc[4-k] * coarse.Q[v][cell_idx(i0+k, cj, ck)];
            }
        } else if (axis == 1) {
            gf_j = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_i = a; gf_k = b;
            const int ci = NG + ix*(NB/2) + local(a)/2;
            const int ck = NG + iz*(NB/2) + local(b)/2;
            if (side == 0) {
                const int j0 = NG + NB - 1;
                for (int k = 0; k < 5; ++k)
                    val += Lc[k] * coarse.Q[v][cell_idx(ci, j0-4+k, ck)];
            } else {
                const int j0 = NG;
                for (int k = 0; k < 5; ++k)
                    val += Lc[4-k] * coarse.Q[v][cell_idx(ci, j0+k, ck)];
            }
        } else {
            gf_k = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_i = a; gf_j = b;
            const int ci = NG + ix*(NB/2) + local(a)/2;
            const int cj = NG + iy*(NB/2) + local(b)/2;
            if (side == 0) {
                const int k0 = NG + NB - 1;
                for (int k = 0; k < 5; ++k)
                    val += Lc[k] * coarse.Q[v][cell_idx(ci, cj, k0-4+k)];
            } else {
                const int k0 = NG;
                for (int k = 0; k < 5; ++k)
                    val += Lc[4-k] * coarse.Q[v][cell_idx(ci, cj, k0+k)];
            }
        }

        fine.Q[v][cell_idx(gf_i, gf_j, gf_k)] = val;
    }

    // P14.1: phi C/F ghost fill — same 5th-order Lagrange stencil
    for (int a = 0; a < NB2; ++a)
    for (int b = 0; b < NB2; ++b) {
        int gf_i, gf_j, gf_k;
        double val = 0.0;

        if (axis == 0) {
            gf_i = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_j = a; gf_k = b;
            const int cj = NG + iy*(NB/2) + local(a)/2;
            const int ck = NG + iz*(NB/2) + local(b)/2;
            if (side == 0) {
                const int i0 = NG + NB - 1;
                for (int k = 0; k < 5; ++k)
                    val += Lc[k] * coarse.phi_data_[cell_idx(i0-4+k, cj, ck)];
            } else {
                const int i0 = NG;
                for (int k = 0; k < 5; ++k)
                    val += Lc[4-k] * coarse.phi_data_[cell_idx(i0+k, cj, ck)];
            }
        } else if (axis == 1) {
            gf_j = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_i = a; gf_k = b;
            const int ci = NG + ix*(NB/2) + local(a)/2;
            const int ck = NG + iz*(NB/2) + local(b)/2;
            if (side == 0) {
                const int j0 = NG + NB - 1;
                for (int k = 0; k < 5; ++k)
                    val += Lc[k] * coarse.phi_data_[cell_idx(ci, j0-4+k, ck)];
            } else {
                const int j0 = NG;
                for (int k = 0; k < 5; ++k)
                    val += Lc[4-k] * coarse.phi_data_[cell_idx(ci, j0+k, ck)];
            }
        } else {
            gf_k = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_i = a; gf_j = b;
            const int ci = NG + ix*(NB/2) + local(a)/2;
            const int cj = NG + iy*(NB/2) + local(b)/2;
            if (side == 0) {
                const int k0 = NG + NB - 1;
                for (int k = 0; k < 5; ++k)
                    val += Lc[k] * coarse.phi_data_[cell_idx(ci, cj, k0-4+k)];
            } else {
                const int k0 = NG;
                for (int k = 0; k < 5; ++k)
                    val += Lc[4-k] * coarse.phi_data_[cell_idx(ci, cj, k0+k)];
            }
        }

        fine.phi_data_[cell_idx(gf_i, gf_j, gf_k)] = val;
    }
    } // gl
}

// =============================================================================
// should_refine / should_coarsen
// =============================================================================
bool should_refine(const CellBlock& blk, double h, double threshold) {
    // FIX C3: centred 2nd-order gradient; loop stays 1 cell away from boundary
    double h2_inv = 0.5 / h;
    for (int k = NG + 1; k < NG + NB - 1; ++k)
    for (int j = NG + 1; j < NG + NB - 1; ++j)
    for (int i = NG + 1; i < NG + NB - 1; ++i) {
        double rho_c = blk.Q[0][cell_idx(i, j, k)];
        double gx = h2_inv * (blk.Q[0][cell_idx(i+1,j,k)] - blk.Q[0][cell_idx(i-1,j,k)]);
        double gy = h2_inv * (blk.Q[0][cell_idx(i,j+1,k)] - blk.Q[0][cell_idx(i,j-1,k)]);
        double gz = h2_inv * (blk.Q[0][cell_idx(i,j,k+1)] - blk.Q[0][cell_idx(i,j,k-1)]);
        double grad_mag = std::sqrt(gx*gx + gy*gy + gz*gz) * h / (std::fabs(rho_c) + 1e-30);
        if (grad_mag > threshold) return true;
    }
    return false;
}

bool should_coarsen(const CellBlock& blk, double h, double threshold) {
    return !should_refine(blk, h, threshold * 0.5);
}
