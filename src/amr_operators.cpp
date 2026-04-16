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
// fill_cf_ghosts: fill one ghost face of `fine` from `coarse` interior
// =============================================================================
void fill_cf_ghosts(CellBlock& fine, const CellBlock& coarse,
                    int child_octant, int axis, int side) {
    int ix = oct_ix(child_octant);
    int iy = oct_iy(child_octant);
    int iz = oct_iz(child_octant);

    auto clamp_c = [](int x) { return std::max(NG, std::min(NG + NB - 1, x)); };
    auto local   = [](int a) {
        return (a >= NG && a < NG + NB) ? (a - NG) : (a < NG ? 0 : NB - 1);
    };

    // Fill NG ghost layers in the normal direction.
    // For side=0 (minus face): ghost layers NG-1..0, reading deeper into coarse interior.
    // For side=1 (plus  face): ghost layers NB2-NG..NB2-1, reading deeper into coarse interior.
    for (int gl = 0; gl < NG; ++gl)
    for (int v  = 0; v  < NVAR; ++v)
    for (int a  = 0; a  < NB2; ++a)
    for (int b  = 0; b  < NB2; ++b) {
        int gf_i, gf_j, gf_k;
        int cf_i, cf_j, cf_k;

        if (axis == 0) {
            gf_i = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_j = a; gf_k = b;
            cf_j = NG + iy * (NB / 2) + local(a) / 2;
            cf_k = NG + iz * (NB / 2) + local(b) / 2;
            {
                int base = (side == 0) ? (NG + ix * (NB / 2) - 1)
                                       : (NG + ix * (NB / 2) + NB / 2);
                cf_i = clamp_c(base + (side == 0 ? -gl : +gl));
            }
        } else if (axis == 1) {
            gf_j = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_i = a; gf_k = b;
            cf_i = NG + ix * (NB / 2) + local(a) / 2;
            cf_k = NG + iz * (NB / 2) + local(b) / 2;
            {
                int base = (side == 0) ? (NG + iy * (NB / 2) - 1)
                                       : (NG + iy * (NB / 2) + NB / 2);
                cf_j = clamp_c(base + (side == 0 ? -gl : +gl));
            }
        } else {
            gf_k = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
            gf_i = a; gf_j = b;
            cf_i = NG + ix * (NB / 2) + local(a) / 2;
            cf_j = NG + iy * (NB / 2) + local(b) / 2;
            {
                int base = (side == 0) ? (NG + iz * (NB / 2) - 1)
                                       : (NG + iz * (NB / 2) + NB / 2);
                cf_k = clamp_c(base + (side == 0 ? -gl : +gl));
            }
        }

        fine.Q[v][cell_idx(gf_i, gf_j, gf_k)] =
            coarse.Q[v][cell_idx(cf_i, cf_j, cf_k)];
    }
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
