#pragma once
// P4.4 — Allaire 5-equation two-phase diffuse interface model
//
// Reference: Allaire, Clerc & Kokh, J. Comput. Phys. 181:577 (2002)
//            Murrone & Guillard, J. Comput. Phys. 202:664 (2005)
//
// 7 transported quantities:
//   Q[0] = α₁ρ₁   partial density phase 1
//   Q[1] = α₂ρ₂   partial density phase 2
//   Q[2] = ρu      mixture x-momentum
//   Q[3] = ρv      mixture y-momentum
//   Q[4] = ρw      mixture z-momentum
//   Q[5] = E       mixture total energy
//   Q[6] = α₁      volume fraction (advected non-conservatively)
//
// EOS — stiffened gas per phase, pressure-equilibrium closure:
//   pₖ = (γₖ-1)ρₖeₖ - γₖp∞ₖ
//   p   = (ρe − Σ αₖγₖp∞ₖ/(γₖ-1)) / (Σ αₖ/(γₖ-1))
//   cₖ² = γₖ(p + p∞ₖ)/ρₖ,   c_mix via Wood formula.
//
// Non-conservative α₁ equation:
//   ∂α₁/∂t + u·∇α₁ = 0
// Numerically split into conservative flux + source (Allaire 2002 §3):
//   RHS[6]_i += (1/h)·sStar_f·(α₁_i − α₁^upwind_f)   per face f.

#include "mesh/cell_block.hpp"
#include <array>
#include <cstring>
#include <cmath>
#include <algorithm>

inline constexpr int NVAR_BN = 7;

// ── EOS parameters (can differ per run) ───────────────────────────────────────
struct BNEosParams {
    double gamma1 = 1.4;      // phase 1 ratio of specific heats (ideal gas default)
    double gamma2 = 4.4;      // phase 2 (water-like stiffened gas, Le Métayer 2004)
    double pinf1  = 0.0;      // phase 1 stiffness [Pa]
    double pinf2  = 6.0e8;    // phase 2 stiffness [Pa]
};

// ── Two-phase primitive state ──────────────────────────────────────────────────
struct Prim2Phase {
    double alpha1;            // volume fraction of phase 1
    double rho1, rho2;        // phasic densities  ρₖ = αₖρₖ / αₖ
    double rho;               // mixture density  ρ = α₁ρ₁ + α₂ρ₂
    double u, v, w;           // mixture velocity
    double p;                 // mixture pressure (equilibrium)
    double c_mix;             // Wood mixture sound speed
};

// ── BNCellBlock ───────────────────────────────────────────────────────────────
// AoSoA storage for 7 variables; otherwise identical API to CellBlock.
struct alignas(64) BNCellBlock {
    static constexpr int NVAR  = NVAR_BN;  // 7
    static constexpr int W     = 8;
    static constexpr int NTILE = NCELL / W;  // 216

    static_assert(NCELL % W == 0, "NCELL must be divisible by W=8");

    // AoSoA buffer: data_[tile * NVAR * W + v * W + lane]
    // tile_ptr(v, t) are 64-byte aligned because NVAR*W = 56 bytes and
    // t*NVAR*W*8 bytes are multiples of 64 for all t,v when NVAR*W = 56 ... actually
    // v*W*8 = v*64, so offsets v=0,1,...,6 are multiples of 64. ✓
    alignas(64) double data_[NTILE * NVAR * W];  // 12 096 doubles = 96 768 bytes

    // ── FieldProxy: Q[v][flat] → AoSoA index ───────────────────────────────
    struct FieldProxy {
        double* data_;
        int     v_;

        double& operator[](int flat) noexcept {
            return data_[(flat >> 3) * (NVAR * W) + v_ * W + (flat & 7)];
        }
        double operator[](int flat) const noexcept {
            return data_[(flat >> 3) * (NVAR * W) + v_ * W + (flat & 7)];
        }

        double*       tile_ptr(int t) noexcept       { return data_ + t*(NVAR*W) + v_*W; }
        const double* tile_ptr(int t) const noexcept { return data_ + t*(NVAR*W) + v_*W; }

        void assign(int /*n*/, double val) noexcept {
            for (int t = 0; t < NTILE; ++t) {
                double* p = tile_ptr(t);
                for (int lane = 0; lane < W; ++lane) p[lane] = val;
            }
        }
    };

    std::array<FieldProxy, NVAR> Q;
    double ox=0, oy=0, oz=0, h=0;

    BNCellBlock() noexcept { init_views(); std::fill(data_, data_+NTILE*NVAR*W, 0.0); }
    explicit BNCellBlock(double ox_, double oy_, double oz_, double h_) noexcept
        : ox(ox_), oy(oy_), oz(oz_), h(h_)
    { init_views(); std::fill(data_, data_+NTILE*NVAR*W, 0.0); }

    BNCellBlock(const BNCellBlock& o) noexcept
        : ox(o.ox), oy(o.oy), oz(o.oz), h(o.h)
    { init_views(); std::memcpy(data_, o.data_, sizeof(data_)); }

    BNCellBlock& operator=(const BNCellBlock& o) noexcept {
        if (this != &o) { ox=o.ox; oy=o.oy; oz=o.oz; h=o.h;
            std::memcpy(data_, o.data_, sizeof(data_)); }
        return *this;
    }
    BNCellBlock(BNCellBlock&&  o) noexcept : BNCellBlock(static_cast<const BNCellBlock&>(o)) {}
    BNCellBlock& operator=(BNCellBlock&& o) noexcept {
        return *this = static_cast<const BNCellBlock&>(o);
    }

    double xc(int i) const noexcept { return ox + (i - NG + 0.5) * h; }
    double yc(int j) const noexcept { return oy + (j - NG + 0.5) * h; }
    double zc(int k) const noexcept { return oz + (k - NG + 0.5) * h; }

private:
    void init_views() noexcept {
        for (int v = 0; v < NVAR; ++v) { Q[v].data_ = data_; Q[v].v_ = v; }
    }
};

// ── EOS: conservative → primitive ─────────────────────────────────────────────
inline Prim2Phase bn_cons_to_prim(
    double a1r1, double a2r2,
    double rhou, double rhov, double rhow,
    double E, double alpha1,
    const BNEosParams& eos) noexcept
{
    constexpr double eps = 1.0e-14;
    Prim2Phase q;
    q.alpha1 = alpha1;
    const double alpha2 = 1.0 - alpha1;

    q.rho  = a1r1 + a2r2;
    q.u    = rhou / q.rho;
    q.v    = rhov / q.rho;
    q.w    = rhow / q.rho;

    const double KE    = 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
    const double rho_e = E - KE;

    q.rho1 = (alpha1 > eps) ? a1r1 / alpha1 : 0.0;
    q.rho2 = (alpha2 > eps) ? a2r2 / alpha2 : 0.0;

    // Pressure from pressure-equilibrium stiffened gas (Allaire 2002 eq. 3.8)
    const double g1 = eos.gamma1, g2 = eos.gamma2;
    const double pi1 = eos.pinf1, pi2 = eos.pinf2;
    const double A = alpha1 / (g1 - 1.0);
    const double B = alpha2 / (g2 - 1.0);
    const double C = alpha1 * g1 * pi1 / (g1 - 1.0);
    const double D = alpha2 * g2 * pi2 / (g2 - 1.0);
    const double denom = A + B;
    q.p = (denom > eps) ? (rho_e - C - D) / denom : 0.0;

    // Phasic sound speeds
    const double c1sq = (q.rho1 > eps) ? g1 * (q.p + pi1) / q.rho1 : 0.0;
    const double c2sq = (q.rho2 > eps) ? g2 * (q.p + pi2) / q.rho2 : 0.0;

    // Wood mixture sound speed: 1/(ρ c²) = Σ αₖ/(ρₖcₖ²)
    double inv = 0.0;
    if (q.rho1 > eps && c1sq > 0.0) inv += alpha1 / (q.rho1 * c1sq);
    if (q.rho2 > eps && c2sq > 0.0) inv += alpha2 / (q.rho2 * c2sq);
    q.c_mix = (q.rho > eps && inv > 0.0) ? std::sqrt(1.0 / (q.rho * inv)) : 0.0;

    return q;
}

// ── BN face-flux result ────────────────────────────────────────────────────────
// F[0..5] = HLLC flux for the 6 conservative variables.
// s_star  = contact-wave speed (for the α₁ non-conservative RHS contribution).
struct BNFaceFlux {
    std::array<double, 6> F;
    double s_star;
};

// ── HLLC-BN flux ──────────────────────────────────────────────────────────────
// Extends 5-variable HLLC to two partial densities α₁ρ₁, α₂ρ₂ using the same
// HLLC wave structure.  The non-conservative α₁ term is handled separately
// in the RHS loop (see bn_operators.cpp).
//
// Sign convention: F is the flux in the +axis direction.
// Davis wave-speed estimates; contact speed sStar from mixture momentum balance.
inline BNFaceFlux hllc_bn_flux(const Prim2Phase& L, const Prim2Phase& R,
                                int axis, const BNEosParams& eos) noexcept
{
    const double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
    const double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

    // Davis wave-speed bounds (mixture sound speed)
    const double sL = std::min(uL - L.c_mix, uR - R.c_mix);
    const double sR = std::max(uL + L.c_mix, uR + R.c_mix);

    // Contact speed from mixture momentum balance (Toro §10.4)
    const double numer = R.p - L.p + L.rho*(sL-uL)*uL - R.rho*(sR-uR)*uR;
    const double denom = L.rho*(sL-uL) - R.rho*(sR-uR);
    const double sStar = (std::abs(denom) > 1.0e-300) ? numer/denom : 0.5*(uL+uR);

    // Helper: reconstruct total energy from Prim2Phase
    auto total_energy = [&](const Prim2Phase& q) -> double {
        const double a2 = 1.0 - q.alpha1;
        const double KE = 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
        const double rho_e = q.alpha1*(q.p + eos.gamma1*eos.pinf1)/(eos.gamma1-1.0)
                           + a2      *(q.p + eos.gamma2*eos.pinf2)/(eos.gamma2-1.0);
        return KE + rho_e;
    };

    // Physical flux F(Q) for the 6 conservative variables
    auto phys_flux = [&](const Prim2Phase& q) -> std::array<double,6> {
        const double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        const double a2 = 1.0 - q.alpha1;
        const double E  = total_energy(q);
        return {q.alpha1 * q.rho1 * un,
                a2       * q.rho2 * un,
                q.rho * q.u * un + (axis==0 ? q.p : 0.0),
                q.rho * q.v * un + (axis==1 ? q.p : 0.0),
                q.rho * q.w * un + (axis==2 ? q.p : 0.0),
                (E + q.p) * un};
    };

    // HLLC star-state flux F*(Q, sK, sStar)
    auto star_flux = [&](const Prim2Phase& q, double sK, double sS)
        -> std::array<double,6>
    {
        const double un  = (axis==0)?q.u:(axis==1)?q.v:q.w;
        const double a2  = 1.0 - q.alpha1;
        const double E   = total_energy(q);
        const double cff = (sK - un) / (sK - sS);  // HLLC density ratio

        // Star-state conserved variables
        const double a1r1_s = q.alpha1 * q.rho1 * cff;
        const double a2r2_s = a2       * q.rho2 * cff;
        const double rho_s  = q.rho * cff;
        const double rhou_s = rho_s * (axis==0 ? sS : q.u);
        const double rhov_s = rho_s * (axis==1 ? sS : q.v);
        const double rhow_s = rho_s * (axis==2 ? sS : q.w);
        const double E_s    = rho_s * (E/q.rho + (sS-un)*(sS + q.p/(q.rho*(sK-un))));

        const auto F = phys_flux(q);
        const double a2val = a2;
        return {F[0] + sK*(a1r1_s  - q.alpha1*q.rho1),
                F[1] + sK*(a2r2_s  - a2val*q.rho2),
                F[2] + sK*(rhou_s  - q.rho*q.u),
                F[3] + sK*(rhov_s  - q.rho*q.v),
                F[4] + sK*(rhow_s  - q.rho*q.w),
                F[5] + sK*(E_s     - E)};
    };

    BNFaceFlux res;
    res.s_star = sStar;

    if      (sL >= 0.0)      res.F = phys_flux(L);
    else if (sR <= 0.0)      res.F = phys_flux(R);
    else if (sStar >= 0.0)   res.F = star_flux(L, sL, sStar);
    else                     res.F = star_flux(R, sR, sStar);

    return res;
}

// ── Ghost fill: periodic BC on a single isolated BNCellBlock ──────────────────
inline void bn_fill_ghosts_periodic(BNCellBlock& blk) noexcept {
    for (int v = 0; v < NVAR_BN; ++v) {
        // X-faces
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j) {
            for (int g = 0; g < NG; ++g) {
                blk.Q[v][cell_idx(g,          j, k)] = blk.Q[v][cell_idx(NB+g,    j, k)];
                blk.Q[v][cell_idx(NB+NG+g,    j, k)] = blk.Q[v][cell_idx(NG+g,    j, k)];
            }
        }
        // Y-faces
        for (int k = 0; k < NB2; ++k)
        for (int i = 0; i < NB2; ++i) {
            for (int g = 0; g < NG; ++g) {
                blk.Q[v][cell_idx(i, g,       k)] = blk.Q[v][cell_idx(i, NB+g,   k)];
                blk.Q[v][cell_idx(i, NB+NG+g, k)] = blk.Q[v][cell_idx(i, NG+g,   k)];
            }
        }
        // Z-faces
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            for (int g = 0; g < NG; ++g) {
                blk.Q[v][cell_idx(i, j, g      )] = blk.Q[v][cell_idx(i, j, NB+g  )];
                blk.Q[v][cell_idx(i, j, NB+NG+g)] = blk.Q[v][cell_idx(i, j, NG+g  )];
            }
        }
    }
}

// ── Forward declaration ───────────────────────────────────────────────────────
// Adds convective contribution (inviscid BN model) to rhs_blk.
// rhs_blk must be zeroed by the caller before the first RHS call.
// Ghost cells of blk must be filled before this call.
void compute_rhs_bn(const BNCellBlock& blk, BNCellBlock& rhs_blk,
                    const BNEosParams& eos) noexcept;

// CFL time step for a single BNCellBlock (interior cells only).
double bn_cfl_dt(const BNCellBlock& blk, double cfl, const BNEosParams& eos) noexcept;
