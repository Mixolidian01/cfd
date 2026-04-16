#pragma once
// DESIGN.md reference: Layer 1 — Cell-Centered Block
//
// A CellBlock stores one NB x NB x NB patch of conserved variables.
// Each field has NG ghost layers on every side → (NB+2·NG)^3 storage.
//
// Memory layout: SoA (Structure of Arrays) — one contiguous array per field.
// Index convention:
//   flat(i,j,k) = k*NB2*NB2 + j*NB2 + i
//   where i,j,k ∈ [0, NB2-1]   (0..NG-1 and NB2-NG..NB2-1 are ghost cells)
//   interior:   i,j,k ∈ [NG, NB+NG-1]
//
// Conserved variables (following DESIGN.md):
//   Q[0] = rho        density
//   Q[1] = rho*u      x-momentum
//   Q[2] = rho*v      y-momentum
//   Q[3] = rho*w      z-momentum
//   Q[4] = E          total energy
//
// Sign convention (DESIGN.md §EOS):
//   p = (gamma-1) * (E - 0.5*rho*(u^2+v^2+w^2))
//   T = p / (rho * R)
//   c = sqrt(gamma * p / rho)

#include <array>
#include <vector>
#include <cstddef>
#include <cmath>
#include <cassert>

// ── Constants ─────────────────────────────────────────────────────────────────────────────────
inline constexpr int    NB     = 8;          // cells per block side
inline constexpr int    NG     = 2;          // ghost layers each side
inline constexpr int    NB2    = NB + 2*NG;  // side with ghosts = 12
inline constexpr int    NCELL  = NB2*NB2*NB2;// total storage per field
inline constexpr int    NVAR   = 5;          // number of conserved variables
inline constexpr double GAMMA  = 1.4;
// FIX P0.5: 287.058 J/(kg·K) — CODATA dry-air value.
// Was 287.0; GPU_R_GAS in gpu_constants.cuh was already 287.058.
// The 0.058 discrepancy produced different pressures at the same (rho,T)
// on CPU vs GPU paths, silently corrupting cross-validation comparisons.
inline constexpr double R_GAS  = 287.058;    // J/(kg·K), dry air (CODATA)
static constexpr double CPU_CP = GAMMA * R_GAS / (GAMMA - 1.0);

// ── Index helpers (always inline, zero overhead) ────────────────────────────────────────────
inline constexpr int cell_idx(int i, int j, int k) noexcept {
    return k * NB2*NB2 + j * NB2 + i;
}
// Interior range: [NG, NB+NG-1] = [1, NB]
inline constexpr int ilo() noexcept { return NG; }
inline constexpr int ihi() noexcept { return NG + NB - 1; }

// ── Primitive variable struct (local, not stored) ──────────────────────────────────────────
struct Prim {
    double rho, u, v, w, p, T, c;
};

// ── Equation of state (inline — called millions of times) ────────────────────────────
inline Prim eos_cons_to_prim(double rho, double rhou, double rhov,
                              double rhow, double E) noexcept
{
    Prim q;
    q.rho = rho;
    q.u   = rhou / rho;
    q.v   = rhov / rho;
    q.w   = rhow / rho;
    q.p   = (GAMMA - 1.0) * (E - 0.5 * rho * (q.u*q.u + q.v*q.v + q.w*q.w));
    q.T   = q.p / (rho * R_GAS);
    q.c   = std::sqrt(GAMMA * q.p / rho);
    return q;
}

inline void eos_prim_to_cons(const Prim& q,
                              double& rho, double& rhou, double& rhov,
                              double& rhow, double& E) noexcept
{
    rho  = q.rho;
    rhou = q.rho * q.u;
    rhov = q.rho * q.v;
    rhow = q.rho * q.w;
    E    = q.p / (GAMMA - 1.0) + 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
}

// ── Sutherland viscosity ─────────────────────────────────────────────────────────────────────────────
inline double sutherland(double T) noexcept {
    constexpr double mu_ref  = 1.716e-5;   // kg/(m·s)
    constexpr double T_ref   = 273.15;     // K
    constexpr double S       = 110.4;      // K
    // FIX P0.4: replaced std::pow(T/T_ref, 1.5) with (T/T_ref)*std::sqrt(T/T_ref).
    // Mathematically identical: x^1.5 = x * sqrt(x).
    // Avoids exp(1.5*log(x)) path in libm (~50 cycles) in favour of
    // one multiply + one sqrt (~5 cycles). Now matches gpu_sutherland().
    double ratio = T / T_ref;
    return mu_ref * ratio * std::sqrt(ratio) * (T_ref + S) / (T + S);
}

// ── CellBlock ───────────────────────────────────────────────────────────────────────────────────────
struct CellBlock {
    // SoA: one array per conserved variable, size NCELL each.
    std::array<std::vector<double>, NVAR> Q;

    // Physical domain origin and cell size for this block.
    double ox = 0, oy = 0, oz = 0;   // origin of interior region
    double h  = 0;                    // uniform cell size (cubic cells)

    // Constructors
    CellBlock() { for (auto& f : Q) f.assign(NCELL, 0.0); }
    explicit CellBlock(double ox_, double oy_, double oz_, double h_)
        : ox(ox_), oy(oy_), oz(oz_), h(h_)
    { for (auto& f : Q) f.assign(NCELL, 0.0); }

    // Field accessors (inline)
    double& rho (int i,int j,int k) noexcept { return Q[0][cell_idx(i,j,k)]; }
    double& rhou(int i,int j,int k) noexcept { return Q[1][cell_idx(i,j,k)]; }
    double& rhov(int i,int j,int k) noexcept { return Q[2][cell_idx(i,j,k)]; }
    double& rhow(int i,int j,int k) noexcept { return Q[3][cell_idx(i,j,k)]; }
    double& E   (int i,int j,int k) noexcept { return Q[4][cell_idx(i,j,k)]; }

    double rho (int i,int j,int k) const noexcept { return Q[0][cell_idx(i,j,k)]; }
    double rhou(int i,int j,int k) const noexcept { return Q[1][cell_idx(i,j,k)]; }
    double rhov(int i,int j,int k) const noexcept { return Q[2][cell_idx(i,j,k)]; }
    double rhow(int i,int j,int k) const noexcept { return Q[3][cell_idx(i,j,k)]; }
    double E   (int i,int j,int k) const noexcept { return Q[4][cell_idx(i,j,k)]; }

    // Primitive variables at interior cell (i,j,k), i,j,k in [ilo,ihi]
    Prim prim(int i, int j, int k) const noexcept {
        return eos_cons_to_prim(rho(i,j,k), rhou(i,j,k),
                                rhov(i,j,k), rhow(i,j,k), E(i,j,k));
    }

    // In cell_block.hpp
    double u(int i,int j,int k) const { Prim q=prim(i,j,k); return q.u; }
    double v(int i,int j,int k) const { Prim q=prim(i,j,k); return q.v; }
    double w(int i,int j,int k) const { Prim q=prim(i,j,k); return q.w; }
    double T(int i,int j,int k) const { Prim q=prim(i,j,k); return q.T; }

    // Physical cell-center coordinates
    double xc(int i) const noexcept { return ox + (i - NG + 0.5) * h; }
    double yc(int j) const noexcept { return oy + (j - NG + 0.5) * h; }
    double zc(int k) const noexcept { return oz + (k - NG + 0.5) * h; }

    // Global mass / momentum / energy (interior cells only)
    double total_mass()     const noexcept;
    double total_energy()   const noexcept;
    double total_momentum_x() const noexcept;

    // CFL time step for this block
    double cfl_dt(double cfl) const noexcept;

    // Zero ghost layer (all 6 faces)
    void zero_ghosts() noexcept;
};
