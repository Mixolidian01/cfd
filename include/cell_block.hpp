#pragma once
// DESIGN.md reference: Layer 1 — Cell-Centered Block
//
// P4.2: AoSoA (Array-of-Structures-of-Arrays) memory layout for AVX-512 CPU.
//
// Layout: data_[tile * NVAR * W + v * W + lane]
//   flat  = cell_idx(i,j,k) = k*NB2² + j*NB2 + i   (same as before)
//   tile  = flat >> 3        (flat / W, W=8)
//   lane  = flat &  7        (flat % W)
//
// With W=8=AVX-512 double lanes and NTILE=NCELL/W=216:
//   tile_ptr(v, t) = data_ + t*NVAR*W + v*W   — 64-byte aligned for all (t,v)
//   because NVAR*W = 40 = 5×8, each tile occupies 5×64 = 320 bytes (5 cache lines).
//
// Conserved variables (Q):
//   Q[0] = rho,  Q[1] = ρu,  Q[2] = ρv,  Q[3] = ρw,  Q[4] = E
//
// Sign convention (DESIGN.md §EOS):
//   p = (γ-1)·(E - ½ρ|u|²),   T = p/(ρR),   c = √(γp/ρ)

#include <array>
#include <cstddef>
#include <cmath>
#include <cassert>
#include <cstring>   // std::memcpy / std::memset
#include <algorithm> // std::fill

// ── Constants ─────────────────────────────────────────────────────────────────────────────────
inline constexpr int    NB     = 8;          // cells per block side (interior)
inline constexpr int    NG     = 2;          // ghost layers each side
inline constexpr int    NB2    = NB + 2*NG;  // total cells per axis = 12
inline constexpr int    NCELL  = NB2*NB2*NB2;// total cells per field = 1728
inline constexpr int    NVAR   = 5;          // ρ, ρu, ρv, ρw, E
inline constexpr double GAMMA  = 1.4;
inline constexpr double R_GAS  = 287.058;    // J/(kg·K), dry air (CODATA)
inline constexpr double PR     = 0.72;        // Prandtl number (dry air)
static constexpr double CPU_CP = GAMMA * R_GAS / (GAMMA - 1.0);

// ── Index helpers (always inline, zero overhead) ────────────────────────────────────────────
inline constexpr int cell_idx(int i, int j, int k) noexcept {
    return k * NB2*NB2 + j * NB2 + i;
}
// Interior range: [NG, NB+NG-1]
inline constexpr int ilo() noexcept { return NG; }
inline constexpr int ihi() noexcept { return NG + NB - 1; }

// ── Primitive variable struct (local, not stored) ──────────────────────────────────────────
struct Prim {
    double rho, u, v, w, p, T, c;
};

// ── Equation of state (inline — called millions of times) ────────────────────
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

// ── Sutherland viscosity ─────────────────────────────────────────────────────
inline double sutherland(double T) noexcept {
    constexpr double mu_ref  = 1.716e-5;
    constexpr double T_ref   = 273.15;
    constexpr double S       = 110.4;
    double ratio = T / T_ref;
    return mu_ref * ratio * std::sqrt(ratio) * (T_ref + S) / (T + S);
}

// ── CellBlock ────────────────────────────────────────────────────────────────
// P4.2: aligned AoSoA storage; identical API to callers (Q[v][flat] syntax).
struct alignas(64) CellBlock {

    static constexpr int W     = 8;            // SIMD width = AVX-512 doubles
    static constexpr int NTILE = NCELL / W;    // 1728 / 8 = 216
    static_assert(NCELL % W == 0, "NCELL must be divisible by W=8");

    // ── AoSoA buffer: first member → offset 0 from struct base → 64-byte aligned ──
    // data_[tile * NVAR * W + v * W + lane]
    // All tile_ptr(v, t) are 64-byte aligned: offset = (t*NVAR + v)*W*8 bytes;
    // NVAR*W=40, v*W=0/8/16/24/32 → all multiples of W*sizeof(double)=64 bytes.
    alignas(64) double data_[NTILE * NVAR * W];  // 8640 doubles = 69 120 bytes

    // ── FieldProxy: Q[v][flat] → double& via AoSoA index ────────────────────
    struct FieldProxy {
        double* data_;  // raw AoSoA buffer (= CellBlock::data_)
        int     v_;     // variable index 0..NVAR-1

        double& operator[](int flat) noexcept {
            return data_[(flat >> 3) * (NVAR * W) + v_ * W + (flat & 7)];
        }
        double operator[](int flat) const noexcept {
            return data_[(flat >> 3) * (NVAR * W) + v_ * W + (flat & 7)];
        }

        // Contiguous pointer to W=8 doubles for variable v_ in tile t.
        // 64-byte aligned for all (v_, t) — see class-level comment.
        double*       tile_ptr(int t) noexcept       { return data_ + t*(NVAR*W) + v_*W; }
        const double* tile_ptr(int t) const noexcept { return data_ + t*(NVAR*W) + v_*W; }

        // Fill all NCELL slots with a scalar (replaces vector::assign).
        void assign(int /*n*/, double val) noexcept {
            for (int t = 0; t < NTILE; ++t) {
                double* p = tile_ptr(t);
                for (int lane = 0; lane < W; ++lane) p[lane] = val;
            }
        }

        // Serialise this variable to a flat SoA array (for checkpoint write).
        void copy_to_flat(double* dst) const noexcept {
            for (int t = 0; t < NTILE; ++t) {
                const double* sp = tile_ptr(t);
                double*       dp = dst + t * W;
                for (int lane = 0; lane < W; ++lane) dp[lane] = sp[lane];
            }
        }

        // Deserialise from a flat SoA array (for checkpoint read).
        void assign_from_flat(const double* src) noexcept {
            for (int t = 0; t < NTILE; ++t) {
                double*       dp = tile_ptr(t);
                const double* sp = src + t * W;
                for (int lane = 0; lane < W; ++lane) dp[lane] = sp[lane];
            }
        }

        // Deep-copy values from src proxy (different block, same var index).
        FieldProxy& operator=(const FieldProxy& src) noexcept {
            if (data_ == src.data_ && v_ == src.v_) return *this;
            for (int t = 0; t < NTILE; ++t)
                std::memcpy(tile_ptr(t), src.tile_ptr(t), W * sizeof(double));
            return *this;
        }
    };

    // ── Public Q accessor array ───────────────────────────────────────────────
    std::array<FieldProxy, NVAR> Q;

    // ── Physical domain metadata ──────────────────────────────────────────────
    double ox = 0, oy = 0, oz = 0;
    double h  = 0;

    // ── Constructors ──────────────────────────────────────────────────────────
    CellBlock() noexcept {
        init_views();
        std::fill(data_, data_ + NTILE*NVAR*W, 0.0);
    }
    explicit CellBlock(double ox_, double oy_, double oz_, double h_) noexcept
        : ox(ox_), oy(oy_), oz(oz_), h(h_)
    {
        init_views();
        std::fill(data_, data_ + NTILE*NVAR*W, 0.0);
    }

    // Deep-copy: memcpy entire AoSoA buffer; views point to this->data_.
    CellBlock(const CellBlock& o) noexcept
        : ox(o.ox), oy(o.oy), oz(o.oz), h(o.h)
    {
        init_views();
        std::memcpy(data_, o.data_, sizeof(data_));
    }
    CellBlock& operator=(const CellBlock& o) noexcept {
        if (this != &o) {
            ox = o.ox; oy = o.oy; oz = o.oz; h = o.h;
            std::memcpy(data_, o.data_, sizeof(data_));
            // Q[v].data_ already == this->data_; no pointer update needed.
        }
        return *this;
    }
    // Move = copy (inline array cannot be moved in memory; noexcept for std::vector).
    CellBlock(CellBlock&& o) noexcept : CellBlock(static_cast<const CellBlock&>(o)) {}
    CellBlock& operator=(CellBlock&& o) noexcept {
        return *this = static_cast<const CellBlock&>(o);
    }

    // ── Named field accessors (inline) ────────────────────────────────────────
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

    // Primitive variables at cell (i,j,k)
    Prim prim(int i, int j, int k) const noexcept {
        return eos_cons_to_prim(rho(i,j,k), rhou(i,j,k),
                                rhov(i,j,k), rhow(i,j,k), E(i,j,k));
    }

    double u(int i,int j,int k) const { Prim q=prim(i,j,k); return q.u; }
    double v(int i,int j,int k) const { Prim q=prim(i,j,k); return q.v; }
    double w(int i,int j,int k) const { Prim q=prim(i,j,k); return q.w; }
    double T(int i,int j,int k) const { Prim q=prim(i,j,k); return q.T; }

    // Physical cell-center coordinates
    double xc(int i) const noexcept { return ox + (i - NG + 0.5) * h; }
    double yc(int j) const noexcept { return oy + (j - NG + 0.5) * h; }
    double zc(int k) const noexcept { return oz + (k - NG + 0.5) * h; }

    // Reductions (interior cells only)
    double total_mass()       const noexcept;
    double total_energy()     const noexcept;
    double total_momentum_x() const noexcept;

    double cfl_dt(double cfl) const noexcept;
    void   zero_ghosts()      noexcept;

private:
    // Bind Q[v].data_ → this->data_ and Q[v].v_ → v.
    // Must be called before any other member use.
    void init_views() noexcept {
        for (int v = 0; v < NVAR; ++v) { Q[v].data_ = data_; Q[v].v_ = v; }
    }
};
