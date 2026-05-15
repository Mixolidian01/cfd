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
#include "mesh/axis.hpp"
#include "vendor/mdspan.hpp"

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
    double gamma_m = GAMMA;  // mixture γ  (default: ideal gas)
    double p_inf_m = 0.0;    // stiffened-gas p∞ (default: 0 → ideal gas)
};

// ── Allaire 2002 mixture EOS (stiffened gas) ─────────────────────────────────
// Allaire et al., J. Comput. Phys. 181 (2002) 577-616, eqs. (3)-(4).
//   1/(γ_m-1) = φ/(γ_A-1) + (1-φ)/(γ_B-1)
//   γ_m·p∞_m/(γ_m-1) = φ·γ_A·p∞_A/(γ_A-1) + (1-φ)·γ_B·p∞_B/(γ_B-1)
inline void mix_eos(double phi,
                    double ga, double gb, double pia, double pib,
                    double& gm, double& pim) noexcept
{
    const double gm1_inv = phi / (ga - 1.0) + (1.0 - phi) / (gb - 1.0);
    gm  = 1.0 + 1.0 / gm1_inv;
    pim = (gm - 1.0) * (phi * ga * pia / (ga - 1.0) + (1.0 - phi) * gb * pib / (gb - 1.0)) / gm;
}

// ── Equation of state (inline — called millions of times) ────────────────────

// Configurable-gamma variant — same formula, gamma passed as argument.
inline Prim eos_cons_to_prim_g(double rho, double rhou, double rhov,
                                double rhow, double E, double gamma) noexcept
{
    Prim q;
    q.rho = rho;
    q.u   = rhou / rho;
    q.v   = rhov / rho;
    q.w   = rhow / rho;
    q.p   = (gamma - 1.0) * (E - 0.5 * rho * (q.u*q.u + q.v*q.v + q.w*q.w));
    q.T   = q.p / (rho * R_GAS);
    q.c   = std::sqrt(gamma * q.p / rho);
    return q;
}

// Fixed-gamma wrapper (gamma = GAMMA = 1.4); gamma_m and p_inf_m keep ideal-gas defaults.
inline Prim eos_cons_to_prim(double rho, double rhou, double rhov,
                              double rhow, double E) noexcept {
    return eos_cons_to_prim_g(rho, rhou, rhov, rhow, E, GAMMA);
}

// Stiffened-gas variant: uses mixture γ_m and p∞_m derived from φ.
inline Prim eos_cons_to_prim_sg(double rho, double rhou, double rhov,
                                 double rhow, double E,
                                 double gm, double pim) noexcept
{
    Prim q;
    q.rho     = rho;
    q.u       = rhou / rho;
    q.v       = rhov / rho;
    q.w       = rhow / rho;
    q.gamma_m = gm;
    q.p_inf_m = pim;
    q.p   = (gm - 1.0) * (E - 0.5 * rho * (q.u*q.u + q.v*q.v + q.w*q.w)) - gm * pim;
    if (q.p < -pim + 1e-10) q.p = -pim + 1e-10;  // positivity floor: p+p∞ > 0
    q.T   = (q.p + pim) / (rho * R_GAS);
    q.c   = std::sqrt(gm * (q.p + pim) / rho);
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
    // Stiffened gas: E = (p + γ·p∞)/(γ-1) + KE = (p+p∞)/(γ-1) + p∞/(γ-1)... simplifies to:
    //   E = p/(γ-1) + γ·p∞/(γ-1) + KE  →  for ideal gas (p∞=0): E = p/(γ-1) + KE ✓
    E = (q.p + q.gamma_m * q.p_inf_m) / (q.gamma_m - 1.0)
        + 0.5 * q.rho * (q.u*q.u + q.v*q.v + q.w*q.w);
}

// ── Sutherland viscosity ─────────────────────────────────────────────────────
inline double sutherland(double T) noexcept {
    constexpr double mu_ref  = 1.716e-5;
    constexpr double T_ref   = 273.15;
    constexpr double S       = 110.4;
    if (T < 1.0) T = 1.0;  // floor prevents NaN from transient negative T
    double ratio = T / T_ref;
    return mu_ref * ratio * std::sqrt(ratio) * (T_ref + S) / (T + S);
}

// ── AoSoAAccessor ─────────────────────────────────────────────────────────────
// Custom mdspan accessor: translates a logical flat cell index into the
// CellBlock AoSoA physical address.
//
//   CellBlock layout: data_[tile*NVAR*W + v*W + lane]
//   where tile = flat >> 3,  lane = flat & 7,  W = 8
//
// Declared before CellBlock so it can appear in CellBlock's type aliases.
struct AoSoAAccessor {
    using element_type     = double;
    using data_handle_type = double*;
    using reference        = double&;
    using offset_policy    = AoSoAAccessor;

    int v_;

    constexpr explicit AoSoAAccessor(int v) noexcept : v_(v) {}

    [[nodiscard]] constexpr reference
    access(data_handle_type p, std::ptrdiff_t flat) const noexcept {
        // W=8, NVAR=5 (compile-time constants; match CellBlock::W and ::NVAR).
        return p[(flat >> 3) * (NVAR * 8) + v_ * 8 + (flat & 7)];
    }

    // offset() is only used for submdspan; not supported with AoSoA storage.
    [[nodiscard]] constexpr data_handle_type
    offset(data_handle_type p, std::ptrdiff_t) const noexcept { return p; }
};

// ── axis_strides<DIR> ─────────────────────────────────────────────────────────
// Returns layout_stride strides so that view(n, a, b) maps to:
//   cell_idx(n,a,b) for X,  cell_idx(a,n,b) for Y,  cell_idx(a,b,n) for Z.
//
// Derivation from cell_idx(i,j,k) = i + j·NB2 + k·NB2²:
//   X: strides (1,    NB2,  NB2²) — n=i, a=j, b=k
//   Y: strides (NB2,  1,    NB2²) — n=j, a=i, b=k
//   Z: strides (NB2², 1,    NB2 ) — n=k, a=i, b=j
template<Axis DIR>
[[nodiscard]] constexpr std::array<int, 3> axis_strides() noexcept {
    if constexpr (DIR == Axis::X) return {1,        NB2,  NB2*NB2};
    if constexpr (DIR == Axis::Y) return {NB2,      1,    NB2*NB2};
    /* Z */                       return {NB2*NB2,  1,    NB2    };
}

// cell_idx_axis<DIR>(n, a, b): flat index for the cell at position n along the
// face-normal direction for Axis DIR, and (a, b) tangentially.
// Equivalent to: X→cell_idx(n,a,b), Y→cell_idx(a,n,b), Z→cell_idx(a,b,n).
template<Axis DIR>
[[nodiscard]] constexpr int cell_idx_axis(int n, int a, int b) noexcept {
    const auto s = axis_strides<DIR>();
    return n * s[0] + a * s[1] + b * s[2];
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

    // P14.1: phase-field scalar φ ∈ [0,1] (0 = phase 1, 1 = phase 2).
    // Stored as flat SoA (separate from the NVAR AoSoA) so NVAR/AVX layout
    // is unchanged.  Ghost layers present (same indexing as Q).
    // Inactive (single-phase) when use_acdi=false — zero-cost if not used.
    double phi_data_[NCELL] = {};

    double  phi(int flat)       const noexcept { return phi_data_[flat]; }
    double& phi(int flat)             noexcept { return phi_data_[flat]; }
    double  phi(int i,int j,int k) const noexcept { return phi_data_[cell_idx(i,j,k)]; }
    double& phi(int i,int j,int k)       noexcept { return phi_data_[cell_idx(i,j,k)]; }

    // ── Constructors ──────────────────────────────────────────────────────────
    CellBlock() noexcept {
        init_views();
        std::fill(data_, data_ + NTILE*NVAR*W, 0.0);
        std::fill(phi_data_, phi_data_ + NCELL, 0.0);
    }
    explicit CellBlock(double ox_, double oy_, double oz_, double h_) noexcept
        : ox(ox_), oy(oy_), oz(oz_), h(h_)
    {
        init_views();
        std::fill(data_, data_ + NTILE*NVAR*W, 0.0);
        std::fill(phi_data_, phi_data_ + NCELL, 0.0);
    }

    // Deep-copy: memcpy entire AoSoA buffer and phi; views point to this->data_.
    CellBlock(const CellBlock& o) noexcept
        : ox(o.ox), oy(o.oy), oz(o.oz), h(o.h)
    {
        init_views();
        std::memcpy(data_,     o.data_,     sizeof(data_));
        std::memcpy(phi_data_, o.phi_data_, sizeof(phi_data_));
    }
    CellBlock& operator=(const CellBlock& o) noexcept {
        if (this != &o) {
            ox = o.ox; oy = o.oy; oz = o.oz; h = o.h;
            std::memcpy(data_,     o.data_,     sizeof(data_));
            std::memcpy(phi_data_, o.phi_data_, sizeof(phi_data_));
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

    // Flat-index variants — complement the (i,j,k) forms above.
    double& rho (int f) noexcept { return Q[0][f]; }
    double& rhou(int f) noexcept { return Q[1][f]; }
    double& rhov(int f) noexcept { return Q[2][f]; }
    double& rhow(int f) noexcept { return Q[3][f]; }
    double& E   (int f) noexcept { return Q[4][f]; }
    double  rho (int f) const noexcept { return Q[0][f]; }
    double  rhou(int f) const noexcept { return Q[1][f]; }
    double  rhov(int f) const noexcept { return Q[2][f]; }
    double  rhow(int f) const noexcept { return Q[3][f]; }
    double  E   (int f) const noexcept { return Q[4][f]; }

    // P14.1c: stiffened-gas EOS parameters — single source of truth.
    // Set via set_sg_eos(); read by prim(), cfl_dt(), and operators.cpp.
    static inline bool   sg_active_ = false;
    static inline double sg_ga_     = GAMMA, sg_gb_ = GAMMA;
    static inline double sg_pia_    = 0.0,   sg_pib_ = 0.0;

    static void set_sg_eos(bool active, double ga, double gb,
                           double pia, double pib) noexcept {
        sg_active_ = active;
        sg_ga_ = ga; sg_gb_ = gb;
        sg_pia_ = pia; sg_pib_ = pib;
    }

    // Primitive variables at cell (i,j,k)
    Prim prim(int i, int j, int k) const noexcept {
        if (!sg_active_) {
            return eos_cons_to_prim(rho(i,j,k), rhou(i,j,k),
                                    rhov(i,j,k), rhow(i,j,k), E(i,j,k));
        }
        const int flat = cell_idx(i,j,k);
        double gm, pim;
        mix_eos(phi_data_[flat], sg_ga_, sg_gb_, sg_pia_, sg_pib_, gm, pim);
        return eos_cons_to_prim_sg(Q[0][flat], Q[1][flat], Q[2][flat],
                                   Q[3][flat], Q[4][flat], gm, pim);
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

    // ── R6: axis-aligned mdspan views ────────────────────────────────────────
    //
    // axis_view<DIR>(v) returns a 3-D mdspan over variable v where:
    //   index 0 (n) runs along the face-normal direction for Axis DIR,
    //   indices 1 and 2 (a, b) run along the two tangential directions.
    //
    // This means view(n, a, b) gives the same cell as:
    //   Q[v][cell_idx(n, a, b)]  for Axis::X
    //   Q[v][cell_idx(a, n, b)]  for Axis::Y
    //   Q[v][cell_idx(a, b, n)]  for Axis::Z
    //
    // The AoSoA remapping is handled by AoSoAAccessor; no data is moved.
    // The layout policy is layout_stride with strides set by axis_strides<DIR>().

    using BlockExtents = md::extents<int, NB2, NB2, NB2>;
    using BlockView    = md::mdspan<double, BlockExtents,
                                    md::layout_stride, AoSoAAccessor>;

    template<Axis DIR>
    [[nodiscard]] BlockView axis_view(int v) noexcept {
        md::layout_stride::mapping<BlockExtents> m{BlockExtents{}, axis_strides<DIR>()};
        return BlockView{data_, m, AoSoAAccessor{v}};
    }
    template<Axis DIR>
    [[nodiscard]] BlockView axis_view(int v) const noexcept {
        md::layout_stride::mapping<BlockExtents> m{BlockExtents{}, axis_strides<DIR>()};
        return BlockView{const_cast<double*>(data_), m, AoSoAAccessor{v}};
    }

private:
    // Bind Q[v].data_ → this->data_ and Q[v].v_ → v.
    // Must be called before any other member use.
    void init_views() noexcept {
        for (int v = 0; v < NVAR; ++v) { Q[v].data_ = data_; Q[v].v_ = v; }
    }
};

