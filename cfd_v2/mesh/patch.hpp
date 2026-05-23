#pragma once

#include <array>
#include <cstddef>

namespace cfd_v2::mesh {

// Ghost width: number of ghost cells on each side per axis.
inline constexpr int NG  = 2;

// Interior cell counts per patch side. These are tunable but must be the same
// across all patches in a given hierarchy. Values like 32^3 are GPU-friendly.
inline constexpr int NI  = 32;
inline constexpr int NJ  = 32;
inline constexpr int NK  = 32;

// Total cells per axis including ghosts.
inline constexpr int NI_TOT = NI + 2 * NG;
inline constexpr int NJ_TOT = NJ + 2 * NG;
inline constexpr int NK_TOT = NK + 2 * NG;

// Total cell count per patch (including ghosts).
inline constexpr int NCELL  = NI_TOT * NJ_TOT * NK_TOT;

// Number of conserved variables for compressible flow: rho, rho*u, rho*v,
// rho*w, total energy.
inline constexpr int NVAR   = 5;

// Geometry metadata for a single patch.
//
// x0,y0,z0   : coordinates of the cell-center at (i=NG, j=NG, k=NG).
// hx,hy,hz   : uniform spacings per axis at this level.
// level      : refinement level index (0 = coarsest).
struct PatchGeometry {
    double x0{0.0}, y0{0.0}, z0{0.0};
    double hx{1.0}, hy{1.0}, hz{1.0};
    int    level{0};
};

// Patch: cell-centered finite-volume block with NG ghost cells on each side.
//
// For now, we store data as a simple structure-of-arrays with one contiguous
// array for all conserved variables and another for the phase field phi.
// This is physically and mathematically correct; AoSoA / GPU-optimized
// layouts can be introduced later without changing the interfaces.
//
// Indexing convention:
//   i in [0, NI_TOT), interior i in [NG, NG+NI-1]
//   j in [0, NJ_TOT), interior j in [NG, NG+NJ-1]
//   k in [0, NK_TOT), interior k in [NG, NG+NK-1]
//
// The flatten() helper maps (i,j,k) to a flat index in [0,NCELL).

template<typename Real = double>
struct Patch {
    using real_type = Real;

    PatchGeometry geom{};

    // Conserved variables Q[v][cell]: v=0..NVAR-1, cell=0..NCELL-1.
    std::array<real_type, NVAR * NCELL> q{};

    // Phase field phi, or other scalar (e.g. volume fraction).
    std::array<real_type, NCELL> phi{};

    // Flatten (i,j,k) to a single index.
    [[nodiscard]] static constexpr int flatten(int i, int j, int k) noexcept {
        return (k * NJ_TOT + j) * NI_TOT + i;
    }

    // Interior index bounds for convenience.
    [[nodiscard]] static constexpr int ilo() noexcept { return NG; }
    [[nodiscard]] static constexpr int ihi_i() noexcept { return NG + NI - 1; }
    [[nodiscard]] static constexpr int ihi_j() noexcept { return NG + NJ - 1; }
    [[nodiscard]] static constexpr int ihi_k() noexcept { return NG + NK - 1; }

    // Accessors for conserved variables by (i,j,k).
    [[nodiscard]] real_type& rho (int i, int j, int k) noexcept {
        return q[0 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] real_type& rhou(int i, int j, int k) noexcept {
        return q[1 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] real_type& rhov(int i, int j, int k) noexcept {
        return q[2 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] real_type& rhow(int i, int j, int k) noexcept {
        return q[3 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] real_type& E   (int i, int j, int k) noexcept {
        return q[4 * NCELL + flatten(i,j,k)];
    }

    [[nodiscard]] const real_type& rho (int i, int j, int k) const noexcept {
        return q[0 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] const real_type& rhou(int i, int j, int k) const noexcept {
        return q[1 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] const real_type& rhov(int i, int j, int k) const noexcept {
        return q[2 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] const real_type& rhow(int i, int j, int k) const noexcept {
        return q[3 * NCELL + flatten(i,j,k)];
    }
    [[nodiscard]] const real_type& E   (int i, int j, int k) const noexcept {
        return q[4 * NCELL + flatten(i,j,k)];
    }

    // Phase field accessors.
    [[nodiscard]] real_type& phi_cell(int i, int j, int k) noexcept {
        return phi[flatten(i,j,k)];
    }
    [[nodiscard]] const real_type& phi_cell(int i, int j, int k) const noexcept {
        return phi[flatten(i,j,k)];
    }

    // Physical cell-center coordinates for given indices.
    [[nodiscard]] double xc(int i) const noexcept {
        return geom.x0 + (static_cast<double>(i - NG) + 0.5) * geom.hx;
    }
    [[nodiscard]] double yc(int j) const noexcept {
        return geom.y0 + (static_cast<double>(j - NG) + 0.5) * geom.hy;
    }
    [[nodiscard]] double zc(int k) const noexcept {
        return geom.z0 + (static_cast<double>(k - NG) + 0.5) * geom.hz;
    }
};

} // namespace cfd_v2::mesh
