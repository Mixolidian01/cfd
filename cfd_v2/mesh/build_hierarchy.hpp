#pragma once

#include <array>

#include "cfd_v2/mesh/hierarchy.hpp"

namespace cfd_v2::mesh {

// Configuration for constructing a uniform root level that tessellates a
// rectangular domain with patches of size NI x NJ x NK interior cells.
struct RootGridConfig {
    std::array<double,3> xlo{0.0, 0.0, 0.0}; // physical lower bounds (x,y,z)
    std::array<double,3> xhi{1.0, 1.0, 1.0}; // physical upper bounds (x,y,z)

    // Number of patches along each axis at the root level. The total number of
    // root-level cells per axis is n_patches[d] * N[d], where N = (NI,NJ,NK).
    std::array<int,3> n_patches{1, 1, 1};
};

// Build a hierarchy with a single root level that uniformly tiles the domain
// specified by cfg. This is the core of Phase 1: it provides a physically
// meaningful mapping from discrete indices to coordinates and ensures that each
// patch has consistent geometry.
//
// Requirements:
//   - (xhi[d] - xlo[d]) > 0 for all d.
//   - n_patches[d] > 0 for all d.
//
// The root level has level_index = 0. Child levels and AMR refinement will be
// added in later phases.
inline Hierarchy make_uniform_root_hierarchy(const RootGridConfig& cfg)
{
    Hierarchy H;
    Level root;
    root.level_index = 0;

    const auto nx_p = cfg.n_patches[0];
    const auto ny_p = cfg.n_patches[1];
    const auto nz_p = cfg.n_patches[2];

    const double Lx = cfg.xhi[0] - cfg.xlo[0];
    const double Ly = cfg.xhi[1] - cfg.xlo[1];
    const double Lz = cfg.xhi[2] - cfg.xlo[2];

    // Total interior cells per axis at the root level.
    const double Nx_tot = static_cast<double>(nx_p * NI);
    const double Ny_tot = static_cast<double>(ny_p * NJ);
    const double Nz_tot = static_cast<double>(nz_p * NK);

    const double hx = Lx / Nx_tot;
    const double hy = Ly / Ny_tot;
    const double hz = Lz / Nz_tot;

    root.patches.reserve(static_cast<std::size_t>(nx_p * ny_p * nz_p));

    for (int kz = 0; kz < nz_p; ++kz) {
        for (int jy = 0; jy < ny_p; ++jy) {
            for (int ix = 0; ix < nx_p; ++ix) {
                Patch<> patch;
                patch.geom.level = 0;
                patch.geom.hx = hx;
                patch.geom.hy = hy;
                patch.geom.hz = hz;

                // Physical coordinates of the first interior cell center in
                // this patch. Each patch covers NI,NJ,NK interior cells.
                const double x0_cell = cfg.xlo[0] + (ix * NI + 0.5) * hx;
                const double y0_cell = cfg.xlo[1] + (jy * NJ + 0.5) * hy;
                const double z0_cell = cfg.xlo[2] + (kz * NK + 0.5) * hz;

                patch.geom.x0 = x0_cell;
                patch.geom.y0 = y0_cell;
                patch.geom.z0 = z0_cell;

                root.patches.emplace_back(std::move(patch));
            }
        }
    }

    H.levels.emplace_back(std::move(root));
    return H;
}

} // namespace cfd_v2::mesh
