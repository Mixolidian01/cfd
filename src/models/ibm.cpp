// ibm.cpp — Ghost-cell immersed boundary method (P7.3)
// Mittal & Iaccarino (2005) ghost-cell direct-forcing IBM.
#include "models/ibm.hpp"
#include "solver/ns_solver.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

// =============================================================================
// classify_ibm_cells
// =============================================================================
void classify_ibm_cells(const CellBlock& blk, const LevelSet& ls,
                        CellType* cell_type) noexcept {
    // First pass: mark FLUID / SOLID
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        double x = blk.ox + (i - NG + 0.5) * blk.h;
        double y = blk.oy + (j - NG + 0.5) * blk.h;
        double z = blk.oz + (k - NG + 0.5) * blk.h;
        cell_type[cell_idx(i,j,k)] = (ls.phi(x,y,z) >= 0.0)
                                   ? CellType::FLUID : CellType::SOLID;
    }

    // Second pass: any FLUID cell adjacent to a SOLID cell becomes IBM_GHOST
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        if (cell_type[cell_idx(i,j,k)] != CellType::FLUID) continue;

        bool adj_solid = false;
        static const int di[6]={1,-1,0,0,0,0};
        static const int dj[6]={0,0,1,-1,0,0};
        static const int dk[6]={0,0,0,0,1,-1};
        for (int f = 0; f < 6; ++f) {
            int ni=i+di[f], nj=j+dj[f], nk=k+dk[f];
            // clamp to interior range (ghost layer cells: treat as FLUID for safety)
            if (ni < NG || ni >= NG+NB ||
                nj < NG || nj >= NG+NB ||
                nk < NG || nk >= NG+NB) continue;
            if (cell_type[cell_idx(ni,nj,nk)] == CellType::SOLID) {
                adj_solid = true;
                break;
            }
        }
        if (adj_solid)
            cell_type[cell_idx(i,j,k)] = CellType::IBM_GHOST;
    }
}

// =============================================================================
// Trilinear interpolation of a conserved variable within one block.
// Returns the interpolated value at physical position (px, py, pz).
// Out-of-interior points are clamped to the nearest interior cell.
// =============================================================================
static double trilinear(const CellBlock& blk, int var,
                        double px, double py, double pz) noexcept {
    // Convert physical coords to continuous cell index (NG-based)
    double fi = (px - blk.ox) / blk.h + NG - 0.5;
    double fj = (py - blk.oy) / blk.h + NG - 0.5;
    double fk = (pz - blk.oz) / blk.h + NG - 0.5;

    // Floor to lower-left cell; clamp so we stay within [NG, NG+NB-1]
    auto clamp = [](double v, int lo, int hi) -> int {
        int iv = (int)v;
        return std::max(lo, std::min(hi-1, iv));
    };
    int i0 = clamp(fi, NG, NG+NB);
    int j0 = clamp(fj, NG, NG+NB);
    int k0 = clamp(fk, NG, NG+NB);
    int i1 = std::min(i0+1, NG+NB-1);
    int j1 = std::min(j0+1, NG+NB-1);
    int k1 = std::min(k0+1, NG+NB-1);

    double tx = fi - i0;  // in [0,1]
    double ty = fj - j0;
    double tz = fk - k0;
    tx = std::max(0.0, std::min(1.0, tx));
    ty = std::max(0.0, std::min(1.0, ty));
    tz = std::max(0.0, std::min(1.0, tz));

    double c000 = blk.Q[var][cell_idx(i0,j0,k0)];
    double c100 = blk.Q[var][cell_idx(i1,j0,k0)];
    double c010 = blk.Q[var][cell_idx(i0,j1,k0)];
    double c110 = blk.Q[var][cell_idx(i1,j1,k0)];
    double c001 = blk.Q[var][cell_idx(i0,j0,k1)];
    double c101 = blk.Q[var][cell_idx(i1,j0,k1)];
    double c011 = blk.Q[var][cell_idx(i0,j1,k1)];
    double c111 = blk.Q[var][cell_idx(i1,j1,k1)];

    double c00 = c000*(1-tx) + c100*tx;
    double c01 = c001*(1-tx) + c101*tx;
    double c10 = c010*(1-tx) + c110*tx;
    double c11 = c011*(1-tx) + c111*tx;
    double c0  = c00*(1-ty)  + c10*ty;
    double c1  = c01*(1-ty)  + c11*ty;
    return c0*(1-tz) + c1*tz;
}

// =============================================================================
// fill_ibm_ghosts
// =============================================================================
void fill_ibm_ghosts(CellBlock& blk, const LevelSet& ls,
                     const CellType* cell_type, const IBMConfig& cfg) noexcept {
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        const int idx = cell_idx(i,j,k);

        // Zero-out solid cells to avoid NaN propagation
        if (cell_type[idx] == CellType::SOLID) {
            for (int v = 0; v < NVAR; ++v) blk.Q[v][idx] = 0.0;
            continue;
        }
        if (cell_type[idx] != CellType::IBM_GHOST) continue;

        // Ghost cell centroid
        double gx = blk.ox + (i - NG + 0.5) * blk.h;
        double gy = blk.oy + (j - NG + 0.5) * blk.h;
        double gz = blk.oz + (k - NG + 0.5) * blk.h;

        // Signed distance and outward normal at ghost cell
        double phi_g = ls.phi(gx, gy, gz);
        double nx, ny, nz;
        ls.normal(gx, gy, gz, nx, ny, nz);

        // Image point: mirror of ghost across the surface
        // x_I = x_G − 2*φ_G * n  (moves inward by 2*dist to mirror inside solid)
        double ix = gx - 2.0 * phi_g * nx;
        double iy = gy - 2.0 * phi_g * ny;
        double iz = gz - 2.0 * phi_g * nz;

        // Interpolate conserved state at image point
        double rho_I = trilinear(blk, 0, ix, iy, iz);
        double ru_I  = trilinear(blk, 1, ix, iy, iz);
        double rv_I  = trilinear(blk, 2, ix, iy, iz);
        double rw_I  = trilinear(blk, 3, ix, iy, iz);
        double E_I   = trilinear(blk, 4, ix, iy, iz);

        // Guard: if image lands in solid, fall back to zero-gradient for momentum
        double rho_safe = (rho_I > 1e-10) ? rho_I : blk.Q[0][idx];

        // Reconstruct ghost velocity from BC
        // No-slip: u_ghost = 2*u_wall − u_image
        double u_I = ru_I / rho_safe;
        double v_I = rv_I / rho_safe;
        double w_I = rw_I / rho_safe;

        double u_g, v_g, w_g;
        switch (cfg.wall_bc) {
        case IBMWallBC::NoSlip:
            u_g = 2.0*cfg.u_wall - u_I;
            v_g = 2.0*cfg.v_wall - v_I;
            w_g = 2.0*cfg.w_wall - w_I;
            break;
        default:
            u_g = u_I; v_g = v_I; w_g = w_I;
            break;
        }

        // Density and temperature: zero-gradient (copy from image)
        double rho_g = rho_safe;
        double T_I   = (E_I - 0.5*rho_safe*(u_I*u_I+v_I*v_I+w_I*w_I))
                       / (rho_safe * (R_GAS / (GAMMA - 1.0)));

        double T_g;
        if (cfg.wall_bc == IBMWallBC::Isothermal)
            T_g = 2.0*cfg.T_wall - T_I;
        else
            T_g = T_I;  // adiabatic / no-slip: zero-gradient T

        T_g = std::max(T_g, 1.0);  // floor to avoid negative T

        // Write ghost cell conserved state
        double e_g = rho_g * R_GAS * T_g / (GAMMA - 1.0);
        blk.Q[0][idx] = rho_g;
        blk.Q[1][idx] = rho_g * u_g;
        blk.Q[2][idx] = rho_g * v_g;
        blk.Q[3][idx] = rho_g * w_g;
        blk.Q[4][idx] = e_g + 0.5*rho_g*(u_g*u_g+v_g*v_g+w_g*w_g);
    }
}

// =============================================================================
// apply_ibm: classify + fill in one call
// =============================================================================
void apply_ibm(CellBlock& blk, const LevelSet& ls,
               const IBMConfig& cfg, CellType* cell_type_scratch) noexcept {
    classify_ibm_cells(blk, ls, cell_type_scratch);
    fill_ibm_ghosts(blk, ls, cell_type_scratch, cfg);
}
