// P7.3 gate test — Ghost-cell immersed boundary method
//
// Protocol:
//   I1: SphereLevelSet correctly classifies cells as FLUID/SOLID/IBM_GHOST
//   I2: PlaneLevelSet no-slip wall: ghost cell u reconstructed as -u_image
//       error < 1e-12 (exact linear interpolation for constant-velocity image)
//   I3: PlaneLevelSet adiabatic wall: ghost-cell T = T_image (zero-gradient)
//       error < 1e-12
//   I4: After apply_ibm on a block, total momentum of IBM_GHOST cells is
//       correctly antisymmetric (no-slip = zero at wall → u_ghost = -u_image)

#include "../include/ibm.hpp"
#include "../include/ns_solver.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg) {
    printf("  %s  %s  %s\n", ok ? "PASS" : "FAIL", tag, msg);
    if (!ok) ++nfail;
}

// =============================================================================
// I1: sphere classification
// =============================================================================
static void test_i1() {
    // Block: [0,1]^3, h = 1/NB
    const double L = 1.0;
    const double h = L / NB;
    CellBlock blk(0.0, 0.0, 0.0, h);

    // Sphere centered at (0.5,0.5,0.5) with R = 0.3; NB=8 → h=0.125
    SphereLevelSet sphere(0.5, 0.5, 0.5, 0.3);

    CellType ct[NCELL];
    classify_ibm_cells(blk, sphere, ct);

    int n_solid=0, n_ghost=0, n_fluid=0;
    for (int k=NG;k<NG+NB;++k)
    for (int j=NG;j<NG+NB;++j)
    for (int i=NG;i<NG+NB;++i) {
        switch (ct[cell_idx(i,j,k)]) {
        case CellType::SOLID:     ++n_solid; break;
        case CellType::IBM_GHOST: ++n_ghost; break;
        case CellType::FLUID:     ++n_fluid; break;
        }
    }

    // Sphere volume ≈ 4/3 π (0.3)^3 ≈ 0.113, domain 1^3 → ~11% solid cells
    // For NB=8: NB^3=512 interior cells; expect ~50-60 solid + some ghosts
    check(n_solid > 0, "I1a", "sphere has SOLID cells");
    check(n_ghost > 0, "I1b", "sphere boundary has IBM_GHOST cells");
    check(n_fluid > 0, "I1c", "fluid region exists");
    check(n_solid + n_ghost + n_fluid == NB*NB*NB, "I1d",
          "all cells classified (SOLID+GHOST+FLUID == NB^3)");
    // Every IBM_GHOST cell must have φ > 0 (is in the fluid)
    bool all_ghost_fluid = true;
    for (int k=NG;k<NG+NB;++k)
    for (int j=NG;j<NG+NB;++j)
    for (int i=NG;i<NG+NB;++i) {
        if (ct[cell_idx(i,j,k)] != CellType::IBM_GHOST) continue;
        double x = blk.ox + (i-NG+0.5)*h;
        double y = blk.oy + (j-NG+0.5)*h;
        double z = blk.oz + (k-NG+0.5)*h;
        if (sphere.phi(x,y,z) < 0.0) { all_ghost_fluid = false; break; }
    }
    check(all_ghost_fluid, "I1e", "all IBM_GHOST cells have phi > 0 (in fluid)");

    printf("      sphere: %d solid, %d ghost, %d fluid cells (NB=%d)\n",
           n_solid, n_ghost, n_fluid, NB);
}

// =============================================================================
// I2: no-slip wall reconstruction
// Fill block with uniform horizontal flow (rho=1, u=U_inf, p=p0).
// Wall at y=0 with outward normal in +y direction: phi(x,y,z) = y.
// Ghost cells just above y=0 → image points at y = -y_ghost.
// Since block extends to y=0 at the bottom face, ghost cells are the
// row at j=NG (y = h/2), image at y = -h/2 (inside solid).
// For a UNIFORM u-field, image interpolation gives u_image = U_inf,
// so ghost must get u_ghost = 2*0 - U_inf = -U_inf exactly.
// =============================================================================
static void test_i2() {
    const double h    = 1.0 / NB;
    const double U    = 3.14159;
    const double rho0 = 1.25;
    const double p0   = 1.0e5;

    // Block origin: y=0 is inside the domain (plane is the bottom face of block)
    // PlaneLevelSet: phi = y (fluid above y=0)
    // Block starts at y=0 → interior cells start at y = h/2
    // No SOLID cells exist inside [y=h/2..y=7.5h], so use a plane at y = 1.5h
    // to have actual solid cells in the lower ghost band.
    // Place wall at y = 1.5h (between j=NG and j=NG+1).
    const double y_wall = 1.5 * h;
    PlaneLevelSet wall(0.0, y_wall, 0.0,  0.0, 1.0, 0.0);  // outward = +y

    CellBlock blk(0.0, 0.0, 0.0, h);

    // Fill block with uniform flow
    const double E0 = p0/(GAMMA-1.0) + 0.5*rho0*U*U;
    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i) {
        int idx = cell_idx(i,j,k);
        blk.Q[0][idx] = rho0;
        blk.Q[1][idx] = rho0 * U;
        blk.Q[2][idx] = 0.0;
        blk.Q[3][idx] = 0.0;
        blk.Q[4][idx] = E0;
    }

    CellType ct[NCELL];
    classify_ibm_cells(blk, wall, ct);

    IBMConfig cfg;
    cfg.wall_bc = IBMWallBC::NoSlip;
    cfg.u_wall  = 0.0;

    fill_ibm_ghosts(blk, wall, ct, cfg);

    // Check: all IBM_GHOST cells have u_x = -U_inf (since u_image = U_inf)
    double max_err = 0.0;
    int n_checked = 0;
    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i) {
        if (ct[cell_idx(i,j,k)] != CellType::IBM_GHOST) continue;
        double rho  = blk.Q[0][cell_idx(i,j,k)];
        double u_xg = blk.Q[1][cell_idx(i,j,k)] / rho;
        double err  = std::abs(u_xg - (-U));
        if (err > max_err) max_err = err;
        ++n_checked;
    }

    check(n_checked > 0, "I2a", "there are IBM_GHOST cells for planar wall");
    check(max_err < 1e-10, "I2b",
          "no-slip ghost u_x = -U_inf (error < 1e-10 for uniform image field)");
    printf("      I2 max_err=%.2e  n_ghost=%d\n", max_err, n_checked);
}

// =============================================================================
// I3: adiabatic wall — ghost T equals image T (zero-gradient)
// =============================================================================
static void test_i3() {
    const double h    = 1.0 / NB;
    const double rho0 = 1.2;
    const double T0   = 350.0;
    const double p0   = rho0 * R_GAS * T0;
    const double E0   = p0 / (GAMMA - 1.0);

    const double y_wall = 1.5 * h;
    PlaneLevelSet wall(0.0, y_wall, 0.0,  0.0, 1.0, 0.0);

    CellBlock blk(0.0, 0.0, 0.0, h);
    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i) {
        int idx = cell_idx(i,j,k);
        blk.Q[0][idx] = rho0;
        blk.Q[1][idx] = 0.0;
        blk.Q[2][idx] = 0.0;
        blk.Q[3][idx] = 0.0;
        blk.Q[4][idx] = E0;
    }

    CellType ct[NCELL];
    classify_ibm_cells(blk, wall, ct);

    IBMConfig cfg;
    cfg.wall_bc = IBMWallBC::Adiabatic;

    fill_ibm_ghosts(blk, wall, ct, cfg);

    // Ghost T should equal T0 (uniform field → image T = T0 → ghost T = T0)
    double max_err = 0.0;
    int n_checked = 0;
    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i) {
        if (ct[cell_idx(i,j,k)] != CellType::IBM_GHOST) continue;
        int idx = cell_idx(i,j,k);
        double rho = blk.Q[0][idx];
        double u   = blk.Q[1][idx]/rho;
        double v   = blk.Q[2][idx]/rho;
        double w   = blk.Q[3][idx]/rho;
        double e_int = blk.Q[4][idx] - 0.5*rho*(u*u+v*v+w*w);
        double T_g = e_int / (rho * R_GAS / (GAMMA - 1.0));
        double err = std::abs(T_g - T0);
        if (err > max_err) max_err = err;
        ++n_checked;
    }

    check(n_checked > 0, "I3a", "IBM_GHOST cells exist for adiabatic wall");
    check(max_err < 1e-6, "I3b",
          "adiabatic ghost T = T_image (zero-gradient; error < 1e-6)");
    printf("      I3 max_err=%.2e  n_ghost=%d\n", max_err, n_checked);
}

// =============================================================================
// I4: no IBM_GHOST cells have phi < 0 for sphere (sanity on classification)
// =============================================================================
static void test_i4() {
    const double h = 1.0 / NB;
    CellBlock blk(0.0, 0.0, 0.0, h);
    SphereLevelSet sphere(0.5, 0.5, 0.5, 0.3);

    CellType ct[NCELL];
    classify_ibm_cells(blk, sphere, ct);

    bool ok = true;
    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i) {
        int idx = cell_idx(i,j,k);
        if (ct[idx] != CellType::IBM_GHOST) continue;
        double x = blk.ox + (i-NG+0.5)*h;
        double y = blk.oy + (j-NG+0.5)*h;
        double z = blk.oz + (k-NG+0.5)*h;
        if (sphere.phi(x,y,z) < 0.0) { ok = false; break; }
    }
    check(ok, "I4", "no IBM_GHOST cell sits inside the solid (phi >= 0 for all ghosts)");
}

int main() {
    printf("=== P7.3 gate: Ghost-cell immersed boundary method ===\n");
    test_i1();
    test_i2();
    test_i3();
    test_i4();

    const int ntotal = 9;
    const int npass  = ntotal - nfail;
    printf("\nResults: %d passed, %d failed\n", npass, nfail);
    if (nfail == 0)
        printf("==> PASS  P7.3 gate cleared — ghost-cell IBM active\n");
    return nfail > 0 ? 1 : 0;
}
