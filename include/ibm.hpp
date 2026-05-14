// ibm.hpp — Ghost-cell immersed boundary method (P7.3)
//
// Implements the direct-forcing ghost-cell IBM of Mittal & Iaccarino (2005).
// Cells are classified as FLUID, SOLID, or IBM_GHOST based on a signed-distance
// level-set function.  Before each RHS evaluation, fill_ibm_ghosts() reconstructs
// IBM ghost-cell state by:
//   1. Finding the image point I = G − 2φ_G·n  (mirror of ghost G across surface)
//   2. Trilinearly interpolating Q at I from surrounding fluid cells
//   3. Applying the wall BC: u_ghost = 2*u_wall − u_image, T/ρ from BC type
//
// References:
//   Mittal & Iaccarino (2005) Annu.Rev.Fluid Mech. 37:239–261
//   Balaras (2004) Comput.Fluids 33:375–404

#pragma once
#include "cell_block.hpp"
#include <cmath>
#include <array>
#include <vector>
#include <string>

// =============================================================================
// Cell classification
// =============================================================================
enum class CellType {
    FLUID    = 0,   // regular fluid cell
    SOLID    = 1,   // inside solid body (skipped in flux computation)
    IBM_GHOST = 2   // fluid cell adjacent to solid (needs IBM reconstruction)
};

// =============================================================================
// Abstract level-set interface
// =============================================================================
class LevelSet {
public:
    virtual ~LevelSet() = default;

    // Signed distance: φ > 0 in fluid, φ < 0 in solid, φ = 0 at surface.
    virtual double phi(double x, double y, double z) const noexcept = 0;

    // Outward surface normal at point (x,y,z) — points toward fluid.
    // Default: finite-difference gradient of phi (overridable for exact normals).
    virtual void normal(double x, double y, double z,
                        double& nx, double& ny, double& nz) const noexcept {
        const double eps = 1e-6;
        double gx = phi(x+eps,y,z) - phi(x-eps,y,z);
        double gy = phi(x,y+eps,z) - phi(x,y-eps,z);
        double gz = phi(x,y,z+eps) - phi(x,y,z-eps);
        double len = std::sqrt(gx*gx + gy*gy + gz*gz) + 1e-300;
        nx = gx/len; ny = gy/len; nz = gz/len;
    }
};

// =============================================================================
// Concrete level-set shapes
// =============================================================================

// Sphere: φ = |r − c| − R
class SphereLevelSet : public LevelSet {
public:
    SphereLevelSet(double cx, double cy, double cz, double R)
        : cx_(cx), cy_(cy), cz_(cz), R_(R) {}

    double phi(double x, double y, double z) const noexcept override {
        double dx=x-cx_, dy=y-cy_, dz=z-cz_;
        return std::sqrt(dx*dx+dy*dy+dz*dz) - R_;
    }
    void normal(double x, double y, double z,
                double& nx, double& ny, double& nz) const noexcept override {
        double dx=x-cx_, dy=y-cy_, dz=z-cz_;
        double r = std::sqrt(dx*dx+dy*dy+dz*dz) + 1e-300;
        nx=dx/r; ny=dy/r; nz=dz/r;
    }
private:
    double cx_, cy_, cz_, R_;
};

// Infinite cylinder along the z-axis: φ = sqrt((x-cx)²+(y-cy)²) − R
class CylinderZLevelSet : public LevelSet {
public:
    CylinderZLevelSet(double cx, double cy, double R)
        : cx_(cx), cy_(cy), R_(R) {}

    double phi(double x, double y, double /*z*/) const noexcept override {
        double dx=x-cx_, dy=y-cy_;
        return std::sqrt(dx*dx+dy*dy) - R_;
    }
    void normal(double x, double y, double z,
                double& nx, double& ny, double& nz) const noexcept override {
        double dx=x-cx_, dy=y-cy_;
        double r = std::sqrt(dx*dx+dy*dy) + 1e-300;
        nx=dx/r; ny=dy/r; nz=0.0;
        (void)z;
    }
private:
    double cx_, cy_, R_;
};

// Half-space: φ = dot(n, x − x0) (plane with outward normal n)
class PlaneLevelSet : public LevelSet {
public:
    PlaneLevelSet(double x0, double y0, double z0,
                  double nx, double ny, double nz)
        : x0_(x0), y0_(y0), z0_(z0), nx_(nx), ny_(ny), nz_(nz) {}

    double phi(double x, double y, double z) const noexcept override {
        return nx_*(x-x0_) + ny_*(y-y0_) + nz_*(z-z0_);
    }
    void normal(double /*x*/, double /*y*/, double /*z*/,
                double& nx, double& ny, double& nz) const noexcept override {
        nx=nx_; ny=ny_; nz=nz_;
    }
private:
    double x0_, y0_, z0_, nx_, ny_, nz_;
};

// =============================================================================
// Wall boundary condition type for IBM ghost-cell reconstruction
// =============================================================================
enum class IBMWallBC { NoSlip, Adiabatic, Isothermal };

struct IBMConfig {
    IBMWallBC wall_bc = IBMWallBC::NoSlip;
    double u_wall  = 0.0;   // wall velocity (x-component; extend if needed)
    double v_wall  = 0.0;
    double w_wall  = 0.0;
    double T_wall  = 300.0; // used when wall_bc == Isothermal
};

// =============================================================================
// Core IBM functions
// =============================================================================

// Classify each interior cell of blk as FLUID, SOLID, or IBM_GHOST.
// cell_type must have NCELL elements; only interior cells [NG..NG+NB-1]³ are set.
void classify_ibm_cells(const CellBlock& blk, const LevelSet& ls,
                        CellType* cell_type) noexcept;

// Fill IBM ghost cells (IBM_GHOST) by ghost-cell reconstruction.
// Modifies blk.Q in place.  cell_type array from classify_ibm_cells.
// Zero solid cells (SOLID): set conserved state to quiescent (avoids NaN).
void fill_ibm_ghosts(CellBlock& blk, const LevelSet& ls,
                     const CellType* cell_type, const IBMConfig& cfg) noexcept;

// Convenience: classify + fill in one call.
void apply_ibm(CellBlock& blk, const LevelSet& ls,
               const IBMConfig& cfg, CellType* cell_type_scratch) noexcept;
