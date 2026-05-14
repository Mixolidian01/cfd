// chemistry.hpp — Multi-species Arrhenius finite-rate chemistry (P7.4)
//
// Design: operator-split approach.  Hydrodynamics (NVAR=5) advances first;
// chemistry updates species mass fractions and internal energy in a separate
// sub-step.  Species Y_k are stored in a SpeciesBlock that parallels CellBlock.
//
// Reaction rate (modified Arrhenius):
//   k_f = A * T^n * exp(-Ea / (R_u * T))   [units consistent with [X]^order]
//   ω̇_k = sum_r (ν_k,r^'' - ν_k,r^') * W_k * k_f,r * prod_j [X_j]^ν_j,r^'
//   [X_j] = ρ Y_j / W_j   (molar concentration, kmol/m³)
//
// IMEX chemistry sub-step: implicit Euler with one Newton iteration.
//   Y^{n+1} - Y^n = dt * ω̇(Y^{n+1}, T^{n+1})
//   Linearise once around Y^n, solve the (N_spec+1)×(N_spec+1) linear system.
//
// References:
//   Poinsot & Veynante (2005) Theoretical and Numerical Combustion, 2nd ed.
//   Westbrook & Dryer (1981) Combust.Sci.Technol. 27:31–43

#pragma once
#include "mesh/cell_block.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <array>

static constexpr int MAX_SPEC = 16;   // max species per mechanism
static constexpr int MAX_REAC = 32;   // max reactions per mechanism
static constexpr double R_UNIV = 8314.46; // universal gas constant [J/(kmol·K)]

// =============================================================================
// Species data
// =============================================================================
struct SpeciesData {
    char   name[16];
    double W;    // molar mass [kg/kmol]
    double h_f0; // standard enthalpy of formation [J/kg] at T_ref=298.15 K
    double cp;   // constant-pressure specific heat [J/(kg·K)] (assumed constant)
};

// =============================================================================
// Single elementary reaction (irreversible for simplicity)
// =============================================================================
struct Reaction {
    double A;    // pre-exponential [SI, consistent units]
    double n;    // temperature exponent
    double Ea;   // activation energy [J/kmol]
    int    ns;   // number of species participating
    int    spec[MAX_SPEC];  // species indices
    double nu_r[MAX_SPEC];  // reactant stoichiometric coefficients
    double nu_p[MAX_SPEC];  // product stoichiometric coefficients
};

// =============================================================================
// Chemical mechanism
// =============================================================================
struct ChemMechanism {
    int         n_spec = 0;
    int         n_reac = 0;
    SpeciesData species[MAX_SPEC];
    Reaction    reactions[MAX_REAC];

    // Mixture cv at constant composition (ideal gas, cp = sum Y_k cp_k)
    double cv_mix(const double* Y) const noexcept {
        double cv = 0.0;
        for (int k = 0; k < n_spec; ++k)
            cv += Y[k] * (species[k].cp - R_UNIV / species[k].W);
        return cv;
    }

    // Mixture cp
    double cp_mix(const double* Y) const noexcept {
        double cp = 0.0;
        for (int k = 0; k < n_spec; ++k) cp += Y[k] * species[k].cp;
        return cp;
    }
};

// =============================================================================
// Per-block species storage (parallel to CellBlock)
// =============================================================================
struct SpeciesBlock {
    int     n_spec;
    // Y_k[k][cell_idx] — mass fractions (N_spec × NCELL)
    std::vector<std::vector<double>> Y;

    explicit SpeciesBlock(int ns) : n_spec(ns), Y(ns, std::vector<double>(NCELL, 0.0)) {}

    void set_uniform(const double* Y_vals) {
        for (int k = 0; k < n_spec; ++k)
            for (int idx = 0; idx < NCELL; ++idx)
                Y[k][idx] = Y_vals[k];
    }
};

// =============================================================================
// Core chemistry functions
// =============================================================================

// Arrhenius forward rate constant for reaction r at temperature T [K].
inline double arrhenius(const Reaction& r, double T) noexcept {
    return r.A * std::pow(T, r.n) * std::exp(-r.Ea / (R_UNIV * T));
}

// Compute chemistry source terms for a single cell.
//   Y[n_spec]   : mass fractions
//   T           : temperature [K]
//   rho         : density [kg/m³]
//   mech        : mechanism
//   dY[n_spec]  : output dY_k/dt  [1/s]
//   dT_dt       : output dT/dt     [K/s]
void chemistry_source(const double* Y, double T, double rho,
                      const ChemMechanism& mech,
                      double* dY, double* dT_dt) noexcept;

// Integrate chemistry for one cell over time interval dt using implicit Euler
// with one Newton iteration (sufficient for moderate stiffness).
// Modifies Y[n_spec] and T in place.
void chemistry_step_cell(double* Y, double& T, double rho, double dt,
                         const ChemMechanism& mech) noexcept;

// Apply chemistry operator to all interior cells of (blk, sblk).
// Updates sblk.Y and blk.Q[4] (total energy) after chemistry.
void apply_chemistry_block(CellBlock& blk, SpeciesBlock& sblk,
                           double dt, const ChemMechanism& mech) noexcept;

// =============================================================================
// Convenience: build common mechanisms
// =============================================================================

// 2-species irreversible A → B (test mechanism)
// A: molar mass W_A, formation enthalpy h_A; B: W_B, h_B
// Rate: k = A_rate * exp(-Ea/(R_u*T)) * [A]
ChemMechanism make_simple_mechanism(
    double W_A, double h_fA, double cp_A,
    double W_B, double h_fB, double cp_B,
    double A_rate, double Ea,
    const char* name_A = "A", const char* name_B = "B");

// H2 / O2 / H2O one-step (Westbrook & Dryer 1981)
ChemMechanism make_h2_o2_mechanism();
