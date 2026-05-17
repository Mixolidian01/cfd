#pragma once
// D5: Single-step Arrhenius reactive flow functor (host+device).
//
// Source term (per unit volume):
//   ω  = A · ρ · Y · exp(−T_act / T)       [reaction rate]
//   dY/dt = −ω / ρ                          [species depletion]
//   dE/dt = +q_heat · ω                     [heat release → total energy]
//
// Temperature from ideal-gas EOS:
//   T = (γ−1) · e_int / R_spec   where e_int = (E − KE) / ρ
//
// Stability note: for detonation the chemistry is stiff (large A·ρ·τ_fluid).
// Use n_sub > 1 to subcycle RK4 within one fluid dt:
//   dt_sub = dt / n_sub  must satisfy  (A·ρ·exp(−T_act/T)) · dt_sub ≤ 2.8

#include "mesh/cell_block.hpp"  // GAMMA
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct ArrheniusParams {
    double A      = 1e4;   // pre-exponential factor
    double T_act  = 10.0;  // activation temperature in code units
    double q_heat = 10.0;  // heat release per unit mass of reactant (code units)
    double R_spec = 1.0;   // specific gas constant (code units)
    int    n_sub  = 8;     // chemistry RK4 substeps per fluid step
};

// Temperature from conserved variables.
__host__ __device__ inline double arrhenius_T(
    double rho, double E, double ru, double rv, double rw, double R_spec)
{
    if (rho < 1e-12) return 0.0;
    double KE    = 0.5 * (ru*ru + rv*rv + rw*rw) / rho;
    double e_int = (E - KE) / rho;
    if (e_int <= 0.0) return 0.0;
    return (GAMMA - 1.0) * e_int / R_spec;
}

// Reaction rate per unit volume.
__host__ __device__ inline double arrhenius_omega(
    double rho, double Y, double T, const ArrheniusParams& p)
{
    return (Y > 0.0 && T > 0.0) ? p.A * rho * Y * std::exp(-p.T_act / T) : 0.0;
}
