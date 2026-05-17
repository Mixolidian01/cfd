#pragma once
// D6: P1 radiation transport (diffusion limit).
//
// Per-step solve: elliptic Helmholtz for mean radiation intensity G:
//   −∇·(D∇G) + κG = κ·a·T⁴        [D = c/(3κ)]
// GPU-resident CG (cuBLAS BLAS-1, matrix-free Helmholtz stencil).
//
// Energy coupling operator-split after SSP-RK3 stage:
//   ΔQ[4] += κ·(G − G_eq) · dt    [G_eq = c·a·T⁴]
//   G > G_eq: matter heats (absorbs radiation);  G < G_eq: matter cools (emits).
//
// Code-unit conventions:
//   T      — dimensionless temperature (T = (γ−1)·e_int/R_spec)
//   G      — radiation energy density × c  (∫I dΩ = G)
//   E_r    — G/c_light  (radiation energy density)
//   a_rad  — radiation constant: E_r_eq = a_rad·T⁴  →  G_eq = c·a_rad·T⁴
//   kappa  — absorption opacity (1/length in code units)
//   D      — c/(3κ)

#include "mesh/cell_block.hpp"  // GAMMA
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct RadiationParams {
    double kappa   = 1.0;   // absorption opacity  [1/length]
    double a_rad   = 1.0;   // radiation constant  [energy / (volume·T⁴)]
    double c_light = 1.0;   // speed of light in code units
    double R_spec  = 1.0;   // specific gas constant (same as ArrheniusParams)
};

// Radiation diffusion coefficient D = c / (3κ).
__host__ __device__ inline double p1_diffusion(const RadiationParams& p) {
    return p.c_light / (3.0 * p.kappa);
}

// Equilibrium radiation intensity: G_eq = c·a·T⁴.
__host__ __device__ inline double p1_emission(double T, const RadiationParams& p) {
    return p.c_light * p.a_rad * T * T * T * T;
}

// Temperature from conserved variables (identical to arrhenius_T).
__host__ __device__ inline double p1_T(
    double rho, double E, double ru, double rv, double rw, double R_spec)
{
    if (rho < 1e-12) return 0.0;
    double KE    = 0.5 * (ru*ru + rv*rv + rw*rw) / rho;
    double e_int = (E - KE) / rho;
    if (e_int <= 0.0) return 0.0;
    return (GAMMA - 1.0) * e_int / R_spec;
}
