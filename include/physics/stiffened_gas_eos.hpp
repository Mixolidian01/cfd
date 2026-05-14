#pragma once
// Layer P — StiffenedGasEOS functor (R10-T3)
// Satisfies EquationOfState (include/schemes/concepts.hpp).
//
// Wraps the Allaire (2002) mixture rule from cell_block.hpp:
//   mix_eos(φ, γ_A, γ_B, p∞_A, p∞_B, γ_m, p∞_m)
//   eos_cons_to_prim_sg(ρ, ρu, ρv, ρw, E, γ_m, p∞_m)
//
// The concept-required single-field form (no φ) uses φ=1 (pure fluid A).

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"
#include "schemes/concepts.hpp"

struct StiffenedGasEOS {
    double gamma_a = 1.4;
    double gamma_b = 1.4;
    double p_inf_a = 0.0;
    double p_inf_b = 0.0;

    // Two-phase form: φ ∈ [0,1] selects the mixture EOS.
    __host__ __device__
    Prim cons_to_prim(double rho, double rhou, double rhov,
                      double rhow, double en, double phi) const noexcept {
        double gm, pim;
        mix_eos(phi, gamma_a, gamma_b, p_inf_a, p_inf_b, gm, pim);
        return eos_cons_to_prim_sg(rho, rhou, rhov, rhow, en, gm, pim);
    }

    // EquationOfState concept: single-component form (φ=1.0 → pure fluid A).
    __host__ __device__
    Prim cons_to_prim(double rho, double rhou, double rhov,
                      double rhow, double en) const noexcept {
        return cons_to_prim(rho, rhou, rhov, rhow, en, 1.0);
    }
};

static_assert(EquationOfState<StiffenedGasEOS>);
