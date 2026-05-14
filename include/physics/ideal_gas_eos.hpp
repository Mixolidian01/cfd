#pragma once
// Layer P — IdealGasEOS functor (CLAUDE.md R5)
// Wraps eos_cons_to_prim() as a stateless EquationOfState.
// __host__ __device__ compatible (delegates to inline function).

// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"  // Prim, eos_cons_to_prim
#include "schemes/concepts.hpp"    // EquationOfState concept

struct IdealGasEOS {
    double gamma = GAMMA;   // default: 1.4; override for other gases

    __host__ __device__
    Prim cons_to_prim(double rho, double rhou, double rhov,
                      double rhow, double en) const noexcept {
        return eos_cons_to_prim_g(rho, rhou, rhov, rhow, en, gamma);
    }
};

// Compile-time concept check (Layer C, CLAUDE.md R1/R5)
static_assert(EquationOfState<IdealGasEOS>);
