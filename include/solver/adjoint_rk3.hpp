#pragma once
// D11-adj: adjoint of one SSP-RK3 step.
//
// adjoint_rk3_one_block: reverse the three Shu-Osher stages for a single leaf block.
//   Qs0: Qn with valid ghosts (post-ghost-fill before stage 1 rhs_call)
//   Qs1: Q^(1) with valid ghosts (post-ghost-fill before stage 2 rhs_call)
//   Qs2: Q^(2) with valid ghosts (post-ghost-fill before stage 3 rhs_call)
//   dt: the same dt used in the forward pass
//   lam_f: ∂J/∂Q^(n+1), interior cells only (ghost zone = 0)
//   lam_Qn: ACCUMULATES ∂J/∂Q^n (zero before call)

#include "mesh/cell_block.hpp"

void adjoint_rk3_one_block(
    const CellBlock& Qs0,
    const CellBlock& Qs1,
    const CellBlock& Qs2,
    double dt,
    const CellBlock& lam_f,
    CellBlock& lam_Qn) noexcept;
