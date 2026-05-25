#pragma once
// D10: adjoint_rhs.hpp — block-level adjoint of convective RHS.
//
// Computes L*(Q) · lambda_rhs and accumulates into lambda_Q.
// L is the convective RHS (accumulate_face loop) with frozen-weight TENO7-A
// and frozen-lam HLLC-ES.
//
// Given lambda_rhs = ∂J/∂rhs_blk (seed from the output of compute_rhs),
// accumulates lambda_Q += L*(Q) · lambda_rhs  into lambda_Q.
//
// has_nbr: same 6-bit mask as compute_rhs (0 = all ghost-fill BCs / no MUSCL
// at block boundaries).

#include "mesh/cell_block.hpp"
#include <cstdint>

void adjoint_rhs(const CellBlock& Q_blk,
                 const CellBlock& lambda_rhs,
                 CellBlock&       lambda_Q,
                 uint8_t          has_nbr = 0) noexcept;
