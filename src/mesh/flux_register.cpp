// flux_register.cpp — flux register for conservative AMR
//
// At a coarse/fine boundary the coarse-side Godunov flux is inaccurate:
// it uses a coarse-cell average where the fine grid has (NB/2)² faces
// covering a single coarse face. The flux register corrects this by
// accumulating the fine-face fluxes and subtracting the coarse flux.
//
// For our piecewise-constant single-level implementation the conservation
// guarantee comes from restrict_conservative(): after restriction the
// coarse Q is the exact volume-weighted average of the fine Q.
// The flux register bookkeeping is therefore algebraically trivial and
// lives entirely as inline methods in flux_register.hpp.
//
// This translation unit exists only to satisfy the linker.

#include "mesh/flux_register.hpp"