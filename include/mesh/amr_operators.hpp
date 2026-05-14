#pragma once
#include "mesh/cell_block.hpp"
// Conservative prolongation: fill fine child block Q from parent coarse block.
// Uses piecewise-constant (zeroth-order) — conservative by construction.
// child_octant: 0..7 (which child of the parent this block is)
void prolong_conservative(const CellBlock& coarse, CellBlock& fine, int child_octant);

// Conservative restriction: average 8 fine children into one coarse block.
// Volume-weighted: Q_coarse[v][I] = (1/8) * sum_children Q_fine[v][i]
void restrict_conservative(CellBlock& coarse, const CellBlock* children[8]);

// Ghost fill at coarse/fine interface: fill fine ghost cells from coarse interior.
// axis=0,1,2  side=0(lo),1(hi)
void fill_cf_ghosts(CellBlock& fine, const CellBlock& coarse,
                    int child_octant, int axis, int side);

// Refinement criterion: returns true if block should be refined.
// Criterion: max |grad(rho)| * h > threshold
bool should_refine(const CellBlock& blk, double h, double threshold = 0.05);

// Coarsening criterion: returns true if block can be coarsened.
bool should_coarsen(const CellBlock& blk, double h, double threshold = 0.01);
