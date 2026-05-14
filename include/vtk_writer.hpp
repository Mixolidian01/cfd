#pragma once
// vtk_writer.hpp — write solver state to VTK legacy format (.vtk)
//
// Produces one structured-points file per leaf block, named:
//   {prefix}_step{NNNNNN}_blk{BBB}.vtk
//
// Fields written (cell-centred, interior only):
//   rho, u, v, w, p, T  (all as SCALARS DOUBLE)
//
// Usage:
//   vtk_write(solver, "output/frame");  // writes all leaf blocks

#include "ns_solver.hpp"
#include <string>

void vtk_write(const NSSolver& s, const std::string& prefix);
