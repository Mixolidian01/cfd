#pragma once
// vtk_writer.hpp — write solver state to VTK format
//
// Legacy ASCII VTK:
//   vtk_write(solver, prefix)  — one .vtk file per leaf block
//
// VTK XML binary (StructuredGrid, appended raw):
//   vtk_write_vts(path, step, blk_id, ox, oy, oz, h, d_Q)
//     — one .vts file for a single AMR leaf; d_Q is a DEVICE pointer
//       [GPU_NVAR × GPU_NCELL], component-major, conserved variables
//   vtk_write_pvts(path_prefix, step, n_blocks)
//     — .pvts collection referencing all per-block .vts files

#include <string>

// Forward-declare NSSolver so this header does not drag in ns_solver.hpp
// and the full CUDA header chain.  vtk_writer.cpp includes ns_solver.hpp directly.
struct NSSolver;

// Legacy ASCII writer (existing)
void vtk_write(const NSSolver& s, const std::string& prefix);

// VTK XML binary writer — implemented in vtk_writer.cu (nvcc-compiled)
// d_Q: device pointer [GPU_NVAR × GPU_NCELL], component-major, conserved vars
void vtk_write_vts(const std::string& path_prefix,
                   int step, int blk_id,
                   double ox, double oy, double oz, double h,
                   const double* d_Q);

void vtk_write_pvts(const std::string& path_prefix, int step, int n_blocks);
