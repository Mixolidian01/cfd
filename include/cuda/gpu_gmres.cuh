#pragma once
// D4: GPU-resident matrix-free GMRES for implicit Helmholtz viscous solve.
// Solves (I − α·∇²)u = rhs on the NB³ interior of a single block using
// GMRES(restart) with Jacobi preconditioning; BLAS-1 via cuBLAS.
//
// d_u / d_rhs: flat NB2³ device arrays indexed by cell_idx(i,j,k).
//   Interior cells: i,j,k ∈ [NG, NG+NB).
//   On entry d_u holds the initial guess (post-RK3 velocity); on exit the solution.
// alpha:   dt · μ/ρ  [m²]
// h:       uniform cell size
// bc:      ghost fill rule for the Helmholtz operator
// handle:  caller-owned cuBLAS handle

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>

enum class GmresBcType : uint8_t {
    Periodic,  // all six faces periodic
    WallY,     // no-slip (Dirichlet u=0) at ±y faces; periodic in x,z
};

struct GmresResult {
    int    iters;    // total iterations taken
    double rel_res;  // final ‖r‖/‖b‖
};

GmresResult gpu_helmholtz_gmres(
    double*        d_u,
    const double*  d_rhs,
    double         alpha,
    double         h,
    GmresBcType    bc,
    cublasHandle_t handle,
    int            max_iter = 50,
    double         tol      = 1e-8,
    int            restart  = 30,
    cudaStream_t   stream   = nullptr
);
