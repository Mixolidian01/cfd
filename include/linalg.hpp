#pragma once
// DESIGN.md reference: Layer 0 — Linear Algebra
// No knowledge of grids, physics, or time integration.
// All dot products accumulate in FP64 with Kahan compensation.
#include <vector>
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// CSR matrix  (host-side, FP64)
// ─────────────────────────────────────────────────────────────────────────────
struct CSRMat {
    int n = 0;
    std::vector<int>    rowptr;   // size n+1
    std::vector<int>    colidx;   // size nnz
    std::vector<double> values;   // size nnz

    int  nnz()   const { return (int)values.size(); }
    bool empty() const { return n == 0; }

    // y = A x  (Kahan accumulation per row)
    void spmv(const std::vector<double>& x,
              std::vector<double>&       y) const;

    // Build from COO; duplicate (row,col) entries are summed.
    static CSRMat from_coo(int n,
                           const std::vector<int>&    rows,
                           const std::vector<int>&    cols,
                           const std::vector<double>& vals);
};

// ─────────────────────────────────────────────────────────────────────────────
// BLAS-1  (FP64, Kahan where noted)
// ─────────────────────────────────────────────────────────────────────────────
double dot_kahan (const std::vector<double>& a,
                  const std::vector<double>& b);   // Kahan compensated
double norm2     (const std::vector<double>& a);   // sqrt(dot(a,a))
void   axpy      (double alpha,
                  const std::vector<double>& x,
                  std::vector<double>&       y);   // y += alpha*x
void   scale     (double alpha, std::vector<double>& x);
void   copy_vec  (const std::vector<double>& src,
                  std::vector<double>&       dst);
void   zero_vec  (std::vector<double>& x);

// ─────────────────────────────────────────────────────────────────────────────
// Conjugate Gradient
// Solves A x = b  (A must be SPD).
// x is the initial guess on entry, solution on exit.
// Returns iteration count.  tol is relative: ||r||/||b|| < tol.
// ─────────────────────────────────────────────────────────────────────────────
int cg_solve(const CSRMat&              A,
             const std::vector<double>& b,
             std::vector<double>&       x,
             int    max_iter = 10000,
             double tol      = 1e-10);

// ─────────────────────────────────────────────────────────────────────────────
// Geometric Multigrid  (3-level V-cycle, Neumann BCs)
//
// Operator: standard 7-point Laplacian on NxNxN cell-centered grid.
//   (Lu)_ijk = sum_{6 neighbours} u_nb - nb_count * u_ijk) / h^2
// Boundary treatment: skip-boundary (Neumann, du/dn = 0).
// Null space: constant — mean is subtracted before solve and after each sweep.
//
// Sign convention (matches DESIGN.md pressure Poisson):
//   Solves  A u = f  where A = +Laplacian (positive semi-definite).
//   For the pressure equation  -∇²p = rhs, pass f = -rhs.
// ─────────────────────────────────────────────────────────────────────────────
struct MGLevel {
    int    nx, ny, nz;
    double hx, hy, hz;
    std::vector<double> u, r, f;
};

struct MGSolver {
    std::vector<MGLevel> levels;   // levels[0] = finest

    // Build 3-level hierarchy for a single NxNxN block (N must be divisible by 4).
    void build(int N, double h);

    // Single V-cycle.  u is updated in-place.
    void vcycle(std::vector<double>&       u,
                const std::vector<double>& f,
                int n_pre  = 2,
                int n_post = 2);

    // Full solve: iterate V-cycles until relative residual < tol.
    // Returns number of cycles taken.
    int solve(std::vector<double>&       u,
              const std::vector<double>& f,
              int    max_cycles = 100,
              double tol        = 1e-10);

    // Exposed for testing.
    double residual_norm  (const MGLevel& lv) const;
    void   apply_laplacian(const MGLevel& lv,
                           const std::vector<double>& u,
                           std::vector<double>&       Lu) const;

private:
    void smooth_rb   (MGLevel& lv, int n_sweeps, bool zero_mean);
    void restrict3   (const MGLevel& fine,   MGLevel& coarse);
    void prolongate3 (const MGLevel& coarse, MGLevel& fine);
};

// ─────────────────────────────────────────────────────────────────────────────
// SSP-RK3 coefficients  (Shu-Osher, 1988)
// Stage s update:  Q_s = alpha[s]*Q^n + beta[s]*(Q_{s-1} + dt*L(Q_{s-1}))
// ─────────────────────────────────────────────────────────────────────────────
struct RK3Coeffs {
    static constexpr double alpha[3] = {1.0,  3.0/4.0, 1.0/3.0};
    static constexpr double beta [3] = {1.0,  1.0/4.0, 2.0/3.0};
};
