// D4 gate — GPU GMRES Helmholtz solver.
//
// G41: GMRES correctly inverts (I − α∇²) on a manufactured Helmholtz problem
//      (Periodic BC, known exact solution).  Rel-error < 1e-8.
// G42: Poiseuille channel flow — IMEX-Euler (body force + implicit Helmholtz)
//      converges to u(y) = F/(2µ)·y·(L−y) within 1e-6 (WallY BC).
// G43: GMRES iteration count ≤ 50 in both G41 and G42.

#include "cuda/gpu_gmres.cuh"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static int nfail = 0;

static void check(bool ok, const char* tag, const char* msg, double val = -1.0) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else {
        if (val >= 0.0) printf("  FAIL  %s  %s  (val = %.3e)\n", tag, msg, val);
        else            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// Upload a NB3 host array into the interior of a NB23 device array (ghosts = 0).
static void upload_interior(double* d_NB23, const double* h_NB3) {
    std::vector<double> h_full(NCELL, 0.0);
    for (int k = 0; k < NB; ++k)
    for (int j = 0; j < NB; ++j)
    for (int i = 0; i < NB; ++i) {
        int flat3  = i + j * NB + k * NB * NB;
        h_full[cell_idx(i + NG, j + NG, k + NG)] = h_NB3[flat3];
    }
    cudaMemcpy(d_NB23, h_full.data(), NCELL * sizeof(double), cudaMemcpyHostToDevice);
}

// Download the interior of a NB23 device array into a NB3 host array.
static void download_interior(double* h_NB3, const double* d_NB23) {
    std::vector<double> h_full(NCELL);
    cudaMemcpy(h_full.data(), d_NB23, NCELL * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < NB; ++k)
    for (int j = 0; j < NB; ++j)
    for (int i = 0; i < NB; ++i) {
        int flat3 = i + j * NB + k * NB * NB;
        h_NB3[flat3] = h_full[cell_idx(i + NG, j + NG, k + NG)];
    }
}

// =============================================================================
// G41: manufactured Helmholtz test (Periodic BC)
// u_exact[i,j,k] = sin(2π·i/NB) — single x-Fourier mode, constant in j,k.
// f = A·u_exact computed analytically; solve GMRES → verify recovery.
// =============================================================================
static void test_g41(cublasHandle_t handle) {
    printf("\n-- G41  GMRES inverts manufactured Helmholtz (Periodic, 1e-8) --\n");

    const double h     = 1.0 / NB;
    const double alpha = 0.05;
    const double aoh2  = alpha / (h * h);

    // u_exact on NB³ interior grid
    std::vector<double> u_exact(NB * NB * NB);
    for (int k = 0; k < NB; ++k)
    for (int j = 0; j < NB; ++j)
    for (int i = 0; i < NB; ++i)
        u_exact[i + j * NB + k * NB * NB] =
            std::sin(2.0 * M_PI * i / NB);

    // For u = sin(2π·i/NB), constant in j,k:
    // Lap(u)[i] = (u[i+1] + u[i-1] − 2u[i]) / h²   (periodic x; y,z cancel)
    //           = 2(cos(2π/NB) − 1) / h² · u[i]
    // So f = u·(1 − α · 2(cos(2π/NB)−1)/h²) = u · λ
    const double lambda = 1.0 + 2.0 * aoh2 * (1.0 - std::cos(2.0 * M_PI / NB));

    std::vector<double> f_exact(NB * NB * NB);
    for (int n = 0; n < NB * NB * NB; ++n) f_exact[n] = lambda * u_exact[n];

    // Upload f as RHS; zero initial guess in d_u
    double *d_u, *d_rhs;
    cudaMalloc(&d_u,   NCELL * sizeof(double));
    cudaMalloc(&d_rhs, NCELL * sizeof(double));
    cudaMemset(d_u, 0, NCELL * sizeof(double));
    upload_interior(d_rhs, f_exact.data());

    GmresResult res = gpu_helmholtz_gmres(
        d_u, d_rhs, alpha, h, GmresBcType::Periodic, handle,
        /*max_iter=*/50, /*tol=*/1e-8, /*restart=*/30);

    printf("   G41: iters=%d  rel_res=%.3e\n", res.iters, res.rel_res);

    // Download solution and compare to u_exact
    std::vector<double> h_u(NB * NB * NB);
    download_interior(h_u.data(), d_u);

    double err = 0.0;
    double ref = 0.0;
    for (int n = 0; n < NB * NB * NB; ++n) {
        err = std::fmax(err, std::fabs(h_u[n] - u_exact[n]));
        ref = std::fmax(ref, std::fabs(u_exact[n]));
    }
    double rel_err = (ref > 0.0) ? err / ref : err;
    printf("   G41: max rel err = %.3e  (tol 1e-8)\n", rel_err);

    check(rel_err < 1e-8, "G41", "GMRES inverts Helmholtz to 1e-8 (Periodic)", rel_err);
    check(res.iters <= 50, "G43a", "GMRES iters ≤ 50 (G41 case)", (double)res.iters);

    cudaFree(d_u); cudaFree(d_rhs);
}

// =============================================================================
// G42 / G43: Poiseuille channel flow, WallY BC.
// IMEX-Euler: explicit body force + implicit Helmholtz on u-velocity.
// dt=100, 50 steps → converges to steady state parabola within 1e-6.
// =============================================================================
// Body-force kernel: add F*dt/rho to interior cells of d_u (x-velocity only).
__global__ static void k_apply_body_force(double* __restrict__ d_u, double increment) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NCELL) return;
    int i = tid % NB2, jk = tid / NB2, j = jk % NB2, k = jk / NB2;
    if (i < NG || i >= NG+NB || j < NG || j >= NG+NB || k < NG || k >= NG+NB) return;
    d_u[tid] += increment;
}

static void test_g42_g43(cublasHandle_t handle) {
    printf("\n-- G42  Poiseuille channel flow (WallY, IMEX, dt=100, 50 steps) --\n");

    const double mu  = 1e-3;
    const double rho = 1.2;
    const double L   = 1.0;          // channel width
    const double h   = L / NB;       // = 0.125

    // Re = rho * u_max * (L/2) / mu = 100 → u_max
    const double u_max = 100.0 * mu / (rho * (L / 2.0));   // ≈ 0.16667
    // Parabola u(y) = F/(2µ)·y·(L−y); max at y=L/2 → F = 8µu_max/L²
    const double F   = 8.0 * mu * u_max / (L * L);          // ≈ 1.3333e-3

    const double dt    = 100.0;
    const double alpha = dt * mu / rho;
    const int    nstep = 50;

    // d_u: NB23 array for x-velocity (u); starts at 0.
    double *d_u, *d_rhs;
    cudaMalloc(&d_u,   NCELL * sizeof(double));
    cudaMalloc(&d_rhs, NCELL * sizeof(double));
    cudaMemset(d_u, 0, NCELL * sizeof(double));

    int max_iters = 0;

    for (int step = 0; step < nstep; ++step) {
        // u* = u^n + (F/rho)*dt  (body force acceleration × dt)
        k_apply_body_force<<<(NCELL+255)/256, 256>>>(d_u, (F / rho) * dt);

        // RHS for Helmholtz = u* (copy d_u → d_rhs then solve in-place)
        cudaMemcpy(d_rhs, d_u, NCELL * sizeof(double), cudaMemcpyDeviceToDevice);

        GmresResult res = gpu_helmholtz_gmres(
            d_u, d_rhs, alpha, h, GmresBcType::WallY, handle,
            /*max_iter=*/50, /*tol=*/1e-8, /*restart=*/30);

        max_iters = std::max(max_iters, res.iters);
    }

    // Download u-velocity from interior
    std::vector<double> h_u(NB * NB * NB);
    download_interior(h_u.data(), d_u);

    // Compare profile at (i=0, k=0) row to the discrete steady state.
    // The ghost-cell Dirichlet BC introduces a constant O(h²) shift relative
    // to the continuous profile: u_ss[j] = F/(2µ)·y_j·(L−y_j) + F·h²/(8µ).
    // (Derived by solving the ghost-cell 1D discrete Poisson exactly.)
    // y_j = (j + 0.5) * h  for j = 0..NB−1 (interior index 0-based)
    const double disc_corr = F * h * h / (8.0 * mu);
    double err = 0.0;
    for (int j = 0; j < NB; ++j) {
        double y_j      = (j + 0.5) * h;
        double u_analyt = (F / (2.0 * mu)) * y_j * (L - y_j) + disc_corr;
        double u_gpu    = h_u[0 + j * NB + 0 * NB * NB];  // i=0, j, k=0
        err = std::fmax(err, std::fabs(u_gpu - u_analyt));
    }
    printf("   G42: max err vs analytical = %.3e  (tol 1e-6)\n", err);
    printf("   G43: max GMRES iters over %d steps = %d\n", nstep, max_iters);

    check(err       < 1e-6,  "G42", "Poiseuille steady state matches analytical (tol 1e-6)", err);
    check(max_iters <= 50,   "G43", "GMRES iters ≤ 50 (all Poiseuille steps)", (double)max_iters);

    cudaFree(d_u); cudaFree(d_rhs);
}

// =============================================================================
int main() {
    printf("=== D4 GPU GMRES Helmholtz gate test ===\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    test_g41(handle);
    test_g42_g43(handle);

    cublasDestroy(handle);

    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
