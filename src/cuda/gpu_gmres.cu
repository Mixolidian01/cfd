// D4: GPU-resident GMRES — Helmholtz viscous solve.
//
// Solves (I − α·∇²)u = rhs on the NB³ interior grid (512 unknowns per
// velocity component per leaf block).
//
// Algorithm: GMRES(restart) with Jacobi (point-diagonal) preconditioning.
//   - Krylov vectors and BLAS-1 ops live entirely on device.
//   - cuBLAS CUBLAS_POINTER_MODE_HOST for dot/nrm2 results (small N → ok).
//   - Hessenberg matrix and Givens rotations on host (at most restart+1 rows).
//
// Ghost fill for the matrix-free apply:
//   Periodic:  ghost = opposite interior face (all six directions).
//   WallY:     ghost_y = −u_mirror  (Dirichlet u=0 at face);
//              ghost_x, ghost_z periodic.
//   Corner/edge ghosts always map to interior cells → no data dependency.
//
// Diagonal of (I − α·∇²):
//   Interior cell with no Dirichlet neighbour: 1 + 6·α/h²
//   Y-boundary interior cell (WallY, 1 Dirichlet ghost):
//     ghost contributes −u → extra α/h² on diagonal → 1 + 7·α/h²

#include "cuda/gpu_gmres.cuh"
#include "mesh/cell_block.hpp"  // NB, NG, NB2, NCELL, cell_idx
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ── compact (NB3) ↔ ghost (NB23) copy kernels ────────────────────────────────
// NB3 flat ordering: tid = i + j*NB + k*NB² (i fastest; i,j,k ∈ [0,NB))
// NB23 flat = cell_idx(i+NG, j+NG, k+NG)

static constexpr int NB3 = NB * NB * NB;

__global__ static void k_put_interior(double* __restrict__ d23,
                                       const double* __restrict__ d3) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    d23[cell_idx(i + NG, j + NG, k + NG)] = d3[tid];
}

__global__ static void k_get_interior(double* __restrict__ d3,
                                       const double* __restrict__ d23) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    d3[tid] = d23[cell_idx(i + NG, j + NG, k + NG)];
}

// ── ghost fill ────────────────────────────────────────────────────────────────
// Every ghost cell maps to an INTERIOR source → no data dependency between threads.
__global__ static void k_fill_ghosts_helm(double* __restrict__ d, int bc_int) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NCELL) return;

    int i = tid % NB2, jk = tid / NB2, j = jk % NB2, k = jk / NB2;

    bool gx = (i < NG || i >= NG + NB);
    bool gy = (j < NG || j >= NG + NB);
    bool gz = (k < NG || k >= NG + NB);
    if (!gx && !gy && !gz) return;  // interior — skip

    int si = i, sj = j, sk = k;
    bool neg = false;

    // x: always periodic
    if (i < NG)        si = i + NB;
    else if (i >= NG + NB) si = i - NB;

    // y: WallY=Dirichlet(u=0) | else periodic
    if (j < NG) {
        if (bc_int == 1) { sj = 2 * NG - 1 - j;           neg = !neg; }
        else             { sj = j + NB; }
    } else if (j >= NG + NB) {
        if (bc_int == 1) { sj = 2 * (NG + NB) - 1 - j;    neg = !neg; }
        else             { sj = j - NB; }
    }

    // z: always periodic
    if (k < NG)        sk = k + NB;
    else if (k >= NG + NB) sk = k - NB;

    d[tid] = neg ? -d[cell_idx(si, sj, sk)] : d[cell_idx(si, sj, sk)];
}

// ── (I − α/h²·Lap) stencil: NB23 (with filled ghosts) → NB3 compact ─────────
__global__ static void k_helm_stencil(double* __restrict__ out3,
                                       const double* __restrict__ in23,
                                       double aoh2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    int ig = i + NG, jg = j + NG, kg = k + NG;
    double v = in23[cell_idx(ig, jg, kg)];
    double lap = (in23[cell_idx(ig+1,jg,  kg  )] +
                  in23[cell_idx(ig-1,jg,  kg  )] +
                  in23[cell_idx(ig,  jg+1,kg  )] +
                  in23[cell_idx(ig,  jg-1,kg  )] +
                  in23[cell_idx(ig,  jg,  kg+1)] +
                  in23[cell_idx(ig,  jg,  kg-1)] - 6.0 * v);
    out3[tid] = v - aoh2 * lap;
}

// ── Jacobi preconditioner diagonal inverse ────────────────────────────────────
// Interior cells adjacent to a Dirichlet y-face have diag 1 + 7·aoh2.
__global__ static void k_compute_diag_inv(double* __restrict__ d, double aoh2, int bc_int) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int j = (tid / NB) % NB;
    double diag = 1.0 + 6.0 * aoh2;
    if (bc_int == 1 && (j == 0 || j == NB - 1)) diag += aoh2;
    d[tid] = 1.0 / diag;
}

// ── pointwise kernels for GMRES vectors ──────────────────────────────────────
__global__ static void k_diag_precond(double* __restrict__ out,
                                       const double* __restrict__ in,
                                       const double* __restrict__ dinv, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) out[t] = in[t] * dinv[t];
}

__global__ static void k_axpby(double* __restrict__ z, double a,
                                 const double* __restrict__ x,
                                 double b, const double* __restrict__ y, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) z[t] = a * x[t] + b * y[t];
}

// ── matrix-free Helmholtz apply: compact NB3 → compact NB3 ───────────────────
static void helm_apply(double* out3, const double* in3,
                        double* scratch23, double aoh2, int bc_int,
                        cudaStream_t stream) {
    const int nb3  = (NB3   + 255) / 256;
    const int nb23 = (NCELL + 255) / 256;
    k_put_interior    <<<nb3,  256, 0, stream>>>(scratch23, in3);
    k_fill_ghosts_helm<<<nb23, 256, 0, stream>>>(scratch23, bc_int);
    k_helm_stencil    <<<nb3,  256, 0, stream>>>(out3, scratch23, aoh2);
}

// ── GMRES(restart) with Jacobi preconditioning ────────────────────────────────
GmresResult gpu_helmholtz_gmres(
    double*        d_u,
    const double*  d_rhs,
    double         alpha,
    double         h,
    GmresBcType    bc,
    cublasHandle_t handle,
    int            max_iter,
    double         tol,
    int            restart,
    cudaStream_t   stream
) {
    const double aoh2   = alpha / (h * h);
    const int    bc_int = (bc == GmresBcType::WallY) ? 1 : 0;
    const int    m      = std::min(restart, max_iter);
    const int    nb3    = (NB3 + 255) / 256;

    // Device allocations
    double *d_x, *d_b, *d_r, *d_w, *d_tmp, *d_diag, *d_scr;
    double *d_V;  // Krylov basis: (m+1) × NB3
    cudaMalloc(&d_x,   NB3 * sizeof(double));
    cudaMalloc(&d_b,   NB3 * sizeof(double));
    cudaMalloc(&d_r,   NB3 * sizeof(double));
    cudaMalloc(&d_w,   NB3 * sizeof(double));
    cudaMalloc(&d_tmp, NB3 * sizeof(double));
    cudaMalloc(&d_diag,NB3 * sizeof(double));
    cudaMalloc(&d_scr, NCELL * sizeof(double));
    cudaMalloc(&d_V,   (m + 1) * NB3 * sizeof(double));

    cublasSetStream(handle, stream);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    // Extract initial guess and RHS from NB23 layout
    k_get_interior<<<nb3, 256, 0, stream>>>(d_x, d_u);
    k_get_interior<<<nb3, 256, 0, stream>>>(d_b, d_rhs);
    k_compute_diag_inv<<<nb3, 256, 0, stream>>>(d_diag, aoh2, bc_int);

    // beta0 = ‖M⁻¹b‖  (left-preconditioned reference norm)
    k_diag_precond<<<nb3, 256, 0, stream>>>(d_w, d_b, d_diag, NB3);
    double beta0 = 0.0;
    cublasDnrm2(handle, NB3, d_w, 1, &beta0);
    if (beta0 < 1e-20) {
        cudaMemsetAsync(d_u, 0, NCELL * sizeof(double), stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_w);
        cudaFree(d_tmp); cudaFree(d_diag); cudaFree(d_scr); cudaFree(d_V);
        return {0, 0.0};
    }

    // Hessenberg (col-major): H[row + col*MX]
    constexpr int MX = 52;
    double H[MX * MX]; std::memset(H, 0, sizeof(H));
    double g[MX];       std::memset(g, 0, sizeof(g));
    double cs[MX], sn[MX];

    int    total_iters = 0;
    double rel_res     = 1.0;
    const double neg1  = -1.0;

    while (total_iters < max_iter) {
        // r = b - A*x
        helm_apply(d_tmp, d_x, d_scr, aoh2, bc_int, stream);
        cudaMemcpyAsync(d_r, d_b, NB3 * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
        cublasDaxpy(handle, NB3, &neg1, d_tmp, 1, d_r, 1);

        // v0 = M⁻¹r / ‖M⁻¹r‖  (left-preconditioned GMRES)
        k_diag_precond<<<nb3, 256, 0, stream>>>(d_w, d_r, d_diag, NB3);
        double beta = 0.0;
        cublasDnrm2(handle, NB3, d_w, 1, &beta);
        rel_res = beta / beta0;
        if (rel_res < tol) break;

        double inv_beta = 1.0 / beta;
        cudaMemcpyAsync(d_V, d_w, NB3 * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
        cublasDscal(handle, NB3, &inv_beta, d_V, 1);

        std::memset(g,  0, sizeof(g));
        std::memset(cs, 0, sizeof(cs));
        std::memset(sn, 0, sizeof(sn));
        g[0] = beta;

        int j = 0;
        for (; j < m && total_iters < max_iter; ++j, ++total_iters) {
            double* d_vj  = d_V +  j      * NB3;
            double* d_vj1 = d_V + (j + 1) * NB3;

            // w = M⁻¹ · A · v_j
            helm_apply(d_tmp, d_vj, d_scr, aoh2, bc_int, stream);
            k_diag_precond<<<nb3, 256, 0, stream>>>(d_w, d_tmp, d_diag, NB3);

            // Modified Gram-Schmidt
            for (int i = 0; i <= j; ++i) {
                double h_ij = 0.0;
                cublasDdot(handle, NB3, d_w, 1, d_V + i * NB3, 1, &h_ij);
                H[i + j * MX] = h_ij;
                double neg_h = -h_ij;
                cublasDaxpy(handle, NB3, &neg_h, d_V + i * NB3, 1, d_w, 1);
            }

            double h_j1j = 0.0;
            cublasDnrm2(handle, NB3, d_w, 1, &h_j1j);
            H[(j + 1) + j * MX] = h_j1j;

            if (h_j1j > 1e-15) {
                double inv_h = 1.0 / h_j1j;
                cudaMemcpyAsync(d_vj1, d_w, NB3 * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
                cublasDscal(handle, NB3, &inv_h, d_vj1, 1);
            }

            // Apply previous Givens rotations to column j
            for (int i = 0; i < j; ++i) {
                double t1 =  cs[i] * H[i     + j * MX] + sn[i] * H[(i+1) + j * MX];
                double t2 = -sn[i] * H[i     + j * MX] + cs[i] * H[(i+1) + j * MX];
                H[i     + j * MX] = t1;
                H[(i+1) + j * MX] = t2;
            }

            // New Givens rotation for entries (j,j) and (j+1,j)
            double a = H[j + j * MX], b_val = H[(j+1) + j * MX];
            double r_val = std::hypot(a, b_val);
            if (r_val > 1e-15) { cs[j] = a / r_val; sn[j] = b_val / r_val; }
            else               { cs[j] = 1.0;        sn[j] = 0.0; }
            H[j     + j * MX] = r_val;
            H[(j+1) + j * MX] = 0.0;
            g[j+1] = -sn[j] * g[j];
            g[j]   =  cs[j] * g[j];

            rel_res = std::fabs(g[j+1]) / beta0;
            if (rel_res < tol) { ++j; break; }
            if (h_j1j <= 1e-15) { ++j; break; }  // lucky breakdown
        }

        // Back-solve R·y = g[0..j-1]
        double y[MX]; std::memset(y, 0, sizeof(y));
        for (int i = j - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int kk = i + 1; kk < j; ++kk)
                y[i] -= H[i + kk * MX] * y[kk];
            if (std::fabs(H[i + i * MX]) > 1e-15)
                y[i] /= H[i + i * MX];
        }

        // x = x + V·y
        for (int i = 0; i < j; ++i)
            cublasDaxpy(handle, NB3, &y[i], d_V + i * NB3, 1, d_x, 1);

        if (rel_res < tol) break;
        std::memset(H, 0, sizeof(H));  // reset for next restart
    }

    // Write solution back to interior of d_u
    k_put_interior<<<nb3, 256, 0, stream>>>(d_u, d_x);
    cudaStreamSynchronize(stream);

    cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_w);
    cudaFree(d_tmp); cudaFree(d_diag); cudaFree(d_scr); cudaFree(d_V);

    return {total_iters, rel_res};
}
