// D4: GpuGraphSolver::advance_imex — IMEX-Euler operator splitting on GPU.
//
// Step 1: SSP-RK3 (explicit convection + viscous) via advance().
// Step 2: Per-leaf, per-component implicit Helmholtz correction:
//           (I − α·∇²) u_i^{n+1} = u_i^*   where α = dt·μ/ρ_avg
//         solved by GPU-resident GMRES (gpu_helmholtz_gmres).
// Step 3: Write back momenta; KE updated, IE unchanged.
//
// In this file so that it is only linked for targets that need advance_imex
// (e.g., t32).  Existing gate tests (t24–t31) do not link this TU and
// therefore require no cuBLAS dependency.

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_gmres.cuh"
#include "mesh/cell_block.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <cmath>

// ── per-leaf kernels ──────────────────────────────────────────────────────────

// Reduce sum of Q[0] (density) over NB³ interior cells.
// Uses shared-memory block reduction; result atomically added to d_rho_sum.
__global__ static void k_rho_sum(const double* __restrict__ d_Q, double* __restrict__ d_sum) {
    extern __shared__ double sh[];
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int i    = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    sh[threadIdx.x] = (tid < NB*NB*NB) ?
        d_Q[0 * NCELL + cell_idx(i + NG, j + NG, k + NG)] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(d_sum, sh[0]);
}

// Extract velocity component v (= Q[v_idx]/Q[0]) from d_Q into NB23 d_vel array.
__global__ static void k_extract_vel(double* __restrict__ d_vel,
                                      const double* __restrict__ d_Q,
                                      int v_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB*NB*NB) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    int flat = cell_idx(i + NG, j + NG, k + NG);
    d_vel[flat] = d_Q[v_idx * NCELL + flat] / d_Q[0 * NCELL + flat];
}

// Write back: Q[v_idx] = rho * u_new; update Q[4] += ΔKE.
// Runs over NB³ interior cells.
__global__ static void k_writeback_vel(double* __restrict__ d_Q,
                                        const double* __restrict__ d_u,
                                        const double* __restrict__ d_v,
                                        const double* __restrict__ d_w) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB*NB*NB) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    int flat  = cell_idx(i + NG, j + NG, k + NG);

    double rho  = d_Q[0 * NCELL + flat];
    double u_old = d_Q[1 * NCELL + flat] / rho;
    double v_old = d_Q[2 * NCELL + flat] / rho;
    double w_old = d_Q[3 * NCELL + flat] / rho;
    double ke_old = 0.5 * rho * (u_old*u_old + v_old*v_old + w_old*w_old);

    double u_new = d_u[flat];
    double v_new = d_v[flat];
    double w_new = d_w[flat];
    double ke_new = 0.5 * rho * (u_new*u_new + v_new*v_new + w_new*w_new);

    d_Q[1 * NCELL + flat] = rho * u_new;
    d_Q[2 * NCELL + flat] = rho * v_new;
    d_Q[3 * NCELL + flat] = rho * w_new;
    d_Q[4 * NCELL + flat] += (ke_new - ke_old);  // IE unchanged
}

// ── advance_imex ──────────────────────────────────────────────────────────────
double GpuGraphSolver::advance_imex(const BlockTree& tree, double cfl, double mu) {
    // Step 1: explicit SSP-RK3
    double dt = advance(tree, cfl);

    // cuBLAS handle (create/destroy here; a persistent handle would be stored
    // in a future refactor when advance_imex is called in a hot loop).
    cublasHandle_t cb;
    cublasCreate(&cb);
    cublasSetStream(cb, stream);

    // GmresBcType from stored bc_type_
    GmresBcType gmres_bc = (bc_type_ == 1) ? GmresBcType::WallY : GmresBcType::Periodic;

    // Device scalars for rho_sum reduction
    double *d_rho_sum_dev, *d_u, *d_v, *d_w, *d_rhs;
    cudaMalloc(&d_rho_sum_dev, sizeof(double));
    cudaMalloc(&d_u,   NCELL * sizeof(double));
    cudaMalloc(&d_v,   NCELL * sizeof(double));
    cudaMalloc(&d_w,   NCELL * sizeof(double));
    cudaMalloc(&d_rhs, NCELL * sizeof(double));

    const int NB3     = NB * NB * NB;
    const int nb3blks = (NB3 + 255) / 256;

    for (auto& [blk, dQ] : download_pairs) {
        // Compute block-averaged density
        double rho_sum = 0.0;
        cudaMemcpy(d_rho_sum_dev, &rho_sum, sizeof(double), cudaMemcpyHostToDevice);
        k_rho_sum<<<nb3blks, 256, 256 * sizeof(double), stream>>>(dQ, d_rho_sum_dev);
        cudaStreamSynchronize(stream);
        cudaMemcpy(&rho_sum, d_rho_sum_dev, sizeof(double), cudaMemcpyDeviceToHost);
        double rho_avg = rho_sum / NB3;
        double alpha   = dt * mu / rho_avg;

        // Extract post-RK3 velocities
        k_extract_vel<<<nb3blks, 256, 0, stream>>>(d_u, dQ, 1);
        k_extract_vel<<<nb3blks, 256, 0, stream>>>(d_v, dQ, 2);
        k_extract_vel<<<nb3blks, 256, 0, stream>>>(d_w, dQ, 3);

        // Solve (I − α∇²) u_i = u_i*  per component
        cudaMemcpy(d_rhs, d_u, NCELL * sizeof(double), cudaMemcpyDeviceToDevice);
        gpu_helmholtz_gmres(d_u, d_rhs, alpha, blk->h, gmres_bc, cb, 50, 1e-8, 30, stream);

        cudaMemcpy(d_rhs, d_v, NCELL * sizeof(double), cudaMemcpyDeviceToDevice);
        gpu_helmholtz_gmres(d_v, d_rhs, alpha, blk->h, gmres_bc, cb, 50, 1e-8, 30, stream);

        cudaMemcpy(d_rhs, d_w, NCELL * sizeof(double), cudaMemcpyDeviceToDevice);
        gpu_helmholtz_gmres(d_w, d_rhs, alpha, blk->h, gmres_bc, cb, 50, 1e-8, 30, stream);

        // Write back and update KE
        k_writeback_vel<<<nb3blks, 256, 0, stream>>>(dQ, d_u, d_v, d_w);
    }

    cudaStreamSynchronize(stream);
    cublasDestroy(cb);
    cudaFree(d_rho_sum_dev);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_w); cudaFree(d_rhs);

    return dt;
}
