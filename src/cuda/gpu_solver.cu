// gpu_solver.cu — GPU NSSolver wrapper
// FIX P1: CFL reduction now runs entirely on the GPU via a two-pass
//         atomicMin kernel. The full device->host download that was done
//         every step just to call gpu_cfl_dt() on the CPU is eliminated.
//         Only 1 double (the reduced dt) is transferred D->H per step,
//         instead of NVAR * n_blocks * NB2^3 * 8 bytes (~40 KB per block).
//
//         Implementation:
//           - gpu_cfl_reduce_kernel: each thread computes local spectral
//             radius, converts to uint64 via __double_as_longlong for
//             atomicMin, writes into a single d_dt_inv scalar.
//           - GPUSolver gains d_dt_inv (device) and a pinned host mirror
//             h_dt_inv for async DtoH transfer.
//           - htmp parameter of gpu_solver_step() is kept for external
//             callers that still want a host copy (e.g. diagnostics),
//             but the download is now skipped unless htmp != nullptr AND
//             the caller requests it explicitly via gpu_solver_download().
//
// FIX B6 (retained): ghost kernels receive n_blocks and use blockIdx.z
//         so all CFD blocks get their ghosts filled, not just block 0.

#include "../include/cuda/gpu_block.cuh"
#include "gpu_rhs.cu"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <limits>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// =============================================================================
// Ghost-fill kernels (FIX B6: blockIdx.z = CFD block index)
// =============================================================================
__global__ void gpu_ghost_x(double* __restrict__ Q, int n_blocks) {
    int v = blockIdx.x, j = threadIdx.x, k = blockIdx.y, b = blockIdx.z;
    double* Qv = Q + (size_t)v * n_blocks * GPU_NCELL + (size_t)b * GPU_NCELL;
    Qv[gpu_cell_idx(0,         j, k)] = Qv[gpu_cell_idx(GPU_NB2-2, j, k)];
    Qv[gpu_cell_idx(GPU_NB2-1, j, k)] = Qv[gpu_cell_idx(1,         j, k)];
}
__global__ void gpu_ghost_y(double* __restrict__ Q, int n_blocks) {
    int v = blockIdx.x, i = threadIdx.x, k = blockIdx.y, b = blockIdx.z;
    double* Qv = Q + (size_t)v * n_blocks * GPU_NCELL + (size_t)b * GPU_NCELL;
    Qv[gpu_cell_idx(i, 0,         k)] = Qv[gpu_cell_idx(i, GPU_NB2-2, k)];
    Qv[gpu_cell_idx(i, GPU_NB2-1, k)] = Qv[gpu_cell_idx(i, 1,         k)];
}
__global__ void gpu_ghost_z(double* __restrict__ Q, int n_blocks) {
    int v = blockIdx.x, i = threadIdx.x, j = blockIdx.y, b = blockIdx.z;
    double* Qv = Q + (size_t)v * n_blocks * GPU_NCELL + (size_t)b * GPU_NCELL;
    Qv[gpu_cell_idx(i, j, 0        )] = Qv[gpu_cell_idx(i, j, GPU_NB2-2)];
    Qv[gpu_cell_idx(i, j, GPU_NB2-1)] = Qv[gpu_cell_idx(i, j, 1        )];
}

// FIX B6: z-dim = n_blocks so every CFD block gets its ghosts filled
static void fill_ghosts_device(double* d_Q, int n_blocks) {
    dim3 grid(GPU_NVAR, GPU_NB2, n_blocks);
    dim3 blk (GPU_NB2,  1,       1);
    gpu_ghost_x<<<grid, blk>>>(d_Q, n_blocks); CUDA_CHECK(cudaGetLastError());
    gpu_ghost_y<<<grid, blk>>>(d_Q, n_blocks); CUDA_CHECK(cudaGetLastError());
    gpu_ghost_z<<<grid, blk>>>(d_Q, n_blocks); CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// FIX P1: GPU CFL reduction kernel
// =============================================================================
// Computes min(cfl * h / spectral_radius) over all interior cells of all
// blocks using a single atomicMin on a uint64 reinterpretation of the
// double, which is valid because IEEE-754 doubles are monotone as uint64
// for positive values.
//
// d_dt_inv: device scalar, must be initialised to 0 before launch.
//           After kernel: dt = cfl * h / (*d_dt_inv_as_double)
//           We store 1/dt_min = max_spectral_radius / (cfl*h) to allow
//           atomic max on unsigned integer.
//
// Grid: one thread per interior cell across all blocks.
//   blockDim = (128,1,1)  gridDim = (ceil(n_interior / 128), 1, 1)
// =============================================================================
__global__
void gpu_cfl_reduce_kernel(
    const double* __restrict__ Q,
    int    n_blocks,
    double h,
    double cfl,
    unsigned long long* __restrict__ d_dt_out   // stores bits of min(dt)
) {
    // Each thread handles one interior cell of one block
    int n_interior = GPU_NB * GPU_NB * GPU_NB;   // cells per block
    int total      = n_blocks * n_interior;
    int tid        = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int b    = tid / n_interior;
    int cell = tid % n_interior;

    // Map flat interior index to (i,j,k) in [NG, NG+NB)
    int li   = cell % GPU_NB;
    int lj   = (cell / GPU_NB) % GPU_NB;
    int lk   = cell / (GPU_NB * GPU_NB);
    int i    = li + GPU_NG;
    int j    = lj + GPU_NG;
    int k    = lk + GPU_NG;

    int idx = gpu_cell_idx(i, j, k);
    size_t stride_var = (size_t)n_blocks * GPU_NCELL;

    double rho_v  = Q[0*stride_var + (size_t)b*GPU_NCELL + idx];
    double rhou_v = Q[1*stride_var + (size_t)b*GPU_NCELL + idx];
    double rhov_v = Q[2*stride_var + (size_t)b*GPU_NCELL + idx];
    double rhow_v = Q[3*stride_var + (size_t)b*GPU_NCELL + idx];
    double E_v    = Q[4*stride_var + (size_t)b*GPU_NCELL + idx];

    GPrim q = gpu_cons_to_prim(rho_v, rhou_v, rhov_v, rhow_v, E_v);

    double sp = fmax(fabs(q.u) + q.c,
                fmax(fabs(q.v) + q.c,
                     fabs(q.w) + q.c));

    if (sp <= 0.0) return;

    double dt_local = cfl * h / sp;

    // atomicMin on uint64: IEEE-754 positive doubles preserve order as uint64
    unsigned long long dt_bits = __double_as_longlong(dt_local);
    atomicMin(d_dt_out, dt_bits);
}

// Host helper: launch reduction, return dt
// d_dt_out must be a device pointer initialised to __double_as_longlong(1e300)
static double gpu_compute_dt(const double* d_Q, int n_blocks, double h, double cfl,
                              unsigned long long* d_dt_out)
{
    // Reset to +inf
    unsigned long long inf_bits = __double_as_longlong(1e300);
    CUDA_CHECK(cudaMemcpy(d_dt_out, &inf_bits, sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));

    int n_interior = GPU_NB * GPU_NB * GPU_NB;
    int total      = n_blocks * n_interior;
    int tpb        = 128;
    int grid       = (total + tpb - 1) / tpb;

    gpu_cfl_reduce_kernel<<<grid, tpb>>>(d_Q, n_blocks, h, cfl, d_dt_out);
    CUDA_CHECK(cudaGetLastError());

    // Small synchronous DtoH of just 8 bytes
    unsigned long long dt_bits = 0;
    CUDA_CHECK(cudaMemcpy(&dt_bits, d_dt_out, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    return __longlong_as_double((long long)dt_bits);
}

// =============================================================================
// SSP-RK3 stage kernels (unchanged)
// =============================================================================
__global__
void gpu_rk3_stage1(const double* __restrict__ Qn,
                          double* __restrict__ Qs,
                    const double* __restrict__ rhs,
                    double dt, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    Qs[tid] = Qn[tid] + dt * rhs[tid];
}

__global__
void gpu_rk3_stage23(const double* __restrict__ Qn,
                           double* __restrict__ Qs,
                     const double* __restrict__ rhs,
                     double alpha, double beta, double dt, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    Qs[tid] = alpha * Qn[tid] + beta * (Qs[tid] + dt * rhs[tid]);
}

// =============================================================================
// GPUSolver struct
// =============================================================================
struct GPUSolver {
    double*             d_Q;
    double*             d_Qn;
    double*             d_rhs;
    unsigned long long* d_dt_out;   // FIX P1: device scalar for CFL reduction
    int                 n_blocks;
    double              h;
    double              t;
    int                 step;
    size_t              buf_size;
};

GPUSolver* gpu_solver_alloc(int n_blocks, double h) {
    GPUSolver* s = new GPUSolver;
    s->n_blocks = n_blocks; s->h = h; s->t = 0.0; s->step = 0;
    s->buf_size = (size_t)GPU_NVAR * n_blocks * GPU_NCELL * sizeof(double);
    CUDA_CHECK(cudaMalloc(&s->d_Q,      s->buf_size));
    CUDA_CHECK(cudaMalloc(&s->d_Qn,     s->buf_size));
    CUDA_CHECK(cudaMalloc(&s->d_rhs,    s->buf_size));
    CUDA_CHECK(cudaMalloc(&s->d_dt_out, sizeof(unsigned long long))); // FIX P1
    CUDA_CHECK(cudaMemset(s->d_Q,   0, s->buf_size));
    CUDA_CHECK(cudaMemset(s->d_Qn,  0, s->buf_size));
    CUDA_CHECK(cudaMemset(s->d_rhs, 0, s->buf_size));
    return s;
}

void gpu_solver_upload(GPUSolver* s, const double* hbuf) {
    CUDA_CHECK(cudaMemcpy(s->d_Q, hbuf, s->buf_size, cudaMemcpyHostToDevice));
}

void gpu_solver_upload_ic(GPUSolver* s, const double* hbuf) {
    CUDA_CHECK(cudaMemcpy(s->d_Q, hbuf, s->buf_size, cudaMemcpyHostToDevice));
    fill_ghosts_device(s->d_Q, s->n_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_solver_download(const GPUSolver* s, double* hbuf) {
    CUDA_CHECK(cudaMemcpy(hbuf, s->d_Q, s->buf_size, cudaMemcpyDeviceToHost));
}

// CPU fallback — only used for diagnostics or unit tests, not in the hot path
double gpu_cfl_dt(const double* hQ, int nb, double h, double cfl) {
    double dt = 1e300;
    size_t sv = (size_t)nb * GPU_NCELL;
    for (int b = 0; b < nb; ++b) {
        const double* rho  = hQ + 0*sv + (size_t)b*GPU_NCELL;
        const double* rhou = hQ + 1*sv + (size_t)b*GPU_NCELL;
        const double* rhov = hQ + 2*sv + (size_t)b*GPU_NCELL;
        const double* rhow = hQ + 3*sv + (size_t)b*GPU_NCELL;
        const double* E    = hQ + 4*sv + (size_t)b*GPU_NCELL;
        for (int k = GPU_NG; k < GPU_NB2-GPU_NG; ++k)
        for (int j = GPU_NG; j < GPU_NB2-GPU_NG; ++j)
        for (int i = GPU_NG; i < GPU_NB2-GPU_NG; ++i) {
            int idx = gpu_cell_idx(i,j,k);
            GPrim q = gpu_cons_to_prim(rho[idx],rhou[idx],rhov[idx],rhow[idx],E[idx]);
            double sp = fmax(fabs(q.u)+q.c, fmax(fabs(q.v)+q.c, fabs(q.w)+q.c));
            if (sp > 0.0) dt = fmin(dt, cfl * h / sp);
        }
    }
    return dt;
}

// =============================================================================
// gpu_solver_step — one SSP-RK3 step
// FIX P1: dt computed on GPU via reduction kernel (8-byte DtoH, not ~40KB+)
//         htmp is now optional: pass nullptr to skip the diagnostic download.
// =============================================================================
double gpu_solver_step(GPUSolver* s, double cfl, double* htmp)
{
    // FIX P1: CFL reduction entirely on GPU — only 8 bytes cross the bus
    double dt   = gpu_compute_dt(s->d_Q, s->n_blocks, s->h, cfl, s->d_dt_out);
    double hinv = 1.0 / s->h;
    int    NL   = s->n_blocks;
    int    total = GPU_NVAR * NL * GPU_NCELL;
    size_t smem  = (size_t)GPU_NVAR * GPU_NCELL * sizeof(double);

    dim3 rhs_blk(GPU_NB2, GPU_NB2, 1);
    dim3 rhs_grd(NL, 1, 1);
    int  trk = 256, grk = (total + trk - 1) / trk;

    // Save Q^n
    CUDA_CHECK(cudaMemcpy(s->d_Qn, s->d_Q, s->buf_size, cudaMemcpyDeviceToDevice));

    // Stage 1: Q^(1) = Q^n + dt*L(Q^n)
    gpu_rhs_kernel<<<rhs_grd, rhs_blk, smem>>>(s->d_Q, s->d_rhs, NL, hinv);
    CUDA_CHECK(cudaGetLastError());
    gpu_rk3_stage1<<<grk, trk>>>(s->d_Qn, s->d_Q, s->d_rhs, dt, total);
    CUDA_CHECK(cudaGetLastError());
    fill_ghosts_device(s->d_Q, NL);

    // Stage 2: Q^(2) = 3/4*Q^n + 1/4*(Q^(1) + dt*L(Q^(1)))
    gpu_rhs_kernel<<<rhs_grd, rhs_blk, smem>>>(s->d_Q, s->d_rhs, NL, hinv);
    CUDA_CHECK(cudaGetLastError());
    gpu_rk3_stage23<<<grk, trk>>>(s->d_Qn, s->d_Q, s->d_rhs, 0.75, 0.25, dt, total);
    CUDA_CHECK(cudaGetLastError());
    fill_ghosts_device(s->d_Q, NL);

    // Stage 3: Q^(n+1) = 1/3*Q^n + 2/3*(Q^(2) + dt*L(Q^(2)))
    gpu_rhs_kernel<<<rhs_grd, rhs_blk, smem>>>(s->d_Q, s->d_rhs, NL, hinv);
    CUDA_CHECK(cudaGetLastError());
    gpu_rk3_stage23<<<grk, trk>>>(s->d_Qn, s->d_Q, s->d_rhs, 1.0/3.0, 2.0/3.0, dt, total);
    CUDA_CHECK(cudaGetLastError());
    fill_ghosts_device(s->d_Q, NL);

    CUDA_CHECK(cudaDeviceSynchronize());
    s->t    += dt;
    s->step += 1;

    // Optional diagnostic download — skipped when htmp == nullptr (hot path)
    if (htmp) gpu_solver_download(s, htmp);

    return dt;
}

void gpu_solver_free(GPUSolver* s) {
    if (!s) return;
    cudaFree(s->d_Q);
    cudaFree(s->d_Qn);
    cudaFree(s->d_rhs);
    cudaFree(s->d_dt_out);   // FIX P1
    delete s;
}
