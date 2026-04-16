// gpu_solver.cu — GPU NSSolver wrapper
//
// P2.5  On-device CFL reduction (atomicMin on uint64 reinterpretation of
//       double): eliminates ~40 MB D→H per step; only 8 bytes cross the bus.
//
// P2.6  CUDA Graph capture of the full RK3 loop (requires P2.5).
//       After the first step the three stages (ghost_fill+rhs+rk3_update) ×3
//       are replayed as a single cudaGraphLaunch.  Variable dt is stored in
//       device memory (d_dt); updated with a single 8-byte cudaMemcpyAsync
//       before each graph exec — still just 8 bytes D→H+H→D per step.
//       Eliminates ~30 s of kernel-launch overhead at 10⁶ steps.
//
//       Implementation details:
//         - All kernel launches during capture use stream_ (never nullptr).
//         - RK3 stage kernels have two variants: value-dt (used for the
//           first non-captured step) and pointer-dt (used inside the graph,
//           dereferencing d_dt at exec time so the captured graph carries
//           a live pointer that is updated before each replay).
//         - Graph is captured ONCE on the first step; graph_valid = true
//           thereafter.  If the block count changes (regrid), invalidate and
//           re-capture (not needed for single-block tests).
//
// Fix log:
//   P1 / P2.5 : CFL reduction on GPU (atomicMin kernel).
//   B6         : ghost kernels receive n_blocks, use blockIdx.z — all blocks
//                get ghosts filled, not just block 0.
//   P2.6       : CUDA Graph capture added; pointer-dt stage kernels added;
//                all kernel launches on stream_; fill_ghosts_device updated
//                to accept stream argument.

#include "../../include/cuda/gpu_block.cuh"
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
// Ghost-fill kernels (B6: blockIdx.z = CFD block index)
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

// P2.6: stream-aware wrapper (stream=nullptr → default; stream=s → capture)
static void fill_ghosts_device(double* d_Q, int n_blocks, cudaStream_t stream = nullptr) {
    dim3 grid(GPU_NVAR, GPU_NB2, n_blocks);
    dim3 blk (GPU_NB2,  1,       1);
    gpu_ghost_x<<<grid, blk, 0, stream>>>(d_Q, n_blocks); CUDA_CHECK(cudaGetLastError());
    gpu_ghost_y<<<grid, blk, 0, stream>>>(d_Q, n_blocks); CUDA_CHECK(cudaGetLastError());
    gpu_ghost_z<<<grid, blk, 0, stream>>>(d_Q, n_blocks); CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// P2.5: GPU CFL reduction kernel
// =============================================================================
// Each thread computes the local spectral radius sp = max(|u|+c, |v|+c, |w|+c)
// and contributes dt_local = cfl*h/sp to a global atomic min stored as uint64.
// IEEE-754 positive doubles are monotone as uint64, so atomicMin(uint64) is
// equivalent to atomicMin(double) for positive values.
// =============================================================================
__global__
void gpu_cfl_reduce_kernel(
    const double* __restrict__ Q,
    int    n_blocks,
    double h,
    double cfl,
    unsigned long long* __restrict__ d_dt_out
) {
    int n_interior = GPU_NB * GPU_NB * GPU_NB;
    int total      = n_blocks * n_interior;
    int tid        = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int b    = tid / n_interior;
    int cell = tid % n_interior;
    int li   = cell % GPU_NB;
    int lj   = (cell / GPU_NB) % GPU_NB;
    int lk   = cell / (GPU_NB * GPU_NB);
    int i    = li + GPU_NG,  j = lj + GPU_NG,  k = lk + GPU_NG;
    int idx  = gpu_cell_idx(i, j, k);
    size_t sv = (size_t)n_blocks * GPU_NCELL;

    double rho_v  = Q[0*sv + (size_t)b*GPU_NCELL + idx];
    double rhou_v = Q[1*sv + (size_t)b*GPU_NCELL + idx];
    double rhov_v = Q[2*sv + (size_t)b*GPU_NCELL + idx];
    double rhow_v = Q[3*sv + (size_t)b*GPU_NCELL + idx];
    double E_v    = Q[4*sv + (size_t)b*GPU_NCELL + idx];
    GPrim q = gpu_cons_to_prim(rho_v, rhou_v, rhov_v, rhow_v, E_v);
    double sp = fmax(fabs(q.u)+q.c, fmax(fabs(q.v)+q.c, fabs(q.w)+q.c));
    if (sp <= 0.0) return;

    double dt_local = cfl * h / sp;
    // atomicMin on uint64: valid for positive IEEE-754 doubles
    atomicMin(d_dt_out, __double_as_longlong(dt_local));
}

static double gpu_compute_dt(const double* d_Q, int n_blocks, double h, double cfl,
                              unsigned long long* d_dt_out, cudaStream_t stream = nullptr)
{
    unsigned long long inf_bits = __double_as_longlong(1e300);
    CUDA_CHECK(cudaMemcpyAsync(d_dt_out, &inf_bits, sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, stream));
    int n_interior = GPU_NB * GPU_NB * GPU_NB;
    int total      = n_blocks * n_interior;
    int tpb = 128, grid = (total + tpb - 1) / tpb;
    gpu_cfl_reduce_kernel<<<grid, tpb, 0, stream>>>(d_Q, n_blocks, h, cfl, d_dt_out);
    CUDA_CHECK(cudaGetLastError());
    // Synchronise + DtoH of 8 bytes only
    if (stream) CUDA_CHECK(cudaStreamSynchronize(stream));
    else        CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long dt_bits = 0;
    CUDA_CHECK(cudaMemcpy(&dt_bits, d_dt_out, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    return __longlong_as_double((long long)dt_bits);
}

// =============================================================================
// SSP-RK3 stage kernels — value-dt variants (used for first/fallback step)
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
// P2.6: pointer-dt variants — read dt from device memory so the captured
// CUDA Graph uses the live d_dt value updated before each replay.
// =============================================================================
__global__
void gpu_rk3_stage1_ptr(const double* __restrict__ Qn,
                               double* __restrict__ Qs,
                         const double* __restrict__ rhs,
                         const double* __restrict__ d_dt, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    double dt = *d_dt;
    Qs[tid] = Qn[tid] + dt * rhs[tid];
}

__global__
void gpu_rk3_stage23_ptr(const double* __restrict__ Qn,
                                double* __restrict__ Qs,
                          const double* __restrict__ rhs,
                          const double* __restrict__ d_dt,
                          double alpha, double beta, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    double dt = *d_dt;
    Qs[tid] = alpha * Qn[tid] + beta * (Qs[tid] + dt * rhs[tid]);
}

// =============================================================================
// GPUSolver struct
// =============================================================================
struct GPUSolver {
    double*             d_Q;        // current state  NVAR×NL×NCELL
    double*             d_Qn;       // saved Q^n
    double*             d_rhs;      // RHS scratch
    unsigned long long* d_dt_out;   // P2.5: CFL reduction output (device uint64)
    double*             d_dt;       // P2.6: dt as device double for pointer-dt kernels
    int                 n_blocks;
    double              h;
    double              t;
    int                 step;
    size_t              buf_size;

    // P2.6: CUDA Graph state
    cudaStream_t    stream_;
    cudaGraph_t     graph_;
    cudaGraphExec_t graph_exec_;
    bool            graph_valid_;
};

GPUSolver* gpu_solver_alloc(int n_blocks, double h) {
    GPUSolver* s = new GPUSolver;
    s->n_blocks     = n_blocks;
    s->h            = h;
    s->t            = 0.0;
    s->step         = 0;
    s->graph_valid_ = false;
    s->buf_size = (size_t)GPU_NVAR * n_blocks * GPU_NCELL * sizeof(double);
    CUDA_CHECK(cudaMalloc(&s->d_Q,      s->buf_size));
    CUDA_CHECK(cudaMalloc(&s->d_Qn,     s->buf_size));
    CUDA_CHECK(cudaMalloc(&s->d_rhs,    s->buf_size));
    CUDA_CHECK(cudaMalloc(&s->d_dt_out, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&s->d_dt,     sizeof(double)));
    CUDA_CHECK(cudaMemset(s->d_Q,   0, s->buf_size));
    CUDA_CHECK(cudaMemset(s->d_Qn,  0, s->buf_size));
    CUDA_CHECK(cudaMemset(s->d_rhs, 0, s->buf_size));
    CUDA_CHECK(cudaStreamCreate(&s->stream_));
    return s;
}

void gpu_solver_upload(GPUSolver* s, const double* hbuf) {
    CUDA_CHECK(cudaMemcpy(s->d_Q, hbuf, s->buf_size, cudaMemcpyHostToDevice));
}

void gpu_solver_upload_ic(GPUSolver* s, const double* hbuf) {
    CUDA_CHECK(cudaMemcpy(s->d_Q, hbuf, s->buf_size, cudaMemcpyHostToDevice));
    fill_ghosts_device(s->d_Q, s->n_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());
    s->graph_valid_ = false;  // IC change invalidates any cached graph
}

void gpu_solver_download(const GPUSolver* s, double* hbuf) {
    CUDA_CHECK(cudaMemcpy(hbuf, s->d_Q, s->buf_size, cudaMemcpyDeviceToHost));
}

// CPU-side CFL dt (host buffer) — used for diagnostics and gate test T07
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
// P2.6: capture the three RK3 stages (rhs + update + ghost) on stream_.
// Called once; subsequent steps replay via graph_exec_.
// Precondition: d_Qn has already been saved (DeviceToDevice copy done before
// capture); the graph references d_Qn, d_Q, d_rhs, d_dt by pointer — all
// remain valid across steps.
// =============================================================================
static void gpu_capture_rk3_graph(GPUSolver* s)
{
    int    NL    = s->n_blocks;
    size_t smem  = (size_t)GPU_NVAR * GPU_NCELL * sizeof(double);
    dim3   rhs_blk(GPU_NB2, GPU_NB2, 1);
    dim3   rhs_grd(NL, 1, 1);
    int    trk   = 256;
    int    grk   = (GPU_NVAR * NL * GPU_NCELL + trk - 1) / trk;
    int    total = GPU_NVAR * NL * GPU_NCELL;

    CUDA_CHECK(cudaStreamBeginCapture(s->stream_, cudaStreamCaptureModeGlobal));

    // Stage 1: Q^(1) = Q^n + dt*L(Q^n)
    gpu_rhs_kernel<<<rhs_grd, rhs_blk, smem, s->stream_>>>(s->d_Q, s->d_rhs, NL, 1.0/s->h);
    gpu_rk3_stage1_ptr<<<grk, trk, 0, s->stream_>>>(s->d_Qn, s->d_Q, s->d_rhs, s->d_dt, total);
    fill_ghosts_device(s->d_Q, NL, s->stream_);

    // Stage 2: Q^(2) = 3/4·Q^n + 1/4·(Q^(1) + dt·L(Q^(1)))
    gpu_rhs_kernel<<<rhs_grd, rhs_blk, smem, s->stream_>>>(s->d_Q, s->d_rhs, NL, 1.0/s->h);
    gpu_rk3_stage23_ptr<<<grk, trk, 0, s->stream_>>>(s->d_Qn, s->d_Q, s->d_rhs, s->d_dt, 0.75, 0.25, total);
    fill_ghosts_device(s->d_Q, NL, s->stream_);

    // Stage 3: Q^{n+1} = 1/3·Q^n + 2/3·(Q^(2) + dt·L(Q^(2)))
    gpu_rhs_kernel<<<rhs_grd, rhs_blk, smem, s->stream_>>>(s->d_Q, s->d_rhs, NL, 1.0/s->h);
    gpu_rk3_stage23_ptr<<<grk, trk, 0, s->stream_>>>(s->d_Qn, s->d_Q, s->d_rhs, s->d_dt, 1.0/3.0, 2.0/3.0, total);
    fill_ghosts_device(s->d_Q, NL, s->stream_);

    CUDA_CHECK(cudaStreamEndCapture(s->stream_, &s->graph_));
    CUDA_CHECK(cudaGraphInstantiate(&s->graph_exec_, s->graph_, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(s->graph_));   // graph_exec_ is self-contained
    s->graph_valid_ = true;
}

// =============================================================================
// gpu_solver_step — one SSP-RK3 step
//
// P2.5: CFL dt computed on GPU (8-byte DtoH instead of ~40 KB).
// P2.6: After first step, the 3 RK3 stages are replayed from a captured
//       CUDA Graph.  Before each replay, dt is written to d_dt (8-byte H2D
//       via cudaMemcpyAsync on stream_).  This eliminates kernel-launch
//       overhead for the hot loop.
//
// htmp: optional host output buffer; pass nullptr to skip diagnostic download.
// =============================================================================
double gpu_solver_step(GPUSolver* s, double cfl, double* htmp)
{
    // P2.5: on-device CFL reduction — only 8 bytes cross the bus
    double dt = gpu_compute_dt(s->d_Q, s->n_blocks, s->h, cfl, s->d_dt_out, nullptr);

    // Save Q^n (DeviceToDevice, very fast)
    CUDA_CHECK(cudaMemcpyAsync(s->d_Qn, s->d_Q, s->buf_size,
                               cudaMemcpyDeviceToDevice, s->stream_));

    if (!s->graph_valid_) {
        // ── First step: run explicitly on stream_ and capture the graph ──
        int    NL    = s->n_blocks;
        size_t smem  = (size_t)GPU_NVAR * GPU_NCELL * sizeof(double);
        dim3   rhs_blk(GPU_NB2, GPU_NB2, 1);
        dim3   rhs_grd(NL, 1, 1);
        int    trk   = 256;
        int    total = GPU_NVAR * NL * GPU_NCELL;
        int    grk   = (total + trk - 1) / trk;

        // Upload dt to device for pointer-dt kernels
        CUDA_CHECK(cudaMemcpyAsync(s->d_dt, &dt, sizeof(double),
                                   cudaMemcpyHostToDevice, s->stream_));
        CUDA_CHECK(cudaStreamSynchronize(s->stream_));

        // Stage 1
        gpu_rhs_kernel<<<rhs_grd,rhs_blk,smem,s->stream_>>>(s->d_Q,s->d_rhs,NL,1.0/s->h);
        gpu_rk3_stage1<<<grk,trk,0,s->stream_>>>(s->d_Qn,s->d_Q,s->d_rhs,dt,total);
        fill_ghosts_device(s->d_Q, NL, s->stream_);
        // Stage 2
        gpu_rhs_kernel<<<rhs_grd,rhs_blk,smem,s->stream_>>>(s->d_Q,s->d_rhs,NL,1.0/s->h);
        gpu_rk3_stage23<<<grk,trk,0,s->stream_>>>(s->d_Qn,s->d_Q,s->d_rhs,0.75,0.25,dt,total);
        fill_ghosts_device(s->d_Q, NL, s->stream_);
        // Stage 3
        gpu_rhs_kernel<<<rhs_grd,rhs_blk,smem,s->stream_>>>(s->d_Q,s->d_rhs,NL,1.0/s->h);
        gpu_rk3_stage23<<<grk,trk,0,s->stream_>>>(s->d_Qn,s->d_Q,s->d_rhs,1.0/3.0,2.0/3.0,dt,total);
        fill_ghosts_device(s->d_Q, NL, s->stream_);
        CUDA_CHECK(cudaStreamSynchronize(s->stream_));

        // Capture the graph for all subsequent steps
        gpu_capture_rk3_graph(s);

    } else {
        // ── P2.6: subsequent steps — update dt and replay graph ──────────
        // 8-byte H2D on stream_ (overlaps with Qn copy if stream is ordered)
        CUDA_CHECK(cudaMemcpyAsync(s->d_dt, &dt, sizeof(double),
                                   cudaMemcpyHostToDevice, s->stream_));
        CUDA_CHECK(cudaGraphLaunch(s->graph_exec_, s->stream_));
        CUDA_CHECK(cudaStreamSynchronize(s->stream_));
    }

    s->t    += dt;
    s->step += 1;

    // Optional diagnostic download — skipped in hot path (htmp == nullptr)
    if (htmp) gpu_solver_download(s, htmp);

    return dt;
}

void gpu_solver_free(GPUSolver* s) {
    if (!s) return;
    if (s->graph_valid_) cudaGraphExecDestroy(s->graph_exec_);
    cudaStreamDestroy(s->stream_);
    cudaFree(s->d_Q);
    cudaFree(s->d_Qn);
    cudaFree(s->d_rhs);
    cudaFree(s->d_dt_out);
    cudaFree(s->d_dt);
    delete s;
}
