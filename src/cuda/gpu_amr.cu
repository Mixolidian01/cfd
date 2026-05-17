// gpu_amr.cu — P8.4 / D1: GPU AMR prolongation, restriction, and sensor kernels
//
// k_prolong: piecewise-constant injection, one CUDA block per (coarse,fine) pair.
//   Thread (li,lj,lk): fine interior cell (NG+li, NG+lj, NG+lk).
//   Coarse index: ic = NG + ix*(NB/2) + li/2  (and analogously for j,k).
//
// k_restrict: volume-weighted average, one CUDA block per coarse block.
//   Thread (li,lj,lk): coarse interior cell (NG+li, NG+lj, NG+lk).
//   Sums the 2×2×2 fine-cell footprint from the owning child octant.
//
// Both kernels write only interior cells and leave ghost layers untouched;
// ghost fill (same-level or C/F) must follow before the next RHS evaluation.

#include "cuda/gpu_amr.cuh"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_check.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// ─────────────────────────────────────────────────────────────────────────────
// k_prolong
// gridDim.x  = n_prolong pairs
// blockDim   = dim3(GPU_NB, GPU_NB, GPU_NB)
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_prolong(const GpuProlongMeta* __restrict__ metas)
{
    const GpuProlongMeta& m = metas[blockIdx.x];

    const int li = threadIdx.x;   // 0 .. GPU_NB-1
    const int lj = threadIdx.y;
    const int lk = threadIdx.z;

    // Octant high-half flags
    const int ix = m.oct       & 1;
    const int iy = (m.oct >> 1) & 1;
    const int iz = (m.oct >> 2) & 1;

    // Fine interior cell
    const int inf_ = GPU_NG + li;
    const int jf   = GPU_NG + lj;
    const int kf   = GPU_NG + lk;

    // Corresponding coarse cell (piecewise-constant: take the parent cell)
    constexpr int half = GPU_NB / 2;
    const int ic = GPU_NG + ix * half + li / 2;
    const int jc = GPU_NG + iy * half + lj / 2;
    const int kc = GPU_NG + iz * half + lk / 2;

    const int fine_flat   = gpu_cell_idx(inf_, jf, kf);
    const int coarse_flat = gpu_cell_idx(ic,   jc, kc);

    for (int v = 0; v < GPU_NVAR; ++v)
        m.d_fine_Q[v * GPU_NCELL + fine_flat] =
            m.d_coarse_Q[v * GPU_NCELL + coarse_flat];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_restrict
// gridDim.x  = n_restrict pairs
// blockDim   = dim3(GPU_NB, GPU_NB, GPU_NB)
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_restrict(const GpuRestrictMeta* __restrict__ metas)
{
    const GpuRestrictMeta& m = metas[blockIdx.x];

    const int li = threadIdx.x;   // 0 .. GPU_NB-1
    const int lj = threadIdx.y;
    const int lk = threadIdx.z;

    constexpr int half = GPU_NB / 2;

    // Which child octant owns this coarse cell?
    const int child = (li / half) | ((lj / half) << 1) | ((lk / half) << 2);

    // Base fine-cell index within that child's interior
    const int fi = GPU_NG + (li % half) * 2;
    const int fj = GPU_NG + (lj % half) * 2;
    const int fk = GPU_NG + (lk % half) * 2;

    const int coarse_flat = gpu_cell_idx(GPU_NG + li, GPU_NG + lj, GPU_NG + lk);

    for (int v = 0; v < GPU_NVAR; ++v) {
        double sum = 0.0;
        for (int dk = 0; dk < 2; ++dk)
        for (int dj = 0; dj < 2; ++dj)
        for (int di = 0; di < 2; ++di)
            sum += m.d_children_Q[child][v * GPU_NCELL +
                                         gpu_cell_idx(fi+di, fj+dj, fk+dk)];
        m.d_coarse_Q[v * GPU_NCELL + coarse_flat] = sum * 0.125;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// k_refine_sensor
// D1: one block per leaf.  Block = dim3(GPU_NB, GPU_NB, GPU_NB) = 512 threads.
// Each thread computes the normalised gradient magnitude at its interior cell.
// A warp-level + block-level max reduction writes the leaf result to d_sensor.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_refine_sensor(const GpuSensorMeta* __restrict__ metas, float* __restrict__ d_sensor)
{
    const GpuSensorMeta& m = metas[blockIdx.x];
    const double* Q = m.d_Q;
    const float   h = m.h;

    const int li = threadIdx.x;
    const int lj = threadIdx.y;
    const int lk = threadIdx.z;
    const int i  = GPU_NG + li;
    const int j  = GPU_NG + lj;
    const int k  = GPU_NG + lk;

    float val = 0.0f;
    // Skip the outermost layer (needs ghost cells for centred gradient)
    if (li > 0 && li < GPU_NB - 1 &&
        lj > 0 && lj < GPU_NB - 1 &&
        lk > 0 && lk < GPU_NB - 1) {
        const double rho_c = Q[gpu_cell_idx(i,   j,   k)];
        const double rho_xp = Q[gpu_cell_idx(i+1, j,   k)];
        const double rho_xm = Q[gpu_cell_idx(i-1, j,   k)];
        const double rho_yp = Q[gpu_cell_idx(i,   j+1, k)];
        const double rho_ym = Q[gpu_cell_idx(i,   j-1, k)];
        const double rho_zp = Q[gpu_cell_idx(i,   j,   k+1)];
        const double rho_zm = Q[gpu_cell_idx(i,   j,   k-1)];
        const double h2inv  = 0.5 / (double)h;
        const double gx = h2inv * (rho_xp - rho_xm);
        const double gy = h2inv * (rho_yp - rho_ym);
        const double gz = h2inv * (rho_zp - rho_zm);
        val = (float)(__dsqrt_rn(gx*gx + gy*gy + gz*gz)
                      * (double)h / (fabs(rho_c) + 1.0e-30));
    }

    // Block-wide max reduction via shared memory
    __shared__ float smax[GPU_NB * GPU_NB * GPU_NB];
    const int tid = li + GPU_NB * (lj + GPU_NB * lk);
    smax[tid] = val;
    __syncthreads();

    for (int s = (GPU_NB * GPU_NB * GPU_NB) / 2; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }
    if (tid == 0) d_sensor[blockIdx.x] = smax[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// gpu_eval_refine_sensor  (host helper — D1)
// ─────────────────────────────────────────────────────────────────────────────
void gpu_eval_refine_sensor(
    const double* const* d_Q_ptrs,
    const float*         h_vals,
    int                  n_leaves,
    float*               d_sensor,
    cudaStream_t         stream)
{
    if (n_leaves == 0) return;

    // Build host-side metadata array and upload.
    std::vector<GpuSensorMeta> h_metas(n_leaves);
    for (int i = 0; i < n_leaves; ++i) {
        h_metas[i].d_Q = d_Q_ptrs[i];
        h_metas[i].h   = h_vals[i];
        h_metas[i]._pad = 0;
    }
    GpuSensorMeta* d_metas = nullptr;
    CUDA_CHECK(cudaMalloc(&d_metas, n_leaves * sizeof(GpuSensorMeta)));
    CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas.data(),
                               n_leaves * sizeof(GpuSensorMeta),
                               cudaMemcpyHostToDevice, stream));

    const dim3 block(GPU_NB, GPU_NB, GPU_NB);
    k_refine_sensor<<<n_leaves, block, 0, stream>>>(d_metas, d_sensor);
    CUDA_CHECK(cudaGetLastError());

    if (!stream) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(d_metas));
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuAmrList methods
// ─────────────────────────────────────────────────────────────────────────────

GpuAmrList::~GpuAmrList() {
    if (d_prolong)  { cudaFree(d_prolong);  d_prolong  = nullptr; }
    if (d_restrict) { cudaFree(d_restrict); d_restrict = nullptr; }
    n_prolong = n_restrict = 0;
}

void GpuAmrList::build_prolong(const std::vector<GpuProlongMeta>& ops) {
    if (d_prolong) { cudaFree(d_prolong); d_prolong = nullptr; }
    n_prolong = (int)ops.size();
    if (n_prolong == 0) return;
    CUDA_CHECK(cudaMalloc(&d_prolong, n_prolong * sizeof(GpuProlongMeta)));
    CUDA_CHECK(cudaMemcpy(d_prolong, ops.data(),
                          n_prolong * sizeof(GpuProlongMeta),
                          cudaMemcpyHostToDevice));
}

void GpuAmrList::build_restrict(const std::vector<GpuRestrictMeta>& ops) {
    if (d_restrict) { cudaFree(d_restrict); d_restrict = nullptr; }
    n_restrict = (int)ops.size();
    if (n_restrict == 0) return;
    CUDA_CHECK(cudaMalloc(&d_restrict, n_restrict * sizeof(GpuRestrictMeta)));
    CUDA_CHECK(cudaMemcpy(d_restrict, ops.data(),
                          n_restrict * sizeof(GpuRestrictMeta),
                          cudaMemcpyHostToDevice));
}

void GpuAmrList::exec_prolong(cudaStream_t stream) const {
    if (n_prolong == 0) return;
    const dim3 block(GPU_NB, GPU_NB, GPU_NB);
    k_prolong<<<n_prolong, block, 0, stream>>>(d_prolong);
    CUDA_CHECK(cudaGetLastError());
    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    else         CUDA_CHECK(cudaStreamSynchronize(stream));
}

void GpuAmrList::exec_restrict(cudaStream_t stream) const {
    if (n_restrict == 0) return;
    const dim3 block(GPU_NB, GPU_NB, GPU_NB);
    k_restrict<<<n_restrict, block, 0, stream>>>(d_restrict);
    CUDA_CHECK(cudaGetLastError());
    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    else         CUDA_CHECK(cudaStreamSynchronize(stream));
}
