// gpu_amr.cu — P8.4: GPU AMR prolongation and restriction kernels
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
