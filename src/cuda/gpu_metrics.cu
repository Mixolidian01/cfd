#include "cuda/gpu_metrics.cuh"
#include "cuda/gpu_check.cuh"
#include <cooperative_groups.h>
#include <cmath>
#include <cstdlib>

namespace cg = cooperative_groups;

// ── k_residual_norm ──────────────────────────────────────────────────────────
// gridDim = dim3(n_leaves, GPU_NVAR), blockDim = 64 (2 warps)
// Accumulates sum of RHS[var]^2 over 512 interior cells of leaf li.
// Writes partial sum to h_out[li*GPU_NVAR + var] (pinned, device-accessible).
__global__ void k_residual_norm(
    const GpuLeafRhsMeta* __restrict__ metas,
    double* __restrict__ d_out,
    int n_leaves)
{
    const int li  = blockIdx.x;
    const int var = blockIdx.y;
    if (li >= n_leaves) return;

    const double* rhs_v = metas[li].d_RHS + (size_t)var * GPU_NCELL;

    double sum_sq = 0.0;
    for (int idx = threadIdx.x; idx < GPU_NB * GPU_NB * GPU_NB; idx += 64) {
        const int kk   = idx / (GPU_NB * GPU_NB);
        const int jj   = (idx / GPU_NB) % GPU_NB;
        const int ii   = idx % GPU_NB;
        const int flat = gpu_cell_idx(GPU_NG + ii, GPU_NG + jj, GPU_NG + kk);
        const double v = rhs_v[flat];
        sum_sq += v * v;
    }

    // 2-warp reduction (same pattern as k_reduce_metrics in gpu_snapshot.cu)
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += warp.shfl_down(sum_sq, off);

    __shared__ double smem[2];
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = sum_sq;
    __syncthreads();
    if (threadIdx.x == 0)
        d_out[(size_t)li * GPU_NVAR + var] = smem[0] + smem[1];
}

// ── GpuResidualList ──────────────────────────────────────────────────────────
GpuResidualList::~GpuResidualList() {
    if (h_out) { cudaFreeHost(h_out); h_out = nullptr; }
}

void GpuResidualList::build(int n) {
    if (h_out) { cudaFreeHost(h_out); h_out = nullptr; }
    n_leaves = n;
    CUDA_CHECK(cudaMallocHost(&h_out, (size_t)n * GPU_NVAR * sizeof(double)));
}

void GpuResidualList::exec(const GpuLeafRhsMeta* d_metas, cudaStream_t s) const {
    if (!n_leaves) return;
    k_residual_norm<<<dim3(n_leaves, GPU_NVAR), 64, 0, s>>>(d_metas, h_out, n_leaves);
}

void GpuResidualList::fold(double* l2) const {
    constexpr int N_INT = GPU_NB * GPU_NB * GPU_NB;
    for (int v = 0; v < GPU_NVAR; ++v) {
        double sum = 0.0;
        for (int li = 0; li < n_leaves; ++li)
            sum += h_out[(size_t)li * GPU_NVAR + v];
        l2[v] = std::sqrt(sum / (n_leaves * N_INT));
    }
}
