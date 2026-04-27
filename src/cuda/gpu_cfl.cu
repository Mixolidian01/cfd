// gpu_cfl.cu — P8.5: GPU CFL warp-shuffle tree reduction
//
// k_cfl_reduce: flat thread loop over all leaves × NB³ interior cells.
//   Each thread computes the local spectral radius sp = max(|u|+c, |v|+c, |w|+c)
//   and dt_local = cfl*h/sp.
//   Warp-level min via __shfl_down_sync; block-level min via shared memory +
//   first-warp reduction; global min via atomicMin on uint64 reinterpretation
//   (IEEE-754 positive doubles are monotone as uint64).
//
// After exec():
//   d_dt_bits holds min(dt) as uint64; a second kernel writes it to d_dt
//   (double) so pointer-dt RK3 kernels can dereference it without D→H copy.

#include "../../include/cuda/gpu_cfl.cuh"
#include "../../include/cuda/gpu_block.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

static inline unsigned long long double_to_ull(double x) noexcept {
    unsigned long long u; memcpy(&u, &x, sizeof(u)); return u;
}
static inline double ull_to_double(unsigned long long u) noexcept {
    double x; memcpy(&x, &u, sizeof(x)); return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// k_cfl_reduce: warp-shuffle min over all leaves × NB³ cells
// blockDim = 256 (8 warps)
// gridDim  = ceil(n_leaves × NB³ / 256)
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_cfl_reduce(const GpuLeafCflMeta* __restrict__ metas,
                  int   n_leaves,
                  double cfl,
                  unsigned long long* __restrict__ d_dt_bits)
{
    constexpr int N_INT = GPU_NB * GPU_NB * GPU_NB;   // 512 interior cells per leaf
    const int total = n_leaves * N_INT;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-thread minimum (stored as uint64 for atomicMin compatibility)
    constexpr unsigned long long INF_BITS = 0x7FEFFFFFFFFFFFFEULL; // max finite double
    unsigned long long local_bits = INF_BITS;

    for (int gid = tid; gid < total; gid += blockDim.x * gridDim.x) {
        const int leaf = gid / N_INT;
        const int cell = gid % N_INT;

        const GpuLeafCflMeta& m = metas[leaf];
        const int li = cell % GPU_NB;
        const int lj = (cell / GPU_NB) % GPU_NB;
        const int lk = cell / (GPU_NB * GPU_NB);
        const int flat = gpu_cell_idx(GPU_NG+li, GPU_NG+lj, GPU_NG+lk);

        GPrim q = gpu_cons_to_prim(
            m.d_Q[0*GPU_NCELL+flat], m.d_Q[1*GPU_NCELL+flat],
            m.d_Q[2*GPU_NCELL+flat], m.d_Q[3*GPU_NCELL+flat],
            m.d_Q[4*GPU_NCELL+flat]);

        const double sp = fmax(fabs(q.u)+q.c, fmax(fabs(q.v)+q.c, fabs(q.w)+q.c));
        if (sp > 0.0) {
            unsigned long long bits = __double_as_longlong(cfl * m.h / sp);
            if (bits < local_bits) local_bits = bits;
        }
    }

    // ── Warp-level reduction (min via shuffle) ────────────────────────────────
    const unsigned int full_mask = 0xffffffff;
    for (int offset = 16; offset >= 1; offset >>= 1) {
        unsigned long long other = __shfl_down_sync(full_mask, local_bits, offset);
        if (other < local_bits) local_bits = other;
    }

    // ── Block-level reduction (shared memory + first warp) ────────────────────
    __shared__ unsigned long long smem[32];   // one slot per warp
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (lane_id == 0) smem[warp_id] = local_bits;
    __syncthreads();

    if (warp_id == 0) {
        const int n_warps = (blockDim.x + 31) / 32;
        local_bits = (lane_id < n_warps) ? smem[lane_id] : INF_BITS;
        for (int offset = 16; offset >= 1; offset >>= 1) {
            unsigned long long other = __shfl_down_sync(full_mask, local_bits, offset);
            if (other < local_bits) local_bits = other;
        }
        if (lane_id == 0) atomicMin(d_dt_bits, local_bits);
    }
}

// ── Tiny kernel: uint64 bits → device double ────────────────────────────────
__global__
void k_bits_to_double(const unsigned long long* __restrict__ bits,
                      double* __restrict__ d_dt) {
    *d_dt = __longlong_as_double((long long)*bits);
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuCflList methods
// ─────────────────────────────────────────────────────────────────────────────

GpuCflList::~GpuCflList() {
    if (d_metas)   { cudaFree(d_metas);   d_metas   = nullptr; }
    if (d_dt_bits) { cudaFree(d_dt_bits); d_dt_bits = nullptr; }
    if (d_dt)      { cudaFree(d_dt);      d_dt      = nullptr; }
}

void GpuCflList::build(const BlockTree& tree) {
    cudaFree(d_metas);   d_metas   = nullptr;
    cudaFree(d_dt_bits); d_dt_bits = nullptr;
    cudaFree(d_dt);      d_dt      = nullptr;

    const auto& leaves = tree.leaf_indices();
    n_leaves = (int)leaves.size();
    if (n_leaves == 0) return;

    std::vector<GpuLeafCflMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const BlockNode& nd = tree.nodes[leaves[li]];
        h_metas[li].d_Q    = nd.block->d_Q;
        h_metas[li].h      = nd.block->h;
        h_metas[li]._pad[0]= h_metas[li]._pad[1] = 0;
    }

    CUDA_CHECK(cudaMalloc(&d_metas,   n_leaves * sizeof(GpuLeafCflMeta)));
    CUDA_CHECK(cudaMalloc(&d_dt_bits, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_dt,      sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas.data(),
                          n_leaves * sizeof(GpuLeafCflMeta),
                          cudaMemcpyHostToDevice));
}

double GpuCflList::exec(double cfl, cudaStream_t stream) const {
    if (n_leaves == 0) return 1.0e300;

    // Reset reduction target to +inf
    constexpr unsigned long long INF_BITS = 0x7FEFFFFFFFFFFFFEULL;
    CUDA_CHECK(cudaMemcpyAsync(d_dt_bits, &INF_BITS, sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, stream));

    // Launch k_cfl_reduce
    constexpr int TPB = 256;
    const int total_cells = n_leaves * GPU_NB * GPU_NB * GPU_NB;
    const int nblocks = (total_cells + TPB - 1) / TPB;
    k_cfl_reduce<<<nblocks, TPB, 0, stream>>>(d_metas, n_leaves, cfl, d_dt_bits);
    CUDA_CHECK(cudaGetLastError());

    // Convert bits → device double (avoids extra type-pun on host)
    k_bits_to_double<<<1, 1, 0, stream>>>(d_dt_bits, d_dt);
    CUDA_CHECK(cudaGetLastError());

    // For the sync return path, wait and copy dt to host
    if (!stream) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    unsigned long long bits = 0;
    CUDA_CHECK(cudaMemcpy(&bits, d_dt_bits, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    return ull_to_double(bits);
}
