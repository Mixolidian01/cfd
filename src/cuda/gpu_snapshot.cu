// GPU slice extraction (Option A) and metric reduction (Option C).
// See include/gpu_snapshot.hpp for architecture notes.

#include "gpu_snapshot.hpp"
#include "gpu_constants.cuh"
#include "gpu_check.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstring>
#include <cstdlib>

namespace cg = cooperative_groups;

// ── Internal opaque CUDA state ────────────────────────────────────────────────
struct SnapImpl {
    cudaStream_t      stream  = nullptr;
    SnapLeafMeta*     d_metas = nullptr;   // device copy of h_metas [max_leaves]
    float*            d_slice = nullptr;   // device-mapped mirror of h_slice
    GpuBlockMetrics*  d_hmet  = nullptr;   // device-mapped mirror of h_metrics
    float*            d_volume = nullptr;  // device-mapped mirror of h_volume
    int               max_leaves = 0;
};

// VOL_MAX_N: maximum N for the volume grid; fixed at compile time.
// Pinned allocation is always VOL_MAX_N³ × sizeof(float) = 8 MB.
static constexpr int VOL_MAX_N = 128;

// ── snap_scalar_val ─────────────────────────────────────────────────────────
// Compute the visualised scalar at interior cell (ci, cj, ck) from the flat SoA
// Q array.  Used by both k_extract_slice and k_build_volume.
__device__ static float snap_scalar_val(
    const double* __restrict__ Q,
    int var_id, int ci, int cj, int ck, float h)
{
#define QVAL(v,i,j,k) Q[(v)*GPU_NCELL + gpu_cell_idx((i),(j),(k))]

    const double rho  = QVAL(0, ci, cj, ck);
    const double rhou = QVAL(1, ci, cj, ck);
    const double rhov = QVAL(2, ci, cj, ck);
    const double rhow = QVAL(3, ci, cj, ck);
    const double E    = QVAL(4, ci, cj, ck);

    float val;
    switch (var_id) {
        case 0: val = (float)rho;  break;
        case 1: {
            const double ke = 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
            val = (float)((GPU_GAMMA - 1.0) * (E - ke));
            break;
        }
        case 2: {
            const double ke = 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
            const double p  = (GPU_GAMMA - 1.0) * (E - ke);
            val = (float)(p / (rho * GPU_R_GAS));
            break;
        }
        case 3: {
            const double u = rhou/rho, v = rhov/rho, w = rhow/rho;
            val = (float)sqrt(u*u + v*v + w*w);
            break;
        }
        case 4: val = (float)rhou; break;
        case 5: val = (float)rhov; break;
        case 6: val = (float)rhow; break;
        case 7: val = (float)E;    break;
        case 8: {
            const double ke = 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) / rho;
            const double p  = (GPU_GAMMA - 1.0) * (E - ke);
            const double c  = sqrt(GPU_GAMMA * p / rho);
            const double u  = rhou/rho, v = rhov/rho, w = rhow/rho;
            val = (float)(sqrt(u*u + v*v + w*w) / max(c, 1e-30));
            break;
        }
        case 9: {
            const double ih2 = 0.5 / h;
            auto vel = [Q](int comp, int i, int j, int k) -> double {
                return Q[(comp+1)*GPU_NCELL + gpu_cell_idx(i,j,k)]
                     / Q[0*GPU_NCELL + gpu_cell_idx(i,j,k)];
            };
            const double wx = (vel(2,ci,cj+1,ck)-vel(2,ci,cj-1,ck))*ih2
                            - (vel(1,ci,cj,ck+1)-vel(1,ci,cj,ck-1))*ih2;
            const double wy = (vel(0,ci,cj,ck+1)-vel(0,ci,cj,ck-1))*ih2
                            - (vel(2,ci+1,cj,ck)-vel(2,ci-1,cj,ck))*ih2;
            const double wz = (vel(1,ci+1,cj,ck)-vel(1,ci-1,cj,ck))*ih2
                            - (vel(0,ci,cj+1,ck)-vel(0,ci,cj-1,ck))*ih2;
            val = (float)sqrt(wx*wx + wy*wy + wz*wz);
            break;
        }
        case 10: {
            const double ih2 = 0.5 / h;
            auto vel = [Q](int comp, int i, int j, int k) -> double {
                return Q[(comp+1)*GPU_NCELL + gpu_cell_idx(i,j,k)]
                     / Q[0*GPU_NCELL + gpu_cell_idx(i,j,k)];
            };
            double A[3][3];
            for (int c = 0; c < 3; ++c) {
                A[c][0] = (vel(c,ci+1,cj,ck)-vel(c,ci-1,cj,ck))*ih2;
                A[c][1] = (vel(c,ci,cj+1,ck)-vel(c,ci,cj-1,ck))*ih2;
                A[c][2] = (vel(c,ci,cj,ck+1)-vel(c,ci,cj,ck-1))*ih2;
            }
            double Qval = 0.0;
            for (int c = 0; c < 3; ++c)
                for (int d = 0; d < 3; ++d)
                    Qval -= 0.5 * A[c][d] * A[d][c];
            val = (float)Qval;
            break;
        }
        case 11: {
            const double ih2 = 0.5 / h;
            const double drx = (QVAL(0,ci+1,cj,ck)-QVAL(0,ci-1,cj,ck))*ih2;
            const double dry = (QVAL(0,ci,cj+1,ck)-QVAL(0,ci,cj-1,ck))*ih2;
            const double drz = (QVAL(0,ci,cj,ck+1)-QVAL(0,ci,cj,ck-1))*ih2;
            val = (float)sqrt(drx*drx + dry*dry + drz*drz);
            break;
        }
        default: val = 0.f;
    }
#undef QVAL
    return val;
}

// ── k_extract_slice ───────────────────────────────────────────────────────────
// One block per leaf; GPU_NB*GPU_NB threads per block.
// Thread t handles cell (a = t%NB, b = t/NB) in the slice plane.
// Writes 0.0f for leaves that don't intersect the slice.
__global__ void k_extract_slice(
    const SnapLeafMeta* __restrict__ metas,
    float*              __restrict__ d_out,   // [n_leaves * NB * NB]
    int var_id, int axis, float slice_phys)
{
    const int li = blockIdx.x;
    const int t  = threadIdx.x;
    const int a  = t % GPU_NB;
    const int b  = t / GPU_NB;

    const SnapLeafMeta& m = metas[li];

    float lo, hi;
    if      (axis == 0) { lo = m.ox; hi = m.ox + GPU_NB * m.h; }
    else if (axis == 1) { lo = m.oy; hi = m.oy + GPU_NB * m.h; }
    else                { lo = m.oz; hi = m.oz + GPU_NB * m.h; }

    if (slice_phys < lo || slice_phys >= hi) {
        d_out[li * GPU_NB * GPU_NB + t] = 0.f;
        return;
    }

    int s = GPU_NG + (int)((slice_phys - lo) / m.h);
    s = max(s, GPU_NG); s = min(s, GPU_NG + GPU_NB - 1);

    const int ia = GPU_NG + a;
    const int ib = GPU_NG + b;
    int ci, cj, ck;
    if      (axis == 0) { ci = s;  cj = ia; ck = ib; }
    else if (axis == 1) { ci = ia; cj = s;  ck = ib; }
    else                { ci = ia; cj = ib; ck = s;  }

    d_out[li * GPU_NB * GPU_NB + t] = snap_scalar_val(m.d_Q, var_id, ci, cj, ck, m.h);
}

// ── k_build_volume ────────────────────────────────────────────────────────────
// One block per leaf; 64 threads per block, each handling NB³/64 = 8 cells.
// Each interior cell scatters its scalar value to the voxel range it covers in
// the N×N×N uniform grid.  Race conditions at C/F boundaries are acceptable for
// a non-critical visualisation use case.
__global__ void k_build_volume(
    const SnapLeafMeta* __restrict__ metas,
    float* __restrict__ d_volume,          // [N*N*N]
    int n_leaves, int N, int var_id, float domain_L)
{
    const int li = blockIdx.x;
    if (li >= n_leaves) return;
    const SnapLeafMeta& m = metas[li];
    const float h     = m.h;
    const float inv_L = 1.0f / domain_L;

    constexpr int N_INT = GPU_NB * GPU_NB * GPU_NB;
    for (int idx = threadIdx.x; idx < N_INT; idx += 64) {
        const int kk = idx / (GPU_NB * GPU_NB);
        const int jj = (idx / GPU_NB) % GPU_NB;
        const int ii = idx % GPU_NB;

        const float cx = m.ox + (ii + 0.5f) * h;
        const float cy = m.oy + (jj + 0.5f) * h;
        const float cz = m.oz + (kk + 0.5f) * h;

        const int vi0 = max(0,   (int)((cx - 0.5f*h) * inv_L * N));
        const int vi1 = min(N-1, (int)((cx + 0.5f*h) * inv_L * N));
        const int vj0 = max(0,   (int)((cy - 0.5f*h) * inv_L * N));
        const int vj1 = min(N-1, (int)((cy + 0.5f*h) * inv_L * N));
        const int vk0 = max(0,   (int)((cz - 0.5f*h) * inv_L * N));
        const int vk1 = min(N-1, (int)((cz + 0.5f*h) * inv_L * N));

        const float val = snap_scalar_val(
            m.d_Q, var_id, GPU_NG + ii, GPU_NG + jj, GPU_NG + kk, h);

        for (int vk = vk0; vk <= vk1; ++vk)
        for (int vj = vj0; vj <= vj1; ++vj)
        for (int vi = vi0; vi <= vi1; ++vi)
            d_volume[vk * N * N + vj * N + vi] = val;
    }
}

// ── k_reduce_metrics ──────────────────────────────────────────────────────────
// One block per leaf; 64 threads per block (8 cells per thread → 512 interior).
// Reduces NB³ interior cells into one GpuBlockMetrics per leaf.
__global__ void k_reduce_metrics(
    const SnapLeafMeta* __restrict__ metas,
    GpuBlockMetrics*    __restrict__ d_out)
{
    const int li = blockIdx.x;
    const SnapLeafMeta& m = metas[li];
    const double h3 = (double)m.h * m.h * m.h;

    double mass = 0, ke = 0, px = 0, py = 0, pz = 0, etot = 0;
    double rho_min = 1e300, rho_max = 0;

    constexpr int T = 64;
    constexpr int N_INT = GPU_NB * GPU_NB * GPU_NB;

    for (int idx = threadIdx.x; idx < N_INT; idx += T) {
        const int kk = idx / (GPU_NB * GPU_NB);
        const int jj = (idx / GPU_NB) % GPU_NB;
        const int ii = idx % GPU_NB;
        const int i = GPU_NG + ii, j = GPU_NG + jj, k = GPU_NG + kk;
        const int flat = gpu_cell_idx(i, j, k);

        const double rho  = m.d_Q[0*GPU_NCELL + flat];
        const double rhou = m.d_Q[1*GPU_NCELL + flat];
        const double rhov = m.d_Q[2*GPU_NCELL + flat];
        const double rhow = m.d_Q[3*GPU_NCELL + flat];
        const double E    = m.d_Q[4*GPU_NCELL + flat];

        mass += rho * h3;
        const double irho = 1.0 / rho;
        ke   += 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) * irho * h3;
        px   += rhou * h3;
        py   += rhov * h3;
        pz   += rhow * h3;
        etot += E * h3;
        rho_min = min(rho_min, rho);
        rho_max = max(rho_max, rho);
    }

    // Warp-level reduction using cooperative groups
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    // Reduce each scalar across the warp
    auto warp_reduce_sum  = [&](double v) {
        for (int off = 16; off > 0; off >>= 1) v += warp.shfl_down(v, off);
        return v;
    };
    auto warp_reduce_min  = [&](double v) {
        for (int off = 16; off > 0; off >>= 1) v = min(v, warp.shfl_down(v, off));
        return v;
    };
    auto warp_reduce_max  = [&](double v) {
        for (int off = 16; off > 0; off >>= 1) v = max(v, warp.shfl_down(v, off));
        return v;
    };

    mass    = warp_reduce_sum(mass);   ke      = warp_reduce_sum(ke);
    px      = warp_reduce_sum(px);     py      = warp_reduce_sum(py);
    pz      = warp_reduce_sum(pz);     etot    = warp_reduce_sum(etot);
    rho_min = warp_reduce_min(rho_min);rho_max = warp_reduce_max(rho_max);

    // Two warps (64 threads) → block-level reduction via shared memory
    __shared__ double smem[8][2];  // [scalar_idx][warp_idx]
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        smem[0][warp_id] = mass;  smem[1][warp_id] = ke;
        smem[2][warp_id] = px;    smem[3][warp_id] = py;   smem[4][warp_id] = pz;
        smem[5][warp_id] = etot;  smem[6][warp_id] = rho_min; smem[7][warp_id] = rho_max;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        GpuBlockMetrics out{};
        out.mass    = smem[0][0] + smem[0][1];
        out.ke      = smem[1][0] + smem[1][1];
        out.px      = smem[2][0] + smem[2][1];
        out.py      = smem[3][0] + smem[3][1];
        out.pz      = smem[4][0] + smem[4][1];
        out.etot    = smem[5][0] + smem[5][1];
        out.rho_min = min(smem[6][0], smem[6][1]);
        out.rho_max = max(smem[7][0], smem[7][1]);
        d_out[li]   = out;
    }
}

// ── GpuSnapshotBuffer methods ─────────────────────────────────────────────────

GpuSnapshotBuffer::GpuSnapshotBuffer() = default;

GpuSnapshotBuffer::~GpuSnapshotBuffer() {
    auto* p = static_cast<SnapImpl*>(impl_);
    if (!p) return;

    if (h_slice)   { cudaFreeHost(h_slice);   h_slice   = nullptr; }
    if (h_metrics) { cudaFreeHost(h_metrics); h_metrics = nullptr; }
    if (h_volume)  { cudaFreeHost(h_volume);  h_volume  = nullptr; }
    if (h_metas)   { free(h_metas);           h_metas   = nullptr; }
    if (p->d_metas){ cudaFree(p->d_metas);    p->d_metas = nullptr; }
    if (p->stream) { cudaStreamDestroy(p->stream); p->stream = nullptr; }
    delete p;
    impl_ = nullptr;
}

void GpuSnapshotBuffer::alloc(int n_leaves_max) {
    // Free existing resources
    auto* p = static_cast<SnapImpl*>(impl_);
    if (p) {
        if (h_slice)   { cudaFreeHost(h_slice);   h_slice   = nullptr; }
        if (h_metrics) { cudaFreeHost(h_metrics); h_metrics = nullptr; }
        if (h_volume)  { cudaFreeHost(h_volume);  h_volume  = nullptr; }
        if (h_metas)   { free(h_metas);           h_metas   = nullptr; }
        if (p->d_metas){ cudaFree(p->d_metas);    p->d_metas = nullptr; }
        if (p->stream) { cudaStreamDestroy(p->stream); p->stream = nullptr; }
        delete p; impl_ = nullptr;
    }

    p = new SnapImpl();
    impl_ = p;
    max_leaves = n_leaves_max;

    const size_t slice_bytes   = (size_t)n_leaves_max * GPU_NB * GPU_NB * sizeof(float);
    const size_t metrics_bytes = (size_t)n_leaves_max * sizeof(GpuBlockMetrics);
    const size_t vol_bytes     = (size_t)VOL_MAX_N * VOL_MAX_N * VOL_MAX_N * sizeof(float);

    // Pinned host-mapped memory: GPU writes directly, CPU reads after sync
    CUDA_CHECK(cudaHostAlloc(&h_slice,   slice_bytes,   cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_metrics, metrics_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_volume,  vol_bytes,     cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&p->d_slice,  h_slice,   0));
    CUDA_CHECK(cudaHostGetDevicePointer(&p->d_hmet,   h_metrics, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&p->d_volume, h_volume,  0));

    // Device-only metadata array (uploaded from CPU each build())
    CUDA_CHECK(cudaMalloc(&p->d_metas, (size_t)n_leaves_max * sizeof(SnapLeafMeta)));
    p->max_leaves = n_leaves_max;

    CUDA_CHECK(cudaStreamCreate(&p->stream));

    // CPU mirror for block descriptors (plain malloc — no CUDA)
    h_metas = static_cast<SnapLeafMeta*>(malloc((size_t)n_leaves_max * sizeof(SnapLeafMeta)));
    n_leaves = 0;
}

// Called from GpuGraphSolver::_upload_snap_metas() — see gpu_graph.cu
// (kept as a free function there; this file provides the buffer primitives)
