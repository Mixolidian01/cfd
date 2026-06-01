#include "cuda/gpu_metrics.cuh"
#include "cuda/gpu_check.cuh"
#include <cooperative_groups.h>
#include <cmath>
#include <cstdlib>
#include <vector>

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

// ── k_surface_forces ─────────────────────────────────────────────────────────
// gridDim = (n_entries+255)/256, blockDim = 256
__global__ void k_surface_forces(
    const GpuSurfaceEntry* __restrict__ entries, int n_entries,
    double* __restrict__ acc,
    float mu,
    double ref_x, double ref_y, double ref_z)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_entries) return;

    const GpuSurfaceEntry& e = entries[gid];

    // Image-point state via trilinear interpolation
    double Q_I[GPU_NVAR] = {};
    for (int s = 0; s < 8; ++s) {
        if (e.w[s] == 0.f || e.stencil[s] == nullptr) continue;
        for (int v = 0; v < GPU_NVAR; ++v)
            Q_I[v] += (double)e.w[s] * e.stencil[s][(size_t)v * GPU_NCELL];
    }

    const double rho_I   = Q_I[0];
    const double inv_rho = 1.0 / max(rho_I, 1e-30);
    const double u_I     = Q_I[1] * inv_rho;
    const double v_I     = Q_I[2] * inv_rho;
    const double w_I     = Q_I[3] * inv_rho;
    const double ke_I    = 0.5 * (Q_I[1]*Q_I[1] + Q_I[2]*Q_I[2] + Q_I[3]*Q_I[3]) * inv_rho;
    const double p_w     = (GPU_GAMMA - 1.0) * (Q_I[4] - ke_I);

    const double A  = (double)e.h * (double)e.h;
    const double nx = (double)e.nx, ny = (double)e.ny, nz = (double)e.nz;

    // Pressure force: fp = -p_w * n * A
    const double fpx = -p_w * nx * A;
    const double fpy = -p_w * ny * A;
    const double fpz = -p_w * nz * A;

    // Wall shear: tangential velocity diff
    const double u_rel = u_I - (double)e.u_wall;
    const double v_rel = v_I - (double)e.v_wall;
    const double w_rel = w_I - (double)e.w_wall;
    const double un    = u_rel * nx + v_rel * ny + w_rel * nz;
    const double ut_x  = u_rel - un * nx;
    const double ut_y  = v_rel - un * ny;
    const double ut_z  = w_rel - un * nz;
    const double inv_d = 1.0 / max((double)e.d, 1e-30);
    const double fvx   = (double)mu * ut_x * inv_d * A;
    const double fvy   = (double)mu * ut_y * inv_d * A;
    const double fvz   = (double)mu * ut_z * inv_d * A;

    const double fx = fpx + fvx;
    const double fy = fpy + fvy;
    const double fz = fpz + fvz;

    // Moment
    const double rx = (double)e.cx - ref_x;
    const double ry = (double)e.cy - ref_y;
    const double rz = (double)e.cz - ref_z;

    atomicAdd(&acc[0], fx);
    atomicAdd(&acc[1], fy);
    atomicAdd(&acc[2], fz);
    atomicAdd(&acc[3], ry * fz - rz * fy);
    atomicAdd(&acc[4], rz * fx - rx * fz);
    atomicAdd(&acc[5], rx * fy - ry * fx);
}

// ── GpuSurfaceList ───────────────────────────────────────────────────────────
GpuSurfaceList::~GpuSurfaceList() {
    if (d_entries) { cudaFree(d_entries); d_entries = nullptr; }
    if (d_acc)     { cudaFree(d_acc);     d_acc     = nullptr; }
    if (h_acc)     { cudaFreeHost(h_acc); h_acc     = nullptr; }
}

void GpuSurfaceList::exec(cudaStream_t s, float mu) const {
    if (!n_entries) return;
    CUDA_CHECK(cudaMemsetAsync(d_acc, 0, 6 * sizeof(double), s));
    k_surface_forces<<<(n_entries + 255) / 256, 256, 0, s>>>(
        d_entries, n_entries, d_acc, mu,
        ref_point[0], ref_point[1], ref_point[2]);
    CUDA_CHECK(cudaMemcpyAsync(h_acc, d_acc, 6 * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
}

void GpuSurfaceList::build(const GpuIbmList& ibm, const SolverConfig::SurfaceConfig& cfg) {
    name      = cfg.name;
    ref_point = cfg.ref_point;
    rho_ref   = cfg.rho_ref;
    u_ref     = cfg.u_ref;
    A_ref     = cfg.A_ref;

    const int NC = GPU_NCELL;
    const int NL = ibm.n_leaves;

    // D2H downloads
    std::vector<int8_t> h_ct(NL * NC);
    std::vector<float>  h_sdf(NL * NC);
    std::vector<float>  h_wn(NL * NC * 3);
    CUDA_CHECK(cudaMemcpy(h_ct.data(),  ibm.d_cell_type_pool,
                          (size_t)NL*NC*sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sdf.data(), ibm.d_sdf_pool,
                          (size_t)NL*NC*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wn.data(),  ibm.d_wnorm_pool,
                          (size_t)NL*NC*3*sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<GhostEntry> h_ghosts(ibm.n_ghosts);
    CUDA_CHECK(cudaMemcpy(h_ghosts.data(), ibm.d_ghosts,
                          (size_t)ibm.n_ghosts*sizeof(GhostEntry), cudaMemcpyDeviceToHost));

    std::vector<GpuIbmMeta> h_ibm_metas(NL);
    CUDA_CHECK(cudaMemcpy(h_ibm_metas.data(), ibm.d_metas,
                          (size_t)NL*sizeof(GpuIbmMeta), cudaMemcpyDeviceToHost));

    std::vector<GpuSurfaceEntry> entries;
    int ghost_idx = 0;

    for (int li = 0; li < NL; ++li) {
        const GpuIbmMeta& m = h_ibm_metas[li];
        for (int flat = 0; flat < NC; ++flat) {
            if (h_ct[(size_t)li * NC + flat] != 2) continue;  // 2 = IBM_GHOST

            const int k_ijk = flat / (GPU_NB2 * GPU_NB2);
            const int j_ijk = (flat / GPU_NB2) % GPU_NB2;
            const int i_ijk = flat % GPU_NB2;
            const float cx = m.ox + (i_ijk + 0.5f) * m.hx;
            const float cy = m.oy + (j_ijk + 0.5f) * m.hy;
            const float cz = m.oz + (k_ijk + 0.5f) * m.hz;

            const float sdf_v = h_sdf[(size_t)li * NC + flat];
            const float nx_v  = h_wn[(size_t)(li * NC * 3 + 0 * NC + flat)];
            const float ny_v  = h_wn[(size_t)(li * NC * 3 + 1 * NC + flat)];
            const float nz_v  = h_wn[(size_t)(li * NC * 3 + 2 * NC + flat)];

            GpuSurfaceEntry se = {};
            const GhostEntry& ge = h_ghosts[ghost_idx++];
            se.ghost_ptr = ge.ghost_ptr;
            for (int s = 0; s < 8; ++s) { se.stencil[s] = ge.stencil[s]; se.w[s] = ge.w[s]; }
            se.nx = nx_v; se.ny = ny_v; se.nz = nz_v;
            se.d  = fabsf(sdf_v);
            se.h  = m.hx;
            se.cx = cx; se.cy = cy; se.cz = cz;
            se.wall_bc = ibm.wall_bc;
            se.u_wall  = ibm.u_wall; se.v_wall = ibm.v_wall; se.w_wall = ibm.w_wall;
            entries.push_back(se);
        }
    }

    // Cleanup old device buffers
    if (d_entries) { cudaFree(d_entries); d_entries = nullptr; }
    if (d_acc)     { cudaFree(d_acc);     d_acc     = nullptr; }
    if (h_acc)     { cudaFreeHost(h_acc); h_acc     = nullptr; }

    n_entries = (int)entries.size();
    if (!n_entries) return;

    CUDA_CHECK(cudaMalloc(&d_entries, n_entries * sizeof(GpuSurfaceEntry)));
    CUDA_CHECK(cudaMemcpy(d_entries, entries.data(),
                          n_entries * sizeof(GpuSurfaceEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_acc, 6 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_acc, 6 * sizeof(double)));
}
