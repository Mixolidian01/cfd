// src/cuda/gpu_adjoint_rhs.cu — G2: GPU adjoint convective RHS.
// PCM reconstruction, full HLLC-ES adjoint (frozen-lam base state).
// One thread-block (NB2²=144 threads) per leaf; shared-mem prim cache.

#include "cuda/gpu_adjoint_rhs.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_constants.cuh"
#include "physics/adjoint_hllc.hpp"   // adjoint_hllces_flux, acc_adj_cons_to_prim
#include <cassert>
#include <cstring>

// ── Shared-memory prim cache kernel ──────────────────────────────────────────
// Layout: sh_prim[v * GPU_NCELL + f]  (SoA, v=0..4)
// Size: 5 * 1728 * 8 = 69120 bytes — requires opt-in carveout.
__global__
__launch_bounds__(GPU_NB2 * GPU_NB2)
static void k_adjoint_rhs(const GpuAdjMeta* __restrict__ metas,
                           const double* __restrict__ d_lam_in,  // [NL*NVAR*NCELL]
                           double* __restrict__       d_lam_out) // [NL*NVAR*NCELL]
{
    extern __shared__ double sh_prim[];  // 5 * GPU_NCELL doubles

    const int li  = blockIdx.x;
    const auto& m = metas[li];
    const double* __restrict__ d_Q_leaf  = m.d_Q;
    const double* __restrict__ d_li_in   = d_lam_in  + (size_t)li * GPU_NVAR * GPU_NCELL;
    double*       __restrict__ d_li_out  = d_lam_out  + (size_t)li * GPU_NVAR * GPU_NCELL;

    // Phase 0: zero output, load prim cache (cooperative, 144 threads × 12 cells = 1728)
    for (int f = threadIdx.x; f < GPU_NCELL; f += blockDim.x) {
        const double rho = d_Q_leaf[0 * GPU_NCELL + f];
        const double u_  = d_Q_leaf[1 * GPU_NCELL + f] / rho;
        const double v_  = d_Q_leaf[2 * GPU_NCELL + f] / rho;
        const double w_  = d_Q_leaf[3 * GPU_NCELL + f] / rho;
        const double ke  = 0.5 * rho * (u_*u_ + v_*v_ + w_*w_);
        const double p_  = (GPU_GAMMA - 1.0) * (d_Q_leaf[4 * GPU_NCELL + f] - ke);
        sh_prim[0 * GPU_NCELL + f] = rho;
        sh_prim[1 * GPU_NCELL + f] = u_;
        sh_prim[2 * GPU_NCELL + f] = v_;
        sh_prim[3 * GPU_NCELL + f] = w_;
        sh_prim[4 * GPU_NCELL + f] = p_;
        for (int v = 0; v < GPU_NVAR; ++v)
            d_li_out[v * GPU_NCELL + f] = 0.0;
    }
    __syncthreads();

    // Phase 1: each thread handles one (a,b) column across all three axes.
    if (threadIdx.x >= GPU_NB2 * GPU_NB2) return;
    const int a  = threadIdx.x % GPU_NB2;
    const int b  = threadIdx.x / GPU_NB2;
    const double ihs[3] = {1.0/m.hx, 1.0/m.hy, 1.0/m.hz};

    for (int ax = 0; ax < 3; ++ax) {
        const double ih = ihs[ax];
        for (int n = 0; n <= GPU_NB2 - 2; ++n) {
            int Li, Ri;
            if      (ax == 0) { Li = gpu_cell_idx(n,   a, b); Ri = gpu_cell_idx(n+1, a, b); }
            else if (ax == 1) { Li = gpu_cell_idx(a,   n, b); Ri = gpu_cell_idx(a, n+1, b); }
            else              { Li = gpu_cell_idx(a,   b, n); Ri = gpu_cell_idx(a, b, n+1); }

            // Interior predicates for this face's two cells
            const int nL = n,   nR = n + 1;
            const bool liI = (nL >= GPU_NG) && (nL < GPU_NB + GPU_NG);
            const bool riI = (nR >= GPU_NG) && (nR < GPU_NB + GPU_NG);
            // Also check transverse interior for axis != 0 case
            // (a, b already iterate all GPU_NB2 values; only interior a/b contribute)
            const bool aI = (a >= GPU_NG) && (a < GPU_NB + GPU_NG);
            const bool bI = (b >= GPU_NG) && (b < GPU_NB + GPU_NG);
            const bool transverse_ok = (ax == 0) ? (aI && bI)
                                     : (ax == 1) ? (aI && bI)
                                     :              (aI && bI);
            if (!transverse_ok) continue;
            if (!liI && !riI) continue;

            // Seed l_F from λ (adjoint of rhs[Li] -= ih*F; rhs[Ri] += ih*F)
            double l_F[GPU_NVAR] = {};
            for (int v = 0; v < GPU_NVAR; ++v) {
                if (liI) l_F[v] -= ih * d_li_in[v * GPU_NCELL + Li];
                if (riI) l_F[v] += ih * d_li_in[v * GPU_NCELL + Ri];
            }

            // Build Prim structs from shared-memory prim cache
            Prim primL{}, primR{};
            primL.rho = sh_prim[0*GPU_NCELL+Li]; primL.u=sh_prim[1*GPU_NCELL+Li];
            primL.v   = sh_prim[2*GPU_NCELL+Li]; primL.w=sh_prim[3*GPU_NCELL+Li];
            primL.p   = sh_prim[4*GPU_NCELL+Li];
            primL.gamma_m = GPU_GAMMA; primL.p_inf_m = 0.0;
            primL.T = primL.p / (primL.rho * (GPU_GAMMA - 1.0));
            primL.c = sqrt(GPU_GAMMA * primL.p / primL.rho);

            primR.rho = sh_prim[0*GPU_NCELL+Ri]; primR.u=sh_prim[1*GPU_NCELL+Ri];
            primR.v   = sh_prim[2*GPU_NCELL+Ri]; primR.w=sh_prim[3*GPU_NCELL+Ri];
            primR.p   = sh_prim[4*GPU_NCELL+Ri];
            primR.gamma_m = GPU_GAMMA; primR.p_inf_m = 0.0;
            primR.T = primR.p / (primR.rho * (GPU_GAMMA - 1.0));
            primR.c = sqrt(GPU_GAMMA * primR.p / primR.rho);

            // Frozen eigenvalue: lam = max(|un_L|+c_L, |un_R|+c_R)
            const double unL = (ax==0) ? primL.u : (ax==1) ? primL.v : primL.w;
            const double unR = (ax==0) ? primR.u : (ax==1) ? primR.v : primR.w;
            const double lam = fmax(fabs(unL) + primL.c, fabs(unR) + primR.c);

            // Adjoint of HLLC-ES flux: scatter l_F into prim adjoints
            double l_pL[GPU_NVAR] = {}, l_pR[GPU_NVAR] = {};
            if (ax == 0)
                adjoint_hllces_flux<Axis::X, true>(primL, primR, l_F, lam, l_pL, l_pR);
            else if (ax == 1)
                adjoint_hllces_flux<Axis::Y, true>(primL, primR, l_F, lam, l_pL, l_pR);
            else
                adjoint_hllces_flux<Axis::Z, true>(primL, primR, l_F, lam, l_pL, l_pR);

            // Convert prim adjoints to conserved via Jacobian transpose
            double l_cL[GPU_NVAR] = {}, l_cR[GPU_NVAR] = {};
            acc_adj_cons_to_prim(primL, l_pL, l_cL);
            acc_adj_cons_to_prim(primR, l_pR, l_cR);

            // Accumulate into output (atomic needed: multiple threads write Li/Ri)
            for (int v = 0; v < GPU_NVAR; ++v) {
                atomicAdd(&d_li_out[v * GPU_NCELL + Li], l_cL[v]);
                atomicAdd(&d_li_out[v * GPU_NCELL + Ri], l_cR[v]);
            }
        }
    }
}

// ── GpuAdjointRhsList ─────────────────────────────────────────────────────────

GpuAdjointRhsList::~GpuAdjointRhsList() {
    if (d_metas) { cudaFree(d_metas); d_metas = nullptr; }
}

void GpuAdjointRhsList::build(const BlockTree& tree, const GpuPool& pool) {
    const auto& leaves = tree.leaf_indices();
    n_leaves = (int)leaves.size();
    if (n_leaves == 0) return;

    if (d_metas) { cudaFree(d_metas); d_metas = nullptr; }

    std::vector<GpuAdjMeta> h_metas(n_leaves);
    for (int ii = 0; ii < n_leaves; ++ii) {
        const int li     = leaves[ii];
        const auto& nd   = tree.nodes[li];
        const CellBlock& blk = *nd.block;
        auto& mt         = h_metas[ii];
        mt.d_Q = pool.d_Q(&blk);
        mt.hx  = blk.h; mt.hy = blk.hy; mt.hz = blk.hz;
    }
    CUDA_CHECK(cudaMalloc((void**)&d_metas,
                          (size_t)n_leaves * sizeof(GpuAdjMeta)));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas.data(),
                          (size_t)n_leaves * sizeof(GpuAdjMeta),
                          cudaMemcpyHostToDevice));
}

void GpuAdjointRhsList::exec(const double* d_lam_in, double* d_lam_out,
                              cudaStream_t s) const {
    if (!n_leaves) return;
    const size_t shmem = (size_t)GPU_NVAR * GPU_NCELL * sizeof(double); // 69120 B
    // Opt-in for >48 KB shared memory
    CUDA_CHECK(cudaFuncSetAttribute(k_adjoint_rhs,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)shmem));
    k_adjoint_rhs<<<n_leaves, GPU_NB2 * GPU_NB2, shmem, s>>>(
        d_metas, d_lam_in, d_lam_out);
}

void GpuAdjointRhsList::exec_sync(const std::vector<double>& h_lam,
                                   std::vector<double>&       h_out) const {
    const size_t sz = (size_t)n_leaves * GPU_NVAR * GPU_NCELL * sizeof(double);
    double *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  sz));
    CUDA_CHECK(cudaMalloc((void**)&d_out, sz));
    CUDA_CHECK(cudaMemcpy(d_in, h_lam.data(), sz, cudaMemcpyHostToDevice));
    exec(d_in, d_out, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    h_out.resize(n_leaves * GPU_NVAR * GPU_NCELL);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sz, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
}
