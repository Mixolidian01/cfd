// D6: GPU P1 radiation transport.
//
// CG solve for −∇·(D∇G) + κG = κ·a·T⁴  (NB³ interior unknowns per leaf).
// cuBLAS HOST pointer mode provides synchronous scalar results for dot/nrm2.
// Ghost cells live in the full NCELL layout; interior is extracted to compact
// NB³ arrays for CG arithmetic, then written back.

#include "cuda/gpu_p1.cuh"
#include "physics/p1_radiation.hpp"
#include "mesh/cell_block.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>

static constexpr int NB3 = NB * NB * NB;
static constexpr int TPB = 256;

// ── compact ↔ ghost layout helpers ───────────────────────────────────────────
__global__ static void k_put_g_interior(double* __restrict__ d23,
                                         const double* __restrict__ d3) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    d23[cell_idx(i + NG, j + NG, k + NG)] = d3[tid];
}

__global__ static void k_get_g_interior(double* __restrict__ d3,
                                         const double* __restrict__ d23) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    d3[tid] = d23[cell_idx(i + NG, j + NG, k + NG)];
}

// ── Ghost fill for G ──────────────────────────────────────────────────────────
// bc_type 0: all-periodic.
// bc_type 1: Dirichlet x-faces (G_left, G_right); y,z periodic.
// Always reads from INTERIOR cells → no ghost-to-ghost dependency.
__global__ static void k_fill_g_ghosts(double* __restrict__ d,
                                        int bc_type,
                                        double G_left, double G_right) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NCELL) return;

    int i = tid % NB2, jk = tid / NB2, j = jk % NB2, k = jk / NB2;
    if (i >= NG && i < NG+NB && j >= NG && j < NG+NB && k >= NG && k < NG+NB)
        return;  // interior — skip

    // Clamp j,k source to interior range for edge/corner ghosts.
    int sj = (j < NG) ? NG : (j >= NG+NB) ? NG+NB-1 : j;
    int sk = (k < NG) ? NG : (k >= NG+NB) ? NG+NB-1 : k;

    if (bc_type == 1 && (i < NG || i >= NG+NB)) {
        // Dirichlet x-face: mirror about face.
        double G_face = (i < NG) ? G_left : G_right;
        int ii = (i < NG) ? (2*NG - 1 - i) : (2*(NG+NB) - 1 - i);
        d[tid] = 2.0 * G_face - d[cell_idx(ii, sj, sk)];
        return;
    }

    // Periodic in x (also handles bc_type=1 for y,z ghost cells):
    int si = i;
    if (i < NG)       si = i + NB;
    else if (i >= NG+NB) si = i - NB;
    if (j < NG)       sj = j + NB;
    else if (j >= NG+NB) sj = j - NB;
    if (k < NG)       sk = k + NB;
    else if (k >= NG+NB) sk = k - NB;

    d[tid] = d[cell_idx(si, sj, sk)];
}

// ── Helmholtz stencil: (−D∇² + κ)G  →  compact NB³ ─────────────────────────
__global__ static void k_p1_stencil(double* __restrict__ out3,
                                     const double* __restrict__ in23,
                                     double D, double kappa, double h2inv) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    int ig = i + NG, jg = j + NG, kg = k + NG;
    double v = in23[cell_idx(ig, jg, kg)];
    double lap = (in23[cell_idx(ig+1,jg,  kg  )] + in23[cell_idx(ig-1,jg,  kg  )] +
                  in23[cell_idx(ig,  jg+1,kg  )] + in23[cell_idx(ig,  jg-1,kg  )] +
                  in23[cell_idx(ig,  jg,  kg+1)] + in23[cell_idx(ig,  jg,  kg-1)] -
                  6.0 * v) * h2inv;
    out3[tid] = -D * lap + kappa * v;
}

// ── RHS: κ·a·T⁴ + Dirichlet correction ──────────────────────────────────────
// For homogeneous-Dirichlet CG, the actual Dirichlet BC is moved to the RHS:
//   b[i=0]   += 2*(D/h²)*G_left    (x=0 face, each cell in the x=0 plane)
//   b[i=N-1] += 2*(D/h²)*G_right   (x=L face)
// For periodic BC (bc_type=0): no correction.
__global__ static void k_p1_rhs(double* __restrict__ rhs3,
                                  const double* __restrict__ Q,
                                  double kappa, double a_rad,
                                  double c_light, double R_spec,
                                  int bc_type, double G_left, double G_right,
                                  double D, double h2inv) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    int f = cell_idx(i + NG, j + NG, k + NG);
    double rho = Q[0 * NCELL + f];
    double ru  = Q[1 * NCELL + f], rv = Q[2 * NCELL + f], rw = Q[3 * NCELL + f];
    double E   = Q[4 * NCELL + f];
    double T   = p1_T(rho, E, ru, rv, rw, R_spec);
    RadiationParams p; p.kappa=kappa; p.a_rad=a_rad; p.c_light=c_light; p.R_spec=R_spec;
    double b = kappa * p1_emission(T, p);
    if (bc_type == 1) {
        if (i == 0)    b += 2.0 * D * h2inv * G_left;
        if (i == NB-1) b += 2.0 * D * h2inv * G_right;
    }
    rhs3[tid] = b;
}

// ── Energy coupling: Q[4] += κ(aT⁴ − G)·dt ──────────────────────────────────
__global__ static void k_p1_couple(double* __restrict__ Q,
                                    const double* __restrict__ d_G,
                                    double dt, double kappa, double a_rad,
                                    double c_light, double R_spec) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NB3) return;
    int i = tid % NB, j = (tid / NB) % NB, k = tid / (NB * NB);
    int f = cell_idx(i + NG, j + NG, k + NG);
    double rho = Q[0 * NCELL + f];
    double ru  = Q[1 * NCELL + f], rv = Q[2 * NCELL + f], rw = Q[3 * NCELL + f];
    double E   = Q[4 * NCELL + f];
    double T   = p1_T(rho, E, ru, rv, rw, R_spec);
    RadiationParams p; p.kappa=kappa; p.a_rad=a_rad; p.c_light=c_light; p.R_spec=R_spec;
    double G_eq = p1_emission(T, p);
    // Matter gains energy = κ(G − G_eq)·dt: when G>G_eq, radiation heats matter;
    // when G<G_eq, matter emits more than it absorbs and cools.
    Q[4 * NCELL + f] += kappa * (d_G[f] - G_eq) * dt;
}

// ── CG vector operations (compact NB³) ───────────────────────────────────────
__global__ static void k_axpy_n(double* __restrict__ y, double alpha,
                                  const double* __restrict__ x, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) y[tid] += alpha * x[tid];
}

__global__ static void k_copy_n(double* __restrict__ dst,
                                  const double* __restrict__ src, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = src[tid];
}

__global__ static void k_r_plus_beta_p(double* __restrict__ p,
                                         double beta,
                                         const double* __restrict__ r, int n) {
    // p = r + beta*p
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) p[tid] = r[tid] + beta * p[tid];
}

// ── GPU-resident CG per leaf ──────────────────────────────────────────────────
// Solves (−D∇² + κ)G = b  where b already includes the Dirichlet correction
// (k_p1_rhs adds 2D/h² × G_face at boundary cells for bc_type=1).
//
// OPERATOR CONVENTION: inside CG, the Helmholtz stencil is applied with
// HOMOGENEOUS Dirichlet (G_face=0).  This makes the matrix A self-adjoint
// with the Dirichlet correction absorbed into b.  The actual inhomogeneous
// BC values are only restored to d_G at the end (for coupling/ghost reads).
//
// d_G (ghost-padded NCELL): interior holds initial guess (zero from build());
// solution on exit.  d_r, d_p, d_Ap, d_x3: compact NB³ work arrays.
// cuBLAS HOST pointer mode: dot/nrm2 synchronise implicitly per call.
static void p1_cg(double* d_G, const double* d_b,
                  double* d_r, double* d_p, double* d_Ap, double* d_x3,
                  double D, double kappa, double h,
                  int bc_type, double G_left, double G_right,
                  cublasHandle_t cublas,
                  int max_iter, double tol,
                  int* out_iters, double* out_rel_res,
                  cudaStream_t stream)
{
    const int n    = NB3;
    const int blk  = (n + TPB - 1) / TPB;
    const int blkN = (NCELL + TPB - 1) / TPB;
    const double h2inv = 1.0 / (h * h);

    cublasSetStream(cublas, stream);

    // x0 from d_G interior (zero from build()).
    k_get_g_interior<<<blk, TPB, 0, stream>>>(d_x3, d_G);

    // r0 = b − A·x0 with HOMOGENEOUS Dirichlet (G_face=0 for stencil).
    // Since x0=0 and A is linear, A·x0=0 with homo-Dirichlet → r0 = b.
    // But for generality: fill ghosts with 0 BCs and apply stencil.
    k_fill_g_ghosts<<<blkN, TPB, 0, stream>>>(d_G, bc_type, 0.0, 0.0);
    k_p1_stencil<<<blk, TPB, 0, stream>>>(d_Ap, d_G, D, kappa, h2inv);
    k_copy_n<<<blk, TPB, 0, stream>>>(d_r, d_b, n);
    k_axpy_n<<<blk, TPB, 0, stream>>>(d_r, -1.0, d_Ap, n);
    k_copy_n<<<blk, TPB, 0, stream>>>(d_p, d_r, n);

    double rr = 0.0;
    cublasDdot(cublas, n, d_r, 1, d_r, 1, &rr);
    double r0_norm = std::sqrt(rr);

    if (r0_norm < 1e-30) {
        // x0 already satisfies Ax=b to machine precision.
        k_put_g_interior<<<blk, TPB, 0, stream>>>(d_G, d_x3);
        k_fill_g_ghosts<<<blkN, TPB, 0, stream>>>(d_G, bc_type, G_left, G_right);
        *out_iters = 0; *out_rel_res = 0.0;
        return;
    }

    double tol_abs = tol * r0_norm;

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        if (std::sqrt(rr) <= tol_abs) break;

        // Ap = A·p with HOMOGENEOUS Dirichlet: p lives in correction space.
        k_put_g_interior<<<blk, TPB, 0, stream>>>(d_G, d_p);
        k_fill_g_ghosts<<<blkN, TPB, 0, stream>>>(d_G, bc_type, 0.0, 0.0);
        k_p1_stencil<<<blk, TPB, 0, stream>>>(d_Ap, d_G, D, kappa, h2inv);

        double pAp = 0.0;
        cublasDdot(cublas, n, d_p, 1, d_Ap, 1, &pAp);
        if (std::fabs(pAp) < 1e-100) break;

        double alpha = rr / pAp;
        k_axpy_n<<<blk, TPB, 0, stream>>>(d_x3,  alpha, d_p,  n);
        k_axpy_n<<<blk, TPB, 0, stream>>>(d_r,  -alpha, d_Ap, n);

        double rr_new = 0.0;
        cublasDdot(cublas, n, d_r, 1, d_r, 1, &rr_new);

        double beta = rr_new / rr;
        rr = rr_new;
        k_r_plus_beta_p<<<blk, TPB, 0, stream>>>(d_p, beta, d_r, n);
    }

    k_put_g_interior<<<blk, TPB, 0, stream>>>(d_G, d_x3);
    k_fill_g_ghosts<<<blkN, TPB, 0, stream>>>(d_G, bc_type, G_left, G_right);

    *out_iters   = iter;
    *out_rel_res = (r0_norm > 0) ? std::sqrt(rr) / r0_norm : 0.0;
}

// =============================================================================
// GpuP1List
// =============================================================================

GpuP1List::GpuP1List() { cublasCreate(&cublas_); }

GpuP1List::~GpuP1List() {
    if (d_metas) cudaFree(d_metas);
    for (auto& [blk, ptr] : g_map_) cudaFree(ptr);
    if (cublas_) cublasDestroy(cublas_);
}

void GpuP1List::build(const BlockTree& tree, const GpuPool& pool, int bc_type) {
    auto leaves = tree.leaf_indices();
    n_leaves = (int)leaves.size();

    for (int li : leaves) {
        const CellBlock* blk = tree.nodes[li].block.get();
        if (!blk || g_map_.count(blk)) continue;
        double* ptr = nullptr;
        cudaMalloc(&ptr, NCELL * sizeof(double));
        cudaMemset(ptr, 0, NCELL * sizeof(double));
        g_map_[blk] = ptr;
    }

    std::vector<GpuP1LeafMeta> h_metas(n_leaves);
    for (int li_idx = 0; li_idx < n_leaves; ++li_idx) {
        int li = leaves[li_idx];
        const CellBlock* blk = tree.nodes[li].block.get();
        auto& m      = h_metas[li_idx];
        m.d_Q        = pool.d_Q(blk);
        m.d_G        = g_map_.at(blk);
        m.h          = blk->h;
        m.bc_type    = (int8_t)bc_type;
        m.G_left     = 0.0;
        m.G_right    = 0.0;
        for (int f = 0; f < 6; ++f) m.d_Gnb[f] = nullptr;
    }

    if (d_metas) cudaFree(d_metas);
    cudaMalloc(&d_metas, n_leaves * sizeof(GpuP1LeafMeta));
    cudaMemcpy(d_metas, h_metas.data(),
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyHostToDevice);
}

void GpuP1List::set_bc(double G_left, double G_right) {
    std::vector<GpuP1LeafMeta> h_metas(n_leaves);
    cudaMemcpy(h_metas.data(), d_metas,
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyDeviceToHost);
    for (auto& m : h_metas) { m.G_left = G_left; m.G_right = G_right; }
    cudaMemcpy(d_metas, h_metas.data(),
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyHostToDevice);
}

void GpuP1List::exec_ghost_g(cudaStream_t stream) const {
    std::vector<GpuP1LeafMeta> h_metas(n_leaves);
    cudaMemcpy(h_metas.data(), d_metas,
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyDeviceToHost);
    for (auto& m : h_metas)
        k_fill_g_ghosts<<<(NCELL+TPB-1)/TPB, TPB, 0, stream>>>(
            m.d_G, (int)m.bc_type, m.G_left, m.G_right);
}

GpuP1CgResult GpuP1List::exec_cg(RadiationParams params, double /*h_unused*/,
                                   int max_iter, double tol, cudaStream_t stream) {
    std::vector<GpuP1LeafMeta> h_metas(n_leaves);
    cudaMemcpy(h_metas.data(), d_metas,
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyDeviceToHost);

    const double D     = p1_diffusion(params);
    const double kappa = params.kappa;

    double *d_r = nullptr, *d_p = nullptr, *d_Ap = nullptr, *d_x3 = nullptr;
    cudaMalloc(&d_r,  NB3 * sizeof(double));
    cudaMalloc(&d_p,  NB3 * sizeof(double));
    cudaMalloc(&d_Ap, NB3 * sizeof(double));
    cudaMalloc(&d_x3, NB3 * sizeof(double));
    double* d_rhs = nullptr;
    cudaMalloc(&d_rhs, NB3 * sizeof(double));

    GpuP1CgResult res{0, 0.0};

    for (auto& m : h_metas) {
        const double h2inv = 1.0 / (m.h * m.h);
        k_p1_rhs<<<(NB3+TPB-1)/TPB, TPB, 0, stream>>>(
            d_rhs, m.d_Q, kappa, params.a_rad, params.c_light, params.R_spec,
            (int)m.bc_type, m.G_left, m.G_right, D, h2inv);

        int iters = 0; double rel_res = 0.0;
        p1_cg(m.d_G, d_rhs,
              d_r, d_p, d_Ap, d_x3,
              D, kappa, m.h,
              (int)m.bc_type, m.G_left, m.G_right,
              cublas_, max_iter, tol,
              &iters, &rel_res, stream);

        // cuBLAS host-mode already synchronised per call; no extra sync needed.
        res.max_iters   = std::max(res.max_iters, iters);
        res.max_rel_res = std::max(res.max_rel_res, rel_res);
    }

    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap); cudaFree(d_x3); cudaFree(d_rhs);
    return res;
}

void GpuP1List::exec_couple(double dt, RadiationParams params,
                             cudaStream_t stream) const {
    std::vector<GpuP1LeafMeta> h_metas(n_leaves);
    cudaMemcpy(h_metas.data(), d_metas,
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyDeviceToHost);
    for (auto& m : h_metas)
        k_p1_couple<<<(NB3+TPB-1)/TPB, TPB, 0, stream>>>(
            m.d_Q, m.d_G, dt, params.kappa, params.a_rad, params.c_light, params.R_spec);
}

void GpuP1List::upload_g(const std::vector<double>& h_G) const {
    std::vector<GpuP1LeafMeta> h_metas(n_leaves);
    cudaMemcpy(h_metas.data(), d_metas,
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyDeviceToHost);
    double* d_tmp = nullptr;
    cudaMalloc(&d_tmp, NB3 * sizeof(double));
    for (int li = 0; li < n_leaves; ++li) {
        cudaMemcpy(d_tmp, h_G.data() + li * NB3, NB3 * sizeof(double),
                   cudaMemcpyHostToDevice);
        k_put_g_interior<<<(NB3+TPB-1)/TPB, TPB>>>(h_metas[li].d_G, d_tmp);
    }
    cudaDeviceSynchronize();
    cudaFree(d_tmp);
}

void GpuP1List::download_g(std::vector<double>& h_G) const {
    h_G.resize(n_leaves * NB3);
    std::vector<GpuP1LeafMeta> h_metas(n_leaves);
    cudaMemcpy(h_metas.data(), d_metas,
               n_leaves * sizeof(GpuP1LeafMeta), cudaMemcpyDeviceToHost);
    double* d_tmp = nullptr;
    cudaMalloc(&d_tmp, NB3 * sizeof(double));
    for (int li = 0; li < n_leaves; ++li) {
        k_get_g_interior<<<(NB3+TPB-1)/TPB, TPB>>>(d_tmp, h_metas[li].d_G);
        cudaDeviceSynchronize();
        cudaMemcpy(h_G.data() + li * NB3, d_tmp, NB3 * sizeof(double),
                   cudaMemcpyDeviceToHost);
    }
    cudaFree(d_tmp);
}
