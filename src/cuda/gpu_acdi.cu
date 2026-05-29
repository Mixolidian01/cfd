// src/cuda/gpu_acdi.cu — G1: GPU ACDI phi transport kernels.
// Ports phi_rhs() and phi_compression_rhs() from src/schemes/rhs_sensors.cpp.

#include "cuda/gpu_acdi.cuh"
#include "cuda/gpu_check.cuh"
#include <cassert>
#include <vector>

// ── GpuPhiPool ────────────────────────────────────────────────────────────────

GpuPhiPool::~GpuPhiPool() {
    for (double* p : free_list_) cudaFree(p);
    for (auto& kv : ptrs_) cudaFree(kv.second);
}

void GpuPhiPool::alloc(const CellBlock* blk) {
    assert(!ptrs_.count(blk));
    double* p = nullptr;
    if (!free_list_.empty()) { p = free_list_.back(); free_list_.pop_back(); }
    else CUDA_CHECK(cudaMalloc((void**)&p, slot_bytes()));
    ptrs_[blk] = p;
}

void GpuPhiPool::free(const CellBlock* blk) {
    auto it = ptrs_.find(blk);
    if (it == ptrs_.end()) return;
    free_list_.push_back(it->second);
    ptrs_.erase(it);
}

double* GpuPhiPool::d_phi(const CellBlock* blk) const noexcept {
    auto it = ptrs_.find(blk);
    return it != ptrs_.end() ? it->second : nullptr;
}

void GpuPhiPool::upload(const CellBlock* blk) {
    CUDA_CHECK(cudaMemcpy(d_phi(blk), blk->phi_data_,
                          slot_bytes(), cudaMemcpyHostToDevice));
}

void GpuPhiPool::download(CellBlock* blk) const {
    CUDA_CHECK(cudaMemcpy(blk->phi_data_, d_phi(blk),
                          slot_bytes(), cudaMemcpyDeviceToHost));
}

// ── Kernels ───────────────────────────────────────────────────────────────────

__global__ static void k_save_phin(const GpuAcdiLeafMeta* __restrict__ m) {
    const auto& mt = m[blockIdx.x];
    for (int f = threadIdx.x; f < GPU_NCELL; f += blockDim.x)
        mt.d_phin[f] = mt.d_phi[f];
}

__global__ static void k_zero_phi_rhs(const GpuAcdiLeafMeta* __restrict__ m) {
    const auto& mt = m[blockIdx.x];
    for (int f = threadIdx.x; f < GPU_NCELL; f += blockDim.x)
        mt.d_phi_rhs[f] = 0.0;
}

__device__ static int phi_cidx(int axis, int ax_val, int a, int b) {
    if (axis == 0) return gpu_cell_idx(ax_val, a, b);
    if (axis == 1) return gpu_cell_idx(a, ax_val, b);
    return              gpu_cell_idx(a, b, ax_val);
}

// Same-level copy / zero-gradient ghost fill for phi scalar.
// Grid: (n_leaves, NFACES).  Block: 64 threads.
__global__ static void k_phi_fill_faces(const GpuAcdiLeafMeta* __restrict__ metas) {
    const int li   = blockIdx.x;
    const int face = blockIdx.y;
    const auto& m  = metas[li];
    const int axis = face / 2;
    const int side = face & 1;
    double* __restrict__ dst = m.d_phi;

    if (m.d_phi_nb[face] == nullptr) {
        // Domain BC: zero-gradient
        const int int_ax = (side == 0) ? GPU_NG : (GPU_NB + GPU_NG - 1);
        for (int ab = threadIdx.x; ab < GPU_NB2 * GPU_NB2; ab += blockDim.x) {
            const int a = ab % GPU_NB2, b = ab / GPU_NB2;
            const double val = dst[phi_cidx(axis, int_ax, a, b)];
            for (int gl = 0; gl < GPU_NG; ++gl) {
                const int g = (side == 0) ? (GPU_NG - 1 - gl) : (GPU_NB + GPU_NG + gl);
                dst[phi_cidx(axis, g, a, b)] = val;
            }
        }
        return;
    }
    const double* __restrict__ src = m.d_phi_nb[face];
    for (int ab = threadIdx.x; ab < GPU_NB2 * GPU_NB2; ab += blockDim.x) {
        const int a = ab % GPU_NB2, b = ab / GPU_NB2;
        for (int gl = 0; gl < GPU_NG; ++gl) {
            const int dst_ax = (side == 0) ? (GPU_NG - 1 - gl) : (GPU_NB + GPU_NG + gl);
            const int src_ax = (side == 0) ? (GPU_NB + GPU_NG - 1 - gl) : (GPU_NG + gl);
            dst[phi_cidx(axis, dst_ax, a, b)] = src[phi_cidx(axis, src_ax, a, b)];
        }
    }
}

// Upwind phi advection: one thread per interior cell (NB³=512 threads per block).
__global__ static void k_phi_rhs(const GpuAcdiLeafMeta* __restrict__ metas) {
    const int li = blockIdx.x;
    const int t  = threadIdx.x;
    const auto& m = metas[li];
    const int ii = t % GPU_NB, ji = (t / GPU_NB) % GPU_NB, ki = t / (GPU_NB * GPU_NB);
    const int i = ii + GPU_NG, j = ji + GPU_NG, k = ki + GPU_NG;
    const double* Q   = m.d_Q;
    const double* phi = m.d_phi;
    const double ihx = 1.0 / m.hx, ihy = 1.0 / m.hy, ihz = 1.0 / m.hz;

    auto rho = [&](int a, int b, int c) { return Q[0*GPU_NCELL + gpu_cell_idx(a,b,c)]; };
    auto u   = [&](int a, int b, int c) { return Q[1*GPU_NCELL + gpu_cell_idx(a,b,c)] / rho(a,b,c); };
    auto v   = [&](int a, int b, int c) { return Q[2*GPU_NCELL + gpu_cell_idx(a,b,c)] / rho(a,b,c); };
    auto w   = [&](int a, int b, int c) { return Q[3*GPU_NCELL + gpu_cell_idx(a,b,c)] / rho(a,b,c); };
    auto p   = [&](int a, int b, int c) { return phi[gpu_cell_idx(a,b,c)]; };

    const double ulx = 0.5*(u(i-1,j,k)+u(i,j,k)), urx = 0.5*(u(i,j,k)+u(i+1,j,k));
    const double flx = (ulx>=0) ? ulx*p(i-1,j,k) : ulx*p(i,j,k);
    const double frx = (urx>=0) ? urx*p(i,j,k)   : urx*p(i+1,j,k);

    const double vly = 0.5*(v(i,j-1,k)+v(i,j,k)), vry = 0.5*(v(i,j,k)+v(i,j+1,k));
    const double fly = (vly>=0) ? vly*p(i,j-1,k) : vly*p(i,j,k);
    const double fry = (vry>=0) ? vry*p(i,j,k)   : vry*p(i,j+1,k);

    const double wlz = 0.5*(w(i,j,k-1)+w(i,j,k)), wrz = 0.5*(w(i,j,k)+w(i,j,k+1));
    const double flz = (wlz>=0) ? wlz*p(i,j,k-1) : wlz*p(i,j,k);
    const double frz = (wrz>=0) ? wrz*p(i,j,k)   : wrz*p(i,j,k+1);

    m.d_phi_rhs[gpu_cell_idx(i,j,k)] +=
        ihx*(flx - frx) + ihy*(fly - fry) + ihz*(flz - frz);
}

// Interface-compression source term. Shared memory: 3*GPU_NCELL doubles = 41472 B.
__global__ __launch_bounds__(512)
static void k_phi_compress_rhs(const GpuAcdiLeafMeta* __restrict__ metas) {
    const int li = blockIdx.x;
    const auto& m = metas[li];
    if (m.ceps <= 0.0) return;
    extern __shared__ double sh[];
    double* Fx = sh, *Fy = sh + GPU_NCELL, *Fz = sh + 2*GPU_NCELL;
    const double* phi = m.d_phi;
    const double hmin = fmin(fmin(m.hx, m.hy), m.hz);
    const double eps  = m.ceps * hmin;
    const double eps2 = 1e-10 / (hmin * hmin);
    const double ihx  = 1.0/m.hx, ihy = 1.0/m.hy, ihz = 1.0/m.hz;

    // Phase 1: flux vectors
    for (int f = threadIdx.x; f < GPU_NCELL; f += blockDim.x) {
        const int i_ = f % GPU_NB2, j_ = (f/GPU_NB2) % GPU_NB2, k_ = f / (GPU_NB2*GPU_NB2);
        Fx[f] = Fy[f] = Fz[f] = 0.0;
        if (i_<1 || i_>GPU_NB2-2 || j_<1 || j_>GPU_NB2-2 || k_<1 || k_>GPU_NB2-2) continue;
        const double dpx = 0.5*ihx*(phi[gpu_cell_idx(i_+1,j_,k_)] - phi[gpu_cell_idx(i_-1,j_,k_)]);
        const double dpy = 0.5*ihy*(phi[gpu_cell_idx(i_,j_+1,k_)] - phi[gpu_cell_idx(i_,j_-1,k_)]);
        const double dpz = 0.5*ihz*(phi[gpu_cell_idx(i_,j_,k_+1)] - phi[gpu_cell_idx(i_,j_,k_-1)]);
        const double im  = rsqrt(dpx*dpx + dpy*dpy + dpz*dpz + eps2);
        const double g   = phi[f] * (1.0 - phi[f]);
        Fx[f] = eps*(dpx - g*dpx*im);
        Fy[f] = eps*(dpy - g*dpy*im);
        Fz[f] = eps*(dpz - g*dpz*im);
    }
    __syncthreads();

    // Phase 2: central divergence → add to rhs
    for (int t = threadIdx.x; t < GPU_NB*GPU_NB*GPU_NB; t += blockDim.x) {
        const int ii = t%GPU_NB, ji = (t/GPU_NB)%GPU_NB, ki = t/(GPU_NB*GPU_NB);
        const int i = ii+GPU_NG, j = ji+GPU_NG, k = ki+GPU_NG;
        const int f = gpu_cell_idx(i,j,k);
        m.d_phi_rhs[f] +=
            0.5*ihx*(Fx[gpu_cell_idx(i+1,j,k)] - Fx[gpu_cell_idx(i-1,j,k)])
          + 0.5*ihy*(Fy[gpu_cell_idx(i,j+1,k)] - Fy[gpu_cell_idx(i,j-1,k)])
          + 0.5*ihz*(Fz[gpu_cell_idx(i,j,k+1)] - Fz[gpu_cell_idx(i,j,k-1)]);
    }
}

// SSP-RK3 stage update for phi (interior cells only).
__global__ static void k_phi_update(const GpuAcdiLeafMeta* __restrict__ metas,
                                    const double* __restrict__ d_dt,
                                    double alpha, bool stage1) {
    const auto& m  = metas[blockIdx.x];
    const double dt   = *d_dt;
    const double beta = 1.0 - alpha;
    for (int t = threadIdx.x; t < GPU_NB*GPU_NB*GPU_NB; t += blockDim.x) {
        const int ii = t%GPU_NB, ji = (t/GPU_NB)%GPU_NB, ki = t/(GPU_NB*GPU_NB);
        const int f  = gpu_cell_idx(ii+GPU_NG, ji+GPU_NG, ki+GPU_NG);
        m.d_phi[f] = stage1
            ? m.d_phin[f] + dt * m.d_phi_rhs[f]
            : alpha * m.d_phin[f] + beta * (m.d_phi[f] + dt * m.d_phi_rhs[f]);
    }
}

// ── GpuAcdiList ───────────────────────────────────────────────────────────────

GpuAcdiList::~GpuAcdiList() {
    if (d_metas)     cudaFree(d_metas);
    if (d_phin_pool) cudaFree(d_phin_pool);
    if (d_rhs_pool)  cudaFree(d_rhs_pool);
}

void GpuAcdiList::build(const BlockTree& tree, const GpuPool& q_pool,
                        const GpuPhiPool& phi_pool, double ceps, int bc_type) {
    const auto& leaves = tree.leaf_indices();
    n_leaves = (int)leaves.size();
    ceps_    = ceps;
    if (n_leaves == 0) return;

    if (d_metas)     { cudaFree(d_metas);     d_metas     = nullptr; }
    if (d_phin_pool) { cudaFree(d_phin_pool); d_phin_pool = nullptr; }
    if (d_rhs_pool)  { cudaFree(d_rhs_pool);  d_rhs_pool  = nullptr; }

    const size_t sz = (size_t)n_leaves * GPU_NCELL * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&d_phin_pool, sz));
    CUDA_CHECK(cudaMalloc((void**)&d_rhs_pool,  sz));

    std::vector<GpuAcdiLeafMeta> h_metas(n_leaves);
    for (int ii = 0; ii < n_leaves; ++ii) {
        const int li      = leaves[ii];
        const auto& nd    = tree.nodes[li];
        const CellBlock&  blk = *nd.block;
        auto& mt          = h_metas[ii];
        mt.d_Q       = q_pool.d_Q(&blk);
        mt.d_phi     = phi_pool.d_phi(&blk);
        mt.d_phin    = d_phin_pool + (size_t)ii * GPU_NCELL;
        mt.d_phi_rhs = d_rhs_pool  + (size_t)ii * GPU_NCELL;
        mt.hx = blk.h; mt.hy = blk.hy; mt.hz = blk.hz;
        mt.ceps = ceps;
        for (int d = 0; d < NFACES; ++d) {
            const int ni = nd.neighbours[d];
            if (ni >= 0 && ni < (int)tree.nodes.size() && tree.nodes[ni].has_block()) {
                mt.d_phi_nb[d]  = phi_pool.d_phi(tree.nodes[ni].block.get());
                mt.level_rel[d] = (int8_t)(tree.nodes[ni].level - nd.level);
            } else {
                mt.d_phi_nb[d]  = nullptr;
                mt.level_rel[d] = +2; // domain BC sentinel
            }
            mt.bc_type[d] = (int8_t)bc_type;
        }
        mt.cf_oct = 0;
        if (nd.parent >= 0) {
            const int fc = tree.nodes[nd.parent].first_child;
            if (fc >= 0) mt.cf_oct = (int8_t)(li - fc);
        }
    }
    CUDA_CHECK(cudaMalloc((void**)&d_metas, (size_t)n_leaves * sizeof(GpuAcdiLeafMeta)));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas.data(),
                          (size_t)n_leaves * sizeof(GpuAcdiLeafMeta),
                          cudaMemcpyHostToDevice));
}

void GpuAcdiList::save_phin(cudaStream_t s) const {
    if (!n_leaves) return;
    k_save_phin<<<n_leaves, 256, 0, s>>>(d_metas);
}
void GpuAcdiList::zero_rhs(cudaStream_t s) const {
    if (!n_leaves) return;
    k_zero_phi_rhs<<<n_leaves, 256, 0, s>>>(d_metas);
}
void GpuAcdiList::fill_ghosts(cudaStream_t s) const {
    if (!n_leaves) return;
    dim3 grid(n_leaves, NFACES);
    k_phi_fill_faces<<<grid, 64, 0, s>>>(d_metas);
}
void GpuAcdiList::rhs_advect(cudaStream_t s) const {
    if (!n_leaves) return;
    k_phi_rhs<<<n_leaves, GPU_NB*GPU_NB*GPU_NB, 0, s>>>(d_metas);
}
void GpuAcdiList::rhs_compress(cudaStream_t s) const {
    if (!n_leaves || ceps_ <= 0.0) return;
    const size_t shmem = 3 * GPU_NCELL * sizeof(double); // 41472 B
    k_phi_compress_rhs<<<n_leaves, 512, shmem, s>>>(d_metas);
}
void GpuAcdiList::update_phi(const double* d_dt, double alpha,
                              bool stage1, cudaStream_t s) const {
    if (!n_leaves) return;
    k_phi_update<<<n_leaves, 256, 0, s>>>(d_metas, d_dt, alpha, stage1);
}
