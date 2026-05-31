#include "cuda/gpu_ibm.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "mesh/cell_block.hpp"
#include <vector>

// k_ibm_classify: one thread per cell, compute SDF via BVH, write cell_type+sdf+wnorm.
// gridDim.x = n_leaves, blockDim.x = 256
__global__
void k_ibm_classify(
    const GpuIbmMeta* __restrict__ metas,
    const BvhNode*    __restrict__ d_nodes,  int n_nodes,
    const float* __restrict__ v0x, const float* __restrict__ v0y, const float* __restrict__ v0z,
    const float* __restrict__ v1x, const float* __restrict__ v1y, const float* __restrict__ v1z,
    const float* __restrict__ v2x, const float* __restrict__ v2y, const float* __restrict__ v2z,
    const float* __restrict__ tnx, const float* __restrict__ tny, const float* __restrict__ tnz)
{
    const GpuIbmMeta& m = metas[blockIdx.x];
    for (int flat = threadIdx.x; flat < GPU_NCELL; flat += blockDim.x) {
        int k =  flat / (GPU_NB2 * GPU_NB2);
        int j = (flat /  GPU_NB2) % GPU_NB2;
        int i =  flat %  GPU_NB2;
        float x = m.ox + (i - GPU_NG + 0.5f) * m.hx;
        float y = m.oy + (j - GPU_NG + 0.5f) * m.hy;
        float z = m.oz + (k - GPU_NG + 0.5f) * m.hz;
        float nx, ny, nz;
        float sdf = bvh_sdf(d_nodes, n_nodes,
                             v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z,
                             tnx,tny,tnz, x, y, z, nx, ny, nz);
        m.d_sdf[flat]       = sdf;
        m.d_wnx[flat]       = nx;
        m.d_wny[flat]       = ny;
        m.d_wnz[flat]       = nz;
        m.d_cell_type[flat] = (sdf < 0.f) ? (int8_t)1 : (int8_t)0;
    }
}

// k_ibm_mark_ghosts: second pass — promote interior FLUID cells adjacent to SOLID to IBM_GHOST.
// gridDim.x = n_leaves, blockDim.x = 256
__global__
void k_ibm_mark_ghosts(const GpuIbmMeta* __restrict__ metas)
{
    const GpuIbmMeta& m = metas[blockIdx.x];
    for (int flat = threadIdx.x; flat < GPU_NCELL; flat += blockDim.x) {
        if (m.d_cell_type[flat] != 0) continue;
        int k =  flat / (GPU_NB2 * GPU_NB2);
        int j = (flat /  GPU_NB2) % GPU_NB2;
        int i =  flat %  GPU_NB2;
        if (i < GPU_NG || i >= GPU_NG+GPU_NB) continue;
        if (j < GPU_NG || j >= GPU_NG+GPU_NB) continue;
        if (k < GPU_NG || k >= GPU_NG+GPU_NB) continue;
        // Check 6 face neighbors
        const int di[6]={-1,1,0,0,0,0};
        const int dj[6]={ 0,0,-1,1,0,0};
        const int dk[6]={ 0,0,0,0,-1,1};
        for (int d = 0; d < 6; ++d) {
            int ni=i+di[d], nj=j+dj[d], nk=k+dk[d];
            if (ni<0||ni>=GPU_NB2||nj<0||nj>=GPU_NB2||nk<0||nk>=GPU_NB2) continue;
            int nf = nk*GPU_NB2*GPU_NB2 + nj*GPU_NB2 + ni;
            if (m.d_cell_type[nf] == 1) {
                m.d_cell_type[flat] = 2;
                break;
            }
        }
    }
}

void GpuIbmList::build(const BlockTree& tree, const GpuPool& pool, const GpuBvh& bvh) {
    // Free old allocations
    if (d_metas)          { cudaFree(d_metas);          d_metas = nullptr; }
    if (d_cell_type_pool) { cudaFree(d_cell_type_pool); d_cell_type_pool = nullptr; }
    if (d_sdf_pool)       { cudaFree(d_sdf_pool);       d_sdf_pool = nullptr; }
    if (d_wnorm_pool)     { cudaFree(d_wnorm_pool);     d_wnorm_pool = nullptr; }
    if (d_ghosts)         { cudaFree(d_ghosts);         d_ghosts = nullptr; }
    n_ghosts = 0;

    std::vector<int> local;
    for (int idx : tree.leaf_indices())
        if (tree.nodes[idx].has_block()) local.push_back(idx);
    n_leaves = (int)local.size();
    if (n_leaves == 0) return;

    CUDA_CHECK(cudaMalloc(&d_cell_type_pool, (size_t)n_leaves * GPU_NCELL * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_sdf_pool,       (size_t)n_leaves * GPU_NCELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wnorm_pool,     (size_t)n_leaves * GPU_NCELL * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cell_type_pool, 0, (size_t)n_leaves * GPU_NCELL * sizeof(int8_t)));

    std::vector<GpuIbmMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const CellBlock* blk = tree.nodes[local[li]].block.get();
        GpuIbmMeta& m = h_metas[li];
        m.d_Q        = pool.d_Q(blk);
        m.d_cell_type= d_cell_type_pool + (size_t)li * GPU_NCELL;
        m.d_sdf      = d_sdf_pool       + (size_t)li * GPU_NCELL;
        m.d_wnx      = d_wnorm_pool     + (size_t)(li * GPU_NCELL * 3 + 0 * GPU_NCELL);
        m.d_wny      = d_wnorm_pool     + (size_t)(li * GPU_NCELL * 3 + 1 * GPU_NCELL);
        m.d_wnz      = d_wnorm_pool     + (size_t)(li * GPU_NCELL * 3 + 2 * GPU_NCELL);
        m.ox = (float)blk->ox; m.oy = (float)blk->oy; m.oz = (float)blk->oz;
        m.hx = (float)blk->h; m.hy = (float)blk->hy; m.hz = (float)blk->hz;
    }
    gpu_upload_meta(d_metas, h_metas);

    constexpr int TPB = 256;
    k_ibm_classify<<<n_leaves, TPB>>>(
        d_metas, bvh.d_nodes, bvh.n_nodes,
        bvh.d_v0x, bvh.d_v0y, bvh.d_v0z,
        bvh.d_v1x, bvh.d_v1y, bvh.d_v1z,
        bvh.d_v2x, bvh.d_v2y, bvh.d_v2z,
        bvh.d_nx,  bvh.d_ny,  bvh.d_nz);
    k_ibm_mark_ghosts<<<n_leaves, TPB>>>(d_metas);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase 2 (ghost list): added in Task 4
}

void GpuIbmList::exec(cudaStream_t /*stream*/) const {}

GpuIbmList::~GpuIbmList() {
    if (d_metas)          cudaFree(d_metas);
    if (d_cell_type_pool) cudaFree(d_cell_type_pool);
    if (d_sdf_pool)       cudaFree(d_sdf_pool);
    if (d_wnorm_pool)     cudaFree(d_wnorm_pool);
    if (d_ghosts)         cudaFree(d_ghosts);
}
