#include "cuda/gpu_ibm.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "mesh/cell_block.hpp"
#include <vector>
#include <cmath>

// gridDim.x = n_leaves, blockDim.x = 256
__global__
void k_ibm_classify(
    const GpuIbmMeta* __restrict__ metas,
    const BvhNode*    __restrict__ d_nodes,  int n_nodes, int n_tris,
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
        float sdf = bvh_sdf(d_nodes, n_nodes, n_tris,
                             v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z,
                             tnx,tny,tnz, x, y, z, nx, ny, nz);
        m.d_sdf[flat]       = sdf;
        m.d_wnx[flat]       = nx;
        m.d_wny[flat]       = ny;
        m.d_wnz[flat]       = nz;
        m.d_cell_type[flat] = (sdf < 0.f) ? (int8_t)1 : (int8_t)0;
    }
}

// Second pass after k_ibm_classify — widening the solid boundary by one cell layer ensures
// the interpolation stencil (Task 4) always has at least one fluid probe point.
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
        // Face-connectivity only (no diagonal) keeps the ghost layer exactly one cell thick
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

// FSI-1: Update u_wall/v_wall/w_wall in ghost entries from rigid-body state.
// One thread per ghost entry.
// gridDim.x = (n_ghosts + 255) / 256,  blockDim.x = 256
__global__
void k_apply_moving_wall(GhostEntry* __restrict__ ghosts, int n_ghosts,
                          double vcm_x, double vcm_y, double vcm_z,
                          double omega_x, double omega_y, double omega_z,
                          double xcm_x, double xcm_y, double xcm_z)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ghosts) return;
    GhostEntry& ge = ghosts[tid];
    if (ge.wall_bc == 3) return;  // SolidFill: no wall BC to update

    // r = x_surf - x_cm
    double rx = ge.x_surf - xcm_x;
    double ry = ge.y_surf - xcm_y;
    double rz = ge.z_surf - xcm_z;

    // v_wall = v_cm + omega × r
    ge.u_wall = (float)(vcm_x + omega_y * rz - omega_z * ry);
    ge.v_wall = (float)(vcm_y + omega_z * rx - omega_x * rz);
    ge.w_wall = (float)(vcm_z + omega_x * ry - omega_y * rx);
}

// FSI-1: Accumulate pressure force and torque on the immersed surface.
// Iterates over all interior IBM_GHOST cells; for each, reads cell pressure
// from d_scratch (primitive layout: [comp][NCELL], comp=4 is pressure).
// dF = p * n_outward * dA,  dT = (x_surf - x_cm) × dF.
// Uses atomicAdd on d_wrench[6].
// gridDim.x = n_leaves, blockDim.x = 256
__global__
void k_surface_forces_ibm(
    const GpuIbmForceMeta* __restrict__ metas,
    int n_leaves,
    double x_cm_x, double x_cm_y, double x_cm_z,
    double* __restrict__ d_wrench)
{
    if (blockIdx.x >= n_leaves) return;
    const GpuIbmForceMeta& m = metas[blockIdx.x];
    const int NC = GPU_NCELL;
    const double dA = (double)(m.hx * m.hy);  // face area = h² for cubic cells

    for (int flat = threadIdx.x; flat < NC; flat += blockDim.x) {
        if (m.d_cell_type[flat] != 2) continue;  // only IBM_GHOST cells

        int k =  flat / (GPU_NB2 * GPU_NB2);
        int j = (flat /  GPU_NB2) % GPU_NB2;
        int i =  flat %  GPU_NB2;

        // Skip ghost layer cells outside interior range
        if (i < GPU_NG || i >= GPU_NG + GPU_NB) continue;
        if (j < GPU_NG || j >= GPU_NG + GPU_NB) continue;
        if (k < GPU_NG || k >= GPU_NG + GPU_NB) continue;

        // Cell pressure from prim scratch (rho=0,u=1,v=2,w=3,p=4 — see SCRATCH_P_IDX)
        double p = m.d_scratch[SCRATCH_P_IDX * NC + flat];
        if (p < 0.0) p = 0.0;

        // Outward wall normal (points from solid into fluid, i.e. outward from surface)
        double nx = (double)m.d_wnx[flat];
        double ny = (double)m.d_wny[flat];
        double nz = (double)m.d_wnz[flat];

        // Surface point: x_ghost - sdf * n  (sdf > 0 for ghost cells near surface)
        double sdf = (double)m.d_sdf[flat];
        double gx = (double)m.ox + (i - GPU_NG + 0.5) * (double)m.hx;
        double gy = (double)m.oy + (j - GPU_NG + 0.5) * (double)m.hy;
        double gz = (double)m.oz + (k - GPU_NG + 0.5) * (double)m.hz;
        double xs = gx - sdf * nx;
        double ys = gy - sdf * ny;
        double zs = gz - sdf * nz;

        // Force contribution: dF = p * n * dA
        double dFx = p * nx * dA;
        double dFy = p * ny * dA;
        double dFz = p * nz * dA;

        // Torque contribution: dT = (x_surf - x_cm) × dF
        double rx = xs - x_cm_x;
        double ry = ys - x_cm_y;
        double rz = zs - x_cm_z;
        double dTx = ry * dFz - rz * dFy;
        double dTy = rz * dFx - rx * dFz;
        double dTz = rx * dFy - ry * dFx;

        atomicAdd(&d_wrench[0], dFx);
        atomicAdd(&d_wrench[1], dFy);
        atomicAdd(&d_wrench[2], dFz);
        atomicAdd(&d_wrench[3], dTx);
        atomicAdd(&d_wrench[4], dTy);
        atomicAdd(&d_wrench[5], dTz);
    }
}

// gridDim.x = (n_ghosts + 255) / 256,  blockDim.x = 256
__global__
void k_ghost_fill_ibm(const GhostEntry* __restrict__ ghosts, int n_ghosts)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ghosts) return;
    const GhostEntry& ge = ghosts[tid];
    const int NC = GPU_NCELL;

    // Normalize weights so sum=1 in fp64; eliminates bias for constant fields
    // (float ge.w[s] can sum to 1+ε for curved-surface image points, seeding instability).
    double wsum = 0.0;
    for (int s = 0; s < 8; ++s) wsum += (double)ge.w[s];
    const double inv_wsum = (wsum > 0.0) ? 1.0 / wsum : 1.0;

    double Q_I[GPU_NVAR] = {};
    for (int s = 0; s < 8; ++s) {
        const double* sp = ge.stencil[s];
        double w = (double)ge.w[s] * inv_wsum;
        for (int v = 0; v < GPU_NVAR; ++v)
            Q_I[v] += w * sp[v * NC];
    }

    double rho_I = Q_I[0];
    if (rho_I < 1e-12) rho_I = 1e-12;
    double u_I   = Q_I[1] / rho_I;
    double v_I   = Q_I[2] / rho_I;
    double w_I   = Q_I[3] / rho_I;
    double E_I   = Q_I[4];
    double p_I   = (GPU_GAMMA - 1.0) * (E_I - 0.5 * rho_I * (u_I*u_I + v_I*v_I + w_I*w_I));
    if (p_I < 1e-12) p_I = 1e-12;
    double T_I   = p_I / (rho_I * GPU_R_GAS);

    // bc==3: SolidFill — copy image-point state directly to SOLID cell; no wall BC
    if (ge.wall_bc == 3) {
        double* gp = ge.ghost_ptr;
        for (int v = 0; v < GPU_NVAR; ++v)
            gp[v * NC] = Q_I[v];
        return;
    }

    double rho_g = rho_I;
    double u_g   = 2.0 * ge.u_wall - u_I;
    double v_g   = 2.0 * ge.v_wall - v_I;
    double w_g   = 2.0 * ge.w_wall - w_I;
    double p_g   = p_I;

    // bc==0: NoSlip + adiabatic: p_ghost = p_I (zero normal gradient for T and p)
    // bc==2: Isothermal: prescribe T_wall
    if (ge.wall_bc == 2) {
        double T_g = 2.0 * ge.T_wall - T_I;
        if (T_g < 1.0) T_g = 1.0;
        p_g = rho_g * GPU_R_GAS * T_g;
    }

    double KE_g = 0.5 * rho_g * (u_g*u_g + v_g*v_g + w_g*w_g);
    double E_g  = p_g / (GPU_GAMMA - 1.0) + KE_g;

    double* gp = ge.ghost_ptr;
    gp[0 * NC] = rho_g;
    gp[1 * NC] = rho_g * u_g;
    gp[2 * NC] = rho_g * v_g;
    gp[3 * NC] = rho_g * w_g;
    gp[4 * NC] = E_g;
}

// gridDim.x = n_leaves, blockDim.x = 256
// Zero d_rhs_pool for every SOLID (1) and IBM_GHOST (2) interior cell.
__global__
void k_ibm_zero_solid_rhs(
    const int8_t* __restrict__ ct_pool,
    double*       __restrict__ rhs_pool,
    int n_leaves)
{
    const int li = blockIdx.x;
    if (li >= n_leaves) return;
    const int NC = GPU_NCELL;
    const int NV = GPU_NVAR;
    const int8_t* ct  = ct_pool  + (size_t)li * NC;
    double*       rhs = rhs_pool + (size_t)li * NV * NC;
    for (int flat = threadIdx.x; flat < NC; flat += blockDim.x) {
        if (ct[flat] != 0) {  // SOLID (1) or IBM_GHOST (2)
            for (int v = 0; v < NV; ++v)
                rhs[v * NC + flat] = 0.0;
        }
    }
}

void GpuIbmList::zero_solid_rhs(double* d_rhs_pool, cudaStream_t stream) const
{
    if (n_leaves == 0 || !d_cell_type_pool || !d_rhs_pool) return;
    k_ibm_zero_solid_rhs<<<n_leaves, 256, 0, stream>>>(
        d_cell_type_pool, d_rhs_pool, n_leaves);
}

struct LeafInfo {
    double ox, oy, oz, hx, hy, hz;
    double* d_Q;
};

static void build_ghost_entries(
    const BlockTree& tree,
    const GpuPool& pool,
    const std::vector<int>& local,
    const int8_t* d_ct_pool,
    const float*  d_sdf_pool,
    const float*  d_wnorm_pool,
    int n_leaves,
    uint8_t wall_bc, float u_wall, float v_wall, float w_wall, float T_wall,
    GhostEntry*& d_ghosts_out,      int& n_ghosts_out,
    GhostEntry*& d_solid_fills_out, int& n_solid_fills_out)
{
    const int NC = GPU_NCELL;
    std::vector<int8_t> h_ct(n_leaves * NC);
    std::vector<float>  h_sdf(n_leaves * NC);
    std::vector<float>  h_wnorm(n_leaves * NC * 3);
    CUDA_CHECK(cudaMemcpy(h_ct.data(),    d_ct_pool,    n_leaves*NC*sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sdf.data(),   d_sdf_pool,   n_leaves*NC*sizeof(float),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wnorm.data(), d_wnorm_pool, n_leaves*NC*3*sizeof(float),cudaMemcpyDeviceToHost));

    std::vector<LeafInfo> info(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const CellBlock* blk = tree.nodes[local[li]].block.get();
        info[li] = {blk->ox, blk->oy, blk->oz,
                    (double)blk->h, (double)blk->hy, (double)blk->hz,
                    pool.d_Q(blk)};
    }

    // Find the leaf and base cell for a physical point (px, py, pz).
    // TODO: replace linear scan with a spatial hash for AMR performance (O(n_leaves) now).
    auto find_cell = [&](double px, double py, double pz,
                         int& out_li, int& out_flat) -> bool {
        for (int li = 0; li < n_leaves; ++li) {
            const auto& inf = info[li];
            double fi = (px - inf.ox) / inf.hx + GPU_NG - 0.5;
            double fj = (py - inf.oy) / inf.hy + GPU_NG - 0.5;
            double fk = (pz - inf.oz) / inf.hz + GPU_NG - 0.5;
            int ci = (int)fi, cj = (int)fj, ck = (int)fk;
            if (ci < 0 || ci >= GPU_NB2-1) continue;
            if (cj < 0 || cj >= GPU_NB2-1) continue;
            if (ck < 0 || ck >= GPU_NB2-1) continue;
            out_li   = li;
            out_flat = ck*GPU_NB2*GPU_NB2 + cj*GPU_NB2 + ci;
            return true;
        }
        return false;
    };

    std::vector<GhostEntry> h_ghosts_, h_solid_fills_;

    // Helper: build trilinear stencil GhostEntry given a cell and its image point
    auto make_entry = [&](int li, int flat, float sdf,
                          float nx, float ny, float nz,
                          uint8_t bc) -> bool {
        const auto& inf = info[li];
        int k =  flat / (GPU_NB2 * GPU_NB2);
        int j = (flat /  GPU_NB2) % GPU_NB2;
        int i =  flat %  GPU_NB2;
        double gx = inf.ox + (i - GPU_NG + 0.5) * inf.hx;
        double gy = inf.oy + (j - GPU_NG + 0.5) * inf.hy;
        double gz = inf.oz + (k - GPU_NG + 0.5) * inf.hz;

        // I = G - 2*sdf_eff*n  (sdf>0 → I inside solid; sdf<0 → I in fluid)
        // IBM_GHOST (sdf>0): push image ≥2*hx into solid so the 8-cell trilinear stencil
        //   is entirely SOLID (no accidental FLUID corners that bypass the wall BC).
        // SOLID (sdf<0, bc==3): image ≥2*hx into fluid → past the IBM_GHOST shell (≤hx wide),
        //   so stencil is pure FLUID and avoids double-reflection that negates no-slip.
        const double sdf_eff = (bc != 3)
            ? (double)std::max(sdf,  (float)(2.0f*inf.hx))    // ghost → ≥2*hx into solid
            : -(double)std::max(-sdf, (float)(2.0f*inf.hx));   // solid → ≥2*hx into fluid
        double ix = gx - 2.0 * sdf_eff * nx;
        double iy = gy - 2.0 * sdf_eff * ny;
        double iz = gz - 2.0 * sdf_eff * nz;

        int sli, sflat;
        if (!find_cell(ix, iy, iz, sli, sflat)) return false;

        const auto& sinf = info[sli];
        int sk0 =  sflat / (GPU_NB2 * GPU_NB2);
        int sj0 = (sflat /  GPU_NB2) % GPU_NB2;
        int si0 =  sflat %  GPU_NB2;

        double fi = (ix - sinf.ox) / sinf.hx + GPU_NG - 0.5;
        double fj = (iy - sinf.oy) / sinf.hy + GPU_NG - 0.5;
        double fk = (iz - sinf.oz) / sinf.hz + GPU_NG - 0.5;
        double tx = fi - si0, ty = fj - sj0, tz = fk - sk0;
        tx = (tx < 0) ? 0 : (tx > 1 ? 1 : tx);
        ty = (ty < 0) ? 0 : (ty > 1 ? 1 : ty);
        tz = (tz < 0) ? 0 : (tz > 1 ? 1 : tz);

        GhostEntry ge{};
        ge.ghost_ptr = inf.d_Q + flat;
        ge.wall_bc   = bc;
        ge.u_wall    = u_wall;
        ge.v_wall    = v_wall;
        ge.w_wall    = w_wall;
        ge.T_wall    = T_wall;
        // FSI-1: surface point = ghost_centroid - sdf * n  (sdf > 0 for ghost cells)
        ge.x_surf    = (float)(gx - (double)sdf * nx);
        ge.y_surf    = (float)(gy - (double)sdf * ny);
        ge.z_surf    = (float)(gz - (double)sdf * nz);

        for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int s = dz*4 + dy*2 + dx;
            int ci = si0+dx, cj = sj0+dy, ck = sk0+dz;
            if (ci >= GPU_NB2) ci = GPU_NB2-1;
            if (cj >= GPU_NB2) cj = GPU_NB2-1;
            if (ck >= GPU_NB2) ck = GPU_NB2-1;
            int sf = ck*GPU_NB2*GPU_NB2 + cj*GPU_NB2 + ci;
            ge.stencil[s] = sinf.d_Q + sf;
            ge.w[s] = (float)(
                (dx ? tx : 1-tx) * (dy ? ty : 1-ty) * (dz ? tz : 1-tz));
        }
        if (bc == 3)
            h_solid_fills_.push_back(ge);
        else
            h_ghosts_.push_back(ge);
        return true;
    };

    // Pass 1: IBM_GHOST cells — wall BC ghost fill
    for (int li = 0; li < n_leaves; ++li)
    for (int k = GPU_NG; k < GPU_NG+GPU_NB; ++k)
    for (int j = GPU_NG; j < GPU_NG+GPU_NB; ++j)
    for (int i = GPU_NG; i < GPU_NG+GPU_NB; ++i) {
        int flat = k*GPU_NB2*GPU_NB2 + j*GPU_NB2 + i;
        if (h_ct[li*NC + flat] != 2) continue;
        float sdf = h_sdf[li*NC + flat];
        float nx  = h_wnorm[li*NC*3 + 0*NC + flat];
        float ny  = h_wnorm[li*NC*3 + 1*NC + flat];
        float nz  = h_wnorm[li*NC*3 + 2*NC + flat];
        make_entry(li, flat, sdf, nx, ny, nz, wall_bc);
    }

    // Pass 2: SOLID cells — suppress RHS accumulation by refreshing from fluid image
    // For sdf<0: I = G - 2*sdf*n_outward = G + 2*|sdf|*n_outward (into fluid).
    // Clamp |sdf| to 1.5*hx so the image point stays near the surface even for
    // deeply interior cells where the unclamped image would overshoot into another
    // SOLID cell or across a block boundary.
    for (int li = 0; li < n_leaves; ++li)
    for (int k = GPU_NG; k < GPU_NG+GPU_NB; ++k)
    for (int j = GPU_NG; j < GPU_NG+GPU_NB; ++j)
    for (int i = GPU_NG; i < GPU_NG+GPU_NB; ++i) {
        int flat = k*GPU_NB2*GPU_NB2 + j*GPU_NB2 + i;
        if (h_ct[li*NC + flat] != 1) continue;
        float sdf = h_sdf[li*NC + flat];
        float nx  = h_wnorm[li*NC*3 + 0*NC + flat];
        float ny  = h_wnorm[li*NC*3 + 1*NC + flat];
        float nz  = h_wnorm[li*NC*3 + 2*NC + flat];
        float min_sdf = -1.5f * (float)info[li].hx; // most-negative allowed
        if (sdf < min_sdf) sdf = min_sdf;
        make_entry(li, flat, sdf, nx, ny, nz, 3);
    }

    n_ghosts_out = (int)h_ghosts_.size();
    if (d_ghosts_out) { cudaFree(d_ghosts_out); d_ghosts_out = nullptr; }
    if (n_ghosts_out > 0) {
        CUDA_CHECK(cudaMalloc(&d_ghosts_out, n_ghosts_out * sizeof(GhostEntry)));
        CUDA_CHECK(cudaMemcpy(d_ghosts_out, h_ghosts_.data(),
                              n_ghosts_out * sizeof(GhostEntry), cudaMemcpyHostToDevice));
    }

    n_solid_fills_out = (int)h_solid_fills_.size();
    if (d_solid_fills_out) { cudaFree(d_solid_fills_out); d_solid_fills_out = nullptr; }
    if (n_solid_fills_out > 0) {
        CUDA_CHECK(cudaMalloc(&d_solid_fills_out, n_solid_fills_out * sizeof(GhostEntry)));
        CUDA_CHECK(cudaMemcpy(d_solid_fills_out, h_solid_fills_.data(),
                              n_solid_fills_out * sizeof(GhostEntry), cudaMemcpyHostToDevice));
    }
}

void GpuIbmList::build(const BlockTree& tree, const GpuPool& pool, const GpuBvh& bvh) {
    if (d_metas)          { cudaFree(d_metas);          d_metas = nullptr; }
    if (d_cell_type_pool) { cudaFree(d_cell_type_pool); d_cell_type_pool = nullptr; }
    if (d_sdf_pool)       { cudaFree(d_sdf_pool);       d_sdf_pool = nullptr; }
    if (d_wnorm_pool)     { cudaFree(d_wnorm_pool);     d_wnorm_pool = nullptr; }
    if (d_ghosts)         { cudaFree(d_ghosts);         d_ghosts = nullptr; }
    if (d_solid_fills)    { cudaFree(d_solid_fills);    d_solid_fills = nullptr; }
    n_ghosts = 0; n_solid_fills = 0;

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
        d_metas, bvh.d_nodes, bvh.n_nodes, bvh.n_tris,
        bvh.d_v0x, bvh.d_v0y, bvh.d_v0z,
        bvh.d_v1x, bvh.d_v1y, bvh.d_v1z,
        bvh.d_v2x, bvh.d_v2y, bvh.d_v2z,
        bvh.d_nx,  bvh.d_ny,  bvh.d_nz);
    k_ibm_mark_ghosts<<<n_leaves, TPB>>>(d_metas);
    CUDA_CHECK(cudaDeviceSynchronize());

    build_ghost_entries(tree, pool, local,
                        d_cell_type_pool, d_sdf_pool, d_wnorm_pool, n_leaves,
                        wall_bc, u_wall, v_wall, w_wall, T_wall,
                        d_ghosts,      n_ghosts,
                        d_solid_fills, n_solid_fills);
}

void GpuIbmList::exec(cudaStream_t stream) const {
    constexpr int TPB = 256;

    // FSI-1: If any rigid-body motion is present, update ghost-entry wall velocities.
    const bool has_motion =
        rigid.v_cm[0] != 0.0 || rigid.v_cm[1] != 0.0 || rigid.v_cm[2] != 0.0 ||
        rigid.omega[0] != 0.0 || rigid.omega[1] != 0.0 || rigid.omega[2] != 0.0;
    if (has_motion && n_ghosts > 0 && d_ghosts) {
        int nb = (n_ghosts + TPB - 1) / TPB;
        k_apply_moving_wall<<<nb, TPB, 0, stream>>>(
            d_ghosts, n_ghosts,
            rigid.v_cm[0], rigid.v_cm[1], rigid.v_cm[2],
            rigid.omega[0], rigid.omega[1], rigid.omega[2],
            rigid.x_cm[0], rigid.x_cm[1], rigid.x_cm[2]);
    }

    // SolidFill first so that ghost-cell stencils reading SOLID cells see fresh fluid values
    if (n_solid_fills > 0 && d_solid_fills) {
        int nb = (n_solid_fills + TPB - 1) / TPB;
        k_ghost_fill_ibm<<<nb, TPB, 0, stream>>>(d_solid_fills, n_solid_fills);
    }
    if (n_ghosts > 0 && d_ghosts) {
        int nb = (n_ghosts + TPB - 1) / TPB;
        k_ghost_fill_ibm<<<nb, TPB, 0, stream>>>(d_ghosts, n_ghosts);
    }
}

GpuIbmList::~GpuIbmList() {
    if (d_metas)          cudaFree(d_metas);
    if (d_cell_type_pool) cudaFree(d_cell_type_pool);
    if (d_sdf_pool)       cudaFree(d_sdf_pool);
    if (d_wnorm_pool)     cudaFree(d_wnorm_pool);
    if (d_ghosts)         cudaFree(d_ghosts);
    if (d_solid_fills)    cudaFree(d_solid_fills);
}
