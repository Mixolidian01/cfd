// gpu_sgs.cu — P-SGS-GPU: Smagorinsky SGS model kernel.
//
// k_sgs_smag: one CUDA block per leaf.
//   Phase 1 (cooperative, 64 threads): compute mu_t[GPU_NCELL] in shared memory.
//   Phase 2 (per column i,j): conservative stress divergence → in-place d_Q update.
//
// Matches CPU SmagorinskyModel::apply() exactly under periodic BC.
// Wall BC: mu_t at inner ghost cells is computed from reflected velocities
//   (slightly inflated vs CPU which zeros them); effect is confined to one cell.

#include "cuda/gpu_sgs.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

// Velocity component comp (0=u,1=v,2=w) at cell (ii,jj,kk) from flat SoA dQ.
__device__ __forceinline__ static double
sgs_vel(const double* __restrict__ dQ, int comp, int ii, int jj, int kk) noexcept
{
    const int f = gpu_cell_idx(ii, jj, kk);
    return dQ[(comp + 1) * GPU_NCELL + f] / dQ[f];
}

// Temperature T = p/(ρ·R) at cell (ii,jj,kk), computed from conserved dQ.
__device__ __forceinline__ static double
sgs_T(const double* __restrict__ dQ, int ii, int jj, int kk) noexcept
{
    const int    f   = gpu_cell_idx(ii, jj, kk);
    const double rho = dQ[f];
    const double u_  = dQ[1*GPU_NCELL+f] / rho;
    const double v_  = dQ[2*GPU_NCELL+f] / rho;
    const double w_  = dQ[3*GPU_NCELL+f] / rho;
    const double E   = dQ[4*GPU_NCELL+f];
    const double p   = (GPU_GAMMA - 1.0) * (E - 0.5 * rho * (u_*u_ + v_*v_ + w_*w_));
    return p / (rho * GPU_R_GAS);
}

// Contribute face stress (axis AX, direction dn=±1) to acc[3].
// mu: face-averaged mu_t.  ih: 1/h.  ihs: 1/(4h) for averaged tang. grads.
// Mirrors CPU VelocityGradAtFace + FaceInterp pattern in apply_sgs_stress_div.
template<int AX>
__device__ __forceinline__ static void
sgs_face_contrib(const double* __restrict__ dQ, int dn, int i, int j, int k,
                 double mu, double ih, double ihs,
                 double& tnn_out, double& tnt1_out, double& tnt2_out) noexcept
{
    constexpr int t1  = (AX + 1) % 3;
    constexpr int t2  = (AX + 2) % 3;
    constexpr int d1i = (t1 == 0), d1j = (t1 == 1), d1k = (t1 == 2);
    constexpr int d2i = (t2 == 0), d2j = (t2 == 1), d2k = (t2 == 2);

    const int ni = i + (AX == 0 ? dn : 0);
    const int nj = j + (AX == 1 ? dn : 0);
    const int nk = k + (AX == 2 ? dn : 0);

    // Normal velocity gradients at this face
    const double dnn  = ih * dn * (sgs_vel(dQ,AX,ni,nj,nk) - sgs_vel(dQ,AX,i,j,k));
    const double dtn1 = ih * dn * (sgs_vel(dQ,t1,ni,nj,nk) - sgs_vel(dQ,t1,i,j,k));
    const double dtn2 = ih * dn * (sgs_vel(dQ,t2,ni,nj,nk) - sgs_vel(dQ,t2,i,j,k));

    // Tangential gradients: face-averaged between ni,nj,nk and i,j,k
    const double dan1 = ihs*((sgs_vel(dQ,AX,ni+d1i,nj+d1j,nk+d1k)-sgs_vel(dQ,AX,ni-d1i,nj-d1j,nk-d1k))
                            +(sgs_vel(dQ,AX,i +d1i,j +d1j,k +d1k)-sgs_vel(dQ,AX,i -d1i,j -d1j,k -d1k)));
    const double dan2 = ihs*((sgs_vel(dQ,AX,ni+d2i,nj+d2j,nk+d2k)-sgs_vel(dQ,AX,ni-d2i,nj-d2j,nk-d2k))
                            +(sgs_vel(dQ,AX,i +d2i,j +d2j,k +d2k)-sgs_vel(dQ,AX,i -d2i,j -d2j,k -d2k)));
    const double dt1t1= ihs*((sgs_vel(dQ,t1,ni+d1i,nj+d1j,nk+d1k)-sgs_vel(dQ,t1,ni-d1i,nj-d1j,nk-d1k))
                            +(sgs_vel(dQ,t1,i +d1i,j +d1j,k +d1k)-sgs_vel(dQ,t1,i -d1i,j -d1j,k -d1k)));
    const double dt2t2= ihs*((sgs_vel(dQ,t2,ni+d2i,nj+d2j,nk+d2k)-sgs_vel(dQ,t2,ni-d2i,nj-d2j,nk-d2k))
                            +(sgs_vel(dQ,t2,i +d2i,j +d2j,k +d2k)-sgs_vel(dQ,t2,i -d2i,j -d2j,k -d2k)));

    const double divu = dnn + dt1t1 + dt2t2;
    tnn_out  = mu * (2.0 * dnn  - (2.0 / 3.0) * divu);
    tnt1_out = mu * (dan1 + dtn1);
    tnt2_out = mu * (dan2 + dtn2);
}

// Accumulate axis AX contribution to acc[3] (momentum divergence).
template<int AX>
__device__ __forceinline__ static void
sgs_axis(const double* __restrict__ dQ, double ih, double ihs,
         int i, int j, int k, double mu_p, double mu_m,
         double (&acc)[3]) noexcept
{
    constexpr int t1 = (AX + 1) % 3;
    constexpr int t2 = (AX + 2) % 3;
    double tnn_p, tnt1_p, tnt2_p;
    double tnn_m, tnt1_m, tnt2_m;
    sgs_face_contrib<AX>(dQ, +1, i,j,k, mu_p, ih, ihs, tnn_p, tnt1_p, tnt2_p);
    sgs_face_contrib<AX>(dQ, -1, i,j,k, mu_m, ih, ihs, tnn_m, tnt1_m, tnt2_m);
    acc[AX] += ih * (tnn_p  - tnn_m );
    acc[t1] += ih * (tnt1_p - tnt1_m);
    acc[t2] += ih * (tnt2_p - tnt2_m);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_sgs_smag: Smagorinsky SGS operator-split in-place update.
// Grid: dim3(n_leaves), Block: dim3(GPU_NB, GPU_NB) = 64 threads.
// Shared: smu_t[GPU_NCELL] = 1728 doubles = 13 824 bytes.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_sgs_smag(const GpuSgsMeta* __restrict__ metas,
                const double*     __restrict__ d_dt)
{
    __shared__ double smu_t[GPU_NCELL];

    const GpuSgsMeta& m    = metas[blockIdx.x];
    double*       dQ       = m.d_Q;
    const double  ih       = 1.0 / m.h;
    const double  ihs      = 0.25 * ih;  // 1/(4h): face-averaged tang. grad. factor
    const double  ihs2     = 0.5  * ih;  // 1/(2h): cell-centred grad. factor
    const double  ih2      = ih   * ih;  // 1/h²: Laplacian factor
    const double  Cs2h2    = m.Cs2h2;
    const double  kap      = m.kap_fac;
    const double  dt       = *d_dt;

    // ── Phase 1: fill smu_t cooperatively over all 64 threads ─────────────────
    // Compute mu_t at [1..NB2-2] using central differences on ghost-filled dQ.
    // Outermost ghost (indices 0 and NB2-1) → 0 (never accessed by face averages
    // at interior cells, since face averages only reach indices [1..NB2-2]).
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    for (int flat = tid; flat < GPU_NCELL; flat += blockDim.x * blockDim.y) {
        const int k_  =  flat / (GPU_NB2 * GPU_NB2);
        const int tmp =  flat % (GPU_NB2 * GPU_NB2);
        const int jj  =  tmp  /  GPU_NB2;
        const int ii  =  tmp  %  GPU_NB2;
        if (ii >= 1 && ii < GPU_NB2-1 && jj >= 1 && jj < GPU_NB2-1 &&
            k_  >= 1 && k_  < GPU_NB2-1) {
            const double dudx = ihs2*(sgs_vel(dQ,0,ii+1,jj,k_)-sgs_vel(dQ,0,ii-1,jj,k_));
            const double dvdy = ihs2*(sgs_vel(dQ,1,ii,jj+1,k_)-sgs_vel(dQ,1,ii,jj-1,k_));
            const double dwdz = ihs2*(sgs_vel(dQ,2,ii,jj,k_+1)-sgs_vel(dQ,2,ii,jj,k_-1));
            const double dudy = ihs2*(sgs_vel(dQ,0,ii,jj+1,k_)-sgs_vel(dQ,0,ii,jj-1,k_));
            const double dudz = ihs2*(sgs_vel(dQ,0,ii,jj,k_+1)-sgs_vel(dQ,0,ii,jj,k_-1));
            const double dvdx = ihs2*(sgs_vel(dQ,1,ii+1,jj,k_)-sgs_vel(dQ,1,ii-1,jj,k_));
            const double dvdz = ihs2*(sgs_vel(dQ,1,ii,jj,k_+1)-sgs_vel(dQ,1,ii,jj,k_-1));
            const double dwdx = ihs2*(sgs_vel(dQ,2,ii+1,jj,k_)-sgs_vel(dQ,2,ii-1,jj,k_));
            const double dwdy = ihs2*(sgs_vel(dQ,2,ii,jj+1,k_)-sgs_vel(dQ,2,ii,jj-1,k_));
            const double Sxx = dudx, Syy = dvdy, Szz = dwdz;
            const double Sxy = 0.5*(dudy + dvdx);
            const double Sxz = 0.5*(dudz + dwdx);
            const double Syz = 0.5*(dvdz + dwdy);
            const double S2  = 2.0 * (Sxx*Sxx + Syy*Syy + Szz*Szz
                                    + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz));
            smu_t[flat] = dQ[flat] * Cs2h2 * sqrt(S2);
        } else {
            smu_t[flat] = 0.0;
        }
    }
    __syncthreads();

    // ── Phase 2: stress divergence for interior cells ──────────────────────────
    // Each thread owns column (i, j, all k ∈ [NG..NG+NB-1]).
    const int i = GPU_NG + (int)threadIdx.x;
    const int j = GPU_NG + (int)threadIdx.y;

    for (int k = GPU_NG; k <= GPU_NG + GPU_NB - 1; ++k) {
        // Face-averaged mu_t (arithmetic mean)
        const double mu_xp = 0.5*(smu_t[gpu_cell_idx(i+1,j,k)]+smu_t[gpu_cell_idx(i,  j,  k  )]);
        const double mu_xm = 0.5*(smu_t[gpu_cell_idx(i,  j,k)]+smu_t[gpu_cell_idx(i-1,j,  k  )]);
        const double mu_yp = 0.5*(smu_t[gpu_cell_idx(i,j+1,k)]+smu_t[gpu_cell_idx(i,  j,  k  )]);
        const double mu_ym = 0.5*(smu_t[gpu_cell_idx(i,j,  k)]+smu_t[gpu_cell_idx(i,  j-1,k  )]);
        const double mu_zp = 0.5*(smu_t[gpu_cell_idx(i,j,k+1)]+smu_t[gpu_cell_idx(i,  j,  k  )]);
        const double mu_zm = 0.5*(smu_t[gpu_cell_idx(i,j,k  )]+smu_t[gpu_cell_idx(i,  j,  k-1)]);

        // Momentum increments: divergence of SGS stress tensor
        double acc[3] = {0.0, 0.0, 0.0};
        sgs_axis<0>(dQ, ih, ihs, i,j,k, mu_xp, mu_xm, acc);
        sgs_axis<1>(dQ, ih, ihs, i,j,k, mu_yp, mu_ym, acc);
        sgs_axis<2>(dQ, ih, ihs, i,j,k, mu_zp, mu_zm, acc);

        // Cell-centred mu_t and velocity gradients for energy terms
        const double mu_c = smu_t[gpu_cell_idx(i,j,k)];

        // Cell-centred velocity gradients (1/(2h) central differences)
        const double dudx_c = ihs2*(sgs_vel(dQ,0,i+1,j,k)-sgs_vel(dQ,0,i-1,j,k));
        const double dvdy_c = ihs2*(sgs_vel(dQ,1,i,j+1,k)-sgs_vel(dQ,1,i,j-1,k));
        const double dwdz_c = ihs2*(sgs_vel(dQ,2,i,j,k+1)-sgs_vel(dQ,2,i,j,k-1));
        const double dudy_c = ihs2*(sgs_vel(dQ,0,i,j+1,k)-sgs_vel(dQ,0,i,j-1,k));
        const double dudz_c = ihs2*(sgs_vel(dQ,0,i,j,k+1)-sgs_vel(dQ,0,i,j,k-1));
        const double dvdx_c = ihs2*(sgs_vel(dQ,1,i+1,j,k)-sgs_vel(dQ,1,i-1,j,k));
        const double dvdz_c = ihs2*(sgs_vel(dQ,1,i,j,k+1)-sgs_vel(dQ,1,i,j,k-1));
        const double dwdx_c = ihs2*(sgs_vel(dQ,2,i+1,j,k)-sgs_vel(dQ,2,i-1,j,k));
        const double dwdy_c = ihs2*(sgs_vel(dQ,2,i,j+1,k)-sgs_vel(dQ,2,i,j-1,k));
        const double divu_c = dudx_c + dvdy_c + dwdz_c;

        // Cell-centred SGS stress components
        const double txx_c = mu_c*(2.0*dudx_c - (2.0/3.0)*divu_c);
        const double tyy_c = mu_c*(2.0*dvdy_c - (2.0/3.0)*divu_c);
        const double tzz_c = mu_c*(2.0*dwdz_c - (2.0/3.0)*divu_c);
        const double txy_c = mu_c*(dudy_c + dvdx_c);
        const double txz_c = mu_c*(dudz_c + dwdx_c);
        const double tyz_c = mu_c*(dvdz_c + dwdy_c);

        // Mechanical dissipation τ:S (removes resolved KE into SGS modes)
        const double visc_work = txx_c*dudx_c + tyy_c*dvdy_c + tzz_c*dwdz_c
                               + txy_c*(dudy_c+dvdx_c)
                               + txz_c*(dudz_c+dwdx_c)
                               + tyz_c*(dvdz_c+dwdy_c);

        // SGS heat conduction: kap * Laplacian(T)
        const double T_c  = sgs_T(dQ, i,   j,   k  );
        const double lap_T = ih2 * (sgs_T(dQ,i+1,j,  k  ) - 2.0*T_c + sgs_T(dQ,i-1,j,  k  )
                                   +sgs_T(dQ,i,  j+1,k  ) - 2.0*T_c + sgs_T(dQ,i,  j-1,k  )
                                   +sgs_T(dQ,i,  j,  k+1) - 2.0*T_c + sgs_T(dQ,i,  j,  k-1));
        const double heat = mu_c * kap * lap_T;

        // In-place update (no atomics: each thread owns a unique cell)
        const int f = gpu_cell_idx(i, j, k);
        dQ[1*GPU_NCELL+f] += dt * acc[0];
        dQ[2*GPU_NCELL+f] += dt * acc[1];
        dQ[3*GPU_NCELL+f] += dt * acc[2];
        dQ[4*GPU_NCELL+f] += dt * (heat - visc_work);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuSgsList
// ─────────────────────────────────────────────────────────────────────────────
GpuSgsList::~GpuSgsList()
{
    if (d_metas) { cudaFree(d_metas); d_metas = nullptr; }
}

void GpuSgsList::build(const BlockTree& tree, const GpuPool& pool,
                       double Cs, double Pr_t)
{
    const auto& leaves = tree.leaf_indices();
    n_leaves = (int)leaves.size();
    if (n_leaves == 0) return;

    const double SGS_CP  = GPU_GAMMA * GPU_R_GAS / (GPU_GAMMA - 1.0);

    std::vector<GpuSgsMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const BlockNode& nd = tree.nodes[leaves[li]];
        const double h      = nd.block->h;
        h_metas[li].d_Q     = pool.d_Q(nd.block.get());
        h_metas[li].h       = h;
        h_metas[li].Cs2h2   = Cs * Cs * h * h;
        h_metas[li].kap_fac = SGS_CP / Pr_t;
    }

    gpu_upload_meta(d_metas, h_metas);
}

void GpuSgsList::exec(const double* d_dt, cudaStream_t stream) const
{
    if (n_leaves == 0) return;
    constexpr size_t shm = GPU_NCELL * sizeof(double);  // 13824 bytes
    k_sgs_smag<<<dim3(n_leaves), dim3(GPU_NB, GPU_NB), shm, stream>>>(d_metas, d_dt);
}
