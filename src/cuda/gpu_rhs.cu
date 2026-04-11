// gpu_rhs.cu -- GPU convective + viscous RHS kernel
// FIX P3: viscous kernel now computes the full grad(div u) cross-partial term,
//         matching the CPU fix in operators.cpp (FIX C1).
//         Previously only diagonal parts were used (d2u/dx2, d2v/dy2, d2w/dz2).
//
//         Cross-partial stencil (same as CPU, NG=1 sufficient):
//           d2f/dxdy = (f[i+1,j+1] - f[i+1,j-1]
//                      -f[i-1,j+1] + f[i-1,j-1]) / (4h^2)
//
// Thread/shared-memory layout unchanged from original:
//   blockIdx.x  = CFD block index b
//   threadIdx.x = i (0..NB2-1), threadIdx.y = j (0..NB2-1)
//   Each thread loops k = ilo..ihi
//   Shared memory: full block Q (5 vars x NB2^3)

#include "../include/cuda/gpu_hllc.cuh"

#define TILE_SIZE (GPU_NVAR * GPU_NCELL)

__device__ __forceinline__
void load_block_var(const double* __restrict__ src,
                    double* __restrict__ smem_var) {
    int i = threadIdx.x, j = threadIdx.y;
    for (int k = 0; k < GPU_NB2; ++k)
        smem_var[gpu_cell_idx(i,j,k)] = src[gpu_cell_idx(i,j,k)];
}

__global__
void gpu_rhs_kernel(
    const double* __restrict__ Q_in,
    double*       __restrict__ rhs_out,
    int n_blocks,
    double h_inv
) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int i = threadIdx.x, j = threadIdx.y;

    extern __shared__ double smem[];

    size_t stride_var = (size_t)n_blocks * GPU_NCELL;
    for (int v = 0; v < GPU_NVAR; ++v) {
        const double* src = Q_in + (size_t)v * stride_var + (size_t)b * GPU_NCELL;
        load_block_var(src, smem + v * GPU_NCELL);
    }
    __syncthreads();

    auto prim_at = [&](int ii, int jj, int kk) -> GPrim {
        int idx = gpu_cell_idx(ii,jj,kk);
        return gpu_cons_to_prim(
            smem[0*GPU_NCELL+idx], smem[1*GPU_NCELL+idx],
            smem[2*GPU_NCELL+idx], smem[3*GPU_NCELL+idx],
            smem[4*GPU_NCELL+idx]);
    };

    // Zero interior rhs
    for (int k = 0; k < GPU_NB2; ++k) {
        if (i >= gpu_ilo() && i <= gpu_ihi() &&
            j >= gpu_ilo() && j <= gpu_ihi() &&
            k >= gpu_ilo() && k <= gpu_ihi()) {
            int idx = gpu_cell_idx(i,j,k);
            for (int v = 0; v < GPU_NVAR; ++v)
                *(rhs_out + (size_t)v*stride_var + (size_t)b*GPU_NCELL + idx) = 0.0;
        }
    }
    __syncthreads();

    if (i < gpu_ilo() || i > gpu_ihi() || j < gpu_ilo() || j > gpu_ihi()) return;

    const double ih      = h_inv;
    const double ih2     = ih * ih;
    const double ih_half = 0.5 * ih;
    const double ih2_q   = 0.25 * ih2;   // 1/(4h^2) for cross-partial stencil

    for (int k = gpu_ilo(); k <= gpu_ihi(); ++k) {

        // Face-axis neighbours
        GPrim c =prim_at(i,j,k);
        GPrim xm=prim_at(i-1,j,k),  xp=prim_at(i+1,j,k);
        GPrim ym=prim_at(i,j-1,k),  yp=prim_at(i,j+1,k);
        GPrim zm=prim_at(i,j,k-1),  zp=prim_at(i,j,k+1);

        // Convective fluxes
        double Fxp[5],Fxm[5],Gyp[5],Gym[5],Hzp[5],Hzm[5];
        gpu_hllc_flux(c,xp,0,Fxp); gpu_hllc_flux(xm,c,0,Fxm);
        gpu_hllc_flux(c,yp,1,Gyp); gpu_hllc_flux(ym,c,1,Gym);
        gpu_hllc_flux(c,zp,2,Hzp); gpu_hllc_flux(zm,c,2,Hzm);

        int idx = gpu_cell_idx(i,j,k);
        for (int v = 0; v < GPU_NVAR; ++v) {
            double div = ih*((Fxp[v]-Fxm[v])+(Gyp[v]-Gym[v])+(Hzp[v]-Hzm[v]));
            *(rhs_out + (size_t)v*stride_var + (size_t)b*GPU_NCELL + idx) -= div;
        }

        // FIX P3: 12 face-diagonal neighbours for cross-partials
        GPrim xpyp=prim_at(i+1,j+1,k), xpym=prim_at(i+1,j-1,k);
        GPrim xmyp=prim_at(i-1,j+1,k), xmym=prim_at(i-1,j-1,k);
        GPrim xpzp=prim_at(i+1,j,k+1), xpzm=prim_at(i+1,j,k-1);
        GPrim xmzp=prim_at(i-1,j,k+1), xmzm=prim_at(i-1,j,k-1);
        GPrim ypzp=prim_at(i,j+1,k+1), ypzm=prim_at(i,j+1,k-1);
        GPrim ymzp=prim_at(i,j-1,k+1), ymzm=prim_at(i,j-1,k-1);

        // Velocity gradients
        double dudx=ih_half*(xp.u-xm.u), dudy=ih_half*(yp.u-ym.u), dudz=ih_half*(zp.u-zm.u);
        double dvdx=ih_half*(xp.v-xm.v), dvdy=ih_half*(yp.v-ym.v), dvdz=ih_half*(zp.v-zm.v);
        double dwdx=ih_half*(xp.w-xm.w), dwdy=ih_half*(yp.w-ym.w), dwdz=ih_half*(zp.w-zm.w);
        double divU = dudx + dvdy + dwdz;

        double mu    = gpu_sutherland(c.T);
        double kappa = mu * GPU_CP / GPU_PR;

        // Stress tensor
        double txx=mu*(2.0*dudx-(2.0/3.0)*divU);
        double tyy=mu*(2.0*dvdy-(2.0/3.0)*divU);
        double tzz=mu*(2.0*dwdz-(2.0/3.0)*divU);
        double txy=mu*(dudy+dvdx), txz=mu*(dudz+dwdx), tyz=mu*(dvdz+dwdy);

        // Laplacians
        double lap_u=ih2*(xp.u-2*c.u+xm.u + yp.u-2*c.u+ym.u + zp.u-2*c.u+zm.u);
        double lap_v=ih2*(xp.v-2*c.v+xm.v + yp.v-2*c.v+ym.v + zp.v-2*c.v+zm.v);
        double lap_w=ih2*(xp.w-2*c.w+xm.w + yp.w-2*c.w+ym.w + zp.w-2*c.w+zm.w);
        double lap_T=ih2*(xp.T-2*c.T+xm.T + yp.T-2*c.T+ym.T + zp.T-2*c.T+zm.T);

        // Diagonal 2nd derivatives
        double d2u_dx2=ih2*(xp.u-2*c.u+xm.u);
        double d2v_dy2=ih2*(yp.v-2*c.v+ym.v);
        double d2w_dz2=ih2*(zp.w-2*c.w+zm.w);

        // FIX P3: cross-partial 2nd derivatives
        double d2v_dxdy=ih2_q*(xpyp.v-xpym.v-xmyp.v+xmym.v);
        double d2w_dxdz=ih2_q*(xpzp.w-xpzm.w-xmzp.w+xmzm.w);
        double d2u_dydx=ih2_q*(xpyp.u-xpym.u-xmyp.u+xmym.u);
        double d2w_dydz=ih2_q*(ypzp.w-ypzm.w-ymzp.w+ymzm.w);
        double d2u_dzdx=ih2_q*(xpzp.u-xpzm.u-xmzp.u+xmzm.u);
        double d2v_dzdy=ih2_q*(ypzp.v-ypzm.v-ymzp.v+ymzm.v);

        // Full grad(div u) -- FIX P3
        double gdivx = d2u_dx2  + d2v_dxdy + d2w_dxdz;
        double gdivy = d2u_dydx + d2v_dy2  + d2w_dydz;
        double gdivz = d2u_dzdx + d2v_dzdy + d2w_dz2;

        double ax = mu * (lap_u + (1.0/3.0)*gdivx);
        double ay = mu * (lap_v + (1.0/3.0)*gdivy);
        double az = mu * (lap_w + (1.0/3.0)*gdivz);

        double visc_pw = txx*dudx + tyy*dvdy + tzz*dwdz
                       + txy*(dudy+dvdx) + txz*(dudz+dwdx) + tyz*(dvdz+dwdy);
        double heat    = kappa * lap_T;

        *(rhs_out+(size_t)1*stride_var+(size_t)b*GPU_NCELL+idx) += ax;
        *(rhs_out+(size_t)2*stride_var+(size_t)b*GPU_NCELL+idx) += ay;
        *(rhs_out+(size_t)3*stride_var+(size_t)b*GPU_NCELL+idx) += az;
        *(rhs_out+(size_t)4*stride_var+(size_t)b*GPU_NCELL+idx) += heat + visc_pw;
    }
}

// Ghost fill kernel (periodic, single block -- multi-block handled on host)
__global__
void gpu_ghost_periodic_single(double* __restrict__ Q, int n_blocks) {
    if (n_blocks != 1) return;
    int v = blockIdx.x, j = threadIdx.x, k = blockIdx.y;
    double* Qv = Q + (size_t)v * GPU_NCELL;
    Qv[gpu_cell_idx(0,         j,k)] = Qv[gpu_cell_idx(GPU_NB2-2,j,k)];
    Qv[gpu_cell_idx(GPU_NB2-1, j,k)] = Qv[gpu_cell_idx(1,        j,k)];
}
