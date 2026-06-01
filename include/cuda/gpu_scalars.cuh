#pragma once
#include "gpu_constants.cuh"

// snap_scalar_val: compute a visualised/probe scalar at interior cell (ci,cj,ck).
// var_id: 0=rho, 1=p, 2=T, 3=|u|, 4=rhou, 5=rhov, 6=rhow, 7=E,
//         8=Mach, 9=|omega|, 10=Q_cr, 11=schlieren(|grad rho|)
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
