#pragma once
// Layer P — Teno5Recon<DIR> physics functor (D3)
// TENO5-A reconstruction with Roe characteristic decomposition.
// Identical architecture to Weno5Recon; scalar kernel replaced by
// physics_teno5_scalar (hard cutoff, q=6 exponent).

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"
#include "schemes/concepts.hpp"
#include "mesh/axis.hpp"
#include "physics/teno5_scalar.hpp"
#include <cmath>

template<Axis DIR>
struct Teno5Recon {
    __host__ __device__
    void operator()(const Prim* pc, int i, int j, int k,
                    Prim& qL_out, Prim& qR_out) const noexcept {
        auto idx_at = [&](int d) noexcept -> int {
            if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
            if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
            return                              cell_idx(i, j, k+d);
        };

        double Q[6][NVAR];
        for (int m = 0; m < 6; ++m) {
            const Prim& p = pc[idx_at(m - 2)];
            Q[m][0] = p.rho;
            Q[m][1] = p.rho * p.u;
            Q[m][2] = p.rho * p.v;
            Q[m][3] = p.rho * p.w;
            Q[m][4] = (p.p + p.gamma_m*p.p_inf_m)/(p.gamma_m-1.0)
                      + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
        }

        const Prim& pL = pc[idx_at(0)];
        const Prim& pR = pc[idx_at(1)];
        const double sqL   = std::sqrt(pL.rho);
        const double sqR   = std::sqrt(pR.rho);
        const double denom = sqL + sqR;
        const double u_roe = (sqL*pL.u + sqR*pR.u) / denom;
        const double v_roe = (sqL*pL.v + sqR*pR.v) / denom;
        const double w_roe = (sqL*pL.w + sqR*pR.w) / denom;
        const double HL    = (Q[2][4] + pL.p) / pL.rho;
        const double HR    = (Q[3][4] + pR.p) / pR.rho;
        const double H_roe = (sqL*HL + sqR*HR) / denom;
        const double KE    = 0.5*(u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
        const double gm_roe = 0.5*(pL.gamma_m + pR.gamma_m);
        const double c2     = std::max((gm_roe-1.0)*(H_roe - KE), 1.0e-300);
        const double c_roe  = std::sqrt(c2);

        constexpr int n_idx  = (DIR==Axis::X) ? 1 : (DIR==Axis::Y) ? 2 : 3;
        constexpr int t1_idx = (DIR==Axis::X) ? 2 : 1;
        constexpr int t2_idx = (DIR==Axis::Z) ? 2 : 3;
        const double un  = (DIR==Axis::X) ? u_roe : (DIR==Axis::Y) ? v_roe : w_roe;
        const double ut1 = (DIR==Axis::X) ? v_roe : u_roe;
        const double ut2 = (DIR==Axis::Z) ? v_roe : w_roe;

        const double b   = (gm_roe-1.0) / c2;
        const double b2  = b * KE;
        const double ioc = 1.0 / c_roe;

        double W[5][6];
        for (int m = 0; m < 6; ++m) {
            const double rho = Q[m][0];
            const double qn  = Q[m][n_idx ];
            const double qt1 = Q[m][t1_idx];
            const double qt2 = Q[m][t2_idx];
            const double E   = Q[m][4];
            const double inner   = b2*rho - b*(un*qn + ut1*qt1 + ut2*qt2) + b*E;
            const double delta_n = ioc*(un*rho - qn);
            W[0][m] = 0.5*(inner + delta_n);
            W[1][m] = (1.0 - b2)*rho + b*(un*qn + ut1*qt1 + ut2*qt2) - b*E;
            W[2][m] = -ut1*rho + qt1;
            W[3][m] = -ut2*rho + qt2;
            W[4][m] = 0.5*(inner - delta_n);
        }

        // TENO5-A scalar per characteristic variable (replaces WENO5-Z)
        double wL[5], wR[5];
        for (int kk = 0; kk < 5; ++kk)
            physics_teno5_scalar(W[kk][0], W[kk][1], W[kk][2],
                                 W[kk][3], W[kk][4], W[kk][5],
                                 wL[kk], wR[kk]);

        auto back_project = [&](const double w[5], double Qrec[NVAR]) noexcept {
            const double w014 = w[0] + w[1] + w[4];
            const double dw04 = w[4] - w[0];
            Qrec[0]      = w014;
            Qrec[n_idx]  = w014*un  + dw04*c_roe;
            Qrec[t1_idx] = w014*ut1 + w[2];
            Qrec[t2_idx] = w014*ut2 + w[3];
            Qrec[4]      = (w[0]+w[4])*H_roe + dw04*un*c_roe
                         + w[1]*KE + w[2]*ut1 + w[3]*ut2;
        };

        double QL[NVAR], QRv[NVAR];
        back_project(wL, QL);
        back_project(wR, QRv);

        auto safe_prim = [](const double Qc[NVAR], const Prim& fallback) noexcept -> Prim {
            const double rho = Qc[0];
            if (rho <= 0.0) return fallback;
            const double u   = Qc[1] / rho;
            const double v   = Qc[2] / rho;
            const double w   = Qc[3] / rho;
            const double gm  = fallback.gamma_m;
            const double pim = fallback.p_inf_m;
            const double p   = (gm-1.0)*(Qc[4] - 0.5*rho*(u*u+v*v+w*w)) - gm*pim;
            if (p + pim <= 0.0) return fallback;
            Prim q; q.rho=rho; q.u=u; q.v=v; q.w=w; q.p=p;
            q.gamma_m=gm; q.p_inf_m=pim;
            q.T=(p+pim)/(rho*R_GAS); q.c=std::sqrt(gm*(p+pim)/rho);
            return q;
        };

        qL_out = safe_prim(QL,  pL);
        qR_out = safe_prim(QRv, pR);
    }
};

static_assert(SpatialReconstruction<Teno5Recon<Axis::X>>);
