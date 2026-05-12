#pragma once
// Layer P — Weno5Recon<DIR> physics functor (CLAUDE.md R2)

// Compatibility: __host__ and __device__ are empty in non-CUDA (GCC/Clang) TUs.
#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif
// WENO5-Z reconstruction with Roe characteristic decomposition (Borges 2008).
// __host__ __device__; template<Axis DIR>; no execution knowledge.

#include "cell_block.hpp"           // Prim, NVAR, cell_idx, R_GAS
#include "concepts.hpp"             // SpatialReconstruction concept
#include "axis.hpp"                 // Axis
#include "physics/weno5z_scalar.hpp"
#include <cmath>

template<Axis DIR>
struct Weno5Recon {
    __host__ __device__
    void operator()(const Prim* pc, int i, int j, int k,
                    Prim& qL_out, Prim& qR_out) const noexcept {
        // Stencil index: offset d along the compile-time axis
        auto idx_at = [&](int d) noexcept -> int {
            if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
            if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
            return                              cell_idx(i, j, k+d);
        };

        // Convert 6 prim states to conservative Q[m][v], m∈[0,5]
        // m=0 → cell i-2,  m=2 → cell i,  m=3 → cell i+1,  m=5 → cell i+3
        double Q[6][NVAR];
        for (int m = 0; m < 6; ++m) {
            const Prim& p = pc[idx_at(m - 2)];
            Q[m][0] = p.rho;
            Q[m][1] = p.rho * p.u;
            Q[m][2] = p.rho * p.v;
            Q[m][3] = p.rho * p.w;
            // P14.1: stiffened-gas E
            Q[m][4] = (p.p + p.gamma_m*p.p_inf_m)/(p.gamma_m-1.0)
                      + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
        }

        // Roe-averaged state between cells i (m=2) and i+1 (m=3)
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
        // P14.1: face-averaged mixture γ for Roe sound speed
        const double gm_roe = 0.5*(pL.gamma_m + pR.gamma_m);
        const double c2     = std::max((gm_roe-1.0)*(H_roe - KE), 1.0e-300);
        const double c_roe  = std::sqrt(c2);

        // Compile-time normal/tangential velocity and Q-array index selection
        //   axis=X: normal=x(idx 1), t1=y(idx 2), t2=z(idx 3)
        //   axis=Y: normal=y(idx 2), t1=x(idx 1), t2=z(idx 3)
        //   axis=Z: normal=z(idx 3), t1=x(idx 1), t2=y(idx 2)
        constexpr int n_idx  = (DIR==Axis::X) ? 1 : (DIR==Axis::Y) ? 2 : 3;
        constexpr int t1_idx = (DIR==Axis::X) ? 2 : 1;
        constexpr int t2_idx = (DIR==Axis::Z) ? 2 : 3;
        const double un  = (DIR==Axis::X) ? u_roe : (DIR==Axis::Y) ? v_roe : w_roe;
        const double ut1 = (DIR==Axis::X) ? v_roe : u_roe;
        const double ut2 = (DIR==Axis::Z) ? v_roe : w_roe;

        const double b   = (gm_roe-1.0) / c2;  // (γ-1)/c²  P14.1
        const double b2  = b * KE;              // (γ-1)*KE/c²
        const double ioc = 1.0 / c_roe;

        // Project 6 conservative states to characteristic space
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

        // Apply WENO5-Z to each characteristic variable independently
        double wL[5], wR[5];
        for (int kk = 0; kk < 5; ++kk)
            physics_weno5z_scalar(W[kk][0], W[kk][1], W[kk][2],
                                  W[kk][3], W[kk][4], W[kk][5],
                                  wL[kk], wR[kk]);

        // Back-project with right eigenvectors to conservative space
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

        // Convert conservative → primitive; fall back to cell-center if non-physical.
        // P14.1: use fallback's mixture EOS for reconstructed state validation.
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

// Compile-time concept check (Layer C)
static_assert(SpatialReconstruction<Weno5Recon<Axis::X>>);
