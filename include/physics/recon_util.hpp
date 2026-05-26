#pragma once
// recon_util.hpp — shared primitives for TENO reconstruction
// prim_to_cons, safe_prim, RoeState, roe_state<DIR>, char_project_one, back_project

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"
#include "mesh/axis.hpp"
#include <cmath>

__host__ __device__ inline void prim_to_cons(const Prim& p, double Q[NVAR]) noexcept {
    Q[0] = p.rho;
    Q[1] = p.rho * p.u;
    Q[2] = p.rho * p.v;
    Q[3] = p.rho * p.w;
    Q[4] = (p.p + p.gamma_m*p.p_inf_m)/(p.gamma_m-1.0)
          + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
}

__host__ __device__ inline Prim safe_prim(const double Qc[NVAR], const Prim& fb) noexcept {
    const double rho = Qc[0];
    if (rho <= 0.0) return fb;
    const double u = Qc[1]/rho, v = Qc[2]/rho, w = Qc[3]/rho;
    const double gm = fb.gamma_m, pim = fb.p_inf_m;
    const double p = (gm-1.0)*(Qc[4] - 0.5*rho*(u*u+v*v+w*w)) - gm*pim;
    if (p + pim <= 0.0) return fb;
    Prim q; q.rho=rho; q.u=u; q.v=v; q.w=w; q.p=p;
    q.gamma_m=gm; q.p_inf_m=pim;
    q.T=(p+pim)/(rho*R_GAS); q.c=std::sqrt(gm*(p+pim)/rho);
    return q;
}

// Roe-average face state for characteristic decomposition
struct RoeState {
    double un, ut1, ut2;        // Roe-avg normal and tangential velocities
    double H_roe, c_roe;        // total enthalpy and sound speed
    double KE, b, b2, ioc;     // ½|u|², (γ-1)/c², b*KE, 1/c_roe
    int n_idx, t1_idx, t2_idx; // conserved momentum indices for normal/tangential axes
};

template<Axis DIR>
__host__ __device__ inline RoeState roe_state(const Prim& pL, const Prim& pR) noexcept {
    const double sqL = std::sqrt(pL.rho), sqR = std::sqrt(pR.rho);
    const double denom = sqL + sqR;
    const double u_roe = (sqL*pL.u + sqR*pR.u)/denom;
    const double v_roe = (sqL*pL.v + sqR*pR.v)/denom;
    const double w_roe = (sqL*pL.w + sqR*pR.w)/denom;
    const double EL = (pL.p + pL.gamma_m*pL.p_inf_m)/(pL.gamma_m-1.0)
                    + 0.5*pL.rho*(pL.u*pL.u + pL.v*pL.v + pL.w*pL.w);
    const double ER = (pR.p + pR.gamma_m*pR.p_inf_m)/(pR.gamma_m-1.0)
                    + 0.5*pR.rho*(pR.u*pR.u + pR.v*pR.v + pR.w*pR.w);
    const double H_roe = (sqL*(EL+pL.p)/pL.rho + sqR*(ER+pR.p)/pR.rho)/denom;
    const double KE    = 0.5*(u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
    const double gm_roe= 0.5*(pL.gamma_m + pR.gamma_m);
    const double c2    = std::max((gm_roe-1.0)*(H_roe - KE), 1.0e-300);
    const double c_roe = std::sqrt(c2);
    constexpr int n_idx  = (DIR==Axis::X)?1:(DIR==Axis::Y)?2:3;
    constexpr int t1_idx = (DIR==Axis::X)?2:1;
    constexpr int t2_idx = (DIR==Axis::Z)?2:3;
    const double un  = (DIR==Axis::X)?u_roe:(DIR==Axis::Y)?v_roe:w_roe;
    const double ut1 = (DIR==Axis::X)?v_roe:u_roe;
    const double ut2 = (DIR==Axis::Z)?v_roe:w_roe;
    const double b   = (gm_roe-1.0)/c2;
    return {un, ut1, ut2, H_roe, c_roe, KE, b, b*KE, 1.0/c_roe, n_idx, t1_idx, t2_idx};
}

// Characteristic projection P: conserved Q[NVAR] → char W[5] (one stencil point)
__host__ __device__ inline void char_project_one(
    const double Q[NVAR], const RoeState& rs, double W[5]) noexcept
{
    const double rho=Q[0], qn=Q[rs.n_idx], qt1=Q[rs.t1_idx], qt2=Q[rs.t2_idx], E=Q[4];
    const double inn   = rs.b2*rho - rs.b*(rs.un*qn+rs.ut1*qt1+rs.ut2*qt2) + rs.b*E;
    const double del_n = rs.ioc*(rs.un*rho - qn);
    W[0] = 0.5*(inn + del_n);
    W[1] = (1.0-rs.b2)*rho + rs.b*(rs.un*qn+rs.ut1*qt1+rs.ut2*qt2) - rs.b*E;
    W[2] = -rs.ut1*rho + qt1;
    W[3] = -rs.ut2*rho + qt2;
    W[4] = 0.5*(inn - del_n);
}

// Back-projection P⁻¹: char w[5] → conserved Qrec[NVAR]
__host__ __device__ inline void back_project(
    const double w[5], const RoeState& rs, double Qrec[NVAR]) noexcept
{
    const double w014 = w[0]+w[1]+w[4], dw04 = w[4]-w[0];
    Qrec[0]          = w014;
    Qrec[rs.n_idx]   = w014*rs.un  + dw04*rs.c_roe;
    Qrec[rs.t1_idx]  = w014*rs.ut1 + w[2];
    Qrec[rs.t2_idx]  = w014*rs.ut2 + w[3];
    Qrec[4]          = (w[0]+w[4])*rs.H_roe + dw04*rs.un*rs.c_roe
                     + w[1]*rs.KE + w[2]*rs.ut1 + w[3]*rs.ut2;
}
