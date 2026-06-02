#pragma once
// gpu_rhs_recon.cuh — device-inline reconstruction helpers for gpu_rhs.cu.
//
// Includes:
//   weno5z_upwind / gpu_weno5z_scalar    — WENO5-Z scalar (Borges 2008)
//   teno5_upwind  / gpu_teno5_scalar     — TENO5-A scalar (Fu/Hu/Adams 2016)
//   teno7_upwind  / gpu_teno7_scalar     — TENO7-A scalar (Fu/Hu/Adams 2019)
//   GpuRoeState, gpu_roe_from_prim       — Roe-averaged state
//   gpu_char_proj / gpu_back_proj        — characteristic ↔ conservative projection
//   gpu_safe_prim_f                      — NaN-safe prim fallback
//   gpu_weno5_face / gpu_teno5_face / gpu_teno7_face — full face reconstructions
//
// All functions are __device__ __forceinline__; this header is safe to include
// in any .cu TU that needs reconstruction.  It must be included after the GPU
// constants header (gpu_constants.cuh) and the primitive-type header (gpu_rhs.cuh).

#include "cuda/gpu_rhs.cuh"
#include "cuda/gpu_hllc.cuh"
#include "physics/teno5_scalar.hpp"

// One-sided WENO5-Z upwind reconstruction (Borges et al. 2008) — full FP64 reference.
// Production path uses weno5z_upwind_mp (FP32 β/τ/ω, FP64 interpolants).
// Stencil [a,b,c,d,e] = [vm2,vm1,v0,vp1,vp2] for the left state.
// For the right state, pass the mirrored stencil [vp3,vp2,vp1,v0,vm1].
__device__ __forceinline__
double weno5z_upwind(double a, double b, double c, double d, double e) noexcept {
    constexpr double eps = 1.0e-36;
    constexpr double d0 = 0.1, d1 = 0.6, d2 = 0.3;
    const double s0 = ( 2.0*a -  7.0*b + 11.0*c) * (1.0/6.0);
    const double s1 = (    -b +  5.0*c +  2.0*d) * (1.0/6.0);
    const double s2 = ( 2.0*c +  5.0*d -      e) * (1.0/6.0);
    const double b0 = (13.0/12.0)*(a-2.0*b+c)*(a-2.0*b+c)
                    +  (1.0/ 4.0)*(a-4.0*b+3.0*c)*(a-4.0*b+3.0*c);
    const double b1 = (13.0/12.0)*(b-2.0*c+d)*(b-2.0*c+d)
                    +  (1.0/ 4.0)*(b-d)*(b-d);
    const double b2 = (13.0/12.0)*(c-2.0*d+e)*(c-2.0*d+e)
                    +  (1.0/ 4.0)*(3.0*c-4.0*d+e)*(3.0*c-4.0*d+e);
    const double tau5 = fabs(b0 - b2);
    const double a0 = d0*(1.0+(tau5/(b0+eps))*(tau5/(b0+eps)));
    const double a1 = d1*(1.0+(tau5/(b1+eps))*(tau5/(b1+eps)));
    const double a2 = d2*(1.0+(tau5/(b2+eps))*(tau5/(b2+eps)));
    return (a0*s0 + a1*s1 + a2*s2) / (a0 + a1 + a2);
}

// FP32 weights are safe for WENO5-Z because ω_k depends only on the relative magnitude
// of β_k — FP32 roundoff (~1e-7) does not misclassify smooth vs. shocked cells.
// TENO5-A/TENO7-A are excluded: their hard cutoff CT≈1e-6 lies within FP32 roundoff
// (~1e-7) for near-threshold cells, corrupting sub-stencil selection.
__device__ __forceinline__
double weno5z_upwind_mp(double a, double b, double c, double d, double e) noexcept {
    // Phase 2 (FP64): sub-stencil interpolants — computed first, inputs still double
    const double s0 = ( 2.0*a -  7.0*b + 11.0*c) * (1.0/6.0);
    const double s1 = (      -b +  5.0*c +  2.0*d) * (1.0/6.0);
    const double s2 = ( 2.0*c +  5.0*d -       e) * (1.0/6.0);
    // Phase 1 (FP32): smoothness indicators β, global indicator τ₅, WENO-Z weights
    const float fa=(float)a, fb=(float)b, fc=(float)c, fd=(float)d, fe=(float)e;
    const float B0 = (13.f/12.f)*(fa-2.f*fb+fc)*(fa-2.f*fb+fc)
                   +  (1.f/ 4.f)*(fa-4.f*fb+3.f*fc)*(fa-4.f*fb+3.f*fc);
    const float B1 = (13.f/12.f)*(fb-2.f*fc+fd)*(fb-2.f*fc+fd)
                   +  (1.f/ 4.f)*(fb-fd)*(fb-fd);
    const float B2 = (13.f/12.f)*(fc-2.f*fd+fe)*(fc-2.f*fd+fe)
                   +  (1.f/ 4.f)*(3.f*fc-4.f*fd+fe)*(3.f*fc-4.f*fd+fe);
    const float tau5 = fabsf(B0 - B2);
    constexpr float eps32 = 1.e-36f;
    constexpr float c0 = 0.1f, c1 = 0.6f, c2 = 0.3f;
    const float t0 = tau5/(B0+eps32), t1 = tau5/(B1+eps32), t2 = tau5/(B2+eps32);
    const float A0 = c0*(1.f+t0*t0), A1 = c1*(1.f+t1*t1), A2 = c2*(1.f+t2*t2);
    const float iAw = 1.f/(A0+A1+A2);
    const double w0=(double)(A0*iAw), w1=(double)(A1*iAw), w2=(double)(A2*iAw);
    // Phase 2 (FP64) continued: weighted sum
    return w0*s0 + w1*s1 + w2*s2;
}

// WENO5-Z scalar reconstruction (Borges et al. 2008)
__device__ __forceinline__
void gpu_weno5z_scalar(double vm2, double vm1, double v0,
                       double vp1, double vp2, double vp3,
                       double& vL, double& vR) noexcept {
    vL = weno5z_upwind_mp(vm2, vm1, v0,  vp1, vp2);   // left state
    vR = weno5z_upwind_mp(vp3, vp2, vp1, v0,  vm1);   // right state (mirrored)
}

// TENO5-A one-sided upwind reconstruction (Fu, Hu, Adams 2016/2019).
// Stencil [a,b,c,d,e] = [vm2,vm1,v0,vp1,vp2] for the left state.
// For the right state, pass the mirrored stencil [vp3,vp2,vp1,v0,vm1].
__device__ __forceinline__
double teno5_upwind(double a, double b, double c, double d, double e) noexcept {
    constexpr double eps = 1.0e-36;
    constexpr double CT  = 1.0e-5;
    constexpr double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    const double s0 = ( 2.0*a -  7.0*b + 11.0*c) * (1.0/6.0);
    const double s1 = (      -b +  5.0*c +  2.0*d) * (1.0/6.0);
    const double s2 = ( 2.0*c +  5.0*d -       e) * (1.0/6.0);

    const double b0 = (13.0/12.0)*(a-2.0*b+c)*(a-2.0*b+c)
                    +  (1.0/ 4.0)*(a-4.0*b+3.0*c)*(a-4.0*b+3.0*c);
    const double b1 = (13.0/12.0)*(b-2.0*c+d)*(b-2.0*c+d)
                    +  (1.0/ 4.0)*(b-d)*(b-d);
    const double b2 = (13.0/12.0)*(c-2.0*d+e)*(c-2.0*d+e)
                    +  (1.0/ 4.0)*(3.0*c-4.0*d+e)*(3.0*c-4.0*d+e);

    const double tau5 = fabs(b0 - b2);

    // χk = (1 + τ5/(βk+ε))^6 via three squarings
    double r0 = 1.0 + tau5/(b0+eps); r0 *= r0; r0 *= r0; r0 *= r0;
    double r1 = 1.0 + tau5/(b1+eps); r1 *= r1; r1 *= r1; r1 *= r1;
    double r2 = 1.0 + tau5/(b2+eps); r2 *= r2; r2 *= r2; r2 *= r2;
    const double csum_inv = 1.0 / (r0 + r1 + r2 + eps);

    // Include substencil k iff γk = χk/Σχ ≥ C_T; excluded substencils get weight 0.
    const double w0 = (r0 * csum_inv >= CT) ? d0 : 0.0;
    const double w1 = (r1 * csum_inv >= CT) ? d1 : 0.0;
    const double w2 = (r2 * csum_inv >= CT) ? d2 : 0.0;
    const double wsum = w0 + w1 + w2;

    if (wsum > 0.0) return (w0*s0 + w1*s1 + w2*s2) / wsum;

    // ENO fallback: all substencils excluded near strong discontinuity.
    if (b0 <= b1 && b0 <= b2) return s0;
    if (b1 <= b2)              return s1;
    return s2;
}

// TENO5-A scalar reconstruction (Borges/Fu et al.)
__device__ __forceinline__
void gpu_teno5_scalar(double vm2, double vm1, double v0,
                      double vp1, double vp2, double vp3,
                      double& vL, double& vR) noexcept {
    vL = teno5_upwind(vm2, vm1, v0,  vp1, vp2);   // left state
    vR = teno5_upwind(vp3, vp2, vp1, v0,  vm1);   // right state (mirrored)
}

// TENO7-A one-sided reconstruction (Fu, Hu, Adams 2016/2019).
// 7-point stencil; Balsara-Shu (2000) WENO7 smoothness indicators; C_T=1e-6, q=6.
// Optimal weights: d0=1/35, d1=12/35, d2=18/35, d3=4/35.
// Smooth limit: (-3,25,-101,407,70,34,-12)/420 (7th-order combination).
//
// Numerical stability: the Balsara-Shu β formulas are quadratic forms that are
// provably shift-invariant (each coefficient row sums to zero).  Computing them
// in expanded form with raw values triggers Inf*coeff - Inf*coeff = NaN when
// the Roe decomposition maps to large characteristic variables (c_roe→0 clamp).
// Fix: center each 4-point sub-stencil on its local mean before computing β;
// the shifted values are O(differences), preventing overflow without changing
// the mathematical result.
__device__ __forceinline__
double teno7_upwind(double a, double b, double c, double d,
                    double e, double f, double g) noexcept {
    constexpr double eps  = 1.0e-36;
    constexpr double CT   = 1.0e-6;
    constexpr double d0   = 1.0/35.0, d1 = 12.0/35.0, d2 = 18.0/35.0, d3 = 4.0/35.0;
    constexpr double i12  = 1.0/12.0;
    constexpr double i240 = 1.0/240.0;

    const double s0 = i12*(-3.0*a + 13.0*b - 23.0*c + 25.0*d);
    const double s1 = i12*( 1.0*b -  5.0*c + 13.0*d +  3.0*e);
    const double s2 = i12*(-1.0*c +  7.0*d +  7.0*e -  1.0*f);
    const double s3 = i12*(25.0*d - 23.0*e + 13.0*f -  3.0*g);

    // Center each 4-point sub-stencil on its local mean before computing β.
    // β is shift-invariant (row sums of the quadratic form = 0), so the result
    // is identical in exact arithmetic; the centering bounds inputs to O(diff)
    // and prevents the expanded-form Inf - Inf = NaN when values are O(1e304).
    const double m0 = 0.25*(a+b+c+d);
    const double A=a-m0, B=b-m0, C=c-m0, D=d-m0;
    const double m1 = 0.25*(b+c+d+e);
    const double Bb=b-m1, Cb=c-m1, Db=d-m1, Eb=e-m1;
    const double m2 = 0.25*(c+d+e+f);
    const double Cc=c-m2, Dc=d-m2, Ec=e-m2, Fc=f-m2;
    const double m3 = 0.25*(d+e+f+g);
    const double Dd=d-m3, Ed=e-m3, Fd=f-m3, Gd=g-m3;

    const double b0 = i240*(547.0*A*A - 3882.0*A*B + 4642.0*A*C - 1854.0*A*D
                           + 7043.0*B*B - 17246.0*B*C + 7042.0*B*D
                           + 11003.0*C*C - 9402.0*C*D + 2107.0*D*D);
    const double b1 = i240*(267.0*Bb*Bb - 1642.0*Bb*Cb + 1602.0*Bb*Db - 494.0*Bb*Eb
                           + 2843.0*Cb*Cb - 5966.0*Cb*Db + 1922.0*Cb*Eb
                           + 3443.0*Db*Db - 2522.0*Db*Eb + 547.0*Eb*Eb);
    const double b2 = i240*(267.0*Cc*Cc - 1642.0*Cc*Dc + 1602.0*Cc*Ec - 494.0*Cc*Fc
                           + 2843.0*Dc*Dc - 5966.0*Dc*Ec + 1922.0*Dc*Fc
                           + 3443.0*Ec*Ec - 2522.0*Ec*Fc + 547.0*Fc*Fc);
    const double b3 = i240*(2107.0*Dd*Dd - 9402.0*Dd*Ed + 7042.0*Dd*Fd - 1854.0*Dd*Gd
                           + 11003.0*Ed*Ed - 17246.0*Ed*Fd + 4642.0*Ed*Gd
                           + 7043.0*Fd*Fd - 3882.0*Fd*Gd + 547.0*Gd*Gd);

    const double tau7 = fabs(b0 - b3);
    double r0 = 1.0 + tau7/(b0+eps); r0 *= r0; r0 *= r0; r0 *= r0;
    double r1 = 1.0 + tau7/(b1+eps); r1 *= r1; r1 *= r1; r1 *= r1;
    double r2 = 1.0 + tau7/(b2+eps); r2 *= r2; r2 *= r2; r2 *= r2;
    double r3 = 1.0 + tau7/(b3+eps); r3 *= r3; r3 *= r3; r3 *= r3;
    const double ci = 1.0 / (r0 + r1 + r2 + r3 + eps);

    const double w0 = (r0*ci >= CT) ? d0 : 0.0;
    const double w1 = (r1*ci >= CT) ? d1 : 0.0;
    const double w2 = (r2*ci >= CT) ? d2 : 0.0;
    const double w3 = (r3*ci >= CT) ? d3 : 0.0;
    const double ws  = w0 + w1 + w2 + w3;
    if (ws > 0.0) {
        // Guard: if only s3 (the downwind sub-stencil) is selected while all
        // upwind sub-stencils s0/s1/s2 are cut, the stencil has crossed to the
        // wrong side of a discontinuity.  Return NaN so the caller's safe_prim
        // (which tests !(x > 0), catching NaN) falls back to the cell-centre.
        if (w0 + w1 + w2 == 0.0) return 0.0 / 0.0;
        return (w0*s0 + w1*s1 + w2*s2 + w3*s3) / ws;
    }

    if (b0 <= b1 && b0 <= b2 && b0 <= b3) return s0;
    if (b1 <= b2 && b1 <= b3)              return s1;
    if (b2 <= b3)                           return s2;
    return s3;
}

__device__ __forceinline__
void gpu_teno7_scalar(double vm3, double vm2, double vm1, double v0,
                      double vp1, double vp2, double vp3,
                      double& vL, double& vR) noexcept {
    vL = teno7_upwind(vm3, vm2, vm1, v0,  vp1, vp2, vp3);
    vR = teno7_upwind(vp3, vp2, vp1, v0,  vm1, vm2, vm3);
}

// ── Shared Roe-decomposition helpers (used by all five face reconstruction fns) ─
struct GpuRoeState {
    double un, ut1, ut2;
    double H_roe, c_roe, KE, bv, b2v, ioc;
    int nidx, t1idx, t2idx;
};

__device__ __forceinline__
GpuRoeState gpu_roe_from_prim(
    double rL, double uL, double vL, double wL, double pL,
    double rR, double uR, double vR, double wR, double pR,
    int axis) noexcept
{
    const double EL    = pL/(GPU_GAMMA-1.0) + 0.5*rL*(uL*uL+vL*vL+wL*wL);
    const double ER    = pR/(GPU_GAMMA-1.0) + 0.5*rR*(uR*uR+vR*vR+wR*wR);
    const double sqL   = sqrt(rL), sqR = sqrt(rR), denom = sqL+sqR;
    const double u_roe = (sqL*uL + sqR*uR)/denom;
    const double v_roe = (sqL*vL + sqR*vR)/denom;
    const double w_roe = (sqL*wL + sqR*wR)/denom;
    const double H_roe = (sqL*(EL+pL)/rL + sqR*(ER+pR)/rR)/denom;
    const double KE    = 0.5*(u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
    const double c2    = fmax((GPU_GAMMA-1.0)*(H_roe-KE), 1.0e-300);
    const double c_roe = sqrt(c2);
    const double un    = (axis==0)?u_roe:(axis==1)?v_roe:w_roe;
    const double ut1   = (axis==0)?v_roe:u_roe;
    const double ut2   = (axis==0)?w_roe:(axis==2)?v_roe:w_roe;
    const int    nidx  = 1+axis;
    const int    t1idx = (axis==0)?2:1;
    const int    t2idx = (axis==2)?2:3;
    const double bv    = (GPU_GAMMA-1.0)/c2;
    return {un, ut1, ut2, H_roe, c_roe, KE, bv, bv*KE, 1.0/c_roe, nidx, t1idx, t2idx};
}

__device__ __forceinline__
void gpu_char_proj(const double Q[GPU_NVAR], const GpuRoeState& rs, double W[5]) noexcept
{
    const double qn  = Q[rs.nidx], qt1 = Q[rs.t1idx], qt2 = Q[rs.t2idx];
    const double inn   = rs.b2v*Q[0] - rs.bv*(rs.un*qn+rs.ut1*qt1+rs.ut2*qt2) + rs.bv*Q[4];
    const double del_n = rs.ioc*(rs.un*Q[0] - qn);
    W[0] = 0.5*(inn + del_n);
    W[1] = (1.0-rs.b2v)*Q[0] + rs.bv*(rs.un*qn+rs.ut1*qt1+rs.ut2*qt2) - rs.bv*Q[4];
    W[2] = -rs.ut1*Q[0] + qt1;
    W[3] = -rs.ut2*Q[0] + qt2;
    W[4] = 0.5*(inn - del_n);
}

__device__ __forceinline__
void gpu_back_proj(const double w[5], const GpuRoeState& rs, double Qrec[GPU_NVAR]) noexcept
{
    const double w014 = w[0]+w[1]+w[4], dw04 = w[4]-w[0];
    Qrec[0]          = w014;
    Qrec[rs.nidx]    = w014*rs.un  + dw04*rs.c_roe;
    Qrec[rs.t1idx]   = w014*rs.ut1 + w[2];
    Qrec[rs.t2idx]   = w014*rs.ut2 + w[3];
    Qrec[4]          = (w[0]+w[4])*rs.H_roe + dw04*rs.un*rs.c_roe
                     + w[1]*rs.KE + w[2]*rs.ut1 + w[3]*rs.ut2;
}

__device__ __forceinline__
GPrim gpu_safe_prim_f(const double Qc[GPU_NVAR], const GPrim& fb) noexcept
{
    const double rho = Qc[0]; if (!(rho > 0.0)) return fb;
    const double u = Qc[1]/rho, v = Qc[2]/rho, w = Qc[3]/rho;
    const double p = (GPU_GAMMA-1.0)*(Qc[4]-0.5*rho*(u*u+v*v+w*w));
    if (!(p > 0.0)) return fb;
    GPrim q; q.rho=rho; q.u=u; q.v=v; q.w=w;
    q.p=p; q.T=p/(rho*GPU_R_GAS); q.c=sqrt(GPU_GAMMA*p/rho);
    return q;
}

// WENO5 face reconstruction with Roe characteristic decomposition.
// Reads prim from d_scratch (comp-major: sp[comp*NCELL + flat]).
// (i,j,k) = left cell of face; axis = normal direction.
// Requires NG=2: stencil offset d ∈ {-2,-1,0,+1,+2,+3} all in-bounds.
__device__ __forceinline__
void gpu_weno5_face(const double* __restrict__ sp,
                    int i, int j, int k, int axis,
                    GPrim& qL_out, GPrim& qR_out) noexcept {
    auto sidx = [&](int d) -> int {
        if (axis == 0) return gpu_cell_idx(i+d, j, k);
        if (axis == 1) return gpu_cell_idx(i, j+d, k);
        return                gpu_cell_idx(i, j, k+d);
    };

    const int fL = sidx(0), fR = sidx(1);
    const double rL = sp[0*GPU_NCELL+fL], uL = sp[1*GPU_NCELL+fL];
    const double vL = sp[2*GPU_NCELL+fL], wL = sp[3*GPU_NCELL+fL];
    const double pL = sp[4*GPU_NCELL+fL], TL = sp[5*GPU_NCELL+fL], cL = sp[6*GPU_NCELL+fL];
    const double rR = sp[0*GPU_NCELL+fR], uR = sp[1*GPU_NCELL+fR];
    const double vR = sp[2*GPU_NCELL+fR], wR = sp[3*GPU_NCELL+fR];
    const double pR = sp[4*GPU_NCELL+fR], TR = sp[5*GPU_NCELL+fR], cR = sp[6*GPU_NCELL+fR];
    const GpuRoeState rs = gpu_roe_from_prim(rL,uL,vL,wL,pL, rR,uR,vR,wR,pR, axis);

    double Q[6][GPU_NVAR];
    for (int m = 0; m < 6; ++m) {
        const int flat = sidx(m-2);
        const double rho = sp[0*GPU_NCELL+flat], u = sp[1*GPU_NCELL+flat];
        const double v   = sp[2*GPU_NCELL+flat], w = sp[3*GPU_NCELL+flat];
        const double p   = sp[4*GPU_NCELL+flat];
        Q[m][0] = rho; Q[m][1] = rho*u; Q[m][2] = rho*v; Q[m][3] = rho*w;
        Q[m][4] = p/(GPU_GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }

    double W[5][6];
    for (int m = 0; m < 6; ++m) { double Wm[5]; gpu_char_proj(Q[m], rs, Wm); for (int c=0;c<5;++c) W[c][m]=Wm[c]; }

    double wL_w[5], wR_w[5];
    for (int kk = 0; kk < 5; ++kk)
        gpu_weno5z_scalar(W[kk][0],W[kk][1],W[kk][2],W[kk][3],W[kk][4],W[kk][5], wL_w[kk], wR_w[kk]);

    double QL[GPU_NVAR], QR[GPU_NVAR];
    gpu_back_proj(wL_w, rs, QL);
    gpu_back_proj(wR_w, rs, QR);
    GPrim fbL; fbL.rho=rL; fbL.u=uL; fbL.v=vL; fbL.w=wL; fbL.p=pL; fbL.T=TL; fbL.c=cL;
    GPrim fbR; fbR.rho=rR; fbR.u=uR; fbR.v=vR; fbR.w=wR; fbR.p=pR; fbR.T=TR; fbR.c=cR;
    qL_out = gpu_safe_prim_f(QL, fbL);
    qR_out = gpu_safe_prim_f(QR, fbR);
}

// D3: TENO5-A face reconstruction — same Roe decomposition as gpu_weno5_face;
// scalar kernel replaced by gpu_teno5_scalar (hard cutoff, q=6 exponent).
__device__ __forceinline__
void gpu_teno5_face(const double* __restrict__ sp,
                    int i, int j, int k, int axis,
                    GPrim& qL_out, GPrim& qR_out) noexcept {
    auto sidx = [&](int d) -> int {
        if (axis == 0) { int ii=i+d; if(ii<0)ii+=GPU_NB; else if(ii>=GPU_NB2)ii-=GPU_NB; return gpu_cell_idx(ii,j,k); }
        if (axis == 1) { int jj=j+d; if(jj<0)jj+=GPU_NB; else if(jj>=GPU_NB2)jj-=GPU_NB; return gpu_cell_idx(i,jj,k); }
        int kk=k+d; if(kk<0)kk+=GPU_NB; else if(kk>=GPU_NB2)kk-=GPU_NB; return gpu_cell_idx(i,j,kk);
    };

    const int fL = sidx(0), fR = sidx(1);
    const double rL = sp[0*GPU_NCELL+fL], uL = sp[1*GPU_NCELL+fL];
    const double vL = sp[2*GPU_NCELL+fL], wL = sp[3*GPU_NCELL+fL];
    const double pL = sp[4*GPU_NCELL+fL], TL = sp[5*GPU_NCELL+fL], cL = sp[6*GPU_NCELL+fL];
    const double rR = sp[0*GPU_NCELL+fR], uR = sp[1*GPU_NCELL+fR];
    const double vR = sp[2*GPU_NCELL+fR], wR = sp[3*GPU_NCELL+fR];
    const double pR = sp[4*GPU_NCELL+fR], TR = sp[5*GPU_NCELL+fR], cR = sp[6*GPU_NCELL+fR];
    const GpuRoeState rs = gpu_roe_from_prim(rL,uL,vL,wL,pL, rR,uR,vR,wR,pR, axis);

    double Q[6][GPU_NVAR];
    for (int m = 0; m < 6; ++m) {
        const int flat = sidx(m-2);
        const double rho = sp[0*GPU_NCELL+flat], u = sp[1*GPU_NCELL+flat];
        const double v   = sp[2*GPU_NCELL+flat], w = sp[3*GPU_NCELL+flat];
        const double p   = sp[4*GPU_NCELL+flat];
        Q[m][0] = rho; Q[m][1] = rho*u; Q[m][2] = rho*v; Q[m][3] = rho*w;
        Q[m][4] = p/(GPU_GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }

    double W[5][6];
    for (int m = 0; m < 6; ++m) { double Wm[5]; gpu_char_proj(Q[m], rs, Wm); for (int c=0;c<5;++c) W[c][m]=Wm[c]; }

    double wL_w[5], wR_w[5];
    for (int kk = 0; kk < 5; ++kk)
        physics_teno5_scalar(W[kk][0],W[kk][1],W[kk][2],W[kk][3],W[kk][4],W[kk][5], wL_w[kk], wR_w[kk]);

    double QL[GPU_NVAR], QR[GPU_NVAR];
    gpu_back_proj(wL_w, rs, QL);
    gpu_back_proj(wR_w, rs, QR);
    GPrim fbL; fbL.rho=rL; fbL.u=uL; fbL.v=vL; fbL.w=wL; fbL.p=pL; fbL.T=TL; fbL.c=cL;
    GPrim fbR; fbR.rho=rR; fbR.u=uR; fbR.v=vR; fbR.w=wR; fbR.p=pR; fbR.T=TR; fbR.c=cR;
    qL_out = gpu_safe_prim_f(QL, fbL);
    qR_out = gpu_safe_prim_f(QR, fbR);
}

// D3: TENO7-A face reconstruction — 7-point stencil with Roe decomposition.
// At face fn=2, d=-3 maps to index -1 which is outside the ghost layer.
// The correct periodic cell is index -1+NB = 7 (an interior cell).
// Using periodic wrap-around (not clamping) ensures fn=2 uses the same
// physical stencil as fn=3..8 and preserves axis symmetry (A73).
__device__ __forceinline__
void gpu_teno7_face(const double* __restrict__ sp,
                    int i, int j, int k, int axis,
                    GPrim& qL_out, GPrim& qR_out) noexcept {
    auto sidx = [&](int d) -> int {
        if (axis == 0) { int ii=i+d; if(ii<0)ii+=GPU_NB; else if(ii>=GPU_NB2)ii-=GPU_NB; return gpu_cell_idx(ii,j,k); }
        if (axis == 1) { int jj=j+d; if(jj<0)jj+=GPU_NB; else if(jj>=GPU_NB2)jj-=GPU_NB; return gpu_cell_idx(i,jj,k); }
        int kk=k+d; if(kk<0)kk+=GPU_NB; else if(kk>=GPU_NB2)kk-=GPU_NB; return gpu_cell_idx(i,j,kk);
    };

    const int fL = sidx(0), fR = sidx(1);
    const double rL = sp[0*GPU_NCELL+fL], uL = sp[1*GPU_NCELL+fL];
    const double vL = sp[2*GPU_NCELL+fL], wL = sp[3*GPU_NCELL+fL];
    const double pL = sp[4*GPU_NCELL+fL], TL = sp[5*GPU_NCELL+fL], cL = sp[6*GPU_NCELL+fL];
    const double rR = sp[0*GPU_NCELL+fR], uR = sp[1*GPU_NCELL+fR];
    const double vR = sp[2*GPU_NCELL+fR], wR = sp[3*GPU_NCELL+fR];
    const double pR = sp[4*GPU_NCELL+fR], TR = sp[5*GPU_NCELL+fR], cR = sp[6*GPU_NCELL+fR];
    const GpuRoeState rs = gpu_roe_from_prim(rL,uL,vL,wL,pL, rR,uR,vR,wR,pR, axis);

    double Q[7][GPU_NVAR];
    for (int m = 0; m < 7; ++m) {
        const int flat = sidx(m-3);
        const double rho = sp[0*GPU_NCELL+flat], u = sp[1*GPU_NCELL+flat];
        const double v   = sp[2*GPU_NCELL+flat], w = sp[3*GPU_NCELL+flat];
        const double p   = sp[4*GPU_NCELL+flat];
        Q[m][0] = rho; Q[m][1] = rho*u; Q[m][2] = rho*v; Q[m][3] = rho*w;
        Q[m][4] = p/(GPU_GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }

    double W[5][7];
    for (int m = 0; m < 7; ++m) { double Wm[5]; gpu_char_proj(Q[m], rs, Wm); for (int c=0;c<5;++c) W[c][m]=Wm[c]; }

    double wL_w[5], wR_w[5];
    for (int kk = 0; kk < 5; ++kk)
        gpu_teno7_scalar(W[kk][0],W[kk][1],W[kk][2],W[kk][3],W[kk][4],W[kk][5],W[kk][6], wL_w[kk], wR_w[kk]);

    // Fallback to TENO5-A if downwind-only NaN sentinel fired for any characteristic.
    for (int kk = 0; kk < 5; ++kk)
        if (!isfinite(wL_w[kk]) || !isfinite(wR_w[kk])) {
            gpu_teno5_face(sp, i, j, k, axis, qL_out, qR_out); return;
        }

    double QL[GPU_NVAR], QR[GPU_NVAR];
    gpu_back_proj(wL_w, rs, QL);
    gpu_back_proj(wR_w, rs, QR);
    GPrim fbL; fbL.rho=rL; fbL.u=uL; fbL.v=vL; fbL.w=wL; fbL.p=pL; fbL.T=TL; fbL.c=cL;
    GPrim fbR; fbR.rho=rR; fbR.u=uR; fbR.v=vR; fbR.w=wR; fbR.p=pR; fbR.T=TR; fbR.c=cR;
    qL_out = gpu_safe_prim_f(QL, fbL);
    qR_out = gpu_safe_prim_f(QR, fbR);
}
