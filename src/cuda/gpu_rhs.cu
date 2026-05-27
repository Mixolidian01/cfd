// gpu_rhs.cu — P8.3: per-block GPU RHS kernels (WENO5-Z + HLLC-ES + viscous)
//
// Three-kernel pipeline per exec() call (one CUDA block per leaf):
//   k_prim_duc  — conservative → prim + µ (Sutherland) + Ducros φ → d_scratch
//   k_rhs_conv  — WENO5-Z/KEP/HLLC-ES face-centred convective flux (atomicAdd)
//   k_rhs_visc  — face-averaged µ viscous divergence (direct write, cell-centred)
//
// GpuRhsList::exec() zeros d_rhs_pool, launches the three kernels in order,
// then returns.  download_rhs() DtoH-copies the RHS back to CellBlock::Q.

#include "cuda/gpu_rhs.cuh"
#include "cuda/gpu_hllc.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "physics/face_interp.hpp"
#include "physics/teno5_scalar.hpp"
#include <cstring>
#include <vector>

// One-sided WENO5-Z upwind reconstruction (Borges et al. 2008).
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

// WENO5-Z scalar reconstruction (Borges et al. 2008)
__device__ __forceinline__
void gpu_weno5z_scalar(double vm2, double vm1, double v0,
                       double vp1, double vp2, double vp3,
                       double& vL, double& vR) noexcept {
    vL = weno5z_upwind(vm2, vm1, v0,  vp1, vp2);   // left state
    vR = weno5z_upwind(vp3, vp2, vp1, v0,  vm1);   // right state (mirrored)
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

// ─────────────────────────────────────────────────────────────────────────────
// k_prim_duc: conservative → primitives + µ + Ducros φ → d_scratch
// Grid: (n_leaves)  Block: (GPU_NB2, GPU_NB2) = 144 threads
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_prim_duc(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    int i = threadIdx.x, j = threadIdx.y;

    // ── Pass 1: prim + µ ─────────────────────────────────────────────────────
    // P_FLOOR: the positivity floor (k_positivity_floor) clamps d_Q so that
    // p >= 1e-12 in conserved space.  However, gpu_cons_to_prim recomputes ke
    // from d_Q via a divide-then-multiply pattern that can differ by ~1 ULP
    // from the floor's multiply-then-divide form, leaving p=0 in the scratch.
    // Matching EPS_POS (1e-12) here ensures the scratch pressure is consistent
    // with the conserved-space floor and keeps HLLC-ES (which divides by p
    // via log_mean) from encountering p=0.
    constexpr double P_FLOOR = 1.0e-12;
    for (int k = 0; k < GPU_NB2; ++k) {
        int flat = gpu_cell_idx(i, j, k);
        GPrim q = gpu_cons_to_prim(
            m.d_Q[0*GPU_NCELL+flat], m.d_Q[1*GPU_NCELL+flat],
            m.d_Q[2*GPU_NCELL+flat], m.d_Q[3*GPU_NCELL+flat],
            m.d_Q[4*GPU_NCELL+flat]);
        if (q.p < P_FLOOR) {
            q.p = P_FLOOR;
            q.T = P_FLOOR / (q.rho * GPU_R_GAS);
            q.c = sqrt(GPU_GAMMA * P_FLOOR / q.rho);
        }
        m.d_scratch[0*GPU_NCELL+flat] = q.rho;
        m.d_scratch[1*GPU_NCELL+flat] = q.u;
        m.d_scratch[2*GPU_NCELL+flat] = q.v;
        m.d_scratch[3*GPU_NCELL+flat] = q.w;
        m.d_scratch[4*GPU_NCELL+flat] = q.p;
        m.d_scratch[5*GPU_NCELL+flat] = q.T;
        m.d_scratch[6*GPU_NCELL+flat] = q.c;
        m.d_scratch[7*GPU_NCELL+flat] = gpu_sutherland(q.T);
    }
    __syncthreads();

    // ── Pass 2: Ducros sensor (reads neighbour prim from d_scratch) ───────────
    constexpr double eps_duc = 1.0e-30;
    for (int k = 0; k < GPU_NB2; ++k) {
        double duc = 0.0;
        if (i >= 1 && i < GPU_NB2-1 && j >= 1 && j < GPU_NB2-1
            && k >= 1 && k < GPU_NB2-1) {
            const double* sp = m.d_scratch;
            const double ih2 = 0.5 / m.h;
            auto U = [=](int ii,int jj,int kk){ return sp[1*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto V = [=](int ii,int jj,int kk){ return sp[2*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto W = [=](int ii,int jj,int kk){ return sp[3*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
            auto P = [=](int ii,int jj,int kk){ return sp[4*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };

            const double dudx = ih2*(U(i+1,j,k)-U(i-1,j,k));
            const double dudy = ih2*(U(i,j+1,k)-U(i,j-1,k));
            const double dudz = ih2*(U(i,j,k+1)-U(i,j,k-1));
            const double dvdx = ih2*(V(i+1,j,k)-V(i-1,j,k));
            const double dvdy = ih2*(V(i,j+1,k)-V(i,j-1,k));
            const double dvdz = ih2*(V(i,j,k+1)-V(i,j,k-1));
            const double dwdx = ih2*(W(i+1,j,k)-W(i-1,j,k));
            const double dwdy = ih2*(W(i,j+1,k)-W(i,j-1,k));
            const double dwdz = ih2*(W(i,j,k+1)-W(i,j,k-1));
            const double divu = dudx + dvdy + dwdz;
            const double ox = dwdy-dvdz, oy = dudz-dwdx, oz = dvdx-dudy;
            const double d2 = divu*divu;
            const double c2 = ox*ox+oy*oy+oz*oz;
            const double phi_vel = d2/(d2+c2+eps_duc);

            const double pC  = P(i,j,k);
            const double dpx = fmax(fabs(P(i+1,j,k)-pC), fabs(P(i-1,j,k)-pC));
            const double dpy = fmax(fabs(P(i,j+1,k)-pC), fabs(P(i,j-1,k)-pC));
            const double dpz = fmax(fabs(P(i,j,k+1)-pC), fabs(P(i,j,k-1)-pC));
            const double phi_p = fmax(dpx, fmax(dpy,dpz)) / (pC+eps_duc);
            const double phi_p_cl = fmin(1.0, fmax(0.0, (phi_p-m.duc_p_thr)*m.duc_blend_inv));

            duc = fmax(phi_vel, phi_p_cl);
        }
        m.d_scratch[8*GPU_NCELL + gpu_cell_idx(i,j,k)] = duc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D0.5: shared-memory reconstruction helper for Y and Z faces (WENO5-Z or TENO5-A).
// Reads the 6-point stencil from the pre-loaded i-plane in shared memory.
// AXIS=1 (Y): face between y=fn and y=fn+1, x=xi fixed, z=tb fixed.
// AXIS=2 (Z): face between z=fn and z=fn+1, x=xi fixed, y=tb fixed.
// s layout: s[comp * NB22 + j * GPU_NB2 + k], NB22 = GPU_NB2*GPU_NB2.
// ─────────────────────────────────────────────────────────────────────────────
template<int AXIS, bool USE_TENO>
__device__ __forceinline__
void gpu_recon_shmem(const double* __restrict__ s,
                     int fn, int tb,
                     GPrim& qL_out, GPrim& qR_out) noexcept {
    // Shmem layout (padded): jk = k * PAD + j  where PAD = NB2+1 = 13.
    // Stride-13 is coprime with 32 → zero bank conflicts for Z-stencil reads.
    constexpr int PAD  = GPU_NB2 + 1;         // 13
    constexpr int NB2P = GPU_NB2 * PAD;        // 156 — per-comp shmem stride

    // jk index for stencil cell d ∈ {-2,-1,0,+1,+2,+3}
    auto jkd = [&](int d) -> int {
        if constexpr (AXIS == 1) return tb        * PAD + (fn + d);  // Y: k=tb, j=fn+d
        else                     return (fn + d)  * PAD + tb;         // Z: k=fn+d, j=tb
    };
    auto jkL = jkd(0), jkR = jkd(1);  // left / right cell of the face

    const double rL = s[0*NB2P+jkL], uL = s[1*NB2P+jkL];
    const double vL = s[2*NB2P+jkL], wL = s[3*NB2P+jkL];
    const double pL = s[4*NB2P+jkL], TL = s[5*NB2P+jkL], cL = s[6*NB2P+jkL];
    const double rR = s[0*NB2P+jkR], uR = s[1*NB2P+jkR];
    const double vR = s[2*NB2P+jkR], wR = s[3*NB2P+jkR];
    const double pR = s[4*NB2P+jkR], TR = s[5*NB2P+jkR], cR = s[6*NB2P+jkR];
    const GpuRoeState rs = gpu_roe_from_prim(rL,uL,vL,wL,pL, rR,uR,vR,wR,pR, AXIS);

    double Q[6][GPU_NVAR];
    for (int m = 0; m < 6; ++m) {
        const int jk = jkd(m - 2);
        const double rho = s[0*NB2P+jk], u = s[1*NB2P+jk];
        const double v   = s[2*NB2P+jk], w = s[3*NB2P+jk];
        const double p   = s[4*NB2P+jk];
        Q[m][0] = rho; Q[m][1] = rho*u; Q[m][2] = rho*v; Q[m][3] = rho*w;
        Q[m][4] = p/(GPU_GAMMA-1.0) + 0.5*rho*(u*u+v*v+w*w);
    }

    double W[5][6];
    for (int m = 0; m < 6; ++m) { double Wm[5]; gpu_char_proj(Q[m], rs, Wm); for (int c=0;c<5;++c) W[c][m]=Wm[c]; }

    double wL_w[5], wR_w[5];
    for (int kk = 0; kk < 5; ++kk)
        if constexpr (USE_TENO)
            physics_teno5_scalar(W[kk][0],W[kk][1],W[kk][2],W[kk][3],W[kk][4],W[kk][5], wL_w[kk], wR_w[kk]);
        else
            gpu_weno5z_scalar(W[kk][0],W[kk][1],W[kk][2],W[kk][3],W[kk][4],W[kk][5], wL_w[kk], wR_w[kk]);

    double QL[GPU_NVAR], QR[GPU_NVAR];
    gpu_back_proj(wL_w, rs, QL);
    gpu_back_proj(wR_w, rs, QR);
    GPrim fbL; fbL.rho=rL; fbL.u=uL; fbL.v=vL; fbL.w=wL; fbL.p=pL; fbL.T=TL; fbL.c=cL;
    GPrim fbR; fbR.rho=rR; fbR.u=uR; fbR.v=vR; fbR.w=wR; fbR.p=pR; fbR.T=TR; fbR.c=cR;
    qL_out = gpu_safe_prim_f(QL, fbL);
    qR_out = gpu_safe_prim_f(QR, fbR);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rhs_conv: hybrid WENO5-Z/KEP/HLLC-ES face-centred convective flux
// Grid: (n_leaves)  Block: (192) flat threads
// Iterates over all 3×(NB+1)×NB² = 1728 faces per leaf.
// Uses atomicAdd because each face writes to two independent cells.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rhs_conv(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const int  ilo    = GPU_NG;
    const int  ihi    = GPU_NG + GPU_NB - 1;
    const double ih   = 1.0 / m.h;
    constexpr double kep_thr = 1.0e-8;

    // Face counts per axis: (NB+1)*NB*NB = 576 ; total = 1728
    constexpr int NF   = GPU_NB + 1;   // 9 faces along normal axis
    constexpr int FPA  = NF * GPU_NB * GPU_NB;  // 576 per axis
    constexpr int FTOT = 3 * FPA;               // 1728

    auto load_prim = [&](int flat, GPrim& q) {
        q.rho = sp[0*GPU_NCELL+flat]; q.u = sp[1*GPU_NCELL+flat];
        q.v   = sp[2*GPU_NCELL+flat]; q.w = sp[3*GPU_NCELL+flat];
        q.p   = sp[4*GPU_NCELL+flat]; q.T = sp[5*GPU_NCELL+flat];
        q.c   = sp[6*GPU_NCELL+flat];
    };

    for (int fid = threadIdx.x; fid < FTOT; fid += blockDim.x) {
        const int axis  = fid / FPA;
        const int fi    = fid % FPA;
        const int f0    = fi % NF;          // face along normal: 0..NB
        const int fa    = (fi / NF) % GPU_NB; // transverse dim 1: 0..NB-1
        const int fb    = fi / (NF * GPU_NB); // transverse dim 2: 0..NB-1

        // Face normal coordinate: f0=0 → ghost face (ilo-1), f0=NB → ghost face (ihi)
        const int fn = ilo - 1 + f0;   // NG-1 .. NG+NB-1
        const int ta = ilo + fa;
        const int tb = ilo + fb;

        // Left/right cell flat indices (axis-dependent)
        int idxL, idxR;
        if (axis == 0) {
            idxL = gpu_cell_idx(fn,   ta, tb);
            idxR = gpu_cell_idx(fn+1, ta, tb);
        } else if (axis == 1) {
            idxL = gpu_cell_idx(ta, fn,   tb);
            idxR = gpu_cell_idx(ta, fn+1, tb);
        } else {
            idxL = gpu_cell_idx(ta, tb, fn  );
            idxR = gpu_cell_idx(ta, tb, fn+1);
        }

        const bool bL = (fn   >= ilo);
        const bool bR = (fn+1 <= ihi);
        if (!bL && !bR) continue;  // ghost-ghost face

        GPrim pL, pR;
        load_prim(idxL, pL); load_prim(idxR, pR);

        const double ducL = sp[8*GPU_NCELL+idxL];
        const double ducR = sp[8*GPU_NCELL+idxR];
        const double theta = fmax(ducL, ducR);
        const bool is_bnd = (fn < ilo || fn+1 > ihi);

        // KEP flux
        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, axis, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            // Pure KEP: smooth interior
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            // Check wall face: p_L == p_R (exact) and anti-symmetric tangential vel
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) -> bool {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }

            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                // Interior: WENO5-Z + HLLC-ES
                GPrim qL, qR;
                if (axis == 0) gpu_weno5_face(sp, fn, ta, tb, 0, qL, qR);
                else if (axis == 1) gpu_weno5_face(sp, ta, fn, tb, 1, qL, qR);
                else               gpu_weno5_face(sp, ta, tb, fn, 2, qL, qR);
                gpu_hllc_es_flux(qL, qR, axis, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, axis, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih*F[v]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D3: k_rhs_conv_teno — TENO5-A/KEP/HLLC-ES face-centred convective flux.
// Replaces k_rhs_conv as the default RHS kernel; WENO5-Z retained for comparison.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rhs_conv_teno(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const int  ilo    = GPU_NG;
    const int  ihi    = GPU_NG + GPU_NB - 1;
    const double ih   = 1.0 / m.h;
    constexpr double kep_thr = 1.0e-8;

    constexpr int NF   = GPU_NB + 1;
    constexpr int FPA  = NF * GPU_NB * GPU_NB;
    constexpr int FTOT = 3 * FPA;

    auto load_prim = [&](int flat, GPrim& q) {
        q.rho = sp[0*GPU_NCELL+flat]; q.u = sp[1*GPU_NCELL+flat];
        q.v   = sp[2*GPU_NCELL+flat]; q.w = sp[3*GPU_NCELL+flat];
        q.p   = sp[4*GPU_NCELL+flat]; q.T = sp[5*GPU_NCELL+flat];
        q.c   = sp[6*GPU_NCELL+flat];
    };

    for (int fid = threadIdx.x; fid < FTOT; fid += blockDim.x) {
        const int axis  = fid / FPA;
        const int fi    = fid % FPA;
        const int f0    = fi % NF;
        const int fa    = (fi / NF) % GPU_NB;
        const int fb    = fi / (NF * GPU_NB);

        const int fn = ilo - 1 + f0;
        const int ta = ilo + fa;
        const int tb = ilo + fb;

        int idxL, idxR;
        if (axis == 0) {
            idxL = gpu_cell_idx(fn,   ta, tb);
            idxR = gpu_cell_idx(fn+1, ta, tb);
        } else if (axis == 1) {
            idxL = gpu_cell_idx(ta, fn,   tb);
            idxR = gpu_cell_idx(ta, fn+1, tb);
        } else {
            idxL = gpu_cell_idx(ta, tb, fn  );
            idxR = gpu_cell_idx(ta, tb, fn+1);
        }

        const bool bL = (fn   >= ilo);
        const bool bR = (fn+1 <= ihi);
        if (!bL && !bR) continue;

        GPrim pL, pR;
        load_prim(idxL, pL); load_prim(idxR, pR);

        const double ducL = sp[8*GPU_NCELL+idxL];
        const double ducR = sp[8*GPU_NCELL+idxR];
        const double theta = fmax(ducL, ducR);
        const bool is_bnd = (fn < ilo || fn+1 > ihi);

        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, axis, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) -> bool {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }

            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                // Interior: TENO5-A + HLLC-ES (D3 default)
                GPrim qL, qR;
                if (axis == 0) gpu_teno5_face(sp, fn, ta, tb, 0, qL, qR);
                else if (axis == 1) gpu_teno5_face(sp, ta, fn, tb, 1, qL, qR);
                else               gpu_teno5_face(sp, ta, tb, fn, 2, qL, qR);
                gpu_hllc_es_flux(qL, qR, axis, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, axis, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih*F[v]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D3: k_rhs_conv_teno7 — TENO7-A/KEP/HLLC-ES face-centred convective flux.
// 7-point stencil; falls back to TENO5-A at boundary-adjacent interior faces.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rhs_conv_teno7(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const int  ilo    = GPU_NG;
    const int  ihi    = GPU_NG + GPU_NB - 1;
    const double ih   = 1.0 / m.h;
    constexpr double kep_thr = 1.0e-8;

    constexpr int NF   = GPU_NB + 1;
    constexpr int FPA  = NF * GPU_NB * GPU_NB;
    constexpr int FTOT = 3 * FPA;

    auto load_prim = [&](int flat, GPrim& q) {
        q.rho = sp[0*GPU_NCELL+flat]; q.u = sp[1*GPU_NCELL+flat];
        q.v   = sp[2*GPU_NCELL+flat]; q.w = sp[3*GPU_NCELL+flat];
        q.p   = sp[4*GPU_NCELL+flat]; q.T = sp[5*GPU_NCELL+flat];
        q.c   = sp[6*GPU_NCELL+flat];
    };

    for (int fid = threadIdx.x; fid < FTOT; fid += blockDim.x) {
        const int axis  = fid / FPA;
        const int fi    = fid % FPA;
        const int f0    = fi % NF;
        const int fa    = (fi / NF) % GPU_NB;
        const int fb    = fi / (NF * GPU_NB);

        const int fn = ilo - 1 + f0;
        const int ta = ilo + fa;
        const int tb = ilo + fb;

        int idxL, idxR;
        if (axis == 0) {
            idxL = gpu_cell_idx(fn,   ta, tb);
            idxR = gpu_cell_idx(fn+1, ta, tb);
        } else if (axis == 1) {
            idxL = gpu_cell_idx(ta, fn,   tb);
            idxR = gpu_cell_idx(ta, fn+1, tb);
        } else {
            idxL = gpu_cell_idx(ta, tb, fn  );
            idxR = gpu_cell_idx(ta, tb, fn+1);
        }

        const bool bL = (fn   >= ilo);
        const bool bR = (fn+1 <= ihi);
        if (!bL && !bR) continue;

        GPrim pL, pR;
        load_prim(idxL, pL); load_prim(idxR, pR);

        const double ducL = sp[8*GPU_NCELL+idxL];
        const double ducR = sp[8*GPU_NCELL+idxR];
        const double theta = fmax(ducL, ducR);
        const bool is_bnd = !m.is_periodic && (fn < ilo || fn+1 > ihi);

        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, axis, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) -> bool {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }
            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                // Interior: TENO7-A + HLLC-ES (D3 7th-order default)
                GPrim qL, qR;
                if (axis == 0) gpu_teno7_face(sp, fn, ta, tb, 0, qL, qR);
                else if (axis == 1) gpu_teno7_face(sp, ta, fn, tb, 1, qL, qR);
                else               gpu_teno7_face(sp, ta, tb, fn, 2, qL, qR);
                gpu_hllc_es_flux(qL, qR, axis, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, axis, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih*F[v]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D0.5 — k_rhs_conv_tiled<USE_TENO>: Y/Z faces use i-plane shmem (bank-conflict-free).
// USE_TENO=false → WENO5-Z;  USE_TENO=true → TENO5-A.
//
// Block: 144 = NB2² threads.  Iterates over NB2/2 = 6 pairs of i-planes:
//   1. Load both xi_a and xi_b slices into padded shmem s[2][8][NB2][PAD]
//      (PAD=NB2+1=13; stride-13 coprime with 32 → zero bank conflicts for Z).
//      tid 0..71  handle xi_a;  tid 72..143 handle xi_b.
//   2. Y-face pass: ALL 144 threads active — 72 for xi_a, 72 for xi_b.
//   3. __syncthreads()
//   4. Z-face pass: ALL 144 threads active — 72 for xi_a, 72 for xi_b.
//   5. __syncthreads()
// X-faces (576 total): global memory, 4 iters/thread.
// ─────────────────────────────────────────────────────────────────────────────
template<bool USE_TENO>
__global__
void k_rhs_conv_tiled(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp  = m.d_scratch;
    double*       rhs = m.d_RHS;
    const int     ilo = GPU_NG;
    const int     ihi = GPU_NG + GPU_NB - 1;
    const double  ih  = 1.0 / m.h;
    constexpr double kep_thr = 1.0e-8;
    constexpr int NF   = GPU_NB + 1;           // 9 faces along normal axis
    constexpr int NF72 = NF * GPU_NB;           // 72 = faces per axis per xi-plane
    constexpr int PAD  = GPU_NB2 + 1;           // 13 (bank-conflict-free stride)
    constexpr int NB2P = GPU_NB2 * PAD;         // 156 per-comp size in padded shmem

    // Shared: 2 xi-planes × 8 comps × NB2 × PAD = 2496 doubles = 19968 bytes.
    // Comp 7 stores Ducros sensor (from scratch comp 8).
    __shared__ double s[2 * 8 * NB2P];

    const int tid     = threadIdx.x;
    const int xi_sel  = tid / (GPU_NB2 * GPU_NB2 / 2);  // 0 or 1
    const int tid_loc = tid % (GPU_NB2 * GPU_NB2 / 2);  // 0..71 within the group

    // Base pointer into shmem for each xi-plane selection
    auto s_xi = [&](int sel) { return s + sel * 8 * NB2P; };

    // Load GPrim from padded shmem at jk = k*PAD+j
    auto sload = [&](const double* sb, int jk, GPrim& q) {
        q.rho = sb[0*NB2P+jk]; q.u = sb[1*NB2P+jk];
        q.v   = sb[2*NB2P+jk]; q.w = sb[3*NB2P+jk];
        q.p   = sb[4*NB2P+jk]; q.T = sb[5*NB2P+jk];
        q.c   = sb[6*NB2P+jk];
    };

    // Shared flux helper (same logic for Y and Z, templated by axis at callsite)
    auto do_yz_face = [&](const double* sb, int ta, int fn, int tb, int axis) {
        const bool is_y   = (axis == 1);
        // jk = k*PAD+j layout:
        //   Y-face (fn,tb): cell (j=fn,k=tb) → jk = tb*PAD + fn
        //   Z-face (tb,fn): cell (j=tb,k=fn) → jk = fn*PAD + tb
        const int jkL = is_y ? tb * PAD + fn     : fn       * PAD + tb;
        const int jkR = is_y ? tb * PAD + (fn+1) : (fn+1)   * PAD + tb;

        const int idxL = is_y ? gpu_cell_idx(ta, fn,   tb) : gpu_cell_idx(ta, tb, fn  );
        const int idxR = is_y ? gpu_cell_idx(ta, fn+1, tb) : gpu_cell_idx(ta, tb, fn+1);
        const bool bL  = (fn   >= ilo);
        const bool bR  = (fn+1 <= ihi);

        GPrim pL, pR;
        sload(sb, jkL, pL);
        sload(sb, jkR, pR);

        const double ducL  = sb[7*NB2P + jkL];
        const double ducR  = sb[7*NB2P + jkR];
        const double theta = fmax(ducL, ducR);
        const bool is_bnd  = (fn < ilo || fn+1 > ihi);

        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, axis, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }
            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                GPrim qL, qR;
                if (axis == 1) gpu_recon_shmem<1, USE_TENO>(sb, fn, tb, qL, qR);
                else           gpu_recon_shmem<2, USE_TENO>(sb, fn, tb, qL, qR);
                gpu_hllc_es_flux(qL, qR, axis, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, axis, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih*F[v]);
    };

    // ── Y/Z faces: loop over pairs of i-planes ────────────────────────────────
    for (int xi_pair = 0; xi_pair < GPU_NB2 / 2; ++xi_pair) {
        const int xi_a = 2 * xi_pair;
        const int xi_b = xi_a + 1;
        const int xi   = (xi_sel == 0) ? xi_a : xi_b;

        // Load: 2 iters per thread (tid_loc covers 0..71; NB2²=144 cells total).
        {
            double* sb = s_xi(xi_sel);
            for (int cell = tid_loc; cell < GPU_NB2 * GPU_NB2; cell += NF72) {
                const int j  = cell % GPU_NB2;
                const int k  = cell / GPU_NB2;
                const int jk = k * PAD + j;
                const int flat = gpu_cell_idx(xi, j, k);
                sb[0*NB2P+jk] = sp[0*GPU_NCELL+flat];
                sb[1*NB2P+jk] = sp[1*GPU_NCELL+flat];
                sb[2*NB2P+jk] = sp[2*GPU_NCELL+flat];
                sb[3*NB2P+jk] = sp[3*GPU_NCELL+flat];
                sb[4*NB2P+jk] = sp[4*GPU_NCELL+flat];
                sb[5*NB2P+jk] = sp[5*GPU_NCELL+flat];
                sb[6*NB2P+jk] = sp[6*GPU_NCELL+flat];
                sb[7*NB2P+jk] = sp[8*GPU_NCELL+flat];  // Ducros at scratch[8]
            }
        }
        __syncthreads();

        // Y-face pass: all 144 threads — each does 1 Y-face for its xi-plane
        {
            const int fn_rel = tid_loc % NF;
            const int tb_rel = tid_loc / NF;
            const int fn     = ilo - 1 + fn_rel;
            const int tb     = ilo + tb_rel;
            do_yz_face(s_xi(xi_sel), xi, fn, tb, 1);
        }
        __syncthreads();  // all Y atomicAdds complete before Z starts

        // Z-face pass: all 144 threads — each does 1 Z-face for its xi-plane
        {
            const int fn_rel = tid_loc % NF;
            const int tb_rel = tid_loc / NF;
            const int fn     = ilo - 1 + fn_rel;
            const int tb     = ilo + tb_rel;
            do_yz_face(s_xi(xi_sel), xi, fn, tb, 2);
        }
        __syncthreads();  // before next pair load
    }

    // ── X-faces: global memory, stride-1 (4 iters/thread) ───────────────────
    constexpr int FX = NF * GPU_NB * GPU_NB;  // 576
    auto gload = [&](int flat, GPrim& q) {
        q.rho = sp[0*GPU_NCELL+flat]; q.u = sp[1*GPU_NCELL+flat];
        q.v   = sp[2*GPU_NCELL+flat]; q.w = sp[3*GPU_NCELL+flat];
        q.p   = sp[4*GPU_NCELL+flat]; q.T = sp[5*GPU_NCELL+flat];
        q.c   = sp[6*GPU_NCELL+flat];
    };
    for (int fid = tid; fid < FX; fid += blockDim.x) {
        const int fn_rel = fid % NF;
        const int fa     = (fid / NF) % GPU_NB;
        const int fb     = fid / (NF * GPU_NB);
        const int fn     = ilo - 1 + fn_rel;
        const int ta     = ilo + fa;
        const int tb     = ilo + fb;

        const int idxL = gpu_cell_idx(fn,   ta, tb);
        const int idxR = gpu_cell_idx(fn+1, ta, tb);
        const bool bL  = (fn   >= ilo);
        const bool bR  = (fn+1 <= ihi);

        GPrim pL, pR;
        gload(idxL, pL); gload(idxR, pR);

        const double theta = fmax(sp[8*GPU_NCELL+idxL], sp[8*GPU_NCELL+idxR]);
        const bool is_bnd  = (fn < ilo || fn+1 > ihi);

        double Fk[GPU_NVAR];
        gpu_kep_flux(pL, pR, 0, Fk);

        double F[GPU_NVAR];
        if (!is_bnd && theta < kep_thr) {
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = Fk[v];
        } else {
            double Fs[GPU_NVAR];
            bool wall = is_bnd;
            if (wall) {
                auto antisym = [](double a, double b) {
                    return fabs(a+b) < 1.0e-8*(fabs(a)+fabs(b)+1.0e-300);
                };
                wall = (pL.p == pR.p) && antisym(pL.u,pR.u)
                                       && antisym(pL.v,pR.v)
                                       && antisym(pL.w,pR.w);
            }
            if (wall) {
                for (int v = 0; v < GPU_NVAR; ++v) Fs[v] = Fk[v];
            } else if (!is_bnd) {
                GPrim qL, qR;
                if constexpr (USE_TENO)
                    gpu_teno5_face(sp, fn, ta, tb, 0, qL, qR);
                else
                    gpu_weno5_face(sp, fn, ta, tb, 0, qL, qR);
                gpu_hllc_es_flux(qL, qR, 0, Fs);
            } else {
                gpu_hllc_es_flux(pL, pR, 0, Fs);
            }
            const double th = is_bnd ? 1.0 : theta;
            const double om = 1.0 - th;
            for (int v = 0; v < GPU_NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        }

        if (bL) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxL], -ih*F[v]);
        if (bR) for (int v = 0; v < GPU_NVAR; ++v)
            atomicAdd(&rhs[v*GPU_NCELL+idxR], +ih*F[v]);
    }
}

// Explicit instantiations required for __global__ function templates.
template __global__ void k_rhs_conv_tiled<false>(const GpuLeafRhsMeta*);
template __global__ void k_rhs_conv_tiled<true> (const GpuLeafRhsMeta*);

// ─────────────────────────────────────────────────────────────────────────────
// Viscous face helper — used by k_rhs_visc
// ─────────────────────────────────────────────────────────────────────────────

// Scratch velocity accessor: comp=0→u,1→v,2→w (scratch offsets 1,2,3).
__device__ __forceinline__ static double
sp_vel(const double* __restrict__ sp, int comp, int ii, int jj, int kk) {
    return sp[(comp + 1) * GPU_NCELL + gpu_cell_idx(ii, jj, kk)];
}

// Viscous stress + energy flux at face AX±½ of cell (i,j,k).
// AX: compile-time axis (0=X,1=Y,2=Z). dn ∈ {+1,-1}: +½ or -½ face.
// dn sign convention: grad_nn = ih*dn*(v_nb - v_c) is positive outward.
// tnn: normal-normal stress; tnt1/tnt2: shear on (AX,t1)/(AX,t2) planes.
// Fe: τ·u + κ∇T at face (energy flux, outward positive).
template<int AX>
__device__ __forceinline__ static void face_visc(
    const double* __restrict__ sp,
    int dn, int i, int j, int k,
    double mu, double kc, double ih, double ihs,
    double& tnn, double& tnt1, double& tnt2, double& Fe)
{
    constexpr int t1 = (AX + 1) % 3;
    constexpr int t2 = (AX + 2) % 3;
    const int ni = i + (AX == 0 ? dn : 0);
    const int nj = j + (AX == 1 ? dn : 0);
    const int nk = k + (AX == 2 ? dn : 0);
    constexpr int d1i = (t1 == 0), d1j = (t1 == 1), d1k = (t1 == 2);
    constexpr int d2i = (t2 == 0), d2j = (t2 == 1), d2k = (t2 == 2);

    // Normal velocity gradients at this face (sign absorbed by dn)
    const double dnn  = ih * dn * (sp_vel(sp, AX, ni, nj, nk) - sp_vel(sp, AX, i, j, k));
    const double dtn1 = ih * dn * (sp_vel(sp, t1, ni, nj, nk) - sp_vel(sp, t1, i, j, k));
    const double dtn2 = ih * dn * (sp_vel(sp, t2, ni, nj, nk) - sp_vel(sp, t2, i, j, k));

    // Tangential gradients: face-averaged between neighbor and center cells
    const double d_ax_dt1 = ihs*(sp_vel(sp,AX,ni+d1i,nj+d1j,nk+d1k)-sp_vel(sp,AX,ni-d1i,nj-d1j,nk-d1k)
                                +sp_vel(sp,AX,i +d1i,j +d1j,k +d1k)-sp_vel(sp,AX,i -d1i,j -d1j,k -d1k));
    const double d_ax_dt2 = ihs*(sp_vel(sp,AX,ni+d2i,nj+d2j,nk+d2k)-sp_vel(sp,AX,ni-d2i,nj-d2j,nk-d2k)
                                +sp_vel(sp,AX,i +d2i,j +d2j,k +d2k)-sp_vel(sp,AX,i -d2i,j -d2j,k -d2k));
    const double d_t1_dt1 = ihs*(sp_vel(sp,t1,ni+d1i,nj+d1j,nk+d1k)-sp_vel(sp,t1,ni-d1i,nj-d1j,nk-d1k)
                                +sp_vel(sp,t1,i +d1i,j +d1j,k +d1k)-sp_vel(sp,t1,i -d1i,j -d1j,k -d1k));
    const double d_t2_dt2 = ihs*(sp_vel(sp,t2,ni+d2i,nj+d2j,nk+d2k)-sp_vel(sp,t2,ni-d2i,nj-d2j,nk-d2k)
                                +sp_vel(sp,t2,i +d2i,j +d2j,k +d2k)-sp_vel(sp,t2,i -d2i,j -d2j,k -d2k));

    const double divu = dnn + d_t1_dt1 + d_t2_dt2;
    tnn  = mu * (2.0 * dnn - (2.0 / 3.0) * divu);
    tnt1 = mu * (d_ax_dt1 + dtn1);
    tnt2 = mu * (d_ax_dt2 + dtn2);

    // Face velocity: FaceInterp<AX> at the lower-index cell of this face.
    // dn=+1 → lower index = (i,j,k); dn=-1 → lower index = (ni,nj,nk).
    const int li = (AX == 0 ? (dn == 1 ? i : ni) : i);
    const int lj = (AX == 1 ? (dn == 1 ? j : nj) : j);
    const int lk = (AX == 2 ? (dn == 1 ? k : nk) : k);
    auto vel_ax = [sp](int ii,int jj,int kk){ return sp_vel(sp, AX, ii, jj, kk); };
    auto vel_t1 = [sp](int ii,int jj,int kk){ return sp_vel(sp, t1, ii, jj, kk); };
    auto vel_t2 = [sp](int ii,int jj,int kk){ return sp_vel(sp, t2, ii, jj, kk); };
    constexpr FaceInterp<static_cast<Axis>(AX)> fi;
    Fe = tnn  * fi(vel_ax, li, lj, lk)
       + tnt1 * fi(vel_t1, li, lj, lk)
       + tnt2 * fi(vel_t2, li, lj, lk)
       + kc * ih * dn * (sp[5*GPU_NCELL + gpu_cell_idx(ni,nj,nk)]
                        - sp[5*GPU_NCELL + gpu_cell_idx(i, j, k)]);
}

// Accumulate viscous stress divergence for one axis AX into acc[] and Fe_acc.
template<int AX>
__device__ __forceinline__ static void visc_axis(
    const double* __restrict__ sp, double ih, double ihs,
    int i, int j, int k, double mu_p, double mu_m,
    double (&acc)[3], double& Fe_acc)
{
    constexpr int t1 = (AX + 1) % 3;
    constexpr int t2 = (AX + 2) % 3;
    double tnn_p, tnt1_p, tnt2_p, Fe_p;
    double tnn_m, tnt1_m, tnt2_m, Fe_m;
    face_visc<AX>(sp, +1, i, j, k, mu_p, mu_p*GPU_CP/GPU_PR, ih, ihs,
                  tnn_p, tnt1_p, tnt2_p, Fe_p);
    face_visc<AX>(sp, -1, i, j, k, mu_m, mu_m*GPU_CP/GPU_PR, ih, ihs,
                  tnn_m, tnt1_m, tnt2_m, Fe_m);
    acc[AX] += ih * (tnn_p  - tnn_m );
    acc[t1] += ih * (tnt1_p - tnt1_m);
    acc[t2] += ih * (tnt2_p - tnt2_m);
    Fe_acc  += ih * (Fe_p   - Fe_m  );
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rhs_visc: face-averaged µ viscous divergence (B5 conservative form)
// Grid: (n_leaves)  Block: (GPU_NB, GPU_NB) = 64 threads
// Each thread (di,dj) handles all k ∈ [ilo,ihi]; direct write (no atomics).
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rhs_visc(const GpuLeafRhsMeta* __restrict__ metas) {
    const GpuLeafRhsMeta& m = metas[blockIdx.x];
    const double* sp = m.d_scratch;
    double*      rhs = m.d_RHS;
    const int    ilo = GPU_NG;
    const int    ihi = GPU_NG + GPU_NB - 1;
    const double ih  = 1.0 / m.h;
    const double ihs = 0.25 * ih;  // 1/(4h): average of two 1/(2h) central diffs

    const int i = GPU_NG + threadIdx.x;
    const int j = GPU_NG + threadIdx.y;

    auto MU = [sp](int ii,int jj,int kk){ return sp[7*GPU_NCELL+gpu_cell_idx(ii,jj,kk)]; };
    constexpr FaceInterp<Axis::X> fi_x;
    constexpr FaceInterp<Axis::Y> fi_y;
    constexpr FaceInterp<Axis::Z> fi_z;

    for (int k = ilo; k <= ihi; ++k) {
        // µ face averaging via FaceInterp<DIR, ArithmeticMean>
        const double mu_xp = fi_x(MU, i,   j,   k  );
        const double mu_xm = fi_x(MU, i-1, j,   k  );
        const double mu_yp = fi_y(MU, i,   j,   k  );
        const double mu_ym = fi_y(MU, i,   j-1, k  );
        const double mu_zp = fi_z(MU, i,   j,   k  );
        const double mu_zm = fi_z(MU, i,   j,   k-1);

        double acc[3] = {0.0, 0.0, 0.0};
        double Fe_acc = 0.0;
        visc_axis<0>(sp, ih, ihs, i, j, k, mu_xp, mu_xm, acc, Fe_acc);
        visc_axis<1>(sp, ih, ihs, i, j, k, mu_yp, mu_ym, acc, Fe_acc);
        visc_axis<2>(sp, ih, ihs, i, j, k, mu_zp, mu_zm, acc, Fe_acc);

        const int flat = gpu_cell_idx(i,j,k);
        rhs[1*GPU_NCELL+flat] += acc[0];
        rhs[2*GPU_NCELL+flat] += acc[1];
        rhs[3*GPU_NCELL+flat] += acc[2];
        rhs[4*GPU_NCELL+flat] += Fe_acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuRhsList methods
// ─────────────────────────────────────────────────────────────────────────────

GpuRhsList::~GpuRhsList() {
    if (d_metas)        { cudaFree(d_metas);        d_metas        = nullptr; }
    if (d_scratch_pool) { cudaFree(d_scratch_pool); d_scratch_pool = nullptr; }
    if (d_rhs_pool)     { cudaFree(d_rhs_pool);     d_rhs_pool     = nullptr; }
}

void GpuRhsList::build(const BlockTree& tree, const GpuPool& pool) {
    // Free previous large-pool allocations (d_metas freed inside gpu_upload_meta)
    cudaFree(d_scratch_pool); d_scratch_pool = nullptr;
    cudaFree(d_rhs_pool);     d_rhs_pool     = nullptr;

    std::vector<int> local;
    for (int idx : tree.leaf_indices())
        if (tree.nodes[idx].has_block()) local.push_back(idx);
    n_leaves = (int)local.size();
    if (n_leaves == 0) return;

    const size_t scratch_bytes = (size_t)SCRATCH_NCOMP * NCELL * n_leaves * sizeof(double);
    const size_t rhs_bytes     = (size_t)NVAR           * NCELL * n_leaves * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_scratch_pool, scratch_bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs_pool,     rhs_bytes    ));

    std::vector<GpuLeafRhsMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const BlockNode& nd = tree.nodes[local[li]];
        GpuLeafRhsMeta& meta = h_metas[li];
        meta.d_Q          = pool.d_Q(nd.block.get());
        meta.d_RHS        = d_rhs_pool     + (size_t)li * NVAR          * NCELL;
        meta.d_scratch    = d_scratch_pool + (size_t)li * SCRATCH_NCOMP * NCELL;
        meta.h            = nd.block->h;
        meta.duc_p_thr    = duc_p_thr_;
        meta.duc_blend_inv= duc_blend_inv_;
        meta.is_periodic  = tree.is_fully_periodic() ? 1u : 0u;
    }

    gpu_upload_meta(d_metas, h_metas);
}

// k_zero_rhs: zero d_rhs_pool as a proper kernel node (graph-capture safe).
// Using a kernel instead of cudaMemsetAsync guarantees explicit stream ordering
// in the captured CUDA graph — no reliance on memset helper streams.
__global__
void k_zero_rhs(double* __restrict__ pool, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        pool[i] = 0.0;
}

void GpuRhsList::exec(cudaStream_t stream, bool zero_rhs) const {
    if (n_leaves == 0) return;

    // Optional zeroing: callers that use cudaMemsetAsync on the same stream
    // BEFORE launching this (e.g. the per-stage graph replay path) pass false
    // to avoid a redundant zero inside any CUDA graph node.
    if (zero_rhs) {
        const int total = NVAR * NCELL * n_leaves;
        const int nblks = (total + 255) / 256;
        k_zero_rhs<<<nblks, 256, 0, stream>>>(d_rhs_pool, total);
    }

    k_prim_duc<<<dim3(n_leaves), dim3(GPU_NB2, GPU_NB2), 0, stream>>>(d_metas);
    switch (scheme) {
    case GpuReconScheme::TENO7A:
        k_rhs_conv_teno7             <<<dim3(n_leaves), 192, 0, stream>>>(d_metas);
        break;
    case GpuReconScheme::TENO5A:
        k_rhs_conv_teno              <<<dim3(n_leaves), 192, 0, stream>>>(d_metas);
        break;
    case GpuReconScheme::WENO5Z_TILED:
        k_rhs_conv_tiled<false>      <<<dim3(n_leaves), 144, 0, stream>>>(d_metas);
        break;
    case GpuReconScheme::TENO5A_TILED:
        k_rhs_conv_tiled<true>       <<<dim3(n_leaves), 144, 0, stream>>>(d_metas);
        break;
    default:  // WENO5Z
        k_rhs_conv                   <<<dim3(n_leaves), 192, 0, stream>>>(d_metas);
        break;
    }
    k_rhs_visc<<<dim3(n_leaves), dim3(GPU_NB, GPU_NB), 0, stream>>>(d_metas);
}

void GpuRhsList::download_rhs(const BlockTree& tree) const {
    if (n_leaves == 0) return;
    const auto& leaves = tree.leaf_indices();
    const size_t blk_bytes = (size_t)NVAR * NCELL * sizeof(double);
    static thread_local double h_buf[NVAR * NCELL];
    for (int li = 0; li < n_leaves; ++li) {
        CellBlock* blk = tree.nodes[leaves[li]].block.get();
        if (!blk) continue;
        CUDA_CHECK(cudaMemcpy(h_buf, d_rhs_pool + (size_t)li * NVAR * NCELL,
                              blk_bytes, cudaMemcpyDeviceToHost));
        for (int v = 0; v < NVAR; ++v)
            blk->Q[v].assign_from_flat(h_buf + v * NCELL);
    }
}
