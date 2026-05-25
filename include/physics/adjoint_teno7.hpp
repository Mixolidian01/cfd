#pragma once
// D10: adjoint_teno7.hpp — frozen-weight TENO7-A characteristic adjoint
//
// Provides:
//   Teno7ScalarFwd       — frozen state from a TENO7-A one_sided() pass
//   physics_teno7_scalar_fwd(...)  — forward pass capturing fwdL/fwdR
//   teno7_one_sided_adj(fw, l_out, la[7]) — adjoint of one_sided()
//   Teno7CharFwd         — frozen state from a full Teno7Recon<DIR> pass
//   teno7_recon_fwd<DIR>(...)      — forward pass capturing Teno7CharFwd
//   teno7_recon_adj<DIR>(...)      — adjoint: l_qL_cons + l_qR_cons → l_pc[flat][var]

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#include "mesh/cell_block.hpp"      // Prim, NVAR, NB2, NCELL, cell_idx
#include "mesh/axis.hpp"             // Axis
#include "physics/adjoint_hllc.hpp"  // acc_adj_prim_to_cons
#include "physics/teno7_scalar.hpp"  // physics_teno7_scalar (for fwd reference)
#include "physics/teno7_recon.hpp"   // Teno7Recon<DIR>
#include "physics/teno5_recon.hpp"   // Teno5Recon<DIR> (fallback path)
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// 1. Teno7ScalarFwd — frozen state from one one_sided() call
// ─────────────────────────────────────────────────────────────────────────────
struct Teno7ScalarFwd {
    double w[4];   // frozen active weights (0 if substencil excluded)
    double ws;     // sum of active weights (>0 → smooth path, ≤0 → ENO fallback)
    double s[4];   // polynomial values s0..s3
    int    eno_k;  // ENO fallback substencil index (0..3); valid only when ws<=0
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. physics_teno7_scalar_fwd — forward TENO7 scalar + capture frozen state
//    Matches physics_teno7_scalar exactly; additionally fills fwdL and fwdR.
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__ inline void physics_teno7_scalar_fwd(
        double vm3, double vm2, double vm1, double v0,
        double vp1, double vp2, double vp3,
        double& vL, double& vR,
        Teno7ScalarFwd& fwdL, Teno7ScalarFwd& fwdR) noexcept
{
    constexpr double eps  = 1.0e-36;
    constexpr double CT   = 1.0e-6;
    constexpr double d0   = 1.0/35.0, d1 = 12.0/35.0, d2 = 18.0/35.0, d3 = 4.0/35.0;
    constexpr double i12  = 1.0/12.0;
    constexpr double i240 = 1.0/240.0;

    auto sq = [](double x) noexcept -> double { return x * x; };

    // one_sided: captures weights and polynomials into fw
    auto one_sided_fwd = [&](double a, double b, double c, double d,
                              double e, double f, double g,
                              Teno7ScalarFwd& fw) noexcept -> double
    {
        // Sub-stencil polynomials
        fw.s[0] = i12*(-3.0*a + 13.0*b - 23.0*c + 25.0*d);
        fw.s[1] = i12*( 1.0*b -  5.0*c + 13.0*d +  3.0*e);
        fw.s[2] = i12*(-1.0*c +  7.0*d +  7.0*e -  1.0*f);
        fw.s[3] = i12*(25.0*d - 23.0*e + 13.0*f -  3.0*g);

        // Smoothness indicators
        const double b0 = i240*(547.0*sq(a) - 3882.0*a*b + 4642.0*a*c - 1854.0*a*d
                               + 7043.0*sq(b) - 17246.0*b*c + 7042.0*b*d
                               + 11003.0*sq(c) - 9402.0*c*d + 2107.0*sq(d));
        const double b1 = i240*(267.0*sq(b) - 1642.0*b*c + 1602.0*b*d - 494.0*b*e
                               + 2843.0*sq(c) - 5966.0*c*d + 1922.0*c*e
                               + 3443.0*sq(d) - 2522.0*d*e + 547.0*sq(e));
        const double b2 = i240*(267.0*sq(c) - 1642.0*c*d + 1602.0*c*e - 494.0*c*f
                               + 2843.0*sq(d) - 5966.0*d*e + 1922.0*d*f
                               + 3443.0*sq(e) - 2522.0*e*f + 547.0*sq(f));
        const double b3 = i240*(2107.0*sq(d) - 9402.0*d*e + 7042.0*d*f - 1854.0*d*g
                               + 11003.0*sq(e) - 17246.0*e*f + 4642.0*e*g
                               + 7043.0*sq(f) - 3882.0*f*g + 547.0*sq(g));

        const double tau7 = (b0 > b3) ? b0 - b3 : b3 - b0;
        auto chi6 = [&](double bk) noexcept -> double {
            double r = 1.0 + tau7 / (bk + eps);
            r *= r; r *= r; r *= r;
            return r;
        };
        const double c0 = chi6(b0), c1 = chi6(b1), c2 = chi6(b2), c3 = chi6(b3);
        const double ci = 1.0 / (c0 + c1 + c2 + c3 + eps);

        fw.w[0] = (c0*ci >= CT) ? d0 : 0.0;
        fw.w[1] = (c1*ci >= CT) ? d1 : 0.0;
        fw.w[2] = (c2*ci >= CT) ? d2 : 0.0;
        fw.w[3] = (c3*ci >= CT) ? d3 : 0.0;
        fw.ws   = fw.w[0] + fw.w[1] + fw.w[2] + fw.w[3];

        if (fw.ws > 0.0)
            return (fw.w[0]*fw.s[0] + fw.w[1]*fw.s[1] +
                    fw.w[2]*fw.s[2] + fw.w[3]*fw.s[3]) / fw.ws;

        // ENO fallback: record which substencil was chosen
        if (b0 <= b1 && b0 <= b2 && b0 <= b3) { fw.eno_k = 0; return fw.s[0]; }
        if (b1 <= b2 && b1 <= b3)              { fw.eno_k = 1; return fw.s[1]; }
        if (b2 <= b3)                           { fw.eno_k = 2; return fw.s[2]; }
        fw.eno_k = 3; return fw.s[3];
    };

    vL = one_sided_fwd(vm3, vm2, vm1, v0,  vp1, vp2, vp3, fwdL);
    vR = one_sided_fwd(vp3, vp2, vp1, v0,  vm1, vm2, vm3, fwdR);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. teno7_one_sided_adj — adjoint of one_sided() with frozen weights
//    fw     : frozen state from the matching forward one_sided_fwd call
//    l_out  : ∂J/∂output (scalar seed)
//    la[7]  : accumulated ∂J/∂{a,b,c,d,e,f,g}  (accumulate, not overwrite)
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__ inline
void teno7_one_sided_adj(const Teno7ScalarFwd& fw,
                         double l_out,
                         double la[7]) noexcept
{
    constexpr double i12 = 1.0/12.0;

    if (fw.ws > 0.0) {
        // Smooth path: output = (w0*s0 + w1*s1 + w2*s2 + w3*s3) / ws
        const double l_s0 = l_out * fw.w[0] / fw.ws;
        const double l_s1 = l_out * fw.w[1] / fw.ws;
        const double l_s2 = l_out * fw.w[2] / fw.ws;
        const double l_s3 = l_out * fw.w[3] / fw.ws;

        // s0 = i12*(-3a + 13b - 23c + 25d)
        la[0] += l_s0 * (-3.0*i12);
        la[1] += l_s0 * (13.0*i12);
        la[2] += l_s0 * (-23.0*i12);
        la[3] += l_s0 * (25.0*i12);

        // s1 = i12*(b - 5c + 13d + 3e)
        la[1] += l_s1 * (i12);
        la[2] += l_s1 * (-5.0*i12);
        la[3] += l_s1 * (13.0*i12);
        la[4] += l_s1 * (3.0*i12);

        // s2 = i12*(-c + 7d + 7e - f)
        la[2] += l_s2 * (-i12);
        la[3] += l_s2 * (7.0*i12);
        la[4] += l_s2 * (7.0*i12);
        la[5] += l_s2 * (-i12);

        // s3 = i12*(25d - 23e + 13f - 3g)
        la[3] += l_s3 * (25.0*i12);
        la[4] += l_s3 * (-23.0*i12);
        la[5] += l_s3 * (13.0*i12);
        la[6] += l_s3 * (-3.0*i12);
    } else {
        // ENO fallback: output = s[eno_k] with frozen substencil selection
        // Adjoint of the selected polynomial only
        switch (fw.eno_k) {
            case 0:  // s0 = i12*(-3a+13b-23c+25d)
                la[0] += l_out * (-3.0*i12);
                la[1] += l_out * (13.0*i12);
                la[2] += l_out * (-23.0*i12);
                la[3] += l_out * (25.0*i12);
                break;
            case 1:  // s1 = i12*(b-5c+13d+3e)
                la[1] += l_out * (i12);
                la[2] += l_out * (-5.0*i12);
                la[3] += l_out * (13.0*i12);
                la[4] += l_out * (3.0*i12);
                break;
            case 2:  // s2 = i12*(-c+7d+7e-f)
                la[2] += l_out * (-i12);
                la[3] += l_out * (7.0*i12);
                la[4] += l_out * (7.0*i12);
                la[5] += l_out * (-i12);
                break;
            default: // s3 = i12*(25d-23e+13f-3g)
                la[3] += l_out * (25.0*i12);
                la[4] += l_out * (-23.0*i12);
                la[5] += l_out * (13.0*i12);
                la[6] += l_out * (-3.0*i12);
                break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Teno7CharFwd — frozen state from a full Teno7Recon<DIR> pass
// ─────────────────────────────────────────────────────────────────────────────
struct Teno7CharFwd {
    // Frozen Roe averages
    double u_roe, v_roe, w_roe, H_roe, c_roe;
    double KE, b, b2, ioc, gm_roe;
    double un, ut1, ut2;

    // Per-characteristic frozen scalar states (5 char fields × left + right)
    Teno7ScalarFwd fwd_L[5];  // frozen state for left one_sided call
    Teno7ScalarFwd fwd_R[5];  // frozen state for right one_sided call (mirrored)

    // Reconstructed characteristic values
    double wL[5];  // left-face char reconstructions
    double wR[5];  // right-face char reconstructions

    // n_idx sentinel: -1 if TENO5 fallback was used (n0 < NG+1)
    int n_idx;
    // t1_idx, t2_idx stored for the adjoint
    int t1_idx;
    int t2_idx;
};

// ─────────────────────────────────────────────────────────────────────────────
// 5. teno7_recon_fwd<DIR> — forward Teno7Recon<DIR> capturing Teno7CharFwd
//    Must match Teno7Recon<DIR>::operator() exactly.
// ─────────────────────────────────────────────────────────────────────────────
template<Axis DIR>
__host__ __device__ inline
void teno7_recon_fwd(const Prim* pc, int i, int j, int k,
                     Prim& qL_out, Prim& qR_out,
                     Teno7CharFwd& cf) noexcept
{
    const int n0 = (DIR==Axis::X) ? i : (DIR==Axis::Y) ? j : k;
    if (n0 < NG + 1) {
        // TENO5 fallback — mark and call the regular forward
        cf.n_idx = -1;
        Teno5Recon<DIR>{}(pc, i, j, k, qL_out, qR_out);
        return;
    }

    constexpr int n_idx_c  = (DIR==Axis::X) ? 1 : (DIR==Axis::Y) ? 2 : 3;
    constexpr int t1_idx_c = (DIR==Axis::X) ? 2 : 1;
    constexpr int t2_idx_c = (DIR==Axis::Z) ? 2 : 3;
    cf.n_idx  = n_idx_c;
    cf.t1_idx = t1_idx_c;
    cf.t2_idx = t2_idx_c;

    auto idx_at = [&](int d) noexcept -> int {
        if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
        if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
        return                              cell_idx(i, j, k+d);
    };

    // 7-point conservative stencil (m=0 → d=-3, m=3 → d=0 left cell)
    double Q[7][NVAR];
    for (int m = 0; m < 7; ++m) {
        const Prim& p = pc[idx_at(m - 3)];
        Q[m][0] = p.rho;
        Q[m][1] = p.rho * p.u;
        Q[m][2] = p.rho * p.v;
        Q[m][3] = p.rho * p.w;
        Q[m][4] = (p.p + p.gamma_m*p.p_inf_m)/(p.gamma_m-1.0)
                  + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
    }

    // Roe averages (left=m=3/d=0, right=m=4/d=1)
    const Prim& pL = pc[idx_at(0)];
    const Prim& pR = pc[idx_at(1)];
    const double sqL   = std::sqrt(pL.rho);
    const double sqR   = std::sqrt(pR.rho);
    const double denom = sqL + sqR;
    cf.u_roe = (sqL*pL.u + sqR*pR.u) / denom;
    cf.v_roe = (sqL*pL.v + sqR*pR.v) / denom;
    cf.w_roe = (sqL*pL.w + sqR*pR.w) / denom;
    const double HL    = (Q[3][4] + pL.p) / pL.rho;
    const double HR    = (Q[4][4] + pR.p) / pR.rho;
    cf.H_roe = (sqL*HL + sqR*HR) / denom;
    cf.KE    = 0.5*(cf.u_roe*cf.u_roe + cf.v_roe*cf.v_roe + cf.w_roe*cf.w_roe);
    cf.gm_roe = 0.5*(pL.gamma_m + pR.gamma_m);
    const double c2    = std::max((cf.gm_roe-1.0)*(cf.H_roe - cf.KE), 1.0e-300);
    cf.c_roe  = std::sqrt(c2);

    cf.un  = (DIR==Axis::X) ? cf.u_roe : (DIR==Axis::Y) ? cf.v_roe : cf.w_roe;
    cf.ut1 = (DIR==Axis::X) ? cf.v_roe : cf.u_roe;
    cf.ut2 = (DIR==Axis::Z) ? cf.v_roe : cf.w_roe;

    cf.b   = (cf.gm_roe-1.0) / c2;
    cf.b2  = cf.b * cf.KE;
    cf.ioc = 1.0 / cf.c_roe;

    // Characteristic projection of 7-point stencil
    double W[5][7];
    for (int m = 0; m < 7; ++m) {
        const double rho = Q[m][0];
        const double qn  = Q[m][n_idx_c];
        const double qt1 = Q[m][t1_idx_c];
        const double qt2 = Q[m][t2_idx_c];
        const double E   = Q[m][4];
        const double inner   = cf.b2*rho - cf.b*(cf.un*qn + cf.ut1*qt1 + cf.ut2*qt2) + cf.b*E;
        const double delta_n = cf.ioc*(cf.un*rho - qn);
        W[0][m] = 0.5*(inner + delta_n);
        W[1][m] = (1.0 - cf.b2)*rho + cf.b*(cf.un*qn + cf.ut1*qt1 + cf.ut2*qt2) - cf.b*E;
        W[2][m] = -cf.ut1*rho + qt1;
        W[3][m] = -cf.ut2*rho + qt2;
        W[4][m] = 0.5*(inner - delta_n);
    }

    // TENO7-A scalar per characteristic variable — capture frozen state
    for (int kk = 0; kk < 5; ++kk)
        physics_teno7_scalar_fwd(W[kk][0], W[kk][1], W[kk][2], W[kk][3],
                                 W[kk][4], W[kk][5], W[kk][6],
                                 cf.wL[kk], cf.wR[kk],
                                 cf.fwd_L[kk], cf.fwd_R[kk]);

    // back_project
    auto back_project = [&](const double w[5], double Qrec[NVAR]) noexcept {
        const double w014 = w[0] + w[1] + w[4];
        const double dw04 = w[4] - w[0];
        Qrec[0]        = w014;
        Qrec[n_idx_c]  = w014*cf.un  + dw04*cf.c_roe;
        Qrec[t1_idx_c] = w014*cf.ut1 + w[2];
        Qrec[t2_idx_c] = w014*cf.ut2 + w[3];
        Qrec[4]        = (w[0]+w[4])*cf.H_roe + dw04*cf.un*cf.c_roe
                       + w[1]*cf.KE + w[2]*cf.ut1 + w[3]*cf.ut2;
    };

    double QL[NVAR], QRv[NVAR];
    back_project(cf.wL, QL);
    back_project(cf.wR, QRv);

    auto safe_prim = [](const double Qc[NVAR], const Prim& fallback) noexcept -> Prim {
        const double rho = Qc[0];
        if (rho <= 0.0) return fallback;
        const double u   = Qc[1] / rho;
        const double v   = Qc[2] / rho;
        const double w_  = Qc[3] / rho;
        const double gm  = fallback.gamma_m;
        const double pim = fallback.p_inf_m;
        const double p   = (gm-1.0)*(Qc[4] - 0.5*rho*(u*u+v*v+w_*w_)) - gm*pim;
        if (p + pim <= 0.0) return fallback;
        Prim q; q.rho=rho; q.u=u; q.v=v; q.w=w_; q.p=p;
        q.gamma_m=gm; q.p_inf_m=pim;
        q.T=(p+pim)/(rho*R_GAS); q.c=std::sqrt(gm*(p+pim)/rho);
        return q;
    };

    qL_out = safe_prim(QL,  pL);
    qR_out = safe_prim(QRv, pR);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. teno7_recon_adj<DIR> — adjoint of teno7_recon_fwd
//
//    Inputs:
//      pc            — primal prim array (needed for acc_adj_prim_to_cons)
//      i, j, k       — face indices (same as forward call)
//      cf            — frozen state captured in teno7_recon_fwd
//      l_qL_cons[5]  — ∂J/∂Qrec_L  (adjoint of reconstructed conserved left state)
//      l_qR_cons[5]  — ∂J/∂Qrec_R  (adjoint of reconstructed conserved right state)
//
//    Output (accumulated):
//      l_pc[NCELL][NVAR] — ∂J/∂prim for each stencil cell
//                          (indexed by flat cell index, accumulated in-place)
//
//    Note: safe_prim is a passthrough when rho>0 and p+p∞>0 (which is the normal
//    case). We treat it as identity for the adjoint (standard frozen-limiter approach).
//    If the fallback was triggered (cf.n_idx == -1), the TENO5 fallback is not
//    differentiated — adjoints are zero for this face (conservative safe choice).
// ─────────────────────────────────────────────────────────────────────────────
template<Axis DIR>
__host__ __device__ inline
void teno7_recon_adj(const Prim* pc, int i, int j, int k,
                     const Teno7CharFwd& cf,
                     const double l_qL_cons[NVAR],
                     const double l_qR_cons[NVAR],
                     double l_pc[][NVAR]) noexcept
{
    // TENO5 fallback: no adjoint propagated (safe conservative choice)
    if (cf.n_idx < 0) return;

    const int n_idx_c  = cf.n_idx;
    const int t1_idx_c = cf.t1_idx;
    const int t2_idx_c = cf.t2_idx;

    auto idx_at = [&](int d) noexcept -> int {
        if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
        if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
        return                              cell_idx(i, j, k+d);
    };

    // ── Step 1: back_project adjoint ──────────────────────────────────────────
    // Forward back_project(w[5]) → Qrec[5]:
    //   w014 = w[0]+w[1]+w[4];  dw04 = w[4]-w[0]
    //   Qrec[0]       = w014
    //   Qrec[n_idx]   = w014*un  + dw04*c_roe
    //   Qrec[t1_idx]  = w014*ut1 + w[2]
    //   Qrec[t2_idx]  = w014*ut2 + w[3]
    //   Qrec[4]       = (w[0]+w[4])*H_roe + dw04*un*c_roe + w[1]*KE + w[2]*ut1 + w[3]*ut2
    //
    // Adjoint (given l_Qrec[5] → l_w[5]):
    //   l_w014 = l_Qrec[0] + l_Qrec[n]*un + l_Qrec[t1]*ut1 + l_Qrec[t2]*ut2
    //   l_dw04 = l_Qrec[n]*c_roe + l_Qrec[4]*un*c_roe
    //   l_w04  = l_Qrec[4]*H_roe
    //   l_w1   = l_Qrec[4]*KE
    //   l_w2   = l_Qrec[t1] + l_Qrec[4]*ut1
    //   l_w3   = l_Qrec[t2] + l_Qrec[4]*ut2
    //   l_w[0] = l_w014 - l_dw04 + l_w04
    //   l_w[1] = l_w014 + l_w1
    //   l_w[2] = l_w2
    //   l_w[3] = l_w3
    //   l_w[4] = l_w014 + l_dw04 + l_w04

    auto back_project_adj = [&](const double l_Qrec[NVAR],
                                 double l_w[5]) noexcept
    {
        const double l_w014 = l_Qrec[0]
                            + l_Qrec[n_idx_c]  * cf.un
                            + l_Qrec[t1_idx_c] * cf.ut1
                            + l_Qrec[t2_idx_c] * cf.ut2;
        const double l_dw04 = l_Qrec[n_idx_c]  * cf.c_roe
                            + l_Qrec[4]          * cf.un * cf.c_roe;
        const double l_w04  = l_Qrec[4] * cf.H_roe;
        const double l_w1   = l_Qrec[4] * cf.KE;
        const double l_w2   = l_Qrec[t1_idx_c] + l_Qrec[4] * cf.ut1;
        const double l_w3   = l_Qrec[t2_idx_c] + l_Qrec[4] * cf.ut2;

        l_w[0] = l_w014 - l_dw04 + l_w04;
        l_w[1] = l_w014 + l_w1;
        l_w[2] = l_w2;
        l_w[3] = l_w3;
        l_w[4] = l_w014 + l_dw04 + l_w04;
    };

    double l_wL[5], l_wR[5];
    back_project_adj(l_qL_cons, l_wL);
    back_project_adj(l_qR_cons, l_wR);

    // ── Step 2: teno7_one_sided_adj for each characteristic field ─────────────
    // For the LEFT state: one_sided(vm3,vm2,vm1,v0,vp1,vp2,vp3)
    //   W[kk][m] for m=0..6 are the 7 stencil values (order: d=-3,..,+3)
    //   la_fwd[0..6] → W[kk][0..6]
    //
    // For the RIGHT state: one_sided(vp3,vp2,vp1,v0,vm1,vm2,vm3) (mirrored)
    //   la_rev[0..6] → (vp3,vp2,vp1,v0,vm1,vm2,vm3) → W[kk][6,5,4,3,2,1,0]
    //   so la_rev[r] → W[kk][6-r]

    // Characteristic adjoints per stencil cell: l_W[char][stencil_m]
    double l_W[5][7] = {};  // zero-init, 5 characteristic fields × 7 stencil cells

    for (int kk = 0; kk < 5; ++kk) {
        // Left state adjoint: la_fwd[0..6] → W[kk][0..6]
        double la_fwd[7] = {};
        teno7_one_sided_adj(cf.fwd_L[kk], l_wL[kk], la_fwd);
        for (int r = 0; r < 7; ++r)
            l_W[kk][r] += la_fwd[r];

        // Right state adjoint: mirrored call → la_rev[0..6] for (vp3,vp2,..,vm3)
        double la_rev[7] = {};
        teno7_one_sided_adj(cf.fwd_R[kk], l_wR[kk], la_rev);
        // Map back: la_rev[r] contributes to W[kk][6-r]
        for (int r = 0; r < 7; ++r)
            l_W[kk][6-r] += la_rev[r];
    }

    // ── Step 3: characteristic projection adjoint (P^T) ───────────────────────
    // Forward: W[char][m] = P * Q[m] where P maps (rho,qn,qt1,qt2,E) → 5 chars.
    // P^T maps l_W[0..4][m] → l_Q[m] = (l_rho, l_qn, l_qt1, l_qt2, l_E)
    //
    // P rows (for each char field):
    //   W[0][m] = 0.5*(b2+un*ioc)*rho + 0.5*(-b*un-ioc)*qn + (-0.5*b*ut1)*qt1 + (-0.5*b*ut2)*qt2 + 0.5*b*E
    //   W[1][m] = (1-b2)*rho + b*un*qn + b*ut1*qt1 + b*ut2*qt2 + (-b)*E
    //   W[2][m] = (-ut1)*rho + qt1
    //   W[3][m] = (-ut2)*rho + qt2
    //   W[4][m] = 0.5*(b2-un*ioc)*rho + 0.5*(-b*un+ioc)*qn + (-0.5*b*ut1)*qt1 + (-0.5*b*ut2)*qt2 + 0.5*b*E
    //
    // P^T:
    //   l_rho = 0.5*(b2+un*ioc)*lW0 + (1-b2)*lW1 + (-ut1)*lW2 + (-ut2)*lW3 + 0.5*(b2-un*ioc)*lW4
    //   l_qn  = 0.5*(-b*un-ioc)*lW0 + b*un*lW1 + 0.5*(-b*un+ioc)*lW4
    //   l_qt1 = (-0.5*b*ut1)*lW0 + b*ut1*lW1 + 1*lW2 + (-0.5*b*ut1)*lW4
    //   l_qt2 = (-0.5*b*ut2)*lW0 + b*ut2*lW1 + 1*lW3 + (-0.5*b*ut2)*lW4
    //   l_E   = 0.5*b*lW0 + (-b)*lW1 + 0.5*b*lW4

    const double b   = cf.b;
    const double b2  = cf.b2;
    const double ioc = cf.ioc;
    const double un  = cf.un;
    const double ut1 = cf.ut1;
    const double ut2 = cf.ut2;

    for (int m = 0; m < 7; ++m) {
        const double lW0 = l_W[0][m];
        const double lW1 = l_W[1][m];
        const double lW2 = l_W[2][m];
        const double lW3 = l_W[3][m];
        const double lW4 = l_W[4][m];

        // Adjoint in conserved space (rho, qn, qt1, qt2, E)
        const double l_rho = 0.5*(b2 + un*ioc)*lW0 + (1.0-b2)*lW1
                           + (-ut1)*lW2 + (-ut2)*lW3
                           + 0.5*(b2 - un*ioc)*lW4;
        const double l_qn  = 0.5*(-b*un - ioc)*lW0 + b*un*lW1
                           + 0.5*(-b*un + ioc)*lW4;
        const double l_qt1 = (-0.5*b*ut1)*lW0 + b*ut1*lW1 + lW2
                           + (-0.5*b*ut1)*lW4;
        const double l_qt2 = (-0.5*b*ut2)*lW0 + b*ut2*lW1 + lW3
                           + (-0.5*b*ut2)*lW4;
        const double l_E   = b*(0.5*lW0 - lW1 + 0.5*lW4);

        // Remap from (rho, qn, qt1, qt2, E) → (rho, Q[1], Q[2], Q[3], E)
        // n_idx_c is the conserved momentum index for the normal direction
        double l_Q[NVAR] = {};
        l_Q[0]        = l_rho;
        l_Q[n_idx_c]  = l_qn;
        l_Q[t1_idx_c] = l_qt1;
        l_Q[t2_idx_c] = l_qt2;
        l_Q[4]        = l_E;

        // ── Step 4: prim→cons adjoint for stencil cell m ──────────────────────
        // Q[m] = K(pc[flat_m]) maps prim → conserved.
        // acc_adj_prim_to_cons(pc[flat_m], l_Q, l_pc[flat_m]) accumulates
        // ∂J/∂prim into l_pc[flat_m][0..4] = [rho, u, v, w, p].
        const int flat_m = idx_at(m - 3);
        acc_adj_prim_to_cons(pc[flat_m], l_Q, l_pc[flat_m]);
    }
}
