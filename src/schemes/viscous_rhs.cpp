// =============================================================================
// viscous_rhs.cpp — R9-E3: viscous RHS kernel extracted from operators.cpp
// =============================================================================
// Contains: viscous_rhs_impl, cf_visc_energy_flux, undo_cf_viscous_energy.
// Non-static functions are forward-declared in operators.cpp.

#include "schemes/operators.hpp"
#include "physics/diff_ops.hpp"
#include "physics/face_interp.hpp"
#include <cmath>

static constexpr double CP = CPU_CP;

// =============================================================================
// viscous_rhs_impl — face-averaged µ, conservative divergence form (B5)
// Non-static: called from compute_rhs / compute_rhs_typed in operators.cpp.
// =============================================================================
void viscous_rhs_impl(const Prim* pc, const double* mu_arr,
                      CellBlock& rhs, double h) noexcept
{
    const double ih = 1.0 / h;

    auto U  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].u; };
    auto V  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].v; };
    auto W  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].w; };
    auto Tf = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].T; };
    auto MU = [&](int ii,int jj,int kk){ return mu_arr[cell_idx(ii,jj,kk)]; };

    constexpr VelocityGradAtFace<Axis::X, 2> VGX;
    constexpr VelocityGradAtFace<Axis::Y, 2> VGY;
    constexpr VelocityGradAtFace<Axis::Z, 2> VGZ;
    constexpr FaceInterp<Axis::X> FI_X;
    constexpr FaceInterp<Axis::Y> FI_Y;
    constexpr FaceInterp<Axis::Z> FI_Z;

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {

        // ── Face-averaged µ at the 6 faces ───────────────────────────────────
        double mu_xp = FI_X(MU, i,   j,   k  );
        double mu_xm = FI_X(MU, i-1, j,   k  );
        double mu_yp = FI_Y(MU, i,   j,   k  );
        double mu_ym = FI_Y(MU, i,   j-1, k  );
        double mu_zp = FI_Z(MU, i,   j,   k  );
        double mu_zm = FI_Z(MU, i,   j,   k-1);

        // ── Velocity gradient tensors at 6 faces (R7: VelocityGradAtFace) ──
        const auto gxp = VGX.plus (U, V, W, i, j, k, h);
        const auto gxm = VGX.minus(U, V, W, i, j, k, h);
        const auto gyp = VGY.plus (U, V, W, i, j, k, h);
        const auto gym = VGY.minus(U, V, W, i, j, k, h);
        const auto gzp = VGZ.plus (U, V, W, i, j, k, h);
        const auto gzm = VGZ.minus(U, V, W, i, j, k, h);

        // ── Face stresses τ = µ·(∂u_a/∂x_b + ∂u_b/∂x_a − δ·⅔ div u) ──────
        // x-faces
        double txx_xp = mu_xp*(2.0*gxp.dun_dxn - (2.0/3.0)*gxp.divu());
        double txy_xp = mu_xp*(gxp.dun_dxt1 + gxp.dut1_dxn);
        double txz_xp = mu_xp*(gxp.dun_dxt2 + gxp.dut2_dxn);
        double txx_xm = mu_xm*(2.0*gxm.dun_dxn - (2.0/3.0)*gxm.divu());
        double txy_xm = mu_xm*(gxm.dun_dxt1 + gxm.dut1_dxn);
        double txz_xm = mu_xm*(gxm.dun_dxt2 + gxm.dut2_dxn);
        // y-faces
        double tyx_yp = mu_yp*(gyp.dut1_dxn + gyp.dun_dxt1);
        double tyy_yp = mu_yp*(2.0*gyp.dun_dxn - (2.0/3.0)*gyp.divu());
        double tyz_yp = mu_yp*(gyp.dun_dxt2 + gyp.dut2_dxn);
        double tyx_ym = mu_ym*(gym.dut1_dxn + gym.dun_dxt1);
        double tyy_ym = mu_ym*(2.0*gym.dun_dxn - (2.0/3.0)*gym.divu());
        double tyz_ym = mu_ym*(gym.dun_dxt2 + gym.dut2_dxn);
        // z-faces
        double tzx_zp = mu_zp*(gzp.dut1_dxn + gzp.dun_dxt1);
        double tzy_zp = mu_zp*(gzp.dut2_dxn + gzp.dun_dxt2);
        double tzz_zp = mu_zp*(2.0*gzp.dun_dxn - (2.0/3.0)*gzp.divu());
        double tzx_zm = mu_zm*(gzm.dut1_dxn + gzm.dun_dxt1);
        double tzy_zm = mu_zm*(gzm.dut2_dxn + gzm.dun_dxt2);
        double tzz_zm = mu_zm*(2.0*gzm.dun_dxn - (2.0/3.0)*gzm.divu());

        // ── Conservative momentum divergences ─────────────────────────────────
        double ax = ih*((txx_xp-txx_xm) + (tyx_yp-tyx_ym) + (tzx_zp-tzx_zm));
        double ay = ih*((txy_xp-txy_xm) + (tyy_yp-tyy_ym) + (tzy_zp-tzy_zm));
        double az = ih*((txz_xp-txz_xm) + (tyz_yp-tyz_ym) + (tzz_zp-tzz_zm));

        // ── Energy: conservative face-flux form  div(τ·u + κ∇T) ──────────────
        auto UF=[&](int ii,int jj,int kk){return 0.5*(U(ii,jj,kk)+U(i,j,k));};
        auto VF=[&](int ii,int jj,int kk){return 0.5*(V(ii,jj,kk)+V(i,j,k));};
        auto WF=[&](int ii,int jj,int kk){return 0.5*(W(ii,jj,kk)+W(i,j,k));};
        // x-faces
        double kxp = mu_xp*CP/PR, kxm = mu_xm*CP/PR;
        double Fex_p = txx_xp*UF(i+1,j,k) + txy_xp*VF(i+1,j,k) + txz_xp*WF(i+1,j,k)
                     + kxp*ih*(Tf(i+1,j,k)-Tf(i,j,k));
        double Fex_m = txx_xm*UF(i-1,j,k) + txy_xm*VF(i-1,j,k) + txz_xm*WF(i-1,j,k)
                     + kxm*ih*(Tf(i,j,k)-Tf(i-1,j,k));
        // y-faces
        double kyp = mu_yp*CP/PR, kym = mu_ym*CP/PR;
        double Fey_p = tyx_yp*UF(i,j+1,k) + tyy_yp*VF(i,j+1,k) + tyz_yp*WF(i,j+1,k)
                     + kyp*ih*(Tf(i,j+1,k)-Tf(i,j,k));
        double Fey_m = tyx_ym*UF(i,j-1,k) + tyy_ym*VF(i,j-1,k) + tyz_ym*WF(i,j-1,k)
                     + kym*ih*(Tf(i,j,k)-Tf(i,j-1,k));
        // z-faces
        double kzp = mu_zp*CP/PR, kzm = mu_zm*CP/PR;
        double Fez_p = tzx_zp*UF(i,j,k+1) + tzy_zp*VF(i,j,k+1) + tzz_zp*WF(i,j,k+1)
                     + kzp*ih*(Tf(i,j,k+1)-Tf(i,j,k));
        double Fez_m = tzx_zm*UF(i,j,k-1) + tzy_zm*VF(i,j,k-1) + tzz_zm*WF(i,j,k-1)
                     + kzm*ih*(Tf(i,j,k)-Tf(i,j,k-1));

        int idx = cell_idx(i,j,k);
        rhs.Q[1][idx] += ax;
        rhs.Q[2][idx] += ay;
        rhs.Q[3][idx] += az;
        rhs.Q[4][idx] += ih*(Fex_p-Fex_m) + ih*(Fey_p-Fey_m) + ih*(Fez_p-Fez_m);
    }
}

// =============================================================================
// cf_visc_energy_flux<AX> — viscous energy flux at one C/F face (R7)
// =============================================================================
template<Axis AX>
static double cf_visc_energy_flux(
    const CellBlock& blk, double h,
    int ci, int cj, int ck, int gi, int gj, int gk, double ns) noexcept
{
    const Prim p_i = blk.prim(ci,cj,ck);
    const Prim p_g = blk.prim(gi,gj,gk);
    const double mu_f  = 0.5*(sutherland(p_i.T) + sutherland(p_g.T));
    const double kappa = mu_f * CP / PR;
    const double u_f   = 0.5*(p_i.u + p_g.u);
    const double v_f   = 0.5*(p_i.v + p_g.v);
    const double w_f   = 0.5*(p_i.w + p_g.w);
    const double ih    = 1.0 / h;
    auto uf = [&](int ii,int jj,int kk){ return blk.prim(ii,jj,kk).u; };
    auto vf = [&](int ii,int jj,int kk){ return blk.prim(ii,jj,kk).v; };
    auto wf = [&](int ii,int jj,int kk){ return blk.prim(ii,jj,kk).w; };
    constexpr VelocityGradAtFace<AX, 2> VGA;
    const auto g = (ns > 0) ? VGA.plus(uf, vf, wf, ci, cj, ck, h)
                             : VGA.minus(uf, vf, wf, ci, cj, ck, h);
    const double tau_nn  = mu_f*(2.0*g.dun_dxn - (2.0/3.0)*g.divu());
    const double tau_nt1 = mu_f*(g.dun_dxt1 + g.dut1_dxn);
    const double tau_nt2 = mu_f*(g.dun_dxt2 + g.dut2_dxn);
    const double kT      = kappa * ns * ih * (p_g.T - p_i.T);
    if constexpr (AX == Axis::X)
        return tau_nn * u_f + tau_nt1 * v_f + tau_nt2 * w_f + kT;
    else if constexpr (AX == Axis::Y)
        return tau_nt1 * u_f + tau_nn * v_f + tau_nt2 * w_f + kT;
    else
        return tau_nt1 * u_f + tau_nt2 * v_f + tau_nn * w_f + kT;
}

// =============================================================================
// undo_cf_viscous_energy — Berger-Colella viscous energy reflux correction
// Non-static: called from tree_rhs in operators.cpp.
// =============================================================================
void undo_cf_viscous_energy(const BlockTree& tree, int node_idx,
                             CellBlock& rhs) noexcept
{
    const auto& nd   = tree.nodes[node_idx];
    const auto& blk  = *nd.block;
    const double h   = blk.h;
    const double ih  = 1.0 / h;

    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

    for (int d = 0; d < NFACES; ++d) {
        const int ni = nd.neighbours[d];
        if (ni < 0 || !tree.nodes[ni].has_block()) continue;
        if (tree.nodes[ni].level == nd.level) continue;

        const int axis   = face_axis[d];
        const int delta  = face_delta[d];
        const double ns  = (double)delta;
        const int bound  = (delta > 0) ? ihi() : ilo();
        const int gbound = bound + delta;

        for (int b = ilo(); b <= ihi(); ++b)
        for (int a = ilo(); a <= ihi(); ++a) {
            int ci, cj, ck, gi, gj, gk;
            if      (axis == 0) { ci=bound; cj=a; ck=b; gi=gbound; gj=a;      gk=b; }
            else if (axis == 1) { ci=a; cj=bound; ck=b; gi=a;      gj=gbound; gk=b; }
            else                { ci=a; cj=b; ck=bound; gi=a;      gj=b;      gk=gbound; }

            double Fvisc_E;
            if      (axis == 0) Fvisc_E = cf_visc_energy_flux<Axis::X>(blk,h,ci,cj,ck,gi,gj,gk,ns);
            else if (axis == 1) Fvisc_E = cf_visc_energy_flux<Axis::Y>(blk,h,ci,cj,ck,gi,gj,gk,ns);
            else                Fvisc_E = cf_visc_energy_flux<Axis::Z>(blk,h,ci,cj,ck,gi,gj,gk,ns);

            rhs.Q[4][cell_idx(ci,cj,ck)] -= ns * ih * Fvisc_E;
        }
    }
}
