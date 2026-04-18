// sgs.cpp — SGS model implementations: Smagorinsky + DynamicSmagorinsky
//
// Fix log:
//   C2      : SGS momentum divergence includes full compressible
//             grad(div u) correction term (1/3)*mu_t*grad(div u).
//   S08-fix : Replace non-conservative cell-centred form mu_t*Lap(u_i)
//             with the face-centred divergence of the full SGS stress
//             tensor div(tau_sgs).  mu_t at each face is the arithmetic
//             mean of the two adjacent cells.  This form telescopes on
//             a periodic domain: sum_i div(tau)_i = 0 exactly, satisfying
//             global momentum conservation (S08 gate).
//   S03-fix : SGS visc_work sign corrected.  tau:S > 0 represents
//             dissipation of resolved KE into SGS modes.  The operator-
//             split energy increment must therefore *subtract* visc_work
//             (dQ[4] -= dt*visc_work), lowering KE as required by S03.
//             Previously the increment was +visc_work (energy injection),
//             causing Smagorinsky KE > NullSGS KE.
//   S08-fix2: Periodic ghost wrap applied to mu_t after the precomputation
//             loop.  Without this, the +x/+y/+z ghost layer has mu_t=0,
//             halving the face viscosity at block boundaries and breaking
//             the telescoping sum => global momentum not conserved.
//   P3.4    : DynamicSmagorinskyModel added.  Germano identity + Lilly LS
//             with 3×3×3 box test filter at 2Δ.  Shared stress-divergence
//             helper apply_sgs_stress_div() extracted to avoid duplication.
#include "../include/sgs.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

// ── Shared thermodynamic constants (same values as in operators.cpp) ─────────
static constexpr double SGS_PR_T_DEFAULT = 0.9;
static constexpr double SGS_CP = GAMMA * R_GAS / (GAMMA - 1.0);

// ── Strain-rate magnitude ─────────────────────────────────────────────────────
double SmagorinskyModel::strain_rate(const CellBlock& b, double h_inv,
                                     int i, int j, int k) {
    double h2 = 0.5 * h_inv;
    double Sxx = h2*(b.u(i+1,j,k) - b.u(i-1,j,k));
    double Syy = h2*(b.v(i,j+1,k) - b.v(i,j-1,k));
    double Szz = h2*(b.w(i,j,k+1) - b.w(i,j,k-1));
    double Sxy = 0.5*h2*((b.u(i,j+1,k)-b.u(i,j-1,k))+(b.v(i+1,j,k)-b.v(i-1,j,k)));
    double Sxz = 0.5*h2*((b.u(i,j,k+1)-b.u(i,j,k-1))+(b.w(i+1,j,k)-b.w(i-1,j,k)));
    double Syz = 0.5*h2*((b.v(i,j,k+1)-b.v(i,j,k-1))+(b.w(i,j+1,k)-b.w(i,j-1,k)));
    return std::sqrt(2.0*(Sxx*Sxx + Syy*Syy + Szz*Szz
                        + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz)));
}

// =============================================================================
// apply_sgs_stress_div — shared helper
//
// Applies the conservative face-centred divergence of the SGS stress tensor
// to blk in-place, given a precomputed mu_t array (size NCELL).
// Inserts periodic ghost values in mu_t before face-average computation.
// Called by both SmagorinskyModel::apply and DynamicSmagorinskyModel::apply.
// =============================================================================
static void apply_sgs_stress_div(CellBlock& blk, double h, double dt,
                                   std::vector<double>& mu_t_arr, double Pr_t_val)
{
    const double h_inv   = 1.0 / h;
    const double ih2     = h_inv * h_inv;
    const double ih_half = 0.5 * h_inv;
    const double kap_fac = SGS_CP / Pr_t_val;

    // S08-fix2: periodic ghost wrap so face averages are correct at block edges.
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j) {
        mu_t_arr[cell_idx(0,     j, k)] = mu_t_arr[cell_idx(NB2-2, j, k)];
        mu_t_arr[cell_idx(NB2-1, j, k)] = mu_t_arr[cell_idx(1,     j, k)];
    }
    for (int k = 0; k < NB2; ++k)
    for (int i = 0; i < NB2; ++i) {
        mu_t_arr[cell_idx(i, 0,     k)] = mu_t_arr[cell_idx(i, NB2-2, k)];
        mu_t_arr[cell_idx(i, NB2-1, k)] = mu_t_arr[cell_idx(i, 1,     k)];
    }
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        mu_t_arr[cell_idx(i, j, 0    )] = mu_t_arr[cell_idx(i, j, NB2-2)];
        mu_t_arr[cell_idx(i, j, NB2-1)] = mu_t_arr[cell_idx(i, j, 1    )];
    }

    static thread_local std::array<std::array<double,NCELL>,NVAR> dQ;
    for (int v = 0; v < NVAR; ++v)
        for (int c2 = 0; c2 < NCELL; ++c2)
            dQ[v][c2] = 0.0;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        Prim q = blk.prim(i, j, k);

        auto vel = [&](int mom, int ii, int jj, int kk) -> double {
            return blk.Q[mom][cell_idx(ii,jj,kk)] / blk.Q[0][cell_idx(ii,jj,kk)];
        };
        auto u = [&](int ii,int jj,int kk){ return vel(1,ii,jj,kk); };
        auto v = [&](int ii,int jj,int kk){ return vel(2,ii,jj,kk); };
        auto w = [&](int ii,int jj,int kk){ return vel(3,ii,jj,kk); };
        auto mu = [&](int ii,int jj,int kk) -> double {
            return mu_t_arr[cell_idx(ii,jj,kk)];
        };

        double mu_xp=0.5*(mu(i,j,k)+mu(i+1,j,k));
        double mu_xm=0.5*(mu(i,j,k)+mu(i-1,j,k));
        double mu_yp=0.5*(mu(i,j,k)+mu(i,j+1,k));
        double mu_ym=0.5*(mu(i,j,k)+mu(i,j-1,k));
        double mu_zp=0.5*(mu(i,j,k)+mu(i,j,k+1));
        double mu_zm=0.5*(mu(i,j,k)+mu(i,j,k-1));

        // x-faces
        double dudx_xp=h_inv*(u(i+1,j,k)-u(i,j,k));
        double dvdx_xp=h_inv*(v(i+1,j,k)-v(i,j,k));
        double dwdx_xp=h_inv*(w(i+1,j,k)-w(i,j,k));
        double dudy_xp=ih_half*(u(i+1,j+1,k)-u(i+1,j-1,k)+u(i,j+1,k)-u(i,j-1,k));
        double dudz_xp=ih_half*(u(i+1,j,k+1)-u(i+1,j,k-1)+u(i,j,k+1)-u(i,j,k-1));
        double dvdy_xp=ih_half*(v(i+1,j+1,k)-v(i+1,j-1,k)+v(i,j+1,k)-v(i,j-1,k));
        double dwdz_xp=ih_half*(w(i+1,j,k+1)-w(i+1,j,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double dudx_xm=h_inv*(u(i,j,k)-u(i-1,j,k));
        double dvdx_xm=h_inv*(v(i,j,k)-v(i-1,j,k));
        double dwdx_xm=h_inv*(w(i,j,k)-w(i-1,j,k));
        double dudy_xm=ih_half*(u(i-1,j+1,k)-u(i-1,j-1,k)+u(i,j+1,k)-u(i,j-1,k));
        double dudz_xm=ih_half*(u(i-1,j,k+1)-u(i-1,j,k-1)+u(i,j,k+1)-u(i,j,k-1));
        double dvdy_xm=ih_half*(v(i-1,j+1,k)-v(i-1,j-1,k)+v(i,j+1,k)-v(i,j-1,k));
        double dwdz_xm=ih_half*(w(i-1,j,k+1)-w(i-1,j,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double divu_xp=dudx_xp+dvdy_xp+dwdz_xp;
        double divu_xm=dudx_xm+dvdy_xm+dwdz_xm;

        // y-faces
        double dvdy_yp=h_inv*(v(i,j+1,k)-v(i,j,k));
        double dudx_yp=ih_half*(u(i+1,j+1,k)-u(i-1,j+1,k)+u(i+1,j,k)-u(i-1,j,k));
        double dwdz_yp=ih_half*(w(i,j+1,k+1)-w(i,j+1,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double dudy_yp=h_inv*(u(i,j+1,k)-u(i,j,k));
        double dvdx_yp=ih_half*(v(i+1,j+1,k)-v(i-1,j+1,k)+v(i+1,j,k)-v(i-1,j,k));
        double dvdz_yp=ih_half*(v(i,j+1,k+1)-v(i,j+1,k-1)+v(i,j,k+1)-v(i,j,k-1));
        double dwdy_yp=h_inv*(w(i,j+1,k)-w(i,j,k));
        double dvdy_ym=h_inv*(v(i,j,k)-v(i,j-1,k));
        double dudx_ym=ih_half*(u(i+1,j-1,k)-u(i-1,j-1,k)+u(i+1,j,k)-u(i-1,j,k));
        double dwdz_ym=ih_half*(w(i,j-1,k+1)-w(i,j-1,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double dudy_ym=h_inv*(u(i,j,k)-u(i,j-1,k));
        double dvdx_ym=ih_half*(v(i+1,j-1,k)-v(i-1,j-1,k)+v(i+1,j,k)-v(i-1,j,k));
        double dvdz_ym=ih_half*(v(i,j-1,k+1)-v(i,j-1,k-1)+v(i,j,k+1)-v(i,j,k-1));
        double dwdy_ym=h_inv*(w(i,j,k)-w(i,j-1,k));
        double divu_yp=dudx_yp+dvdy_yp+dwdz_yp;
        double divu_ym=dudx_ym+dvdy_ym+dwdz_ym;

        // z-faces
        double dwdz_zp=h_inv*(w(i,j,k+1)-w(i,j,k));
        double dudx_zp=ih_half*(u(i+1,j,k+1)-u(i-1,j,k+1)+u(i+1,j,k)-u(i-1,j,k));
        double dvdy_zp=ih_half*(v(i,j+1,k+1)-v(i,j-1,k+1)+v(i,j+1,k)-v(i,j-1,k));
        double dudz_zp=h_inv*(u(i,j,k+1)-u(i,j,k));
        double dwdx_zp=ih_half*(w(i+1,j,k+1)-w(i-1,j,k+1)+w(i+1,j,k)-w(i-1,j,k));
        double dvdz_zp=h_inv*(v(i,j,k+1)-v(i,j,k));
        double dwdy_zp=ih_half*(w(i,j+1,k+1)-w(i,j-1,k+1)+w(i,j+1,k)-w(i,j-1,k));
        double dwdz_zm=h_inv*(w(i,j,k)-w(i,j,k-1));
        double dudx_zm=ih_half*(u(i+1,j,k-1)-u(i-1,j,k-1)+u(i+1,j,k)-u(i-1,j,k));
        double dvdy_zm=ih_half*(v(i,j+1,k-1)-v(i,j-1,k-1)+v(i,j+1,k)-v(i,j-1,k));
        double dudz_zm=h_inv*(u(i,j,k)-u(i,j,k-1));
        double dwdx_zm=ih_half*(w(i+1,j,k-1)-w(i-1,j,k-1)+w(i+1,j,k)-w(i-1,j,k));
        double dvdz_zm=h_inv*(v(i,j,k)-v(i,j,k-1));
        double dwdy_zm=ih_half*(w(i,j+1,k-1)-w(i,j-1,k-1)+w(i,j+1,k)-w(i,j-1,k));
        double divu_zp=dudx_zp+dvdy_zp+dwdz_zp;
        double divu_zm=dudx_zm+dvdy_zm+dwdz_zm;

        double txx_xp=mu_xp*(2.0*dudx_xp-(2.0/3.0)*divu_xp);
        double txy_xp=mu_xp*(dudy_xp+dvdx_xp);
        double txz_xp=mu_xp*(dudz_xp+dwdx_xp);
        double txx_xm=mu_xm*(2.0*dudx_xm-(2.0/3.0)*divu_xm);
        double txy_xm=mu_xm*(dudy_xm+dvdx_xm);
        double txz_xm=mu_xm*(dudz_xm+dwdx_xm);
        double tyx_yp=mu_yp*(dudy_yp+dvdx_yp);
        double tyy_yp=mu_yp*(2.0*dvdy_yp-(2.0/3.0)*divu_yp);
        double tyz_yp=mu_yp*(dvdz_yp+dwdy_yp);
        double tyx_ym=mu_ym*(dudy_ym+dvdx_ym);
        double tyy_ym=mu_ym*(2.0*dvdy_ym-(2.0/3.0)*divu_ym);
        double tyz_ym=mu_ym*(dvdz_ym+dwdy_ym);
        double tzx_zp=mu_zp*(dudz_zp+dwdx_zp);
        double tzy_zp=mu_zp*(dvdz_zp+dwdy_zp);
        double tzz_zp=mu_zp*(2.0*dwdz_zp-(2.0/3.0)*divu_zp);
        double tzx_zm=mu_zm*(dudz_zm+dwdx_zm);
        double tzy_zm=mu_zm*(dvdz_zm+dwdy_zm);
        double tzz_zm=mu_zm*(2.0*dwdz_zm-(2.0/3.0)*divu_zm);

        double ax=h_inv*((txx_xp-txx_xm)+(tyx_yp-tyx_ym)+(tzx_zp-tzx_zm));
        double ay=h_inv*((txy_xp-txy_xm)+(tyy_yp-tyy_ym)+(tzy_zp-tzy_zm));
        double az=h_inv*((txz_xp-txz_xm)+(tyz_yp-tyz_ym)+(tzz_zp-tzz_zm));

        // Cell-centred tau:S for energy dissipation and heat conduction
        double dudx_c=ih_half*(u(i+1,j,k)-u(i-1,j,k));
        double dudy_c=ih_half*(u(i,j+1,k)-u(i,j-1,k));
        double dudz_c=ih_half*(u(i,j,k+1)-u(i,j,k-1));
        double dvdx_c=ih_half*(v(i+1,j,k)-v(i-1,j,k));
        double dvdy_c=ih_half*(v(i,j+1,k)-v(i,j-1,k));
        double dvdz_c=ih_half*(v(i,j,k+1)-v(i,j,k-1));
        double dwdx_c=ih_half*(w(i+1,j,k)-w(i-1,j,k));
        double dwdy_c=ih_half*(w(i,j+1,k)-w(i,j-1,k));
        double dwdz_c=ih_half*(w(i,j,k+1)-w(i,j,k-1));
        double divu_c=dudx_c+dvdy_c+dwdz_c;
        double mu_c=mu_t_arr[cell_idx(i,j,k)];
        double txx_c=mu_c*(2.0*dudx_c-(2.0/3.0)*divu_c);
        double tyy_c=mu_c*(2.0*dvdy_c-(2.0/3.0)*divu_c);
        double tzz_c=mu_c*(2.0*dwdz_c-(2.0/3.0)*divu_c);
        double txy_c=mu_c*(dudy_c+dvdx_c);
        double txz_c=mu_c*(dudz_c+dwdx_c);
        double tyz_c=mu_c*(dvdz_c+dwdy_c);
        double visc_work=txx_c*dudx_c+tyy_c*dvdy_c+tzz_c*dwdz_c
                        +txy_c*(dudy_c+dvdx_c)+txz_c*(dudz_c+dwdx_c)+tyz_c*(dvdz_c+dwdy_c);

        double lap_T=ih2*(blk.T(i+1,j,k)-2.0*q.T+blk.T(i-1,j,k)
                         +blk.T(i,j+1,k)-2.0*q.T+blk.T(i,j-1,k)
                         +blk.T(i,j,k+1)-2.0*q.T+blk.T(i,j,k-1));
        double heat=mu_c*kap_fac*lap_T;

        int c2=cell_idx(i,j,k);
        dQ[1][c2]+=dt*ax;
        dQ[2][c2]+=dt*ay;
        dQ[3][c2]+=dt*az;
        dQ[4][c2]+=dt*(heat-visc_work);
    }

    for (int k=NG; k<NG+NB; ++k)
    for (int j=NG; j<NG+NB; ++j)
    for (int i=NG; i<NG+NB; ++i) {
        int c2=cell_idx(i,j,k);
        for (int v=0; v<NVAR; ++v)
            blk.Q[v][c2]+=dQ[v][c2];
    }
}

void SmagorinskyModel::apply(CellBlock& blk, double h, double dt) const {
    const double h_inv = 1.0 / h;
    const double CsD2  = (Cs * h) * (Cs * h);

    static thread_local std::vector<double> mu_t_arr;
    mu_t_arr.assign(NCELL, 0.0);
    for (int k = 1; k < NB2-1; ++k)
    for (int j = 1; j < NB2-1; ++j)
    for (int i = 1; i < NB2-1; ++i) {
        double rho_ijk = blk.Q[0][cell_idx(i,j,k)];
        double S       = strain_rate(blk, h_inv, i, j, k);
        mu_t_arr[cell_idx(i,j,k)] = rho_ijk * CsD2 * S;
    }

    apply_sgs_stress_div(blk, h, dt, mu_t_arr, Pr_t);
}

// =============================================================================
// DynamicSmagorinskyModel::apply — Germano + Lilly LS, 3×3×3 test filter
// =============================================================================
void DynamicSmagorinskyModel::apply(CellBlock& blk, double h, double dt) const
{
    const double h_inv  = 1.0 / h;
    const double ih2    = 0.5 * h_inv;  // factor for central diffs
    const double Delta2 = h * h;
    constexpr double inv27 = 1.0 / 27.0;

    auto vel = [&](int comp, int i, int j, int k) -> double {
        return blk.Q[comp][cell_idx(i,j,k)] / blk.Q[0][cell_idx(i,j,k)];
    };

    // ── Step 1: S_ij and |S̄| for [1..NB2-2] ─────────────────────────────────
    // Layout: Sij[idx][0..5] = {Sxx,Sxy,Sxz,Syy,Syz,Szz}
    static thread_local std::array<std::array<double,6>, NCELL> Sij;
    static thread_local std::array<double,               NCELL> Smag;
    // Zero ghost-layer boundary indices so test-filter accesses at i=1,di=-1
    // don't read stale values from the previous leaf processed by this thread.
    Sij.fill({});
    Smag.fill(0.0);

    for (int k = 1; k < NB2-1; ++k)
    for (int j = 1; j < NB2-1; ++j)
    for (int i = 1; i < NB2-1; ++i) {
        int idx = cell_idx(i,j,k);
        double Sxx = ih2*(vel(1,i+1,j,k)-vel(1,i-1,j,k));
        double Syy = ih2*(vel(2,i,j+1,k)-vel(2,i,j-1,k));
        double Szz = ih2*(vel(3,i,j,k+1)-vel(3,i,j,k-1));
        double Sxy = 0.5*ih2*((vel(1,i,j+1,k)-vel(1,i,j-1,k))
                              +(vel(2,i+1,j,k)-vel(2,i-1,j,k)));
        double Sxz = 0.5*ih2*((vel(1,i,j,k+1)-vel(1,i,j,k-1))
                              +(vel(3,i+1,j,k)-vel(3,i-1,j,k)));
        double Syz = 0.5*ih2*((vel(2,i,j,k+1)-vel(2,i,j,k-1))
                              +(vel(3,i,j+1,k)-vel(3,i,j-1,k)));
        Sij[idx][0]=Sxx; Sij[idx][1]=Sxy; Sij[idx][2]=Sxz;
        Sij[idx][3]=Syy; Sij[idx][4]=Syz; Sij[idx][5]=Szz;
        Smag[idx] = std::sqrt(2.0*(Sxx*Sxx+Syy*Syy+Szz*Szz
                              + 2.0*(Sxy*Sxy+Sxz*Sxz+Syz*Syz)));
    }

    // ── Step 2: 3×3×3 box test-filter over [1..NB2-2] ────────────────────────
    // Computes ũ_i, (u_i u_j)~, (|S̄| S_ij)~
    static thread_local std::array<double,NCELL> u_tf, v_tf, w_tf;
    static thread_local std::array<std::array<double,6>,NCELL> uiuj_tf;  // uu,uv,uw,vv,vw,ww
    static thread_local std::array<std::array<double,6>,NCELL> SmSij_tf; // (|S̄|S_ij)~
    u_tf.fill(0.0); v_tf.fill(0.0); w_tf.fill(0.0);
    uiuj_tf.fill({}); SmSij_tf.fill({});

    for (int k = 1; k < NB2-1; ++k)
    for (int j = 1; j < NB2-1; ++j)
    for (int i = 1; i < NB2-1; ++i) {
        int idx = cell_idx(i,j,k);
        double su=0, sv=0, sw=0;
        double suiuj[6]={0,0,0,0,0,0};
        double sSmSij[6]={0,0,0,0,0,0};
        for (int dk=-1; dk<=1; ++dk)
        for (int dj=-1; dj<=1; ++dj)
        for (int di=-1; di<=1; ++di) {
            int ni=i+di, nj=j+dj, nk=k+dk;
            int nidx = cell_idx(ni,nj,nk);
            double ui=vel(1,ni,nj,nk), vi=vel(2,ni,nj,nk), wi=vel(3,ni,nj,nk);
            double sm=Smag[nidx];
            su+=ui; sv+=vi; sw+=wi;
            suiuj[0]+=ui*ui; suiuj[1]+=ui*vi; suiuj[2]+=ui*wi;
            suiuj[3]+=vi*vi; suiuj[4]+=vi*wi; suiuj[5]+=wi*wi;
            for (int c=0; c<6; ++c) sSmSij[c]+=sm*Sij[nidx][c];
        }
        u_tf[idx]=su*inv27; v_tf[idx]=sv*inv27; w_tf[idx]=sw*inv27;
        for (int c=0; c<6; ++c) {
            uiuj_tf[idx][c] =suiuj[c]*inv27;
            SmSij_tf[idx][c]=sSmSij[c]*inv27;
        }
    }

    // ── Steps 3–4: Germano identity + Lilly LS over interior ─────────────────
    // S̃_ij from central diffs of ũ,ṽ,w̃ (available at [1..NB2-2])
    double sum_LM = 0.0, sum_MM = 0.0;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int idx = cell_idx(i,j,k);

        auto utf = [&](int ii,int jj,int kk){ return u_tf[cell_idx(ii,jj,kk)]; };
        auto vtf = [&](int ii,int jj,int kk){ return v_tf[cell_idx(ii,jj,kk)]; };
        auto wtf = [&](int ii,int jj,int kk){ return w_tf[cell_idx(ii,jj,kk)]; };

        double Stxx=ih2*(utf(i+1,j,k)-utf(i-1,j,k));
        double Styy=ih2*(vtf(i,j+1,k)-vtf(i,j-1,k));
        double Stzz=ih2*(wtf(i,j,k+1)-wtf(i,j,k-1));
        double Stxy=0.5*ih2*((utf(i,j+1,k)-utf(i,j-1,k))+(vtf(i+1,j,k)-vtf(i-1,j,k)));
        double Stxz=0.5*ih2*((utf(i,j,k+1)-utf(i,j,k-1))+(wtf(i+1,j,k)-wtf(i-1,j,k)));
        double Styz=0.5*ih2*((vtf(i,j,k+1)-vtf(i,j,k-1))+(wtf(i,j+1,k)-wtf(i,j-1,k)));
        double Stmag=std::sqrt(2.0*(Stxx*Stxx+Styy*Styy+Stzz*Stzz
                               + 2.0*(Stxy*Stxy+Stxz*Stxz+Styz*Styz)));

        // Leonard stress L_ij = (u_i u_j)~ - ũ_i ũ_j
        double uf=u_tf[idx], vf=v_tf[idx], wf=w_tf[idx];
        double Lxx=uiuj_tf[idx][0]-uf*uf, Lxy=uiuj_tf[idx][1]-uf*vf;
        double Lxz=uiuj_tf[idx][2]-uf*wf, Lyy=uiuj_tf[idx][3]-vf*vf;
        double Lyz=uiuj_tf[idx][4]-vf*wf, Lzz=uiuj_tf[idx][5]-wf*wf;

        // M_ij = 2Δ²(4|S̃|S̃_ij - (|S̄|S_ij)~)
        double M[6];
        M[0]=2.0*Delta2*(4.0*Stmag*Stxx-SmSij_tf[idx][0]);
        M[1]=2.0*Delta2*(4.0*Stmag*Stxy-SmSij_tf[idx][1]);
        M[2]=2.0*Delta2*(4.0*Stmag*Stxz-SmSij_tf[idx][2]);
        M[3]=2.0*Delta2*(4.0*Stmag*Styy-SmSij_tf[idx][3]);
        M[4]=2.0*Delta2*(4.0*Stmag*Styz-SmSij_tf[idx][4]);
        M[5]=2.0*Delta2*(4.0*Stmag*Stzz-SmSij_tf[idx][5]);

        sum_LM += Lxx*M[0]+Lyy*M[3]+Lzz*M[5]
                + 2.0*(Lxy*M[1]+Lxz*M[2]+Lyz*M[4]);
        sum_MM += M[0]*M[0]+M[3]*M[3]+M[5]*M[5]
                + 2.0*(M[1]*M[1]+M[2]*M[2]+M[4]*M[4]);
    }

    // Lilly LS: C_s² ≥ 0 (no backscatter)
    double Cs2 = (sum_MM > 1e-300) ? std::max(0.0, sum_LM / sum_MM) : 0.0;

    // ── Step 5: mu_t = ρ · C_s² · Δ² · |S̄|  →  shared stress-div ─────────
    static thread_local std::vector<double> mu_t_arr;
    mu_t_arr.assign(NCELL, 0.0);
    for (int k = 1; k < NB2-1; ++k)
    for (int j = 1; j < NB2-1; ++j)
    for (int i = 1; i < NB2-1; ++i) {
        int idx = cell_idx(i,j,k);
        mu_t_arr[idx] = blk.Q[0][idx] * Cs2 * Delta2 * Smag[idx];
    }

    apply_sgs_stress_div(blk, h, dt, mu_t_arr, Pr_t);
}
