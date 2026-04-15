// sgs.cpp — Smagorinsky SGS model implementation
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
#include "../include/sgs.hpp"
#include <cmath>
#include <algorithm>

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

// ── SmagorinskyModel::apply ───────────────────────────────────────────────────
//
// Conservative face-centred divergence of the SGS stress tensor.
//
// For each cell (i,j,k) the momentum increments are:
//
//   dQ[1]/dt = (1/h)*[tau_xx|_{i+1/2} - tau_xx|_{i-1/2}
//                   + tau_xy|_{j+1/2} - tau_xy|_{j-1/2}
//                   + tau_xz|_{k+1/2} - tau_xz|_{k-1/2}]
//
// and analogously for y and z.  The face stresses use face-averaged mu_t:
//
//   tau_xx|_{i+1/2} = mu_t_{i+1/2} * (2*du/dx - (2/3)*div_u) |_{i+1/2}
//
// where mu_t_{i+1/2} = 0.5*(mu_t(i,j,k) + mu_t(i+1,j,k))
// and the velocity gradients are taken as 1st-order forward/backward
// differences at the face (equivalent to 2nd-order centred at the face
// using the two neighbouring cell centres).
//
// The energy increment subtracts the SGS viscous work (dissipation):
//   dQ[4]/dt = -(tau:S) + kap_t * lap_T
//             = -(tau_xx*dudx + ... + 2*tau_xy*Sxy + ...) + kap_t*lap_T
//
// Note: the negative sign for visc_work is physically required — a positive
// tau:S represents conversion of resolved KE to SGS modes (dissipation).
// =============================================================================
void SmagorinskyModel::apply(CellBlock& blk, double h, double dt) const {
    const double h_inv   = 1.0 / h;
    const double ih2     = h_inv * h_inv;
    const double ih_half = 0.5 * h_inv;
    const double CsD2    = (Cs * h) * (Cs * h);
    const double kap_fac = SGS_CP / Pr_t;  // kap_t = mu_t * kap_fac

    // Precompute mu_t at every cell (interior + 1-ghost layer for face averages)
    static thread_local std::vector<double> mu_t_arr;
    mu_t_arr.assign(NCELL, 0.0);
    for (int k = NG-1; k < NG+NB+1; ++k)
    for (int j = NG-1; j < NG+NB+1; ++j)
    for (int i = NG-1; i < NG+NB+1; ++i) {
        if (i < 0 || i >= NB2 || j < 0 || j >= NB2 || k < 0 || k >= NB2) continue;
        // Only compute for cells that have a full stencil for strain_rate
        if (i < 1 || i > NB2-2 || j < 1 || j > NB2-2 || k < 1 || k > NB2-2) continue;
        double rho_ijk = blk.Q[0][cell_idx(i,j,k)];
        double S       = strain_rate(blk, h_inv, i, j, k);
        mu_t_arr[cell_idx(i,j,k)] = rho_ijk * CsD2 * S;
    }

    // S08-fix2: apply periodic wrap to mu_t ghost cells so that face
    // averages mu_{i+1/2} = 0.5*(mu[i]+mu[i+1]) are correct at block
    // boundaries.  Without this, the +x/+y/+z ghost layer (index NB2-1)
    // has mu_t=0, halving the face viscosity and breaking the telescoping
    // sum => global momentum not conserved (S07/S08).
    // x-direction
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j) {
        mu_t_arr[cell_idx(0,     j, k)] = mu_t_arr[cell_idx(NB2-2, j, k)];
        mu_t_arr[cell_idx(NB2-1, j, k)] = mu_t_arr[cell_idx(1,     j, k)];
    }
    // y-direction
    for (int k = 0; k < NB2; ++k)
    for (int i = 0; i < NB2; ++i) {
        mu_t_arr[cell_idx(i, 0,     k)] = mu_t_arr[cell_idx(i, NB2-2, k)];
        mu_t_arr[cell_idx(i, NB2-1, k)] = mu_t_arr[cell_idx(i, 1,     k)];
    }
    // z-direction
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        mu_t_arr[cell_idx(i, j, 0    )] = mu_t_arr[cell_idx(i, j, NB2-2)];
        mu_t_arr[cell_idx(i, j, NB2-1)] = mu_t_arr[cell_idx(i, j, 1    )];
    }

    static thread_local std::array<std::array<double,NCELL>,NVAR> dQ;
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            dQ[v][idx] = 0.0;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        Prim q = blk.prim(i, j, k);

        // ── Velocity accessor (rhou/rho) ────────────────────────────────────
        auto vel = [&](int mom, int ii, int jj, int kk) -> double {
            return blk.Q[mom][cell_idx(ii,jj,kk)]
                 / blk.Q[0  ][cell_idx(ii,jj,kk)];
        };
        auto u = [&](int ii,int jj,int kk){ return vel(1,ii,jj,kk); };
        auto v = [&](int ii,int jj,int kk){ return vel(2,ii,jj,kk); };
        auto w = [&](int ii,int jj,int kk){ return vel(3,ii,jj,kk); };

        // ── mu_t accessor ───────────────────────────────────────────────────
        auto mu = [&](int ii, int jj, int kk) -> double {
            return mu_t_arr[cell_idx(ii,jj,kk)];
        };

        // ── Face-averaged mu_t ──────────────────────────────────────────────
        double mu_xp = 0.5*(mu(i,j,k)+mu(i+1,j,k));
        double mu_xm = 0.5*(mu(i,j,k)+mu(i-1,j,k));
        double mu_yp = 0.5*(mu(i,j,k)+mu(i,j+1,k));
        double mu_ym = 0.5*(mu(i,j,k)+mu(i,j-1,k));
        double mu_zp = 0.5*(mu(i,j,k)+mu(i,j,k+1));
        double mu_zm = 0.5*(mu(i,j,k)+mu(i,j,k-1));

        // ── Velocity gradients at each face (forward/backward 1st-order) ────
        // x-faces (i+1/2 and i-1/2):
        double dudx_xp = h_inv*(u(i+1,j,k)-u(i,j,k));
        double dvdx_xp = h_inv*(v(i+1,j,k)-v(i,j,k));
        double dwdx_xp = h_inv*(w(i+1,j,k)-w(i,j,k));
        double dudy_xp = ih_half*(u(i+1,j+1,k)-u(i+1,j-1,k)+u(i,j+1,k)-u(i,j-1,k));
        double dudz_xp = ih_half*(u(i+1,j,k+1)-u(i+1,j,k-1)+u(i,j,k+1)-u(i,j,k-1));
        double dvdy_xp = ih_half*(v(i+1,j+1,k)-v(i+1,j-1,k)+v(i,j+1,k)-v(i,j-1,k));
        double dwdz_xp = ih_half*(w(i+1,j,k+1)-w(i+1,j,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double dvdx_xm = h_inv*(v(i,j,k)-v(i-1,j,k));
        double dwdx_xm = h_inv*(w(i,j,k)-w(i-1,j,k));
        double dudx_xm = h_inv*(u(i,j,k)-u(i-1,j,k));
        double dudy_xm = ih_half*(u(i-1,j+1,k)-u(i-1,j-1,k)+u(i,j+1,k)-u(i,j-1,k));
        double dudz_xm = ih_half*(u(i-1,j,k+1)-u(i-1,j,k-1)+u(i,j,k+1)-u(i,j,k-1));
        double dvdy_xm = ih_half*(v(i-1,j+1,k)-v(i-1,j-1,k)+v(i,j+1,k)-v(i,j-1,k));
        double dwdz_xm = ih_half*(w(i-1,j,k+1)-w(i-1,j,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double divu_xp = dudx_xp + dvdy_xp + dwdz_xp;
        double divu_xm = dudx_xm + dvdy_xm + dwdz_xm;

        // y-faces (j+1/2 and j-1/2):
        double dvdy_yp = h_inv*(v(i,j+1,k)-v(i,j,k));
        double dudx_yp = ih_half*(u(i+1,j+1,k)-u(i-1,j+1,k)+u(i+1,j,k)-u(i-1,j,k));
        double dwdz_yp = ih_half*(w(i,j+1,k+1)-w(i,j+1,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double dudy_yp = h_inv*(u(i,j+1,k)-u(i,j,k));
        double dvdx_yp = ih_half*(v(i+1,j+1,k)-v(i-1,j+1,k)+v(i+1,j,k)-v(i-1,j,k));
        double dvdz_yp = ih_half*(v(i,j+1,k+1)-v(i,j+1,k-1)+v(i,j,k+1)-v(i,j,k-1));
        double dwdy_yp = h_inv*(w(i,j+1,k)-w(i,j,k));
        double dvdy_ym = h_inv*(v(i,j,k)-v(i,j-1,k));
        double dudx_ym = ih_half*(u(i+1,j-1,k)-u(i-1,j-1,k)+u(i+1,j,k)-u(i-1,j,k));
        double dwdz_ym = ih_half*(w(i,j-1,k+1)-w(i,j-1,k-1)+w(i,j,k+1)-w(i,j,k-1));
        double dudy_ym = h_inv*(u(i,j,k)-u(i,j-1,k));
        double dvdx_ym = ih_half*(v(i+1,j-1,k)-v(i-1,j-1,k)+v(i+1,j,k)-v(i-1,j,k));
        double dvdz_ym = ih_half*(v(i,j-1,k+1)-v(i,j-1,k-1)+v(i,j,k+1)-v(i,j,k-1));
        double dwdy_ym = h_inv*(w(i,j,k)-w(i,j-1,k));
        double divu_yp = dudx_yp + dvdy_yp + dwdz_yp;
        double divu_ym = dudx_ym + dvdy_ym + dwdz_ym;

        // z-faces (k+1/2 and k-1/2):
        double dwdz_zp = h_inv*(w(i,j,k+1)-w(i,j,k));
        double dudx_zp = ih_half*(u(i+1,j,k+1)-u(i-1,j,k+1)+u(i+1,j,k)-u(i-1,j,k));
        double dvdy_zp = ih_half*(v(i,j+1,k+1)-v(i,j-1,k+1)+v(i,j+1,k)-v(i,j-1,k));
        double dudz_zp = h_inv*(u(i,j,k+1)-u(i,j,k));
        double dwdx_zp = ih_half*(w(i+1,j,k+1)-w(i-1,j,k+1)+w(i+1,j,k)-w(i-1,j,k));
        double dvdz_zp = h_inv*(v(i,j,k+1)-v(i,j,k));
        double dwdy_zp = ih_half*(w(i,j+1,k+1)-w(i,j-1,k+1)+w(i,j+1,k)-w(i,j-1,k));
        double dwdz_zm = h_inv*(w(i,j,k)-w(i,j,k-1));
        double dudx_zm = ih_half*(u(i+1,j,k-1)-u(i-1,j,k-1)+u(i+1,j,k)-u(i-1,j,k));
        double dvdy_zm = ih_half*(v(i,j+1,k-1)-v(i,j-1,k-1)+v(i,j+1,k)-v(i,j-1,k));
        double dudz_zm = h_inv*(u(i,j,k)-u(i,j,k-1));
        double dwdx_zm = ih_half*(w(i+1,j,k-1)-w(i-1,j,k-1)+w(i+1,j,k)-w(i-1,j,k));
        double dvdz_zm = h_inv*(v(i,j,k)-v(i,j,k-1));
        double dwdy_zm = ih_half*(w(i,j+1,k-1)-w(i,j-1,k-1)+w(i,j+1,k)-w(i,j-1,k));
        double divu_zp = dudx_zp + dvdy_zp + dwdz_zp;
        double divu_zm = dudx_zm + dvdy_zm + dwdz_zm;

        // ── Face stresses ───────────────────────────────────────────────────
        // tau_xx|_{i+1/2}, tau_xy|_{i+1/2}, tau_xz|_{i+1/2} etc.
        // tau_xx = mu*(2*du/dx - (2/3)*div_u)
        // tau_xy = mu*(du/dy + dv/dx)   [symmetric: tau_yx = tau_xy]
        // tau_xz = mu*(du/dz + dw/dx)   etc.
        double txx_xp = mu_xp*(2.0*dudx_xp - (2.0/3.0)*divu_xp);
        double txy_xp = mu_xp*(dudy_xp + dvdx_xp);
        double txz_xp = mu_xp*(dudz_xp + dwdx_xp);

        double txx_xm = mu_xm*(2.0*dudx_xm - (2.0/3.0)*divu_xm);
        double txy_xm = mu_xm*(dudy_xm + dvdx_xm);
        double txz_xm = mu_xm*(dudz_xm + dwdx_xm);

        double tyx_yp = mu_yp*(dudy_yp + dvdx_yp);   // tau_yx = tau_xy
        double tyy_yp = mu_yp*(2.0*dvdy_yp - (2.0/3.0)*divu_yp);
        double tyz_yp = mu_yp*(dvdz_yp + dwdy_yp);

        double tyx_ym = mu_ym*(dudy_ym + dvdx_ym);
        double tyy_ym = mu_ym*(2.0*dvdy_ym - (2.0/3.0)*divu_ym);
        double tyz_ym = mu_ym*(dvdz_ym + dwdy_ym);

        double tzx_zp = mu_zp*(dudz_zp + dwdx_zp);   // tau_zx = tau_xz
        double tzy_zp = mu_zp*(dvdz_zp + dwdy_zp);   // tau_zy = tau_yz
        double tzz_zp = mu_zp*(2.0*dwdz_zp - (2.0/3.0)*divu_zp);

        double tzx_zm = mu_zm*(dudz_zm + dwdx_zm);
        double tzy_zm = mu_zm*(dvdz_zm + dwdy_zm);
        double tzz_zm = mu_zm*(2.0*dwdz_zm - (2.0/3.0)*divu_zm);

        // ── Conservative divergence of SGS stress → momentum increments ────
        double ax = h_inv*((txx_xp - txx_xm) + (tyx_yp - tyx_ym) + (tzx_zp - tzx_zm));
        double ay = h_inv*((txy_xp - txy_xm) + (tyy_yp - tyy_ym) + (tzy_zp - tzy_zm));
        double az = h_inv*((txz_xp - txz_xm) + (tyz_yp - tyz_ym) + (tzz_zp - tzz_zm));

        // ── Cell-centred stress/strain for energy term ──────────────────────
        // Use cell-centred values for the dissipation integral tau:S.
        double dudx_c=ih_half*(u(i+1,j,k)-u(i-1,j,k));
        double dudy_c=ih_half*(u(i,j+1,k)-u(i,j-1,k));
        double dudz_c=ih_half*(u(i,j,k+1)-u(i,j,k-1));
        double dvdx_c=ih_half*(v(i+1,j,k)-v(i-1,j,k));
        double dvdy_c=ih_half*(v(i,j+1,k)-v(i,j-1,k));
        double dvdz_c=ih_half*(v(i,j,k+1)-v(i,j,k-1));
        double dwdx_c=ih_half*(w(i+1,j,k)-w(i-1,j,k));
        double dwdy_c=ih_half*(w(i,j+1,k)-w(i,j-1,k));
        double dwdz_c=ih_half*(w(i,j,k+1)-w(i,j,k-1));
        double divu_c = dudx_c + dvdy_c + dwdz_c;
        double mu_c   = mu_t_arr[cell_idx(i,j,k)];
        double txx_c  = mu_c*(2.0*dudx_c - (2.0/3.0)*divu_c);
        double tyy_c  = mu_c*(2.0*dvdy_c - (2.0/3.0)*divu_c);
        double tzz_c  = mu_c*(2.0*dwdz_c - (2.0/3.0)*divu_c);
        double txy_c  = mu_c*(dudy_c + dvdx_c);
        double txz_c  = mu_c*(dudz_c + dwdx_c);
        double tyz_c  = mu_c*(dvdz_c + dwdy_c);
        // tau:S = tau_ij * S_ij = tau_ij * (du_i/dx_j + du_j/dx_i)/2
        //       = txx*dudx + tyy*dvdy + tzz*dwdz
        //       + txy*(dudy+dvdx) + txz*(dudz+dwdx) + tyz*(dvdz+dwdy)
        double visc_work = txx_c*dudx_c + tyy_c*dvdy_c + tzz_c*dwdz_c
                         + txy_c*(dudy_c+dvdx_c)
                         + txz_c*(dudz_c+dwdx_c)
                         + tyz_c*(dvdz_c+dwdy_c);

        // Heat conduction: kap_t * Lap(T)
        double lap_T = ih2*(blk.T(i+1,j,k)-2.0*q.T+blk.T(i-1,j,k)
                              + blk.T(i,j+1,k)-2.0*q.T+blk.T(i,j-1,k)
                              + blk.T(i,j,k+1)-2.0*q.T+blk.T(i,j,k-1));
        double kap_t = mu_c * kap_fac;
        double heat  = kap_t * lap_T;

        // ── Accumulate increments ──────────────────────────────────────────
        int idx = cell_idx(i,j,k);
        dQ[1][idx] += dt * ax;
        dQ[2][idx] += dt * ay;
        dQ[3][idx] += dt * az;
        // S03-fix: visc_work is dissipation (tau:S > 0), subtract from energy
        dQ[4][idx] += dt * (heat - visc_work);
    }

    // Apply increments to interior cells
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int idx = cell_idx(i,j,k);
        for (int v = 0; v < NVAR; ++v)
            blk.Q[v][idx] += dQ[v][idx];
    }
}
