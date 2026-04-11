// sgs.cpp — Smagorinsky SGS model implementation
// FIX C2: SGS momentum divergence now includes the full compressible
//         grad(div u) correction term (1/3)*mu_t*grad(div u), consistent
//         with the molecular viscous operator fixed in C1/P3.
//         Previously only mu_t*Lap(u_i) was applied — correct only for
//         incompressible flow (div u = 0).
//         Cross-partial stencil is the same 4-point corner stencil as C1;
//         all 12 edge neighbours are within NG=1 reach for interior cells.
#include "../include/sgs.hpp"
#include <cmath>
#include <algorithm>

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
void SmagorinskyModel::apply(CellBlock& blk, double h, double dt) const {
    const double h_inv  = 1.0 / h;
    const double ih2    = h_inv * h_inv;
    const double ih_half = 0.5 * h_inv;
    const double ih2_q  = 0.25 * ih2;   // 1/(4h^2) for cross-partial stencil
    const double CsD2   = (Cs * h) * (Cs * h);

    const int NC = NB2 * NB2 * NB2;
    static thread_local double dQ[NVAR][NC];
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NC; ++idx)
            dQ[v][idx] = 0.0;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        Prim q  = blk.prim(i, j, k);
        double S = strain_rate(blk, h_inv, i, j, k);

        double nu_t  = CsD2 * S;
        double mu_t  = q.rho * nu_t;
        double kap_t = mu_t * CPU_CP / Pr_t;

        // ── Velocity at stencil points (u = rhou/rho) ─────────────────────
        auto vel = [&](int mom, int ii, int jj, int kk) {
            return blk.Q[mom][cell_idx(ii,jj,kk)]
                 / blk.Q[0  ][cell_idx(ii,jj,kk)];
        };
        // Convenience wrappers
        auto u = [&](int ii,int jj,int kk){ return vel(1,ii,jj,kk); };
        auto v = [&](int ii,int jj,int kk){ return vel(2,ii,jj,kk); };
        auto w = [&](int ii,int jj,int kk){ return vel(3,ii,jj,kk); };

        // ── Velocity gradients ────────────────────────────────────────────
        double dudx=ih_half*(u(i+1,j,k)-u(i-1,j,k));
        double dudy=ih_half*(u(i,j+1,k)-u(i,j-1,k));
        double dudz=ih_half*(u(i,j,k+1)-u(i,j,k-1));
        double dvdx=ih_half*(v(i+1,j,k)-v(i-1,j,k));
        double dvdy=ih_half*(v(i,j+1,k)-v(i,j-1,k));
        double dvdz=ih_half*(v(i,j,k+1)-v(i,j,k-1));
        double dwdx=ih_half*(w(i+1,j,k)-w(i-1,j,k));
        double dwdy=ih_half*(w(i,j+1,k)-w(i,j-1,k));
        double dwdz=ih_half*(w(i,j,k+1)-w(i,j,k-1));
        double divU = dudx + dvdy + dwdz;

        // ── SGS stress tensor (for energy work term) ──────────────────────
        double txx=mu_t*(2.0*dudx-(2.0/3.0)*divU);
        double tyy=mu_t*(2.0*dvdy-(2.0/3.0)*divU);
        double tzz=mu_t*(2.0*dwdz-(2.0/3.0)*divU);
        double txy=mu_t*(dudy+dvdx);
        double txz=mu_t*(dudz+dwdx);
        double tyz=mu_t*(dvdz+dwdy);

        // ── Laplacians of velocity ─────────────────────────────────────────
        double lap_u=ih2*(u(i+1,j,k)-2*q.u+u(i-1,j,k)
                        + u(i,j+1,k)-2*q.u+u(i,j-1,k)
                        + u(i,j,k+1)-2*q.u+u(i,j,k-1));
        double lap_v=ih2*(v(i+1,j,k)-2*q.v+v(i-1,j,k)
                        + v(i,j+1,k)-2*q.v+v(i,j-1,k)
                        + v(i,j,k+1)-2*q.v+v(i,j,k-1));
        double lap_w=ih2*(w(i+1,j,k)-2*q.w+w(i-1,j,k)
                        + w(i,j+1,k)-2*q.w+w(i,j-1,k)
                        + w(i,j,k+1)-2*q.w+w(i,j,k-1));
        double lap_T=ih2*(blk.T(i+1,j,k)-2*q.T+blk.T(i-1,j,k)
                        + blk.T(i,j+1,k)-2*q.T+blk.T(i,j-1,k)
                        + blk.T(i,j,k+1)-2*q.T+blk.T(i,j,k-1));

        // ── Diagonal 2nd derivatives ───────────────────────────────────────
        double d2u_dx2=ih2*(u(i+1,j,k)-2*q.u+u(i-1,j,k));
        double d2v_dy2=ih2*(v(i,j+1,k)-2*q.v+v(i,j-1,k));
        double d2w_dz2=ih2*(w(i,j,k+1)-2*q.w+w(i,j,k-1));

        // ── FIX C2: cross-partial 2nd derivatives (4-point corner stencil) ─
        double d2v_dxdy=ih2_q*(v(i+1,j+1,k)-v(i+1,j-1,k)-v(i-1,j+1,k)+v(i-1,j-1,k));
        double d2w_dxdz=ih2_q*(w(i+1,j,k+1)-w(i+1,j,k-1)-w(i-1,j,k+1)+w(i-1,j,k-1));
        double d2u_dydx=ih2_q*(u(i+1,j+1,k)-u(i+1,j-1,k)-u(i-1,j+1,k)+u(i-1,j-1,k));
        double d2w_dydz=ih2_q*(w(i,j+1,k+1)-w(i,j+1,k-1)-w(i,j-1,k+1)+w(i,j-1,k-1));
        double d2u_dzdx=ih2_q*(u(i+1,j,k+1)-u(i+1,j,k-1)-u(i-1,j,k+1)+u(i-1,j,k-1));
        double d2v_dzdy=ih2_q*(v(i,j+1,k+1)-v(i,j+1,k-1)-v(i,j-1,k+1)+v(i,j-1,k-1));

        // ── Full grad(div u) — FIX C2 ──────────────────────────────────────
        double gdivx = d2u_dx2  + d2v_dxdy + d2w_dxdz;
        double gdivy = d2u_dydx + d2v_dy2  + d2w_dydz;
        double gdivz = d2u_dzdx + d2v_dzdy + d2w_dz2;

        // ── SGS momentum acceleration: mu_t*(Lap + (1/3)*grad(div u)) ──────
        double ax = mu_t * (lap_u + (1.0/3.0)*gdivx);
        double ay = mu_t * (lap_v + (1.0/3.0)*gdivy);
        double az = mu_t * (lap_w + (1.0/3.0)*gdivz);

        // ── Energy: SGS viscous work + SGS heat conduction ─────────────────
        double visc_work = txx*dudx + tyy*dvdy + tzz*dwdz
                         + txy*(dudy+dvdx) + txz*(dudz+dwdx) + tyz*(dvdz+dwdy);
        double heat      = kap_t * lap_T;

        // ── Accumulate increments ──────────────────────────────────────────
        int idx = cell_idx(i,j,k);
        dQ[1][idx] += dt * ax;
        dQ[2][idx] += dt * ay;
        dQ[3][idx] += dt * az;
        dQ[4][idx] += dt * (visc_work + heat);
    }

    // Apply increments
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int idx = cell_idx(i,j,k);
        for (int v = 0; v < NVAR; ++v)
            blk.Q[v][idx] += dQ[v][idx];
    }
}
