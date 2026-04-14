// DESIGN.md reference: Layer 2 — Discrete Operators
#include "operators.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>

// =============================================================================
// HLLC Riemann flux
// =============================================================================
// Reference: Toro (2009) §10.4, Batten et al. (1997).
// axis: 0=x, 1=y, 2=z  — selects which velocity component is normal.
//
// Sign convention: flux is in the +axis direction.
// The five components of the returned flux correspond to:
//   [0] mass flux        rho * u_n
//   [1] x-momentum flux  rho*u*u_n + p*nx  (nx=1 for axis=0, else 0)
//   [2] y-momentum flux
//   [3] z-momentum flux
//   [4] energy flux      (E+p)*u_n

std::array<double,5> hllc_flux(const Prim& L, const Prim& R, int axis) noexcept
{
    double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
    double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

    double sL = std::min(uL - L.c, uR - R.c);
    double sR = std::max(uL + L.c, uR + R.c);

    double numer = R.p - L.p + L.rho*uL*(sL - uL) - R.rho*uR*(sR - uR);
    double denom = L.rho*(sL - uL) - R.rho*(sR - uR);
    double sStar = (std::abs(denom) > 1e-300) ? numer / denom : 0.5*(uL + uR);

    auto phys_flux = [&](const Prim& q) -> std::array<double,5> {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GAMMA-1.0) + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        return {q.rho*un,
                q.rho*q.u*un + (axis==0?q.p:0.0),
                q.rho*q.v*un + (axis==1?q.p:0.0),
                q.rho*q.w*un + (axis==2?q.p:0.0),
                (E+q.p)*un};
    };

    auto star_flux = [&](const Prim& q, double sK, double sS)
        -> std::array<double,5>
    {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GAMMA-1.0) + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        double coeff = q.rho * (sK - un) / (sK - sS);
        double rho_s  = coeff;
        double rhou_s = coeff * (axis==0 ? sS : q.u);
        double rhov_s = coeff * (axis==1 ? sS : q.v);
        double rhow_s = coeff * (axis==2 ? sS : q.w);
        double E_s    = coeff * (E/q.rho + (sS - un)*(sS + q.p/(q.rho*(sK-un))));

        auto F = phys_flux(q);
        double rho  = q.rho;
        double rhou = rho*q.u, rhov = rho*q.v, rhow = rho*q.w;
        return {F[0] + sK*(rho_s  - rho ),
                F[1] + sK*(rhou_s - rhou),
                F[2] + sK*(rhov_s - rhov),
                F[3] + sK*(rhow_s - rhow),
                F[4] + sK*(E_s    - E   )};
    };

    if      (sL >= 0.0) return phys_flux(L);
    else if (sR <= 0.0) return phys_flux(R);
    else if (sStar >= 0.0) return star_flux(L, sL, sStar);
    else                   return star_flux(R, sR, sStar);
}

// =============================================================================
// Convective RHS  — HLLC
// =============================================================================
void convective_rhs(const CellBlock& blk, CellBlock& rhs) noexcept
{
    double ih = 1.0 / blk.h;

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        Prim c  = blk.prim(i,  j,  k );
        Prim xm = blk.prim(i-1,j,  k );
        Prim xp = blk.prim(i+1,j,  k );
        Prim ym = blk.prim(i,  j-1,k );
        Prim yp = blk.prim(i,  j+1,k );
        Prim zm = blk.prim(i,  j,  k-1);
        Prim zp = blk.prim(i,  j,  k+1);

        auto Fxp = hllc_flux(c,  xp, 0);
        auto Fxm = hllc_flux(xm, c,  0);
        auto Gyp = hllc_flux(c,  yp, 1);
        auto Gym = hllc_flux(ym, c,  1);
        auto Hzp = hllc_flux(c,  zp, 2);
        auto Hzm = hllc_flux(zm, c,  2);

        for (int v = 0; v < NVAR; ++v) {
            rhs.Q[v][cell_idx(i,j,k)] -=
                ih * ((Fxp[v]-Fxm[v]) + (Gyp[v]-Gym[v]) + (Hzp[v]-Hzm[v]));
        }
    }
}

// =============================================================================
// Viscous RHS -- full compressible N-S viscous operator
// =============================================================================
// Momentum: a_i = mu*Lap(u_i) + (1/3)*mu * d(div u)/dx_i
//
// grad(div u) components (FIX C1 -- all cross-partials now included):
//   d(div u)/dx = d2u/dx2  + d2v/dxdy + d2w/dxdz
//   d(div u)/dy = d2u/dydx + d2v/dy2  + d2w/dydz
//   d(div u)/dz = d2u/dzdx + d2v/dzdy + d2w/dz2
//
// Cross-partial: d2f/dxdy = (f_{i+1,j+1} - f_{i+1,j-1}
//                           -f_{i-1,j+1} + f_{i-1,j-1}) / (4h^2)
// =============================================================================
static constexpr double PR = 0.72;
static constexpr double CP = GAMMA * R_GAS / (GAMMA - 1.0);

void viscous_rhs(const CellBlock& blk, CellBlock& rhs) noexcept
{
    const double ih      = 1.0 / blk.h;
    const double ih2     = ih * ih;
    const double ih_half = 0.5 * ih;
    const double ih2_q   = 0.25 * ih2;

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {

        Prim c =blk.prim(i,j,k);
        Prim xm=blk.prim(i-1,j,k),  xp=blk.prim(i+1,j,k);
        Prim ym=blk.prim(i,j-1,k),  yp=blk.prim(i,j+1,k);
        Prim zm=blk.prim(i,j,k-1),  zp=blk.prim(i,j,k+1);

        Prim xpyp=blk.prim(i+1,j+1,k), xpym=blk.prim(i+1,j-1,k);
        Prim xmyp=blk.prim(i-1,j+1,k), xmym=blk.prim(i-1,j-1,k);
        Prim xpzp=blk.prim(i+1,j,k+1), xpzm=blk.prim(i+1,j,k-1);
        Prim xmzp=blk.prim(i-1,j,k+1), xmzm=blk.prim(i-1,j,k-1);
        Prim ypzp=blk.prim(i,j+1,k+1), ypzm=blk.prim(i,j+1,k-1);
        Prim ymzp=blk.prim(i,j-1,k+1), ymzm=blk.prim(i,j-1,k-1);

        double dudx=ih_half*(xp.u-xm.u), dudy=ih_half*(yp.u-ym.u), dudz=ih_half*(zp.u-zm.u);
        double dvdx=ih_half*(xp.v-xm.v), dvdy=ih_half*(yp.v-ym.v), dvdz=ih_half*(zp.v-zm.v);
        double dwdx=ih_half*(xp.w-xm.w), dwdy=ih_half*(yp.w-ym.w), dwdz=ih_half*(zp.w-zm.w);
        double divU = dudx + dvdy + dwdz;

        double mu    = sutherland(c.T);
        double kappa = mu * CP / PR;

        double txx=mu*(2.0*dudx-(2.0/3.0)*divU);
        double tyy=mu*(2.0*dvdy-(2.0/3.0)*divU);
        double tzz=mu*(2.0*dwdz-(2.0/3.0)*divU);
        double txy=mu*(dudy+dvdx), txz=mu*(dudz+dwdx), tyz=mu*(dvdz+dwdy);

        double lap_u=ih2*(xp.u-2*c.u+xm.u + yp.u-2*c.u+ym.u + zp.u-2*c.u+zm.u);
        double lap_v=ih2*(xp.v-2*c.v+xm.v + yp.v-2*c.v+ym.v + zp.v-2*c.v+zm.v);
        double lap_w=ih2*(xp.w-2*c.w+xm.w + yp.w-2*c.w+ym.w + zp.w-2*c.w+zm.w);
        double lap_T=ih2*(xp.T-2*c.T+xm.T + yp.T-2*c.T+ym.T + zp.T-2*c.T+zm.T);

        double d2u_dx2=ih2*(xp.u-2*c.u+xm.u);
        double d2v_dy2=ih2*(yp.v-2*c.v+ym.v);
        double d2w_dz2=ih2*(zp.w-2*c.w+zm.w);

        double d2v_dxdy=ih2_q*(xpyp.v-xpym.v-xmyp.v+xmym.v);
        double d2w_dxdz=ih2_q*(xpzp.w-xpzm.w-xmzp.w+xmzm.w);
        double d2u_dydx=ih2_q*(xpyp.u-xpym.u-xmyp.u+xmym.u);
        double d2w_dydz=ih2_q*(ypzp.w-ypzm.w-ymzp.w+ymzm.w);
        double d2u_dzdx=ih2_q*(xpzp.u-xpzm.u-xmzp.u+xmzm.u);
        double d2v_dzdy=ih2_q*(ypzp.v-ypzm.v-ymzp.v+ymzm.v);

        double gdivx = d2u_dx2  + d2v_dxdy + d2w_dxdz;
        double gdivy = d2u_dydx + d2v_dy2  + d2w_dydz;
        double gdivz = d2u_dzdx + d2v_dzdy + d2w_dz2;

        double ax = mu * (lap_u + (1.0/3.0)*gdivx);
        double ay = mu * (lap_v + (1.0/3.0)*gdivy);
        double az = mu * (lap_w + (1.0/3.0)*gdivz);

        double visc_pw = txx*dudx + tyy*dvdy + tzz*dwdz
                       + txy*(dudy+dvdx) + txz*(dudz+dwdx) + tyz*(dvdz+dwdy);
        double heat    = kappa * lap_T;

        int idx = cell_idx(i,j,k);
        rhs.Q[1][idx] += ax;
        rhs.Q[2][idx] += ay;
        rhs.Q[3][idx] += az;
        rhs.Q[4][idx] += heat + visc_pw;
    }
}

// =============================================================================
// Full RHS
// =============================================================================
void compute_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    for (int v = 0; v < NVAR; ++v)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i)
            rhs_blk.Q[v][cell_idx(i,j,k)] = 0.0;

    convective_rhs(blk, rhs_blk);
    viscous_rhs   (blk, rhs_blk);
}

// =============================================================================
// Berger-Colella reflux: undo_cf_face_flux
// ─────────────────────────────────────────
// Undoes the coarse HLLC contribution at each CF face so that
// apply_flux_correction() can add the correct fine-side average.
// Net effect after correction:
//   dQ/dt += (1/h) * [F_fine_avg - F_coarse_own]
// =============================================================================
static void undo_cf_face_flux(const BlockTree& tree, int node_idx,
                               CellBlock& rhs) noexcept
{
    const auto& nd  = tree.nodes[node_idx];
    const auto& blk = *nd.block;
    const double ih = 1.0 / blk.h;

    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

    for (int d = 0; d < NFACES; ++d) {
        const int ni = nd.neighbours[d];
        if (ni < 0 || !tree.nodes[ni].has_block()) continue;
        if (tree.nodes[ni].level <= nd.level) continue;

        const int axis  = face_axis[d];
        const int delta = face_delta[d];
        const int bound = (delta > 0) ? ihi() : ilo();

        for (int b = ilo(); b <= ihi(); ++b)
        for (int a = ilo(); a <= ihi(); ++a) {
            int ci, cj, ck, gi, gj, gk;
            if (axis == 0) {
                ci=bound; cj=a;     ck=b;
                gi=bound+delta; gj=a; gk=b;
            } else if (axis == 1) {
                ci=a;     cj=bound; ck=b;
                gi=a; gj=bound+delta; gk=b;
            } else {
                ci=a;     cj=b;     ck=bound;
                gi=a; gj=b; gk=bound+delta;
            }

            Prim interior = blk.prim(ci, cj, ck);
            Prim ghost    = blk.prim(gi, gj, gk);

            std::array<double,5> F;
            if (delta > 0)
                F = hllc_flux(interior, ghost, axis);
            else
                F = hllc_flux(ghost, interior, axis);

            const double sign = (delta > 0) ? +1.0 : -1.0;
            const int idx = cell_idx(ci, cj, ck);
            for (int v = 0; v < NVAR; ++v)
                rhs.Q[v][idx] += sign * ih * F[v];
        }
    }
}

// =============================================================================
// accumulate_cf_fine_fluxes
// ─────────────────────────
// A05-fix4: stage_weight is the SSP-RK3 quadrature weight for this stage:
//   stage 1: w = 1/6,  stage 2: w = 1/6,  stage 3: w = 2/3
// Registers are zeroed ONCE before stage 1 (in advance()).
// Each call accumulates w * F_fine; apply_flux_correction(dt) then applies
// the time-averaged fine flux, giving exact Berger-Colella conservation
// across all three RK3 sub-steps.
//
// Register layout (must match apply_flux_correction in block_tree.cpp):
//   reg[v*NB*NB + jc*NB + ic]
//   axis=0 (x-face, YZ plane): jc = a-ilo() (y), ic = b-ilo() (z)
//   axis=1 (y-face, XZ plane): jc = b-ilo() (z), ic = a-ilo() (x)
//   axis=2 (z-face, XY plane): jc = b-ilo() (y), ic = a-ilo() (x)
//
// FIX A05-fix5: the branch condition for jc/ic was (axis==1), which put
// axis=2 (z-face, a→x, b→y) into the else branch and stored
// jc=a-ilo()=x-offset, ic=b-ilo()=y-offset — transposing x↔y in every
// z-face register entry.  apply_flux_correction reads ic as x and jc as y,
// so the correction landed on the wrong (ci,cj) cell pair, breaking mass
// conservation on any non-isotropic 3-D field.
// Corrected: use (axis==0) as the branch so axis=1 and axis=2 both take
// the b→first-index, a→second-index path, matching the documented layout.
// =============================================================================
static void accumulate_cf_fine_fluxes(BlockTree& tree,
                                       double stage_weight) noexcept
{
    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

    for (int li : tree.leaf_indices()) {
        const auto& nd  = tree.nodes[li];
        const auto& blk = *nd.block;

        for (int d = 0; d < NFACES; ++d) {
            const int ni = nd.neighbours[d];
            if (ni < 0 || !tree.nodes[ni].has_block()) continue;
            if (tree.nodes[ni].level >= nd.level) continue;

            const int axis  = face_axis[d];
            const int delta = face_delta[d];
            const int bound = (delta > 0) ? ihi() : ilo();

            std::vector<double> face_flux(NVAR * NB * NB, 0.0);

            for (int b = ilo(); b <= ihi(); ++b)
            for (int a = ilo(); a <= ihi(); ++a) {
                int ci, cj, ck, gi, gj, gk;
                if (axis == 0) {
                    ci=bound; cj=a;     ck=b;
                    gi=bound+delta; gj=a; gk=b;
                } else if (axis == 1) {
                    ci=a;     cj=bound; ck=b;
                    gi=a; gj=bound+delta; gk=b;
                } else {
                    ci=a;     cj=b;     ck=bound;
                    gi=a; gj=b; gk=bound+delta;
                }

                Prim interior = blk.prim(ci, cj, ck);
                Prim ghost    = blk.prim(gi, gj, gk);

                std::array<double,5> F;
                if (delta > 0) F = hllc_flux(interior, ghost, axis);
                else           F = hllc_flux(ghost, interior, axis);

                // FIX A05-fix5: use (axis==0) instead of (axis==1) so that
                // axis=2 (z-face, a→x, b→y) maps jc=b-ilo()=y, ic=a-ilo()=x,
                // consistent with apply_flux_correction (ci=ilo()+ic, cj=ilo()+jc).
                int jc = (axis == 0) ? (a - ilo()) : (b - ilo());
                int ic = (axis == 0) ? (b - ilo()) : (a - ilo());
                for (int v = 0; v < NVAR; ++v)
                    face_flux[v*NB*NB + jc*NB + ic] = stage_weight * F[v];
            }
            tree.accumulate_fine_flux(li, static_cast<FaceDir>(d), face_flux);
        }
    }
}

// =============================================================================
// tree_rhs
// ─────────
// A05-fix4: stage_weight is the SSP-RK3 quadrature weight (1/6, 1/6, 2/3).
// The caller (advance()) must zero flux registers once before stage 1 and
// must NOT zero them between stages.
// =============================================================================
void tree_rhs(BlockTree& tree,
              std::vector<CellBlock>& rhs_blocks,
              bool periodic,
              double stage_weight) noexcept
{
    if (periodic)
        tree.fill_ghosts_periodic();
    else
        tree.fill_ghosts_wall();

    const auto& leaves = tree.leaf_indices();
    assert((int)rhs_blocks.size() == (int)leaves.size());
    for (int li = 0; li < (int)leaves.size(); ++li) {
        const int node_idx = leaves[li];
        compute_rhs(*tree.nodes[node_idx].block, rhs_blocks[li]);
        undo_cf_face_flux(tree, node_idx, rhs_blocks[li]);
    }

    accumulate_cf_fine_fluxes(tree, stage_weight);
}

// =============================================================================
// CFL time step
// =============================================================================
double tree_cfl_dt(const BlockTree& tree, double cfl) noexcept
{
    double dt = 1e300;
    for (auto li : tree.leaf_indices())
        dt = std::min(dt, tree.nodes[li].block->cfl_dt(cfl));
    return dt;
}
