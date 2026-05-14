// =============================================================================
// rhs_sensors.cpp — R9-E3: shock sensors and source terms from operators.cpp
// =============================================================================
// Contains: fill_ducros_cache, phi_rhs, phi_compression_rhs, tree_sat_penalty.
// fill_ducros_cache is non-static: forward-declared and called in operators.cpp.

#include "schemes/operators.hpp"
#include "physics/diff_ops.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

// =============================================================================
// fill_ducros_cache — P3.2: combined Ducros + pressure-ratio sensor
// Non-static: called from convective_rhs / compute_rhs / compute_rhs_typed
//             in operators.cpp via forward declaration.
// =============================================================================
void fill_ducros_cache(const Prim* pc, double* duc, double h,
                       const DucrosConfig& ducros) noexcept
{
    constexpr double eps_duc = 1.0e-30;
    constexpr CellGrad<Axis::X, 2> dX;
    constexpr CellGrad<Axis::Y, 2> dY;
    constexpr CellGrad<Axis::Z, 2> dZ;

    auto U = [pc](int i, int j, int k){ return pc[cell_idx(i,j,k)].u; };
    auto V = [pc](int i, int j, int k){ return pc[cell_idx(i,j,k)].v; };
    auto W = [pc](int i, int j, int k){ return pc[cell_idx(i,j,k)].w; };

    std::fill(duc, duc + NCELL, 0.0);

    for (int k = 1; k < NB2-1; ++k)
    for (int j = 1; j < NB2-1; ++j)
    for (int i = 1; i < NB2-1; ++i) {
        const double dudx = dX(U, i, j, k, h);
        const double dudy = dY(U, i, j, k, h);
        const double dudz = dZ(U, i, j, k, h);
        const double dvdx = dX(V, i, j, k, h);
        const double dvdy = dY(V, i, j, k, h);
        const double dvdz = dZ(V, i, j, k, h);
        const double dwdx = dX(W, i, j, k, h);
        const double dwdy = dY(W, i, j, k, h);
        const double dwdz = dZ(W, i, j, k, h);

        const double divu = dudx + dvdy + dwdz;
        const double ox   = dwdy - dvdz;
        const double oy   = dudz - dwdx;
        const double oz   = dvdx - dudy;
        const double d2   = divu*divu;
        const double c2   = ox*ox + oy*oy + oz*oz;
        const double phi_vel = d2 / (d2 + c2 + eps_duc);

        const double pC   = pc[cell_idx(i,j,k)].p;
        const double dpx  = std::max(std::abs(pc[cell_idx(i+1,j,k)].p - pC),
                                     std::abs(pc[cell_idx(i-1,j,k)].p - pC));
        const double dpy  = std::max(std::abs(pc[cell_idx(i,j+1,k)].p - pC),
                                     std::abs(pc[cell_idx(i,j-1,k)].p - pC));
        const double dpz  = std::max(std::abs(pc[cell_idx(i,j,k+1)].p - pC),
                                     std::abs(pc[cell_idx(i,j,k-1)].p - pC));
        const double phi_p = std::max({dpx, dpy, dpz}) / (pC + eps_duc);
        const double blend = (ducros.blend_width > 1e-30) ? ducros.blend_width : 1e-30;
        const double phi_p_clamped = std::min(1.0, std::max(0.0,
            (phi_p - ducros.p_threshold) / blend));

        duc[cell_idx(i,j,k)] = std::max(phi_vel, phi_p_clamped);
    }
}

// =============================================================================
// P14.1 — ACDI phase-field advection RHS
// =============================================================================
void phi_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    const double inv_h = 1.0 / blk.h;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        const int flat = cell_idx(i,j,k);

        const Prim pC  = blk.prim(i,  j,  k  );
        const Prim pXm = blk.prim(i-1,j,  k  );
        const Prim pXp = blk.prim(i+1,j,  k  );
        const Prim pYm = blk.prim(i,  j-1,k  );
        const Prim pYp = blk.prim(i,  j+1,k  );
        const Prim pZm = blk.prim(i,  j,  k-1);
        const Prim pZp = blk.prim(i,  j,  k+1);

        const double phi_C  = blk.phi(i,  j,  k  );
        const double phi_Xm = blk.phi(i-1,j,  k  );
        const double phi_Xp = blk.phi(i+1,j,  k  );
        const double phi_Ym = blk.phi(i,  j-1,k  );
        const double phi_Yp = blk.phi(i,  j+1,k  );
        const double phi_Zm = blk.phi(i,  j,  k-1);
        const double phi_Zp = blk.phi(i,  j,  k+1);

        const double u_lx = 0.5*(pXm.u + pC.u);
        const double u_rx = 0.5*(pC.u  + pXp.u);
        const double f_lx = (u_lx >= 0.0) ? u_lx * phi_Xm : u_lx * phi_C;
        const double f_rx = (u_rx >= 0.0) ? u_rx * phi_C  : u_rx * phi_Xp;

        const double v_ly = 0.5*(pYm.v + pC.v);
        const double v_ry = 0.5*(pC.v  + pYp.v);
        const double f_ly = (v_ly >= 0.0) ? v_ly * phi_Ym : v_ly * phi_C;
        const double f_ry = (v_ry >= 0.0) ? v_ry * phi_C  : v_ry * phi_Yp;

        const double w_lz = 0.5*(pZm.w + pC.w);
        const double w_rz = 0.5*(pC.w  + pZp.w);
        const double f_lz = (w_lz >= 0.0) ? w_lz * phi_Zm : w_lz * phi_C;
        const double f_rz = (w_rz >= 0.0) ? w_rz * phi_C  : w_rz * phi_Zp;

        rhs_blk.phi_data_[flat] += inv_h * ((f_lx - f_rx) + (f_ly - f_ry) + (f_lz - f_rz));
    }
}

// =============================================================================
// P14.1b — ACDI interface-compression source
// =============================================================================
void phi_compression_rhs(const CellBlock& blk, CellBlock& rhs_blk,
                          double ceps) noexcept
{
    const double h      = blk.h;
    const double eps    = ceps * h;
    const double eps_sq = 1e-10 / (h * h);

    alignas(64) double Fx[NCELL] = {};
    alignas(64) double Fy[NCELL] = {};
    alignas(64) double Fz[NCELL] = {};

    static_assert(NG >= 2, "phi_compression_rhs needs NG>=2 for halo stencil");

    constexpr CellGrad<Axis::X, 2> dX;
    constexpr CellGrad<Axis::Y, 2> dY;
    constexpr CellGrad<Axis::Z, 2> dZ;
    auto Phi = [&blk](int i, int j, int k){ return blk.phi(i, j, k); };

    // Pass 1: compute interface-compression flux F = ε·(∇φ − φ(1−φ)·n̂)
    for (int k = NG-1; k <= NG+NB; ++k)
    for (int j = NG-1; j <= NG+NB; ++j)
    for (int i = NG-1; i <= NG+NB; ++i) {
        const double dpx   = dX(Phi, i, j, k, h);
        const double dpy   = dY(Phi, i, j, k, h);
        const double dpz   = dZ(Phi, i, j, k, h);
        const double mag2  = dpx*dpx + dpy*dpy + dpz*dpz + eps_sq;
        const double inv_mag = 1.0 / std::sqrt(mag2);
        const double phi_c = blk.phi(i, j, k);
        const double g     = phi_c * (1.0 - phi_c);
        const int flat     = cell_idx(i, j, k);
        Fx[flat] = eps * (dpx - g * dpx * inv_mag);
        Fy[flat] = eps * (dpy - g * dpy * inv_mag);
        Fz[flat] = eps * (dpz - g * dpz * inv_mag);
    }

    // Pass 2: central-difference divergence of F → add to rhs phi
    constexpr CellDiv<2> divOp;
    auto Fxa = [&Fx](int i, int j, int k){ return Fx[cell_idx(i,j,k)]; };
    auto Fya = [&Fy](int i, int j, int k){ return Fy[cell_idx(i,j,k)]; };
    auto Fza = [&Fz](int i, int j, int k){ return Fz[cell_idx(i,j,k)]; };

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        rhs_blk.phi_data_[cell_idx(i,j,k)] += divOp(Fxa, Fya, Fza, i, j, k, h);
}

// =============================================================================
// P13.5 — SBP-SAT interface penalty at AMR C/F boundaries
// =============================================================================
void tree_sat_penalty(BlockTree& tree,
                      std::vector<CellBlock>& rhs_blocks,
                      double tau) noexcept
{
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();
    if (NL == 0) return;

    std::vector<int> node_to_rhs(tree.nodes.size(), -1);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].has_block())
            node_to_rhs[leaves[ii]] = ii;
    }

    for (int ii = 0; ii < NL; ++ii) {
        int li = leaves[ii];
        const BlockNode& nd = tree.nodes[li];
        if (!nd.has_block()) continue;
        const CellBlock& blk = *nd.block;

        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            if (ni < 0 || ni >= (int)tree.nodes.size()) continue;
            const BlockNode& nnd = tree.nodes[ni];
            if (!nnd.is_leaf() || !nnd.has_block()) continue;
            if (nnd.block->h <= blk.h) continue;

            const int axis   = d / 2;
            const int side   = d % 2;
            const int face_i = (side == 0) ? ilo() : ihi();
            const int ghost_i = (side == 0) ? (ilo()-1) : (ihi()+1);

            const double sigma_f = tau / blk.h;
            const double sigma_c = sigma_f;

            CellBlock& rhs_f = rhs_blocks[ii];

            int ii_c = node_to_rhs[ni];
            CellBlock* rhs_c = (ii_c >= 0) ? &rhs_blocks[ii_c] : nullptr;

            const int fine_parent = nd.parent;
            if (fine_parent < 0) continue;
            const BlockNode& fp = tree.nodes[fine_parent];
            if (fp.first_child < 0) continue;
            const int oct = li - fp.first_child;
            const int oix = oct_ix(oct);
            const int oiy = oct_iy(oct);
            const int oiz = oct_iz(oct);

            const int half = NB / 2;
            int ta_off, tb_off;
            if (axis == 0) { ta_off = oiy * half; tb_off = oiz * half; }
            else if (axis == 1) { ta_off = oix * half; tb_off = oiz * half; }
            else               { ta_off = oix * half; tb_off = oiy * half; }

            const int c_face_i = (side == 0) ? ihi() : ilo();

            std::array<std::array<std::array<double,NVAR>, 5>, 5> coarse_acc{};
            static_assert(NB/2 <= 4, "coarse_acc sized for NB<=8");

            for (int a = ilo(); a <= ihi(); ++a)
            for (int b = ilo(); b <= ihi(); ++b) {
                int fi, fj, fk, gi, gj, gk;
                if (axis == 0) {
                    fi=face_i; fj=a; fk=b;
                    gi=ghost_i; gj=a; gk=b;
                } else if (axis == 1) {
                    fi=a; fj=face_i; fk=b;
                    gi=a; gj=ghost_i; gk=b;
                } else {
                    fi=a; fj=b; fk=face_i;
                    gi=a; gj=b; gk=ghost_i;
                }
                const int a_local = a - ilo();
                const int b_local = b - ilo();
                const int ca = a_local / 2;
                const int cb = b_local / 2;

                for (int v = 0; v < NVAR; ++v) {
                    const double jump = blk.Q[v][cell_idx(gi,gj,gk)]
                                      - blk.Q[v][cell_idx(fi,fj,fk)];
                    rhs_f.Q[v][cell_idx(fi,fj,fk)] += sigma_f * jump;
                    coarse_acc[ca][cb][v] += jump;
                }
            }

            if (rhs_c) {
                const double inv4 = 0.25;
                for (int ca = 0; ca < half; ++ca)
                for (int cb = 0; cb < half; ++cb) {
                    const int ca_c = NG + ta_off + ca;
                    const int cb_c = NG + tb_off + cb;
                    int ci, cj, ck;
                    if (axis == 0) { ci=c_face_i; cj=ca_c; ck=cb_c; }
                    else if (axis == 1) { ci=ca_c; cj=c_face_i; ck=cb_c; }
                    else               { ci=ca_c; cj=cb_c; ck=c_face_i; }
                    for (int v = 0; v < NVAR; ++v)
                        rhs_c->Q[v][cell_idx(ci,cj,ck)] -= sigma_c * inv4 * coarse_acc[ca][cb][v];
                }
            }
        }
    }
}
