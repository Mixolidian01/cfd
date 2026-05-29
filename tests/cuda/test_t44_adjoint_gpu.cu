// tests/cuda/test_t44_adjoint_gpu.cu
// Gate t44: GPU adjoint convective RHS dot-product test.
// Verifies <L_froz(Q)δQ, λ> = <δQ, L*_froz(Q)λ> to 1e-8 relative error.
// Forward L_froz(Q)δQ: 4-point O(h^4) centered FD of PCM+HLLC-ES with
//   FROZEN lam (base-state spectral radius).
// Adjoint L*_froz(Q)λ: GPU k_adjoint_rhs kernel (SKIP_LAM_ADJ=true, frozen lam).
#include "cuda/gpu_adjoint_rhs.cuh"
#include "cuda/gpu_constants.cuh"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "gpu_pool.hpp"
#include "schemes/operators.hpp"    // hllc_es_flux_t<DIR>, cell_idx_axis<DIR>
#include "physics/log_mean.hpp"     // physics_log_mean
#include <cstdio>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>
#include <array>

static constexpr int ILO = NG;
static constexpr int IHI = NG + NB - 1;

// Prim from flat index (ideal gas, p_inf=0)
static Prim prim_flat(const CellBlock& blk, int f) {
    const int i = f % NB2, j = (f/NB2) % NB2, k = f / (NB2*NB2);
    return blk.prim(i, j, k);
}

// Frozen-lam HLLC-ES: use F_EC(pL_pert, pR_pert) with lam from base state.
// F = F_EC - 0.5 * lam_frozen * (Q_R_pert - Q_L_pert)
template<Axis DIR>
static std::array<double, NVAR>
hllces_frozen_lam(const Prim& L, const Prim& R, double lam_frozen) noexcept
{
    constexpr int ax = static_cast<int>(DIR);
    const double rho_a   = 0.5*(L.rho + R.rho);
    const double u_a     = 0.5*(L.u   + R.u  );
    const double v_a     = 0.5*(L.v   + R.v  );
    const double w_a     = 0.5*(L.w   + R.w  );
    const double beta_L  = L.rho / (2.0*L.p);
    const double beta_R  = R.rho / (2.0*R.p);
    const double beta_a  = 0.5*(beta_L + beta_R);
    const double rho_ln  = physics_log_mean(L.rho, R.rho);
    const double beta_ln = physics_log_mean(beta_L, beta_R);
    const double p_hat   = rho_a / (2.0*beta_a);
    const double un_L    = (ax==0)?L.u:(ax==1)?L.v:L.w;
    const double un_R    = (ax==0)?R.u:(ax==1)?R.v:R.w;
    const double un_a    = 0.5*(un_L + un_R);
    const double mass    = rho_ln * un_a;
    const double gm_face = 0.5*(L.gamma_m + R.gamma_m);
    const double KE_hat  = 0.5*(u_a*u_a + v_a*v_a + w_a*w_a);
    const double H_hat   = 1.0/(2.0*(gm_face-1.0)*beta_ln) + KE_hat + p_hat/rho_ln;

    std::array<double,NVAR> F;
    F[0] = mass;
    F[1] = mass*u_a + (ax==0 ? p_hat : 0.0);
    F[2] = mass*v_a + (ax==1 ? p_hat : 0.0);
    F[3] = mass*w_a + (ax==2 ? p_hat : 0.0);
    F[4] = mass*H_hat;

    // Frozen dissipation
    const double gm1 = L.gamma_m - 1.0;
    const double E_L = L.p/gm1 + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R = R.p/gm1 + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    F[0] -= 0.5*lam_frozen*(R.rho         - L.rho    );
    F[1] -= 0.5*lam_frozen*(R.rho*R.u     - L.rho*L.u);
    F[2] -= 0.5*lam_frozen*(R.rho*R.v     - L.rho*L.v);
    F[3] -= 0.5*lam_frozen*(R.rho*R.w     - L.rho*L.w);
    F[4] -= 0.5*lam_frozen*(E_R           - E_L       );
    return F;
}

// Frozen-lam PCM RHS: lam from blk_base, EC flux from blk_pert.
template<Axis DIR>
static void pcm_rhs_ax(const CellBlock& blk_base, const CellBlock& blk_pert,
                        CellBlock& rhs, double ih) {
    for (int b2 = ILO; b2 <= IHI; ++b2)
    for (int b1 = ILO; b1 <= IHI; ++b1)
    for (int n = ILO-1; n <= IHI; ++n) {
        const int Li = cell_idx_axis<DIR>(n,   b1, b2);
        const int Ri = cell_idx_axis<DIR>(n+1, b1, b2);
        const bool liI = (n   >= ILO);
        const bool riI = (n+1 <= IHI);
        const Prim bL = prim_flat(blk_base, Li);
        const Prim bR = prim_flat(blk_base, Ri);
        constexpr int ax = static_cast<int>(DIR);
        const double unL_b = (ax==0)?bL.u:(ax==1)?bL.v:bL.w;
        const double unR_b = (ax==0)?bR.u:(ax==1)?bR.v:bR.w;
        const double lam   = std::max(std::abs(unL_b)+bL.c, std::abs(unR_b)+bR.c);

        const Prim pL = prim_flat(blk_pert, Li);
        const Prim pR = prim_flat(blk_pert, Ri);
        const auto F  = hllces_frozen_lam<DIR>(pL, pR, lam);
        if (liI) for (int v=0;v<NVAR;++v) rhs.Q[v][Li] -= ih * F[v];
        if (riI) for (int v=0;v<NVAR;++v) rhs.Q[v][Ri] += ih * F[v];
    }
}

static void pcm_rhs(const CellBlock& blk_base, const CellBlock& blk_pert,
                    CellBlock& rhs) {
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f) rhs.Q[v][f]=0.0;
    pcm_rhs_ax<Axis::X>(blk_base, blk_pert, rhs, 1.0/blk_pert.h);
    pcm_rhs_ax<Axis::Y>(blk_base, blk_pert, rhs, 1.0/blk_pert.hy);
    pcm_rhs_ax<Axis::Z>(blk_base, blk_pert, rhs, 1.0/blk_pert.hz);
}

// Interior dot product
static double dot_int(const CellBlock& a, const CellBlock& b) {
    double s = 0.0;
    for (int v=0;v<NVAR;++v)
    for (int k=ILO;k<=IHI;++k) for (int j=ILO;j<=IHI;++j) for (int i=ILO;i<=IHI;++i)
        s += a.Q[v][cell_idx(i,j,k)] * b.Q[v][cell_idx(i,j,k)];
    return s;
}

int main() {
    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic_axes(true, true, true);
    CellBlock& blk = *tree.nodes[0].block;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.3, 0.3);

    for (int k=0;k<NB2;++k) for (int j=0;j<NB2;++j) for (int i=0;i<NB2;++i) {
        const int f = cell_idx(i,j,k);
        blk.Q[0][f] = std::max(1.0 + dist(rng), 0.1);
        blk.Q[1][f] = 0.5  + dist(rng);
        blk.Q[2][f] = 0.3  + dist(rng);
        blk.Q[3][f] = 0.2  + dist(rng);
        blk.Q[4][f] = std::max(2.5 + dist(rng), 1.5);
    }
    tree.fill_ghosts_periodic();

    // GPU pool and upload
    GpuPool pool;
    pool.alloc(&blk);
    pool.upload(&blk);

    // Build adjoint list
    GpuAdjointRhsList adj;
    adj.build(tree, pool);

    // Generate δQ and λ (interior cells only)
    CellBlock dQ_blk{}, lam_blk{};
    dQ_blk.h = lam_blk.h = blk.h;
    dQ_blk.hy = lam_blk.hy = blk.hy;
    dQ_blk.hz = lam_blk.hz = blk.hz;
    {
        std::uniform_real_distribution<double> rd(-0.01, 0.01);
        for (int v=0;v<NVAR;++v)
        for (int k=ILO;k<=IHI;++k) for (int j=ILO;j<=IHI;++j) for (int i=ILO;i<=IHI;++i) {
            const int f = cell_idx(i,j,k);
            dQ_blk.Q[v][f] = rd(rng);
            lam_blk.Q[v][f] = rd(rng);
        }
    }

    // 4-point O(h^4) centered FD for L_froz(Q)δQ (frozen-lam forward)
    const double eps = 1e-4;
    CellBlock blk_p1 = blk, blk_m1 = blk, blk_p2 = blk, blk_m2 = blk;
    blk_p1.h = blk_m1.h = blk_p2.h = blk_m2.h = blk.h;
    blk_p1.hy = blk_m1.hy = blk_p2.hy = blk_m2.hy = blk.hy;
    blk_p1.hz = blk_m1.hz = blk_p2.hz = blk_m2.hz = blk.hz;
    for (int v=0;v<NVAR;++v)
    for (int k=ILO;k<=IHI;++k) for (int j=ILO;j<=IHI;++j) for (int i=ILO;i<=IHI;++i) {
        const int f = cell_idx(i,j,k);
        blk_p1.Q[v][f] += eps * dQ_blk.Q[v][f];
        blk_m1.Q[v][f] -= eps * dQ_blk.Q[v][f];
        blk_p2.Q[v][f] += 2.0*eps * dQ_blk.Q[v][f];
        blk_m2.Q[v][f] -= 2.0*eps * dQ_blk.Q[v][f];
    }

    CellBlock rhs_p1{}, rhs_m1{}, rhs_p2{}, rhs_m2{};
    pcm_rhs(blk, blk_p1, rhs_p1);
    pcm_rhs(blk, blk_m1, rhs_m1);
    pcm_rhs(blk, blk_p2, rhs_p2);
    pcm_rhs(blk, blk_m2, rhs_m2);

    // L_froz(Q)δQ = (-rhs_p2 + 8*rhs_p1 - 8*rhs_m1 + rhs_m2) / (12*eps)
    CellBlock LdQ{};
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        LdQ.Q[v][f] = (-rhs_p2.Q[v][f] + 8.0*rhs_p1.Q[v][f]
                       - 8.0*rhs_m1.Q[v][f] + rhs_m2.Q[v][f]) / (12.0*eps);

    // GPU adjoint: L*_froz(Q)λ
    std::vector<double> h_lam(GPU_NVAR * GPU_NCELL, 0.0);
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        h_lam[v*NCELL + f] = lam_blk.Q[v][f];

    std::vector<double> h_adj_out;
    adj.exec_sync(h_lam, h_adj_out);

    CellBlock adj_blk{};
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        adj_blk.Q[v][f] = h_adj_out[v*NCELL + f];

    const double ip_fwd = dot_int(LdQ,     lam_blk);  // <L_froz(Q)δQ, λ>
    const double ip_adj = dot_int(dQ_blk,  adj_blk);  // <δQ, L*_froz(Q)λ>

    const double err = std::fabs(ip_fwd - ip_adj)
                     / (std::fabs(ip_fwd) + std::fabs(ip_adj) + 1e-30);
    std::printf("<LdQ,lam>=%.6e  <dQ,L*lam>=%.6e  rel_err=%.3e\n",
                ip_fwd, ip_adj, err);
    std::fflush(stdout);
    assert(err < 1e-8 && "GPU adjoint dot-product test failed");
    std::puts("PASS");

    pool.free(&blk);
}
