// Gate t42: dot-product test for adjoint_rk3_one_block.
// Verifies <dQf, lambda_f> ≈ <dQn, lambda_Qn> via 4-point FD through
// frozen-weight RK3 forward (same linearisation as adjoint_rhs uses internally).
#include "solver/adjoint_rk3.hpp"
#include "schemes/adjoint_rhs.hpp"
#include "physics/adjoint_teno7.hpp"
#include "physics/adjoint_hllc.hpp"
#include "schemes/operators.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>

static int nfail = 0;
static void check(const char* tag, const char* msg, bool ok, double val, double tol) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else  { printf("  FAIL  %s  %s  (val=%.3e tol=%.3e)\n", tag, msg, val, tol); ++nfail; }
}

template<Axis DIR>
static void accum_face_frozen(const Prim* pc0, const Prim* pc_pert,
    CellBlock& rhs, double ih, int n, int a, int b) noexcept
{
    const int Li = cell_idx_axis<DIR>(n,   a, b);
    const int Ri = cell_idx_axis<DIR>(n+1, a, b);
    const bool is_bnd = (n < ilo()) || (n+1 > ihi());
    Prim qL, qR;
    if (!is_bnd) {
        int xi, yi, zi;
        if constexpr (DIR == Axis::X) { xi=n; yi=a; zi=b; }
        else if constexpr (DIR == Axis::Y) { xi=a; yi=n; zi=b; }
        else                               { xi=a; yi=b; zi=n; }
        Prim qL0, qR0; Teno7CharFwd cf;
        teno7_recon_fwd<DIR>(pc0, xi, yi, zi, qL0, qR0, cf);
        teno7_recon_apply_frozen<DIR>(pc_pert, cf, xi, yi, zi, qL, qR);
    } else {
        qL = pc_pert[Li]; qR = pc_pert[Ri];
    }
    auto F = hllc_es_flux_t<DIR>(qL, qR);
    if (n >= ilo())   for (int v=0;v<NVAR;++v) rhs.axis_view<DIR>(v)(n,  a,b) -= ih*F[v];
    if (n+1 <= ihi()) for (int v=0;v<NVAR;++v) rhs.axis_view<DIR>(v)(n+1,a,b) += ih*F[v];
}

static void frozen_rhs(const Prim* pc0, const Prim* pc_pert, CellBlock& rhs, double h) noexcept {
    const double ih = 1.0/h;
    for (int k=ilo();k<=ihi();++k)  { for (int j=ilo();j<=ihi();++j)  for (int i=ilo()-1;i<=ihi();++i) accum_face_frozen<Axis::X>(pc0,pc_pert,rhs,ih,i,j,k); }
    for (int k=ilo();k<=ihi();++k)  { for (int j=ilo()-1;j<=ihi();++j) for (int i=ilo();i<=ihi();++i)  accum_face_frozen<Axis::Y>(pc0,pc_pert,rhs,ih,j,i,k); }
    for (int k=ilo()-1;k<=ihi();++k){ for (int j=ilo();j<=ihi();++j)  for (int i=ilo();i<=ihi();++i)   accum_face_frozen<Axis::Z>(pc0,pc_pert,rhs,ih,k,i,j); }
}

static void build_prim(const CellBlock& blk, Prim pc[NCELL]) noexcept {
    for (int k=0;k<NB2;++k) for (int j=0;j<NB2;++j) for (int i=0;i<NB2;++i)
        pc[cell_idx(i,j,k)] = blk.prim(i,j,k);
}

static void frozen_rk3_step(
    const CellBlock& Qs0, const CellBlock& Qs1, const CellBlock& Qs2,
    const CellBlock& Qn_pert, double dt, CellBlock& Qf_pert) noexcept
{
    const double h = Qs0.h;
    Prim pc0[NCELL], pc1[NCELL], pc2[NCELL];
    build_prim(Qs0, pc0); build_prim(Qs1, pc1); build_prim(Qs2, pc2);
    Prim pcn[NCELL]; build_prim(Qn_pert, pcn);

    CellBlock Q1{}; Q1.h = h;
    CellBlock rhs1{}; rhs1.h = h;
    frozen_rhs(pc0, pcn, rhs1, h);
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        Q1.Q[v][f] = Qn_pert.Q[v][f] + dt*rhs1.Q[v][f];

    Prim pc1p[NCELL]; build_prim(Q1, pc1p);
    CellBlock rhs2{}; rhs2.h = h;
    frozen_rhs(pc1, pc1p, rhs2, h);
    CellBlock Q2{}; Q2.h = h;
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        Q2.Q[v][f] = 0.75*Qn_pert.Q[v][f] + 0.25*(Q1.Q[v][f] + dt*rhs2.Q[v][f]);

    Prim pc2p[NCELL]; build_prim(Q2, pc2p);
    CellBlock rhs3{}; rhs3.h = h;
    frozen_rhs(pc2, pc2p, rhs3, h);
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        Qf_pert.Q[v][f] = (1.0/3.0)*Qn_pert.Q[v][f] + (2.0/3.0)*(Q2.Q[v][f] + dt*rhs3.Q[v][f]);
}

static double dot_interior(const CellBlock& a, const CellBlock& b) noexcept {
    double s = 0.0;
    for (int v=0;v<NVAR;++v) for (int k=ilo();k<=ihi();++k) for (int j=ilo();j<=ihi();++j) for (int i=ilo();i<=ihi();++i)
        s += a.Q[v][cell_idx(i,j,k)] * b.Q[v][cell_idx(i,j,k)];
    return s;
}

static void fill_rand(CellBlock& blk, std::mt19937& rng, double scale) {
    std::uniform_real_distribution<double> d(-scale, scale);
    for (int k=0;k<NB2;++k) for (int j=0;j<NB2;++j) for (int i=0;i<NB2;++i) {
        const int f = cell_idx(i,j,k);
        blk.Q[0][f] = std::max(1.0+d(rng), 0.1);
        blk.Q[1][f] = 0.3+d(rng); blk.Q[2][f] = d(rng);
        blk.Q[3][f] = d(rng);
        blk.Q[4][f] = std::max(2.0+d(rng), 1.0);
    }
}

int main() {
    printf("=== D11-adj: adjoint_rk3_one_block dot-product test (t42) ===\n\n");

    BlockTree tree; tree.init(1.0); tree.set_periodic_axes(true,true,true);
    const double h  = tree.nodes[0].block->h;
    const double dt = 1e-4;

    std::mt19937 rng(137);
    CellBlock Qn{}; Qn.h = h;
    fill_rand(Qn, rng, 0.3);
    tree.nodes[0].block->operator=(Qn);
    tree.fill_ghosts_periodic();
    CellBlock Qs0 = *tree.nodes[0].block;  // Qn + valid ghosts

    // Build Q1 via frozen forward
    Prim pc0[NCELL]; build_prim(Qs0, pc0);
    CellBlock rhs1{}; rhs1.h = h;
    frozen_rhs(pc0, pc0, rhs1, h);
    CellBlock Qs1{}; Qs1.h = h;
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        Qs1.Q[v][f] = Qs0.Q[v][f] + dt*rhs1.Q[v][f];

    // Build Q2 via frozen forward
    Prim pc1[NCELL]; build_prim(Qs1, pc1);
    CellBlock rhs2{}; rhs2.h = h;
    frozen_rhs(pc1, pc1, rhs2, h);
    CellBlock Qs2{}; Qs2.h = h;
    for (int v=0;v<NVAR;++v) for (int f=0;f<NCELL;++f)
        Qs2.Q[v][f] = 0.75*Qs0.Q[v][f] + 0.25*(Qs1.Q[v][f] + dt*rhs2.Q[v][f]);

    // Random dQn and lambda_f (interior only)
    CellBlock dQn{}; dQn.h = h;
    CellBlock lam_f{}; lam_f.h = h;
    {
        std::uniform_real_distribution<double> d(-0.01, 0.01);
        for (int v=0;v<NVAR;++v) for (int k=ilo();k<=ihi();++k) for (int j=ilo();j<=ihi();++j) for (int i=ilo();i<=ihi();++i) {
            const int f = cell_idx(i,j,k);
            dQn.Q[v][f] = d(rng); lam_f.Q[v][f] = d(rng);
        }
    }

    // 4-point FD
    const double eps = 1e-4;
    CellBlock Qn_p1=Qs0, Qn_m1=Qs0, Qn_p2=Qs0, Qn_m2=Qs0;
    for (int v=0;v<NVAR;++v) for (int k=ilo();k<=ihi();++k) for (int j=ilo();j<=ihi();++j) for (int i=ilo();i<=ihi();++i) {
        const int f = cell_idx(i,j,k);
        Qn_p1.Q[v][f]+=eps*dQn.Q[v][f]; Qn_m1.Q[v][f]-=eps*dQn.Q[v][f];
        Qn_p2.Q[v][f]+=2*eps*dQn.Q[v][f]; Qn_m2.Q[v][f]-=2*eps*dQn.Q[v][f];
    }
    CellBlock Qf_p1{}; Qf_p1.h=h; CellBlock Qf_m1{}; Qf_m1.h=h;
    CellBlock Qf_p2{}; Qf_p2.h=h; CellBlock Qf_m2{}; Qf_m2.h=h;
    frozen_rk3_step(Qs0,Qs1,Qs2,Qn_p1,dt,Qf_p1);
    frozen_rk3_step(Qs0,Qs1,Qs2,Qn_m1,dt,Qf_m1);
    frozen_rk3_step(Qs0,Qs1,Qs2,Qn_p2,dt,Qf_p2);
    frozen_rk3_step(Qs0,Qs1,Qs2,Qn_m2,dt,Qf_m2);

    double lhs = 0.0;
    for (int v=0;v<NVAR;++v) for (int k=ilo();k<=ihi();++k) for (int j=ilo();j<=ihi();++j) for (int i=ilo();i<=ihi();++i) {
        const int f = cell_idx(i,j,k);
        const double dQf = (-Qf_p2.Q[v][f]+8*Qf_p1.Q[v][f]-8*Qf_m1.Q[v][f]+Qf_m2.Q[v][f])/(12*eps);
        lhs += dQf * lam_f.Q[v][f];
    }

    CellBlock lam_Qn{}; lam_Qn.h = h;
    adjoint_rk3_one_block(Qs0, Qs1, Qs2, dt, lam_f, lam_Qn);
    const double rhs_val = dot_interior(dQn, lam_Qn);

    const double rel = std::abs(lhs-rhs_val)/(0.5*(std::abs(lhs)+std::abs(rhs_val))+1e-300);
    printf("   A01: LHS=%.10e  RHS=%.10e  rel_err=%.3e\n", lhs, rhs_val, rel);
    check("A01", "adjoint_rk3 dot-product rel error < 1e-7", rel < 1e-7, rel, 1e-7);

    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail ? 1 : 0;
}
