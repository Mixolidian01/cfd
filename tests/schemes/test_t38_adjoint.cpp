// D10 gate — discrete adjoint dot-product test (A01)
//
// A01: <(rhs_frozen(Q+eps*dQ) - rhs_frozen(Q-eps*dQ))/(2eps), λ> = <dQ, L*(Q)·λ>
//      to 1e-9 relative error (single block, frozen-weight/frozen-lam TENO7+HLLC-ES).
//      rhs_frozen uses TENO7 weights and spectral radius frozen at the base state Q,
//      exactly matching the linearisation that adjoint_rhs implements.
// A02: adjoint_hllces_flux vs FD (direct Jacobian check on a single face)

#include "schemes/adjoint_rhs.hpp"
#include "physics/adjoint_teno7.hpp"   // Teno7CharFwd, teno7_recon_fwd
#include "physics/adjoint_hllc.hpp"    // adjoint_hllces_flux, physics_log_mean
#include "schemes/operators.hpp"       // hllc_es_flux_t<DIR>, cell_idx_axis
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

// convective_rhs_impl is not in any header; forward-declare it here.
// Defined in src/schemes/convective_rhs.cpp (external linkage).
void convective_rhs_impl(const Prim* pc, const double* duc,
                          CellBlock& rhs, double h,
                          uint8_t has_nbr) noexcept;

static int nfail = 0;
static void check(const char* tag, const char* msg, bool ok, double val, double tol) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else { printf("  FAIL  %s  %s  (val=%.3e  tol=%.3e)\n", tag, msg, val, tol); ++nfail; }
}

// Dot product over interior cells only
static double dot_interior(const CellBlock& a, const CellBlock& b) noexcept {
    double s = 0.0;
    for (int v = 0; v < NVAR; ++v) {
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i, j, k);
            s += a.Q[v][f] * b.Q[v][f];
        }
    }
    return s;
}

// Fill all cells (including ghosts) with a physically-valid random state
static void fill_random_all(CellBlock& blk, std::mt19937& rng, double scale) {
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const int f = cell_idx(i, j, k);
        blk.Q[0][f] = std::max(1.0 + dist(rng), 0.1);    // rho ≈ 1
        blk.Q[1][f] = 0.5 + dist(rng);                    // rho*u ≈ 0.5
        blk.Q[2][f] = dist(rng);
        blk.Q[3][f] = dist(rng);
        blk.Q[4][f] = std::max(2.5 + dist(rng), 1.5);    // E ≈ 2.5
    }
}

// Build Prim array from all cells in block
static void build_prim_array(const CellBlock& blk, Prim pc[NCELL]) noexcept {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i)
        pc[cell_idx(i,j,k)] = blk.prim(i,j,k);
}

// =============================================================================
// Frozen-weight, frozen-lam forward RHS helpers for A01
// =============================================================================
// These mirror the linearisation that adjoint_rhs implements:
//   - TENO7/TENO5 weights frozen at base state Q0
//   - Spectral radius (lam) frozen at base reconstructed states
// This makes the FD test exactly consistent with the frozen-weight adjoint.

// Apply frozen TENO7 one-sided weights to a new 7-point stencil.
static double teno7_one_sided_frozen(const Teno7ScalarFwd& fw,
    double a, double b, double c, double d, double e, double f, double g) noexcept
{
    constexpr double i12 = 1.0/12.0;
    const double s0 = i12*(-3.0*a + 13.0*b - 23.0*c + 25.0*d);
    const double s1 = i12*( 1.0*b -  5.0*c + 13.0*d +  3.0*e);
    const double s2 = i12*(-1.0*c +  7.0*d +  7.0*e -  1.0*f);
    const double s3 = i12*(25.0*d - 23.0*e + 13.0*f -  3.0*g);
    if (fw.ws > 0.0)
        return (fw.w[0]*s0 + fw.w[1]*s1 + fw.w[2]*s2 + fw.w[3]*s3) / fw.ws;
    switch (fw.eno_k) {
        case 0: return s0;
        case 1: return s1;
        case 2: return s2;
        default: return s3;
    }
}

// Apply frozen TENO5 one-sided weights to a new 5-point stencil.
static double teno5_one_sided_frozen(const Teno5ScalarFwd& fw,
    double a, double b, double c, double d, double e) noexcept
{
    constexpr double i6 = 1.0/6.0;
    const double s0 = ( 2.0*a -  7.0*b + 11.0*c) * i6;
    const double s1 = (     -b +  5.0*c +  2.0*d) * i6;
    const double s2 = ( 2.0*c +  5.0*d -      e) * i6;
    if (fw.ws > 0.0)
        return (fw.w[0]*s0 + fw.w[1]*s1 + fw.w[2]*s2) / fw.ws;
    switch (fw.eno_k) {
        case 0: return s0;
        case 1: return s1;
        default: return s2;
    }
}

// Apply frozen characteristic reconstruction (from cf) to a perturbed prim array pc_pert.
// Roe eigenvectors and TENO7/TENO5 weights are taken from cf (computed at base state).
// Stencil conserved values are taken from pc_pert.
template<Axis DIR>
static void teno7_recon_apply_frozen(
    const Prim* pc_pert, const Teno7CharFwd& cf, int i, int j, int k,
    Prim& qL_out, Prim& qR_out) noexcept
{
    auto idx_at = [&](int d) noexcept -> int {
        if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
        if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
        return                              cell_idx(i, j, k+d);
    };

    const int n_idx_c  = cf.n_idx;
    const int t1_idx_c = cf.t1_idx;
    const int t2_idx_c = cf.t2_idx;
    const double b    = cf.b,   b2   = cf.b2,   ioc  = cf.ioc;
    const double un   = cf.un,  ut1  = cf.ut1,  ut2  = cf.ut2;
    const double c_roe = cf.c_roe, H_roe = cf.H_roe, KE = cf.KE;

    // Shared back_project: char values → conserved (same formula for TENO5 and TENO7)
    auto back_project = [&](const double w[5], double Qrec[NVAR]) noexcept {
        const double w014 = w[0] + w[1] + w[4];
        const double dw04 = w[4] - w[0];
        Qrec[0]        = w014;
        Qrec[n_idx_c]  = w014*un   + dw04*c_roe;
        Qrec[t1_idx_c] = w014*ut1  + w[2];
        Qrec[t2_idx_c] = w014*ut2  + w[3];
        Qrec[4]        = (w[0]+w[4])*H_roe + dw04*un*c_roe
                       + w[1]*KE + w[2]*ut1 + w[3]*ut2;
    };

    auto safe_prim = [](const double Qc[NVAR], const Prim& fb) noexcept -> Prim {
        const double rho = Qc[0];
        if (rho <= 0.0) return fb;
        const double u_ = Qc[1]/rho, v_ = Qc[2]/rho, w_ = Qc[3]/rho;
        const double gm = fb.gamma_m, pim = fb.p_inf_m;
        const double p  = (gm-1.0)*(Qc[4]-0.5*rho*(u_*u_+v_*v_+w_*w_)) - gm*pim;
        if (p + pim <= 0.0) return fb;
        Prim q; q.rho=rho; q.u=u_; q.v=v_; q.w=w_; q.p=p;
        q.gamma_m=gm; q.p_inf_m=pim;
        q.T=(p+pim)/(rho*R_GAS); q.c=std::sqrt(gm*(p+pim)/rho);
        return q;
    };

    const Prim& fbL = pc_pert[idx_at(0)];
    const Prim& fbR = pc_pert[idx_at(1)];

    if (cf.is_teno5) {
        // 6-point stencil: m=0 → d=-2 .. m=5 → d=+3
        double Q5[6][NVAR];
        for (int m = 0; m < 6; ++m) {
            const Prim& p = pc_pert[idx_at(m - 2)];
            Q5[m][0] = p.rho;
            Q5[m][1] = p.rho * p.u;
            Q5[m][2] = p.rho * p.v;
            Q5[m][3] = p.rho * p.w;
            Q5[m][4] = (p.p + p.gamma_m*p.p_inf_m)/(p.gamma_m-1.0)
                      + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
        }
        // Char projection with frozen Roe eigenvectors
        double W5[5][6];
        for (int m = 0; m < 6; ++m) {
            const double rho = Q5[m][0];
            const double qn  = Q5[m][n_idx_c];
            const double qt1 = Q5[m][t1_idx_c];
            const double qt2 = Q5[m][t2_idx_c];
            const double E   = Q5[m][4];
            const double inn   = b2*rho - b*(un*qn+ut1*qt1+ut2*qt2) + b*E;
            const double del_n = ioc*(un*rho - qn);
            W5[0][m] = 0.5*(inn + del_n);
            W5[1][m] = (1.0 - b2)*rho + b*(un*qn+ut1*qt1+ut2*qt2) - b*E;
            W5[2][m] = -ut1*rho + qt1;
            W5[3][m] = -ut2*rho + qt2;
            W5[4][m] = 0.5*(inn - del_n);
        }
        // Apply frozen TENO5 weights
        double wL[5], wR[5];
        for (int kk = 0; kk < 5; ++kk) {
            wL[kk] = teno5_one_sided_frozen(cf.fwd5_L[kk],
                W5[kk][0], W5[kk][1], W5[kk][2], W5[kk][3], W5[kk][4]);
            // Right: reversed order (vp3,vp2,vp1,v0,vm1)
            wR[kk] = teno5_one_sided_frozen(cf.fwd5_R[kk],
                W5[kk][5], W5[kk][4], W5[kk][3], W5[kk][2], W5[kk][1]);
        }
        double QL[NVAR], QR[NVAR];
        back_project(wL, QL);
        back_project(wR, QR);
        qL_out = safe_prim(QL, fbL);
        qR_out = safe_prim(QR, fbR);
        return;
    }

    // 7-point stencil: m=0 → d=-3 .. m=6 → d=+3
    double Q[7][NVAR];
    for (int m = 0; m < 7; ++m) {
        const Prim& p = pc_pert[idx_at(m - 3)];
        Q[m][0] = p.rho;
        Q[m][1] = p.rho * p.u;
        Q[m][2] = p.rho * p.v;
        Q[m][3] = p.rho * p.w;
        Q[m][4] = (p.p + p.gamma_m*p.p_inf_m)/(p.gamma_m-1.0)
                  + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
    }
    // Char projection with frozen Roe eigenvectors
    double W[5][7];
    for (int m = 0; m < 7; ++m) {
        const double rho = Q[m][0];
        const double qn  = Q[m][n_idx_c];
        const double qt1 = Q[m][t1_idx_c];
        const double qt2 = Q[m][t2_idx_c];
        const double E   = Q[m][4];
        const double inner   = b2*rho - b*(un*qn+ut1*qt1+ut2*qt2) + b*E;
        const double delta_n = ioc*(un*rho - qn);
        W[0][m] = 0.5*(inner + delta_n);
        W[1][m] = (1.0 - b2)*rho + b*(un*qn+ut1*qt1+ut2*qt2) - b*E;
        W[2][m] = -ut1*rho + qt1;
        W[3][m] = -ut2*rho + qt2;
        W[4][m] = 0.5*(inner - delta_n);
    }
    // Apply frozen TENO7 weights
    double wL[5], wR[5];
    for (int kk = 0; kk < 5; ++kk) {
        wL[kk] = teno7_one_sided_frozen(cf.fwd_L[kk],
            W[kk][0], W[kk][1], W[kk][2], W[kk][3], W[kk][4], W[kk][5], W[kk][6]);
        // Right: reversed order (vp3,vp2,vp1,v0,vm1,vm2,vm3)
        wR[kk] = teno7_one_sided_frozen(cf.fwd_R[kk],
            W[kk][6], W[kk][5], W[kk][4], W[kk][3], W[kk][2], W[kk][1], W[kk][0]);
    }
    double QL[NVAR], QR[NVAR];
    back_project(wL, QL);
    back_project(wR, QR);
    qL_out = safe_prim(QL, fbL);
    qR_out = safe_prim(QR, fbR);
}

// Accumulate one face into rhs using frozen TENO7 weights (from pc0) and full HLLC-ES.
// The HLLC-ES flux recomputes lam from qL_pert, qR_pert — consistent with adjoint_hllces_flux
// which includes the lam sensitivity ∂lam/∂(qL,qR) in the adjoint.
template<Axis DIR>
static void accum_face_frozen(
    const Prim* pc0, const Prim* pc_pert,
    CellBlock& rhs, double ih,
    int n, int a, int b) noexcept
{
    const int Li = cell_idx_axis<DIR>(n,   a, b);
    const int Ri = cell_idx_axis<DIR>(n+1, a, b);

    const bool is_bnd = (n < ilo()) || (n+1 > ihi());

    Prim qL, qR;

    if (!is_bnd) {
        // Unpack face (n,a,b) → (xi,yi,zi) for teno7_recon_fwd
        int xi, yi, zi;
        if constexpr (DIR == Axis::X) { xi = n; yi = a; zi = b; }
        else if constexpr (DIR == Axis::Y) { xi = a; yi = n; zi = b; }
        else                               { xi = a; yi = b; zi = n; }

        // Compute frozen TENO7 weights from base state, apply to perturbed stencil
        Prim qL0, qR0;
        Teno7CharFwd cf;
        teno7_recon_fwd<DIR>(pc0, xi, yi, zi, qL0, qR0, cf);
        teno7_recon_apply_frozen<DIR>(pc_pert, cf, xi, yi, zi, qL, qR);
    } else {
        // Boundary: PCM — perturbed state directly
        qL = pc_pert[Li];
        qR = pc_pert[Ri];
    }

    // Full HLLC-ES: lam recomputed from qL,qR (consistent with adjoint lam sensitivity)
    auto F = hllc_es_flux_t<DIR>(qL, qR);

    // rhs[Li] -= ih*F,  rhs[Ri] += ih*F  (interior cells only)
    if (n >= ilo()) {
        for (int v = 0; v < NVAR; ++v)
            rhs.axis_view<DIR>(v)(n, a, b) -= ih * F[v];
    }
    if (n+1 <= ihi()) {
        for (int v = 0; v < NVAR; ++v)
            rhs.axis_view<DIR>(v)(n+1, a, b) += ih * F[v];
    }
}

// Frozen-weight, frozen-lam convective RHS.
// pc0: base state (TENO7 weights and lam frozen here).
// pc_pert: perturbed state (stencil values used for reconstruction).
// Mirrors adjoint_rhs face-loop bounds exactly.
static void convective_rhs_frozen(
    const Prim* pc0, const Prim* pc_pert,
    CellBlock& rhs, double h) noexcept
{
    const double ih = 1.0 / h;

    // X faces: n=i, a=j, b=k
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo()-1; i <= ihi(); ++i)
        accum_face_frozen<Axis::X>(pc0, pc_pert, rhs, ih, i, j, k);

    // Y faces: n=j, a=i, b=k
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo()-1; j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        accum_face_frozen<Axis::Y>(pc0, pc_pert, rhs, ih, j, i, k);

    // Z faces: n=k, a=i, b=j
    for (int k = ilo()-1; k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        accum_face_frozen<Axis::Z>(pc0, pc_pert, rhs, ih, k, i, j);
}

// =============================================================================
// A01: frozen-weight dot-product test
// =============================================================================
static void test_a01() {
    printf("\n-- A01  adjoint_rhs dot-product test (frozen-weight FD) --\n");

    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic_axes(true, true, true);
    CellBlock& blk = *tree.nodes[0].block;

    std::mt19937 rng(42);
    // Fill ALL cells (including ghosts) with random but physically valid data
    // so that ghost cells have a valid state after the periodic fill.
    fill_random_all(blk, rng, 0.3);
    tree.fill_ghosts_periodic();

    // Random dQ and lambda (interior cells only, ghost zone values = 0)
    CellBlock dQ{}, lambda{};
    dQ.h = lambda.h = blk.h;
    {
        std::uniform_real_distribution<double> d(-0.01, 0.01);
        for (int v = 0; v < NVAR; ++v)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i,j,k);
            dQ.Q[v][f] = d(rng);
            lambda.Q[v][f] = d(rng);
        }
    }

    // 4-point O(h^4) centered FD: (-f(+2h)+8f(+h)-8f(-h)+f(-2h))/(12h)
    // Eliminates O(h^2) truncation from HLLC-ES nonlinearity.
    // h=1e-4: truncation O(h^4)~1e-16, round-off ~2e-10 relative — both < 1e-9.
    const double h = 1e-4;

    // Build 4 perturbed blocks (interior cells only, ghosts unchanged).
    CellBlock blk_p1 = blk, blk_m1 = blk, blk_p2 = blk, blk_m2 = blk;
    for (int v = 0; v < NVAR; ++v)
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int f = cell_idx(i,j,k);
        blk_p1.Q[v][f] += h * dQ.Q[v][f];
        blk_m1.Q[v][f] -= h * dQ.Q[v][f];
        blk_p2.Q[v][f] += 2.0*h * dQ.Q[v][f];
        blk_m2.Q[v][f] -= 2.0*h * dQ.Q[v][f];
    }

    // Build Prim arrays for base and perturbed states.
    Prim pc0[NCELL], pc_p1[NCELL], pc_m1[NCELL], pc_p2[NCELL], pc_m2[NCELL];
    build_prim_array(blk,    pc0);
    build_prim_array(blk_p1, pc_p1);
    build_prim_array(blk_m1, pc_m1);
    build_prim_array(blk_p2, pc_p2);
    build_prim_array(blk_m2, pc_m2);

    // Frozen-weight forward RHS at all four perturbation levels.
    CellBlock rhs_p1{}, rhs_m1{}, rhs_p2{}, rhs_m2{};
    rhs_p1.h = rhs_m1.h = rhs_p2.h = rhs_m2.h = blk.h;
    convective_rhs_frozen(pc0, pc_p1, rhs_p1, blk.h);
    convective_rhs_frozen(pc0, pc_m1, rhs_m1, blk.h);
    convective_rhs_frozen(pc0, pc_p2, rhs_p2, blk.h);
    convective_rhs_frozen(pc0, pc_m2, rhs_m2, blk.h);

    // LHS: <(-f(+2h)+8f(+h)-8f(-h)+f(-2h))/(12h), lambda>
    double lhs = 0.0;
    for (int v = 0; v < NVAR; ++v)
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int f = cell_idx(i,j,k);
        const double drhs = -rhs_p2.Q[v][f] + 8.0*rhs_p1.Q[v][f]
                            - 8.0*rhs_m1.Q[v][f] + rhs_m2.Q[v][f];
        lhs += drhs / (12.0 * h) * lambda.Q[v][f];
    }

    // RHS: <dQ, L*(Q)·lambda>
    // adjoint_rhs accumulates into lambda_Q for interior cells only.
    CellBlock l_Q{};
    l_Q.h = blk.h;
    adjoint_rhs(blk, lambda, l_Q);
    const double rhs_val = dot_interior(dQ, l_Q);

    const double rel = std::abs(lhs - rhs_val)
                     / (0.5 * (std::abs(lhs) + std::abs(rhs_val)) + 1e-300);
    printf("   A01: LHS=%.10e  RHS=%.10e  rel_err=%.3e\n", lhs, rhs_val, rel);
    check("A01", "adjoint_rhs dot-product rel error < 1e-9", rel < 1e-9, rel, 1e-9);
}

// A02: direct FD check on adjoint_hllces_flux for a single face
// Builds a non-trivial L, R state and checks each prim adjoint entry
// against a finite difference of the flux.
static void test_a02() {
    printf("\n-- A02  adjoint_hllces_flux direct FD test (X-face) --\n");

    Prim L{}, R{};
    L.rho=1.0; L.u=0.5; L.v=0.1; L.w=-0.05; L.p=0.95;
    L.gamma_m=GAMMA; L.p_inf_m=0.0;
    L.T=L.p/(L.rho*R_GAS); L.c=std::sqrt(L.gamma_m*L.p/L.rho);
    R.rho=1.1; R.u=0.4; R.v=-0.1; R.w=0.02; R.p=1.0;
    R.gamma_m=GAMMA; R.p_inf_m=0.0;
    R.T=R.p/(R.rho*R_GAS); R.c=std::sqrt(R.gamma_m*R.p/R.rho);

    const double unL = L.u, unR = R.u;
    const double lam = std::max(std::abs(unL)+L.c, std::abs(unR)+R.c);

    // Random flux seed
    double l_F[NVAR] = {0.13, -0.22, 0.07, 0.18, -0.11};

    // Compute adjoint
    double l_pL[NVAR] = {}, l_pR[NVAR] = {};
    adjoint_hllces_flux<Axis::X>(L, R, l_F, lam, l_pL, l_pR);

    const double eps = 1e-7;
    const char* pnames[5] = {"rho","u","v","w","p"};
    int a02_fail = 0;

    auto perturb = [](Prim p, int v, double dv) -> Prim {
        if (v==0) p.rho+=dv; else if (v==1) p.u+=dv;
        else if (v==2) p.v+=dv; else if (v==3) p.w+=dv; else p.p+=dv;
        p.T=p.p/(p.rho*R_GAS); p.c=std::sqrt(p.gamma_m*p.p/p.rho);
        return p;
    };

    for (int v = 0; v < NVAR; ++v) {
        // FD for l_pL[v]: perturb L prim variable v
        Prim Lp = perturb(L, v, +eps);
        Prim Lm = perturb(L, v, -eps);
        auto Fp = hllc_es_flux_t<Axis::X>(Lp, R);
        auto Fm = hllc_es_flux_t<Axis::X>(Lm, R);
        double fd = 0.0;
        for (int i = 0; i < NVAR; ++i) fd += l_F[i] * (Fp[i]-Fm[i]) / (2.0*eps);
        const double rel = std::abs(l_pL[v]-fd) / (0.5*(std::abs(l_pL[v])+std::abs(fd))+1e-300);
        printf("   l_pL[%s]: adj=%.6e  fd=%.6e  rel=%.2e  %s\n",
               pnames[v], l_pL[v], fd, rel, rel<1e-6?"OK":"FAIL");
        if (rel > 1e-6) ++a02_fail;
    }
    for (int v = 0; v < NVAR; ++v) {
        // FD for l_pR[v]: perturb R prim variable v
        Prim Rp = perturb(R, v, +eps);
        Prim Rm = perturb(R, v, -eps);
        auto Fp = hllc_es_flux_t<Axis::X>(L, Rp);
        auto Fm = hllc_es_flux_t<Axis::X>(L, Rm);
        double fd = 0.0;
        for (int i = 0; i < NVAR; ++i) fd += l_F[i] * (Fp[i]-Fm[i]) / (2.0*eps);
        const double rel = std::abs(l_pR[v]-fd) / (0.5*(std::abs(l_pR[v])+std::abs(fd))+1e-300);
        printf("   l_pR[%s]: adj=%.6e  fd=%.6e  rel=%.2e  %s\n",
               pnames[v], l_pR[v], fd, rel, rel<1e-6?"OK":"FAIL");
        if (rel > 1e-6) ++a02_fail;
    }

    if (a02_fail==0) printf("  PASS  A02  adjoint_hllces_flux vs FD\n");
    else { printf("  FAIL  A02  %d entries wrong\n", a02_fail); ++nfail; }
}

int main() {
    printf("=== D10: Discrete Adjoint gate (t38) ===\n");
    test_a01();
    test_a02();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail ? 1 : 0;
}
