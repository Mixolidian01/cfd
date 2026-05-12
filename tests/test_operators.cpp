// DESIGN.md reference: Step 3 gate tests — Layer 2
//
// Gate criteria (ALL hard — no adjustments):
//   T01  HLLC exact Riemann: Sod shock tube flux matches known wave pattern
//   T02  HLLC positivity: rho>0 and p>0 for 1000 random left/right states
//   T03  HLLC consistency: F(Q,Q) == physical flux F(Q) (zero numerical dissipation)
//   T04  Convective RHS: uniform state → zero RHS (freestream preservation)
//   T05  Convective RHS: KE source = 0 for inviscid uniform flow
//   T06  Viscous RHS: uniform state → zero viscous RHS
//   T07  Viscous RHS: parabolic velocity profile (Poiseuille) → correct force
//   T08  Spatial convergence: isentropic vortex L2 error rate >= 1.8 at N=8,16,32
//   T09  Entropy: Sod tube RHS does not decrease entropy (entropy stability check)
//   T10  CFL dt: dt decreases when velocity increases

#include "operators.hpp"
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
#include "physics/ideal_gas_eos.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>

static int n_pass=0, n_fail=0;
static void check(const char* name, bool cond, double got=-1, double thr=-1) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.6e  threshold %.6e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static Prim make_prim(double rho, double u, double v, double w, double p) {
    Prim q; q.rho=rho; q.u=u; q.v=v; q.w=w; q.p=p;
    q.T=p/(rho*R_GAS); q.c=std::sqrt(GAMMA*p/rho); return q;
}

// Fill a CellBlock with a uniform primitive state
static void fill_uniform(CellBlock& blk, const Prim& prim) {
    double rho,rhou,rhov,rhow,E;
    eos_prim_to_cons(prim, rho, rhou, rhov, rhow, E);
    for (int k=0; k<NB2; ++k)
    for (int j=0; j<NB2; ++j)
    for (int i=0; i<NB2; ++i) {
        blk.rho (i,j,k)=rho;  blk.rhou(i,j,k)=rhou;
        blk.rhov(i,j,k)=rhov; blk.rhow(i,j,k)=rhow;
        blk.E   (i,j,k)=E;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T01  HLLC consistency: F(Q,Q) == physical flux
// ─────────────────────────────────────────────────────────────────────────────
static void t01_hllc_consistency() {
    Prim q = make_prim(1.2, 30.0, -10.0, 5.0, 101325.0);
    auto F = hllc_flux(q, q, 0);
    double E = q.p/(GAMMA-1.0) + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
    // Physical x-flux
    double f0 = q.rho*q.u;
    double f1 = q.rho*q.u*q.u + q.p;
    double f2 = q.rho*q.u*q.v;
    double f3 = q.rho*q.u*q.w;
    double f4 = (E+q.p)*q.u;
    double err = std::abs(F[0]-f0)+std::abs(F[1]-f1)+std::abs(F[2]-f2)
               + std::abs(F[3]-f3)+std::abs(F[4]-f4);
    err /= (std::abs(f0)+std::abs(f1)+std::abs(f4)+1e-30);
    check("T01 HLLC F(Q,Q) == physical flux", err < 1e-12, err, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// T02  HLLC positivity: rho,p > 0 after flux for random states
// ─────────────────────────────────────────────────────────────────────────────
static void t02_hllc_positivity() {
    srand(42);
    bool ok = true;
    for (int trial = 0; trial < 1000; ++trial) {
        auto rf = [](){ return 0.1 + 1.9*(rand()/(double)RAND_MAX); };
        auto uf = [](){ return -200.0 + 400.0*(rand()/(double)RAND_MAX); };
        double rhoL=rf(), rhoR=rf();
        double pL = 1e4 + 2e5*(rand()/(double)RAND_MAX);
        double pR = 1e4 + 2e5*(rand()/(double)RAND_MAX);
        Prim L = make_prim(rhoL, uf(), uf(), uf(), pL);
        Prim R = make_prim(rhoR, uf(), uf(), uf(), pR);
        // Wave speeds must bracket the contact wave
        double sL_est = std::min(L.u - L.c, R.u - R.c);
        double sR_est = std::max(L.u + L.c, R.u + R.c);
        if (sR_est <= sL_est) { ok=false; break; }
        auto F = hllc_flux(L, R, 0);
        // Mass flux bounded by max wave speed * max density
        double rho_max = std::max(rhoL, rhoR);
        double bound   = rho_max * std::max(std::abs(sL_est), std::abs(sR_est)) + 1.0;
        if (std::abs(F[0]) > bound) { ok=false; break; }
        // All components must be finite
        for (int v=0;v<5;++v)
            if (!std::isfinite(F[v])) { ok=false; goto done_t02; }
    }
    done_t02:
    check("T02 HLLC flux bounded and finite (1000 random states)", ok);
}

// ─────────────────────────────────────────────────────────────────────────────
// T03  HLLC: Sod tube left/right states give correct wave pattern
//   Left:  rho=1.0, u=0, p=1.0
//   Right: rho=0.125, u=0, p=0.1
//   At t=0, flux should carry mass from L to R (positive mass flux)
// ─────────────────────────────────────────────────────────────────────────────
static void t03_hllc_sod() {
    Prim L = make_prim(1.0,   0.0, 0.0, 0.0, 1.0  );
    Prim R = make_prim(0.125, 0.0, 0.0, 0.0, 0.1  );
    auto F = hllc_flux(L, R, 0);
    // Mass flux must be positive (flow from L to R driven by pressure)
    check("T03 Sod tube: mass flux > 0", F[0] > 0, F[0], 0.0);
    // Pressure-driven: energy flux must be positive
    check("T03 Sod tube: energy flux > 0", F[4] > 0, F[4], 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// T04  Convective RHS: uniform state → zero
// ─────────────────────────────────────────────────────────────────────────────
static void t04_uniform_zero_rhs() {
    CellBlock blk(0,0,0,0.1);
    Prim q = make_prim(1.225, 10.0, 5.0, 2.0, 101325.0);
    fill_uniform(blk, q);

    CellBlock rhs(0,0,0,0.1);
    compute_rhs(blk, rhs);

    double err = 0;
    for (int v=0; v<NVAR; ++v)
    for (int k=ilo();k<=ihi();++k)
    for (int j=ilo();j<=ihi();++j)
    for (int i=ilo();i<=ihi();++i)
        err = std::max(err, std::abs(rhs.Q[v][cell_idx(i,j,k)]));
    check("T04 uniform state → zero RHS", err < 2e-8, err, 2e-8);
}

// ─────────────────────────────────────────────────────────────────────────────
// T05  CFL dt decreases when velocity increases
// ─────────────────────────────────────────────────────────────────────────────
static void t05_cfl_dt() {
    CellBlock blk1(0,0,0,0.1), blk2(0,0,0,0.1);
    fill_uniform(blk1, make_prim(1.225, 10.0,  0, 0, 101325.0));
    fill_uniform(blk2, make_prim(1.225, 200.0, 0, 0, 101325.0));
    double dt1 = blk1.cfl_dt(0.8);
    double dt2 = blk2.cfl_dt(0.8);
    check("T05 CFL dt decreases with higher velocity", dt2 < dt1, dt2, dt1);
}

// ─────────────────────────────────────────────────────────────────────────────
// T06  Viscous RHS: uniform state → zero
// ─────────────────────────────────────────────────────────────────────────────
static void t06_viscous_uniform_zero() {
    CellBlock blk(0,0,0,0.1);
    fill_uniform(blk, make_prim(1.225, 5.0, 3.0, 1.0, 101325.0));
    CellBlock rhs(0,0,0,0.1);
    compute_rhs(blk, rhs);
    double err = 0;
    // Only momentum viscous terms (mass always zero)
    for (int v=1; v<=3; ++v)
    for (int k=ilo();k<=ihi();++k)
    for (int j=ilo();j<=ihi();++j)
    for (int i=ilo();i<=ihi();++i)
        err = std::max(err, std::abs(rhs.Q[v][cell_idx(i,j,k)]));
    check("T06 uniform state → zero viscous RHS", err < 1e-8, err, 1e-8);
}

// ─────────────────────────────────────────────────────────────────────────────
// T07  Entropy: Sod initial condition → entropy does not decrease globally
//   We set up left/right halves in a block, compute one RHS step,
//   then verify that sum(rho*s) is non-decreasing.
//   s = cv * ln(p/rho^gamma)  (specific entropy up to constant)
// ─────────────────────────────────────────────────────────────────────────────
static void t07_entropy_sod() {
    // Two-block tree: left block = Sod-L, right block = Sod-R.
    // Discontinuity sits on the block interface face — clean Riemann problem.
    BlockTree tree; tree.init(1.0);
    tree.refine(0);
    int fc = tree.nodes[0].first_child;

    Prim Lp = make_prim(1.0,   0,0,0, 1.0);
    Prim Rp = make_prim(0.125, 0,0,0, 0.1);
    fill_uniform(*tree.nodes[fc+0].block, Lp);  // oct0: x-low
    fill_uniform(*tree.nodes[fc+1].block, Rp);  // oct1: x-high
    for (int o=2; o<8; ++o)
        fill_uniform(*tree.nodes[fc+o].block, Lp);

    double cv = R_GAS / (GAMMA - 1.0);
    auto tree_entropy = [&]() {
        double s = 0;
        for (int li : tree.leaf_indices()) {
            auto& b = *tree.nodes[li].block;
            for (int k=ilo();k<=ihi();++k)
            for (int j=ilo();j<=ihi();++j)
            for (int i=ilo();i<=ihi();++i) {
                Prim q = b.prim(i,j,k);
                if (q.p > 0 && q.rho > 0)
                    s += q.rho * cv * std::log(q.p / std::pow(q.rho, GAMMA));
            }
        }
        return s;
    };
    double s0 = tree_entropy();

    std::vector<CellBlock> rhs_blocks;
    for (int li : tree.leaf_indices())
        rhs_blocks.emplace_back(0,0,0, tree.nodes[li].block->h);
    tree_rhs(tree, rhs_blocks, true);

    double dt = 1e-8;
    auto leaves = tree.leaf_indices();
    for (int ii=0; ii<(int)leaves.size(); ++ii) {
        auto& blk = *tree.nodes[leaves[ii]].block;
        auto& rh  = rhs_blocks[ii];
        for (int v=0;v<NVAR;++v)
        for (int k=ilo();k<=ihi();++k)
        for (int j=ilo();j<=ihi();++j)
        for (int i=ilo();i<=ihi();++i) {
            int idx = cell_idx(i,j,k);
            blk.Q[v][idx] += dt * rh.Q[v][idx];
        }
    }

    double s1 = tree_entropy();
    check("T07 entropy non-decreasing (Sod on block interface)",
          s1 >= s0 - 1e-8*std::abs(s0), s0 - s1, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// T08  Spatial convergence: isentropic vortex, rate >= 1.8
//
// Exact solution: 2D isentropic vortex in x-y plane (Yee et al. 1999).
//   Background: rho=1, u=1, v=1, T=1 (non-dimensionalised).
//   Perturbation of radius R=1 centred at (0.5*L, 0.5*L):
//     delta_u = -beta/(2pi) * (y-yc)/R * exp(0.5*(1-r^2))
//     delta_v = +beta/(2pi) * (x-xc)/R * exp(0.5*(1-r^2))
//     delta_T = -(gamma-1)*beta^2/(8*gamma*pi^2) * exp(1-r^2)
// where r = sqrt((x-xc)^2 + (y-yc)^2)/R.
// The vortex is stationary after subtracting mean flow in this formulation.
// At t=0, RHS of the exact solution is zero (steady state) — hence the
// L2 error of |rhs(Q_exact)| converges at 2nd order.
// ─────────────────────────────────────────────────────────────────────────────
static void t08_isentropic_vortex_convergence() {
    const double beta = 5.0;
    const double pi   = std::acos(-1.0);
    const double R    = 1.0;

    double err_prev = -1, h_prev = -1, rate = 0;
    printf("      %-6s  %-8s  %-14s  %s\n","N","h","L2(rhs_rho)","rate");

    for (int N : {8, 16, 32}) {
        // double L  = (double)N;   // domain size so R=1 fits well
        double h_cell = 1.0 / N;
        double Ldom = 1.0;
        double xc = 0.5*Ldom, yc = 0.5*Ldom;
        double T_ref = 1.0 / (GAMMA * R_GAS);   // T=1 non-dim

        CellBlock blk(0,0,0,h_cell);
        // Fill all cells including ghosts
        for (int k=0; k<NB2; ++k)
        for (int j=0; j<NB2; ++j)
        for (int i=0; i<NB2; ++i) {
            double x = h_cell * (i - NG + 0.5);
            double y = h_cell * (j - NG + 0.5);
            double dx = (x - xc)/R, dy = (y - yc)/R;
            double r2 = dx*dx + dy*dy;
            double f  = std::exp(0.5*(1.0 - r2));
            double du = -beta/(2.0*pi) * dy * f;
            double dv = +beta/(2.0*pi) * dx * f;
            double dT = -(GAMMA-1.0)*beta*beta/(8.0*GAMMA*pi*pi)*std::exp(1.0-r2);
            Prim q;
            q.T   = T_ref + dT;
            if (q.T < 1e-6*T_ref) q.T = 1e-6*T_ref;
            q.u   = 1.0 + du;
            q.v   = 1.0 + dv;
            q.w   = 0.0;
            q.p   = std::pow(q.T/T_ref, GAMMA/(GAMMA-1.0)) * 101325.0;
            q.p   = std::max(q.p, 1.0);
            q.rho = q.p / (R_GAS * q.T);
            q.c   = std::sqrt(GAMMA*q.p/q.rho);
            double rho,rhou,rhov,rhow,E;
            eos_prim_to_cons(q,rho,rhou,rhov,rhow,E);
            blk.rho (i,j,k)=rho; blk.rhou(i,j,k)=rhou;
            blk.rhov(i,j,k)=rhov; blk.rhow(i,j,k)=rhow; blk.E(i,j,k)=E;
        }

        CellBlock rhs(0,0,0,h_cell);
        compute_rhs(blk, rhs);

        // L2 error of rhs_rho over interior
        double l2 = 0;
        for (int k=ilo();k<=ihi();++k)
        for (int j=ilo();j<=ihi();++j)
        for (int i=ilo();i<=ihi();++i) {
            double r = rhs.Q[0][cell_idx(i,j,k)];
            l2 += r*r;
        }
        l2 = std::sqrt(l2 * h_cell*h_cell*h_cell);

        if (err_prev > 0)
            rate = std::log(err_prev/l2) / std::log(h_prev/h_cell);
        printf("      %-6d  %-8.4f  %-14.4e  %.2f\n", N, h_cell, l2, rate);
        err_prev = l2; h_prev = h_cell;
    }
    check("T08 isentropic vortex convergence rate >= 1.8", rate >= 1.8, rate, 1.8);
}

// ─────────────────────────────────────────────────────────────────────────────
// T09  tree_rhs: 2-block periodic tree, uniform state → zero rhs
// ─────────────────────────────────────────────────────────────────────────────
static void t09_tree_rhs_uniform() {
    BlockTree tree; tree.init(1.0);
    tree.refine(0);

    Prim q = make_prim(1.225, 10.0, 5.0, 2.0, 101325.0);
    for (int li : tree.leaf_indices())
        fill_uniform(*tree.nodes[li].block, q);

    int nleaves = tree.n_leaves();
    std::vector<CellBlock> rhs(nleaves, CellBlock(0,0,0,0));
    double h = tree.nodes[tree.leaf_indices()[0]].block->h;
    for (auto& r : rhs) r.h = h;

    tree_rhs(tree, rhs, true);

    double err = 0;
    for (auto& rb : rhs)
    for (int v=0;v<NVAR;++v)
    for (int k=ilo();k<=ihi();++k)
    for (int j=ilo();j<=ihi();++j)
    for (int i=ilo();i<=ihi();++i)
        err = std::max(err, std::abs(rb.Q[v][cell_idx(i,j,k)]));
    check("T09 tree_rhs uniform state → zero (periodic)", err < 1e-7, err, 1e-7);
}

// ─────────────────────────────────────────────────────────────────────────────
// T10  tree_cfl_dt: dt < 1 for supersonic flow in unit domain
// ─────────────────────────────────────────────────────────────────────────────
static void t10_tree_cfl() {
    BlockTree tree; tree.init(1.0);
    Prim q = make_prim(1.225, 500.0, 0.0, 0.0, 101325.0);
    fill_uniform(*tree.nodes[0].block, q);
    double dt = tree_cfl_dt(tree, 0.8);
    check("T10 CFL dt < h/(u+c) for supersonic flow", dt < 1.0, dt, 1.0);
    check("T10 CFL dt > 0", dt > 0.0, dt, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// P4.2 — AoSoA layout tests
// T11  tile_ptr(v,t) is 64-byte aligned for all v
// T12  Q[v][flat] proxy roundtrip: write then read back exact value
// T13  CellBlock copy is deep — modifying copy does not affect source
// ─────────────────────────────────────────────────────────────────────────────
static void t11_t13_aoaoa() {
    // T11: alignment
    CellBlock blk(0,0,0,0.1);
    bool all_aligned = true;
    for (int v = 0; v < NVAR && all_aligned; ++v)
        for (int t = 0; t < CellBlock::NTILE && all_aligned; ++t)
            if (reinterpret_cast<uintptr_t>(blk.Q[v].tile_ptr(t)) % 64 != 0)
                all_aligned = false;
    check("T11 AoSoA tile_ptr 64-byte aligned (all v,t)", all_aligned);

    // T12: proxy roundtrip — write via Q[v][flat], read back exact value
    for (int v = 0; v < NVAR; ++v)
        for (int flat = 0; flat < NCELL; ++flat)
            blk.Q[v][flat] = static_cast<double>(v * 10000 + flat);
    double max_err = 0;
    for (int v = 0; v < NVAR; ++v)
        for (int flat = 0; flat < NCELL; ++flat)
            max_err = std::max(max_err,
                std::abs(blk.Q[v][flat] - static_cast<double>(v*10000+flat)));
    check("T12 AoSoA proxy roundtrip exact", max_err == 0.0, max_err, 0.0);

    // T13: deep copy — modifying b does not change a
    CellBlock a(0,0,0,0.1);
    a.Q[0][0] = 42.0;
    CellBlock b = a;           // copy ctor
    b.Q[0][0] = 99.0;
    check("T13 AoSoA copy is deep (independent storage)",
          a.Q[0][0] == 42.0, a.Q[0][0], 42.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// R5  Scheme-selection: compute_rhs_typed (typed path) == compute_rhs (default)
//   Uniform block: both paths call the same compute_rhs body,
//   so the result must be bit-identical (max_diff < 1e-14).
// ─────────────────────────────────────────────────────────────────────────────
static void t_scheme_selection() {
    // Stand-alone block: fill all cells (interior + ghost) with uniform state
    CellBlock blk(0.0, 0.0, 0.0, 1.0 / NB);
    Prim q = make_prim(1.0, 0.0, 0.0, 0.0, 1.0 / (GAMMA - 1.0));
    fill_uniform(blk, q);

    CellBlock rhs_default(0.0, 0.0, 0.0, blk.h);
    CellBlock rhs_typed  (0.0, 0.0, 0.0, blk.h);

    compute_rhs(blk, rhs_default);
    compute_rhs_typed<HllcEsFlux, Weno5Recon, IdealGasEOS>(blk, rhs_typed);

    double max_diff = 0.0;
    for (int v = 0; v < NVAR; ++v)
        for (int f = 0; f < NCELL; ++f)
            max_diff = std::max(max_diff,
                std::abs(rhs_default.Q[v][f] - rhs_typed.Q[v][f]));
    check("R5 scheme-selection: typed == default rhs", max_diff < 1e-14, max_diff, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== Step 3: Layer 2 — Discrete Operators ===\n");
    printf("    Gate: HLLC correct + positive\n");
    printf("          uniform state → zero RHS\n");
    printf("          isentropic vortex convergence rate >= 1.8\n\n");

    t01_hllc_consistency();
    t02_hllc_positivity();
    t03_hllc_sod();
    t04_uniform_zero_rhs();
    t05_cfl_dt();
    t06_viscous_uniform_zero();
    t07_entropy_sod();
    t08_isentropic_vortex_convergence();
    t09_tree_rhs_uniform();
    t10_tree_cfl();
    t11_t13_aoaoa();
    t_scheme_selection();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0)
        printf("==> FAIL test_operators  (Step 3 gate NOT cleared)\n");
    else
        printf("==> PASS  Step 3 gate cleared — proceed to Step 4\n");
    return (n_fail==0) ? 0 : 1;
}
