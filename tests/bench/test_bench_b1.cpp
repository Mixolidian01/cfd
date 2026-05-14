// B.1 benchmark — Sod shock tube (1D, Euler)
//
// Physical validation of the HLLC Riemann solver and WENO5-Z reconstruction
// against the exact Riemann solution.
//
// Reference: Sod, G.A. (1978) "A survey of several finite difference methods
//   for systems of nonlinear hyperbolic conservation laws", J. Comput. Phys.
//
// Setup:
//   Initial discontinuity at x=0.5, domain [0,1]^3.
//   Left:  ρ=1,     u=0, p=1.0  (γ=1.4, code units)
//   Right: ρ=0.125, u=0, p=0.1
//   Uniform level-2 refinement → 32 interior cells in x (32³=32768 total cells).
//   Viscosity → 0 at these temperatures (T≈0.0035 K → Sutherland μ≈10⁻¹² kg/(m·s)).
//   Run to t = 0.2.
//
// Exact solution (Toro 2009, §4):
//   p* ≈ 0.30313,  u* ≈ 0.92745
//   ρ_*L ≈ 0.42632, ρ_*R ≈ 0.26557
//   Rarefaction head  x ≈ 0.2634
//   Rarefaction foot  x ≈ 0.4861
//   Contact           x ≈ 0.6855
//   Shock             x ≈ 0.8504
//
// Gate criteria (all required):
//   B1a  L1(ρ) / ‖ρ_exact‖₁ < 8 %   (32-cell resolution, HLLC+WENO5)
//   B1b  L1(p) / ‖p_exact‖₁ < 8 %
//   B1c  mass conservation < 1e-10  (periodic domain, exact)
//   B1d  shock position within 2 cells of exact (tolerance ±2/32)
//
// CMake target: tb1

#include "solver/ns_solver.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool ok, double got = -1, double thr = -1) {
    if (ok) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.3e  thr %.3e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact Riemann solver for ideal gas (Toro 2009, §4.3 — iterative p* via NR)
// ─────────────────────────────────────────────────────────────────────────────
struct SodState { double rho, u, p; };

// f_K: wave function (rarefaction or shock side K)
static double f_wave(double p, double pk, double rhok, double ck, double gam) {
    const double g1 = (gam - 1.0) / (2.0 * gam);
    const double g5 = 2.0 / (gam + 1.0);
    const double g6 = (gam - 1.0) / (gam + 1.0);
    if (p <= pk)
        return (2.0 * ck / (gam - 1.0)) * (std::pow(p / pk, g1) - 1.0);
    else {
        double A = g5 / rhok;
        double B = g6 * pk;
        return (p - pk) * std::sqrt(A / (p + B));
    }
}

static double df_wave(double p, double pk, double rhok, double ck, double gam) {
    const double g1 = (gam - 1.0) / (2.0 * gam);
    const double g2 = (gam + 1.0) / (2.0 * gam);
    const double g5 = 2.0 / (gam + 1.0);
    const double g6 = (gam - 1.0) / (gam + 1.0);
    if (p <= pk) {
        return (1.0 / (rhok * ck)) * std::pow(p / pk, -g2);
    } else {
        double A = g5 / rhok;
        double B = g6 * pk;
        double sq = std::sqrt(A / (p + B));
        return sq * (1.0 - (p - pk) / (2.0 * (p + B)));
    }
}

// Sample exact Riemann solution at (x, t); discontinuity initially at x0=0.5
static SodState exact_sod(double x, double t, double x0 = 0.5)
{
    const double gam  = GAMMA;     // 1.4
    const double g1   = (gam - 1.0) / (2.0 * gam);    // 0.1667
    const double g4   = 2.0 / (gam - 1.0);             // 5.0
    const double g6   = (gam - 1.0) / (gam + 1.0);    // 0.1667

    const double rhoL = 1.0,   uL = 0.0, pL = 1.0;
    const double rhoR = 0.125, uR = 0.0, pR = 0.1;
    const double cL   = std::sqrt(gam * pL / rhoL);   // 1.1832
    const double cR   = std::sqrt(gam * pR / rhoR);   // 1.0583

    // Newton-Raphson for p*
    double p_star = 0.5 * (pL + pR);
    for (int it = 0; it < 200; ++it) {
        double f  = f_wave(p_star, pL, rhoL, cL, gam)
                  + f_wave(p_star, pR, rhoR, cR, gam) + (uR - uL);
        double df = df_wave(p_star, pL, rhoL, cL, gam)
                  + df_wave(p_star, pR, rhoR, cR, gam);
        double dp = -f / df;
        p_star   += dp;
        if (p_star < 1e-12) p_star = 1e-12;
        if (std::fabs(dp) < 1e-12 * p_star) break;
    }
    double u_star = 0.5 * (uL + uR)
                  + 0.5 * (f_wave(p_star, pR, rhoR, cR, gam)
                          - f_wave(p_star, pL, rhoL, cL, gam));

    // Star densities
    double rho_sL = (p_star <= pL)
        ? rhoL * std::pow(p_star / pL, 1.0 / gam)
        : rhoL * (p_star / pL + g6) / (g6 * p_star / pL + 1.0);
    double rho_sR = (p_star <= pR)
        ? rhoR * std::pow(p_star / pR, 1.0 / gam)
        : rhoR * (p_star / pR + g6) / (g6 * p_star / pR + 1.0);

    // Wave speeds
    double cSL   = cL * std::pow(p_star / pL, g1);    // rarefaction tail speed
    double S_L   = uL - cL;                            // rarefaction head
    double S_R   = uR + cR * std::sqrt((gam + 1.0) / (2.0 * gam) * p_star / pR
                                      + (gam - 1.0) / (2.0 * gam));  // shock

    double xi = (t > 1e-100) ? (x - x0) / t : ((x < x0) ? -1e30 : 1e30);

    if        (xi <= S_L) {
        return {rhoL, uL, pL};
    } else if (xi <= u_star - cSL) {
        // Inside rarefaction fan (isentropic expansion)
        double u_fan = (2.0 / (gam + 1.0)) * (cL + (gam - 1.0) / 2.0 * uL + xi);
        double c_fan = (2.0 / (gam + 1.0)) * (cL + (gam - 1.0) / 2.0 * (uL - xi));
        double rho_f = rhoL * std::pow(c_fan / cL, g4);
        double p_f   = pL   * std::pow(c_fan / cL, 2.0 * gam / (gam - 1.0));
        return {rho_f, u_fan, p_f};
    } else if (xi <= u_star) {
        return {rho_sL, u_star, p_star};
    } else if (xi <= S_R) {
        return {rho_sR, u_star, p_star};
    } else {
        return {rhoR, uR, pR};
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: fill all leaf blocks with IC; collect x-profile
// ─────────────────────────────────────────────────────────────────────────────
static void fill_leaves(NSSolver& s,
    const std::function<Prim(double,double,double)>& ic)
{
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            double x = blk.ox + (i - NG + 0.5) * blk.h;
            double y = blk.oy + (j - NG + 0.5) * blk.h;
            double z = blk.oz + (k - NG + 0.5) * blk.h;
            Prim p   = ic(x, y, z);
            int  idx = cell_idx(i, j, k);
            blk.Q[0][idx] = p.rho;
            blk.Q[1][idx] = p.rho * p.u;
            blk.Q[2][idx] = p.rho * p.v;
            blk.Q[3][idx] = p.rho * p.w;
            blk.Q[4][idx] = p.p / (GAMMA - 1.0)
                           + 0.5 * p.rho * (p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
}

struct CellSample { double x, rho, p_gas; };

static std::vector<CellSample> x_profile(const NSSolver& s)
{
    std::vector<CellSample> pts;
    const int jc = NG + NB / 2;
    const int kc = NG + NB / 2;
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        for (int i = NG; i < NG + NB; ++i) {
            double xc  = blk.ox + (i - NG + 0.5) * blk.h;
            int    idx = cell_idx(i, jc, kc);
            double rho = blk.Q[0][idx];
            double KE  = 0.5 * (blk.Q[1][idx]*blk.Q[1][idx]
                               + blk.Q[2][idx]*blk.Q[2][idx]
                               + blk.Q[3][idx]*blk.Q[3][idx]) / rho;
            double pg  = (GAMMA - 1.0) * (blk.Q[4][idx] - KE);
            pts.push_back({xc, rho, pg});
        }
    }
    std::sort(pts.begin(), pts.end(), [](const CellSample& a, const CellSample& b){
        return a.x < b.x;
    });
    return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// B1 — Sod shock tube
// ─────────────────────────────────────────────────────────────────────────────
static void b1_sod_shock_tube()
{
    printf("\n-- B1  Sod shock tube (Sod 1978) --\n");

    const double L   = 1.0;
    const double T_END = 0.2;

    auto sod_ic = [](double x, double, double) -> Prim {
        Prim q;
        if (x < 0.5) { q.rho = 1.0;   q.p = 1.0;  }
        else          { q.rho = 0.125; q.p = 0.1;  }
        q.u = 0.0; q.v = 0.0; q.w = 0.0;
        q.T = q.p / (q.rho * R_GAS);
        q.c = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    };

    NSSolver s;
    s.cfg.bc.variant = WallBC{};   // reflecting walls; waves don't reach
    s.cfg.time.cfl            = 0.4;            // boundaries by t=0.2 so no reflection
    s.cfg.amr.regrid_interval = 0;             // fixed mesh during run
    s.cfg.amr.max_level      = 0;
    s.cfg.io.verbose        = false;
    s.init(L, sod_ic);

    // Uniform level-2 refinement → 32 interior cells in each direction
    for (int lvl = 0; lvl < 2; ++lvl) {
        auto lv = s.tree.leaf_indices();
        for (int li : lv) s.tree.refine(li);
        s.tree.rebuild_neighbours();
    }
    fill_leaves(s, sod_ic);
    s.alloc_scratch();

    // Mass before run
    double mass0 = 0.0;
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        double h3 = blk.h * blk.h * blk.h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            mass0 += blk.Q[0][cell_idx(i,j,k)] * h3;
    }

    s.cfg.time.t_end    = T_END;
    s.cfg.time.max_steps = 2000;
    s.run();
    double t_final = s.t;

    // Extract x-profile at t_final
    auto prof = x_profile(s);

    // L1 errors vs exact solution
    double l1_rho = 0.0, norm_rho = 0.0;
    double l1_p   = 0.0, norm_p   = 0.0;
    double shock_x_numerical = 0.5;   // position of maximum dp/dx (shock)

    // Locate shock via pressure threshold crossing: find the RIGHTMOST x where
    // p > p_threshold, i.e. the trailing edge of the shock transition.
    // Threshold = midpoint between star-right pressure (~0.303) and undisturbed
    // right-state pressure (0.1).  At this level of detail a threshold crossing
    // is far more robust than a max-dp scan for a diffuse 1st-order HLLC shock.
    const double p_threshold = 0.5 * (0.30313 + 0.1);   // ≈ 0.2016
    shock_x_numerical = 0.5;   // fallback
    {   // Build unique-x list (first occurrence at each x)
        double lx = -1e30;
        std::vector<std::pair<double,double>> ux_p;  // (x, p)
        for (auto& pt : prof) {
            if (pt.x > lx + 1e-14) { ux_p.push_back({pt.x, pt.p_gas}); lx = pt.x; }
        }
        // Rightmost cell with p > threshold
        for (int ii = (int)ux_p.size()-1; ii >= 0; --ii) {
            if (ux_p[ii].second > p_threshold) {
                shock_x_numerical = ux_p[ii].first;
                break;
            }
        }
    }

    for (auto& pt : prof) {
        SodState ex = exact_sod(pt.x, t_final);
        double dx   = 1.0 / (double)prof.size();  // uniform cell width
        l1_rho   += std::fabs(pt.rho   - ex.rho) * dx;
        l1_p     += std::fabs(pt.p_gas - ex.p)   * dx;
        norm_rho += std::fabs(ex.rho) * dx;
        norm_p   += std::fabs(ex.p)   * dx;
    }
    double rel_rho = l1_rho / norm_rho;
    double rel_p   = l1_p   / norm_p;

    // Exact shock position at t_final
    const double cR   = std::sqrt(GAMMA * 0.1 / 0.125);
    const double pstar = 0.30313;
    double S_R_exact = 0.0 + cR * std::sqrt((GAMMA+1.0)/(2.0*GAMMA) * pstar/0.1
                                             + (GAMMA-1.0)/(2.0*GAMMA));
    double shock_x_exact = 0.5 + S_R_exact * t_final;
    double shock_err     = std::fabs(shock_x_numerical - shock_x_exact);
    // cell_width = h of finest leaf (not prof.size() which counts y-z duplicates)
    double cell_width    = s.tree.leaf_indices().empty() ? 1.0/32.0
                         : s.tree.nodes[s.tree.leaf_indices()[0]].block->h;

    // Mass after run
    double mass1 = 0.0;
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        double h3 = blk.h * blk.h * blk.h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            mass1 += blk.Q[0][cell_idx(i,j,k)] * h3;
    }
    double mass_err = std::fabs(mass1 - mass0) / std::fabs(mass0);

    printf("   t_final = %.4f\n", t_final);
    printf("   L1(ρ)/‖ρ‖₁ = %.3f%%   L1(p)/‖p‖₁ = %.3f%%\n",
           rel_rho*100.0, rel_p*100.0);
    printf("   shock_x_num = %.4f  shock_x_exact = %.4f  err = %.4f  (%.1f cells)\n",
           shock_x_numerical, shock_x_exact, shock_err, shock_err / cell_width);
    printf("   mass_err = %.2e\n", mass_err);

    check("B1a  L1(ρ) < 8%",     rel_rho < 0.08, rel_rho, 0.08);
    check("B1b  L1(p) < 8%",     rel_p   < 0.08, rel_p,   0.08);
    check("B1c  mass conservation < 1e-10", mass_err < 1e-10, mass_err, 1e-10);
    check("B1d  shock position within 2 cells of exact",
          shock_err < 2.0 * cell_width, shock_err, 2.0 * cell_width);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.1 gate: Sod shock tube ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d\n", NB, NG, NB2, NCELL);
    b1_sod_shock_tube();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.1 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
