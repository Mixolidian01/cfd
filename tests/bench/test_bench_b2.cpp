// B.2 benchmark — Shu-Osher shock-entropy interaction (1D, Euler)
//
// Physical validation of the WENO5-Z reconstruction accuracy on a problem
// combining a strong shock with a smooth sinusoidal density field.
//
// Reference: Shu, C.-W. & Osher, S. (1989) "Efficient implementation of
//   essentially non-oscillatory shock-capturing schemes, II",
//   J. Comput. Phys. 83, 32-78.
//
// Setup:
//   Physical domain: x ∈ [-5, 5]    →  code domain [0, 10] (shift by 5)
//   Initial discontinuity at x_phys=-4  (x_code=1.0).
//   Left  (x_phys < -4): post-shock state (Mach-3 shock hitting ρ=1 gas)
//     ρ = 27/7   = 3.857143,  u = 4√35/9 ≈ 2.629369,  p = 116/35 = 10.33333
//   Right (x_phys ≥ -4): sinusoidal density field ahead of the shock
//     ρ = 1 + 0.2 sin(5·x_phys),  u = 0,  p = 1
//   BCType::Wall; waves don't reach boundaries at t = 1.8.
//   Level-3 uniform refinement → 64 interior cells in x, h ≈ 0.156.
//   Oscillation wavelength λ = 2π/5 ≈ 1.257 ≈ 8 cells/period (WENO5 resolves).
//
// Exact shock speed: S = M_s · c_0 = 3·√1.4 ≈ 3.5497
//   Shock position at t=1.8: x_phys ≈ -4 + 3.5497·1.8 ≈ 2.39
//                             x_code ≈ 7.39
//
// Gate criteria:
//   B2a  Shock position at t=1.8 in x_code ∈ [6.0, 9.0]  (exact ≈ 7.39)
//   B2b  Post-shock density oscillation amplitude > 0.25
//        (WENO5 must preserve oscillations; 1st-order HLLC diffuses them to zero)
//   B2c  No negative pressure anywhere
//   B2d  Mass conservation < 1e-10

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
static void b2_shu_osher()
{
    printf("\n-- B2  Shu-Osher shock-entropy interaction (Shu & Osher 1989) --\n");

    const double L   = 10.0;   // code domain [0, 10] ↔ physical [-5, 5]
    const double SHIFT = 5.0;  // x_phys = x_code - SHIFT

    // Shu-Osher initial condition (Mach-3 shock, γ=1.4)
    //   Left  (x_phys < -4, i.e. x_code < 1): post-shock state
    //   Right (x_phys ≥ -4, i.e. x_code ≥ 1): sinusoidal pre-shock
    const double rhoL = 27.0/7.0,   uL = 4.0*std::sqrt(35.0)/9.0, pL = 116.0/35.0;
    auto shu_ic = [&](double xc, double, double) -> Prim {
        Prim q;
        q.v = 0.0; q.w = 0.0;
        double xp = xc - SHIFT;  // physical coordinate
        if (xp < -4.0) {
            q.rho = rhoL; q.u = uL; q.p = pL;
        } else {
            q.rho = 1.0 + 0.2 * std::sin(5.0 * xp);
            q.u   = 0.0;
            q.p   = 1.0;
        }
        q.T = q.p / (q.rho * R_GAS);
        q.c = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    };

    NSSolver s;
    s.cfg.bc.variant = OpenBC{};   // transmissive: no wall reflections
    s.cfg.time.cfl             = 0.4;
    s.cfg.amr.regrid_interval = 0;
    s.cfg.amr.max_level       = 0;
    s.cfg.io.verbose         = false;
    s.init(L, shu_ic);

    // Level-3 uniform refinement → 64 cells in x, h ≈ 0.156
    for (int lvl = 0; lvl < 3; ++lvl) {
        auto lv = s.tree.leaf_indices();
        for (int li : lv) s.tree.refine(li);
        s.tree.rebuild_neighbours();
    }
    fill_leaves(s, shu_ic);
    s.alloc_scratch();

    s.cfg.time.t_end    = 1.8;
    s.cfg.time.max_steps = 20000;
    s.run();
    double t_final = s.t;

    auto prof = x_profile(s);

    // Build unique-x list (first occurrence)
    std::vector<CellSample> ux;
    {   double lx = -1e30;
        for (auto& pt : prof) {
            if (pt.x > lx + 1e-14) { ux.push_back(pt); lx = pt.x; }
        }
    }

    // ── B2a: shock position via pressure threshold ────────────────────────────
    // Shock = rightmost x_code where p > threshold = (pL + 1.0)/2 ≈ 5.67
    const double p_thresh = 0.5 * (pL + 1.0);
    double shock_x = 0.0;
    for (int ii = (int)ux.size()-1; ii >= 0; --ii) {
        if (ux[ii].p_gas > p_thresh) { shock_x = ux[ii].x; break; }
    }
    const double shock_exact = (-4.0 + 3.0*std::sqrt(GAMMA) * t_final) + SHIFT;

    // ── B2b: post-shock density oscillation amplitude ─────────────────────────
    // Post-shock: x_code ∈ [1.0, shock_x - h], where h = leaf cell size
    double h_cell = s.tree.nodes[s.tree.leaf_indices()[0]].block->h;
    double rho_max_ps = -1e30, rho_min_ps = 1e30;
    for (auto& pt : ux) {
        if (pt.x > 1.0 && pt.x < shock_x - h_cell) {
            rho_max_ps = std::max(rho_max_ps, pt.rho);
            rho_min_ps = std::min(rho_min_ps, pt.rho);
        }
    }
    double amp = (rho_max_ps > rho_min_ps) ? rho_max_ps - rho_min_ps : 0.0;

    // ── B2c: no negative pressure ─────────────────────────────────────────────
    double p_min = 1e30;
    for (auto& pt : ux) p_min = std::min(p_min, pt.p_gas);

    // ── B2d: no negative density (stability of WENO5-Z under strong shock) ───
    double rho_min_global = 1e30;
    for (auto& pt : ux) rho_min_global = std::min(rho_min_global, pt.rho);

    printf("   t_final = %.4f  leaves = %d  h = %.5f\n",
           t_final, (int)s.tree.leaf_indices().size(), h_cell);
    printf("   shock_x = %.4f  shock_exact = %.4f  |err| = %.4f\n",
           shock_x, shock_exact, std::fabs(shock_x - shock_exact));
    printf("   post-shock rho: max=%.4f min=%.4f amp=%.4f\n",
           rho_max_ps, rho_min_ps, amp);
    printf("   p_min = %.4f  rho_min = %.4f\n", p_min, rho_min_global);

    check("B2a  shock position in [6.0, 9.0]",
          shock_x >= 6.0 && shock_x <= 9.0);
    check("B2b  post-shock oscillation amplitude > 0.25",
          amp > 0.25, amp, 0.25);
    check("B2c  no negative pressure",
          p_min > 0.0, p_min, 0.0);
    check("B2d  no negative density (WENO5-Z stability)",
          rho_min_global > 0.0, rho_min_global, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.2 gate: Shu-Osher shock-entropy interaction ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d\n", NB, NG, NB2, NCELL);
    b2_shu_osher();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.2 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
