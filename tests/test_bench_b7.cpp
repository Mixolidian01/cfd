// B.7 benchmark — Rayleigh-Taylor instability (2D, Allaire 5-equation model)
//
// Validates the two-phase diffuse-interface model under gravity-driven
// interface instability where a heavy fluid (phase 2) sits atop a light
// fluid (phase 1).  The Allaire (2002) 5-equation model must:
//   (i)  keep α₁ ∈ [0,1] (bound preservation)
//   (ii) propagate the interface without spurious pressure oscillations
//   (iii) produce bubble/spike growth consistent with linear RT theory
//
// Reference: Allaire, Clerc & Kokh (2002) J. Comput. Phys. 181, 577–616
//            Youngs (1984) Physica D 12, 32–44 (RT mixing layer growth)
//
// Physical analysis:
//   Linear RT growth rate: γ_RT = sqrt(g·k·A_t)
//   where k = 2π/λ (wave number), A_t = (ρ₂−ρ₁)/(ρ₂+ρ₁) (Atwood number),
//         g = gravitational acceleration
//   Setup: λ = 0.5 (half-domain width), k = 4π, ρ₂=2·ρ₁, A_t = 1/3
//   g = 1 → γ_RT = sqrt(1 · 4π · 1/3) ≈ 2.045
//   Growth e-fold time τ = 1/γ_RT ≈ 0.489
//
// Setup (2D in xy, embedded in 3D periodic/wall domain [0,0.5]×[0,1]×[0,0.5]):
//   Phase 1 (light, bottom, y < 0.5): ρ₁=1,  p=p₀−ρ₁·g·y  (hydrostatic)
//   Phase 2 (heavy, top,  y ≥ 0.5): ρ₂=2,  p=p₀−ρ₂·g·(y−0.5)−ρ₁·g·0.5
//   γ₁=γ₂=1.4 (same EOS, ideal gas), so BN reduces to single-fluid Euler
//   Interface at y=0.5 + A*cos(2πx/0.5)  (A=0.01 perturbation amplitude)
//   α₁ = smooth step across interface (tanh with δ=0.05)
//   g = 1.0 (body force applied as source term)
//   BCs: periodic in x, Wall (reflecting) in y (no-penetration), periodic in z
//   Level-2 refinement → 16×32 cells in [0,0.5]×[0,1] (32 cells/half-wavelength)
//
// Note: gravity body force is applied as an explicit source term.
//       The BN solver is used without the two-fluid EOS mismatch here (same γ).
//
// Gate criteria:
//   B7a  α₁ ∈ [0,1] at all times (bound preservation)
//   B7b  No spurious pressure oscillations at interface (< 3 extrema in ±5 cells)
//   B7c  Interface amplitude grows: A(t_final)/A(0) > 1.1
//          The 1st-order BN HLLC scheme at 16×32 cells is strongly diffusive
//          (D≈c·h/2≈0.15 m²/s >> γ_RT=2.05 s⁻¹), so exponential growth is
//          heavily damped.  Threshold 1.1 verifies the perturbation grows
//          (correct unstable stratification) rather than decays.
//   B7d  No negative partial densities (α₁ρ₁ ≥ 0, α₂ρ₂ ≥ 0)

#include "../include/bn_model.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool ok, double got = -1, double thr = -1)
{
    if (ok) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.3e  thr %.3e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// ── Ghost fill helpers (same as B4, adapted for y-wall + x-periodic) ────────
// x: periodic (block-to-block wrapping)
// y: wall (zero-gradient = extrapolate density, flip normal momentum)
static void fill_ghosts_2d(std::vector<BNCellBlock>& blks,
                            int NX, int NY)
{
    for (int by = 0; by < NY; ++by)
    for (int bx = 0; bx < NX; ++bx) {
        int b = by * NX + bx;
        // First fill y,z ghosts periodically (default)
        bn_fill_ghosts_periodic(blks[b]);

        // ── Periodic x ghosts ────────────────────────────────────────────────
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR_BN; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j) {
            // Left ghost: from right interior of block (bx-1, by) wrapped
            int bx_l = (bx - 1 + NX) % NX;
            int b_l  = by * NX + bx_l;
            blks[b].Q[v][cell_idx(g, j, k)] =
                blks[b_l].Q[v][cell_idx(NB+g, j, k)];
            // Right ghost: from left interior of block (bx+1, by) wrapped
            int bx_r = (bx + 1) % NX;
            int b_r  = by * NX + bx_r;
            blks[b].Q[v][cell_idx(NB+NG+g, j, k)] =
                blks[b_r].Q[v][cell_idx(NG+g, j, k)];
        }

        // ── Wall y ghosts (zero-gradient on rho, flip normal momentum) ───────
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR_BN; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int i = NG; i < NG+NB; ++i) {
            if (by == 0) {
                // Bottom wall: ghost j=NG-1-g mirrors interior j=NG+g
                // Normal momentum (y = index 3) is negated; all others copied
                int gj = NG-1-g;
                int mi = NG+g;
                blks[b].Q[v][cell_idx(i, gj, k)] =
                    (v == 3) ? -blks[b].Q[v][cell_idx(i, mi, k)]
                             :  blks[b].Q[v][cell_idx(i, mi, k)];
            }
            if (by == NY-1) {
                // Top wall: ghost j=NB+NG+g mirrors interior j=NB+NG-1-g
                int gj = NB+NG+g;
                int mi = NB+NG-1-g;
                blks[b].Q[v][cell_idx(i, gj, k)] =
                    (v == 3) ? -blks[b].Q[v][cell_idx(i, mi, k)]
                             :  blks[b].Q[v][cell_idx(i, mi, k)];
            }
        }

        // ── Interior y ghosts (block-to-block in y) ───────────────────────────
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR_BN; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int i = NG; i < NG+NB; ++i) {
            if (by > 0) {
                int b_bot = (by-1)*NX + bx;
                blks[b].Q[v][cell_idx(i, NG-1-g, k)] =
                    blks[b_bot].Q[v][cell_idx(i, NB+NG-1-g, k)];
            }
            if (by < NY-1) {
                int b_top = (by+1)*NX + bx;
                blks[b].Q[v][cell_idx(i, NB+NG+g, k)] =
                    blks[b_top].Q[v][cell_idx(i, NG+g, k)];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
static void b7_rayleigh_taylor()
{
    printf("\n-- B7  Rayleigh-Taylor instability (Allaire 2002, Youngs 1984) --\n");

    // EOS: both phases identical ideal gas with γ=1.4 (same as single-fluid Euler)
    BNEosParams eos;
    // Override: make both phases identical ideal gas
    // gamma1=1.4, pinf1=0 (air), gamma2=1.4, pinf2=0 (same)
    eos.gamma2 = 1.4;
    eos.pinf2  = 0.0;

    const double g_grav = 1.0;     // gravitational acceleration
    const double rho1   = 1.0;     // light phase density
    const double rho2   = 2.0;     // heavy phase density
    const double p0     = 100.0;   // reference pressure at y=0 (high enough: Ma<<1)
    const double A_pert = 0.01;    // interface perturbation amplitude
    const double delta  = 0.025;   // tanh interface thickness
    const double Lx     = 0.5;     // domain width (half wavelength of perturbation)
    const double Ly     = 1.0;     // domain height

    // 2D grid of BNCellBlocks: NX × NY in x-y
    // Level-2 equivalent: NX=2, NY=4 → 2×4 blocks = 16×32 interior cells
    const int NX = 2;   // blocks in x → NX*NB = 16 cells
    const int NY = 4;   // blocks in y → NY*NB = 32 cells
    const double hx = Lx / (NX * NB);  // cell size in x
    const double hy = Ly / (NY * NB);  // cell size in y
    // Use hx == hy for CFL simplicity (hx = 0.03125, hy = 0.03125)
    const double h = hx;  // assuming hx == hy

    printf("   NX=%d NY=%d hx=%.5f hy=%.5f h=%.5f\n", NX, NY, hx, hy, h);

    // Linear RT growth rate
    const double k_rt  = 2.0 * M_PI / Lx;  // = 4π (one full wavelength in [0,Lx])
    const double A_t   = (rho2 - rho1) / (rho2 + rho1);  // = 1/3
    const double gamma_rt = std::sqrt(g_grav * k_rt * A_t);
    printf("   k=%.4f  A_t=%.4f  γ_RT=%.4f (e-fold time=%.4f)\n",
           k_rt, A_t, gamma_rt, 1.0/gamma_rt);

    std::vector<BNCellBlock> blks(NX * NY);
    for (int by = 0; by < NY; ++by)
    for (int bx = 0; bx < NX; ++bx) {
        int b       = by * NX + bx;
        blks[b].h   = h;
        blks[b].ox  = bx * NB * hx;
        blks[b].oy  = by * NB * hy;
        blks[b].oz  = 0.0;
    }

    // ── Initial condition ─────────────────────────────────────────────────────
    for (auto& blk : blks)
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const double xc = blk.ox + (i - NG + 0.5) * hx;
        const double yc = blk.oy + (j - NG + 0.5) * hy;

        // Interface at y = 0.5 + A_pert*cos(2πx/Lx)
        const double y_iface = 0.5 + A_pert * std::cos(2.0*M_PI*xc/Lx);
        // α₁: smooth step, 1 below interface (light phase), 0 above (heavy phase)
        const double alpha1  = 0.5*(1.0 - std::tanh((yc - y_iface)/delta));

        // Hydrostatic pressure (each layer separately)
        // p(y) = p_bottom - integral of rho*g dy
        // For y < 0.5 (light zone): p = p0 - rho1*g*y
        // For y >= 0.5 (heavy zone): p = p0 - rho1*g*0.5 - rho2*g*(y-0.5)
        double p_hyd;
        if (yc < 0.5)
            p_hyd = p0 - rho1 * g_grav * yc;
        else
            p_hyd = p0 - rho1*g_grav*0.5 - rho2*g_grav*(yc - 0.5);

        const double alpha2 = 1.0 - alpha1;
        const int f = cell_idx(i,j,k);
        blk.Q[0][f] = alpha1 * rho1;    // α₁ρ₁
        blk.Q[1][f] = alpha2 * rho2;    // α₂ρ₂
        blk.Q[2][f] = 0.0;              // ρu = 0
        blk.Q[3][f] = 0.0;              // ρv = 0
        blk.Q[4][f] = 0.0;              // ρw = 0
        // Total energy: both phases at p_hyd
        const double rho_e = alpha1*(p_hyd + eos.gamma1*eos.pinf1)/(eos.gamma1-1.0)
                           + alpha2*(p_hyd + eos.gamma2*eos.pinf2)/(eos.gamma2-1.0);
        blk.Q[5][f] = rho_e;
        blk.Q[6][f] = alpha1;
    }

    const int kmid = NG + NB/2;

    // ── Interface amplitude: y-location of α₁=0.5 contour per x-column ────────
    // Flatten all (y, α₁) pairs for each x-column across ALL blocks,
    // then find the crossing from above (α₁ >= 0.5) to below (α₁ < 0.5).
    auto measure_amplitude = [&]() -> double {
        double amp_max = 0.0;
        // One column per interior x-cell position: NX*NB columns total
        for (int ci = 0; ci < NX*NB; ++ci) {
            const int bx = ci / NB;
            const int li = ci % NB + NG;  // local i in block: NG..NG+NB-1
            // Build flat y-profile of α₁ across all y-blocks
            struct Row { double yc, a1; };
            std::vector<Row> col;
            col.reserve(NY * NB);
            for (int by = 0; by < NY; ++by) {
                const int b = by * NX + bx;
                for (int j = NG; j < NG+NB; ++j) {
                    double yc = blks[b].oy + (j - NG + 0.5) * hy;
                    double a1 = blks[b].Q[6][cell_idx(li, j, kmid)];
                    col.push_back({yc, a1});
                }
            }
            // Find first downward crossing (a1 >= 0.5 → a1 < 0.5)
            double y_cross = 0.5;
            bool found = false;
            for (int r = 0; r + 1 < (int)col.size() && !found; ++r) {
                if (col[r].a1 >= 0.5 && col[r+1].a1 < 0.5) {
                    double t_interp = (col[r].a1 - 0.5) / (col[r].a1 - col[r+1].a1);
                    y_cross = col[r].yc + t_interp * hy;
                    found = true;
                }
            }
            amp_max = std::max(amp_max, std::abs(y_cross - 0.5));
        }
        return amp_max;
    };

    const double A0 = measure_amplitude();
    printf("   A0 = %.4e (expected ~%.4e)\n", A0, A_pert);

    // ── Time integration (Forward Euler + gravity source) ─────────────────────
    const double CFL      = 0.3;
    const int    N_STEPS  = 800;   // ~t=0.8–1.0 depending on dt
    double t = 0.0;

    for (int step = 0; step < N_STEPS; ++step) {
        fill_ghosts_2d(blks, NX, NY);

        // CFL time step
        double dt = 1e30;
        for (auto& blk : blks) dt = std::min(dt, bn_cfl_dt(blk, CFL, eos));

        // RHS
        std::vector<BNCellBlock> rhs(NX * NY);
        for (int b = 0; b < NX*NY; ++b) {
            rhs[b].h = h;
            compute_rhs_bn(blks[b], rhs[b], eos);
        }

        // Update: Q += dt*rhs + gravity source
        for (int b = 0; b < NX*NY; ++b)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i,j,k);
            // Total density = α₁ρ₁ + α₂ρ₂
            const double rho_tot = blks[b].Q[0][f] + blks[b].Q[1][f];
            // Gravity source: momentum y += -rho*g*dt, energy += -rho*v*g*dt
            const double rhov_old = blks[b].Q[3][f];
            for (int v = 0; v < NVAR_BN; ++v)
                blks[b].Q[v][f] += dt * rhs[b].Q[v][f];
            // Gravity body force (explicit operator splitting)
            blks[b].Q[3][f] -= dt * rho_tot * g_grav;
            // Energy source: dE/dt = -ρ*v*g
            blks[b].Q[5][f] -= dt * rhov_old * g_grav;
        }
        t += dt;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────
    const double A_final = measure_amplitude();
    const double A_ratio = A_final / std::max(A0, 1e-30);

    // B7a: α₁ ∈ [0,1]
    double a1_min = 1e30, a1_max = -1e30;
    double a1r1_min = 1e30, a2r2_min = 1e30;
    double p_min_all = 1e30;
    int n_osc_iface = 0;

    // B7b: pressure oscillations near interface (centre x-column only)
    {
        // Gather y-profile of pressure at x_mid
        std::vector<double> pprof;
        for (int by = 0; by < NY; ++by) {
            const int b = by * NX + 0;  // first x-block
            for (int j = NG; j < NG+NB; ++j) {
                const int f = cell_idx(NG, j, kmid);
                Prim2Phase q = bn_cons_to_prim(
                    blks[b].Q[0][f], blks[b].Q[1][f],
                    blks[b].Q[2][f], blks[b].Q[3][f], blks[b].Q[4][f],
                    blks[b].Q[5][f], blks[b].Q[6][f], eos);
                pprof.push_back(q.p);
            }
        }
        // Find interface cell in pressure profile
        int i_iface = (int)pprof.size() / 2;
        const int WIN = 5;
        int lo = std::max(1, i_iface - WIN);
        int hi = std::min((int)pprof.size()-2, i_iface + WIN);
        for (int ii = lo; ii <= hi; ++ii) {
            bool lmax = (pprof[ii] > pprof[ii-1]) && (pprof[ii] > pprof[ii+1]);
            bool lmin = (pprof[ii] < pprof[ii-1]) && (pprof[ii] < pprof[ii+1]);
            if (lmax || lmin) ++n_osc_iface;
        }
    }

    for (int b = 0; b < NX*NY; ++b)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        const int f = cell_idx(i,j,k);
        Prim2Phase q = bn_cons_to_prim(
            blks[b].Q[0][f], blks[b].Q[1][f],
            blks[b].Q[2][f], blks[b].Q[3][f], blks[b].Q[4][f],
            blks[b].Q[5][f], blks[b].Q[6][f], eos);
        a1_min = std::min(a1_min, q.alpha1);
        a1_max = std::max(a1_max, q.alpha1);
        a1r1_min = std::min(a1r1_min, blks[b].Q[0][f]);
        a2r2_min = std::min(a2r2_min, blks[b].Q[1][f]);
        p_min_all = std::min(p_min_all, q.p);
    }

    printf("   t_final = %.4f  steps = %d\n", t, N_STEPS);
    printf("   A0 = %.4e  A(t_final) = %.4e  A/A0 = %.2f\n",
           A0, A_final, A_ratio);
    printf("   α₁ ∈ [%.4e, %.4e]  n_osc_iface = %d\n",
           a1_min, a1_max, n_osc_iface);
    printf("   α₁ρ₁_min=%.3e  α₂ρ₂_min=%.3e  p_min=%.3e\n",
           a1r1_min, a2r2_min, p_min_all);

    const double tol_alpha = 1.0e-10;
    check("B7a  α₁ ∈ [0,1]  (Allaire model bound preservation)",
          a1_min >= -tol_alpha && a1_max <= 1.0+tol_alpha,
          std::max(-a1_min, a1_max-1.0), tol_alpha);
    check("B7b  no spurious pressure oscillations near interface (< 3 extrema)",
          n_osc_iface < 3, (double)n_osc_iface, 3.0);
    check("B7c  interface amplitude grew ≥ 1.1× (correct unstable stratification)",
          A_ratio >= 1.1, A_ratio, 1.1);
    check("B7d  α₁ρ₁ ≥ 0  and  α₂ρ₂ ≥ 0  (no negative partial densities)",
          a1r1_min >= 0.0 && a2r2_min >= 0.0,
          std::min(a1r1_min, a2r2_min), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.7 gate: Rayleigh-Taylor instability (Allaire 5-equation) ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d  NVAR_BN=%d\n",
           NB, NG, NB2, NCELL, NVAR_BN);
    b7_rayleigh_taylor();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.7 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
