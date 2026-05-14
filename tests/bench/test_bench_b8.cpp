// B.8 benchmark — Lid-driven cavity Re=1000 (2D, viscous NS)
//
// Validates the compressible Navier-Stokes solver (HLLC-ES + Sutherland
// viscosity) against the classic incompressible reference of Ghia et al. (1982).
//
// Physical parameters are chosen so that Sutherland's law gives the target
// Reynolds number at a manageable Mach number:
//   ρ₀ = 1e-3 kg/m³  (dilute gas, p₀=86 Pa, T₀=300 K)
//   μ₀ = sutherland(T₀) ≈ 1.847e-5 Pa·s
//   U_lid = Re·μ₀/(ρ₀·L) ≈ 18.47 m/s
//   c₀ = sqrt(γ·p₀/ρ₀) ≈ 347 m/s  →  Ma ≈ 0.053  (quasi-incompressible)
//
// Grid: 4×4 blocks → 32×32 interior cells in x-y, 8 cells in z (periodic).
// BCs:  left/right/bottom walls: adiabatic no-slip (u=v=w=0);
//       top lid: adiabatic no-slip with u=U_lid.
// Time: SSP-RK3 (Shu-Osher form), CFL=0.4, t_end=1.0 s (≈18 convective times).
//
// Reference: Ghia, Ghia & Shin (1982) J. Comput. Phys. 48, 387–411.
//
// Gate criteria (coarse grid, 32×32):
//   B8a  u_min/U_lid < −0.15  (main vortex return flow present)
//   B8b  y(u_min) ∈ [0.05, 0.45]  (located in lower/central region)
//   B8c  p > 0 everywhere  (no blow-up)
//   B8d  |Δmass|/mass₀ < 1e-8  (closed domain, no-slip → no mass flux)
//
// Note: Forward Euler + WENO5 in 2D is only stable at CFL < ~0.2.
// SSP-RK3 extends the stability margin to CFL ≈ 0.6 and is the correct
// time integrator for WENO5 schemes (Shu & Osher 1988).

#include "solver/ns_solver.hpp"
#include "schemes/operators.hpp"
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

// ── Ghost fill for lid-driven cavity ─────────────────────────────────────────
// x: no-slip walls on left (bx=0) and right (bx=NX-1)
// y: no-slip bottom (by=0), moving lid top (by=NY-1, u=U_lid)
// z: periodic (one block deep)
static void fill_cavity_ghosts(std::vector<CellBlock>& blks,
                                int NX, int NY, double U_lid)
{
    for (int by = 0; by < NY; ++by)
    for (int bx = 0; bx < NX; ++bx) {
        const int b = by * NX + bx;

        // ── z-periodic (single block wraps onto itself) ───────────────────────
        for (int v = 0; v < NVAR; ++v)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            for (int g = 0; g < NG; ++g) {
                blks[b].Q[v][cell_idx(i, j, NG-1-g)]   =
                    blks[b].Q[v][cell_idx(i, j, NB+NG-1-g)];
                blks[b].Q[v][cell_idx(i, j, NB+NG+g)]  =
                    blks[b].Q[v][cell_idx(i, j, NG+g)];
            }
        }

        // ── Interior x neighbours ─────────────────────────────────────────────
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j) {
            if (bx > 0) {
                blks[b].Q[v][cell_idx(NG-1-g, j, k)] =
                    blks[by*NX + bx-1].Q[v][cell_idx(NB+NG-1-g, j, k)];
            }
            if (bx < NX-1) {
                blks[b].Q[v][cell_idx(NB+NG+g, j, k)] =
                    blks[by*NX + bx+1].Q[v][cell_idx(NG+g, j, k)];
            }
        }

        // ── Interior y neighbours ─────────────────────────────────────────────
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int i = NG; i < NG+NB; ++i) {
            if (by > 0) {
                blks[b].Q[v][cell_idx(i, NG-1-g, k)] =
                    blks[(by-1)*NX + bx].Q[v][cell_idx(i, NB+NG-1-g, k)];
            }
            if (by < NY-1) {
                blks[b].Q[v][cell_idx(i, NB+NG+g, k)] =
                    blks[(by+1)*NX + bx].Q[v][cell_idx(i, NG+g, k)];
            }
        }

        // ── x-wall no-slip (negate all velocity components; E unchanged) ───────
        // Left wall (bx=0): ghost i=NG-1-g mirrors interior i=NG+g
        if (bx == 0) {
            for (int g = 0; g < NG; ++g)
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j) {
                const int gi = cell_idx(NG-1-g, j, k);
                const int ii = cell_idx(NG+g,   j, k);
                blks[b].Q[0][gi] =  blks[b].Q[0][ii];   // ρ: copy
                blks[b].Q[1][gi] = -blks[b].Q[1][ii];   // ρu: reflect
                blks[b].Q[2][gi] = -blks[b].Q[2][ii];   // ρv: reflect
                blks[b].Q[3][gi] = -blks[b].Q[3][ii];   // ρw: reflect
                blks[b].Q[4][gi] =  blks[b].Q[4][ii];   // E: unchanged (|u|² same)
            }
        }
        // Right wall (bx=NX-1): ghost i=NB+NG+g mirrors interior i=NB+NG-1-g
        if (bx == NX-1) {
            for (int g = 0; g < NG; ++g)
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j) {
                const int gi = cell_idx(NB+NG+g,   j, k);
                const int ii = cell_idx(NB+NG-1-g, j, k);
                blks[b].Q[0][gi] =  blks[b].Q[0][ii];
                blks[b].Q[1][gi] = -blks[b].Q[1][ii];
                blks[b].Q[2][gi] = -blks[b].Q[2][ii];
                blks[b].Q[3][gi] = -blks[b].Q[3][ii];
                blks[b].Q[4][gi] =  blks[b].Q[4][ii];
            }
        }

        // ── y-wall ghost fill helper ──────────────────────────────────────────
        auto fill_y_wall = [&](int j_ghost, int j_int, double u_wall) {
            for (int k = NG; k < NG+NB; ++k)
            for (int i = NG; i < NG+NB; ++i) {
                const int gi  = cell_idx(i, j_ghost, k);
                const int ii  = cell_idx(i, j_int,   k);
                const double rho  = blks[b].Q[0][ii];
                const double ru   = blks[b].Q[1][ii];
                const double rv   = blks[b].Q[2][ii];
                const double rw   = blks[b].Q[3][ii];
                const double E_in = blks[b].Q[4][ii];
                const double KE_in = 0.5*(ru*ru + rv*rv + rw*rw)/rho;
                const double p_in  = (GAMMA - 1.0) * (E_in - KE_in);

                // Adiabatic BC: T_ghost = T_int → ρ_ghost = ρ_int
                const double ru_g = 2.0*rho*u_wall - ru;  // u_face = u_wall
                const double rv_g = -rv;                   // v=0 at wall
                const double rw_g = -rw;                   // w=0 at wall
                const double KE_g = 0.5*(ru_g*ru_g + rv_g*rv_g + rw_g*rw_g)/rho;

                blks[b].Q[0][gi] = rho;
                blks[b].Q[1][gi] = ru_g;
                blks[b].Q[2][gi] = rv_g;
                blks[b].Q[3][gi] = rw_g;
                blks[b].Q[4][gi] = p_in/(GAMMA-1.0) + KE_g;
            }
        };

        // Bottom wall (by=0): no-slip, u_wall=0
        if (by == 0) {
            for (int g = 0; g < NG; ++g)
                fill_y_wall(NG-1-g, NG+g, 0.0);
        }
        // Top wall / moving lid (by=NY-1): u_wall=U_lid
        if (by == NY-1) {
            for (int g = 0; g < NG; ++g)
                fill_y_wall(NB+NG+g, NB+NG-1-g, U_lid);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
static void b8_lid_driven_cavity()
{
    printf("\n-- B8  Lid-driven cavity Re=1000 (Ghia et al. 1982) --\n");

    const int    NX = 4, NY = 4;         // 32×32 cells (4 blocks × NB=8 per axis)
    const double L  = 1.0;               // cavity side length
    const double hx = L / (NX * NB);    // = 0.03125 m
    const double hy = L / (NY * NB);    // = 0.03125 m
    const double hz = hx;               // one cell deep in z

    // Physical parameters: ρ₀=1e-3 so that Sutherland gives μ₀≈1.85e-5 at T₀=300K
    // and Re=ρ₀*U_lid*L/μ₀ = 1000 at Ma ≈ 0.053.
    const double rho0  = 1.0e-3;                          // kg/m³
    const double T0    = 300.0;                            // K
    const double p0    = rho0 * R_GAS * T0;               // ≈ 86.12 Pa
    const double mu0   = sutherland(T0);                   // ≈ 1.847e-5 Pa·s
    const double Re    = 1000.0;
    const double U_lid = Re * mu0 / (rho0 * L);           // ≈ 18.47 m/s
    const double c0    = std::sqrt(GAMMA * p0 / rho0);    // ≈ 347 m/s
    const double Ma    = U_lid / c0;

    printf("   NX=%d NY=%d h=%.5f U_lid=%.4f c0=%.2f Ma=%.4f mu0=%.3e\n",
           NX, NY, hx, U_lid, c0, Ma, mu0);
    printf("   Re=%.0f (measured=%.1f)\n", Re, rho0*U_lid*L/mu0);

    // Allocate blocks
    const int NB2_ = NX * NY;
    std::vector<CellBlock> blks(NB2_);
    for (int by = 0; by < NY; ++by)
    for (int bx = 0; bx < NX; ++bx) {
        int b = by * NX + bx;
        blks[b].h  = hx;
        blks[b].ox = bx * NB * hx;
        blks[b].oy = by * NB * hy;
        blks[b].oz = 0.0;
    }

    // ── IC: uniform rest state ────────────────────────────────────────────────
    const double E0 = p0 / (GAMMA - 1.0);
    for (auto& blk : blks)
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const int f = cell_idx(i, j, k);
        blk.Q[0][f] = rho0;
        blk.Q[1][f] = 0.0;
        blk.Q[2][f] = 0.0;
        blk.Q[3][f] = 0.0;
        blk.Q[4][f] = E0;
    }

    // ── Initial mass ──────────────────────────────────────────────────────────
    double mass0 = 0.0;
    for (auto& blk : blks) {
        const double dV = blk.h * blk.h * hz;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            mass0 += blk.Q[0][cell_idx(i,j,k)] * dV;
    }

    // ── SSP-RK3 work arrays ───────────────────────────────────────────────────
    // Shu-Osher form: Q1 = Q + dt*L(Q)
    //                 Q2 = 3/4*Q + 1/4*Q1 + 1/4*dt*L(Q1)
    //                 Q  = 1/3*Q + 2/3*Q2 + 2/3*dt*L(Q2)
    //
    // Initialize Qn/Q1/Q2 as FULL copies of blks so that corner ghost cells
    // (e.g. (i=1,j=1,k) — x+y corner, never filled by fill_cavity_ghosts)
    // carry valid IC values (rho0, u=0).  The viscous tangential-gradient
    // stencil accesses these corners; rho=0 from default-init gives NaN.
    std::vector<CellBlock> Qn(blks), Q1(blks), Q2(blks), rhs(NB2_);
    for (int b = 0; b < NB2_; ++b)
        rhs[b].h = blks[b].h;

    // Helper: copy interior cells from src to dst
    auto copy_int = [&](std::vector<CellBlock>& dst,
                        const std::vector<CellBlock>& src) {
        for (int b = 0; b < NB2_; ++b)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i,j,k);
            dst[b].Q[v][f] = src[b].Q[v][f];
        }
    };

    // Helper: Q_out[f] = a*Qa[f] + b*Qb[f] + c*Rhs[f]  (interior only)
    auto lc3 = [&](std::vector<CellBlock>& out,
                   double a, const std::vector<CellBlock>& Qa,
                   double b_c, const std::vector<CellBlock>& Qb,
                   double c_c, double dt, const std::vector<CellBlock>& Rc) {
        for (int b = 0; b < NB2_; ++b)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i,j,k);
            out[b].Q[v][f] = a*Qa[b].Q[v][f]
                           + b_c*Qb[b].Q[v][f]
                           + c_c*dt*Rc[b].Q[v][f];
        }
    };

    // ── Time integration (SSP-RK3) ────────────────────────────────────────────
    const double CFL      = 0.4;
    const double t_end    = 1.0;    // ≈ 18 convective times (U_lid * t / L ≈ 18)
    const int    max_steps = 100000;
    double t = 0.0;
    int    step = 0;

    for (; step < max_steps && t < t_end; ++step) {
        // CFL time step (from current state)
        double dt = 1e30;
        for (auto& blk : blks) {
            for (int k = NG; k < NG+NB; ++k)
            for (int j = NG; j < NG+NB; ++j)
            for (int i = NG; i < NG+NB; ++i) {
                const int f = cell_idx(i, j, k);
                const double rho = blk.Q[0][f];
                const double ru  = blk.Q[1][f], rv = blk.Q[2][f], rw = blk.Q[3][f];
                const double E   = blk.Q[4][f];
                const double KE  = 0.5*(ru*ru+rv*rv+rw*rw)/rho;
                const double p   = (GAMMA-1.0)*(E-KE);
                const double c   = std::sqrt(GAMMA*p/rho);
                const double sp  = (std::abs(ru)+std::abs(rv)+std::abs(rw))/rho + c;
                if (sp > 0.0) dt = std::min(dt, CFL * blk.h / sp);
            }
        }
        if (t + dt > t_end) dt = t_end - t;

        // Stage 1: Q1 = Qn + dt*L(Qn)
        copy_int(Qn, blks);
        fill_cavity_ghosts(blks, NX, NY, U_lid);
        for (int b = 0; b < NB2_; ++b) compute_rhs(blks[b], rhs[b]);
        lc3(Q1, 1.0, Qn, 0.0, Qn, 1.0, dt, rhs);

        // Stage 2: Q2 = 3/4*Qn + 1/4*Q1 + 1/4*dt*L(Q1)
        fill_cavity_ghosts(Q1, NX, NY, U_lid);
        for (int b = 0; b < NB2_; ++b) compute_rhs(Q1[b], rhs[b]);
        lc3(Q2, 0.75, Qn, 0.25, Q1, 0.25, dt, rhs);

        // Stage 3: Qn+1 = 1/3*Qn + 2/3*Q2 + 2/3*dt*L(Q2)
        fill_cavity_ghosts(Q2, NX, NY, U_lid);
        for (int b = 0; b < NB2_; ++b) compute_rhs(Q2[b], rhs[b]);
        lc3(blks, 1.0/3.0, Qn, 2.0/3.0, Q2, 2.0/3.0, dt, rhs);

        t += dt;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────
    // Extract u/U_lid profile along vertical centerline x ≈ L/2.
    // Use the leftmost interior cell of bx=NX/2 (x ≈ L/2 + h/2).
    struct CentPt { double y, u_nd; };
    std::vector<CentPt> prof;
    const int bx_mid = NX / 2;       // = 2
    const int i_loc  = NG;           // first interior cell of that block

    for (int by = 0; by < NY; ++by) {
        const int b  = by * NX + bx_mid;
        const int kc = NG + NB / 2;  // z-midplane
        for (int j = NG; j < NG+NB; ++j) {
            const double yc = blks[b].oy + (j - NG + 0.5) * hy;
            const int    f  = cell_idx(i_loc, j, kc);
            const double rho = blks[b].Q[0][f];
            const double u   = blks[b].Q[1][f] / rho;
            prof.push_back({yc, u / U_lid});
        }
    }

    // Find minimum u/U_lid and its y-location
    double u_min = 1e30, y_at_min = 0.0;
    for (auto& pt : prof) {
        if (pt.u_nd < u_min) { u_min = pt.u_nd; y_at_min = pt.y; }
    }

    // Minimum pressure
    double p_min = 1e30;
    for (auto& blk : blks) {
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i, j, k);
            const double rho = blk.Q[0][f];
            const double KE  = 0.5*(blk.Q[1][f]*blk.Q[1][f]
                                  + blk.Q[2][f]*blk.Q[2][f]
                                  + blk.Q[3][f]*blk.Q[3][f]) / rho;
            p_min = std::min(p_min, (GAMMA-1.0)*(blk.Q[4][f]-KE));
        }
    }

    // Final mass
    double mass1 = 0.0;
    for (auto& blk : blks) {
        const double dV = blk.h * blk.h * hz;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            mass1 += blk.Q[0][cell_idx(i,j,k)] * dV;
    }
    const double mass_err = std::fabs(mass1 - mass0) / std::fabs(mass0);

    printf("   t_final=%.4f s  steps=%d  (%.1f conv. times)\n",
           t, step, t * U_lid / L);
    printf("   u_min/U_lid = %.4f at y = %.4f\n", u_min, y_at_min);
    printf("   p_min = %.3e Pa  mass_err = %.2e\n", p_min, mass_err);
    printf("   Profile (u/U_lid vs y):\n");
    for (auto& pt : prof)
        printf("     y=%.3f  u/U=% .4f\n", pt.y, pt.u_nd);
    printf("   Ref Ghia Re=1000: u_min/U ≈ −0.383 at y ≈ 0.17\n");

    check("B8a  u_min/U_lid < −0.15  (main vortex return flow)",
          u_min < -0.15, -u_min, 0.15);
    check("B8b  y(u_min) ∈ [0.05, 0.45]  (lower/central region)",
          y_at_min >= 0.05 && y_at_min <= 0.45, y_at_min, 0.20);
    check("B8c  p > 0 everywhere  (no blow-up)",
          p_min > 0.0, p_min, 0.0);
    check("B8d  mass conservation |Δm|/m₀ < 1e-8  (no-flux walls)",
          mass_err < 1e-8, mass_err, 1e-8);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.8 gate: Lid-driven cavity Re=1000 (2D viscous NS) ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d\n", NB, NG, NB2, NCELL);
    b8_lid_driven_cavity();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.8 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
