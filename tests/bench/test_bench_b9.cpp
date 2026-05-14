// B.9 benchmark — Turbulent channel flow Re_τ=180 (3D LES)
//
// Validates: Dynamic Smagorinsky SGS, mixed periodic+wall BCs, body force
// Domain: Lx × Ly × Lz = 4 × 2 × 2 m  (δ=1 m half-channel height)
// Grid:   NX=4 × NY=2 × NZ=2 blocks → 32×16×16 cells,  h=0.125 m (Δ⁺≈22.5)
// BCs:    periodic x/z,  no-slip walls at y=0 and y=Ly
// Drive:  constant body force f_x = ρ₀ u_τ²/δ  (= mean pressure gradient)
// SGS:    DynamicSmagorinsky (operator splitting after each RK3 step)
//
// Physical parameters (T₀=300 K, Ma_bulk≈0.1):
//   c₀   = √(γ R T₀) ≈ 347.2 m/s
//   U_bk  = 0.1·c₀ ≈ 34.72 m/s    (Ma_bulk = 0.1)
//   u_τ  = U_bk/15.6 ≈ 2.226 m/s  (empirical for Re_τ=180)
//   μ₀   = sutherland(T₀) ≈ 1.847e-5 Pa·s
//   ρ₀   = Re_τ·μ₀/u_τ ≈ 1.494e-3 kg/m³
//   ν    = μ₀/ρ₀ = u_τ/Re_τ ✓
//
// IC: van Driest log-law profile + sinusoidal perturbations (amp = 0.1 u_τ)
//
// Gate criteria (coarse grid, ~1.3 flow-through times):
//   B9a: u_mean_centerline / u_τ ∈ [5, 30]    (body force drives flow)
//   B9b: u_bulk / U_bulk ∈ [0.3, 3.0]         (bulk velocity in correct range)
//   B9c: |Δmass|/mass₀ < 1e-7                 (mass conserved)
//   B9d: p_min > 0                             (no blow-up)
//
// References: Moser, Kim & Mansour (1999) Phys. Fluids 11(4):943–945
//             Germano et al. (1991) Phys. Fluids A 3(7):1760–1765

#include "solver/ns_solver.hpp"
#include "schemes/operators.hpp"
#include "models/sgs.hpp"
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

// ── Grid layout ───────────────────────────────────────────────────────────────
static constexpr int NX  = 4, NY = 2, NZ = 2;  // blocks per direction
static constexpr int NB_ = NX * NY * NZ;         // 16 blocks total

static inline int bid(int ix, int iy, int iz) noexcept {
    return (iy * NZ + iz) * NX + ix;
}

// ── Ghost fill: periodic x/z, adiabatic no-slip walls at y=0 and y=Ly ────────
static void fill_channel_ghosts(std::vector<CellBlock>& blks)
{
    for (int iy = 0; iy < NY; ++iy)
    for (int iz = 0; iz < NZ; ++iz)
    for (int ix = 0; ix < NX; ++ix) {
        CellBlock& blk = blks[bid(ix, iy, iz)];

        // ── x-periodic ───────────────────────────────────────────────────────
        const CellBlock& xm = blks[bid((ix - 1 + NX) % NX, iy, iz)];
        const CellBlock& xp = blks[bid((ix + 1)      % NX, iy, iz)];
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j) {
            blk.Q[v][cell_idx(NG-1-g,  j, k)] = xm.Q[v][cell_idx(NB+NG-1-g, j, k)];
            blk.Q[v][cell_idx(NB+NG+g, j, k)] = xp.Q[v][cell_idx(NG+g,      j, k)];
        }

        // ── z-periodic ───────────────────────────────────────────────────────
        const CellBlock& zm = blks[bid(ix, iy, (iz - 1 + NZ) % NZ)];
        const CellBlock& zp = blks[bid(ix, iy, (iz + 1)      % NZ)];
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR; ++v)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            blk.Q[v][cell_idx(i, j, NG-1-g)]   = zm.Q[v][cell_idx(i, j, NB+NG-1-g)];
            blk.Q[v][cell_idx(i, j, NB+NG+g)]  = zp.Q[v][cell_idx(i, j, NG+g)];
        }

        // ── y-interior: copy from y-neighbor ─────────────────────────────────
        for (int g = 0; g < NG; ++g)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int i = NG; i < NG+NB; ++i) {
            if (iy > 0)
                blk.Q[v][cell_idx(i, NG-1-g,  k)] =
                    blks[bid(ix, iy-1, iz)].Q[v][cell_idx(i, NB+NG-1-g, k)];
            if (iy < NY-1)
                blk.Q[v][cell_idx(i, NB+NG+g, k)] =
                    blks[bid(ix, iy+1, iz)].Q[v][cell_idx(i, NG+g, k)];
        }

        // ── y-wall no-slip (adiabatic: E_ghost = E_int, negate momenta) ──────
        if (iy == 0) {
            for (int g = 0; g < NG; ++g)
            for (int k = NG; k < NG+NB; ++k)
            for (int i = NG; i < NG+NB; ++i) {
                const int gi = cell_idx(i, NG-1-g, k);
                const int ii = cell_idx(i, NG+g,   k);
                blk.Q[0][gi] =  blk.Q[0][ii];
                blk.Q[1][gi] = -blk.Q[1][ii];
                blk.Q[2][gi] = -blk.Q[2][ii];
                blk.Q[3][gi] = -blk.Q[3][ii];
                blk.Q[4][gi] =  blk.Q[4][ii];
            }
        }
        if (iy == NY-1) {
            for (int g = 0; g < NG; ++g)
            for (int k = NG; k < NG+NB; ++k)
            for (int i = NG; i < NG+NB; ++i) {
                const int gi = cell_idx(i, NB+NG+g,   k);
                const int ii = cell_idx(i, NB+NG-1-g, k);
                blk.Q[0][gi] =  blk.Q[0][ii];
                blk.Q[1][gi] = -blk.Q[1][ii];
                blk.Q[2][gi] = -blk.Q[2][ii];
                blk.Q[3][gi] = -blk.Q[3][ii];
                blk.Q[4][gi] =  blk.Q[4][ii];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
static void b9_channel_flow()
{
    printf("\n-- B9  Turbulent channel flow Re_τ=180 (Moser et al. 1999) --\n");

    constexpr double PI = 3.14159265358979323846;

    const double T0     = 300.0;
    const double c0     = std::sqrt(GAMMA * R_GAS * T0);   // ≈347.2 m/s
    const double mu0    = sutherland(T0);                    // ≈1.847e-5 Pa·s
    const double RE_TAU = 180.0;
    const double UB_UT  = 15.6;                             // U_bulk/u_tau at Re_tau=180
    const double Ubulk  = 0.1 * c0;                         // Ma_bulk = 0.1
    const double u_tau  = Ubulk / UB_UT;                    // ≈2.226 m/s
    const double nu0    = u_tau / RE_TAU;                   // kinematic viscosity
    const double rho0   = mu0 / nu0;                        // ≈1.494e-3 kg/m³
    const double delta  = 1.0;                              // half-channel height [m]
    const double Ly     = 2.0 * delta;
    const double h      = Ly / (NY * NB);                   // = 0.125 m (Δ⁺=22.5)
    const double Lx     = NX * NB * h;                     // = 4.0 m
    const double Lz     = NZ * NB * h;                     // = 2.0 m
    const double p0     = rho0 * R_GAS * T0;
    const double E0     = p0 / (GAMMA - 1.0);
    // Body force = mean pressure gradient equivalent: dP/dx = -rho0*u_tau²/delta
    const double f_body = rho0 * u_tau * u_tau / delta;     // [N/m³]

    printf("   NX=%d NY=%d NZ=%d h=%.4f  Lx=%.2f Ly=%.2f Lz=%.2f\n",
           NX, NY, NZ, h, Lx, Ly, Lz);
    printf("   Re_tau=%.0f  u_tau=%.4f  rho0=%.4e  nu=%.4e\n",
           RE_TAU, u_tau, rho0, nu0);
    printf("   Ma_bulk=0.10  U_bulk=%.3f  c0=%.2f  f_body=%.4e\n",
           Ubulk, c0, f_body);

    // ── Allocate blocks ───────────────────────────────────────────────────────
    std::vector<CellBlock> blks(NB_);
    for (int iy = 0; iy < NY; ++iy)
    for (int iz = 0; iz < NZ; ++iz)
    for (int ix = 0; ix < NX; ++ix) {
        int b = bid(ix, iy, iz);
        blks[b].h  = h;
        blks[b].ox = ix * NB * h;
        blks[b].oy = iy * NB * h;
        blks[b].oz = iz * NB * h;
    }

    // Law of the wall: u/u_tau as function of y_wall distance
    auto llaw = [&](double y_wall) -> double {
        const double yp = y_wall * u_tau / nu0;   // y+
        return u_tau * (yp <= 11.6 ? yp : (1.0/0.41 * std::log(yp) + 5.2));
    };

    // ── IC: Step 1 — initialise ALL cells (incl. ghosts) to rest state ────────
    // Ensures corner ghost cells (never filled by fill_channel_ghosts) carry
    // valid rho=rho0 so that u=ρu/ρ never produces NaN in viscous stencils.
    for (auto& blk : blks) {
        for (int f = 0; f < NCELL; ++f) {
            blk.Q[0][f] = rho0;
            blk.Q[1][f] = 0.0;
            blk.Q[2][f] = 0.0;
            blk.Q[3][f] = 0.0;
            blk.Q[4][f] = E0;
        }
    }

    // ── IC: Step 2 — interior cells: log-law profile + sinusoidal perturbation ─
    for (int iy = 0; iy < NY; ++iy)
    for (int iz = 0; iz < NZ; ++iz)
    for (int ix = 0; ix < NX; ++ix) {
        auto& blk = blks[bid(ix, iy, iz)];
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int    f  = cell_idx(i, j, k);
            const double x  = blk.ox + (i - NG + 0.5) * h;
            const double y  = blk.oy + (j - NG + 0.5) * h;
            const double z  = blk.oz + (k - NG + 0.5) * h;
            const double yw = std::min(y, Ly - y);          // distance from nearest wall
            const double um = llaw(yw);                      // mean velocity

            // Sinusoidal perturbation — vanishes at y=0,Ly (sin(πy/Ly)=0)
            const double A  = 0.1 * u_tau;
            const double sy = std::sin(PI * y / Ly);
            const double du = A * sy * std::sin(4*PI*x/Lx) * std::sin(2*PI*z/Lz);
            const double dv = A * sy * std::cos(4*PI*x/Lx);
            const double dw = A * sy * std::sin(2*PI*x/Lx) * std::cos(2*PI*z/Lz);

            const double ru = rho0 * (um + du);
            const double rv = rho0 * dv;
            const double rw = rho0 * dw;
            const double KE = 0.5 * (ru*ru + rv*rv + rw*rw) / rho0;
            blk.Q[0][f] = rho0;
            blk.Q[1][f] = ru;
            blk.Q[2][f] = rv;
            blk.Q[3][f] = rw;
            blk.Q[4][f] = E0 + KE;
        }
    }

    // ── Initial mass ──────────────────────────────────────────────────────────
    double mass0 = 0.0;
    const double dV = h * h * h;
    for (auto& blk : blks)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        mass0 += blk.Q[0][cell_idx(i,j,k)] * dV;

    // ── RK3 work arrays (copy blks so corner ghosts carry IC values) ──────────
    std::vector<CellBlock> Qn(blks), Q1(blks), Q2(blks), rhs(NB_);
    for (int b = 0; b < NB_; ++b) rhs[b].h = blks[b].h;

    auto copy_int = [&](std::vector<CellBlock>& dst,
                        const std::vector<CellBlock>& src) {
        for (int b = 0; b < NB_; ++b)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i,j,k);
            dst[b].Q[v][f] = src[b].Q[v][f];
        }
    };

    // SSP-RK3 linear combination: out = a*Qa + bc*Qb + cc*dt*R  (interior only)
    auto lc3 = [&](std::vector<CellBlock>& out,
                   double a,   const std::vector<CellBlock>& Qa,
                   double bc,  const std::vector<CellBlock>& Qb,
                   double cc, double dt, const std::vector<CellBlock>& R) {
        for (int b = 0; b < NB_; ++b)
        for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i,j,k);
            out[b].Q[v][f] = a*Qa[b].Q[v][f] + bc*Qb[b].Q[v][f] + cc*dt*R[b].Q[v][f];
        }
    };

    // Add body force f_x = rho0*u_tau²/delta to x-momentum and energy RHS
    auto add_forcing = [&](const std::vector<CellBlock>& stg,
                            std::vector<CellBlock>& r) {
        for (int b = 0; b < NB_; ++b)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int f = cell_idx(i, j, k);
            const double u = stg[b].Q[1][f] / stg[b].Q[0][f];
            r[b].Q[1][f] += f_body;
            r[b].Q[4][f] += f_body * u;
        }
    };

    // ── SGS model ─────────────────────────────────────────────────────────────
    DynamicSmagorinskyModel sgs;

    // ── Statistics: u_mean and u_rms by global y-index jg ∈ [0, NY*NB-1] ─────
    const int JG = NY * NB;
    std::vector<double> sum_u(JG, 0.0), sum_u2(JG, 0.0);
    long long n_avg = 0;

    // ── Time integration (SSP-RK3) ────────────────────────────────────────────
    const double CFL       = 0.4;
    const double t_end     = 0.15;    // ≈ 1.3 flow-through times (Tft = Lx/Ubulk)
    const double t_stat    = 0.075;   // begin statistics after ~0.65 Tft
    const int    max_steps = 200000;
    double t = 0.0;
    int    step = 0;

    for (; step < max_steps && t < t_end; ++step) {
        // CFL time step
        double dt = 1e30;
        for (auto& blk : blks)
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            const int    f   = cell_idx(i, j, k);
            const double rho = blk.Q[0][f];
            const double ru  = blk.Q[1][f], rv = blk.Q[2][f], rw = blk.Q[3][f];
            const double E   = blk.Q[4][f];
            const double KE  = 0.5*(ru*ru+rv*rv+rw*rw)/rho;
            const double p   = (GAMMA-1.0)*(E-KE);
            const double cc  = std::sqrt(GAMMA*p/rho);
            const double sp  = (std::abs(ru)+std::abs(rv)+std::abs(rw))/rho + cc;
            if (sp > 0.0) dt = std::min(dt, CFL * h / sp);
        }
        if (t + dt > t_end) dt = t_end - t;

        // Stage 1: Q1 = Qn + dt·L(Qn)
        copy_int(Qn, blks);
        fill_channel_ghosts(blks);
        for (int b = 0; b < NB_; ++b) compute_rhs(blks[b], rhs[b]);
        add_forcing(blks, rhs);
        lc3(Q1, 1.0, Qn, 0.0, Qn, 1.0, dt, rhs);

        // Stage 2: Q2 = ¾Qn + ¼(Q1 + dt·L(Q1))
        fill_channel_ghosts(Q1);
        for (int b = 0; b < NB_; ++b) compute_rhs(Q1[b], rhs[b]);
        add_forcing(Q1, rhs);
        lc3(Q2, 0.75, Qn, 0.25, Q1, 0.25, dt, rhs);

        // Stage 3: Q^{n+1} = ⅓Qn + ⅔(Q2 + dt·L(Q2))
        fill_channel_ghosts(Q2);
        for (int b = 0; b < NB_; ++b) compute_rhs(Q2[b], rhs[b]);
        add_forcing(Q2, rhs);
        lc3(blks, 1.0/3.0, Qn, 2.0/3.0, Q2, 2.0/3.0, dt, rhs);

        t += dt;

        // SGS: operator splitting — fill ghosts then apply DynamicSmagorinsky
        fill_channel_ghosts(blks);
        for (int b = 0; b < NB_; ++b)
            sgs.apply(blks[b], h, dt);

        // Accumulate statistics
        if (t >= t_stat) {
            ++n_avg;
            for (int iy = 0; iy < NY; ++iy)
            for (int iz = 0; iz < NZ; ++iz)
            for (int ix = 0; ix < NX; ++ix) {
                const auto& blk = blks[bid(ix, iy, iz)];
                for (int j = NG; j < NG+NB; ++j) {
                    const int jg = iy * NB + (j - NG);
                    for (int k = NG; k < NG+NB; ++k)
                    for (int i = NG; i < NG+NB; ++i) {
                        const int    f = cell_idx(i, j, k);
                        const double u = blk.Q[1][f] / blk.Q[0][f];
                        sum_u [jg] += u;
                        sum_u2[jg] += u * u;
                    }
                }
            }
        }
    }

    // ── Post-processing ───────────────────────────────────────────────────────
    const long long nxz = n_avg * (long long)(NX * NZ * NB * NB);

    std::vector<double> u_mean(JG), u_rms(JG), yp(JG);
    for (int jg = 0; jg < JG; ++jg) {
        u_mean[jg] = (nxz > 0) ? sum_u[jg]  / (double)nxz : 0.0;
        double u2m = (nxz > 0) ? sum_u2[jg] / (double)nxz : 0.0;
        u_rms [jg] = std::sqrt(std::max(0.0, u2m - u_mean[jg]*u_mean[jg]));
        double y_c = (jg + 0.5) * h;
        yp    [jg] = std::min(y_c, Ly - y_c) * u_tau / nu0;
    }

    const double u_center  = u_mean[JG/2 - 1];  // cell nearest y=delta from below
    const double urms_max  = *std::max_element(u_rms.begin(), u_rms.end());

    // Wall shear diagnostic: τ_w = μ·u_first/(0.5·h)  (coarse-grid estimate)
    const double tau_w_meas  = mu0 * u_mean[0] / (0.5 * h);
    const double tau_w_ref   = rho0 * u_tau * u_tau;
    const double tau_w_ratio = tau_w_meas / tau_w_ref;

    // Bulk velocity: mean of mean profile over the full channel height
    double u_bulk = 0.0;
    for (int jg = 0; jg < JG; ++jg) u_bulk += u_mean[jg];
    u_bulk /= JG;

    // Final mass
    double mass1 = 0.0;
    for (auto& blk : blks)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        mass1 += blk.Q[0][cell_idx(i,j,k)] * dV;
    const double mass_err = std::fabs(mass1 - mass0) / std::fabs(mass0);

    // Minimum pressure
    double p_min = 1e30;
    for (auto& blk : blks)
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        const int    f   = cell_idx(i, j, k);
        const double rho = blk.Q[0][f];
        const double KE  = 0.5*(blk.Q[1][f]*blk.Q[1][f]
                              + blk.Q[2][f]*blk.Q[2][f]
                              + blk.Q[3][f]*blk.Q[3][f]) / rho;
        p_min = std::min(p_min, (GAMMA-1.0)*(blk.Q[4][f]-KE));
    }

    printf("   t_final=%.4f s  steps=%d  n_avg=%lld\n", t, step, (long long)n_avg);
    printf("   u_center=%.3f m/s  u_center/u_tau=%.2f  (ref ≈18.7)\n",
           u_center, u_center/u_tau);
    printf("   u_rms_max/u_tau=%.3f  (ref ≈2.7 at y+≈14)\n", urms_max/u_tau);
    printf("   u_bulk=%.3f m/s  u_bulk/Ubulk=%.3f  (target 1.0)\n",
           u_bulk, u_bulk/Ubulk);
    printf("   tau_w_meas=%.4e  tau_w_ref=%.4e  ratio=%.3f  [diagnostic]\n",
           tau_w_meas, tau_w_ref, tau_w_ratio);
    printf("   mass_err=%.2e  p_min=%.3e\n", mass_err, p_min);
    printf("   y+  u_mean/u_tau  u_rms/u_tau:\n");
    for (int jg = 0; jg < JG; ++jg)
        printf("     jg=%2d  y+=%.1f  u+=%.2f  u_rms/u_tau=%.3f\n",
               jg, yp[jg], u_mean[jg]/u_tau, u_rms[jg]/u_tau);

    check("B9a  u_center/u_tau ∈ [5, 30]  (body force drives flow correctly)",
          u_center/u_tau >= 5.0 && u_center/u_tau <= 30.0, u_center/u_tau, 5.0);
    check("B9b  u_bulk/U_bulk ∈ [0.3, 3.0]  (body force drives flow to target bulk velocity)",
          u_bulk/Ubulk >= 0.3 && u_bulk/Ubulk <= 3.0, u_bulk/Ubulk, 1.0);
    check("B9c  mass conservation |Δm|/m₀ < 1e-7",
          mass_err < 1e-7, mass_err, 1e-7);
    check("B9d  p > 0 everywhere  (no blow-up)",
          p_min > 0.0, p_min, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== B.9 gate: Turbulent channel flow Re_τ=180 (3D LES, Dynamic Smagorinsky) ===\n");
    printf("  NB=%d  NG=%d  NB2=%d  NCELL=%d\n", NB, NG, NB2, NCELL);
    b9_channel_flow();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) printf("==> PASS  B.9 gate cleared\n");
    else             printf("==> FAIL\n");
    return n_fail ? 1 : 0;
}
