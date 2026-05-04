// DESIGN.md reference: Layer 2 — Discrete Operators
//
// Fix log:
//   B1  : viscous energy equation: dE/dt|_visc includes u*ax + v*ay + w*az
//         (u·div(τ) work term was missing).
//   P2.2: convective_rhs refactored to face-centred loop.
//         Each of 3*(NB+1)*NB^2 = 1728 face fluxes computed once
//         vs 3*2*NB^3 = 3072 in the original cell loop — 43.7% fewer HLLC
//         evaluations per block.
//         Sign convention: F is the flux leaving the left cell / entering the
//         right cell.  rhs[left] -= ih*F;  rhs[right] += ih*F.
//   P2.3: Primitive variable cache pc[NCELL] pre-computed ONCE in compute_rhs
//         and shared by both convective and viscous operators.
//         Reduces EOS inversions from ~3584 (convective) + ~6656 (viscous)
//         to NCELL=1000 per compute_rhs call.  Enables SIMD on inner i-loop.
//   B5  : viscous_rhs_impl restructured to face-averaged µ.
//         Conservative divergence form div(τ) with µ_face = ½(µ_L + µ_R)
//         replaces the constant-µ expansion µ·(Δu + ⅓∇div u).
//         A separate µ cache (µ_arr[NCELL]) is built once in compute_rhs
//         alongside the prim cache, adding NCELL Sutherland calls (1000)
//         at negligible cost relative to the gradient evaluation.
//         Energy: cell-centred tau:S and κ·ΔT are unchanged (O(h²)).
//   P3.3: Entropy-stable HLLC-ES flux (Chandrashekar 2013).
//         EC base uses log-mean ρ_ln, β_ln where β=ρ/(2p); face pressure
//         p̂ = ρ̄/(2β̄) from arithmetic means.  Lax-Friedrichs dissipation
//         (λ_max/2)*ΔQ restores entropy stability.  Replaces hllc_flux in
//         convective_rhs_impl, undo_cf_face_flux, accumulate_cf_fine_fluxes.
//   P3.1: WENO5-Z reconstruction with characteristic decomposition
//         (Borges et al. 2008).  Roe-eigenvector projection of 6-cell
//         stencil; scalar WENO5-Z per characteristic variable; back-project
//         to conservative space.  Applied to interior faces [ilo, ihi-1] per
//         axis (full 6-cell stencil fits within NG=2 ghost layers).  Ghost
//         faces ilo-1 and ihi fall back to 1st-order PCM + HLLC-ES.
//   P3.2: Ducros sensor + hybrid KE-preserving/WENO5-ES switch.
//         Ducros (1999): Φ = (div u)²/((div u)²+|curl u|²+ε).
//         In smooth/vortex-dominated regions (Φ≈0) the KE-consistent flux
//         F_KEP (Pirozzoli 2011 arithmetic-mean form) preserves discrete KE
//         and reduces aliasing.  Near shocks (Φ≈1) the WENO5+HLLC-ES flux
//         ensures entropy stability and monotonicity.
//         Blend: F = (1−θ)·F_KEP + θ·F_WENO_ES,  θ = max(Φ_L, Φ_R).
//         When θ < 1e-8 (smooth), WENO5 is skipped entirely (pure KEP path).
//   B9  : No-slip wall faces detected in convective_rhs_impl.
//         HLLC-ES at ghost↔interior boundary faces applies LF dissipation
//         −½λ·ΔQ on all components.  For anti-symmetric no-slip ghost fill
//         (Q[tang]_ghost = −Q[tang]_interior) this creates a tangential
//         momentum drain of order λ·ρ·u/h >> body force or viscosity, locking
//         the wall-adjacent cell to near-zero velocity.
//         Fix: detect anti-symmetric tangential velocity (|u_L+u_R|/|u_L+u_R+ε| < 1e-8)
//         and substitute F_KEP (arithmetic means → u_a=0 → zero tangential flux,
//         correct wall pressure in normal direction) for HLLC-ES at those faces.
//         AMR C/F faces are not affected (neighbour states are not anti-symmetric).

#include "operators.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>
#ifdef _OPENMP
#  include <omp.h>
#endif

// =============================================================================
// HLLC Riemann flux
// =============================================================================
// Reference: Toro (2009) §10.4, Batten et al. (1997).
// axis: 0=x, 1=y, 2=z  — selects which velocity component is normal.
//
// Sign convention: flux is in the +axis direction.
// The five components of the returned flux correspond to:
//   [0] mass flux        rho * u_n
//   [1] x-momentum flux  rho*u*u_n + p*nx  (nx=1 for axis=0, else 0)
//   [2] y-momentum flux
//   [3] z-momentum flux
//   [4] energy flux      (E+p)*u_n

std::array<double,5> hllc_flux(const Prim& L, const Prim& R, int axis) noexcept
{
    double uL = (axis==0)?L.u:(axis==1)?L.v:L.w;
    double uR = (axis==0)?R.u:(axis==1)?R.v:R.w;

    // Roe-average wave speeds (Einfeldt-Roe): entropy-consistent near sonic points.
    // H = c²/(γ-1) + ½|u|²  is the specific total enthalpy.
    const double sqL = std::sqrt(L.rho);
    const double sqR = std::sqrt(R.rho);
    const double isq = 1.0 / (sqL + sqR);
    const double HL  = L.c*L.c/(GAMMA-1.0) + 0.5*(L.u*L.u + L.v*L.v + L.w*L.w);
    const double HR  = R.c*R.c/(GAMMA-1.0) + 0.5*(R.u*R.u + R.v*R.v + R.w*R.w);
    const double uh  = (sqL*L.u + sqR*R.u)*isq;
    const double vh  = (sqL*L.v + sqR*R.v)*isq;
    const double wh  = (sqL*L.w + sqR*R.w)*isq;
    const double Hh  = (sqL*HL  + sqR*HR )*isq;
    const double c2h = (GAMMA-1.0)*(Hh - 0.5*(uh*uh + vh*vh + wh*wh));
    const double ch  = (c2h > 0.0) ? std::sqrt(c2h) : 0.5*(L.c + R.c);
    const double u_h = (axis==0) ? uh : (axis==1) ? vh : wh;  // normal component
    double sL = std::min(uL - L.c, u_h - ch);
    double sR = std::max(uR + R.c, u_h + ch);

    double numer = R.p - L.p + L.rho*uL*(sL - uL) - R.rho*uR*(sR - uR);
    double denom = L.rho*(sL - uL) - R.rho*(sR - uR);
    double sStar = (std::abs(denom) > 1e-300) ? numer / denom : 0.5*(uL + uR);

    auto phys_flux = [&](const Prim& q) -> std::array<double,5> {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GAMMA-1.0) + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        return {q.rho*un,
                q.rho*q.u*un + (axis==0?q.p:0.0),
                q.rho*q.v*un + (axis==1?q.p:0.0),
                q.rho*q.w*un + (axis==2?q.p:0.0),
                (E+q.p)*un};
    };

    auto star_flux = [&](const Prim& q, double sK, double sS)
        -> std::array<double,5>
    {
        double un = (axis==0)?q.u:(axis==1)?q.v:q.w;
        double E  = q.p/(GAMMA-1.0) + 0.5*q.rho*(q.u*q.u+q.v*q.v+q.w*q.w);
        double coeff = q.rho * (sK - un) / (sK - sS);
        double rho_s  = coeff;
        double rhou_s = coeff * (axis==0 ? sS : q.u);
        double rhov_s = coeff * (axis==1 ? sS : q.v);
        double rhow_s = coeff * (axis==2 ? sS : q.w);
        double E_s    = coeff * (E/q.rho + (sS - un)*(sS + q.p/(q.rho*(sK-un))));

        auto F = phys_flux(q);
        double rho  = q.rho;
        double rhou = rho*q.u, rhov = rho*q.v, rhow = rho*q.w;
        return {F[0] + sK*(rho_s  - rho ),
                F[1] + sK*(rhou_s - rhou),
                F[2] + sK*(rhov_s - rhov),
                F[3] + sK*(rhow_s - rhow),
                F[4] + sK*(E_s    - E   )};
    };

    if      (sL >= 0.0) return phys_flux(L);
    else if (sR <= 0.0) return phys_flux(R);
    else if (sStar >= 0.0) return star_flux(L, sL, sStar);
    else                   return star_flux(R, sR, sStar);
}

// =============================================================================
// P3.3 — Entropy-stable HLLC-ES flux (Chandrashekar 2013)
// =============================================================================
// Reference: Chandrashekar, "Kinetic Energy Preserving and Entropy Stable
//            Finite Volume Schemes", Comm. Comput. Phys. 14(5), 2013.
//
// Physical basis:
//   Entropy variable: η = -ρ·s/(γ-1) where s = ln(p·ρ^{-γ}).
//   The entropy-CONSERVATIVE flux F_EC satisfies (w_R-w_L)·F_EC = ψ_R-ψ_L
//   exactly, where w = ∂η/∂Q are the entropy variables.
//   Adding the non-negative LF dissipation −(λ_max/2)·ΔQ converts this
//   to an entropy-STABLE flux: the discrete entropy inequality holds
//   cell-by-cell without any additional limiting.
//
// Key quantities:
//   β      = ρ/(2p)  (proportional to inverse temperature: β=1/(2RT))
//   ρ_ln   = log-mean(ρ_L, ρ_R)
//   β_ln   = log-mean(β_L, β_R)
//   p̂      = ρ̄/(2β̄)  with ρ̄, β̄ = arithmetic means  (Eq. 3.10)
//   Ĥ      = 1/(2(γ-1)β_ln) + ½(ū²+v̄²+w̄²) + p̂/ρ_ln

static inline double sq(double x) noexcept { return x * x; }

// Numerically stable log-mean using Ismail-Roe series near a≈b.
// For |u|² < 1e-4 the Taylor series  F = 1 + u²/3 + u⁴/5 + u⁶/7 + ...
// (where u=(a-b)/(a+b)) is used; otherwise the exact log formula.
static inline double log_mean(double a, double b) noexcept {
    const double xi = a / b;
    const double f  = (xi - 1.0) / (xi + 1.0);
    const double u2 = f * f;
    const double F  = (u2 < 1.0e-4)
                    ? 1.0 + u2 * (1.0/3.0 + u2 * (1.0/5.0 + u2 / 7.0))
                    : std::log(xi) / (2.0 * f);
    return (a + b) / (2.0 * F);
}

std::array<double,NVAR> hllc_es_flux(const Prim& L, const Prim& R, int axis) noexcept
{
    // Arithmetic means of primitive quantities
    const double rho_a  = 0.5*(L.rho + R.rho);
    const double u_a    = 0.5*(L.u   + R.u  );
    const double v_a    = 0.5*(L.v   + R.v  );
    const double w_a    = 0.5*(L.w   + R.w  );
    const double beta_L = L.rho / (2.0*L.p);
    const double beta_R = R.rho / (2.0*R.p);
    const double beta_a = 0.5*(beta_L + beta_R);

    // Log-mean quantities (entropy-variable averages)
    const double rho_ln  = log_mean(L.rho,  R.rho );
    const double beta_ln = log_mean(beta_L, beta_R);

    // Face pressure (Chandrashekar Eq. 3.10): ρ̄/(2β̄)
    const double p_hat = rho_a / (2.0 * beta_a);

    // Normal velocity
    const double un_L = (axis==0)?L.u:(axis==1)?L.v:L.w;
    const double un_R = (axis==0)?R.u:(axis==1)?R.v:R.w;
    const double un_a = 0.5*(un_L + un_R);

    const double mass = rho_ln * un_a;

    // Face total enthalpy: Ĥ = 1/(2(γ-1)β_ln) + KE_hat + p̂/ρ_ln
    const double KE_hat = 0.5*(u_a*u_a + v_a*v_a + w_a*w_a);
    const double H_hat  = 1.0/(2.0*(GAMMA-1.0)*beta_ln) + KE_hat + p_hat/rho_ln;

    // Entropy-conservative flux (Chandrashekar Eq. 3.10)
    std::array<double,NVAR> F_EC;
    F_EC[0] = mass;
    F_EC[1] = mass*u_a + (axis==0 ? p_hat : 0.0);
    F_EC[2] = mass*v_a + (axis==1 ? p_hat : 0.0);
    F_EC[3] = mass*w_a + (axis==2 ? p_hat : 0.0);
    F_EC[4] = mass*H_hat;

    // Lax-Friedrichs scalar dissipation: F_ES = F_EC − (λ_max/2)·ΔQ
    const double lam = std::max(std::abs(un_L)+L.c, std::abs(un_R)+R.c);
    const double E_L = L.p/(GAMMA-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R = R.p/(GAMMA-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);

    std::array<double,NVAR> F_ES;
    F_ES[0] = F_EC[0] - 0.5*lam*(R.rho       - L.rho      );
    F_ES[1] = F_EC[1] - 0.5*lam*(R.rho*R.u   - L.rho*L.u  );
    F_ES[2] = F_EC[2] - 0.5*lam*(R.rho*R.v   - L.rho*L.v  );
    F_ES[3] = F_EC[3] - 0.5*lam*(R.rho*R.w   - L.rho*L.w  );
    F_ES[4] = F_EC[4] - 0.5*lam*(E_R         - E_L         );
    return F_ES;
}

// =============================================================================
// P2.3 — Primitive variable and viscosity caches
// =============================================================================
// Pre-compute all NCELL primitive states (and Sutherland µ) once per
// compute_rhs call.  Interior + ghost cells filled → stencil = pure lookup.
static void fill_prim_cache(const CellBlock& blk, Prim* pc) noexcept {
    // P4.2: tile loop — 5 aligned 64-byte loads per tile, enables SIMD EOS.
    for (int t = 0; t < CellBlock::NTILE; ++t) {
        const double* rho_p  = blk.Q[0].tile_ptr(t);
        const double* rhou_p = blk.Q[1].tile_ptr(t);
        const double* rhov_p = blk.Q[2].tile_ptr(t);
        const double* rhow_p = blk.Q[3].tile_ptr(t);
        const double* E_p    = blk.Q[4].tile_ptr(t);
        Prim* pc_t = pc + t * CellBlock::W;
        for (int lane = 0; lane < CellBlock::W; ++lane)
            pc_t[lane] = eos_cons_to_prim(rho_p[lane], rhou_p[lane],
                                           rhov_p[lane], rhow_p[lane], E_p[lane]);
    }
}

// B5: µ cache — one Sutherland call per cell, shared across viscous operator.
static void fill_mu_cache(const Prim* pc, double* mu_arr) noexcept {
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const int idx = cell_idx(i,j,k);
        mu_arr[idx] = sutherland(pc[idx].T);
    }
}

// =============================================================================
// P3.2 — KE-consistent (skew-symmetric) flux  (Pirozzoli 2011)
// =============================================================================
// Physical basis:
//   The skew-symmetric / KE-consistent form splits the convective derivative
//   symmetrically between divergence and advective form.  The resulting flux
//   conserves discrete kinetic energy (dKE/dt = −work done by pressure)
//   to machine precision on smooth, periodic grids, eliminating the spurious
//   KE production that plagues standard upwind schemes in DNS/LES.
//
// Flux components (all arithmetic means, no upwinding):
//   F[0] = ρ̄ · ū_n
//   F[mom] = ρ̄ · ū_n · ū_i + p̄ · δ_{in}
//   F[4]   = ρ̄ · ū_n · H̄   where H̄ = ½(H_L + H_R), H = (E+p)/ρ
// Runtime fallback — used by undo_cf_face_flux / accumulate_fine_fluxes
static std::array<double,NVAR> kep_flux(const Prim& L, const Prim& R,
                                         int axis) noexcept
{
    const double u_a   = 0.5*(L.u   + R.u  );
    const double v_a   = 0.5*(L.v   + R.v  );
    const double w_a   = 0.5*(L.w   + R.w  );
    const double p_a   = 0.5*(L.p   + R.p  );
    const double E_L   = L.p/(GAMMA-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R   = R.p/(GAMMA-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    const double H_L   = (E_L + L.p) / L.rho;
    const double H_R   = (E_R + R.p) / R.rho;
    const double H_a   = 0.5*(H_L + H_R);
    // P13.2: FDKEC mass flux (Subbareddy & Candler 2009):
    //   mass = ½(ρL·u_nL + ρR·u_nR)  vs  KG: ρ_avg·u_n_avg
    const double un_L  = (axis==0)?L.u:(axis==1)?L.v:L.w;
    const double un_R  = (axis==0)?R.u:(axis==1)?R.v:R.w;
    const double mass  = 0.5*(L.rho*un_L + R.rho*un_R);

    std::array<double,NVAR> F;
    F[0] = mass;
    F[1] = mass*u_a + (axis==0 ? p_a : 0.0);
    F[2] = mass*v_a + (axis==1 ? p_a : 0.0);
    F[3] = mass*w_a + (axis==2 ? p_a : 0.0);
    F[4] = mass*H_a;
    return F;
}

// P13.1 stage 3 — compile-time axis: dead branches eliminated by constexpr if
template<Axis DIR>
static std::array<double,NVAR> kep_flux_t(const Prim& L, const Prim& R) noexcept {
    const double u_a   = 0.5*(L.u   + R.u  );
    const double v_a   = 0.5*(L.v   + R.v  );
    const double w_a   = 0.5*(L.w   + R.w  );
    const double p_a   = 0.5*(L.p   + R.p  );
    const double E_L   = L.p/(GAMMA-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R   = R.p/(GAMMA-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
    const double H_a   = 0.5*((E_L+L.p)/L.rho + (E_R+R.p)/R.rho);

    // P13.2: FDKEC mass flux (Subbareddy & Candler 2009)
    const double un_L = (DIR==Axis::X) ? L.u : (DIR==Axis::Y) ? L.v : L.w;
    const double un_R = (DIR==Axis::X) ? R.u : (DIR==Axis::Y) ? R.v : R.w;
    const double mass = 0.5*(L.rho*un_L + R.rho*un_R);

    std::array<double,NVAR> F;
    F[0] = mass;
    F[1] = mass*u_a + (DIR==Axis::X ? p_a : 0.0);
    F[2] = mass*v_a + (DIR==Axis::Y ? p_a : 0.0);
    F[3] = mass*w_a + (DIR==Axis::Z ? p_a : 0.0);
    F[4] = mass*H_a;
    return F;
}

// =============================================================================
// P3.2 — Combined shock sensor: Ducros (1999) + pressure-ratio
// =============================================================================
// Physical basis:
//   Φ_vel = (div u)² / ((div u)² + |curl u|² + ε)   — Ducros velocity sensor
//   Φ_p   = |Δp|_max / (p_local + ε)  smoothly blended [0.1→0.2] → [0→1]
//   Φ     = max(Φ_vel, Φ_p)
//
//   Φ_vel: fires near compressive shocks with non-zero velocity.
//   Φ_p:   fires near stationary pressure discontinuities (e.g., t=0 Riemann IC),
//          inactive for smooth flows (TGV: Φ_p ~ 0.07%, threshold 10%).
//
// Configurable pressure-sensor thresholds (P11.6) — set via set_ducros_thresholds().
static double g_ducros_p_thr = 0.1;  // |Δp|/p below this → phi_p = 0
static double g_ducros_blend = 0.1;  // linear blend width above threshold

void set_ducros_thresholds(double p_threshold, double blend_width) noexcept {
    g_ducros_p_thr = p_threshold;
    g_ducros_blend = (blend_width > 1e-30) ? blend_width : 1e-30;
}

// Computed for cells in [1, NB2-2] per axis.
// Output range: duc[cell_idx(i,j,k)] ∈ [0,1] for all valid cells.
static void fill_ducros_cache(const Prim* pc, double* duc, double h) noexcept
{
    constexpr double eps_duc = 1.0e-30;
    const double ih2 = 0.5 / h;   // 1/(2h)

    // Zero all cells first (covers absolute ghost boundary 0 and NB2-1)
    std::fill(duc, duc + NCELL, 0.0);

    // Compute for all cells one layer inside the absolute boundary
    for (int k = 1; k < NB2-1; ++k)
    for (int j = 1; j < NB2-1; ++j)
    for (int i = 1; i < NB2-1; ++i) {
        // Central-difference velocity gradients (all 9 components)
        const double dudx = ih2*(pc[cell_idx(i+1,j,k)].u - pc[cell_idx(i-1,j,k)].u);
        const double dudy = ih2*(pc[cell_idx(i,j+1,k)].u - pc[cell_idx(i,j-1,k)].u);
        const double dudz = ih2*(pc[cell_idx(i,j,k+1)].u - pc[cell_idx(i,j,k-1)].u);
        const double dvdx = ih2*(pc[cell_idx(i+1,j,k)].v - pc[cell_idx(i-1,j,k)].v);
        const double dvdy = ih2*(pc[cell_idx(i,j+1,k)].v - pc[cell_idx(i,j-1,k)].v);
        const double dvdz = ih2*(pc[cell_idx(i,j,k+1)].v - pc[cell_idx(i,j,k-1)].v);
        const double dwdx = ih2*(pc[cell_idx(i+1,j,k)].w - pc[cell_idx(i-1,j,k)].w);
        const double dwdy = ih2*(pc[cell_idx(i,j+1,k)].w - pc[cell_idx(i,j-1,k)].w);
        const double dwdz = ih2*(pc[cell_idx(i,j,k+1)].w - pc[cell_idx(i,j,k-1)].w);

        // Dilatation (div u) and vorticity magnitude squared
        const double divu = dudx + dvdy + dwdz;
        const double ox   = dwdy - dvdz;   // ω_x = ∂w/∂y − ∂v/∂z
        const double oy   = dudz - dwdx;   // ω_y = ∂u/∂z − ∂w/∂x
        const double oz   = dvdx - dudy;   // ω_z = ∂v/∂x − ∂u/∂y
        const double d2   = divu*divu;
        const double c2   = ox*ox + oy*oy + oz*oz;
        const double phi_vel = d2 / (d2 + c2 + eps_duc);

        // Pressure-ratio sensor: catches stationary shocks/contact discontinuities
        // where the velocity-based Ducros term is zero (u=0 at t=0).
        // Fires when any neighbor has |Δp| / p_local > 0.1 (10% relative jump).
        // Inactive for smooth flows (TGV: ~0.07%) and smooth pressure variations.
        const double pC   = pc[cell_idx(i,j,k)].p;
        const double dpx  = std::max(std::abs(pc[cell_idx(i+1,j,k)].p - pC),
                                     std::abs(pc[cell_idx(i-1,j,k)].p - pC));
        const double dpy  = std::max(std::abs(pc[cell_idx(i,j+1,k)].p - pC),
                                     std::abs(pc[cell_idx(i,j-1,k)].p - pC));
        const double dpz  = std::max(std::abs(pc[cell_idx(i,j,k+1)].p - pC),
                                     std::abs(pc[cell_idx(i,j,k-1)].p - pC));
        const double phi_p = std::max({dpx, dpy, dpz}) / (pC + eps_duc);
        // Smooth blend: phi_p < thr → 0; phi_p > thr+width → 1; linear in between
        const double phi_p_clamped = std::min(1.0, std::max(0.0,
            (phi_p - g_ducros_p_thr) / g_ducros_blend));

        duc[cell_idx(i,j,k)] = std::max(phi_vel, phi_p_clamped);
    }
}

// =============================================================================
// P3.1 — WENO5-Z scalar reconstruction (Borges et al. 2008)
// =============================================================================
// Reconstructs left (vL) and right (vR) states at the face between v0 and vp1
// from the 6-cell stencil {vm2, vm1, v0, vp1, vp2, vp3}.
//
// Mathematical basis:
//   Three 3rd-order candidate polynomials per side, combined with non-linear
//   WENO-Z weights α_k = d_k·(1 + (τ₅/(β_k+ε))²), ε=1e-36.
//   τ₅ = |β₀−β₂| (Borges global smoothness indicator).  At smooth extrema
//   WENO-Z reduces the artificial dissipation of WENO-JS (Jiang-Shu 1996),
//   recovering close to the 5th-order optimal stencil weight.
//
// Optimal weights (Shu 2009 lecture notes):
//   Left:  d₀=1/10, d₁=3/5, d₂=3/10   (bias toward right-of-face)
//   Right: same values (bias toward left-of-face, mirrored stencil)
static void weno5z_scalar(double vm2, double vm1, double v0,
                           double vp1, double vp2, double vp3,
                           double& vL, double& vR) noexcept
{
    constexpr double eps = 1.0e-36;
    constexpr double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    // ── Left state (reconstructed from the left side of face v0|vp1) ─────────
    // Candidate 3rd-order polynomials (Shu 2009 Eq. 2.16–2.18)
    const double L0 = ( 2.0*vm2 -  7.0*vm1 + 11.0*v0 ) * (1.0/6.0);
    const double L1 = (     -vm1 +  5.0*v0  +  2.0*vp1) * (1.0/6.0);
    const double L2 = ( 2.0*v0  +  5.0*vp1  -      vp2) * (1.0/6.0);

    // Jiang-Shu smoothness indicators (Eq. 2.61–2.63)
    const double b0L = (13.0/12.0)*sq(vm2 - 2.0*vm1 + v0 )
                     +  (1.0/ 4.0)*sq(vm2 - 4.0*vm1 + 3.0*v0);
    const double b1L = (13.0/12.0)*sq(vm1 - 2.0*v0  + vp1)
                     +  (1.0/ 4.0)*sq(vm1 - vp1);
    const double b2L = (13.0/12.0)*sq(v0  - 2.0*vp1 + vp2)
                     +  (1.0/ 4.0)*sq(3.0*v0 - 4.0*vp1 + vp2);

    const double tau5L = std::abs(b0L - b2L);
    const double a0L = d0 * (1.0 + sq(tau5L / (b0L + eps)));
    const double a1L = d1 * (1.0 + sq(tau5L / (b1L + eps)));
    const double a2L = d2 * (1.0 + sq(tau5L / (b2L + eps)));
    vL = (a0L*L0 + a1L*L1 + a2L*L2) / (a0L + a1L + a2L);

    // ── Right state (reconstructed from the right side, mirrored stencil) ────
    const double R0 = ( 2.0*vp3 -  7.0*vp2 + 11.0*vp1) * (1.0/6.0);
    const double R1 = (     -vp2 +  5.0*vp1 +  2.0*v0 ) * (1.0/6.0);
    const double R2 = ( 2.0*vp1 +  5.0*v0   -      vm1) * (1.0/6.0);

    const double b0R = (13.0/12.0)*sq(vp1  - 2.0*vp2 + vp3)
                     +  (1.0/ 4.0)*sq(3.0*vp1 - 4.0*vp2 + vp3);
    const double b1R = (13.0/12.0)*sq(v0   - 2.0*vp1 + vp2)
                     +  (1.0/ 4.0)*sq(v0 - vp2);
    const double b2R = (13.0/12.0)*sq(vm1  - 2.0*v0  + vp1)
                     +  (1.0/ 4.0)*sq(vm1 - 4.0*v0 + 3.0*vp1);

    const double tau5R = std::abs(b0R - b2R);
    const double a0R = d0 * (1.0 + sq(tau5R / (b0R + eps)));
    const double a1R = d1 * (1.0 + sq(tau5R / (b1R + eps)));
    const double a2R = d2 * (1.0 + sq(tau5R / (b2R + eps)));
    vR = (a0R*R0 + a1R*R1 + a2R*R2) / (a0R + a1R + a2R);
}

// =============================================================================
// P3.1 — WENO5 face reconstruction with Roe characteristic decomposition
// =============================================================================
// Reconstructs left primitive state qL and right primitive state qR at the
// face between cells (i,j,k) and (i+1,j,k) [axis=0] etc.
//
// Mathematical basis:
//   1. Convert 6-cell stencil to conservative variables Q[m], m=0..5.
//   2. Compute Roe-averaged state at face: ρ̃, ũ, H̃, c̃.
//   3. Left eigenvectors L_k (rows of R^{-1}) project each Q[m] to the
//      k-th characteristic variable  w_{k,m} = L_k · Q[m].
//   4. Apply weno5z_scalar independently to each characteristic sequence.
//   5. Back-project with right eigenvectors R_k (columns):
//      Q_face = Σ_k w_k * r_k.
//   6. Convert to primitive; clamp ρ,p > 0 for robustness.
//
// PRECONDITION: full 6-cell stencil must be in-bounds.
//   For NG=2: interior faces i ∈ [ilo(), ihi()-1] per axis only.
//   Ghost faces (i=ilo()-1, i=ihi()) must use PCM fallback.
//
// Roe eigenvector layout (axis-general):
//   k=0: acoustic  (u_n - c)     right: [1, un-c, ut1, ut2, H-un*c]^T
//   k=1: entropy wave (u_n)      right: [1, un,   ut1, ut2, KE    ]^T
//   k=2: shear 1  (u_n)          right: [0, 0,    1,   0,   ut1   ]^T
//   k=3: shear 2  (u_n)          right: [0, 0,    0,   1,   ut2   ]^T
//   k=4: acoustic  (u_n + c)     right: [1, un+c, ut1, ut2, H+un*c]^T
//   (ut1/ut2 are the tangential velocity components for the given axis)
static void weno5_face(const Prim* pc, int i, int j, int k, int axis,
                        Prim& qL_out, Prim& qR_out) noexcept
{
    // ── Stencil index helper: position offset d along axis from cell (i,j,k) ─
    auto idx_at = [&](int d) -> int {
        if (axis == 0) return cell_idx(i+d, j, k);
        if (axis == 1) return cell_idx(i, j+d, k);
        return                cell_idx(i, j, k+d);
    };

    // ── Convert 6 prim states to conservative Q[m][v], m∈[0,5] ──────────────
    // m=0 → cell i-2,  m=2 → cell i,  m=3 → cell i+1,  m=5 → cell i+3
    double Q[6][NVAR];
    for (int m = 0; m < 6; ++m) {
        const Prim& p = pc[idx_at(m - 2)];
        Q[m][0] = p.rho;
        Q[m][1] = p.rho * p.u;
        Q[m][2] = p.rho * p.v;
        Q[m][3] = p.rho * p.w;
        Q[m][4] = p.p/(GAMMA-1.0) + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
    }

    // ── Roe-averaged state between cells i (m=2) and i+1 (m=3) ──────────────
    const Prim& pL = pc[idx_at(0)];
    const Prim& pR = pc[idx_at(1)];
    const double sqL   = std::sqrt(pL.rho);
    const double sqR   = std::sqrt(pR.rho);
    const double denom = sqL + sqR;
    const double u_roe = (sqL*pL.u + sqR*pR.u) / denom;
    const double v_roe = (sqL*pL.v + sqR*pR.v) / denom;
    const double w_roe = (sqL*pL.w + sqR*pR.w) / denom;
    const double HL    = (Q[2][4] + pL.p) / pL.rho;
    const double HR    = (Q[3][4] + pR.p) / pR.rho;
    const double H_roe = (sqL*HL + sqR*HR) / denom;
    const double KE    = 0.5*(u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
    const double c2    = std::max((GAMMA-1.0)*(H_roe - KE), 1.0e-300);
    const double c_roe = std::sqrt(c2);

    // Normal and tangential Roe velocities + conservative-array index mapping
    //   axis=0: normal=x(idx 1), t1=y(idx 2), t2=z(idx 3)
    //   axis=1: normal=y(idx 2), t1=x(idx 1), t2=z(idx 3)
    //   axis=2: normal=z(idx 3), t1=x(idx 1), t2=y(idx 2)
    const double un  = (axis==0)?u_roe:(axis==1)?v_roe:w_roe;
    const double ut1 = (axis==0)?v_roe:(axis==1)?u_roe:u_roe;
    const double ut2 = (axis==0)?w_roe:(axis==1)?w_roe:v_roe;
    const int    n_idx  = 1 + axis;
    const int    t1_idx = (axis == 0) ? 2 : 1;
    const int    t2_idx = (axis == 2) ? 2 : 3;

    const double b  = (GAMMA-1.0) / c2;   // (γ-1)/c²
    const double b2 = b * KE;             // (γ-1)*KE/c²
    const double ioc = 1.0 / c_roe;       // 1/c

    // ── Project 6 conservative states to characteristic space ────────────────
    // w[k][m] = L_k · Q[m]  (left eigenvectors applied to each stencil point)
    //
    // Left eigenvectors (rows of R^{-1}), axis-general form:
    //   L₀ = ½·[b2+un/c, -(b*un+1/c) at n-slot, -b*ut1, -b*ut2, b]
    //   L₄ = ½·[b2-un/c, -(b*un-1/c) at n-slot, -b*ut1, -b*ut2, b]
    //
    // Expanding L₀·Q:
    //   = ½*(b2*ρ - b*(un*qn+ut1*qt1+ut2*qt2) + b*E)  [=inner/2]
    //   + ½*(un/c)*ρ - ½*(1/c)*qn
    //   = ½*(inner + ioc*(un*ρ - qn))         ← the un*ρ term is essential
    //
    // The term (un*ρ - qn)/c vanishes for the two Roe-reference cells (m=2,3)
    // only when the state is uniform; for non-uniform stencils it is non-zero
    // and must be included to ensure L·R = I exactly.
    //
    // L₁ = [(1-b2), b*un, b*ut1, b*ut2, -b]  → w[1] = (1-b2)*ρ + b*(un*qn+...) - b*E
    // L₂ = [-ut1, 0, 1@t1-slot, 0, 0]         → w[2] = -ut1*ρ + qt1
    // L₃ = [-ut2, 0, 0, 1@t2-slot, 0]         → w[3] = -ut2*ρ + qt2
    double W[5][6];
    for (int m = 0; m < 6; ++m) {
        const double rho = Q[m][0];
        const double qn  = Q[m][n_idx ];
        const double qt1 = Q[m][t1_idx];
        const double qt2 = Q[m][t2_idx];
        const double E   = Q[m][4];
        const double inner  = b2*rho - b*(un*qn + ut1*qt1 + ut2*qt2) + b*E;
        const double delta_n = ioc*(un*rho - qn);  // (un_roe*ρ_cell - qn_cell)/c
        W[0][m] = 0.5*(inner + delta_n);
        W[1][m] = (1.0 - b2)*rho + b*(un*qn + ut1*qt1 + ut2*qt2) - b*E;
        W[2][m] = -ut1*rho + qt1;
        W[3][m] = -ut2*rho + qt2;
        W[4][m] = 0.5*(inner - delta_n);
    }

    // ── Apply WENO5-Z to each characteristic variable independently ───────────
    double wL[5], wR[5];
    for (int kk = 0; kk < 5; ++kk)
        weno5z_scalar(W[kk][0], W[kk][1], W[kk][2],
                      W[kk][3], W[kk][4], W[kk][5],
                      wL[kk], wR[kk]);

    // ── Back-project with right eigenvectors to conservative space ────────────
    // Q = Σ_k w_k · r_k  where columns r_k are listed above.
    // Using the factored form (w014 = w[0]+w[1]+w[4]):
    //   Q[0]      = w014
    //   Q[n_idx]  = w014*un + (w[4]-w[0])*c
    //   Q[t1_idx] = w014*ut1 + w[2]
    //   Q[t2_idx] = w014*ut2 + w[3]
    //   Q[4]      = (w[0]+w[4])*H + (w[4]-w[0])*un*c + w[1]*KE + w[2]*ut1 + w[3]*ut2
    auto back_project = [&](const double w[5], double Qrec[NVAR]) {
        const double w014 = w[0] + w[1] + w[4];
        const double dw04 = w[4] - w[0];
        Qrec[0]      = w014;
        Qrec[n_idx]  = w014*un  + dw04*c_roe;
        Qrec[t1_idx] = w014*ut1 + w[2];
        Qrec[t2_idx] = w014*ut2 + w[3];
        Qrec[4]      = (w[0]+w[4])*H_roe + dw04*un*c_roe
                     + w[1]*KE + w[2]*ut1 + w[3]*ut2;
    };

    double QL[NVAR], QRv[NVAR];
    back_project(wL, QL);
    back_project(wR, QRv);

    // ── Convert conservative → primitive; fall back to cell-center if non-physical ──
    // A reconstructed state with ρ < 0 or p < 0 means the WENO5 polynomial
    // overshoots at a strong discontinuity. Rather than clamping to 1e-300
    // (which makes log_mean(β_L,β_R) degenerate in hllc_es_flux), fall back
    // to the cell-center primitive at the interface: pL for qL, pR for qR.
    auto safe_prim = [](const double Qc[NVAR], const Prim& fallback) -> Prim {
        const double rho = Qc[0];
        if (rho <= 0.0) return fallback;
        const double u   = Qc[1] / rho;
        const double v   = Qc[2] / rho;
        const double w   = Qc[3] / rho;
        const double p   = (GAMMA-1.0)*(Qc[4]-0.5*rho*(u*u+v*v+w*w));
        if (p <= 0.0) return fallback;
        Prim pr;
        pr.rho = rho;  pr.u = u;   pr.v = v;   pr.w = w;
        pr.p   = p;    pr.T = p/(rho*R_GAS);
        pr.c   = std::sqrt(GAMMA*p/rho);
        return pr;
    };

    qL_out = safe_prim(QL,  pL);
    qR_out = safe_prim(QRv, pR);
}

// P13.1 stage 3 — compile-time axis: idx_at + Roe-velocity selection via if constexpr
template<Axis DIR>
static void weno5_face_t(const Prim* pc, int i, int j, int k,
                         Prim& qL_out, Prim& qR_out) noexcept {
    // Stencil index: offset d along the compile-time axis
    auto idx_at = [&](int d) -> int {
        if constexpr (DIR == Axis::X) return cell_idx(i+d, j, k);
        if constexpr (DIR == Axis::Y) return cell_idx(i, j+d, k);
        return                              cell_idx(i, j, k+d);
    };

    double Q[6][NVAR];
    for (int m = 0; m < 6; ++m) {
        const Prim& p = pc[idx_at(m - 2)];
        Q[m][0] = p.rho;
        Q[m][1] = p.rho * p.u;
        Q[m][2] = p.rho * p.v;
        Q[m][3] = p.rho * p.w;
        Q[m][4] = p.p/(GAMMA-1.0) + 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
    }

    const Prim& pL = pc[idx_at(0)];
    const Prim& pR = pc[idx_at(1)];
    const double sqL   = std::sqrt(pL.rho);
    const double sqR   = std::sqrt(pR.rho);
    const double denom = sqL + sqR;
    const double u_roe = (sqL*pL.u + sqR*pR.u) / denom;
    const double v_roe = (sqL*pL.v + sqR*pR.v) / denom;
    const double w_roe = (sqL*pL.w + sqR*pR.w) / denom;
    const double HL    = (Q[2][4] + pL.p) / pL.rho;
    const double HR    = (Q[3][4] + pR.p) / pR.rho;
    const double H_roe = (sqL*HL + sqR*HR) / denom;
    const double KE    = 0.5*(u_roe*u_roe + v_roe*v_roe + w_roe*w_roe);
    const double c2    = std::max((GAMMA-1.0)*(H_roe - KE), 1.0e-300);
    const double c_roe = std::sqrt(c2);

    // Compile-time normal/tangential velocity and Q-array index selection
    constexpr int n_idx  = (DIR==Axis::X) ? 1 : (DIR==Axis::Y) ? 2 : 3;
    constexpr int t1_idx = (DIR==Axis::X) ? 2 : 1;
    constexpr int t2_idx = (DIR==Axis::Z) ? 2 : 3;
    const double un  = (DIR==Axis::X) ? u_roe : (DIR==Axis::Y) ? v_roe : w_roe;
    const double ut1 = (DIR==Axis::X) ? v_roe : u_roe;
    const double ut2 = (DIR==Axis::Z) ? v_roe : w_roe;

    const double b  = (GAMMA-1.0) / c2;
    const double b2 = b * KE;
    const double ioc = 1.0 / c_roe;

    double W[5][6];
    for (int m = 0; m < 6; ++m) {
        const double rho = Q[m][0];
        const double qn  = Q[m][n_idx ];
        const double qt1 = Q[m][t1_idx];
        const double qt2 = Q[m][t2_idx];
        const double E   = Q[m][4];
        const double inner   = b2*rho - b*(un*qn + ut1*qt1 + ut2*qt2) + b*E;
        const double delta_n = ioc*(un*rho - qn);
        W[0][m] = 0.5*(inner + delta_n);
        W[1][m] = (1.0 - b2)*rho + b*(un*qn + ut1*qt1 + ut2*qt2) - b*E;
        W[2][m] = -ut1*rho + qt1;
        W[3][m] = -ut2*rho + qt2;
        W[4][m] = 0.5*(inner - delta_n);
    }

    double wL[5], wR[5];
    for (int kk = 0; kk < 5; ++kk)
        weno5z_scalar(W[kk][0], W[kk][1], W[kk][2],
                      W[kk][3], W[kk][4], W[kk][5],
                      wL[kk], wR[kk]);

    auto back_project = [&](const double w[5], double Qrec[NVAR]) {
        const double w014 = w[0] + w[1] + w[4];
        const double dw04 = w[4] - w[0];
        Qrec[0]      = w014;
        Qrec[n_idx]  = w014*un  + dw04*c_roe;
        Qrec[t1_idx] = w014*ut1 + w[2];
        Qrec[t2_idx] = w014*ut2 + w[3];
        Qrec[4]      = (w[0]+w[4])*H_roe + dw04*un*c_roe
                     + w[1]*KE + w[2]*ut1 + w[3]*ut2;
    };

    double QL[NVAR], QRv[NVAR];
    back_project(wL, QL);
    back_project(wR, QRv);

    auto safe_prim = [](const double Qc[NVAR], const Prim& fallback) -> Prim {
        const double rho = Qc[0];
        if (rho <= 0.0) return fallback;
        const double u   = Qc[1] / rho;
        const double v   = Qc[2] / rho;
        const double w   = Qc[3] / rho;
        const double p   = (GAMMA-1.0)*(Qc[4] - 0.5*rho*(u*u+v*v+w*w));
        if (p <= 0.0) return fallback;
        Prim q; q.rho=rho; q.u=u; q.v=v; q.w=w; q.p=p;
        q.T=p/(rho*R_GAS); q.c=std::sqrt(GAMMA*p/rho);
        return q;
    };

    qL_out = safe_prim(QL,  pL);
    qR_out = safe_prim(QRv, pR);
}

// =============================================================================
// P2.2/P3.2 — Convective RHS: face-centred hybrid loop
// =============================================================================
// Iterates over faces rather than cells.  For axis d, there are
// (NB+1)*NB*NB interior+boundary faces; each evaluated exactly once.
//
// Hybrid scheme (P3.2):
//   θ = max(Ducros_L, Ducros_R) ∈ [0,1]
//   F = (1−θ)·F_KEP + θ·F_shock
//
//   F_KEP  : Pirozzoli (2011) KE-preserving flux (zero added dissipation)
//   F_shock: WENO5+HLLC-ES for interior faces; PCM+HLLC-ES for ghost faces
//
// Fast path (θ < 1e-8): smooth/turbulent region → pure KEP, skip WENO5.
// Shock path (θ ≥ 1e-8): compute WENO5 and blend with KEP.
//
// Flux sign: F positive leaving the left cell:
//   rhs[left] -= ih*F,  rhs[right] += ih*F  (telescopes for conservation)
static void convective_rhs_impl(const Prim* pc, const double* duc,
                                 CellBlock& rhs, double h) noexcept
{
    const double ih = 1.0 / h;
    constexpr double kep_threshold = 1.0e-8;  // below → pure KEP, no WENO5

    // Helper: blend two flux arrays
    auto blend = [](const std::array<double,NVAR>& Fk,
                    const std::array<double,NVAR>& Fs,
                    double th) -> std::array<double,NVAR>
    {
        std::array<double,NVAR> F;
        const double om = 1.0 - th;
        for (int v = 0; v < NVAR; ++v) F[v] = om*Fk[v] + th*Fs[v];
        return F;
    };

    // Helper: detect no-slip wall ghost face.
    //
    // Discriminator: pL.p == pR.p (exact float equality) as primary check, plus
    // all three velocity components anti-symmetric.
    //
    // Rationale:
    //   Wall ghost (fill_channel_ghosts): Q[4]_g = Q[4]_i and Q[0]_g = Q[0]_i by
    //   direct assignment; KE_g = KE_i (since |u_g|=|u_i|); so p_g = p_i EXACTLY.
    //   AMR/periodic ghosts: values from coarse interpolation or periodic neighbour —
    //   pressure differs in general (only coincidentally equal, e.g. uniform flow).
    //   Adding all-velocity anti-symmetry excludes uniform-flow edge cases.
    auto is_wall_face = [](const Prim& pL, const Prim& pR, bool bnd) -> bool {
        if (!bnd) return false;
        if (pL.p != pR.p) return false;    // AMR/periodic: pressure differs
        auto antisym = [](double a, double b) -> bool {
            const double s = std::fabs(a) + std::fabs(b) + 1e-300;
            return std::fabs(a + b) < 1e-8 * s;
        };
        return antisym(pL.u, pR.u) && antisym(pL.v, pR.v) && antisym(pL.w, pR.w);
    };

    // ── X-direction ────────────────────────────────────────────────────────────
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo()-1; i <= ihi(); ++i) {
        const Prim& pL = pc[cell_idx(i,  j,k)];
        const Prim& pR = pc[cell_idx(i+1,j,k)];
        const double theta = std::max(duc[cell_idx(i,  j,k)],
                                      duc[cell_idx(i+1,j,k)]);
        // Block-boundary faces (ghost↔interior) must use hllc_es so that
        // undo_cf_face_flux and accumulate_cf_fine_fluxes compute the same flux,
        // giving exact Berger-Colella reflux cancellation at C/F interfaces.
        const bool is_bnd = (i < ilo() || i+1 > ihi());
        std::array<double,NVAR> F;
        if (!is_bnd && theta < kep_threshold) {
            // Pure KEP: smooth/turbulent interior region
            F = kep_flux_t<Axis::X>(pL, pR);
        } else {
            // Shock region, WENO5+ES interior, or block-boundary face (hllc_es)
            const auto Fk = kep_flux_t<Axis::X>(pL, pR);
            std::array<double,NVAR> Fs;
            if (!is_bnd && i >= ilo() && i < ihi()) {
                Prim qL, qR;
                weno5_face_t<Axis::X>(pc, i, j, k, qL, qR);
                Fs = hllc_es_flux_t<Axis::X>(qL, qR);
            } else if (is_wall_face(pL, pR, is_bnd)) {
                Fs = Fk;  // wall: KEP = {0, p, 0, 0, 0}, no LF tangential drain
            } else {
                Fs = hllc_es_flux_t<Axis::X>(pL, pR);
            }
            F = blend(Fk, Fs, is_bnd ? 1.0 : theta);
        }
        if (i >= ilo())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,  j,k)] -= ih * F[v];
        if (i+1 <= ihi())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i+1,j,k)] += ih * F[v];
    }

    // ── Y-direction ────────────────────────────────────────────────────────────
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo()-1; j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const Prim& pL = pc[cell_idx(i,j,  k)];
        const Prim& pR = pc[cell_idx(i,j+1,k)];
        const double theta = std::max(duc[cell_idx(i,j,  k)],
                                      duc[cell_idx(i,j+1,k)]);
        const bool is_bnd = (j < ilo() || j+1 > ihi());
        std::array<double,NVAR> F;
        if (!is_bnd && theta < kep_threshold) {
            F = kep_flux_t<Axis::Y>(pL, pR);
        } else {
            const auto Fk = kep_flux_t<Axis::Y>(pL, pR);
            std::array<double,NVAR> Fs;
            if (!is_bnd && j >= ilo() && j < ihi()) {
                Prim qL, qR;
                weno5_face_t<Axis::Y>(pc, i, j, k, qL, qR);
                Fs = hllc_es_flux_t<Axis::Y>(qL, qR);
            } else if (is_wall_face(pL, pR, is_bnd)) {
                Fs = Fk;  // wall: KEP = {0, 0, p, 0, 0}, no LF tangential drain
            } else {
                Fs = hllc_es_flux_t<Axis::Y>(pL, pR);
            }
            F = blend(Fk, Fs, is_bnd ? 1.0 : theta);
        }
        if (j >= ilo())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j,  k)] -= ih * F[v];
        if (j+1 <= ihi())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j+1,k)] += ih * F[v];
    }

    // ── Z-direction ────────────────────────────────────────────────────────────
    for (int k = ilo()-1; k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const Prim& pL = pc[cell_idx(i,j,k  )];
        const Prim& pR = pc[cell_idx(i,j,k+1)];
        const double theta = std::max(duc[cell_idx(i,j,k  )],
                                      duc[cell_idx(i,j,k+1)]);
        const bool is_bnd = (k < ilo() || k+1 > ihi());
        std::array<double,NVAR> F;
        if (!is_bnd && theta < kep_threshold) {
            F = kep_flux_t<Axis::Z>(pL, pR);
        } else {
            const auto Fk = kep_flux_t<Axis::Z>(pL, pR);
            std::array<double,NVAR> Fs;
            if (!is_bnd && k >= ilo() && k < ihi()) {
                Prim qL, qR;
                weno5_face_t<Axis::Z>(pc, i, j, k, qL, qR);
                Fs = hllc_es_flux_t<Axis::Z>(qL, qR);
            } else if (is_wall_face(pL, pR, is_bnd)) {
                Fs = Fk;  // wall: KEP = {0, 0, 0, p, 0}, no LF tangential drain
            } else {
                Fs = hllc_es_flux_t<Axis::Z>(pL, pR);
            }
            F = blend(Fk, Fs, is_bnd ? 1.0 : theta);
        }
        if (k >= ilo())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j,k  )] -= ih * F[v];
        if (k+1 <= ihi())
            for (int v = 0; v < NVAR; ++v) rhs.Q[v][cell_idx(i,j,k+1)] += ih * F[v];
    }
}

// =============================================================================
// B5 — Viscous RHS: face-averaged µ, conservative divergence form
// =============================================================================
// Replaces the old constant-µ expansion  a_i = µ·(Δu_i + ⅓ ∂(div u)/∂x_i).
// That form is only equivalent to div(τ) when µ is spatially constant; it
// misses the ∇µ·(∂u/∂x + ...) term at AMR coarse-fine interfaces.
//
// New form: for each cell (i,j,k), compute the Conservative divergence
//
//   a_x = h⁻¹ [(τ_xx|_{i+½} - τ_xx|_{i-½})
//              + (τ_yx|_{j+½} - τ_yx|_{j-½})
//              + (τ_zx|_{k+½} - τ_zx|_{k-½})]
//
// where τ_ab|_{face} = µ_face · (∂u_a/∂x_b + ∂u_b/∂x_a - δ_ab⅔ div u)|_{face}
// and µ_face = ½(µ_left + µ_right)  (arithmetic average).
//
// Velocity gradients at face i+½:
//   normal:    (u[i+1] - u[i]) / h           (1st-order forward, 2nd-order at face)
//   tangential: average of centred differences on both sides of the face, e.g.
//               du/dy|_{i+½} = ((u[i+1,j+1]-u[i+1,j-1]) + (u[i,j+1]-u[i,j-1])) / 4h
//
// Energy: cell-centred τ:S and κ·Δ(T) are unchanged (O(h²), conserved on
// periodic domains to machine precision as confirmed by T04/S07).
// =============================================================================
static constexpr double CP = CPU_CP;

static void viscous_rhs_impl(const Prim* pc, const double* mu_arr,
                              CellBlock& rhs, double h) noexcept
{
    const double ih      = 1.0 / h;
    const double ih2     = ih * ih;
    const double ih_half = 0.5 * ih;   // 1/(2h): face tangential gradients

    // Velocity and µ accessors from caches
    auto U  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].u; };
    auto V  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].v; };
    auto W  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].w; };
    auto Tf = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].T; };
    auto MU = [&](int ii,int jj,int kk){ return mu_arr[cell_idx(ii,jj,kk)]; };

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {

        // ── Face-averaged µ at the 6 faces ──────────────────────────────────
        double mu_xp = 0.5*(MU(i,j,k) + MU(i+1,j,  k  ));
        double mu_xm = 0.5*(MU(i,j,k) + MU(i-1,j,  k  ));
        double mu_yp = 0.5*(MU(i,j,k) + MU(i,  j+1,k  ));
        double mu_ym = 0.5*(MU(i,j,k) + MU(i,  j-1,k  ));
        double mu_zp = 0.5*(MU(i,j,k) + MU(i,  j,  k+1));
        double mu_zm = 0.5*(MU(i,j,k) + MU(i,  j,  k-1));

        // ── Velocity gradients at x-faces (i±½) ─────────────────────────────
        // x+½
        double dudx_xp = ih*(U(i+1,j,k)-U(i,j,k));
        double dvdx_xp = ih*(V(i+1,j,k)-V(i,j,k));
        double dwdx_xp = ih*(W(i+1,j,k)-W(i,j,k));
        double dudy_xp = ih_half*(U(i+1,j+1,k)-U(i+1,j-1,k) + U(i,j+1,k)-U(i,j-1,k));
        double dudz_xp = ih_half*(U(i+1,j,k+1)-U(i+1,j,k-1) + U(i,j,k+1)-U(i,j,k-1));
        double dvdy_xp = ih_half*(V(i+1,j+1,k)-V(i+1,j-1,k) + V(i,j+1,k)-V(i,j-1,k));
        double dwdz_xp = ih_half*(W(i+1,j,k+1)-W(i+1,j,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_xp = dudx_xp + dvdy_xp + dwdz_xp;
        // x-½
        double dudx_xm = ih*(U(i,j,k)-U(i-1,j,k));
        double dvdx_xm = ih*(V(i,j,k)-V(i-1,j,k));
        double dwdx_xm = ih*(W(i,j,k)-W(i-1,j,k));
        double dudy_xm = ih_half*(U(i-1,j+1,k)-U(i-1,j-1,k) + U(i,j+1,k)-U(i,j-1,k));
        double dudz_xm = ih_half*(U(i-1,j,k+1)-U(i-1,j,k-1) + U(i,j,k+1)-U(i,j,k-1));
        double dvdy_xm = ih_half*(V(i-1,j+1,k)-V(i-1,j-1,k) + V(i,j+1,k)-V(i,j-1,k));
        double dwdz_xm = ih_half*(W(i-1,j,k+1)-W(i-1,j,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_xm = dudx_xm + dvdy_xm + dwdz_xm;

        // ── Velocity gradients at y-faces (j±½) ─────────────────────────────
        // j+½
        double dudy_yp = ih*(U(i,j+1,k)-U(i,j,k));
        double dvdx_yp = ih_half*(V(i+1,j+1,k)-V(i-1,j+1,k) + V(i+1,j,k)-V(i-1,j,k));
        double dvdy_yp = ih*(V(i,j+1,k)-V(i,j,k));
        double dvdz_yp = ih_half*(V(i,j+1,k+1)-V(i,j+1,k-1) + V(i,j,k+1)-V(i,j,k-1));
        double dwdy_yp = ih*(W(i,j+1,k)-W(i,j,k));
        double dudx_yp = ih_half*(U(i+1,j+1,k)-U(i-1,j+1,k) + U(i+1,j,k)-U(i-1,j,k));
        double dwdz_yp = ih_half*(W(i,j+1,k+1)-W(i,j+1,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_yp = dudx_yp + dvdy_yp + dwdz_yp;
        // j-½
        double dudy_ym = ih*(U(i,j,k)-U(i,j-1,k));
        double dvdx_ym = ih_half*(V(i+1,j-1,k)-V(i-1,j-1,k) + V(i+1,j,k)-V(i-1,j,k));
        double dvdy_ym = ih*(V(i,j,k)-V(i,j-1,k));
        double dvdz_ym = ih_half*(V(i,j-1,k+1)-V(i,j-1,k-1) + V(i,j,k+1)-V(i,j,k-1));
        double dwdy_ym = ih*(W(i,j,k)-W(i,j-1,k));
        double dudx_ym = ih_half*(U(i+1,j-1,k)-U(i-1,j-1,k) + U(i+1,j,k)-U(i-1,j,k));
        double dwdz_ym = ih_half*(W(i,j-1,k+1)-W(i,j-1,k-1) + W(i,j,k+1)-W(i,j,k-1));
        double divu_ym = dudx_ym + dvdy_ym + dwdz_ym;

        // ── Velocity gradients at z-faces (k±½) ─────────────────────────────
        // k+½
        double dudz_zp = ih*(U(i,j,k+1)-U(i,j,k));
        double dvdz_zp = ih*(V(i,j,k+1)-V(i,j,k));
        double dwdz_zp = ih*(W(i,j,k+1)-W(i,j,k));
        double dwdx_zp = ih_half*(W(i+1,j,k+1)-W(i-1,j,k+1) + W(i+1,j,k)-W(i-1,j,k));
        double dwdy_zp = ih_half*(W(i,j+1,k+1)-W(i,j-1,k+1) + W(i,j+1,k)-W(i,j-1,k));
        double dudx_zp = ih_half*(U(i+1,j,k+1)-U(i-1,j,k+1) + U(i+1,j,k)-U(i-1,j,k));
        double dvdy_zp = ih_half*(V(i,j+1,k+1)-V(i,j-1,k+1) + V(i,j+1,k)-V(i,j-1,k));
        double divu_zp = dudx_zp + dvdy_zp + dwdz_zp;
        // k-½
        double dudz_zm = ih*(U(i,j,k)-U(i,j,k-1));
        double dvdz_zm = ih*(V(i,j,k)-V(i,j,k-1));
        double dwdz_zm = ih*(W(i,j,k)-W(i,j,k-1));
        double dwdx_zm = ih_half*(W(i+1,j,k-1)-W(i-1,j,k-1) + W(i+1,j,k)-W(i-1,j,k));
        double dwdy_zm = ih_half*(W(i,j+1,k-1)-W(i,j-1,k-1) + W(i,j+1,k)-W(i,j-1,k));
        double dudx_zm = ih_half*(U(i+1,j,k-1)-U(i-1,j,k-1) + U(i+1,j,k)-U(i-1,j,k));
        double dvdy_zm = ih_half*(V(i,j+1,k-1)-V(i,j-1,k-1) + V(i,j+1,k)-V(i,j-1,k));
        double divu_zm = dudx_zm + dvdy_zm + dwdz_zm;

        // ── Face stresses τ_ab|_{face} = µ_face·(∂u_a/∂x_b + ∂u_b/∂x_a - δ·⅔ div u)
        // x-faces: components needed for ax(τ_xx), ay(τ_xy), az(τ_xz)
        double txx_xp = mu_xp*(2.0*dudx_xp - (2.0/3.0)*divu_xp);
        double txy_xp = mu_xp*(dudy_xp + dvdx_xp);
        double txz_xp = mu_xp*(dudz_xp + dwdx_xp);
        double txx_xm = mu_xm*(2.0*dudx_xm - (2.0/3.0)*divu_xm);
        double txy_xm = mu_xm*(dudy_xm + dvdx_xm);
        double txz_xm = mu_xm*(dudz_xm + dwdx_xm);

        // y-faces: τ_yx(=τ_xy), τ_yy, τ_yz
        double tyx_yp = mu_yp*(dudy_yp + dvdx_yp);
        double tyy_yp = mu_yp*(2.0*dvdy_yp - (2.0/3.0)*divu_yp);
        double tyz_yp = mu_yp*(dvdz_yp + dwdy_yp);
        double tyx_ym = mu_ym*(dudy_ym + dvdx_ym);
        double tyy_ym = mu_ym*(2.0*dvdy_ym - (2.0/3.0)*divu_ym);
        double tyz_ym = mu_ym*(dvdz_ym + dwdy_ym);

        // z-faces: τ_zx(=τ_xz), τ_zy(=τ_yz), τ_zz
        double tzx_zp = mu_zp*(dudz_zp + dwdx_zp);
        double tzy_zp = mu_zp*(dvdz_zp + dwdy_zp);
        double tzz_zp = mu_zp*(2.0*dwdz_zp - (2.0/3.0)*divu_zp);
        double tzx_zm = mu_zm*(dudz_zm + dwdx_zm);
        double tzy_zm = mu_zm*(dvdz_zm + dwdy_zm);
        double tzz_zm = mu_zm*(2.0*dwdz_zm - (2.0/3.0)*divu_zm);

        // ── Conservative momentum divergences ────────────────────────────────
        double ax = ih*((txx_xp-txx_xm) + (tyx_yp-tyx_ym) + (tzx_zp-tzx_zm));
        double ay = ih*((txy_xp-txy_xm) + (tyy_yp-tyy_ym) + (tzy_zp-tzy_zm));
        double az = ih*((txz_xp-txz_xm) + (tyz_yp-tyz_ym) + (tzz_zp-tzz_zm));

        // ── Energy: conservative face-flux form  div(τ·u + κ∇T) ─────────────
        // Using the already-computed face stresses τ_ab|face and face averages.
        // This form telescopes at any boundary type (periodic, wall, C/F) so that
        // total energy is conserved when viscous face fluxes vanish at boundaries.
        // F_visc_E_x|face = τxx*u_face + τxy*v_face + τxz*w_face + κ*dT/dx|face
        const Prim& c = pc[cell_idx(i,j,k)];
        auto UF=[&](int ii,int jj,int kk){return 0.5*(U(ii,jj,kk)+U(i,j,k));};
        auto VF=[&](int ii,int jj,int kk){return 0.5*(V(ii,jj,kk)+V(i,j,k));};
        auto WF=[&](int ii,int jj,int kk){return 0.5*(W(ii,jj,kk)+W(i,j,k));};
        // x-faces
        double kxp = mu_xp*CP/PR, kxm = mu_xm*CP/PR;
        double Fex_p = txx_xp*UF(i+1,j,k) + txy_xp*VF(i+1,j,k) + txz_xp*WF(i+1,j,k)
                     + kxp*ih*(Tf(i+1,j,k)-Tf(i,j,k));
        double Fex_m = txx_xm*UF(i-1,j,k) + txy_xm*VF(i-1,j,k) + txz_xm*WF(i-1,j,k)
                     + kxm*ih*(Tf(i,j,k)-Tf(i-1,j,k));
        // y-faces
        double kyp = mu_yp*CP/PR, kym = mu_ym*CP/PR;
        double Fey_p = tyx_yp*UF(i,j+1,k) + tyy_yp*VF(i,j+1,k) + tyz_yp*WF(i,j+1,k)
                     + kyp*ih*(Tf(i,j+1,k)-Tf(i,j,k));
        double Fey_m = tyx_ym*UF(i,j-1,k) + tyy_ym*VF(i,j-1,k) + tyz_ym*WF(i,j-1,k)
                     + kym*ih*(Tf(i,j,k)-Tf(i,j-1,k));
        // z-faces
        double kzp = mu_zp*CP/PR, kzm = mu_zm*CP/PR;
        double Fez_p = tzx_zp*UF(i,j,k+1) + tzy_zp*VF(i,j,k+1) + tzz_zp*WF(i,j,k+1)
                     + kzp*ih*(Tf(i,j,k+1)-Tf(i,j,k));
        double Fez_m = tzx_zm*UF(i,j,k-1) + tzy_zm*VF(i,j,k-1) + tzz_zm*WF(i,j,k-1)
                     + kzm*ih*(Tf(i,j,k)-Tf(i,j,k-1));

        int idx = cell_idx(i,j,k);
        rhs.Q[1][idx] += ax;
        rhs.Q[2][idx] += ay;
        rhs.Q[3][idx] += az;
        rhs.Q[4][idx] += ih*(Fex_p-Fex_m) + ih*(Fey_p-Fey_m) + ih*(Fez_p-Fez_m);
    }
}

// =============================================================================
// Public wrappers — preserve API for standalone test calls
// =============================================================================
void convective_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> duc;
    fill_prim_cache(blk, pc.data());
    fill_ducros_cache(pc.data(), duc.data(), blk.h);
    convective_rhs_impl(pc.data(), duc.data(), rhs_blk, blk.h);
}

void viscous_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> mu_arr;
    fill_prim_cache(blk, pc.data());
    fill_mu_cache(pc.data(), mu_arr.data());
    viscous_rhs_impl(pc.data(), mu_arr.data(), rhs_blk, blk.h);
}

// =============================================================================
// Full RHS — P2.3 + B5 + P3.2: prim, µ, and Ducros caches built once
// =============================================================================
void compute_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> mu_arr;
    static thread_local std::array<double, NCELL> duc;
    fill_prim_cache(blk, pc.data());
    fill_mu_cache(pc.data(), mu_arr.data());
    fill_ducros_cache(pc.data(), duc.data(), blk.h);

    for (int v = 0; v < NVAR; ++v)
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i)
            rhs_blk.Q[v][cell_idx(i,j,k)] = 0.0;

    convective_rhs_impl(pc.data(), duc.data(), rhs_blk, blk.h);
    viscous_rhs_impl   (pc.data(), mu_arr.data(),         rhs_blk, blk.h);
}

// =============================================================================
// Berger-Colella reflux: undo_cf_face_flux
// ─────────────────────────────────────────
// Undoes the coarse HLLC contribution at each CF face so that
// apply_flux_correction() can add the correct fine-side average.
// Net effect after correction:
//   dQ/dt += (1/h) * [F_fine_avg - F_coarse_own]
// =============================================================================
static void undo_cf_face_flux(const BlockTree& tree, int node_idx,
                               CellBlock& rhs) noexcept
{
    const auto& nd  = tree.nodes[node_idx];
    const auto& blk = *nd.block;
    const double ih = 1.0 / blk.h;

    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

    for (int d = 0; d < NFACES; ++d) {
        const int ni = nd.neighbours[d];
        if (ni < 0 || !tree.nodes[ni].has_block()) continue;
        if (tree.nodes[ni].level <= nd.level) continue;

        const int axis  = face_axis[d];
        const int delta = face_delta[d];
        const int bound = (delta > 0) ? ihi() : ilo();

        for (int b = ilo(); b <= ihi(); ++b)
        for (int a = ilo(); a <= ihi(); ++a) {
            int ci, cj, ck, gi, gj, gk;
            if (axis == 0) {
                ci=bound; cj=a;     ck=b;
                gi=bound+delta; gj=a; gk=b;
            } else if (axis == 1) {
                ci=a;     cj=bound; ck=b;
                gi=a; gj=bound+delta; gk=b;
            } else {
                ci=a;     cj=b;     ck=bound;
                gi=a; gj=b; gk=bound+delta;
            }

            Prim interior = blk.prim(ci, cj, ck);
            Prim ghost    = blk.prim(gi, gj, gk);

            std::array<double,5> F;
            if (delta > 0)
                F = hllc_es_flux(interior, ghost, axis);
            else
                F = hllc_es_flux(ghost, interior, axis);

            const double sign = (delta > 0) ? +1.0 : -1.0;
            const int idx = cell_idx(ci, cj, ck);
            for (int v = 0; v < NVAR; ++v)
                rhs.Q[v][idx] += sign * ih * F[v];
        }
    }
}

// =============================================================================
// Berger-Colella reflux: undo_cf_viscous_energy
// ─────────────────────────────────────────────
// Root cause of T10b energy drift: even with zero-gradient ghost fill, the
// TANGENTIAL viscous stress at a C/F face is non-zero.  Example for x+½ face:
//   dudy_xp = ih_half*(U(ghost,j+1)+U(interior,j+1)-U(ghost,j-1)-U(interior,j-1))
// With zero-grad ghost: U(ghost,j,k)=U(interior,j,k) → dudy_xp = ih*(U(i,j+1)-U(i,j-1)).
// This gives txy_xp ≠ 0 → non-zero energy flux txy*v_face at C/F.
// Neither the convective flux register nor undo_cf_face_flux corrects this.
//
// Fix: exactly reproduce the viscous energy flux at each C/F face using the
// same stencil as viscous_rhs_impl, then subtract it from rhs.Q[4].
// Called only when cf_coarse_zero_grad=true (both fine and coarse sub-steps).
//
// Sign: viscous_rhs_impl adds +delta*ih*Fvisc_E to the boundary cell.
//   For delta>0 (XP): rhs += +ih*Fex_p → subtract ih*Fex_p.
//   For delta<0 (XM): rhs += -ih*Fex_m → subtract -ih*Fex_m = add ih*Fex_m.
// Combined: rhs.Q[4] -= delta * ih * Fvisc_E_face.
// =============================================================================
static void undo_cf_viscous_energy(const BlockTree& tree, int node_idx,
                                    CellBlock& rhs) noexcept
{
    const auto& nd   = tree.nodes[node_idx];
    const auto& blk  = *nd.block;
    const double h   = blk.h;
    const double ih  = 1.0 / h;
    const double ihhalf = 0.5 * ih;

    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

    auto U = [&](int ii,int jj,int kk){ return blk.prim(ii,jj,kk).u; };
    auto V = [&](int ii,int jj,int kk){ return blk.prim(ii,jj,kk).v; };
    auto W = [&](int ii,int jj,int kk){ return blk.prim(ii,jj,kk).w; };

    for (int d = 0; d < NFACES; ++d) {
        const int ni = nd.neighbours[d];
        if (ni < 0 || !tree.nodes[ni].has_block()) continue;
        if (tree.nodes[ni].level == nd.level) continue;  // same-level: skip

        const int axis   = face_axis[d];
        const int delta  = face_delta[d];
        const double ns  = (double)delta;  // ±1: normal direction sign
        const int bound  = (delta > 0) ? ihi() : ilo();
        const int gbound = bound + delta;  // ghost cell index along normal

        for (int b = ilo(); b <= ihi(); ++b)
        for (int a = ilo(); a <= ihi(); ++a) {
            int ci, cj, ck, gi, gj, gk;
            if      (axis == 0) { ci=bound; cj=a; ck=b; gi=gbound; gj=a;      gk=b; }
            else if (axis == 1) { ci=a; cj=bound; ck=b; gi=a;      gj=gbound; gk=b; }
            else                { ci=a; cj=b; ck=bound; gi=a;      gj=b;      gk=gbound; }

            const Prim p_i = blk.prim(ci,cj,ck);
            const Prim p_g = blk.prim(gi,gj,gk);
            const double mu_f  = 0.5*(sutherland(p_i.T) + sutherland(p_g.T));
            const double kappa = mu_f * CP / PR;
            const double u_f   = 0.5*(p_i.u + p_g.u);
            const double v_f   = 0.5*(p_i.v + p_g.v);
            const double w_f   = 0.5*(p_i.w + p_g.w);

            // Reproduce the same face-stress stencil as viscous_rhs_impl.
            // Normal gradients (i.e. across the face) need the sign factor ns
            // because (gi,gj,gk) is the ghost: for delta>0, ghost is "right"
            // (forward); for delta<0, ghost is "left" (backward).
            // Tangential gradients (cross-stencil) are symmetric and need no ns.
            double Fvisc_E;
            if (axis == 0) {
                double dudx = ns*ih*(U(gi,gj,gk)-U(ci,cj,ck));
                double dvdx = ns*ih*(V(gi,gj,gk)-V(ci,cj,ck));
                double dwdx = ns*ih*(W(gi,gj,gk)-W(ci,cj,ck));
                double dudy = ihhalf*(U(gi,gj+1,gk)-U(gi,gj-1,gk)+U(ci,cj+1,ck)-U(ci,cj-1,ck));
                double dudz = ihhalf*(U(gi,gj,gk+1)-U(gi,gj,gk-1)+U(ci,cj,ck+1)-U(ci,cj,ck-1));
                double dvdy = ihhalf*(V(gi,gj+1,gk)-V(gi,gj-1,gk)+V(ci,cj+1,ck)-V(ci,cj-1,ck));
                double dwdz = ihhalf*(W(gi,gj,gk+1)-W(gi,gj,gk-1)+W(ci,cj,ck+1)-W(ci,cj,ck-1));
                double divu = dudx + dvdy + dwdz;
                double txx  = mu_f*(2.0*dudx - (2.0/3.0)*divu);
                double txy  = mu_f*(dudy + dvdx);
                double txz  = mu_f*(dudz + dwdx);
                Fvisc_E = txx*u_f + txy*v_f + txz*w_f + kappa*ns*ih*(p_g.T-p_i.T);
            } else if (axis == 1) {
                double dvdy = ns*ih*(V(gi,gj,gk)-V(ci,cj,ck));
                double dudy = ns*ih*(U(gi,gj,gk)-U(ci,cj,ck));
                double dwdy = ns*ih*(W(gi,gj,gk)-W(ci,cj,ck));
                double dvdx = ihhalf*(V(gi+1,gj,gk)-V(gi-1,gj,gk)+V(ci+1,cj,ck)-V(ci-1,cj,ck));
                double dvdz = ihhalf*(V(gi,gj,gk+1)-V(gi,gj,gk-1)+V(ci,cj,ck+1)-V(ci,cj,ck-1));
                double dudx = ihhalf*(U(gi+1,gj,gk)-U(gi-1,gj,gk)+U(ci+1,cj,ck)-U(ci-1,cj,ck));
                double dwdz = ihhalf*(W(gi,gj,gk+1)-W(gi,gj,gk-1)+W(ci,cj,ck+1)-W(ci,cj,ck-1));
                double divu = dudx + dvdy + dwdz;
                double tyx  = mu_f*(dudy + dvdx);
                double tyy  = mu_f*(2.0*dvdy - (2.0/3.0)*divu);
                double tyz  = mu_f*(dvdz + dwdy);
                Fvisc_E = tyx*u_f + tyy*v_f + tyz*w_f + kappa*ns*ih*(p_g.T-p_i.T);
            } else {
                double dwdz = ns*ih*(W(gi,gj,gk)-W(ci,cj,ck));
                double dudz = ns*ih*(U(gi,gj,gk)-U(ci,cj,ck));
                double dvdz = ns*ih*(V(gi,gj,gk)-V(ci,cj,ck));
                double dwdx = ihhalf*(W(gi+1,gj,gk)-W(gi-1,gj,gk)+W(ci+1,cj,ck)-W(ci-1,cj,ck));
                double dwdy = ihhalf*(W(gi,gj+1,gk)-W(gi,gj-1,gk)+W(ci,cj+1,ck)-W(ci,cj-1,ck));
                double dudx = ihhalf*(U(gi+1,gj,gk)-U(gi-1,gj,gk)+U(ci+1,cj,ck)-U(ci-1,cj,ck));
                double dvdy = ihhalf*(V(gi,gj+1,gk)-V(gi,gj-1,gk)+V(ci,cj+1,ck)-V(ci,cj-1,ck));
                double divu = dudx + dvdy + dwdz;
                double tzx  = mu_f*(dudz + dwdx);
                double tzy  = mu_f*(dvdz + dwdy);
                double tzz  = mu_f*(2.0*dwdz - (2.0/3.0)*divu);
                Fvisc_E = tzx*u_f + tzy*v_f + tzz*w_f + kappa*ns*ih*(p_g.T-p_i.T);
            }

            // viscous_rhs_impl added ns*ih*Fvisc_E to rhs.Q[4]; subtract it.
            rhs.Q[4][cell_idx(ci,cj,ck)] -= ns * ih * Fvisc_E;
        }
    }
}

// =============================================================================
// accumulate_cf_fine_fluxes
// ─────────────────────────
// A05-fix4: stage_weight is the SSP-RK3 quadrature weight for this stage:
//   stage 1: w = 1/6,  stage 2: w = 1/6,  stage 3: w = 2/3
// Registers are zeroed ONCE before stage 1 (in advance()).
// Each call accumulates w * F_fine; apply_flux_correction(dt) then applies
// the time-averaged fine flux, giving exact Berger-Colella conservation
// across all three RK3 sub-steps.
//
// Register layout (must match apply_flux_correction in block_tree.cpp):
//   reg[v*NB*NB + jc*NB + ic]
//   axis=0 (x-face, YZ plane): jc = coarse_y_idx (0..NB-1), ic = coarse_z_idx
//   axis=1 (y-face, XZ plane): jc = coarse_z_idx,           ic = coarse_x_idx
//   axis=2 (z-face, XY plane): jc = coarse_y_idx,           ic = coarse_x_idx
//
// A05-fix7: correct fine-to-coarse index mapping for axis=1 and axis=2.
//
// The face loop runs `a` over the first transverse direction and `b` over the
// second.  The physical direction these represent depends on the face axis:
//   axis=0 (x-face): a→y (j), b→z (k)  → jc uses a_local, ic uses b_local ✓
//   axis=1 (y-face): a→x (i), b→z (k)  → jc (→z) must use b_local,
//                                          ic (→x) must use a_local
//   axis=2 (z-face): a→x (i), b→y (j)  → jc (→y) must use b_local,
//                                          ic (→x) must use a_local
//
// The previous code (A05-fix6) used a_local for jc and b_local for ic
// uniformly across all axes.  This was correct only for axis=0; for axis=1
// and axis=2 the a/b roles are swapped, sending fine fluxes from non-diagonal
// octants to the wrong coarse register slots and producing a systematic mass
// leak whenever y- or z-face CF interfaces exist.
// =============================================================================
// level_filter: when >= 0 only leaves at that level contribute fine fluxes.
// This is required for P4.1 LTS so that fine-level sub-steps do not
// double-accumulate during the subsequent coarse step (and vice-versa).
static void accumulate_cf_fine_fluxes(BlockTree& tree,
                                       double stage_weight,
                                       int    level_filter = -1) noexcept
{
    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};
    static constexpr int HALF = NB / 2;

    for (int li : tree.leaf_indices()) {
        if (level_filter >= 0 && tree.nodes[li].level != level_filter) continue;
        const auto& nd  = tree.nodes[li];
        if (!nd.has_block()) continue;  // P7.1: remote leaf
        const auto& blk = *nd.block;

        // Octant of this fine leaf relative to its parent.
        // Needed to compute the quadrant offset on the coarse face.
        // oct ∈ {0..7}: bit0=ix, bit1=iy, bit2=iz.
        const int parent_idx = nd.parent;
        const int oct = (parent_idx >= 0)
                        ? li - tree.nodes[parent_idx].first_child
                        : 0;
        const int o_ix = oct_ix(oct);  // 0 or 1
        const int o_iy = oct_iy(oct);
        const int o_iz = oct_iz(oct);

        for (int d = 0; d < NFACES; ++d) {
            const int ni = nd.neighbours[d];
            if (ni < 0 || !tree.nodes[ni].has_block()) continue;
            if (tree.nodes[ni].level >= nd.level) continue;

            const int axis  = face_axis[d];
            const int delta = face_delta[d];
            const int bound = (delta > 0) ? ihi() : ilo();

            // Transverse octant offsets for this face axis.
            // off1 is the octant component in the jc direction (1st register index).
            // off2 is the octant component in the ic direction (2nd register index).
            //   axis=0: jc→y, ic→z  → off1=o_iy, off2=o_iz
            //   axis=1: jc→z, ic→x  → off1=o_iz, off2=o_ix
            //   axis=2: jc→y, ic→x  → off1=o_iy, off2=o_ix
            int off1, off2;
            if      (axis == 0) { off1 = o_iy; off2 = o_iz; }
            else if (axis == 1) { off1 = o_iz; off2 = o_ix; }
            else                { off1 = o_iy; off2 = o_ix; }

            // Accumulate into a NB×NB register of COARSE-cell averaged fluxes.
            // Each fine block contributes to one (NB/2)×(NB/2) quadrant.
            // Four fine cells per coarse slot → each contributes 1/4 (applied
            // by area_ratio=0.25 inside accumulate_fine_flux).
            std::vector<double> face_flux(NVAR * NB * NB, 0.0);

            for (int b = ilo(); b <= ihi(); ++b)
            for (int a = ilo(); a <= ihi(); ++a) {
                int ci, cj, ck, gi, gj, gk;
                if (axis == 0) {
                    ci=bound; cj=a;     ck=b;
                    gi=bound+delta; gj=a; gk=b;
                } else if (axis == 1) {
                    ci=a;     cj=bound; ck=b;
                    gi=a; gj=bound+delta; gk=b;
                } else {
                    ci=a;     cj=b;     ck=bound;
                    gi=a; gj=b; gk=bound+delta;
                }

                Prim interior = blk.prim(ci, cj, ck);
                Prim ghost    = blk.prim(gi, gj, gk);

                std::array<double,5> F;
                if (delta > 0) F = hllc_es_flux(interior, ghost, axis);
                else           F = hllc_es_flux(ghost, interior, axis);

                // Map fine cell (a,b) to the coarse-cell register index.
                // The mapping depends on which physical direction a and b
                // represent for this face axis (A05-fix7):
                //   axis=0: a→y→jc, b→z→ic  → jc uses a_local, ic uses b_local
                //   axis=1: a→x→ic, b→z→jc  → jc uses b_local, ic uses a_local
                //   axis=2: a→x→ic, b→y→jc  → jc uses b_local, ic uses a_local
                const int a_local = a - ilo();
                const int b_local = b - ilo();
                int jc, ic;
                if (axis == 0) {
                    jc = off1 * HALF + a_local / 2;
                    ic = off2 * HALF + b_local / 2;
                } else {
                    // axis=1 and axis=2: a→x, b→(z or y); jc indexes the
                    // non-x transverse direction, so it must use b_local.
                    jc = off1 * HALF + b_local / 2;
                    ic = off2 * HALF + a_local / 2;
                }

                // Accumulate (+=): 4 fine cells contribute to the same coarse
                // slot; area_ratio=0.25 in accumulate_fine_flux completes the
                // area-weighted average.
                for (int v = 0; v < NVAR; ++v)
                    face_flux[v*NB*NB + jc*NB + ic] += stage_weight * F[v];
            }
            tree.accumulate_fine_flux(li, static_cast<FaceDir>(d), face_flux);
        }
    }
}

// =============================================================================
// tree_rhs
// ─────────
// A05-fix4: stage_weight is the SSP-RK3 quadrature weight (1/6, 1/6, 2/3).
// The caller (advance()) must zero flux registers once before stage 1 and
// must NOT zero them between stages.  apply_flux_correction(dt) is called
// after stage 3.
//
// P2.4: leaf loop parallelised with OpenMP.
//   - Ghost fill runs BEFORE the parallel region: it touches shared neighbour
//     data and is not thread-safe to parallelise.
//   - compute_rhs and undo_cf_face_flux are fully independent per leaf: each
//     reads from its own block (+ already-filled ghost cells, read-only) and
//     writes only to its own rhs_blocks[li] slot.
//   - static thread_local prim caches in compute_rhs are safe: each thread
//     gets its own copy.
//   - accumulate_cf_fine_fluxes writes to shared flux registers; it runs
//     serially after the parallel loop.
//   - Leaves are sorted by Morton code before the loop so spatially adjacent
//     blocks land on the same CPU socket / L3 cache slice.
// =============================================================================
void tree_rhs(BlockTree& tree,
              std::vector<CellBlock>& rhs_blocks,
              bool periodic,
              double stage_weight,
              int    level_filter,
              bool   cf_coarse_zero_grad,
              bool   open_bc) noexcept
{
    // ── 1. Ghost fill — always global (C/F fills need the full tree) ─────────
    if (periodic)
        tree.fill_ghosts_periodic(cf_coarse_zero_grad);
    else if (open_bc)
        tree.fill_ghosts_open(cf_coarse_zero_grad);
    else
        tree.fill_ghosts_wall(cf_coarse_zero_grad);

    const auto& leaves = tree.leaf_indices();
    const int n_leaves = (int)leaves.size();
    assert((int)rhs_blocks.size() == n_leaves);

    // ── 2. Build Morton-sorted slot order for L3 cache locality ──────────────
    std::vector<int> order;
    order.reserve(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        if (level_filter >= 0 && tree.nodes[leaves[li]].level != level_filter)
            continue;
        order.push_back(li);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tree.nodes[leaves[a]].morton < tree.nodes[leaves[b]].morton;
    });
    const int n_active = (int)order.size();

    // ── 3. Per-leaf RHS — parallel (fully independent per slot) ──────────────
#pragma omp parallel for schedule(dynamic,4)
    for (int oi = 0; oi < n_active; ++oi) {
        const int li       = order[oi];
        const int node_idx = leaves[li];
        if (!tree.nodes[node_idx].has_block()) continue;  // P7.1: remote leaf
        compute_rhs(*tree.nodes[node_idx].block, rhs_blocks[li]);
        undo_cf_face_flux(tree, node_idx, rhs_blocks[li]);
        if (cf_coarse_zero_grad)
            undo_cf_viscous_energy(tree, node_idx, rhs_blocks[li]);
    }

    // ── 4. Flux register accumulation — serial (shared CF register writes) ───
    // Pass level_filter so that only leaves at the active level contribute;
    // prevents double-accumulation during LTS coarse/fine interleaving.
    accumulate_cf_fine_fluxes(tree, stage_weight, level_filter);
}

// =============================================================================
// CFL time step
// =============================================================================
double tree_cfl_dt(const BlockTree& tree, double cfl) noexcept
{
    double dt = 1e300;
    for (auto li : tree.leaf_indices()) {
        if (!tree.nodes[li].has_block()) continue;  // P7.1: remote leaf
        dt = std::min(dt, tree.nodes[li].block->cfl_dt(cfl));
    }
    return dt;
}

// P4.1: minimum CFL dt over leaves at a specific refinement level.
// Used by advance_lts() to set the fine-level time step independently.
double level_cfl_dt(const BlockTree& tree, int level, double cfl) noexcept
{
    double dt = 1e300;
    for (auto li : tree.leaf_indices()) {
        if (tree.nodes[li].level != level) continue;
        if (!tree.nodes[li].has_block()) continue;  // P7.1: remote leaf
        dt = std::min(dt, tree.nodes[li].block->cfl_dt(cfl));
    }
    return dt;
}
