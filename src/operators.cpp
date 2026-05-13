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
#include "concepts.hpp"
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
#include "physics/diff_ops.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>
#ifdef _OPENMP
#  include <omp.h>
#endif

// ── R7: Differential operator concept verification ─────────────────────────
static_assert(ScalarCellOperator<CellGrad<Axis::X>>);
static_assert(ScalarCellOperator<CellGrad<Axis::X, 4>>);
static_assert(ScalarFaceOperator<FaceGrad<Axis::X>>);
static_assert(ScalarFaceOperator<FaceGrad<Axis::X, 4>>);
static_assert(TensorFaceOperator<VelocityGradAtFace<Axis::X>>);
static_assert(TensorFaceOperator<VelocityGradAtFace<Axis::X, 4>>);

// ── FaceInterp: all four mean policies ─────────────────────────────────────
static_assert(FaceInterpolator<FaceInterp<Axis::X>>);
static_assert(FaceInterpolator<FaceInterp<Axis::X, GeometricMean>>);
static_assert(FaceInterpolator<FaceInterp<Axis::X, LogMean>>);
static_assert(FaceInterpolator<FaceInterp<Axis::X, HarmonicMean>>);

// =============================================================================
// P14.1: Stiffened-gas EOS global state
// Set once per advance() from NSSolver when use_acdi && use_sg_eos is true.
// =============================================================================
static bool   g_sg_active = false;
static double g_sg_ga  = GAMMA, g_sg_gb  = GAMMA;
static double g_sg_pia = 0.0,   g_sg_pib = 0.0;

void set_sg_eos(bool active, double ga, double gb, double pia, double pib) noexcept {
    g_sg_active = active;
    g_sg_ga = ga; g_sg_gb = gb;
    g_sg_pia = pia; g_sg_pib = pib;
    // P14.1c: propagate to CellBlock statics so prim() and cfl_dt() use the correct EOS
    CellBlock::sg_active_ = active;
    CellBlock::sg_ga_     = ga;  CellBlock::sg_gb_  = gb;
    CellBlock::sg_pia_    = pia; CellBlock::sg_pib_ = pib;
}

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
    // R2: delegate to HllcFlux<DIR> physics functor (include/physics/hllc_flux.hpp)
    switch (axis) {
        case 0:  return HllcFlux<Axis::X>{}(L, R);
        case 1:  return HllcFlux<Axis::Y>{}(L, R);
        default: return HllcFlux<Axis::Z>{}(L, R);
    }
}

// =============================================================================
// P2.3 — Primitive variable and viscosity caches
// =============================================================================
// Pre-compute all NCELL primitive states (and Sutherland µ) once per
// compute_rhs call.  Interior + ghost cells filled → stencil = pure lookup.
static void fill_prim_cache(const CellBlock& blk, Prim* pc) noexcept {
    if (!g_sg_active) {
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
    } else {
        // P14.1: stiffened-gas per-cell mixture EOS from phi_data_
        for (int flat = 0; flat < NCELL; ++flat) {
            double gm, pim;
            mix_eos(blk.phi_data_[flat], g_sg_ga, g_sg_gb, g_sg_pia, g_sg_pib, gm, pim);
            pc[flat] = eos_cons_to_prim_sg(blk.Q[0][flat], blk.Q[1][flat],
                                            blk.Q[2][flat], blk.Q[3][flat],
                                            blk.Q[4][flat], gm, pim);
        }
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
// P13.1 stage 3 — compile-time axis: dead branches eliminated by constexpr if
template<Axis DIR>
static std::array<double,NVAR> kep_flux_t(const Prim& L, const Prim& R) noexcept {
    const double u_a   = 0.5*(L.u   + R.u  );
    const double v_a   = 0.5*(L.v   + R.v  );
    const double w_a   = 0.5*(L.w   + R.w  );
    const double p_a   = 0.5*(L.p   + R.p  );
    // P14.1: stiffened-gas E for H computation
    const double E_L   = (L.p + L.gamma_m*L.p_inf_m)/(L.gamma_m-1.0) + 0.5*L.rho*(L.u*L.u+L.v*L.v+L.w*L.w);
    const double E_R   = (R.p + R.gamma_m*R.p_inf_m)/(R.gamma_m-1.0) + 0.5*R.rho*(R.u*R.u+R.v*R.v+R.w*R.w);
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
    constexpr CellGrad<Axis::X, 2> dX;
    constexpr CellGrad<Axis::Y, 2> dY;
    constexpr CellGrad<Axis::Z, 2> dZ;

    auto U = [pc](int i, int j, int k){ return pc[cell_idx(i,j,k)].u; };
    auto V = [pc](int i, int j, int k){ return pc[cell_idx(i,j,k)].v; };
    auto W = [pc](int i, int j, int k){ return pc[cell_idx(i,j,k)].w; };

    std::fill(duc, duc + NCELL, 0.0);

    for (int k = 1; k < NB2-1; ++k)
    for (int j = 1; j < NB2-1; ++j)
    for (int i = 1; i < NB2-1; ++i) {
        const double dudx = dX(U, i, j, k, h);
        const double dudy = dY(U, i, j, k, h);
        const double dudz = dZ(U, i, j, k, h);
        const double dvdx = dX(V, i, j, k, h);
        const double dvdy = dY(V, i, j, k, h);
        const double dvdz = dZ(V, i, j, k, h);
        const double dwdx = dX(W, i, j, k, h);
        const double dwdy = dY(W, i, j, k, h);
        const double dwdz = dZ(W, i, j, k, h);

        const double divu = dudx + dvdy + dwdz;
        const double ox   = dwdy - dvdz;
        const double oy   = dudz - dwdx;
        const double oz   = dvdx - dudy;
        const double d2   = divu*divu;
        const double c2   = ox*ox + oy*oy + oz*oz;
        const double phi_vel = d2 / (d2 + c2 + eps_duc);

        const double pC   = pc[cell_idx(i,j,k)].p;
        const double dpx  = std::max(std::abs(pc[cell_idx(i+1,j,k)].p - pC),
                                     std::abs(pc[cell_idx(i-1,j,k)].p - pC));
        const double dpy  = std::max(std::abs(pc[cell_idx(i,j+1,k)].p - pC),
                                     std::abs(pc[cell_idx(i,j-1,k)].p - pC));
        const double dpz  = std::max(std::abs(pc[cell_idx(i,j,k+1)].p - pC),
                                     std::abs(pc[cell_idx(i,j,k-1)].p - pC));
        const double phi_p = std::max({dpx, dpy, dpz}) / (pC + eps_duc);
        const double phi_p_clamped = std::min(1.0, std::max(0.0,
            (phi_p - g_ducros_p_thr) / g_ducros_blend));

        duc[cell_idx(i,j,k)] = std::max(phi_vel, phi_p_clamped);
    }
}

// P13.1 stage 3 — compile-time axis; R2: delegates to Weno5Recon<DIR> functor.
template<Axis DIR>
static void weno5_face_t(const Prim* pc, int i, int j, int k,
                         Prim& qL_out, Prim& qR_out) noexcept {
    Weno5Recon<DIR>{}(pc, i, j, k, qL_out, qR_out);
}

// =============================================================================
// P15.1 — Basilisk foreach_dimension analogue: template axis helpers
// =============================================================================
//
// These helpers encode rotational invariance of the compressible flux vector
// (F_X, F_Y, F_Z are the same function under axis permutation) as C++ compile-
// time template parameters.  Three instantiations (X/Y/Z) are generated at
// compile time — no runtime axis branches in the flux computation itself.
//
// This is the C++/CUDA equivalent of Basilisk's foreach_dimension() preprocessor
// macro: the face-flux logic is written exactly once and reused for all three
// directions.  Any bug fix or scheme change propagates automatically to all axes,
// eliminating the class of asymmetric bugs where one axis's scheme diverges from
// the others.
//
// Boundary face reconstruction (is_bnd=true, non-wall):
//   PCM + HLLC-ES: use cell-centre prim values directly.  MUSCL limiters
//   (minmod P15.2 attempt, van Albada P15.2 retry) both degrade T08
//   isentropic-vortex convergence rate from ≥1.8 to ~1.1.  Root cause: T08
//   fills ghost cells with the vortex IC extended to x<0/x>1, creating O(1)
//   slopes at boundary faces that MUSCL cannot reduce below O(h).  PCM avoids
//   this via Lax-Wendroff cancellation (O(h²) flux divergence on smooth data).
//   undo_cf_one_face and cf_accum_one_face use the same PCM+HLLC-ES formula
//   for exact Berger-Colella cancellation.  Future fix: gate MUSCL on whether
//   the ghost cells contain real physics data vs. BC padding (needs topology
//   info in accumulate_face — not implemented).
// =============================================================================

// is_wall_ghost: detect no-slip wall ghost face.
// Exact float equality p_L == p_R (from symmetric energy ghost fill) plus
// anti-symmetric velocities uniquely identifies wall fill_ghosts_wall output.
// AMR/periodic/open ghosts differ in pressure or have symmetric velocities.
static bool is_wall_ghost(const Prim& pL, const Prim& pR) noexcept {
    if (pL.p != pR.p) return false;
    auto antisym = [](double a, double b) noexcept -> bool {
        return std::fabs(a + b) < 1e-8 * (std::fabs(a) + std::fabs(b) + 1e-300);
    };
    return antisym(pL.u, pR.u) && antisym(pL.v, pR.v) && antisym(pL.w, pR.w);
}

// face_lr_idx<DIR>: flat cell indices for the left and right cells of face (n,a,b).
// n = normal-axis loop variable (ilo()-1 .. ihi()); a, b = transverse (ilo()..ihi()).
template <Axis DIR>
static inline std::pair<int,int> face_lr_idx(int n, int a, int b) noexcept {
    if constexpr (DIR == Axis::X) return {cell_idx(n,  a,b), cell_idx(n+1,a,b)};
    if constexpr (DIR == Axis::Y) return {cell_idx(a,n,  b), cell_idx(a,n+1,b)};
    return                              {cell_idx(a,b,n  ), cell_idx(a,b,n+1)};
}

// face_to_ijk<DIR>: convert face coords (n, a, b) to (xi, yi, zi) of the left cell.
// Matches the calling convention of weno5_face_t<DIR>.
template <Axis DIR>
static inline void face_to_ijk(int n, int a, int b,
                                int& xi, int& yi, int& zi) noexcept {
    if constexpr (DIR == Axis::X) { xi = n; yi = a; zi = b; }
    else if constexpr (DIR == Axis::Y) { xi = a; yi = n; zi = b; }
    else                               { xi = a; yi = b; zi = n; }
}

// accumulate_face<DIR>: compute the flux at one face and accumulate into rhs.
//
// Encodes the full Pirozzoli (2011) hybrid scheme (KEP + Ducros-blended
// WENO5-ES) in a single axis-agnostic template — the C++ equivalent of
// Basilisk's foreach_dimension().  Instantiated for X, Y, Z at compile time.
//
// Face classification:
//   is_bnd=false: interior face — WENO5-Z + HLLC-ES (full 5-point stencil)
//                 or pure KEP when θ < 1e-8 (smooth/vortex-dominated region)
//   is_bnd=true, wall: KEP only (exact wall pressure, zero LF tangential drain)
//   is_bnd=true, other: PCM + HLLC-ES (cell-centre prim → Lax-Wendroff O(h²))
//
// PCM+HLLC-ES is required for Berger-Colella consistency: undo_cf_one_face and
// cf_accum_one_face use the same PCM+HLLC-ES formula, so their contributions
// cancel exactly at C/F interfaces.
template <Axis DIR>
static void accumulate_face(const Prim* pc, const double* duc,
                             CellBlock& rhs, double ih,
                             int n, int a, int b) noexcept {
    constexpr double kep_threshold = 1.0e-8;
    const auto [Li, Ri] = face_lr_idx<DIR>(n, a, b);
    const Prim& pL = pc[Li];
    const Prim& pR = pc[Ri];
    const double theta = std::max(duc[Li], duc[Ri]);
    const bool is_bnd = (n < ilo() || n+1 > ihi());

    std::array<double,NVAR> F;
    if (!is_bnd && theta < kep_threshold) {
        // Pure KEP: smooth/vortex-dominated interior — zero artificial dissipation
        F = kep_flux_t<DIR>(pL, pR);
    } else {
        const auto Fk = kep_flux_t<DIR>(pL, pR);
        std::array<double,NVAR> Fs;
        if (!is_bnd) {
            // Interior face: full WENO5-Z + HLLC-ES (5-point stencil always fits)
            int xi, yi, zi;
            face_to_ijk<DIR>(n, a, b, xi, yi, zi);
            Prim qL, qR;
            weno5_face_t<DIR>(pc, xi, yi, zi, qL, qR);
            Fs = hllc_es_flux_t<DIR>(qL, qR);
        } else if (is_wall_ghost(pL, pR)) {
            // No-slip wall: KEP gives {0, p_wall, 0, 0, 0} — correct wall pressure,
            // zero tangential flux, avoids LF momentum drain on anti-symmetric ghosts
            Fs = Fk;
        } else {
            // Block boundary (same-level or C/F): PCM + HLLC-ES
            Fs = hllc_es_flux_t<DIR>(pL, pR);
        }
        // Ducros-weighted blend: boundary → full shock flux; interior → hybrid
        const double th = is_bnd ? 1.0 : theta;
        const double om = 1.0 - th;
        for (int v = 0; v < NVAR; ++v) F[v] = om * Fk[v] + th * Fs[v];
    }
    // Conservative divergence: F leaves left cell (Li), enters right cell (Ri)
    if (n >= ilo())
        for (int v = 0; v < NVAR; ++v) rhs.Q[v][Li] -= ih * F[v];
    if (n+1 <= ihi())
        for (int v = 0; v < NVAR; ++v) rhs.Q[v][Ri] += ih * F[v];
}

// =============================================================================
// P2.2/P3.2/P15.1 — Convective RHS: face-centred hybrid loop
// =============================================================================
// Three direction sweeps, each calling accumulate_face<DIR>.  The flux
// computation is written once in accumulate_face; the loops only differ in
// which variable is n (normal) and which are a, b (transverse).
//
// Loop ordering (innermost = i, stride-1 in cell_idx) is preserved from the
// original three-section code to maintain CPU cache performance.
//
//   F_KEP  : Pirozzoli (2011) KE-preserving flux (zero added dissipation)
//   F_bnd  : PCM+HLLC-ES at block boundary faces (Lax-Wendroff O(h²))
//   F_shock: WENO5-Z+HLLC-ES for interior shock-region faces (P3.1/P3.2)
//   Blend  : F = (1−θ)·F_KEP + θ·F_shock,  θ = max(Ducros_L, Ducros_R)
//
// Flux sign: F positive leaving the left cell:
//   rhs[left] -= ih*F,  rhs[right] += ih*F  (telescopes for conservation)
static void convective_rhs_impl(const Prim* pc, const double* duc,
                                 CellBlock& rhs, double h) noexcept
{
    const double ih = 1.0 / h;

    // X: n=i (normal), a=j, b=k
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo()-1; i <= ihi(); ++i)
        accumulate_face<Axis::X>(pc, duc, rhs, ih, i, j, k);

    // Y: n=j (normal), a=i (innermost for stride-1), b=k
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo()-1; j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        accumulate_face<Axis::Y>(pc, duc, rhs, ih, j, i, k);

    // Z: n=k (normal), a=i (innermost for stride-1), b=j
    for (int k = ilo()-1; k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        accumulate_face<Axis::Z>(pc, duc, rhs, ih, k, i, j);
}

// =============================================================================
// R5 — Typed convective kernel: Flux × Recon resolved at compile time
// =============================================================================
// accumulate_face_typed mirrors accumulate_face exactly, replacing the
// hardcoded weno5_face_t / hllc_es_flux_t calls with functor invocations
// Recon<DIR>{}(...) and Flux<DIR>{}(...).
template<Axis DIR, template<Axis> class Flux, template<Axis> class Recon>
static void accumulate_face_typed(const Prim* pc, const double* duc,
                                   CellBlock& rhs, double ih,
                                   int n, int a, int b) noexcept {
    constexpr double kep_threshold = 1.0e-8;
    const auto [Li, Ri] = face_lr_idx<DIR>(n, a, b);
    const Prim& pL = pc[Li];
    const Prim& pR = pc[Ri];
    const double theta = std::max(duc[Li], duc[Ri]);
    const bool is_bnd = (n < ilo() || n+1 > ihi());

    std::array<double,NVAR> F;
    if (!is_bnd && theta < kep_threshold) {
        F = kep_flux_t<DIR>(pL, pR);
    } else {
        const auto Fk = kep_flux_t<DIR>(pL, pR);
        std::array<double,NVAR> Fs;
        if (!is_bnd) {
            int xi, yi, zi;
            face_to_ijk<DIR>(n, a, b, xi, yi, zi);
            Prim qL, qR;
            Recon<DIR>{}(pc, xi, yi, zi, qL, qR);
            Fs = Flux<DIR>{}(qL, qR);
        } else if (is_wall_ghost(pL, pR)) {
            Fs = Fk;
        } else {
            Fs = Flux<DIR>{}(pL, pR);
        }
        const double th = is_bnd ? 1.0 : theta;
        const double om = 1.0 - th;
        for (int v = 0; v < NVAR; ++v) F[v] = om * Fk[v] + th * Fs[v];
    }
    if (n >= ilo())
        for (int v = 0; v < NVAR; ++v) rhs.Q[v][Li] -= ih * F[v];
    if (n+1 <= ihi())
        for (int v = 0; v < NVAR; ++v) rhs.Q[v][Ri] += ih * F[v];
}

template<template<Axis> class Flux, template<Axis> class Recon>
static void convective_rhs_impl_typed(const Prim* pc, const double* duc,
                                       CellBlock& rhs, double h) noexcept {
    const double ih = 1.0 / h;

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo()-1; i <= ihi(); ++i)
        accumulate_face_typed<Axis::X,Flux,Recon>(pc, duc, rhs, ih, i, j, k);

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo()-1; j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        accumulate_face_typed<Axis::Y,Flux,Recon>(pc, duc, rhs, ih, j, i, k);

    for (int k = ilo()-1; k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        accumulate_face_typed<Axis::Z,Flux,Recon>(pc, duc, rhs, ih, k, i, j);
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
    const double ih = 1.0 / h;

    auto U  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].u; };
    auto V  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].v; };
    auto W  = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].w; };
    auto Tf = [&](int ii,int jj,int kk){ return pc[cell_idx(ii,jj,kk)].T; };
    auto MU = [&](int ii,int jj,int kk){ return mu_arr[cell_idx(ii,jj,kk)]; };

    constexpr VelocityGradAtFace<Axis::X, 2> VGX;
    constexpr VelocityGradAtFace<Axis::Y, 2> VGY;
    constexpr VelocityGradAtFace<Axis::Z, 2> VGZ;
    constexpr FaceInterp<Axis::X> FI_X;
    constexpr FaceInterp<Axis::Y> FI_Y;
    constexpr FaceInterp<Axis::Z> FI_Z;

    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {

        // ── Face-averaged µ at the 6 faces (FaceInterp<DIR, ArithmeticMean>) ─
        double mu_xp = FI_X(MU, i,   j,   k  );
        double mu_xm = FI_X(MU, i-1, j,   k  );
        double mu_yp = FI_Y(MU, i,   j,   k  );
        double mu_ym = FI_Y(MU, i,   j-1, k  );
        double mu_zp = FI_Z(MU, i,   j,   k  );
        double mu_zm = FI_Z(MU, i,   j,   k-1);

        // ── Velocity gradient tensors at 6 faces (R7: VelocityGradAtFace) ──
        const auto gxp = VGX.plus (U, V, W, i, j, k, h);
        const auto gxm = VGX.minus(U, V, W, i, j, k, h);
        const auto gyp = VGY.plus (U, V, W, i, j, k, h);
        const auto gym = VGY.minus(U, V, W, i, j, k, h);
        const auto gzp = VGZ.plus (U, V, W, i, j, k, h);
        const auto gzm = VGZ.minus(U, V, W, i, j, k, h);

        // ── Face stresses τ = µ·(∂u_a/∂x_b + ∂u_b/∂x_a − δ·⅔ div u) ────
        // x-faces
        double txx_xp = mu_xp*(2.0*gxp.dun_dxn - (2.0/3.0)*gxp.divu());
        double txy_xp = mu_xp*(gxp.dun_dxt1 + gxp.dut1_dxn);
        double txz_xp = mu_xp*(gxp.dun_dxt2 + gxp.dut2_dxn);
        double txx_xm = mu_xm*(2.0*gxm.dun_dxn - (2.0/3.0)*gxm.divu());
        double txy_xm = mu_xm*(gxm.dun_dxt1 + gxm.dut1_dxn);
        double txz_xm = mu_xm*(gxm.dun_dxt2 + gxm.dut2_dxn);
        // y-faces
        double tyx_yp = mu_yp*(gyp.dut1_dxn + gyp.dun_dxt1);
        double tyy_yp = mu_yp*(2.0*gyp.dun_dxn - (2.0/3.0)*gyp.divu());
        double tyz_yp = mu_yp*(gyp.dun_dxt2 + gyp.dut2_dxn);
        double tyx_ym = mu_ym*(gym.dut1_dxn + gym.dun_dxt1);
        double tyy_ym = mu_ym*(2.0*gym.dun_dxn - (2.0/3.0)*gym.divu());
        double tyz_ym = mu_ym*(gym.dun_dxt2 + gym.dut2_dxn);
        // z-faces
        double tzx_zp = mu_zp*(gzp.dut1_dxn + gzp.dun_dxt1);
        double tzy_zp = mu_zp*(gzp.dut2_dxn + gzp.dun_dxt2);
        double tzz_zp = mu_zp*(2.0*gzp.dun_dxn - (2.0/3.0)*gzp.divu());
        double tzx_zm = mu_zm*(gzm.dut1_dxn + gzm.dun_dxt1);
        double tzy_zm = mu_zm*(gzm.dut2_dxn + gzm.dun_dxt2);
        double tzz_zm = mu_zm*(2.0*gzm.dun_dxn - (2.0/3.0)*gzm.divu());

        // ── Conservative momentum divergences ────────────────────────────────
        double ax = ih*((txx_xp-txx_xm) + (tyx_yp-tyx_ym) + (tzx_zp-tzx_zm));
        double ay = ih*((txy_xp-txy_xm) + (tyy_yp-tyy_ym) + (tzy_zp-tzy_zm));
        double az = ih*((txz_xp-txz_xm) + (tyz_yp-tyz_ym) + (tzz_zp-tzz_zm));

        // ── Energy: conservative face-flux form  div(τ·u + κ∇T) ─────────────
        // Using the already-computed face stresses τ_ab|face and face averages.
        // This form telescopes at any boundary type (periodic, wall, C/F) so that
        // total energy is conserved when viscous face fluxes vanish at boundaries.
        // F_visc_E_x|face = τxx*u_face + τxy*v_face + τxz*w_face + κ*dT/dx|face
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

// R2: SpatialReconstruction for weno5_face_t now verified directly on Weno5Recon<DIR>
// in include/physics/weno5_recon.hpp. No wrapper stub needed here.

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
// R5 — compute_rhs_typed: typed dispatch for (Flux × Recon × EOS) combinations
// =============================================================================
// Body lives here (not in operators.hpp) because all static helpers
// (kep_flux_t, fill_prim_cache, fill_ducros_cache, etc.) are TU-private.
// Explicit instantiations at the bottom of this file satisfy the ODR for all
// supported combinations; callers link to those pre-compiled specialisations.
template<template<Axis> class Flux, template<Axis> class Recon, class EOS>
    requires RiemannFlux<Flux<Axis::X>>
          && SpatialReconstruction<Recon<Axis::X>>
          && EquationOfState<EOS>
void compute_rhs_typed(const CellBlock& blk, CellBlock& rhs_blk) noexcept
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

    convective_rhs_impl_typed<Flux,Recon>(pc.data(), duc.data(), rhs_blk, blk.h);
    viscous_rhs_impl          (pc.data(), mu_arr.data(),          rhs_blk, blk.h);
}

// Explicit instantiations — one row per supported (Flux × Recon × EOS) combination.
// Add a new row here when a new scheme is introduced.
#include "physics/ideal_gas_eos.hpp"

template void compute_rhs_typed<HllcEsFlux, Weno5Recon, IdealGasEOS>(
    const CellBlock&, CellBlock&) noexcept;
template void compute_rhs_typed<HllcFlux,   Weno5Recon, IdealGasEOS>(
    const CellBlock&, CellBlock&) noexcept;

// =============================================================================
// Berger-Colella reflux: undo_cf_face_flux
// ─────────────────────────────────────────
// Undoes the coarse HLLC contribution at each CF face so that
// apply_flux_correction() can add the correct fine-side average.
// Net effect after correction:
//   dQ/dt += (1/h) * [F_fine_avg - F_coarse_own]
// =============================================================================
// undo_cf_one_face<DIR>: removes the PCM+HLLC-ES boundary flux that compute_rhs
// added at one coarse C/F face, so that accumulate_cf_fine_fluxes can replace it
// with the correctly area-averaged fine flux.  Same PCM+HLLC-ES formula as
// accumulate_face<DIR>, guaranteeing exact Berger-Colella cancellation.
template <Axis DIR>
static void undo_cf_one_face(const CellBlock& blk, CellBlock& rhs, int delta) noexcept {
    const double ih   = 1.0 / blk.h;
    const double sign = (double)delta;  // +1 (right face) or -1 (left face)
    const int    bound = (delta > 0) ? ihi() : ilo();
    for (int b = ilo(); b <= ihi(); ++b)
    for (int a = ilo(); a <= ihi(); ++a) {
        int ci, cj, ck, gi, gj, gk;
        if constexpr (DIR == Axis::X) {
            ci=bound; cj=a; ck=b; gi=bound+delta; gj=a; gk=b;
        } else if constexpr (DIR == Axis::Y) {
            ci=a; cj=bound; ck=b; gi=a; gj=bound+delta; gk=b;
        } else {
            ci=a; cj=b; ck=bound; gi=a; gj=b; gk=bound+delta;
        }
        const Prim interior = blk.prim(ci, cj, ck);
        const Prim ghost    = blk.prim(gi, gj, gk);
        const auto F = (delta > 0) ? hllc_es_flux_t<DIR>(interior, ghost)
                                   : hllc_es_flux_t<DIR>(ghost,    interior);
        const int idx = cell_idx(ci, cj, ck);
        for (int v = 0; v < NVAR; ++v)
            rhs.Q[v][idx] += sign * ih * F[v];
    }
}

static void undo_cf_face_flux(const BlockTree& tree, int node_idx,
                               CellBlock& rhs) noexcept
{
    const auto& nd  = tree.nodes[node_idx];
    const auto& blk = *nd.block;

    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

    for (int d = 0; d < NFACES; ++d) {
        const int ni = nd.neighbours[d];
        if (ni < 0 || !tree.nodes[ni].has_block()) continue;
        if (tree.nodes[ni].level <= nd.level) continue;

        const int axis  = face_axis[d];
        const int delta = face_delta[d];
        switch (axis) {
            case 0: undo_cf_one_face<Axis::X>(blk, rhs, delta); break;
            case 1: undo_cf_one_face<Axis::Y>(blk, rhs, delta); break;
            case 2: undo_cf_one_face<Axis::Z>(blk, rhs, delta); break;
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
// cf_accum_one_face<DIR>: accumulates PCM+HLLC-ES fine-face flux into the
// Berger-Colella register for one face direction.  Same reconstruction as
// accumulate_face<DIR> and undo_cf_one_face<DIR> — exact cancellation at C/F.
// (off1, off2) are the octant quadrant offsets; (jc, ic) register mapping is
// axis-specific and encoded via if constexpr so all three axes share one body.
template <Axis DIR>
static void cf_accum_one_face(const CellBlock& blk, int delta, int bound,
                               int off1, int off2, double stage_weight,
                               std::vector<double>& face_flux) noexcept {
    static constexpr int HALF = NB / 2;
    for (int b = ilo(); b <= ihi(); ++b)
    for (int a = ilo(); a <= ihi(); ++a) {
        int ci, cj, ck, gi, gj, gk;
        if constexpr (DIR == Axis::X) {
            ci=bound; cj=a; ck=b; gi=bound+delta; gj=a; gk=b;
        } else if constexpr (DIR == Axis::Y) {
            ci=a; cj=bound; ck=b; gi=a; gj=bound+delta; gk=b;
        } else {
            ci=a; cj=b; ck=bound; gi=a; gj=b; gk=bound+delta;
        }
        const Prim interior = blk.prim(ci, cj, ck);
        const Prim ghost    = blk.prim(gi, gj, gk);
        const auto F = (delta > 0) ? hllc_es_flux_t<DIR>(interior, ghost)
                                   : hllc_es_flux_t<DIR>(ghost,    interior);
        const int a_local = a - ilo();
        const int b_local = b - ilo();
        // axis=X: a→y (jc direction), b→z (ic direction)
        // axis=Y/Z: a→x (ic direction), b→(z or y) (jc direction) — A05-fix7
        int jc, ic;
        if constexpr (DIR == Axis::X) {
            jc = off1 * HALF + a_local / 2;
            ic = off2 * HALF + b_local / 2;
        } else {
            jc = off1 * HALF + b_local / 2;
            ic = off2 * HALF + a_local / 2;
        }
        for (int v = 0; v < NVAR; ++v)
            face_flux[v*NB*NB + jc*NB + ic] += stage_weight * F[v];
    }
}

// level_filter: when >= 0 only leaves at that level contribute fine fluxes.
// This is required for P4.1 LTS so that fine-level sub-steps do not
// double-accumulate during the subsequent coarse step (and vice-versa).
static void accumulate_cf_fine_fluxes(BlockTree& tree,
                                       double stage_weight,
                                       int    level_filter = -1) noexcept
{
    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};

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
            switch (axis) {
                case 0: cf_accum_one_face<Axis::X>(blk, delta, bound, off1, off2, stage_weight, face_flux); break;
                case 1: cf_accum_one_face<Axis::Y>(blk, delta, bound, off1, off2, stage_weight, face_flux); break;
                case 2: cf_accum_one_face<Axis::Z>(blk, delta, bound, off1, off2, stage_weight, face_flux); break;
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
// P13.5 — SBP-SAT interface penalty at AMR C/F boundaries
// =============================================================================
// Physical basis (Del Rey Fernández et al. 2020, Lundgren & Nordström 2020):
//   At a 2:1 AMR interface, the solution is discontinuous by construction.
//   A Simultaneous Approximation Term (SAT) penalty drives the jump to zero
//   at rate σ = τ/h_f, giving semi-discrete energy stability.
//
// Conservative pairing (matches Berger-Colella volume weighting):
//   Fine leaf:   RHS[bdry] += σ * (Q_ghost − Q_interior) / h_f
//   Coarse leaf: RHS[bdry] -= σ * avg_patch(Q_ghost − Q_interior) / h_c
//
//   where Q_ghost is the coarse-interpolated value in the fine ghost cell
//   (already filled by fill_ghosts_*) and avg_patch is the 2×2 average
//   over the fine cells in the coarse cell's transverse patch.
//   Volume-weighted sum: Fine contrib = NB² * jump * h_f³ * σ/h_f
//                        Coarse contrib = NB²/4 * avg * h_c³ * σ_c/h_c
//   Setting σ_c = σ and h_c = 2*h_f yields equal and opposite totals → conservation.
//
// Ghost cells must be filled before calling this (fills fine ghost from coarse).
// tau >= 0.5 ensures energy stability (mirrors Berger-Colella strength).
void tree_sat_penalty(BlockTree& tree,
                      std::vector<CellBlock>& rhs_blocks,
                      double tau) noexcept
{
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();
    if (NL == 0) return;

    // Map: node_idx → rhs position (only for leaves with blocks)
    std::vector<int> node_to_rhs(tree.nodes.size(), -1);
    for (int ii = 0; ii < NL; ++ii) {
        if (tree.nodes[leaves[ii]].has_block())
            node_to_rhs[leaves[ii]] = ii;
    }

    for (int ii = 0; ii < NL; ++ii) {
        int li = leaves[ii];
        const BlockNode& nd = tree.nodes[li];
        if (!nd.has_block()) continue;
        const CellBlock& blk = *nd.block;

        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            if (ni < 0 || ni >= (int)tree.nodes.size()) continue;
            const BlockNode& nnd = tree.nodes[ni];
            if (!nnd.is_leaf() || !nnd.has_block()) continue;
            // Only process faces where this leaf is FINER than the neighbor
            if (nnd.block->h <= blk.h) continue;

            const int axis   = d / 2;
            const int side   = d % 2;  // 0=minus-face, 1=plus-face
            // Fine interior cell adjacent to the C/F face
            const int face_i = (side == 0) ? ilo() : ihi();
            // Fine ghost cell (holds coarse-interpolated value from fill_ghosts)
            const int ghost_i = (side == 0) ? (ilo()-1) : (ihi()+1);

            // σ = τ/h_f  (penalty per unit volume)
            const double sigma_f = tau / blk.h;
            const double sigma_c = sigma_f;  // equal σ; volume weighting cancels

            CellBlock& rhs_f = rhs_blocks[ii];

            // Coarse leaf RHS (may be nullptr if remote rank)
            int ii_c = node_to_rhs[ni];
            CellBlock* rhs_c = (ii_c >= 0) ? &rhs_blocks[ii_c] : nullptr;

            // Octant of this fine leaf relative to its parent
            // Used to determine which quadrant of the coarse face we cover.
            const int fine_parent = nd.parent;
            if (fine_parent < 0) continue;
            const BlockNode& fp = tree.nodes[fine_parent];
            if (fp.first_child < 0) continue;
            const int oct = li - fp.first_child;  // 0..7
            const int oix = oct_ix(oct);
            const int oiy = oct_iy(oct);
            const int oiz = oct_iz(oct);

            // Transverse quadrant offset in the coarse face (for conservative correction)
            // The fine block covers coarse cells [NG+ta_off .. NG+ta_off+NB/2) × similar
            const int half = NB / 2;
            int ta_off, tb_off;  // offset in coarse transverse indices
            if (axis == 0) { ta_off = oiy * half; tb_off = oiz * half; }
            else if (axis == 1) { ta_off = oix * half; tb_off = oiz * half; }
            else               { ta_off = oix * half; tb_off = oiy * half; }

            // Coarse interior cell index (1 cell inside the C/F face)
            // Fine side=1 (plus-face) → neighbor is on the "left" → coarse at ilo()
            const int c_face_i = (side == 0) ? ihi() : ilo();

            // Accumulate per-coarse-cell penalty sums (NB/2 × NB/2 coarse cells)
            // coarse_acc[ca][cb][v] = sum of (ghost-interior) over 2×2 fine patch
            // Stack array — NB/2 ≤ 4 for NB=8; zero-initialised each face pass.
            std::array<std::array<std::array<double,NVAR>, 5>, 5> coarse_acc{};
            static_assert(NB/2 <= 4, "coarse_acc sized for NB<=8");

            // Fine boundary loop: add penalty to fine RHS, accumulate for coarse
            for (int a = ilo(); a <= ihi(); ++a)
            for (int b = ilo(); b <= ihi(); ++b) {
                int fi, fj, fk, gi, gj, gk;
                if (axis == 0) {
                    fi=face_i; fj=a; fk=b;
                    gi=ghost_i; gj=a; gk=b;
                } else if (axis == 1) {
                    fi=a; fj=face_i; fk=b;
                    gi=a; gj=ghost_i; gk=b;
                } else {
                    fi=a; fj=b; fk=face_i;
                    gi=a; gj=b; gk=ghost_i;
                }
                const int a_local = a - ilo();
                const int b_local = b - ilo();
                const int ca = a_local / 2;
                const int cb = b_local / 2;

                for (int v = 0; v < NVAR; ++v) {
                    const double jump = blk.Q[v][cell_idx(gi,gj,gk)]
                                      - blk.Q[v][cell_idx(fi,fj,fk)];
                    rhs_f.Q[v][cell_idx(fi,fj,fk)] += sigma_f * jump;
                    coarse_acc[ca][cb][v] += jump;
                }
            }

            // Coarse RHS: subtract the 2×2-averaged penalty
            if (rhs_c) {
                const double inv4 = 0.25;
                for (int ca = 0; ca < half; ++ca)
                for (int cb = 0; cb < half; ++cb) {
                    // Coarse transverse index (within [NG, NG+NB))
                    const int ca_c = NG + ta_off + ca;
                    const int cb_c = NG + tb_off + cb;
                    int ci, cj, ck;
                    if (axis == 0) { ci=c_face_i; cj=ca_c; ck=cb_c; }
                    else if (axis == 1) { ci=ca_c; cj=c_face_i; ck=cb_c; }
                    else               { ci=ca_c; cj=cb_c; ck=c_face_i; }
                    for (int v = 0; v < NVAR; ++v)
                        rhs_c->Q[v][cell_idx(ci,cj,ck)] -= sigma_c * inv4 * coarse_acc[ca][cb][v];
                }
            }
        }
    }
}

// =============================================================================
// P14.1 — ACDI phase-field advection RHS
// =============================================================================
// Conservative 1st-order upwind: ∂φ/∂t = -∇·(φu).
// Face velocity = arithmetic mean of neighbour cell-centre velocities.
// Upwind state: φ_face = φ_L if u_face > 0, φ_R if u_face ≤ 0.
// Requires NG ≥ 1 ghost layers for φ (already guaranteed by NG=2).
void phi_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    const double inv_h = 1.0 / blk.h;

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        const int flat = cell_idx(i,j,k);

        // Pre-fetch primitive velocities at cell and all ±1 neighbours.
        const Prim pC  = blk.prim(i,  j,  k  );
        const Prim pXm = blk.prim(i-1,j,  k  );
        const Prim pXp = blk.prim(i+1,j,  k  );
        const Prim pYm = blk.prim(i,  j-1,k  );
        const Prim pYp = blk.prim(i,  j+1,k  );
        const Prim pZm = blk.prim(i,  j,  k-1);
        const Prim pZp = blk.prim(i,  j,  k+1);

        const double phi_C  = blk.phi(i,  j,  k  );
        const double phi_Xm = blk.phi(i-1,j,  k  );
        const double phi_Xp = blk.phi(i+1,j,  k  );
        const double phi_Ym = blk.phi(i,  j-1,k  );
        const double phi_Yp = blk.phi(i,  j+1,k  );
        const double phi_Zm = blk.phi(i,  j,  k-1);
        const double phi_Zp = blk.phi(i,  j,  k+1);

        // X: left face (i-1/2) and right face (i+1/2)
        const double u_lx = 0.5*(pXm.u + pC.u);
        const double u_rx = 0.5*(pC.u  + pXp.u);
        const double f_lx = (u_lx >= 0.0) ? u_lx * phi_Xm : u_lx * phi_C;
        const double f_rx = (u_rx >= 0.0) ? u_rx * phi_C  : u_rx * phi_Xp;

        // Y: bottom face (j-1/2) and top face (j+1/2)
        const double v_ly = 0.5*(pYm.v + pC.v);
        const double v_ry = 0.5*(pC.v  + pYp.v);
        const double f_ly = (v_ly >= 0.0) ? v_ly * phi_Ym : v_ly * phi_C;
        const double f_ry = (v_ry >= 0.0) ? v_ry * phi_C  : v_ry * phi_Yp;

        // Z: back face (k-1/2) and front face (k+1/2)
        const double w_lz = 0.5*(pZm.w + pC.w);
        const double w_rz = 0.5*(pC.w  + pZp.w);
        const double f_lz = (w_lz >= 0.0) ? w_lz * phi_Zm : w_lz * phi_C;
        const double f_rz = (w_rz >= 0.0) ? w_rz * phi_C  : w_rz * phi_Zp;

        rhs_blk.phi_data_[flat] += inv_h * ((f_lx - f_rx) + (f_ly - f_ry) + (f_lz - f_rz));
    }
}

// =============================================================================
// P14.1b — ACDI interface-compression source
// =============================================================================
// Two-pass cell-centred scheme (requires NG >= 2):
//   Pass 1: compute compressive flux vector F = ε·(∇φ − φ(1−φ)·n̂) at a
//           (NB+2)³ halo (indices NG-1..NG+NB on each axis).
//           ∇φ: central differences, n̂ = ∇φ/|∇φ| (regularised).
//   Pass 2: central-difference divergence of F at interior cells.
// Net effect: drives interface toward tanh profile of thickness ε=Cε·h.
// Conservative for periodic and natural wall BCs (net flux through ∂Ω = 0).
void phi_compression_rhs(const CellBlock& blk, CellBlock& rhs_blk,
                          double ceps) noexcept
{
    const double h      = blk.h;
    const double eps    = ceps * h;
    const double eps_sq = 1e-10 / (h * h);

    alignas(64) double Fx[NCELL] = {};
    alignas(64) double Fy[NCELL] = {};
    alignas(64) double Fz[NCELL] = {};

    static_assert(NG >= 2, "phi_compression_rhs needs NG>=2 for halo stencil");

    constexpr CellGrad<Axis::X, 2> dX;
    constexpr CellGrad<Axis::Y, 2> dY;
    constexpr CellGrad<Axis::Z, 2> dZ;
    auto Phi = [&blk](int i, int j, int k){ return blk.phi(i, j, k); };

    // Pass 1: compute interface-compression flux F = ε·(∇φ − φ(1−φ)·n̂)
    for (int k = NG-1; k <= NG+NB; ++k)
    for (int j = NG-1; j <= NG+NB; ++j)
    for (int i = NG-1; i <= NG+NB; ++i) {
        const double dpx   = dX(Phi, i, j, k, h);
        const double dpy   = dY(Phi, i, j, k, h);
        const double dpz   = dZ(Phi, i, j, k, h);
        const double mag2  = dpx*dpx + dpy*dpy + dpz*dpz + eps_sq;
        const double inv_mag = 1.0 / std::sqrt(mag2);
        const double phi_c = blk.phi(i, j, k);
        const double g     = phi_c * (1.0 - phi_c);
        const int flat     = cell_idx(i, j, k);
        Fx[flat] = eps * (dpx - g * dpx * inv_mag);
        Fy[flat] = eps * (dpy - g * dpy * inv_mag);
        Fz[flat] = eps * (dpz - g * dpz * inv_mag);
    }

    // Pass 2: central-difference divergence of F → add to rhs phi
    constexpr CellDiv<2> divOp;
    auto Fxa = [&Fx](int i, int j, int k){ return Fx[cell_idx(i,j,k)]; };
    auto Fya = [&Fy](int i, int j, int k){ return Fy[cell_idx(i,j,k)]; };
    auto Fza = [&Fz](int i, int j, int k){ return Fz[cell_idx(i,j,k)]; };

    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        rhs_blk.phi_data_[cell_idx(i,j,k)] += divOp(Fxa, Fya, Fza, i, j, k, h);
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
