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
//
// R9-E3: operators.cpp now contains only top-level glue — tree_rhs, compute_rhs,
//        tree_cfl_dt, and AMR flux-correction calls.  Kernel implementations have
//        been extracted into focused translation units:
//          convective_rhs.cpp  — convective_rhs_impl + face-loop helpers
//          viscous_rhs.cpp     — viscous_rhs_impl + cf_visc_energy_flux + undo_cf_viscous_energy
//          rhs_sensors.cpp     — fill_ducros_cache + phi_rhs + phi_compression_rhs + tree_sat_penalty

#include "schemes/operators.hpp"
#include "schemes/concepts.hpp"
#include "physics/hllc_flux.hpp"
#include "physics/weno5_recon.hpp"
#include "physics/diff_ops.hpp"
#include "profiling/profiler.hpp"
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

// P14.1: Stiffened-gas EOS state lives in CellBlock statics (single source of truth).
// operators.cpp reads CellBlock::sg_active_ / sg_ga_ / sg_gb_ / sg_pia_ / sg_pib_ directly.

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
    if (!CellBlock::sg_active_) {
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
            mix_eos(blk.phi_data_[flat], CellBlock::sg_ga_, CellBlock::sg_gb_,
                    CellBlock::sg_pia_, CellBlock::sg_pib_, gm, pim);
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
// Forward declarations for kernels extracted to separate translation units
// (R9-E3).  These are non-static functions with internal linkage removed so
// they can be resolved at link time.
// =============================================================================
void convective_rhs_impl(const Prim* pc, const double* duc,
                          CellBlock& rhs, double h) noexcept;
void viscous_rhs_impl(const Prim* pc, const double* mu_arr,
                      CellBlock& rhs, double h) noexcept;
void undo_cf_viscous_energy(const BlockTree& tree, int node_idx,
                             CellBlock& rhs) noexcept;
void fill_ducros_cache(const Prim* pc, double* duc, double h,
                       const DucrosConfig& ducros) noexcept;

// =============================================================================
// R5 — Typed convective kernel: Flux × Recon resolved at compile time
// =============================================================================
// accumulate_face_typed / convective_rhs_impl_typed are template helpers that
// live here because compute_rhs_typed (also a template) instantiates them.
// kep_flux_t is inline in operators.hpp (shared with convective_rhs.cpp).

template<Axis DIR, template<Axis> class Flux, template<Axis> class Recon>
static void accumulate_face_typed(const Prim* pc, const double* duc,
                                   CellBlock& rhs, double ih,
                                   int n, int a, int b) noexcept {
    constexpr double kep_threshold = 1.0e-8;
    // face_lr_idx / face_to_ijk logic inlined here (static templates in
    // convective_rhs.cpp have TU-private linkage; duplicate minimal helpers).
    auto face_lr = [](int nn, int aa, int bb) -> std::pair<int,int> {
        if constexpr (DIR == Axis::X) return {cell_idx(nn,  aa,bb), cell_idx(nn+1,aa,bb)};
        if constexpr (DIR == Axis::Y) return {cell_idx(aa,nn,  bb), cell_idx(aa,nn+1,bb)};
        return                              {cell_idx(aa,bb,nn  ), cell_idx(aa,bb,nn+1)};
    };
    auto face_to_ijk = [](int nn, int aa, int bb, int& xi, int& yi, int& zi) {
        if constexpr (DIR == Axis::X) { xi = nn; yi = aa; zi = bb; }
        else if constexpr (DIR == Axis::Y) { xi = aa; yi = nn; zi = bb; }
        else                               { xi = aa; yi = bb; zi = nn; }
    };

    const auto [Li, Ri] = face_lr(n, a, b);
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
            face_to_ijk(n, a, b, xi, yi, zi);
            Prim qL, qR;
            Recon<DIR>{}(pc, xi, yi, zi, qL, qR);
            Fs = Flux<DIR>{}(qL, qR);
        } else {
            // is_wall_ghost detection: exact p equality + antisymmetric velocities.
            // Reproduce inline (static helper is TU-private in convective_rhs.cpp).
            bool wall = (pL.p == pR.p);
            if (wall) {
                auto antisym = [](double a_, double b_) {
                    return std::fabs(a_+b_) < 1e-8*(std::fabs(a_)+std::fabs(b_)+1e-300);
                };
                wall = antisym(pL.u,pR.u) && antisym(pL.v,pR.v) && antisym(pL.w,pR.w);
            }
            Fs = wall ? Fk : Flux<DIR>{}(pL, pR);
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
// Public wrappers — preserve API for standalone test calls
// =============================================================================
void convective_rhs(const CellBlock& blk, CellBlock& rhs_blk) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> duc;
    fill_prim_cache(blk, pc.data());
    fill_ducros_cache(pc.data(), duc.data(), blk.h, DucrosConfig{});
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
void compute_rhs(const CellBlock& blk, CellBlock& rhs_blk,
                 const DucrosConfig& ducros) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> mu_arr;
    static thread_local std::array<double, NCELL> duc;
    fill_prim_cache(blk, pc.data());
    fill_mu_cache(pc.data(), mu_arr.data());
    fill_ducros_cache(pc.data(), duc.data(), blk.h, ducros);

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
// (fill_prim_cache, fill_ducros_cache, etc.) are TU-private.
// Explicit instantiations at the bottom of this file satisfy the ODR for all
// supported combinations; callers link to those pre-compiled specialisations.
template<template<Axis> class Flux, template<Axis> class Recon, class EOS>
    requires RiemannFlux<Flux<Axis::X>>
          && SpatialReconstruction<Recon<Axis::X>>
          && EquationOfState<EOS>
void compute_rhs_typed(const CellBlock& blk, CellBlock& rhs_blk,
                       const DucrosConfig& ducros) noexcept
{
    static thread_local std::array<Prim,   NCELL> pc;
    static thread_local std::array<double, NCELL> mu_arr;
    static thread_local std::array<double, NCELL> duc;
    fill_prim_cache(blk, pc.data());
    fill_mu_cache(pc.data(), mu_arr.data());
    fill_ducros_cache(pc.data(), duc.data(), blk.h, ducros);

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
    const CellBlock&, CellBlock&, const DucrosConfig&) noexcept;
template void compute_rhs_typed<HllcFlux,   Weno5Recon, IdealGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&) noexcept;

#include "physics/stiffened_gas_eos.hpp"

template void compute_rhs_typed<HllcEsFlux, Weno5Recon, StiffenedGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&) noexcept;
template void compute_rhs_typed<HllcFlux, Weno5Recon, StiffenedGasEOS>(
    const CellBlock&, CellBlock&, const DucrosConfig&) noexcept;

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
// accumulate_cf_fine_fluxes
// =============================================================================
// cf_accum_one_face<DIR>: accumulates PCM+HLLC-ES fine-face flux into the
// Berger-Colella register for one face direction.
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

        const int parent_idx = nd.parent;
        const int oct = (parent_idx >= 0)
                        ? li - tree.nodes[parent_idx].first_child
                        : 0;
        const int o_ix = oct_ix(oct);
        const int o_iy = oct_iy(oct);
        const int o_iz = oct_iz(oct);

        for (int d = 0; d < NFACES; ++d) {
            const int ni = nd.neighbours[d];
            if (ni < 0 || !tree.nodes[ni].has_block()) continue;
            if (tree.nodes[ni].level >= nd.level) continue;

            const int axis  = face_axis[d];
            const int delta = face_delta[d];
            const int bound = (delta > 0) ? ihi() : ilo();

            int off1, off2;
            if      (axis == 0) { off1 = o_iy; off2 = o_iz; }
            else if (axis == 1) { off1 = o_iz; off2 = o_ix; }
            else                { off1 = o_iy; off2 = o_ix; }

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
// =============================================================================
void tree_rhs(BlockTree& tree,
              std::vector<CellBlock>& rhs_blocks,
              bool periodic,
              double stage_weight,
              int    level_filter,
              bool   cf_coarse_zero_grad,
              bool   open_bc,
              const DucrosConfig& ducros) noexcept
{
    PROFILE_SCOPE("tree_rhs");

    // ── 1. Ghost fill — always global (C/F fills need the full tree) ─────────
    { PROFILE_SCOPE("tree_rhs/ghost_fill");
      if (periodic)
          tree.fill_ghosts_periodic(cf_coarse_zero_grad);
      else if (open_bc)
          tree.fill_ghosts_open(cf_coarse_zero_grad);
      else
          tree.fill_ghosts_wall(cf_coarse_zero_grad);
    }

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
    { PROFILE_SCOPE("tree_rhs/compute");
#pragma omp parallel for schedule(dynamic,4)
      for (int oi = 0; oi < n_active; ++oi) {
          const int li       = order[oi];
          const int node_idx = leaves[li];
          if (!tree.nodes[node_idx].has_block()) continue;  // P7.1: remote leaf
          compute_rhs(*tree.nodes[node_idx].block, rhs_blocks[li], ducros);
          undo_cf_face_flux(tree, node_idx, rhs_blocks[li]);
          if (cf_coarse_zero_grad)
              undo_cf_viscous_energy(tree, node_idx, rhs_blocks[li]);
      }
    }

    // ── 4. Flux register accumulation — serial (shared CF register writes) ───
    { PROFILE_SCOPE("tree_rhs/flux_accum");
      accumulate_cf_fine_fluxes(tree, stage_weight, level_filter);
    }
}

// =============================================================================
// CFL time step
// =============================================================================
double tree_cfl_dt(const BlockTree& tree, double cfl) noexcept
{
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();
    double dt = 1e300;
#pragma omp parallel for reduction(min:dt)
    for (int i = 0; i < NL; ++i) {
        int li = leaves[i];
        if (!tree.nodes[li].has_block()) continue;  // P7.1: remote leaf
        double cdt = tree.nodes[li].block->cfl_dt(cfl);
        if (cdt < dt) dt = cdt;
    }
    return dt;
}

// P4.1: minimum CFL dt over leaves at a specific refinement level.
double level_cfl_dt(const BlockTree& tree, int level, double cfl) noexcept
{
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();
    double dt = 1e300;
#pragma omp parallel for reduction(min:dt)
    for (int i = 0; i < NL; ++i) {
        int li = leaves[i];
        if (tree.nodes[li].level != level) continue;
        if (!tree.nodes[li].has_block()) continue;  // P7.1: remote leaf
        double cdt = tree.nodes[li].block->cfl_dt(cfl);
        if (cdt < dt) dt = cdt;
    }
    return dt;
}
