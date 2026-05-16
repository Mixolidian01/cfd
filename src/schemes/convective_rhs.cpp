// =============================================================================
// convective_rhs.cpp — R9-E3: convective RHS kernel extracted from operators.cpp
// =============================================================================
// Contains: weno5_face_t, is_wall_ghost, face_lr_idx, face_to_ijk,
//           accumulate_face, convective_rhs_impl.
// kep_flux_t lives in operators.hpp (inline template, shared with operators.cpp).
// Called from: compute_rhs / compute_rhs_typed (operators.cpp) via forward decl.

#include "schemes/operators.hpp"
#include "physics/weno5_recon.hpp"
#include "profiling/profiler.hpp"

#include <cmath>
#include <algorithm>

// P13.1 stage 3 — compile-time axis; R2: delegates to Weno5Recon<DIR> functor.
template<Axis DIR>
static void weno5_face_t(const Prim* pc, int i, int j, int k,
                         Prim& qL_out, Prim& qR_out) noexcept {
    Weno5Recon<DIR>{}(pc, i, j, k, qL_out, qR_out);
}

// is_wall_ghost: detect no-slip wall ghost face.
// Exact float equality p_L == p_R (from symmetric energy ghost fill) plus
// anti-symmetric velocities uniquely identifies wall fill_ghosts_wall output.
static bool is_wall_ghost(const Prim& pL, const Prim& pR) noexcept {
    if (pL.p != pR.p) return false;
    auto antisym = [](double a, double b) noexcept -> bool {
        return std::fabs(a + b) < 1e-8 * (std::fabs(a) + std::fabs(b) + 1e-300);
    };
    return antisym(pL.u, pR.u) && antisym(pL.v, pR.v) && antisym(pL.w, pR.w);
}

// face_to_ijk<DIR>: convert face coords (n, a, b) to (xi, yi, zi) of the left
// cell, for passing into Weno5Recon which still uses natural (i,j,k) indexing.
template <Axis DIR>
static inline void face_to_ijk(int n, int a, int b,
                                int& xi, int& yi, int& zi) noexcept {
    if constexpr (DIR == Axis::X) { xi = n; yi = a; zi = b; }
    else if constexpr (DIR == Axis::Y) { xi = a; yi = n; zi = b; }
    else                               { xi = a; yi = b; zi = n; }
}

// accumulate_face<DIR>: compute the flux at one face and accumulate into rhs.
// R6: rhs updates use axis_view<DIR> — no axis dispatch in the accumulation.
// See operators.cpp P15.1 / P3.2 design notes for the full hybrid scheme doc.
template <Axis DIR>
static void accumulate_face(const Prim* pc, const double* duc,
                             CellBlock& rhs, double ih,
                             int n, int a, int b) noexcept {
    constexpr double kep_threshold = 1.0e-8;

    // R6: axis_view maps (n,a,b) → the correct flat index for each axis.
    // Li = cell_idx for the left cell, Ri for the right — no axis branch needed.
    const int Li = cell_idx_axis<DIR>(n,   a, b);
    const int Ri = cell_idx_axis<DIR>(n+1, a, b);
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
            weno5_face_t<DIR>(pc, xi, yi, zi, qL, qR);
            Fs = hllc_es_flux_t<DIR>(qL, qR);
        } else if (is_wall_ghost(pL, pR)) {
            Fs = Fk;
        } else {
            Fs = hllc_es_flux_t<DIR>(pL, pR);
        }
        const double th = is_bnd ? 1.0 : theta;
        const double om = 1.0 - th;
        for (int v = 0; v < NVAR; ++v) F[v] = om * Fk[v] + th * Fs[v];
    }

    // R6: write through axis_view — accumulation is axis-agnostic.
    if (n >= ilo())
        for (int v = 0; v < NVAR; ++v) rhs.axis_view<DIR>(v)(n,   a, b) -= ih * F[v];
    if (n+1 <= ihi())
        for (int v = 0; v < NVAR; ++v) rhs.axis_view<DIR>(v)(n+1, a, b) += ih * F[v];
}

// =============================================================================
// convective_rhs_impl — the face-centred hybrid loop (P2.2/P3.2/P15.1)
// Non-static: called from compute_rhs / compute_rhs_typed in operators.cpp.
// =============================================================================
void convective_rhs_impl(const Prim* pc, const double* duc,
                          CellBlock& rhs, double h) noexcept
{
    PROFILE_SCOPE("convective_rhs_impl");
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
