// D11-adj: adjoint_rk3_one_block — reverses three SSP-RK3 Shu-Osher stages.
// Depends only on adjoint_rhs (operators library) and CellBlock (block library).
#include "solver/adjoint_rk3.hpp"
#include "schemes/adjoint_rhs.hpp"

void adjoint_rk3_one_block(
    const CellBlock& Qs0,
    const CellBlock& Qs1,
    const CellBlock& Qs2,
    double dt,
    const CellBlock& lam_f,
    CellBlock& lam_Qn) noexcept
{
    CellBlock seed{}; seed.h = Qs0.h;
    CellBlock lam_Q2{}; lam_Q2.h = Qs0.h;
    CellBlock lam_Q1{}; lam_Q1.h = Qs0.h;

    // ── Reverse stage 3 ────────────────────────────────────────────────────
    for (int v = 0; v < NVAR; ++v)
    for (int f = 0; f < NCELL; ++f)
        seed.Q[v][f] = (2.0/3.0)*dt * lam_f.Q[v][f];
    adjoint_rhs(Qs2, seed, lam_Q2);
    for (int v = 0; v < NVAR; ++v)
    for (int f = 0; f < NCELL; ++f) {
        lam_Q2.Q[v][f] += (2.0/3.0)*lam_f.Q[v][f];
        lam_Qn.Q[v][f] += (1.0/3.0)*lam_f.Q[v][f];
    }

    // ── Reverse stage 2 ────────────────────────────────────────────────────
    for (int v = 0; v < NVAR; ++v)
    for (int f = 0; f < NCELL; ++f)
        seed.Q[v][f] = (1.0/4.0)*dt * lam_Q2.Q[v][f];
    adjoint_rhs(Qs1, seed, lam_Q1);
    for (int v = 0; v < NVAR; ++v)
    for (int f = 0; f < NCELL; ++f) {
        lam_Q1.Q[v][f] += (1.0/4.0)*lam_Q2.Q[v][f];
        lam_Qn.Q[v][f] += (3.0/4.0)*lam_Q2.Q[v][f];
    }

    // ── Reverse stage 1 ────────────────────────────────────────────────────
    for (int v = 0; v < NVAR; ++v)
    for (int f = 0; f < NCELL; ++f)
        seed.Q[v][f] = dt * lam_Q1.Q[v][f];
    adjoint_rhs(Qs0, seed, lam_Qn);
    for (int v = 0; v < NVAR; ++v)
    for (int f = 0; f < NCELL; ++f)
        lam_Qn.Q[v][f] += lam_Q1.Q[v][f];
}
