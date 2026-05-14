// test_sat.cpp — F2: SBP-SAT penalty energy balance test
//
// Gate criterion:
//   TSat  tree_sat_penalty is globally conservative: the volume-weighted sum
//         of all RHS contributions across every leaf block is zero to < 1e-13.
//
// Physical basis:
//   tree_sat_penalty adds sigma*(Q_ghost - Q_interior)/h_f to fine boundary
//   cells and subtracts the matching averaged correction from the coarse
//   neighbour.  The two contributions cancel to machine precision, so no net
//   energy (or any conserved variable) is added or removed by the penalty.
//
// Setup:
//   2-level tree: root block + 8 fine children (one refine of root).
//   Periodic BC.  Smooth initial condition (uniform density, pressure and
//   small sinusoidal velocity) so ghosts are non-trivial and penalties are
//   non-zero.

#include "ns_solver.hpp"
#include "operators.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

static int n_pass = 0, n_fail = 0;

static void check(const char* name, bool cond, double got = -1, double thr = -1) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.6e  threshold %.6e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// =============================================================================
// TSat — SBP-SAT penalty is globally conservative (zero net energy injection)
// =============================================================================
static void test_sat_energy_balance() {
    // ── Build a 2-level tree ──────────────────────────────────────────────────
    NSSolver s;
    s.cfg.time.cfl             = 0.3;
    s.cfg.time.max_steps       = 1;        // minimal — we only need tree setup, not time-stepping
    s.cfg.time.t_end           = 1e30;
    s.cfg.bc.variant      = PeriodicBC{};
    s.cfg.io.verbose         = false;
    s.cfg.io.diag_interval   = 1000;
    s.cfg.amr.regrid_interval = 0;
    s.cfg.amr.max_level       = 1;
    s.cfg.numerics.sat_tau         = 0.5;

    // Smooth IC: uniform flow with a small sinusoidal density perturbation so
    // that ghost values differ from interior values and penalties are non-zero.
    auto ic = [](double x, double y, double z) {
        Prim q;
        q.rho = 1.0 + 0.1 * std::sin(2.0 * M_PI * x)
                     + 0.05 * std::cos(2.0 * M_PI * y)
                     + 0.03 * std::sin(2.0 * M_PI * z);
        q.u   = 0.05 * std::sin(2.0 * M_PI * y);
        q.v   = 0.03 * std::cos(2.0 * M_PI * x);
        q.w   = 0.02 * std::sin(2.0 * M_PI * z);
        q.p   = 101325.0;
        q.T   = q.p / (q.rho * R_GAS);
        q.c   = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    };

    s.init(1.0, ic);

    // Refine the root to create 8 level-1 fine leaves — a single coarse block
    // is now replaced by 8 fine blocks, giving C/F interfaces (the root's
    // former periodic neighbours are now level-0 and level-1 blocks).
    // Because all children share the same parent, all 6 faces of every fine
    // block are adjacent to a fine sibling except the outer faces which are
    // adjacent to the root-level periodic image (coarse).  However,
    // fill_ghosts_periodic uses the root image which may be the coarse level.
    // For simplicity we just check that the function runs without crashing and
    // that the global sum is zero.
    s.tree.refine(s.tree.root());
    s.tree.balance();
    s.tree.rebuild_neighbours();
    s.tree.fill_ghosts_periodic();

    // ── Build rhs scratch matching leaf geometry ──────────────────────────────
    const auto& leaves = s.tree.leaf_indices();
    const int NL = (int)leaves.size();

    std::vector<CellBlock> rhs_blocks;
    rhs_blocks.reserve(NL);
    for (int li : leaves) {
        const auto& blk = *s.tree.nodes[li].block;
        rhs_blocks.emplace_back(blk.ox, blk.oy, blk.oz, blk.h);
        // rhs is zero-initialised by CellBlock default constructor
    }

    // ── Apply SAT penalty ─────────────────────────────────────────────────────
    tree_sat_penalty(s.tree, rhs_blocks, s.cfg.numerics.sat_tau);

    // ── Compute volume-weighted RHS sum for each conserved variable ───────────
    // delta_Q[v] = sum_{leaves} sum_{interior cells} rhs[v][i,j,k] * h^3
    // Must be zero for all v if the penalty is conservative.
    std::array<double, NVAR> delta_Q{};
    for (int ii = 0; ii < NL; ++ii) {
        const CellBlock& rb = rhs_blocks[ii];
        const double h3 = rb.h * rb.h * rb.h;
        for (int v = 0; v < NVAR; ++v) {
            double sum = 0.0;
            for (int k = ilo(); k <= ihi(); ++k)
            for (int j = ilo(); j <= ihi(); ++j)
            for (int i = ilo(); i <= ihi(); ++i)
                sum += rb.Q[v][cell_idx(i, j, k)];
            delta_Q[v] += sum * h3;
        }
    }

    const char* var_names[NVAR] = {"rho","rhou","rhov","rhow","E"};
    for (int v = 0; v < NVAR; ++v) {
        char label[64];
        std::snprintf(label, sizeof(label),
                      "TSat SAT penalty is conservative for %s (|sum| < 1e-13)",
                      var_names[v]);
        check(label, std::abs(delta_Q[v]) < 1e-13, std::abs(delta_Q[v]), 1e-13);
    }
}

int main() {
    printf("=== test_sat: SBP-SAT penalty energy balance ===\n");
    test_sat_energy_balance();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    return (n_fail == 0) ? 0 : 1;
}
