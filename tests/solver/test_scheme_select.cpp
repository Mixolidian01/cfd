// R5 gate tests — instantiation matrix + runtime scheme selection
//
// S01  HLLC and HLLC-ES produce distinct density after 10 steps on a Sod IC
//      (entropy stabilisation changes the numerical flux — results must differ)
// S02  HLLC and HLLC-ES conserve mass to the same tolerance (both are conservative)
// S03  Scheme selection survives regrid (rhs_fn_ still valid after tree changes)

#include "solver/ns_solver.hpp"
#include "mesh/bc_types.hpp"
#include <cmath>
#include <cstdio>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool cond, double got = -1, double thr = -1) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.6e  thr %.6e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// Sod shock tube: left/right states separated at x=0.5
static Prim sod_ic(double x, double /*y*/, double /*z*/) {
    Prim q;
    if (x < 0.5) { q.rho = 1.0; q.p = 1.0; }
    else          { q.rho = 0.125; q.p = 0.1; }
    q.u = q.v = q.w = 0.0;
    q.T = q.p / (q.rho * R_GAS);
    q.c = std::sqrt(GAMMA * q.p / q.rho);
    return q;
}

static SolverConfig make_cfg(SolverConfig::FluxScheme fs) {
    SolverConfig c;
    c.exec.flux_scheme  = fs;
    c.bc.variant        = WallBC{};   // wall BC so Sod doesn't go periodic
    c.time.cfl          = 0.4;
    c.time.max_steps    = 10;
    c.time.t_end        = 1e30;
    c.io.verbose        = false;
    c.io.diag_interval  = 100;
    return c;
}

// ─── S01  distinct results ────────────────────────────────────────────────────
static void s01_distinct_results() {
    printf("S01  HLLC vs HLLC-ES → distinct density after 10 steps\n");

    NSSolver es_solver, hllc_solver;
    es_solver.cfg   = make_cfg(SolverConfig::FluxScheme::HLLC_ES);
    hllc_solver.cfg = make_cfg(SolverConfig::FluxScheme::HLLC);

    es_solver.init(1.0, sod_ic);
    hllc_solver.init(1.0, sod_ic);
    es_solver.run();
    hllc_solver.run();

    // Compute max pointwise density difference
    const auto& leaves = es_solver.tree.leaf_indices();
    double max_diff = 0.0;
    for (int ni : leaves) {
        const CellBlock* bES   = es_solver.tree.nodes[ni].block.get();
        const CellBlock* bHLLC = hllc_solver.tree.nodes[ni].block.get();
        if (!bES || !bHLLC) continue;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i, j, k);
            max_diff = std::max(max_diff, std::abs(bES->Q[0][f] - bHLLC->Q[0][f]));
        }
    }
    check("S01 HLLC_ES != HLLC (max density diff > 1e-10)", max_diff > 1e-10, max_diff, 1e-10);
}

// ─── S02  both schemes conserve mass ─────────────────────────────────────────
static void s02_both_conserve_mass() {
    printf("S02  both HLLC and HLLC-ES conserve mass (periodic BC, 20 steps)\n");

    for (auto fs : {SolverConfig::FluxScheme::HLLC_ES, SolverConfig::FluxScheme::HLLC}) {
        NSSolver s;
        s.cfg = make_cfg(fs);
        s.cfg.bc.variant       = PeriodicBC{};
        s.cfg.time.max_steps   = 20;
        // Smooth periodic IC
        s.init(1.0, [](double x, double y, double z) {
            (void)y; (void)z;
            Prim q;
            q.rho = 1.0 + 0.1 * std::sin(2 * M_PI * x);
            q.u = 0.1; q.v = 0.0; q.w = 0.0;
            q.p = 101325.0;
            q.T = q.p / (q.rho * R_GAS);
            q.c = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        });
        const double m0 = s.compute_diag().mass;
        s.run();
        const double m1   = s.compute_diag().mass;
        const double err  = std::abs(m1 - m0) / std::abs(m0);
        const char*  name = (fs == SolverConfig::FluxScheme::HLLC_ES)
                          ? "S02 HLLC-ES mass conservation < 1e-10"
                          : "S02 HLLC mass conservation < 1e-10";
        check(name, err < 1e-10, err, 1e-10);
    }
}

// ─── S03  scheme survives regrid ─────────────────────────────────────────────
static void s03_scheme_survives_regrid() {
    printf("S03  scheme selection valid after regrid (HLLC-ES, AMR)\n");

    NSSolver s;
    s.cfg = make_cfg(SolverConfig::FluxScheme::HLLC_ES);
    s.cfg.amr.max_level       = 1;
    s.cfg.amr.regrid_interval = 2;
    s.cfg.time.max_steps      = 6;
    s.cfg.bc.variant          = WallBC{};

    s.init(1.0, sod_ic);
    bool ok = true;
    try { s.run(); } catch (...) { ok = false; }
    check("S03 HLLC-ES + AMR regrid: no exception in 6 steps", ok);
}

int main() {
    printf("=== R5 scheme selection gate tests ===\n");
    s01_distinct_results();
    s02_both_conserve_mass();
    s03_scheme_survives_regrid();
    printf("=== %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
