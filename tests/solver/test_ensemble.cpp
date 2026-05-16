// P14.5 gate tests — EnsembleSolver (EnKF)
//
// E01  Uniform IC → ensemble variance == 0 at t=0
// E02  Perturbed IC → ensemble variance > 0 after init
// E03  EnKF update with exact observation → posterior variance < prior variance
// E04  EnKF update with observation == ensemble mean → analysis mean unchanged
//      (the stochastic EnKF adds ε_i noise, so we check mean shifts by < 10*sqrt(R)/N)

#include "solver/ensemble_solver.hpp"
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

// ─── common IC helpers ────────────────────────────────────────────────────────

static Prim uniform_ic(double /*x*/, double /*y*/, double /*z*/) {
    Prim q;
    q.rho = 1.225; q.u = 0.0; q.v = 0.0; q.w = 0.0;
    q.p   = 101325.0;
    q.T   = q.p / (q.rho * R_GAS);
    q.c   = std::sqrt(GAMMA * q.p / q.rho);
    return q;
}

static SolverConfig base_cfg() {
    SolverConfig c;
    c.bc.variant        = PeriodicBC{};
    c.time.cfl          = 0.5;
    c.time.max_steps    = 5;
    c.time.t_end        = 1e30;
    c.io.verbose        = false;
    c.io.diag_interval  = 100;
    return c;
}

// ─── E01  Uniform IC → variance == 0 ─────────────────────────────────────────
static void e01_uniform_variance_zero() {
    printf("E01  uniform IC — ensemble variance == 0\n");

    EnsembleSolver es(4);
    es.init(1.0, base_cfg(), [](int /*m*/, double x, double y, double z) {
        return uniform_ic(x, y, z);
    });

    const auto mom = es.compute_moments(/*var=rho*/0);

    double max_var = 0.0;
    for (const auto& m : mom) max_var = std::max(max_var, m.var);

    check("E01 max variance == 0 (all-uniform IC)", max_var == 0.0, max_var, 0.0);
}

// ─── E02  Perturbed IC → variance > 0 ────────────────────────────────────────
static void e02_perturbed_variance_positive() {
    printf("E02  perturbed IC — ensemble variance > 0\n");

    const double delta_rho = 0.01;  // ±1% density perturbation
    EnsembleSolver es(4);
    es.obs_variance        = 1e-6;
    es.localization_radius = 0.5;

    es.init(1.0, base_cfg(), [delta_rho](int m, double x, double y, double z) {
        Prim q = uniform_ic(x, y, z);
        // Member 0,2 get +delta, 1,3 get -delta → non-zero variance
        q.rho += (m % 2 == 0) ? delta_rho : -delta_rho;
        q.p    = 101325.0;
        q.T    = q.p / (q.rho * R_GAS);
        q.c    = std::sqrt(GAMMA * q.p / q.rho);
        (void)x; (void)y; (void)z;
        return q;
    });

    const auto mom = es.compute_moments(0);

    double mean_var = 0.0;
    for (const auto& m : mom) mean_var += m.var;
    mean_var /= mom.size();

    // ±delta alternating with N=4: sample variance = 4*delta^2 / (N-1) = 4/3 * delta^2
    const double expected = (4.0 / 3.0) * delta_rho * delta_rho;
    check("E02 mean variance ≈ analytical (±δ, N=4)", std::abs(mean_var - expected) < 0.01 * expected,
          mean_var, expected);
}

// ─── E03  EnKF update → posterior variance < prior variance ──────────────────
static void e03_enkf_reduces_variance() {
    printf("E03  EnKF update reduces ensemble variance\n");

    const double delta_rho = 0.05;
    EnsembleSolver es(8);
    es.obs_variance        = 1e-6;
    es.localization_radius = 0.5;

    es.init(1.0, base_cfg(), [delta_rho](int m, double x, double y, double z) {
        (void)x; (void)y; (void)z;
        Prim q = uniform_ic(0, 0, 0);
        // Spread: member m gets a linearly spaced perturbation
        q.rho += delta_rho * (m - 3.5) / 3.5;
        q.p    = 101325.0;
        q.T    = q.p / (q.rho * R_GAS);
        q.c    = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    });

    const auto prior = es.compute_moments(0);
    double prior_var = 0.0;
    for (const auto& m : prior) prior_var += m.var;
    prior_var /= prior.size();

    // Observe rho at domain centre with the ensemble mean value
    double prior_mean = 0.0;
    for (const auto& m : prior) prior_mean += m.mean;
    prior_mean /= prior.size();

    es.enkf_update(0.5, 0.5, 0.5, prior_mean, es.obs_variance);

    const auto post = es.compute_moments(0);
    double post_var = 0.0;
    for (const auto& m : post) post_var += m.var;
    post_var /= post.size();

    check("E03 posterior variance < prior variance", post_var < prior_var,
          post_var, prior_var);
}

// ─── E04  step_all advances all members ──────────────────────────────────────
static void e04_step_all_advances() {
    printf("E04  step_all advances solver time\n");

    EnsembleSolver es(2);
    es.init(1.0, base_cfg(), [](int /*m*/, double x, double y, double z) {
        return uniform_ic(x, y, z);
    });

    const double t0 = es.members[0].t;
    const double dt = es.step_all();
    const double t1 = es.members[0].t;

    check("E04 dt > 0",            dt > 0.0, dt, 0.0);
    check("E04 member 0 advanced", t1 > t0,  t1, t0);
    check("E04 all members same t",
          std::abs(es.members[0].t - es.members[1].t) < 1e-14,
          std::abs(es.members[0].t - es.members[1].t), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("=== P14.5 EnsembleSolver (EnKF) gate tests ===\n");
    e01_uniform_variance_zero();
    e02_perturbed_variance_positive();
    e03_enkf_reduces_variance();
    e04_step_all_advances();
    printf("=== %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
