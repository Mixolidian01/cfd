#pragma once
// P14.5: GPU ensemble UQ + Ensemble Kalman Filter
//
// EnsembleSolver drives N independent NSSolver members and provides:
//   step_all()        — advance all members one CFL step; returns min dt
//   compute_moments() — ensemble mean + variance per (leaf, cell) for one variable
//   enkf_update()     — stochastic EnKF analysis at a scalar point observation
//
// Each member is a fully independent NSSolver with identical cfg but perturbed IC.
// The GPU path is engaged per-member when cfg.exec.use_gpu = true.
// MPI: mpi_allreduce_sum is used for ensemble moment reduction when
//      EnsembleSolver::mpi is non-null (multi-rank runs).

#include "solver/ns_solver.hpp"
#include <functional>
#include <random>
#include <vector>

struct EnsembleSolver {

    // Per-cell ensemble moments (one Moments per interior cell per leaf)
    struct Moments {
        double mean = 0.0;
        double var  = 0.0;   // unbiased sample variance (divided by N-1)
    };

    // ── Configuration ────────────────────────────────────────────────────────
    int    n_members           = 4;
    double obs_variance        = 1e-4;   // R: measurement noise variance
    double localization_radius = 0.3;    // spatial localization length scale

    // Optional MPI partition shared across all members (same topology).
    MpiPartition* mpi = nullptr;

    // ── Members ───────────────────────────────────────────────────────────────
    // Indexed [0, n_members). Each NSSolver has identical cfg except the IC.
    std::vector<NSSolver> members;

    explicit EnsembleSolver(int n = 4);

    // Initialize all members.
    //   domain_L : physical domain side length
    //   base_cfg : common solver config applied to every member
    //   ic_gen   : ic_gen(member_idx, x, y, z) → Prim — generates per-member IC
    void init(double domain_L,
              const SolverConfig& base_cfg,
              const std::function<Prim(int, double, double, double)>& ic_gen);

    // Advance all members by one CFL timestep; returns min dt across members.
    double step_all();

    // Compute ensemble mean and variance for conserved variable var_idx
    // (0=ρ, 1=ρu, 2=ρv, 3=ρw, 4=E) at every interior cell of every leaf.
    // Layout: [leaf 0 cell 0, ..., leaf 0 cell NB³-1, leaf 1 cell 0, ...]
    // Leaf order = tree.leaf_indices() of member 0 (all members share same topology).
    std::vector<Moments> compute_moments(int var_idx = 0) const;

    // Stochastic EnKF analysis step at a point density observation.
    //   obs_x/y/z : physical location of the observation (density)
    //   obs_val   : observed ρ value
    //   obs_var   : measurement noise variance (<0 → use this->obs_variance)
    // Updates Q[0] (density) for every interior cell of every member.
    // Localization: Gaspari-Cohn function with cut-off 2×localization_radius.
    void enkf_update(double obs_x, double obs_y, double obs_z,
                     double obs_val, double obs_var = -1.0);

private:
    std::mt19937 rng_{42};  // fixed seed — reproducible across test runs

    // Evaluate H(member i): density at (px, py, pz) via nearest-grid-point.
    // Returns NaN if point is outside every leaf's interior.
    std::vector<double> apply_H(double px, double py, double pz) const;

    // Gaspari-Cohn fifth-order piecewise-polynomial localization weight.
    // c = localization_radius; returns 1.0 at r=0, 0.0 for r >= 2c.
    static double gc_weight(double r, double c) noexcept;
};
