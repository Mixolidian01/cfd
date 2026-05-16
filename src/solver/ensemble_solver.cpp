// P14.5: Ensemble Kalman Filter + GPU ensemble UQ
//
// Stochastic EnKF (Burgers et al. 1998) for a scalar point density observation.
// Localization uses the Gaspari-Cohn (1999) fifth-order polynomial, cut off at 2c.
//
// Variable layout: Q[v][cell_idx(i,j,k)] with i,j,k ∈ [NG, NG+NB-1] interior.

#include "solver/ensemble_solver.hpp"
#include "mpi/mpi_comm.hpp"
#include "mesh/cell_block.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

// ─── helpers ─────────────────────────────────────────────────────────────────

// Gaspari-Cohn 1999, eq. (4.10).  r in [0, 2c] → [0, 1].
double EnsembleSolver::gc_weight(double r, double c) noexcept {
    if (c <= 0.0) return 1.0;
    const double z = r / c;
    if (z >= 2.0) return 0.0;
    if (z < 1.0) {
        return 1.0 - (5.0/3.0)*z*z + (5.0/8.0)*z*z*z + (1.0/2.0)*z*z*z*z - (1.0/4.0)*z*z*z*z*z;
    }
    // 1 ≤ z < 2
    return (4.0 - 5.0*z + (5.0/3.0)*z*z + (5.0/8.0)*z*z*z
            - (1.0/2.0)*z*z*z*z + (1.0/12.0)*z*z*z*z*z) - (2.0/(3.0*z));
}

// ─── constructor ─────────────────────────────────────────────────────────────

EnsembleSolver::EnsembleSolver(int n) : n_members(n) {
    if (n < 2) throw std::invalid_argument("EnsembleSolver: n_members must be >= 2");
    members.resize(n);
}

// ─── init ────────────────────────────────────────────────────────────────────

void EnsembleSolver::init(double domain_L,
                          const SolverConfig& base_cfg,
                          const std::function<Prim(int, double, double, double)>& ic_gen) {
    for (int m = 0; m < n_members; ++m) {
        members[m].cfg = base_cfg;
        if (mpi) members[m].set_mpi(mpi);
        members[m].init(domain_L, [m, &ic_gen](double x, double y, double z) {
            return ic_gen(m, x, y, z);
        });
    }
}

// ─── step_all ────────────────────────────────────────────────────────────────

double EnsembleSolver::step_all() {
    double dt_min = std::numeric_limits<double>::max();
    for (auto& s : members) {
        const double dt = s.advance();
        dt_min = std::min(dt_min, dt);
    }
    return mpi_allreduce_min(dt_min, mpi);
}

// ─── compute_moments ─────────────────────────────────────────────────────────

std::vector<EnsembleSolver::Moments>
EnsembleSolver::compute_moments(int var_idx) const {
    assert(!members.empty());
    const auto& ref_tree = members[0].tree;
    const auto& leaves   = ref_tree.leaf_indices();
    const int   N        = n_members;

    // Flat output: all interior cells of all leaves, in leaf_indices() order.
    const int n_interior = NB * NB * NB;
    std::vector<Moments> out(leaves.size() * n_interior);

    for (int li = 0; li < (int)leaves.size(); ++li) {
        const int ni = leaves[li];
        for (int iz = 0; iz < NB; ++iz) {
        for (int iy = 0; iy < NB; ++iy) {
        for (int ix = 0; ix < NB; ++ix) {
            const int flat  = cell_idx(ix+NG, iy+NG, iz+NG);
            const int out_i = li * n_interior + iz*NB*NB + iy*NB + ix;

            // Welford online mean + variance
            double mean = 0.0, M2 = 0.0;
            for (int m = 0; m < N; ++m) {
                const CellBlock* blk = members[m].tree.nodes[ni].block.get();
                const double val = blk->Q[var_idx][flat];
                const double delta = val - mean;
                mean += delta / (m + 1);
                M2   += delta * (val - mean);
            }
            out[out_i].mean = mean;
            out[out_i].var  = (N > 1) ? M2 / (N - 1) : 0.0;
        }}}
    }
    return out;
}

// ─── apply_H: nearest-grid-point density at (px, py, pz) ────────────────────

std::vector<double> EnsembleSolver::apply_H(double px, double py, double pz) const {
    std::vector<double> hx(n_members, std::numeric_limits<double>::quiet_NaN());

    const auto& leaves = members[0].tree.leaf_indices();

    for (int li = 0; li < (int)leaves.size(); ++li) {
        const int ni = leaves[li];
        const CellBlock* ref = members[0].tree.nodes[ni].block.get();
        if (!ref) continue;

        const double bx = ref->ox, by = ref->oy, bz = ref->oz;
        const double h  = ref->h;
        const double bxe = bx + NB * h, bye = by + NB * h, bze = bz + NB * h;

        if (px < bx || px >= bxe) continue;
        if (py < by || py >= bye) continue;
        if (pz < bz || pz >= bze) continue;

        // Convert physical → interior cell index ∈ [0, NB-1]
        int ix = static_cast<int>((px - bx) / h);
        int iy = static_cast<int>((py - by) / h);
        int iz = static_cast<int>((pz - bz) / h);
        ix = std::clamp(ix, 0, NB-1);
        iy = std::clamp(iy, 0, NB-1);
        iz = std::clamp(iz, 0, NB-1);

        const int flat = cell_idx(ix+NG, iy+NG, iz+NG);
        for (int m = 0; m < n_members; ++m) {
            const CellBlock* blk = members[m].tree.nodes[ni].block.get();
            hx[m] = blk->Q[0][flat];   // density
        }
        return hx;  // found — early exit
    }
    return hx;  // NaN: point outside all leaves
}

// ─── enkf_update ─────────────────────────────────────────────────────────────

void EnsembleSolver::enkf_update(double obs_x, double obs_y, double obs_z,
                                  double obs_val, double obs_var) {
    if (obs_var < 0.0) obs_var = this->obs_variance;
    const int N = n_members;

    // 1. Forward operator: ŷ_i = H(x_i)
    const std::vector<double> hx = apply_H(obs_x, obs_y, obs_z);

    // 2. Ensemble mean of predicted observation
    double hy_mean = 0.0;
    for (double v : hx) hy_mean += v;
    hy_mean /= N;

    // 3. Innovation variance: P_yy + R
    double sigma2_yy = obs_var;
    for (double v : hx)
        sigma2_yy += (v - hy_mean) * (v - hy_mean) / (N - 1);

    if (sigma2_yy < 1e-30) return;  // degenerate: no update

    // 4. Observation perturbations ε_i ~ N(0, R) for stochastic EnKF
    std::normal_distribution<double> noise(0.0, std::sqrt(obs_var));
    std::vector<double> eps(N);
    for (double& e : eps) e = noise(rng_);

    // 5. Update density (Q[0]) at every interior cell of every leaf
    const auto& leaves = members[0].tree.leaf_indices();
    const double c = localization_radius;

    for (int li = 0; li < (int)leaves.size(); ++li) {
        const int ni = leaves[li];
        const CellBlock* ref = members[0].tree.nodes[ni].block.get();
        if (!ref) continue;

        for (int iz = 0; iz < NB; ++iz) {
        for (int iy = 0; iy < NB; ++iy) {
        for (int ix = 0; ix < NB; ++ix) {
            const int flat = cell_idx(ix+NG, iy+NG, iz+NG);

            // Physical cell center
            const double xc = ref->ox + (ix + 0.5) * ref->h;
            const double yc = ref->oy + (iy + 0.5) * ref->h;
            const double zc = ref->oz + (iz + 0.5) * ref->h;
            const double r  = std::sqrt((xc-obs_x)*(xc-obs_x)
                                      + (yc-obs_y)*(yc-obs_y)
                                      + (zc-obs_z)*(zc-obs_z));
            const double L  = gc_weight(r, c);
            if (L == 0.0) continue;

            // Cross-covariance sigma_xy = cov(Q[0][cell], H(x))
            double q_mean = 0.0;
            for (int m = 0; m < N; ++m)
                q_mean += members[m].tree.nodes[ni].block->Q[0][flat];
            q_mean /= N;

            double sigma_xy = 0.0;
            for (int m = 0; m < N; ++m) {
                const double dq  = members[m].tree.nodes[ni].block->Q[0][flat] - q_mean;
                const double dhy = hx[m] - hy_mean;
                sigma_xy += dq * dhy;
            }
            sigma_xy /= (N - 1);

            // Localized Kalman gain
            const double K = L * sigma_xy / sigma2_yy;

            // Analysis update: Q_i += K * (y + ε_i - ŷ_i)
            for (int m = 0; m < N; ++m) {
                const double innovation = obs_val + eps[m] - hx[m];
                members[m].tree.nodes[ni].block->Q[0][flat] += K * innovation;
            }
        }}}
    }
}
