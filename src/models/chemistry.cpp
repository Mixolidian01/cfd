// chemistry.cpp — Multi-species Arrhenius chemistry (P7.4)
// Operator-split: explicit hydro + implicit chemistry sub-step.
#include "models/chemistry.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

// =============================================================================
// chemistry_source: compute dY_k/dt and dT/dt for one cell
// =============================================================================
void chemistry_source(const double* Y, double T, double rho,
                      const ChemMechanism& mech,
                      double* dY, double* dT_dt) noexcept {
    // Zero output
    for (int k = 0; k < mech.n_spec; ++k) dY[k] = 0.0;
    *dT_dt = 0.0;

    for (int r = 0; r < mech.n_reac; ++r) {
        const Reaction& rxn = mech.reactions[r];
        double kf = arrhenius(rxn, T);

        // Progress rate: q_r = kf * prod_j [X_j]^nu_r_j
        double q = kf;
        for (int s = 0; s < rxn.ns; ++s) {
            int k = rxn.spec[s];
            if (rxn.nu_r[s] > 0.0) {
                double Xk = rho * std::max(Y[k], 0.0) / mech.species[k].W;
                q *= std::pow(Xk, rxn.nu_r[s]);
            }
        }

        // Species production: ω̇_k = (ν_k'' - ν_k') * W_k * q_r
        double dH_r = 0.0;
        for (int s = 0; s < rxn.ns; ++s) {
            int k  = rxn.spec[s];
            double nu_net = rxn.nu_p[s] - rxn.nu_r[s];
            double omega_k = nu_net * mech.species[k].W * q;  // [kg/(m³·s)]
            dY[k] += omega_k / rho;
            // Heat release contribution: -h_f0 * omega_k / rho  → dT
            dH_r += mech.species[k].h_f0 * omega_k;
        }
        // dT/dt += -ΔH_reaction / (rho * cv)
        // (heat release = sum of formation enthalpies weighted by consumption)
        *dT_dt -= dH_r / (rho * mech.cv_mix(Y));
    }
}

// =============================================================================
// chemistry_step_cell: implicit Euler + one Newton iteration
//
// Solve: φ^{n+1} = φ^n + dt * f(φ^{n+1})
// Linearise: φ^{n+1} ≈ φ^n + dt * [f(φ^n) + J*(φ^{n+1}-φ^n)]
// → (I - dt*J) * (φ^{n+1} - φ^n) = dt * f(φ^n)
// Solve with Gaussian elimination on the (n_spec+1)×(n_spec+1) system.
// =============================================================================
void chemistry_step_cell(double* Y, double& T, double rho, double dt,
                         const ChemMechanism& mech) noexcept {
    const int N = mech.n_spec + 1;  // state: [Y_0..Y_{n-1}, T]

    // Build f(φ^n)
    double f[MAX_SPEC + 1];
    {
        double dT_dt = 0.0;
        chemistry_source(Y, T, rho, mech, f, &dT_dt);
        f[mech.n_spec] = dT_dt;
    }

    // Build Jacobian J = df/dφ by finite differences (ε = 1e-6 * |φ_k| + 1e-10)
    double J[MAX_SPEC+1][MAX_SPEC+1];
    {
        double Y_pert[MAX_SPEC];
        double f_pert[MAX_SPEC + 1];
        double dT_pert;

        for (int j = 0; j < mech.n_spec; ++j) {
            std::memcpy(Y_pert, Y, mech.n_spec * sizeof(double));
            double eps = 1e-6 * std::fabs(Y[j]) + 1e-10;
            Y_pert[j] += eps;
            chemistry_source(Y_pert, T, rho, mech, f_pert, &dT_pert);
            f_pert[mech.n_spec] = dT_pert;
            for (int i = 0; i < N; ++i)
                J[i][j] = (f_pert[i] - f[i]) / eps;
        }
        // Column for T perturbation
        {
            double Tp = T + 1e-3;
            chemistry_source(Y, Tp, rho, mech, f_pert, &dT_pert);
            f_pert[mech.n_spec] = dT_pert;
            for (int i = 0; i < N; ++i)
                J[i][mech.n_spec] = (f_pert[i] - f[i]) / 1e-3;
        }
    }

    // Form (I - dt*J) and rhs = dt * f
    double A[MAX_SPEC+1][MAX_SPEC+1];
    double b[MAX_SPEC+1];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            A[i][j] = (i==j ? 1.0 : 0.0) - dt * J[i][j];
        b[i] = dt * f[i];
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < N; ++col) {
        // Find pivot
        int pivot = col;
        for (int row = col+1; row < N; ++row)
            if (std::fabs(A[row][col]) > std::fabs(A[pivot][col]))
                pivot = row;
        // Swap
        for (int k = 0; k < N; ++k) std::swap(A[col][k], A[pivot][k]);
        std::swap(b[col], b[pivot]);
        // Eliminate
        double diag = A[col][col];
        if (std::fabs(diag) < 1e-300) continue;
        for (int row = col+1; row < N; ++row) {
            double factor = A[row][col] / diag;
            for (int k = col; k < N; ++k)
                A[row][k] -= factor * A[col][k];
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    double delta[MAX_SPEC+1];
    for (int i = N-1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i+1; j < N; ++j) sum -= A[i][j] * delta[j];
        delta[i] = (std::fabs(A[i][i]) > 1e-300) ? sum / A[i][i] : 0.0;
    }

    // Apply update, enforcing Y_k >= 0 and T > 0
    for (int k = 0; k < mech.n_spec; ++k)
        Y[k] = std::max(0.0, Y[k] + delta[k]);
    T = std::max(1.0, T + delta[mech.n_spec]);
}

// =============================================================================
// apply_chemistry_block
// =============================================================================
void apply_chemistry_block(CellBlock& blk, SpeciesBlock& sblk,
                           double dt, const ChemMechanism& mech) noexcept {
    assert(sblk.n_spec == mech.n_spec);
    double Y[MAX_SPEC];

    for (int k_ = NG; k_ < NG+NB; ++k_)
    for (int j_ = NG; j_ < NG+NB; ++j_)
    for (int i_ = NG; i_ < NG+NB; ++i_) {
        const int idx = cell_idx(i_, j_, k_);

        double rho = blk.Q[0][idx];
        if (rho < 1e-10) continue;

        // Extract species mass fractions
        for (int k = 0; k < mech.n_spec; ++k)
            Y[k] = sblk.Y[k][idx];

        // Extract temperature from conserved state
        double ru = blk.Q[1][idx], rv = blk.Q[2][idx], rw = blk.Q[3][idx];
        double ke = 0.5*(ru*ru + rv*rv + rw*rw) / rho;
        double e_int = blk.Q[4][idx] - ke;
        double cv = mech.cv_mix(Y);
        double T  = e_int / (rho * cv);
        T = std::max(T, 1.0);

        // Record energy before chemistry for conservation check
        double e_int_before = e_int;
        double Y_before[MAX_SPEC];
        for (int k = 0; k < mech.n_spec; ++k) Y_before[k] = Y[k];

        // Integrate chemistry
        chemistry_step_cell(Y, T, rho, dt, mech);

        // Update conserved energy: ΔE = rho*(cv_new*T_new - cv_old*T_old)
        // (composition change shifts cv, so recompute both sides)
        double cv_new = mech.cv_mix(Y);
        blk.Q[4][idx] = rho * cv_new * T + ke;

        // Update species
        // Renormalize to enforce sum(Y_k) = 1 (numerical drift correction)
        double Y_sum = 0.0;
        for (int k = 0; k < mech.n_spec; ++k) Y_sum += Y[k];
        if (Y_sum > 1e-10)
            for (int k = 0; k < mech.n_spec; ++k) Y[k] /= Y_sum;

        for (int k = 0; k < mech.n_spec; ++k)
            sblk.Y[k][idx] = Y[k];

        (void)e_int_before; (void)Y_before;
    }
}

// =============================================================================
// make_simple_mechanism: A → B single-step
// =============================================================================
ChemMechanism make_simple_mechanism(
    double W_A, double h_fA, double cp_A,
    double W_B, double h_fB, double cp_B,
    double A_rate, double Ea,
    const char* name_A, const char* name_B)
{
    ChemMechanism m;
    m.n_spec = 2;
    m.n_reac = 1;

    // Species A (index 0)
    std::strncpy(m.species[0].name, name_A, 15);
    m.species[0].W    = W_A;
    m.species[0].h_f0 = h_fA;
    m.species[0].cp   = cp_A;

    // Species B (index 1)
    std::strncpy(m.species[1].name, name_B, 15);
    m.species[1].W    = W_B;
    m.species[1].h_f0 = h_fB;
    m.species[1].cp   = cp_B;

    // Reaction: A → B
    Reaction& r = m.reactions[0];
    r.A  = A_rate;
    r.n  = 0.0;
    r.Ea = Ea;
    r.ns = 2;
    r.spec[0] = 0; r.spec[1] = 1;
    r.nu_r[0] = 1.0; r.nu_r[1] = 0.0;  // 1 A consumed
    r.nu_p[0] = 0.0; r.nu_p[1] = 1.0;  // 1 B produced (same mass → W_A = W_B assumed for mass balance)

    return m;
}

// =============================================================================
// H2 + 0.5 O2 → H2O  (Westbrook & Dryer 1981 single-step)
// species: H2(0), O2(1), H2O(2)
// rate:    k = 1.8e10 * exp(-16800/T) * [H2]^1 * [O2]^1  [kmol/(m³·s)]
// =============================================================================
ChemMechanism make_h2_o2_mechanism() {
    ChemMechanism m;
    m.n_spec = 3;
    m.n_reac = 1;

    // H2
    std::strncpy(m.species[0].name, "H2",  15);
    m.species[0].W    = 2.016;      // kg/kmol
    m.species[0].h_f0 = 0.0;        // J/kg (reference)
    m.species[0].cp   = 14307.0;    // J/(kg·K)

    // O2
    std::strncpy(m.species[1].name, "O2",  15);
    m.species[1].W    = 31.999;
    m.species[1].h_f0 = 0.0;
    m.species[1].cp   = 918.0;

    // H2O
    std::strncpy(m.species[2].name, "H2O", 15);
    m.species[2].W    = 18.015;
    m.species[2].h_f0 = -1.3435e7;  // J/kg (−241.8 kJ/mol / 0.018015 kg/mol)
    m.species[2].cp   = 1870.0;

    // H2 + 0.5 O2 → H2O
    // Per mole H2: consume 1 H2, 0.5 O2, produce 1 H2O
    Reaction& r = m.reactions[0];
    r.A  = 1.8e10;       // [(kmol/m³)^{-1} · s^{-1}]  (bimolecular)
    r.n  = 0.0;
    r.Ea = 16800.0 * R_UNIV;  // Ea = 16800 K in R_u*T units → J/kmol
    r.ns = 3;
    r.spec[0]=0; r.spec[1]=1; r.spec[2]=2;
    r.nu_r[0]=1.0; r.nu_r[1]=0.5; r.nu_r[2]=0.0;  // reactants
    r.nu_p[0]=0.0; r.nu_p[1]=0.0; r.nu_p[2]=1.0;  // products

    return m;
}
