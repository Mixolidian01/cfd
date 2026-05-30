// test_t48_bn_gpu.cu — G6: GPU Baer-Nunziato two-phase solver gate
//
// T48a: GPU BN runs 10 steps without crash (α₁ ∈ (0,1), ρ > 0).
// T48b: Mass is conserved |Δm|/m < 1e-10 over 10 steps (periodic).
// T48c: CPU and GPU pressures agree max|p_CPU − p_GPU|/p_ref < 1e-3 after 10 steps
//       with the same fixed dt sequence (eliminates adaptive-dt mismatch).
//
// EOS: both phases ideal gas (γ₁=1.4, γ₂=1.6, π∞=0) for dimensionless IC.

#include "cuda/gpu_bn.cuh"
#include "models/bn_model.hpp"
#include "mesh/cell_block.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

static int nfail = 0;

static void check(bool ok, const char* tag, const char* msg, double val = -1.0)
{
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else {
        if (val >= 0.0) printf("  FAIL  %s  %s  (val = %.3e)\n", tag, msg, val);
        else            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// Both phases ideal gas, dimensionless pressures O(1).
static const BNEosParams EOS_IDEAL{ 1.4, 1.6, 0.0, 0.0 };

// 1D two-fluid Sod IC in x: left half α₁=0.9 p=1, right half α₁=0.1 p=0.1.
static void fill_sod(BNCellBlock& blk)
{
    const BNEosParams& e = EOS_IDEAL;
    for (int k = GPU_NG; k < GPU_NB+GPU_NG; ++k)
    for (int j = GPU_NG; j < GPU_NB+GPU_NG; ++j)
    for (int i = GPU_NG; i < GPU_NB+GPU_NG; ++i) {
        const int f = cell_idx(i, j, k);
        const bool left = (i < GPU_NG + GPU_NB/2);
        const double a1 = left ? 0.9 : 0.1;
        const double a2 = 1.0 - a1;
        const double p  = left ? 1.0 : 0.1;
        const double rho_e = a1*(p + e.gamma1*e.pinf1)/(e.gamma1-1.0)
                           + a2*(p + e.gamma2*e.pinf2)/(e.gamma2-1.0);
        blk.Q[0][f] = a1;      // α₁ρ₁ (ρ₁=1)
        blk.Q[1][f] = a2;      // α₂ρ₂ (ρ₂=1)
        blk.Q[2][f] = 0.0;
        blk.Q[3][f] = 0.0;
        blk.Q[4][f] = 0.0;
        blk.Q[5][f] = rho_e;
        blk.Q[6][f] = a1;
    }
}

static double total_mass_bn(const BNCellBlock& blk)
{
    const double dV = blk.h * blk.h * blk.h;
    double m = 0.0;
    for (int k = GPU_NG; k < GPU_NB+GPU_NG; ++k)
    for (int j = GPU_NG; j < GPU_NB+GPU_NG; ++j)
    for (int i = GPU_NG; i < GPU_NB+GPU_NG; ++i) {
        const int f = cell_idx(i,j,k);
        m += (blk.Q[0][f] + blk.Q[1][f]) * dV;
    }
    return m;
}

// CPU SSP-RK3 with fixed dt, mirrors GpuBnList::advance() stage structure.
static void cpu_rk3(BNCellBlock& Q, double dt, const BNEosParams& eos)
{
    BNCellBlock Qn(0,0,0,Q.h), Qs(0,0,0,Q.h), rhs(0,0,0,Q.h);
    std::memcpy(Qn.data_, Q.data_, sizeof(Qn.data_));

    auto rk_step = [&](BNCellBlock& q_in, BNCellBlock& q_out, double a, double b) {
        bn_fill_ghosts_periodic(q_in);
        for (int v = 0; v < NVAR_BN; ++v)
            for (int f = 0; f < NCELL; ++f) rhs.Q[v][f] = 0.0;
        compute_rhs_bn(q_in, rhs, eos);
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            const int f = cell_idx(i,j,k);
            for (int v = 0; v < NVAR_BN; ++v)
                q_out.Q[v][f] = a*Qn.Q[v][f] + b*(q_in.Q[v][f] + dt*rhs.Q[v][f]);
        }
    };

    // Stage 1: Qs = 0*Qn + 1*(Qn + dt*RHS) = Qn + dt*RHS(Qn)
    rk_step(Q,  Qs, 0.0, 1.0);
    // Stage 2: Q  = 3/4*Qn + 1/4*(Qs + dt*RHS(Qs))
    rk_step(Qs, Q,  0.75, 0.25);
    // Stage 3: Q  = 1/3*Qn + 2/3*(Q + dt*RHS(Q))
    rk_step(Q,  Q,  1.0/3.0, 2.0/3.0);
}

// ─────────────────────────────────────────────────────────────────────────────
static void test_t48a()
{
    printf("\n-- T48a  GPU BN 10 steps no crash --\n");
    const double h = 1.0 / GPU_NB;
    BNCellBlock blk(0,0,0,h);
    fill_sod(blk);
    bn_fill_ghosts_periodic(blk);

    GpuBnList bn;
    bn.build(h, 1, EOS_IDEAL);
    bn.upload({blk});

    const double dt = bn_cfl_dt(blk, 0.4, EOS_IDEAL);
    printf("   h=%.4f  dt=%.4e\n", h, dt);

    bool crashed = false;
    for (int s = 0; s < 10 && !crashed; ++s) {
        bn.advance(dt);
        std::vector<BNCellBlock> out(1, BNCellBlock(0,0,0,h));
        bn.download(out);
        for (int k = GPU_NG; k < GPU_NB+GPU_NG && !crashed; ++k)
        for (int j = GPU_NG; j < GPU_NB+GPU_NG && !crashed; ++j)
        for (int i = GPU_NG; i < GPU_NB+GPU_NG && !crashed; ++i) {
            const int f = cell_idx(i,j,k);
            const double rho = out[0].Q[0][f] + out[0].Q[1][f];
            const double a1  = out[0].Q[6][f];
            if (!std::isfinite(rho) || rho <= 0.0)         crashed = true;
            if (!std::isfinite(a1) || a1 <= 0.0 || a1 >= 1.0) crashed = true;
        }
        printf("   step=%d  %s\n", s, crashed ? "CRASH" : "ok");
    }
    check(!crashed, "T48a", "GPU BN 10 steps no crash");
}

// ─────────────────────────────────────────────────────────────────────────────
static void test_t48b()
{
    printf("\n-- T48b  GPU BN mass conservation |Δm|/m < 1e-10 --\n");
    const double h = 1.0 / GPU_NB;
    BNCellBlock blk(0,0,0,h);
    fill_sod(blk);
    bn_fill_ghosts_periodic(blk);
    const double mass0 = total_mass_bn(blk);

    GpuBnList bn;
    bn.build(h, 1, EOS_IDEAL);
    bn.upload({blk});
    const double dt = bn_cfl_dt(blk, 0.4, EOS_IDEAL);
    for (int s = 0; s < 10; ++s) bn.advance(dt);

    std::vector<BNCellBlock> out(1, BNCellBlock(0,0,0,h));
    bn.download(out);
    const double mass1 = total_mass_bn(out[0]);
    const double err   = std::fabs(mass1 - mass0) / (mass0 + 1e-300);
    printf("   mass_err = %.3e  (tol 1e-10)\n", err);
    check(err < 1e-10, "T48b", "GPU BN mass conservation |Δm|/m < 1e-10", err);
}

// ─────────────────────────────────────────────────────────────────────────────
static void test_t48c()
{
    printf("\n-- T48c  CPU vs GPU pressure agreement (relative) < 1e-3 --\n");
    const double h = 1.0 / GPU_NB;
    BNCellBlock blk(0,0,0,h);
    fill_sod(blk);
    bn_fill_ghosts_periodic(blk);
    const double dt = bn_cfl_dt(blk, 0.4, EOS_IDEAL);
    printf("   h=%.4f  dt=%.4e\n", h, dt);

    // CPU: 10 fixed-dt RK3 steps
    BNCellBlock cpu_Q = blk;
    for (int s = 0; s < 10; ++s) cpu_rk3(cpu_Q, dt, EOS_IDEAL);

    // GPU: same IC, same dt
    GpuBnList bn;
    bn.build(h, 1, EOS_IDEAL);
    bn.upload({blk});
    for (int s = 0; s < 10; ++s) bn.advance(dt);
    std::vector<BNCellBlock> gpu_out(1, BNCellBlock(0,0,0,h));
    bn.download(gpu_out);

    double max_err = 0.0;
    for (int k = GPU_NG; k < GPU_NB+GPU_NG; ++k)
    for (int j = GPU_NG; j < GPU_NB+GPU_NG; ++j)
    for (int i = GPU_NG; i < GPU_NB+GPU_NG; ++i) {
        const int f = cell_idx(i,j,k);
        const Prim2Phase pc = bn_cons_to_prim(
            cpu_Q.Q[0][f], cpu_Q.Q[1][f], cpu_Q.Q[2][f], cpu_Q.Q[3][f],
            cpu_Q.Q[4][f], cpu_Q.Q[5][f], cpu_Q.Q[6][f], EOS_IDEAL);
        const Prim2Phase pg = bn_cons_to_prim(
            gpu_out[0].Q[0][f], gpu_out[0].Q[1][f], gpu_out[0].Q[2][f],
            gpu_out[0].Q[3][f], gpu_out[0].Q[4][f], gpu_out[0].Q[5][f],
            gpu_out[0].Q[6][f], EOS_IDEAL);
        const double ref = std::max(std::fabs(pc.p), 1.0e-6);
        max_err = std::max(max_err, std::fabs(pc.p - pg.p) / ref);
    }
    printf("   max rel|p_CPU - p_GPU| = %.3e  (tol 1e-3)\n", max_err);
    check(max_err < 1e-3, "T48c", "CPU vs GPU relative pressure agreement < 1e-3", max_err);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== G6 GPU Baer-Nunziato two-phase solver gate test ===\n");
    test_t48a();
    test_t48b();
    test_t48c();
    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
