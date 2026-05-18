// Gate t36: GPU snapshot correctness — Option A (slice) + Option C (metrics).
//
// Sets up a 2×2×2 leaf tree (Sod IC), runs 20 GPU steps with a
// GpuSnapshotBuffer attached.  After 20 steps:
//   SnapA1: h_slice contains non-zero values (kernel ran for the intersecting axis)
//   SnapA2: GPU slice values match CPU build_frame() within float32 tolerance (1e-4)
//   SnapC1: GPU mass matches CPU total_mass() sum to 1e-8 (relative)
//   SnapC2: GPU rho_min/rho_max match CPU scan to 1e-8

#include "solver/ns_solver.hpp"
#include "cuda/gpu_graph.cuh"
#include "gpu_pool.hpp"
#include "gpu_snapshot.hpp"
#include "io/live_streamer.hpp"
#include "mesh/bc_types.hpp"
#include "mesh/cell_block.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <vector>

static bool g_ok = true;
#define CHECK(label, cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s\n", label); \
        g_ok = false; \
    } else { \
        fprintf(stdout, "PASS %s\n", label); \
    } \
} while(0)

static double rel(double a, double b) {
    return std::abs(a - b) / (std::abs(b) + 1e-300);
}

int main()
{
    // ── Solver setup (2×2×2 leaf Sod IC) ─────────────────────────────────────
    NSSolver solver;
    solver.cfg.time.cfl        = 0.4;
    solver.cfg.time.t_end      = 1e30;
    solver.cfg.time.max_steps  = 20;
    solver.cfg.bc.variant      = PeriodicBC{};
    solver.cfg.io.verbose      = false;
    solver.cfg.io.diag_interval = 999;

    solver.init(1.0, [](double x, double /*y*/, double /*z*/) -> Prim {
        Prim q{};
        q.rho = (x < 0.5) ? 1.0 : 0.125;
        q.u = q.v = q.w = 0.0;
        q.p = (x < 0.5) ? 1.0 : 0.1;
        q.T = q.p / (q.rho * R_GAS);
        q.c = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    });

    // ── GPU pool + solver injection ───────────────────────────────────────────
    GpuPool pool;
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (!blk) continue;
        pool.alloc(blk);
        pool.upload(blk);
    }

    GpuGraphSolver graph_solver;

    // ── Snapshot buffer (Option A/C) ─────────────────────────────────────────
    const int n_leaves = static_cast<int>(solver.tree.leaf_indices().size());
    GpuSnapshotBuffer snap;
    snap.alloc(n_leaves);
    snap.var_id   = 0;    // RHO
    snap.axis     = 2;    // Z-slice
    snap.norm_pos = 0.5f; // midplane
    snap.domain_L = 1.0f;

    // Wire snapshot buffer BEFORE build() so _upload_snap_metas() runs inside build().
    graph_solver.set_snapshot_buffer(&snap);
    graph_solver.build(solver.tree, pool, bc_to_int(solver.cfg.bc.variant));
    solver.set_gpu_pool(&pool);
    solver.set_gpu_solver(&graph_solver);
    solver.set_gpu_snapshot(&snap);

    // ── Run 20 steps ─────────────────────────────────────────────────────────
    solver.run();

    // advance() launched snapshot kernels + synced stream each step.
    // After run(), download_q() was called at diag_interval=999 (so NOT called).
    // Call it now explicitly so we can compare with CPU data.
    graph_solver.download_q(solver.tree);

    // ── SnapA1: h_slice has non-zero values ───────────────────────────────────
    const float* tile = snap.h_slice;
    bool any_nonzero = false;
    for (int i = 0; i < snap.n_leaves * NB * NB; ++i)
        if (tile[i] != 0.f) { any_nonzero = true; break; }
    CHECK("SnapA1: h_slice non-zero", any_nonzero);

    // ── SnapA2: GPU slice matches CPU build_frame ─────────────────────────────
    // Build CPU slice manually: iterate leaf_indices, extract axis=2, pos=0.5.
    const auto leaves = solver.tree.leaf_indices();
    const float slice_phys = snap.norm_pos * snap.domain_L;
    double max_abs_err = 0.0;
    int n_compared = 0;
    for (int idx = 0; idx < (int)leaves.size(); ++idx) {
        const BlockNode& nd = solver.tree.nodes[leaves[idx]];
        const CellBlock& blk = *nd.block;
        const float lo = static_cast<float>(nd.oz);
        const float hi = lo + NB * blk.h;
        if (slice_phys < lo || slice_phys >= hi) continue;

        int s = NG + static_cast<int>((slice_phys - lo) / blk.h);
        s = std::clamp(s, NG, NG + NB - 1);

        const float* gpu_tile = snap.h_slice + idx * NB * NB;
        for (int b = 0; b < NB; ++b)
        for (int a = 0; a < NB; ++a) {
            float cpu_val = static_cast<float>(blk.rho(NG+a, NG+b, s));
            float gpu_val = gpu_tile[b * NB + a];
            double err = std::abs((double)(cpu_val - gpu_val));
            if (err > max_abs_err) max_abs_err = err;
            ++n_compared;
        }
    }
    CHECK("SnapA2: slice cells compared > 0", n_compared > 0);
    fprintf(stdout, "  max_abs_err=%.2e  n_compared=%d\n", max_abs_err, n_compared);
    CHECK("SnapA2: GPU slice matches CPU (|err|<1e-4)", max_abs_err < 1e-4);

    // ── SnapC1: GPU mass matches CPU mass sum ─────────────────────────────────
    double gpu_mass = 0.0;
    double gpu_rho_min = 1e300, gpu_rho_max = 0.0;
    for (int i = 0; i < snap.n_leaves; ++i) {
        gpu_mass    += snap.h_metrics[i].mass;
        gpu_rho_min  = std::min(gpu_rho_min, snap.h_metrics[i].rho_min);
        gpu_rho_max  = std::max(gpu_rho_max, snap.h_metrics[i].rho_max);
    }

    double cpu_mass = 0.0;
    double cpu_rho_min = 1e300, cpu_rho_max = 0.0;
    for (int li : leaves) {
        const CellBlock& blk = *solver.tree.nodes[li].block;
        const double h3 = blk.h * blk.h * blk.h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i) {
            double rho = blk.rho(i,j,k);
            cpu_mass    += rho * h3;
            cpu_rho_min  = std::min(cpu_rho_min, rho);
            cpu_rho_max  = std::max(cpu_rho_max, rho);
        }
    }

    double mass_rel = rel(gpu_mass, cpu_mass);
    fprintf(stdout, "  gpu_mass=%.10e  cpu_mass=%.10e  rel=%.2e\n",
            gpu_mass, cpu_mass, mass_rel);
    CHECK("SnapC1: GPU mass matches CPU (rel<1e-8)", mass_rel < 1e-8);

    // ── SnapC2: GPU rho_min/rho_max match CPU ────────────────────────────────
    fprintf(stdout, "  gpu_rho_min=%.6e  cpu_rho_min=%.6e\n", gpu_rho_min, cpu_rho_min);
    fprintf(stdout, "  gpu_rho_max=%.6e  cpu_rho_max=%.6e\n", gpu_rho_max, cpu_rho_max);
    CHECK("SnapC2: GPU rho_min matches CPU (rel<1e-8)", rel(gpu_rho_min, cpu_rho_min) < 1e-8);
    CHECK("SnapC2: GPU rho_max matches CPU (rel<1e-8)", rel(gpu_rho_max, cpu_rho_max) < 1e-8);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    for (int li : leaves) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }

    if (g_ok) {
        fprintf(stdout, "[t36] ALL GATES PASS\n");
        return 0;
    }
    return 1;
}
