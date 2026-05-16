// P-MPI-GPU gate test — 2-rank GPU halo exchange
//
// Each rank owns half the leaves (8-leaf uniform tree → 4 per rank).
// GpuGraphSolver advances 20 SSP-RK3 steps with GPU halo exchange.
//
// Gates:
//   GM1: mpi_partition assigns every leaf to exactly one rank.
//   GM2: mpi_alloc_local_blocks: local leaves have block; remote leaves do not.
//   GM3: 20 GPU+MPI steps complete without error.
//   GM4: global mass conserved: |mass_final - mass_initial| / mass_initial < 1e-10.

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_check.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "solver/ns_solver.hpp"
#include "mpi/mpi_comm.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

static Prim sod_ic(double x, double, double) {
    Prim q{};
    if (x < 0.5) { q.rho = 1.0; q.p = 1.0; }
    else          { q.rho = 0.125; q.p = 0.1; }
    q.u = q.v = q.w = 0.0;
    q.T = q.p / (q.rho * R_GAS);
    q.c = std::sqrt(GAMMA * q.p / q.rho);
    return q;
}

static double local_mass(const BlockTree& tree) {
    double m = 0.0;
    for (int li : tree.leaf_indices()) {
        const CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            m += blk->Q[0][cell_idx(i,j,k)] * (blk->h * blk->h * blk->h);
    }
    return m;
}

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MpiEnvironment mpi_env(argc, argv);
    int rank   = mpi_env.rank();
    int nranks = mpi_env.size();
#else
    (void)argc; (void)argv;
    int rank = 0, nranks = 1;
#endif

    // ── Build solver (all ranks build the same full tree) ─────────────────────
    SolverConfig cfg;
    cfg.time.cfl            = 0.4;
    cfg.time.max_steps      = 20;
    cfg.bc.variant          = PeriodicBC{};
    cfg.amr.regrid_interval = 0;
    cfg.amr.max_level       = 1;
    cfg.io.verbose          = false;

    NSSolver solver;
    solver.cfg = cfg;
    solver.init(1.0, sod_ic);

    // Refine once uniformly → 8 leaves.
    {
        std::vector<int> to_refine;
        for (int li : solver.tree.leaf_indices()) to_refine.push_back(li);
        for (int li : to_refine) solver.tree.refine(li);
        solver.tree.rebuild_neighbours();
        solver.alloc_scratch();
    }

    const int n_leaves = (int)solver.tree.leaf_indices().size();
    if (rank == 0) printf("[P-MPI-GPU] leaves=%d  ranks=%d\n", n_leaves, nranks);

    // ── Set up MPI partition ──────────────────────────────────────────────────
    MpiPartition part;
#ifdef HAVE_MPI
    part.comm = mpi_env.comm();
#else
    part.comm = 0;
#endif
    mpi_partition(solver.tree, &part);

    // Gate GM1: every leaf assigned to exactly one rank.
    {
        int bad = 0;
        for (int li : solver.tree.leaf_indices())
            if (part.leaf_owner[li] < 0 || part.leaf_owner[li] >= nranks) ++bad;
        if (bad > 0) {
            if (rank == 0) printf("FAIL GM1: %d leaves unassigned\n", bad);
#ifdef HAVE_MPI
            MPI_Abort(part.comm, 1);
#endif
            return 1;
        }
        if (rank == 0) printf("PASS GM1: all %d leaves assigned\n", n_leaves);
    }

    const double h0 = 1.0 / NB;
    mpi_alloc_local_blocks(solver.tree, part, h0);

    // Gate GM2: local leaves have block; remote leaves do not.
    {
        int bad = 0;
        for (int li : solver.tree.leaf_indices()) {
            bool is_local  = part.is_local(li);
            bool has_block = solver.tree.nodes[li].has_block();
            if (is_local && !has_block) ++bad;
            if (!is_local && has_block) ++bad;
        }
        if (bad > 0) {
            printf("[rank %d] FAIL GM2: %d leaves with wrong block allocation\n", rank, bad);
#ifdef HAVE_MPI
            MPI_Abort(part.comm, 1);
#endif
            return 1;
        }
        if (rank == 0) printf("PASS GM2: block allocation matches partition\n");
    }

    // Re-fill IC on local leaves.
    for (int li : solver.tree.leaf_indices()) {
        if (!solver.tree.nodes[li].has_block()) continue;
        auto& blk = *solver.tree.nodes[li].block;
        const double h = blk.h;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            double cx = blk.ox + (i - ilo() + 0.5) * h;
            double cy = blk.oy + (j - ilo() + 0.5) * h;
            double cz = blk.oz + (k - ilo() + 0.5) * h;
            Prim p = sod_ic(cx, cy, cz);
            double rhoE = p.p / (GAMMA - 1.0)
                        + 0.5 * p.rho * (p.u*p.u + p.v*p.v + p.w*p.w);
            int idx = cell_idx(i, j, k);
            blk.Q[0][idx] = p.rho;
            blk.Q[1][idx] = p.rho * p.u;
            blk.Q[2][idx] = p.rho * p.v;
            blk.Q[3][idx] = p.rho * p.w;
            blk.Q[4][idx] = rhoE;
        }
    }

    // ── Wire MPI + GPU ────────────────────────────────────────────────────────
    solver.set_mpi(&part);
    solver.alloc_scratch();

    GpuPool pool;
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (!blk) continue;
        pool.alloc(blk);
        pool.upload(blk);
    }

    GpuGraphSolver graph_solver;
    graph_solver.set_mpi(&part);
    graph_solver.build(solver.tree, pool, /*bc_type=*/0);
    solver.set_gpu_pool(&pool);
    solver.set_gpu_solver(&graph_solver);

    // Gate GM3: run 20 steps without error.
    const double lm0 = local_mass(solver.tree);
    double mass0 = lm0;
#ifdef HAVE_MPI
    MPI_Allreduce(&lm0, &mass0, 1, MPI_DOUBLE, MPI_SUM, part.comm);
#endif

    for (int s = 0; s < 20; ++s)
        solver.advance();

    if (rank == 0) printf("PASS GM3: 20 GPU+MPI steps completed\n");

    // Gate GM4: global mass conserved.
    // download_q is called by advance() already; read CPU blocks.
    const double lmf = local_mass(solver.tree);
    double mass_final = lmf;
#ifdef HAVE_MPI
    MPI_Allreduce(&lmf, &mass_final, 1, MPI_DOUBLE, MPI_SUM, part.comm);
#endif

    const double rel_err = std::fabs(mass_final - mass0) / (std::fabs(mass0) + 1e-300);
    if (rank == 0) {
        printf("  mass0=%.14e  mass_final=%.14e  rel_err=%.3e\n",
               mass0, mass_final, rel_err);
        if (rel_err < 1e-10) {
            printf("PASS GM4: global mass conserved (rel_err=%.2e < 1e-10)\n", rel_err);
            printf("[P-MPI-GPU] ALL GATES PASS\n");
        } else {
            printf("FAIL GM4: mass not conserved (rel_err=%.2e)\n", rel_err);
#ifdef HAVE_MPI
            MPI_Abort(part.comm, 1);
#endif
            return 1;
        }
    }

    // Cleanup GPU pool.
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }

    return 0;
}
