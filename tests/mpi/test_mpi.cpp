// P7.1 gate test — 2-rank Sod shock tube via MPI domain decomposition
//
// Protocol:
//   1. Every rank builds the same full tree (1 uniform refinement → 8 leaves).
//   2. mpi_partition() assigns 4 leaves per rank (or as evenly as possible).
//   3. mpi_alloc_local_blocks() allocates CellBlock data only on owning ranks.
//   4. 20 SSP-RK3 steps advance the Sod IC under periodic BC (simplest wiring).
//   5. Global mass is reduced across ranks; must be conserved to 1e-10 relative.
//
// Gates:
//   M1: mpi_partition assigns every leaf to exactly one rank.
//   M2: mpi_alloc_local_blocks: local leaves have block; remote leaves do not.
//   M3: 20 steps complete without MPI error.
//   M4: global mass conserved: |mass_final - mass_initial| / mass_initial < 1e-10.
//
// Compilation: must link with MPI.  When HAVE_MPI is not defined the test
// still runs as a single-rank smoke test (all gates pass trivially for M1/M2).

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include "solver/ns_solver.hpp"
#include "mpi/mpi_comm.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>

// ── Sod IC ────────────────────────────────────────────────────────────────────
static Prim sod_ic(double x, double /*y*/, double /*z*/) {
    const double rhoL = 1.0, pL = 1.0;
    const double rhoR = 0.125, pR = 0.1;
    if (x < 0.5) {
        Prim q{}; q.rho = rhoL; q.u = 0; q.v = 0; q.w = 0;
        q.p = pL; q.T = pL / (rhoL * R_GAS);
        q.c = std::sqrt(GAMMA * pL / rhoL);
        return q;
    } else {
        Prim q{}; q.rho = rhoR; q.u = 0; q.v = 0; q.w = 0;
        q.p = pR; q.T = pR / (rhoR * R_GAS);
        q.c = std::sqrt(GAMMA * pR / rhoR);
        return q;
    }
}

// ── Run all gates ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
#ifdef HAVE_MPI
    MpiEnvironment mpi_env(argc, argv);
    int rank = mpi_env.rank();
    int nranks = mpi_env.size();
#else
    (void)argc; (void)argv;
    int rank = 0, nranks = 1;
#endif

    // ── Build solver (all ranks build the same full tree) ─────────────────────
    SolverConfig cfg;
    cfg.time.cfl            = 0.4;
    cfg.time.t_end          = 1e30;   // we control step count manually
    cfg.time.max_steps      = 20;
    cfg.bc.variant = PeriodicBC{};
    cfg.io.verbose        = false;
    cfg.amr.regrid_interval = 0;
    cfg.amr.max_level      = 1;

    NSSolver solver;
    solver.cfg = cfg;
    solver.init(1.0, sod_ic);

    // Refine once uniformly so we have 8 leaves.
    {
        std::vector<int> to_refine;
        for (int li : solver.tree.leaf_indices()) to_refine.push_back(li);
        for (int li : to_refine) solver.tree.refine(li);
        solver.tree.rebuild_neighbours();
        solver.alloc_scratch();
    }

    const int n_leaves = (int)solver.tree.leaf_indices().size();
    if (rank == 0) printf("[P7.1] leaves=%d  ranks=%d\n", n_leaves, nranks);

    // ── Set up MPI partition ──────────────────────────────────────────────────
    MpiPartition part;
#ifdef HAVE_MPI
    part.comm = mpi_env.comm();
#else
    part.comm = 0;
#endif
    mpi_partition(solver.tree, &part);

    // Gate M1: every leaf assigned to exactly one rank
    {
        int unassigned = 0;
        for (int li : solver.tree.leaf_indices())
            if (part.leaf_owner[li] < 0 || part.leaf_owner[li] >= nranks)
                ++unassigned;
        if (unassigned > 0) {
            if (rank == 0) printf("FAIL M1: %d leaves unassigned\n", unassigned);
#ifdef HAVE_MPI
            MPI_Abort(part.comm, 1);
#endif
            return 1;
        }
        if (rank == 0) printf("PASS M1: all %d leaves assigned\n", n_leaves);
    }

    // Allocate blocks only on owning ranks.
    // h0 = L / NB for the root block at level 0; after 1 refine → h = L/(2*NB).
    const double h0 = 1.0 / NB;
    mpi_alloc_local_blocks(solver.tree, part, h0);

    // Gate M2: local leaves have block; remote leaves do not.
    {
        int bad = 0;
        for (int li : solver.tree.leaf_indices()) {
            bool is_local  = part.is_local(li);
            bool has_block = solver.tree.nodes[li].has_block();
            if (is_local && !has_block) ++bad;
            if (!is_local && has_block) ++bad;
        }
        if (bad > 0) {
            printf("[rank %d] FAIL M2: %d leaves with wrong block allocation\n", rank, bad);
#ifdef HAVE_MPI
            MPI_Abort(part.comm, 1);
#endif
            return 1;
        }
        if (rank == 0) printf("PASS M2: block allocation matches partition\n");
    }

    // Re-fill IC on each rank's local leaves (mpi_alloc_local_blocks may have
    // allocated zero-initialised blocks; we need the Sod IC there).
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

    // Wire MPI into solver (also wires into tree).
    solver.set_mpi(&part);
    solver.alloc_scratch();

    // Gate M3: run 20 steps without error.
    // Compute initial mass for conservation check.
    const StepDiag d0 = solver.compute_diag();
    const double mass0 = d0.mass;

    for (int s = 0; s < 20; ++s)
        solver.advance();

    if (rank == 0) printf("PASS M3: 20 steps completed\n");

    // Gate M4: global mass conserved.
    const StepDiag df = solver.compute_diag();
    const double mass_final = df.mass;
    const double rel_err = std::abs(mass_final - mass0) / (std::abs(mass0) + 1e-300);

    if (rank == 0) {
        printf("  mass0=%.14e  mass_final=%.14e  rel_err=%.3e\n",
               mass0, mass_final, rel_err);
        if (rel_err < 1e-10) {
            printf("PASS M4: global mass conserved (rel_err=%.2e < 1e-10)\n", rel_err);
            printf("[P7.1] ALL GATES PASS\n");
        } else {
            printf("FAIL M4: mass not conserved (rel_err=%.2e)\n", rel_err);
#ifdef HAVE_MPI
            MPI_Abort(part.comm, 1);
#endif
            return 1;
        }
    }

    return 0;
}
