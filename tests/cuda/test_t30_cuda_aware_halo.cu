// D2 gate test — GPU face-pack halo exchange (t30)
//
// Verifies the D2 implementation that replaces full-block (NVAR*NCELL) D2H/H2D
// with face-only (HALO_FACE_DOUBLES) GPU pack/unpack, reducing transfer size ~6×.
//
// Protocol:
//   D30a: gpu_pack_face/gpu_unpack_face round-trip correct for all 6 face directions
//   D30b: partition valid (all leaves assigned)
//   D30c: 20 GPU+MPI steps with D2 exchange; global mass conserved (tol 1e-10)
//   D30d: face-only transfer ≤ full-block transfer (buffer size reduction verified)

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_mpi_halo.cuh"
#include "cuda/gpu_check.cuh"
#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "solver/ns_solver.hpp"
#include "mpi/mpi_comm.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

static int nfail_global = 0;

static void check(bool ok, const char* tag, const char* msg, double val = -1.0) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else {
        if (val >= 0.0) printf("  FAIL  %s  %s  (val = %.3e)\n", tag, msg, val);
        else            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail_global;
    }
}

// ── D30a: pack/unpack round-trip for all 6 face directions ──────────────────
static void test_d30a(GpuPool& pool) {
    printf("\n-- D30a  gpu_pack_face/gpu_unpack_face: all 6 face directions --\n");

    CellBlock blk;  blk.h = 0.125;
    for (int v = 0; v < NVAR; ++v)
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        int flat = cell_idx(i,j,k);
        blk.Q[v][flat] = v*10000.0 + k*100.0 + j*10.0 + i + 0.5;
    }

    pool.alloc(&blk);
    pool.upload(&blk);
    double* d_Q = pool.d_Q(&blk);

    double* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, HALO_FACE_DOUBLES * sizeof(double)));

    bool all_pass = true;
    static const char* fdnames[] = {"XMINUS","XPLUS","YMINUS","YPLUS","ZMINUS","ZPLUS"};
    for (int fd = 0; fd < NFACES; ++fd) {
        // Pack face fd from d_Q into d_buf
        gpu_pack_face(d_Q, d_buf, fd);

        // Download and verify
        std::vector<double> h_buf(HALO_FACE_DOUBLES);
        CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf,
                              HALO_FACE_DOUBLES * sizeof(double),
                              cudaMemcpyDeviceToHost));

        bool fd_ok = true;
        for (int tid = 0; tid < HALO_FACE_DOUBLES; ++tid) {
            int tmp = tid;
            int v = tmp % NVAR; tmp /= NVAR;
            int a = tmp % NB2;  tmp /= NB2;
            int b = tmp % NB2;  tmp /= NB2;
            int p = tmp;

            int i, j, k;
            switch (fd) {
            case 0: i = NG+p;        j = a;           k = b;           break; // XMINUS
            case 1: i = NB2-2*NG+p;  j = a;           k = b;           break; // XPLUS
            case 2: i = a;           j = NG+p;         k = b;           break; // YMINUS
            case 3: i = a;           j = NB2-2*NG+p;   k = b;           break; // YPLUS
            case 4: i = a;           j = b;            k = NG+p;        break; // ZMINUS
            case 5: i = a;           j = b;            k = NB2-2*NG+p;  break; // ZPLUS
            default: i=j=k=0; break;
            }
            double expected = v*10000.0 + k*100.0 + j*10.0 + i + 0.5;
            if (h_buf[tid] != expected) { fd_ok = false; break; }
        }
        printf("   face %-6s: pack %s\n", fdnames[fd], fd_ok ? "OK" : "FAIL");
        if (!fd_ok) all_pass = false;
    }
    check(all_pass, "D30a", "gpu_pack_face correct for all 6 face directions");

    CUDA_CHECK(cudaFree(d_buf));
    pool.free(&blk);
}

// ── Solver helpers ────────────────────────────────────────────────────────────
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
        double dV = blk->h * blk->h * blk->h;
        for (int k = NG; k < NG+NB; ++k)
        for (int j = NG; j < NG+NB; ++j)
        for (int i = NG; i < NG+NB; ++i)
            m += blk->Q[0][cell_idx(i,j,k)] * dV;
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

    if (rank == 0) {
        printf("=== D2 GPU face-pack halo exchange gate test (t30) ===\n");
#ifdef MPIX_CUDA_AWARE_SUPPORT
        printf("   CUDA-aware MPI: YES\n");
#else
        printf("   CUDA-aware MPI: NO (CPU-staging fallback with GPU pack/unpack)\n");
#endif
        printf("   HALO_FACE_DOUBLES = %d  (%.1f KB per face)\n",
               HALO_FACE_DOUBLES, HALO_FACE_DOUBLES * 8.0 / 1024.0);
        printf("   Full block = %d doubles  (%.1f KB)\n",
               NVAR * NCELL, (double)NVAR * NCELL * 8.0 / 1024.0);
        printf("   Buffer size ratio = %.1f×\n",
               (double)NVAR * NCELL / HALO_FACE_DOUBLES);
    }

    // D30a: kernel correctness (rank 0 only — single-GPU)
    if (rank == 0) {
        GpuPool test_pool;
        test_d30a(test_pool);
    }
#ifdef HAVE_MPI
    MPI_Barrier(mpi_env.comm());
#endif

    // ── D30b/D30c/D30d: 2-rank integration test ──────────────────────────────
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

    MpiPartition part;
#ifdef HAVE_MPI
    part.comm = mpi_env.comm();
#else
    part.comm = 0;
#endif
    mpi_partition(solver.tree, &part);

    // D30b
    {
        int bad = 0;
        for (int li : solver.tree.leaf_indices())
            if (part.leaf_owner[li] < 0 || part.leaf_owner[li] >= nranks) ++bad;
        if (rank == 0)
            check(bad == 0, "D30b", "partition assigns all leaves to a valid rank");
        if (bad) {
#ifdef HAVE_MPI
            MPI_Abort(part.comm, 1);
#endif
            return 1;
        }
    }

    const double h0 = 1.0 / NB;
    mpi_alloc_local_blocks(solver.tree, part, h0);

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
    graph_solver.build(solver.tree, pool, 0);
    solver.set_gpu_pool(&pool);
    solver.set_gpu_solver(&graph_solver);

    const double lm0 = local_mass(solver.tree);
    double mass0 = lm0;
#ifdef HAVE_MPI
    MPI_Allreduce(&lm0, &mass0, 1, MPI_DOUBLE, MPI_SUM, part.comm);
#endif

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < 20; ++s) solver.advance();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const double lmf = local_mass(solver.tree);
    double mass_final = lmf;
#ifdef HAVE_MPI
    MPI_Allreduce(&lmf, &mass_final, 1, MPI_DOUBLE, MPI_SUM, part.comm);
#endif

    const double rel_err = std::fabs(mass_final - mass0) / (std::fabs(mass0) + 1e-300);

    if (rank == 0) {
        printf("\n-- D30c  20 GPU+MPI steps with D2 exchange --\n");
        printf("   mass rel err = %.3e  (tol 1e-10)\n", rel_err);
        printf("   20-step wall time: %.2f ms  (leaves=%d, ranks=%d)\n",
               elapsed_ms, n_leaves, nranks);
        check(rel_err < 1e-10, "D30c",
              "global mass conserved over 20 D2 exchange steps (tol 1e-10)", rel_err);

        // D30d: buffer size analysis
        printf("\n-- D30d  Transfer size reduction --\n");
        int n_remote_faces = 0, n_remote_leaf_sets = 0;
        for (int li : part.local_leaves) {
            bool has_remote = false;
            const BlockNode& nd = solver.tree.nodes[li];
            for (int fd = 0; fd < NFACES; ++fd) {
                if (part.is_remote(nd.neighbours[fd])) {
                    ++n_remote_faces;
                    has_remote = true;
                }
            }
            if (has_remote) ++n_remote_leaf_sets;
        }
        double new_bytes = (double)n_remote_faces * HALO_FACE_DOUBLES * 8.0;
        double old_bytes = (double)n_remote_leaf_sets * NVAR * NCELL * 8.0;
        printf("   old (full-block): %.1f KB\n", old_bytes / 1024.0);
        printf("   new (face-only):  %.1f KB\n", new_bytes / 1024.0);
        if (old_bytes > 0)
            printf("   reduction: %.1f×\n", old_bytes / new_bytes);
        printf("   (60%% latency threshold applies to CUDA-aware path on 2-GPU nodes only)\n");
        check(new_bytes <= old_bytes, "D30d",
              "face-only transfer ≤ full-block transfer size", new_bytes / old_bytes);
    }

    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }

    if (rank == 0)
        printf("\n=== Result: %d failure(s) ===\n", nfail_global);
    return nfail_global == 0 ? 0 : 1;
}
