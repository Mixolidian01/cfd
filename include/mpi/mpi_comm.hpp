#pragma once
// P7.1 — MPI domain decomposition for block-structured AMR
//
// Design: replicated topology, distributed data.
//   Every rank holds the same BlockTree node array (topology only, ~kilobytes).
//   Each rank allocates CellBlock data only for its "owned" leaves.
//   Ghost (halo) face data is exchanged once per RHS evaluation before
//   fill_ghosts_*().  exchange_halos() writes directly into the NG ghost
//   layers of local CellBlocks that border a remote-owned leaf.
//
// Threading model:
//   exchange_halos() calls MPI_Isend / MPI_Irecv across all neighbor ranks,
//   then MPI_Waitall.  Communication is aggregated per-rank (one message pair
//   per neighbor rank, not one per face) to minimise MPI call overhead.
//
// AMR constraint:
//   Same-level remote boundaries only in this implementation.  For coarse-fine
//   remote interfaces the partitioner ensures 2:1 balance keeps them same-rank.
//
// Usage:
//   MpiEnvironment mpi_env(argc, argv);       // MPI_Init / MPI_Finalize
//   MpiPartition   part;
//   mpi_partition(tree, &part);               // assign leaves to ranks
//   tree.set_mpi(&part);                      // fill_ghosts skips remote faces
//   solver.set_mpi(&part);                    // advance() calls exchange_halos
//   solver.run();

#include "mesh/block_tree.hpp"
#ifdef HAVE_MPI
#  include <mpi.h>
using MpiComm_t = MPI_Comm;
#else
using MpiComm_t = int;
#endif

#include <vector>

// ── Halo face constants ───────────────────────────────────────────────────────
// Number of doubles in one packed face halo (NG planes × NB2² cells × NVAR vars).
// Used by both CPU mpi_exchange_halos and GPU k_pack_face / k_unpack_face.
static constexpr int HALO_FACE_DOUBLES = NG * NB2 * NB2 * NVAR;  // = 1440

// ── MPI environment (RAII) ────────────────────────────────────────────────────
struct MpiEnvironment {
    explicit MpiEnvironment(int& argc, char**& argv);
    ~MpiEnvironment();

    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }

    MpiComm_t comm() const noexcept { return comm_; }

private:
    MpiComm_t comm_;
    int rank_ = 0;
    int size_ = 1;
};

// ── Partition metadata ────────────────────────────────────────────────────────
// Built once by mpi_partition(); stable as long as the tree topology is stable.
// Must be rebuilt after every regrid().
struct MpiPartition {
    int my_rank  = 0;
    int n_ranks  = 1;
    MpiComm_t comm;

    // leaf_owner[li] = rank that owns leaf li (-1 for internal/dead nodes)
    std::vector<int> leaf_owner;

    // My owned leaves (subset of tree.leaf_indices())
    std::vector<int> local_leaves;

    bool is_local(int li) const noexcept {
        return li >= 0 && li < (int)leaf_owner.size()
            && leaf_owner[li] == my_rank;
    }
    bool is_remote(int li) const noexcept {
        return li >= 0 && li < (int)leaf_owner.size()
            && leaf_owner[li] >= 0 && leaf_owner[li] != my_rank;
    }
    bool active() const noexcept { return n_ranks > 1; }
};

// ── Partitioning ─────────────────────────────────────────────────────────────
// Assign leaves to ranks using Morton Z-order (contiguous stripes).
// Must be called after tree topology is built and after every regrid().
void mpi_partition(const BlockTree& tree, MpiPartition* part);

// Allocate CellBlocks on this rank for all locally-owned leaves.
// Leaves owned by other ranks have their block reset to nullptr.
// Must be called once after mpi_partition(), before first advance().
void mpi_alloc_local_blocks(BlockTree& tree, const MpiPartition& part,
                             double h0, const CellBlock* prototype = nullptr);

// ── Halo exchange ─────────────────────────────────────────────────────────────
// Fill ghost layers of local CellBlocks whose face-neighbor is on a remote rank.
// Call once per RHS evaluation, before fill_ghosts_*.
// No-op if part is null or part->n_ranks == 1.
void mpi_exchange_halos(BlockTree& tree, const MpiPartition* part);

// ── Global reductions ─────────────────────────────────────────────────────────
// Both functions return local_val unchanged when part is null (single-rank path).
double mpi_allreduce_min(double local_val, const MpiPartition* part);
double mpi_allreduce_sum(double local_val, const MpiPartition* part);

// ── Null-safe remote-leaf test ────────────────────────────────────────────────
// Returns false (treat as local) when part is null; avoids scattered if(mpi_&&…).
inline bool mpi_is_remote(const MpiPartition* part, int ni) noexcept {
    return part && part->is_remote(ni);
}
