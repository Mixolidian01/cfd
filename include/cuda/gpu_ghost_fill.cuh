#pragma once
// P8.2: GPU ghost fill for same-level and C/F interfaces.
//
// Data flow:
//   GpuGhostFillList::build(tree, bc) — rebuild after each regrid
//   GpuGhostFillList::exec(stream)    — called once per RK3 stage
//
// Face kernel (k_fill_faces): one CUDA block per (leaf, face) pair.
//   Grid: (n_leaves, NFACES)  Block: 256 threads
//   Handles: same-level copy, CF fine←coarse 5th-order Lagrange (P7.2),
//            wall reflect (all momenta), open zero-gradient, root periodic.
//   CF coarse←fine: zero-gradient fallback (upgraded to conservative average
//   in P8.4 when coarse-block fine-child pointers are added to metadata).
//
// Edge/corner kernel (k_fill_edges_corners): one CUDA block per leaf.
//   Grid: n_leaves  Block: 256 threads
//   Reads from own interior cells — no cross-block dependency.
//   Source formula: src_coord = (ghost < NG) ? ghost+NB : ghost-NB
//
// BCType encoding:  0=periodic  1=wall  2=open  3=nscbc  (must match ns_solver.hpp)

#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <cstdint>

// Forward declaration — full definition in mpi/mpi_comm.hpp.
struct MpiPartition;

// ── Per-leaf GPU ghost fill metadata ─────────────────────────────────────────
// Built on host, uploaded to device, reused every RK3 stage until next regrid.
struct alignas(64) GpuLeafGhostMeta {
    double*       d_Q;                    // offset 0,  8 B
    const double* d_nb[NFACES];           // offset 8,  48 B
    int8_t  level_rel[NFACES];            // offset 56, 6 B
    int8_t  cf_oct;                       // offset 62, 1 B
    int8_t  bc_type[NFACES];              // offset 63, 6 B
    int8_t  is_mpi_face[NFACES];          // offset 69, 6 B
    float   open_p_inf[NFACES];           // offset 76, 24 B (padded to 100 B → sizeof=128)
};
static_assert(sizeof(GpuLeafGhostMeta) <= 128, "GpuLeafGhostMeta too large");

// ── Ghost fill list ───────────────────────────────────────────────────────────
struct GpuGhostFillList {
    GpuLeafGhostMeta* d_metas = nullptr;  // device array
    int               n_leaves = 0;

    GpuGhostFillList() = default;
    GpuGhostFillList(const GpuGhostFillList&) = delete;
    GpuGhostFillList& operator=(const GpuGhostFillList&) = delete;
    ~GpuGhostFillList();

    // Rebuild after regrid.  bc_type: 0=periodic, 1=wall, 2=open (all faces same).
    // mpi_part: when non-null, marks remote-rank faces as is_mpi_face so the
    // kernel skips them (ghost cells are pre-filled by GpuMpiHaloList::exchange).
    void build(const BlockTree& tree, const GpuPool& pool, int bc_type,
               const MpiPartition* mpi_part = nullptr);

    // Per-face variant: bc_types[d] gives the BC for face d (0=periodic,1=wall,2=open).
    void build(const BlockTree& tree, const GpuPool& pool,
               const std::array<int,6>& bc_types,
               const MpiPartition* mpi_part = nullptr);

    // bc_types + per-face p_inf for NSCBC (bc_type==3).
    void build(const BlockTree& tree, const GpuPool& pool,
               const std::array<int,6>& bc_types,
               const std::array<float,6>& p_infs,
               const MpiPartition* mpi_part = nullptr);

    // Convenience overload: extracts bc_types and p_infs from FaceBCArray.
    void build(const BlockTree& tree, const GpuPool& pool,
               const FaceBCArray& face_bcs,
               const MpiPartition* mpi_part = nullptr);

    // Launch kernels for face fill + edge/corner fill on the given stream.
    void exec(cudaStream_t stream = nullptr) const;
};

// ── Device kernel declarations (for direct use from other .cu files) ─────────
extern "C" {
    void gpu_ghost_fill_faces_launch(const GpuLeafGhostMeta* d_metas,
                                     int n_leaves, cudaStream_t stream);
    void gpu_ghost_fill_ec_launch(const GpuLeafGhostMeta* d_metas,
                                   int n_leaves, cudaStream_t stream);
}
