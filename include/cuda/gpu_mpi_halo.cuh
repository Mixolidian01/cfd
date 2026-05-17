#pragma once
// D2: GPU face-pack halo exchange.
//
// Replaces the full-block D2H→CPU MPI→H2D staging (P-MPI-GPU) with:
//   1. GPU k_pack_face: extract NG ghost planes from d_Q into a face buffer
//      (HALO_FACE_DOUBLES doubles ≈ 11.5 KB per face, vs 69 KB full block).
//   2. MPI_Irecv posted BEFORE cudaStreamSynchronize to maximise overlap.
//   3. cudaStreamSynchronize — wait for RHS kernels on stream.
//   4. D2H: copy face pack buffer to pinned host.
//   5. MPI_Isend (CPU-staging) or GPU-buffer pointer (CUDA-aware, #ifdef guarded).
//   6. MPI_Waitall.
//   7. GPU k_unpack_face: write received ghost data directly into d_Q ghost planes.
//
// CUDA-aware path: guarded by MPIX_CUDA_AWARE_SUPPORT.  Falls back to
// CPU-staging (D2H → MPI → H2D) if not available at compile time.
//
// Face count precomputed at build() → no per-step MPI_Alltoall.

#include "mesh/block_tree.hpp"
#include "mpi/mpi_comm.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <vector>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif

struct GpuMpiHaloList {
    // Per (leaf, face_direction) send/recv entry
    struct FaceEntry {
        double* d_Q;    // device block pointer
        int      li;    // local leaf index
        int      fd;    // face direction (FaceDir enum value, 0..5)
        int      rank;  // remote MPI rank
        int      slot;  // offset (in HALO_FACE_DOUBLES units) within rank's buffer
    };

    // Per-rank I/O buffers (indexed by rank, NR entries)
    struct RankBuf {
        int n_send = 0;   // send face count
        int n_recv = 0;   // recv face count
        double* h_send = nullptr;  // pinned host send buffer (n_send * HALO_FACE_DOUBLES doubles)
        double* h_recv = nullptr;  // pinned host recv buffer (n_recv * HALO_FACE_DOUBLES doubles)
        // D2: GPU send/recv buffers (CUDA-aware path)
        double* d_send = nullptr;
        double* d_recv = nullptr;
    };

    std::vector<FaceEntry> send_entries_;
    std::vector<FaceEntry> recv_entries_;
    std::vector<RankBuf>   rank_bufs_;

    BlockTree*    tree_ = nullptr;
    MpiPartition* mpi_  = nullptr;

    GpuMpiHaloList() = default;
    GpuMpiHaloList(const GpuMpiHaloList&) = delete;
    GpuMpiHaloList& operator=(const GpuMpiHaloList&) = delete;
    ~GpuMpiHaloList();

    bool active() const noexcept {
        return mpi_ && mpi_->active() && !send_entries_.empty();
    }

    // Rebuild after regrid.  Pass nullptr mpi to deactivate.
    void build(const BlockTree& tree, const GpuPool& pool, MpiPartition* mpi);

    // GPU pack → D2H → MPI Isend/Irecv → Waitall → H2D → GPU unpack.
    // MPI_Irecv is posted before cudaStreamSynchronize for overlap.
    void exchange(cudaStream_t stream);
};

// ── Test helpers (single-face pack/unpack for unit tests) ─────────────────────
// Pack NG real planes of face fd from d_Q (NVAR*NCELL doubles) into
// d_buf (HALO_FACE_DOUBLES doubles). Synchronous.
void gpu_pack_face(const double* d_Q, double* d_buf, int fd,
                   cudaStream_t stream = nullptr);

// Unpack d_buf (HALO_FACE_DOUBLES doubles) into the ghost planes of face fd
// in d_Q (NVAR*NCELL doubles). Synchronous.
void gpu_unpack_face(double* d_Q, const double* d_buf, int fd,
                     cudaStream_t stream = nullptr);
