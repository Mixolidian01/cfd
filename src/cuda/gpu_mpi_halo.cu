// D2: GPU face-pack halo exchange.
//
// GPU pack/unpack kernels operate on HALO_FACE_DOUBLES = NG*NB2*NB2*NVAR = 1440
// doubles per face — 6× smaller than the full-block (NVAR*NCELL = 8640) staging
// used by the previous P-MPI-GPU implementation.
//
// GPU send/recv buffers are always allocated (both paths below):
//   CUDA-aware path (#ifdef MPIX_CUDA_AWARE_SUPPORT):
//     pack → d_send → MPI_Isend with GPU ptr → MPI_Waitall → GPU unpack from d_recv
//   CPU-staging path (default):
//     pack → d_send → D2H → h_send → MPI_Isend → MPI_Waitall → h_recv → H2D → d_recv → unpack
//
// MPI_Irecv is posted BEFORE cudaStreamSynchronize for comm/compute overlap.
// Face count precomputed at build() → no per-step MPI_Alltoall.

#include "cuda/gpu_mpi_halo.cuh"
#include "cuda/gpu_check.cuh"
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include <algorithm>
#include <cassert>
#include <vector>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif

// ── GPU pack kernel ───────────────────────────────────────────────────────────
// HALO_FACE_DOUBLES threads per face; decodes tid → (v, a, b, p) → (i,j,k).
// face_dir: 0=XMINUS, 1=XPLUS, 2=YMINUS, 3=YPLUS, 4=ZMINUS, 5=ZPLUS
__global__
void k_pack_face(const double* __restrict__ d_Q,
                 double*       __restrict__ d_buf,
                 int face_dir)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= HALO_FACE_DOUBLES) return;

    int tmp = tid;
    const int v = tmp % NVAR;  tmp /= NVAR;
    const int a = tmp % NB2;   tmp /= NB2;   // first transverse axis (j or i)
    const int b = tmp % NB2;   tmp /= NB2;   // second transverse axis (k or j)
    const int p = tmp;                         // plane index [0, NG)

    int i, j, k;
    switch (face_dir) {
    case 0: i = NG + p;           j = a;              k = b;              break; // XMINUS
    case 1: i = NB2 - 2*NG + p;   j = a;              k = b;              break; // XPLUS
    case 2: i = a;                 j = NG + p;         k = b;              break; // YMINUS
    case 3: i = a;                 j = NB2 - 2*NG + p; k = b;              break; // YPLUS
    case 4: i = a;                 j = b;              k = NG + p;         break; // ZMINUS
    case 5: i = a;                 j = b;              k = NB2 - 2*NG + p; break; // ZPLUS
    default: return;
    }
    d_buf[tid] = d_Q[v * NCELL + cell_idx(i, j, k)];
}

// ── GPU unpack kernel ─────────────────────────────────────────────────────────
// Writes received ghost data into the ghost planes of d_Q (local face direction).
__global__
void k_unpack_face(double*       __restrict__ d_Q,
                   const double* __restrict__ d_buf,
                   int face_dir)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= HALO_FACE_DOUBLES) return;

    int tmp = tid;
    const int v = tmp % NVAR;  tmp /= NVAR;
    const int a = tmp % NB2;   tmp /= NB2;
    const int b = tmp % NB2;   tmp /= NB2;
    const int p = tmp;

    int i, j, k;
    switch (face_dir) {
    case 0: i = p;                 j = a;              k = b;              break; // XMINUS ghost
    case 1: i = NB2 - NG + p;      j = a;              k = b;              break; // XPLUS  ghost
    case 2: i = a;                  j = p;              k = b;              break; // YMINUS ghost
    case 3: i = a;                  j = NB2 - NG + p;   k = b;              break; // YPLUS  ghost
    case 4: i = a;                  j = b;              k = p;              break; // ZMINUS ghost
    case 5: i = a;                  j = b;              k = NB2 - NG + p;   break; // ZPLUS  ghost
    default: return;
    }
    d_Q[v * NCELL + cell_idx(i, j, k)] = d_buf[tid];
}

// ── Destructor ────────────────────────────────────────────────────────────────
GpuMpiHaloList::~GpuMpiHaloList()
{
    for (auto& rb : rank_bufs_) {
        if (rb.h_send) cudaFreeHost(rb.h_send);
        if (rb.h_recv) cudaFreeHost(rb.h_recv);
        if (rb.d_send) cudaFree(rb.d_send);
        if (rb.d_recv) cudaFree(rb.d_recv);
    }
}

// ── build ─────────────────────────────────────────────────────────────────────
void GpuMpiHaloList::build(const BlockTree& tree, const GpuPool& pool,
                            MpiPartition* mpi)
{
    for (auto& rb : rank_bufs_) {
        if (rb.h_send) cudaFreeHost(rb.h_send);
        if (rb.h_recv) cudaFreeHost(rb.h_recv);
        if (rb.d_send) cudaFree(rb.d_send);
        if (rb.d_recv) cudaFree(rb.d_recv);
    }
    send_entries_.clear();
    recv_entries_.clear();
    rank_bufs_.clear();
    tree_ = const_cast<BlockTree*>(&tree);
    mpi_  = mpi;

    if (!mpi || !mpi->active()) return;

    const int NR = mpi->n_ranks;
    rank_bufs_.resize(NR);

    // Build send entries: each local leaf × remote face → send to remote rank.
    for (int li : mpi->local_leaves) {
        const BlockNode& nd = tree.nodes[li];
        if (!nd.has_block()) continue;
        double* dq = pool.d_Q(nd.block.get());
        for (int fd = 0; fd < NFACES; ++fd) {
            int ni = nd.neighbours[fd];
            if (!mpi->is_remote(ni)) continue;
            int rr   = mpi->leaf_owner[ni];
            int slot = rank_bufs_[rr].n_send++;
            send_entries_.push_back({dq, li, fd, rr, slot});
        }
    }

#ifdef HAVE_MPI
    // Exchange face counts so each rank knows how many faces to receive.
    {
        std::vector<int> lc(NR, 0), rc(NR, 0);
        for (const auto& e : send_entries_) lc[e.rank]++;
        MPI_Alltoall(lc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, mpi->comm);
        for (int r = 0; r < NR; ++r) rank_bufs_[r].n_recv = rc[r];
    }
#endif

    // Build recv entries.
    // ORDERING INVARIANT: recv entry slot i from rank R must correspond to
    // rank R's send entry slot i (both iterate rank R's local_leaves × faces).
    // Since the tree topology is replicated, we reconstruct rank R's local_leaves
    // (Morton-sorted stripe, same algorithm as mpi_partition) and iterate them here.
    {
        // Reconstruct Morton-sorted leaf order (identical to mpi_partition's sorted list).
        std::vector<int> sorted_leaves = tree.leaf_indices();
        std::sort(sorted_leaves.begin(), sorted_leaves.end(), [&](int a, int b) {
            return tree.nodes[a].morton < tree.nodes[b].morton;
        });

        std::vector<int> recv_slot(NR, 0);
        for (int r = 0; r < NR; ++r) {
            if (rank_bufs_[r].n_recv == 0) continue;
            // Simulate rank r's iteration order over its local_leaves.
            for (int li : sorted_leaves) {
                if (mpi->leaf_owner[li] != r) continue;  // only rank r's leaves
                // Rank r iterates its face dirs; for each face where neighbor is ours:
                for (int fd = 0; fd < NFACES; ++fd) {
                    int ni = tree.nodes[li].neighbours[fd];
                    if (!mpi->is_local(ni)) continue;     // ni must be our local leaf
                    if (!tree.nodes[ni].has_block()) continue;
                    // Rank r sends face (li, fd); we receive into our leaf ni at ghost fd_ghost.
                    // ghost face direction: opposite of fd (sender's face XPLUS fills our XMINUS ghost)
                    int fd_ghost = opposite(static_cast<FaceDir>(fd));
                    double* dq = pool.d_Q(tree.nodes[ni].block.get());
                    recv_entries_.push_back({dq, ni, fd_ghost, r, recv_slot[r]++});
                }
            }
        }
    }

    // Allocate pinned host + GPU buffers per rank (both paths use GPU pack/unpack).
    for (int r = 0; r < NR; ++r) {
        auto& rb = rank_bufs_[r];
        if (rb.n_send > 0) {
            size_t bytes = (size_t)rb.n_send * HALO_FACE_DOUBLES * sizeof(double);
            CUDA_CHECK(cudaMallocHost(&rb.h_send, bytes));
            CUDA_CHECK(cudaMalloc(&rb.d_send, bytes));
        }
        if (rb.n_recv > 0) {
            size_t bytes = (size_t)rb.n_recv * HALO_FACE_DOUBLES * sizeof(double);
            CUDA_CHECK(cudaMallocHost(&rb.h_recv, bytes));
            CUDA_CHECK(cudaMalloc(&rb.d_recv, bytes));
        }
    }
}

// ── exchange ──────────────────────────────────────────────────────────────────
void GpuMpiHaloList::exchange(cudaStream_t stream)
{
    if (!active()) return;

#ifdef HAVE_MPI
    const int NR  = mpi_->n_ranks;
    static constexpr int THR = 256;
    static constexpr int BLK = (HALO_FACE_DOUBLES + THR - 1) / THR;

    // ── 1. Post MPI_Irecv BEFORE cudaStreamSynchronize for comm/compute overlap
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * NR);
    for (int r = 0; r < NR; ++r) {
        auto& rb = rank_bufs_[r];
        if (rb.n_recv == 0 || r == mpi_->my_rank) continue;
        int count = rb.n_recv * HALO_FACE_DOUBLES;
        MPI_Request rq;
#ifdef MPIX_CUDA_AWARE_SUPPORT
        MPI_Irecv(rb.d_recv, count, MPI_DOUBLE, r, 43, mpi_->comm, &rq);
#else
        MPI_Irecv(rb.h_recv, count, MPI_DOUBLE, r, 43, mpi_->comm, &rq);
#endif
        reqs.push_back(rq);
    }

    // ── 2. Wait for GPU RHS kernels before touching d_Q ─────────────────────
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 3. GPU pack: extract face planes from d_Q into d_send per rank ───────
    for (const auto& e : send_entries_) {
        double* dst = rank_bufs_[e.rank].d_send + (size_t)e.slot * HALO_FACE_DOUBLES;
        k_pack_face<<<BLK, THR, 0, stream>>>(e.d_Q, dst, e.fd);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef MPIX_CUDA_AWARE_SUPPORT
    // ── CUDA-aware: send GPU buffers directly ────────────────────────────────
    for (int r = 0; r < NR; ++r) {
        auto& rb = rank_bufs_[r];
        if (rb.n_send == 0 || r == mpi_->my_rank) continue;
        int count = rb.n_send * HALO_FACE_DOUBLES;
        MPI_Request rq;
        MPI_Isend(rb.d_send, count, MPI_DOUBLE, r, 43, mpi_->comm, &rq);
        reqs.push_back(rq);
    }
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    // GPU unpack directly from d_recv
    for (const auto& e : recv_entries_) {
        const double* src = rank_bufs_[e.rank].d_recv + (size_t)e.slot * HALO_FACE_DOUBLES;
        k_unpack_face<<<BLK, THR, 0, stream>>>(e.d_Q, src, e.fd);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
#else
    // ── CPU-staging: D2H → MPI → H2D ────────────────────────────────────────
    for (int r = 0; r < NR; ++r) {
        auto& rb = rank_bufs_[r];
        if (rb.n_send == 0) continue;
        size_t bytes = (size_t)rb.n_send * HALO_FACE_DOUBLES * sizeof(double);
        CUDA_CHECK(cudaMemcpy(rb.h_send, rb.d_send, bytes, cudaMemcpyDeviceToHost));
    }
    for (int r = 0; r < NR; ++r) {
        auto& rb = rank_bufs_[r];
        if (rb.n_send == 0 || r == mpi_->my_rank) continue;
        int count = rb.n_send * HALO_FACE_DOUBLES;
        MPI_Request rq;
        MPI_Isend(rb.h_send, count, MPI_DOUBLE, r, 43, mpi_->comm, &rq);
        reqs.push_back(rq);
    }
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    for (int r = 0; r < NR; ++r) {
        auto& rb = rank_bufs_[r];
        if (rb.n_recv == 0) continue;
        size_t bytes = (size_t)rb.n_recv * HALO_FACE_DOUBLES * sizeof(double);
        CUDA_CHECK(cudaMemcpy(rb.d_recv, rb.h_recv, bytes, cudaMemcpyHostToDevice));
    }
    for (const auto& e : recv_entries_) {
        const double* src = rank_bufs_[e.rank].d_recv + (size_t)e.slot * HALO_FACE_DOUBLES;
        k_unpack_face<<<BLK, THR, 0, stream>>>(e.d_Q, src, e.fd);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif  // MPIX_CUDA_AWARE_SUPPORT

#else  // !HAVE_MPI
    (void)stream;
#endif
}

// ── Test helpers ──────────────────────────────────────────────────────────────
static constexpr int _THR = 256;
static constexpr int _BLK = (HALO_FACE_DOUBLES + _THR - 1) / _THR;

void gpu_pack_face(const double* d_Q, double* d_buf, int fd, cudaStream_t stream)
{
    k_pack_face<<<_BLK, _THR, 0, stream>>>(d_Q, d_buf, fd);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_unpack_face(double* d_Q, const double* d_buf, int fd, cudaStream_t stream)
{
    k_unpack_face<<<_BLK, _THR, 0, stream>>>(d_Q, d_buf, fd);
    CUDA_CHECK(cudaDeviceSynchronize());
}
