// P7.1 — MPI domain decomposition implementation
//
// See include/mpi_comm.hpp for architecture notes.

#include "mpi/mpi_comm.hpp"
#include "mesh/cell_block.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <vector>

// =============================================================================
// MpiEnvironment
// =============================================================================

MpiEnvironment::MpiEnvironment(int& argc, char**& argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    comm_ = MPI_COMM_WORLD;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
#else
    (void)argc; (void)argv;
    comm_ = 0; rank_ = 0; size_ = 1;
#endif
}

MpiEnvironment::~MpiEnvironment() {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}

// =============================================================================
// mpi_partition — Morton Z-order stripe assignment
// =============================================================================

void mpi_partition(const BlockTree& tree, MpiPartition* part) {
#ifdef HAVE_MPI
    MPI_Comm_rank(part->comm, &part->my_rank);
    MPI_Comm_size(part->comm, &part->n_ranks);
#else
    part->my_rank = 0;
    part->n_ranks = 1;
#endif

    const auto& leaves = tree.leaf_indices();
    const int   NL     = static_cast<int>(leaves.size());

    // Sort leaves by Morton code (Z-order spatial locality)
    std::vector<int> sorted = leaves;
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
        return tree.nodes[a].morton < tree.nodes[b].morton;
    });

    part->leaf_owner.assign(tree.nodes.size(), -1);
    part->local_leaves.clear();

    for (int i = 0; i < NL; ++i) {
        int rank = (i * part->n_ranks) / NL;   // contiguous stripe
        int li   = sorted[i];
        part->leaf_owner[li] = rank;
        if (rank == part->my_rank) part->local_leaves.push_back(li);
    }
}

// =============================================================================
// mpi_alloc_local_blocks
// =============================================================================

void mpi_alloc_local_blocks(BlockTree& tree, const MpiPartition& part,
                             double h0, const CellBlock*)
{
    for (auto& nd : tree.nodes) {
        if (!nd.is_leaf()) continue;
        int li = static_cast<int>(&nd - tree.nodes.data());
        if (!part.is_local(li)) {
            // Free block memory on non-owning ranks (keep topology intact)
            nd.block.reset();
        } else if (!nd.block) {
            // Allocate if missing (e.g. after regrid on this rank)
            // Compute h from level: h = h0 / 2^level  where h0 = L/NB
            double h = h0;
            for (int l = 0; l < nd.level; ++l) h *= 0.5;
            nd.block = std::make_unique<CellBlock>(nd.ox, nd.oy, nd.oz, h);
        }
    }
}

// =============================================================================
// mpi_exchange_halos
//
// Protocol (per neighbor rank R):
//   1. For each local leaf li and each face d where neighbor ni is owned by R:
//        pack NG planes of real cells from li (outgoing side) into send buffer
//   2. Compute recv sizes by symmetry (same formula on R)
//   3. MPI_Isend(send_buf[R]) + MPI_Irecv(recv_buf[R])
//   4. MPI_Waitall
//   5. Unpack: for each received face, write NG ghost planes of local leaf li
//
// Face plane layout in the buffer (one face entry):
//   [4-byte LE leaf_index][4-byte LE face_dir][NG×NB2×NB2×NVAR doubles]
// =============================================================================

// Number of doubles in one halo face
static constexpr int HALO_FACE_DOUBLES = NG * NB2 * NB2 * NVAR;
static constexpr int HALO_FACE_HEADER  = 2;   // leaf_index + face_dir (as int32)

struct FaceEntry {
    int li;
    int face;
};

static void pack_face(const CellBlock& blk, FaceDir d,
                      double* buf /* HALO_FACE_DOUBLES doubles */)
{
    // Pack the NG real-cell planes adjacent to face d.
    // These will fill the NG ghost planes of the neighbor on the opposite side.
    int ptr = 0;

    auto pack_cell = [&](int i, int j, int k) {
        int idx = cell_idx(i, j, k);
        for (int v = 0; v < NVAR; ++v)
            buf[ptr++] = blk.Q[v][idx];
    };

    // Real planes to send for each direction:
    //   XMINUS: send planes i = [NG, 2*NG-1]         → fills neighbor's XPLUS  ghost i=[NB2-NG,NB2-1]
    //   XPLUS : send planes i = [NB2-2*NG, NB2-NG-1] → fills neighbor's XMINUS ghost i=[0,NG-1]
    //   YMINUS: send planes j = [NG, 2*NG-1]
    //   YPLUS : send planes j = [NB2-2*NG, NB2-NG-1]
    //   ZMINUS: send planes k = [NG, 2*NG-1]
    //   ZPLUS : send planes k = [NB2-2*NG, NB2-NG-1]

    switch (d) {
    case XMINUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
            pack_cell(NG + p, j, k);
        break;
    case XPLUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
            pack_cell(NB2 - 2*NG + p, j, k);
        break;
    case YMINUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int i = 0; i < NB2; ++i)
            pack_cell(i, NG + p, k);
        break;
    case YPLUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int i = 0; i < NB2; ++i)
            pack_cell(i, NB2 - 2*NG + p, k);
        break;
    case ZMINUS:
        for (int p = 0; p < NG; ++p)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i)
            pack_cell(i, j, NG + p);
        break;
    case ZPLUS:
        for (int p = 0; p < NG; ++p)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i)
            pack_cell(i, j, NB2 - 2*NG + p);
        break;
    default: break;
    }
    assert(ptr == HALO_FACE_DOUBLES);
}

static void unpack_face(CellBlock& blk, FaceDir d,
                        const double* buf /* HALO_FACE_DOUBLES doubles */)
{
    // Unpack data sent by the neighbor for face d of blk.
    // The sender packed its real cells; we place them into our ghost cells.
    //   Received for our XMINUS ghost (d=XMINUS): write i=[0,NG-1]
    //   Received for our XPLUS  ghost (d=XPLUS ): write i=[NB2-NG,NB2-1]
    //   etc.

    int ptr = 0;
    auto unpack_cell = [&](int i, int j, int k) {
        int idx = cell_idx(i, j, k);
        for (int v = 0; v < NVAR; ++v)
            blk.Q[v][idx] = buf[ptr++];
    };

    switch (d) {
    case XMINUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
            unpack_cell(p, j, k);
        break;
    case XPLUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
            unpack_cell(NB2 - NG + p, j, k);
        break;
    case YMINUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int i = 0; i < NB2; ++i)
            unpack_cell(i, p, k);
        break;
    case YPLUS:
        for (int p = 0; p < NG; ++p)
        for (int k = 0; k < NB2; ++k)
        for (int i = 0; i < NB2; ++i)
            unpack_cell(i, NB2 - NG + p, k);
        break;
    case ZMINUS:
        for (int p = 0; p < NG; ++p)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i)
            unpack_cell(i, j, p);
        break;
    case ZPLUS:
        for (int p = 0; p < NG; ++p)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i)
            unpack_cell(i, j, NB2 - NG + p);
        break;
    default: break;
    }
    assert(ptr == HALO_FACE_DOUBLES);
}

void mpi_exchange_halos(BlockTree& tree, const MpiPartition* mpi_part) {
    if (!mpi_part || !mpi_part->active()) return;
    const MpiPartition& part = *mpi_part;

#ifdef HAVE_MPI
    const int NR = part.n_ranks;

    // Identify all remote face pairs: (local_leaf, face_dir, remote_rank)
    struct RemoteFace { int li; FaceDir d; int remote_rank; };
    std::vector<RemoteFace> send_faces;  // faces we send data for
    // recv_faces == same list (symmetric: neighbor ri sends us data for face opposite(d))

    for (int li : part.local_leaves) {
        const BlockNode& nd = tree.nodes[li];
        for (int fd = 0; fd < NFACES; ++fd) {
            int ni = nd.neighbours[fd];
            if (ni < 0) continue;
            if (part.is_remote(ni)) {
                send_faces.push_back({li, static_cast<FaceDir>(fd), part.leaf_owner[ni]});
            }
        }
    }

    // Build per-rank send buffers
    // send_buf[r] contains all face data going to rank r
    // Each face: [int32 li][int32 face_dir][HALO_FACE_DOUBLES doubles]
    const int ENTRY_INTS    = 2;
    const int ENTRY_DOUBLES = HALO_FACE_DOUBLES;
    // We'll pack everything as doubles (2 ints packed into 1 double each — or use bytes).
    // Simpler: separate int header + double payload, or pack as MPI_BYTE.
    // Use MPI_BYTE for generality.

    // Bytes per face entry: 2 int32s + HALO_FACE_DOUBLES doubles
    const int ENTRY_BYTES = ENTRY_INTS * 4 + ENTRY_DOUBLES * 8;

    std::vector<std::vector<uint8_t>> send_bufs(NR);
    for (const auto& sf : send_faces) {
        auto& buf = send_bufs[sf.remote_rank];
        size_t off = buf.size();
        buf.resize(off + ENTRY_BYTES);
        // Header: li, face_dir
        int32_t hdr[2] = {static_cast<int32_t>(sf.li), static_cast<int32_t>(sf.d)};
        std::memcpy(buf.data() + off, hdr, 8);
        // Payload: packed real cells
        double tmp[HALO_FACE_DOUBLES];
        pack_face(*tree.nodes[sf.li].block, sf.d, tmp);
        std::memcpy(buf.data() + off + 8, tmp, ENTRY_DOUBLES * 8);
    }

    // Exchange buffer sizes first (each rank needs to know how much to receive)
    std::vector<int> send_sizes(NR, 0), recv_sizes(NR, 0);
    for (int r = 0; r < NR; ++r)
        send_sizes[r] = static_cast<int>(send_bufs[r].size());
    MPI_Alltoall(send_sizes.data(), 1, MPI_INT,
                 recv_sizes.data(), 1, MPI_INT, part.comm);

    // Allocate recv buffers
    std::vector<std::vector<uint8_t>> recv_bufs(NR);
    for (int r = 0; r < NR; ++r)
        if (recv_sizes[r] > 0) recv_bufs[r].resize(recv_sizes[r]);

    // Non-blocking send/recv
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * NR);

    for (int r = 0; r < NR; ++r) {
        if (r == part.my_rank) continue;
        if (recv_sizes[r] > 0) {
            MPI_Request rq;
            MPI_Irecv(recv_bufs[r].data(), recv_sizes[r], MPI_BYTE,
                      r, 42, part.comm, &rq);
            reqs.push_back(rq);
        }
    }
    for (int r = 0; r < NR; ++r) {
        if (r == part.my_rank) continue;
        if (send_sizes[r] > 0) {
            MPI_Request rq;
            MPI_Isend(send_bufs[r].data(), send_sizes[r], MPI_BYTE,
                      r, 42, part.comm, &rq);
            reqs.push_back(rq);
        }
    }
    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // Unpack recv buffers: fill ghost cells of local leaves
    for (int r = 0; r < NR; ++r) {
        if (recv_sizes[r] == 0) continue;
        const uint8_t* p   = recv_bufs[r].data();
        const uint8_t* end = p + recv_sizes[r];
        while (p < end) {
            // Header: sender's li (the remote leaf), sender's face_dir (their outgoing face)
            int32_t hdr[2];
            std::memcpy(hdr, p, 8); p += 8;
            int     remote_li   = hdr[0];
            FaceDir remote_face = static_cast<FaceDir>(hdr[1]);

            // The sender packed their face `remote_face` of leaf `remote_li`.
            // That leaf is the neighbor of some local leaf on face opposite(remote_face).
            // Find which local leaf has remote_li as its `opposite(remote_face)` neighbor.
            FaceDir local_recv_face = opposite(remote_face);

            double tmp[HALO_FACE_DOUBLES];
            std::memcpy(tmp, p, HALO_FACE_DOUBLES * 8); p += HALO_FACE_DOUBLES * 8;

            // Find the local leaf whose neighbor in direction local_recv_face is remote_li
            // (guaranteed to exist since send_faces was built symmetrically)
            const BlockNode& remote_nd = tree.nodes[remote_li];
            int local_li = remote_nd.neighbours[static_cast<int>(remote_face)];
            if (local_li < 0 || !part.is_local(local_li)) continue;
            if (!tree.nodes[local_li].block) continue;

            unpack_face(*tree.nodes[local_li].block, local_recv_face, tmp);
        }
    }
#else
    (void)tree; (void)part;
#endif
}

// =============================================================================
// Global reductions
// =============================================================================

double mpi_allreduce_min(double local_val, const MpiPartition* part) {
#ifdef HAVE_MPI
    if (!part || !part->active()) return local_val;
    double global_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MIN, part->comm);
    return global_val;
#else
    (void)part;
    return local_val;
#endif
}

double mpi_allreduce_sum(double local_val, const MpiPartition* part) {
#ifdef HAVE_MPI
    if (!part || !part->active()) return local_val;
    double global_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, part->comm);
    return global_val;
#else
    (void)part;
    return local_val;
#endif
}
