#pragma once
// P-MPI-GPU: GPU halo exchange via D2H → MPI → H2D staging.
//
// Protocol per RK3 stage (called before ghost_list.exec()):
//   1. cudaStreamSynchronize — wait for GPU RHS from previous stage
//   2. cudaMemcpy D2H — download d_Q for each local leaf with remote face(s)
//   3. assign_from_flat — refresh CPU CellBlock from flat buffer
//   4. mpi_exchange_halos — CPU MPI pack/send/recv/unpack into CellBlock ghosts
//   5. copy_to_flat — flatten updated CellBlock (real + MPI ghost cells)
//   6. cudaMemcpyAsync H2D — upload updated block back to GPU (on stream)
//
// ghost_list's is_mpi_face flags prevent k_fill_faces from overwriting the
// MPI-filled ghost cells with periodic self-wrap.
//
// CFL: after cfl_list.exec() returns local dt, mpi_allreduce_min reduces to
// global dt and the result is written back to cfl_list.d_dt on device so all
// RK3 update kernels use the same dt.

#include "mesh/block_tree.hpp"
#include "mpi/mpi_comm.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <vector>

struct GpuMpiHaloList {
    struct LeafEntry { double* d_Q; int li; };

    std::vector<LeafEntry> entries_;   // local leaves with ≥1 remote-rank face
    std::vector<double*>   h_bufs_;   // pinned host staging (NVAR*NCELL doubles each)

    BlockTree*    tree_ = nullptr;
    MpiPartition*    mpi_  = nullptr;

    GpuMpiHaloList() = default;
    GpuMpiHaloList(const GpuMpiHaloList&) = delete;
    GpuMpiHaloList& operator=(const GpuMpiHaloList&) = delete;
    ~GpuMpiHaloList();

    bool active() const noexcept { return mpi_ && mpi_->active() && !entries_.empty(); }

    // Rebuild after regrid.  Pass nullptr mpi to deactivate.
    void build(const BlockTree& tree, const GpuPool& pool, MpiPartition* mpi);

    // D2H → mpi_exchange_halos (CPU) → H2D.
    // Synchronises stream before D2H; queues H2D on stream.
    void exchange(cudaStream_t stream);
};
