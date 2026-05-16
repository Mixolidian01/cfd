// P-MPI-GPU: GPU halo exchange implementation.
//
// Download/upload is done for the FULL d_Q array of each leaf with remote faces
// (NVAR*NCELL doubles = 69 KB per leaf).  This reuses the existing CPU
// mpi_exchange_halos() without custom pack/unpack kernels.

#include "cuda/gpu_mpi_halo.cuh"
#include "cuda/gpu_check.cuh"
#include "mesh/cell_block.hpp"
#include <cassert>

GpuMpiHaloList::~GpuMpiHaloList()
{
    for (double* p : h_bufs_) if (p) cudaFreeHost(p);
}

void GpuMpiHaloList::build(const BlockTree& tree, const GpuPool& pool, MpiPartition* mpi)
{
    for (double* p : h_bufs_) if (p) cudaFreeHost(p);
    h_bufs_.clear();
    entries_.clear();
    tree_ = const_cast<BlockTree*>(&tree);
    mpi_  = mpi;

    if (!mpi || !mpi->active()) return;

    for (int li : mpi->local_leaves) {
        const BlockNode& nd = tree.nodes[li];
        if (!nd.has_block()) continue;
        bool has_remote = false;
        for (int d = 0; d < NFACES; ++d) {
            if (mpi_is_remote(mpi, nd.neighbours[d])) { has_remote = true; break; }
        }
        if (!has_remote) continue;
        double* h = nullptr;
        CUDA_CHECK(cudaMallocHost(&h, (size_t)NVAR * NCELL * sizeof(double)));
        entries_.push_back({pool.d_Q(nd.block.get()), li});
        h_bufs_.push_back(h);
    }
}

void GpuMpiHaloList::exchange(cudaStream_t stream)
{
    if (!active()) return;

    // Wait for any GPU kernels queued before this call (e.g. RHS from previous stage)
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // D2H: download current d_Q → flat pinned buffer → refresh CPU CellBlock
    for (int i = 0; i < (int)entries_.size(); ++i) {
        const LeafEntry& e = entries_[i];
        CUDA_CHECK(cudaMemcpy(h_bufs_[i], e.d_Q,
                              (size_t)NVAR * NCELL * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CellBlock* blk = tree_->nodes[e.li].block.get();
        assert(blk);
        for (int v = 0; v < NVAR; ++v)
            blk->Q[v].assign_from_flat(h_bufs_[i] + v * NCELL);
    }

    // CPU MPI exchange: pack real faces → send; recv → unpack into ghost cells
    mpi_exchange_halos(*tree_, mpi_);

    // H2D: flatten updated CPU blocks (real + MPI ghost cells) → upload to GPU
    for (int i = 0; i < (int)entries_.size(); ++i) {
        const LeafEntry& e = entries_[i];
        CellBlock* blk = tree_->nodes[e.li].block.get();
        for (int v = 0; v < NVAR; ++v)
            blk->Q[v].copy_to_flat(h_bufs_[i] + v * NCELL);
        CUDA_CHECK(cudaMemcpyAsync(e.d_Q, h_bufs_[i],
                                   (size_t)NVAR * NCELL * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
}
