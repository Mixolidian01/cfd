#pragma once
// P4.3 — Thread Block Cluster (TBC) ghost exchange for H100+ GPUs.
//
// When two adjacent CFD blocks form a CTA cluster, each CTA's shared memory
// is directly readable by the other (Distributed Shared Memory, DSMEM).
// Face data that would otherwise round-trip through global memory can be
// exchanged inside the cluster without any L2 traffic.
//
// Algorithm (x-direction pair):
//   CTA rank 0 → left block b0,  CTA rank 1 → right block b1.
//   Each CTA loads GPU_NG interior face layers into shmem.
//   After cluster.sync(), each CTA reads the peer's shmem via
//   cooperative_groups::this_cluster().map_shared_rank() and writes
//   the face data into its own ghost layers in global memory.
//
// Hardware requirement: sm_90+ (CUDA 12+, H100/B200).
// On sm_86 and below the function gpu_ghost_exchange_x falls back to a
// standard global-memory copy that is functionally identical.
//
// API:
//   gpu_ghost_exchange_x(Q, pairs, n_pairs, n_blocks, stream)
//   gpu_ghost_exchange_y(...)
//   gpu_ghost_exchange_z(...)
//
// pairs[p*2+0] = index of left/lower block for pair p
// pairs[p*2+1] = index of right/upper block for pair p
// Q layout: Q[v * n_blocks * GPU_NCELL + b * GPU_NCELL + flat_cell]

#include "gpu_constants.cuh"
#include <cuda_runtime.h>

// Face cell count (per variable, per direction) = NB2 × NB2 × NG
static constexpr int GPU_FACE_SIZE = GPU_NB2 * GPU_NB2 * GPU_NG;

// ── Host-callable dispatch functions ─────────────────────────────────────────
void gpu_ghost_exchange_x(double* Q,
                          const int* d_pairs, int n_pairs,
                          int n_blocks,
                          cudaStream_t stream = nullptr);

void gpu_ghost_exchange_y(double* Q,
                          const int* d_pairs, int n_pairs,
                          int n_blocks,
                          cudaStream_t stream = nullptr);

void gpu_ghost_exchange_z(double* Q,
                          const int* d_pairs, int n_pairs,
                          int n_blocks,
                          cudaStream_t stream = nullptr);

// Query whether TBC kernels were actually used (sm_90+) or the fallback ran.
bool gpu_tbc_available();
