#pragma once
// P8.5: GPU CFL reduction — warp-shuffle tree reduction over all leaves.
//
// Design goals:
//   • Zero D→H bandwidth on interior (non-diagnostic) steps.
//     The dt value lives in d_dt (device memory) and is passed directly
//     to RK3 update kernels as a pointer.  No host copy unless requested.
//   • Warp-shuffle min reduction eliminates shared-memory atomics within each
//     warp; only one atomicMin per SM crosses to global memory.
//   • Single kernel launch over all leaves × interior cells in one grid.
//
// Data flow:
//   GpuCflList::build(tree)     — rebuild after each regrid
//   GpuCflList::exec(cfl, stream) — launches k_cfl_reduce; returns host dt
//   GpuCflList::d_dt            — device pointer to dt for pointer-dt kernels
//
// Kernel layout:
//   Grid: ceil(n_leaves × NB³ / blockDim.x) blocks
//   Block: 256 threads (8 warps, 32 lanes)
//   Each thread reduces one or more cells; warp-shuffle gives intra-warp min;
//   shared-memory + first-warp pattern gives block min; atomicMin to d_dt_bits.

#include "../cell_block.hpp"
#include "../block_tree.hpp"
#include <cuda_runtime.h>
#include <cstdint>

// ── Per-leaf CFL metadata ────────────────────────────────────────────────────
struct alignas(32) GpuLeafCflMeta {
    const double* d_Q;   // device pointer (flat SoA: Q[v*NCELL + flat])
    double        h;     // cell size
    int32_t       _pad[2];
};
static_assert(sizeof(GpuLeafCflMeta) <= 32, "GpuLeafCflMeta too large");

// ── CFL list ─────────────────────────────────────────────────────────────────
struct GpuCflList {
    GpuLeafCflMeta*     d_metas    = nullptr;
    unsigned long long* d_dt_bits  = nullptr;  // atomicMin scratch (uint64)
    double*             d_dt       = nullptr;  // live device dt (double)
    int                 n_leaves   = 0;

    GpuCflList() = default;
    GpuCflList(const GpuCflList&) = delete;
    GpuCflList& operator=(const GpuCflList&) = delete;
    ~GpuCflList();

    // Rebuild after regrid.
    void build(const BlockTree& tree);

    // Launch k_cfl_reduce; synchronises; copies d_dt to host; returns dt.
    // If stream != nullptr, the memcpy is async on that stream and this
    // function returns 0.0 — caller must sync and read d_dt explicitly.
    double exec(double cfl, cudaStream_t stream = nullptr) const;
};
