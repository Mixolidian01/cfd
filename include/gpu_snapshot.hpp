#pragma once
// GPU-side 2D slice extraction + conserved-variable metric reduction.
// Option A: GPU slice → pinned buffer (avoids full D2H of Q for streaming).
// Option C: GPU reductions for MetricsSnapshot (avoids CPU scan of Q).
//
// Compiles under both g++ and nvcc — no CUDA types exposed here.
// CUDA implementation lives in src/cuda/gpu_snapshot.cu.
//
// Data flow:
//   NSSolver::advance()         sets snap.var_id/axis/norm_pos/domain_L
//   GpuGraphSolver::advance()   calls _launch_snapshot() before stream sync
//   → k_extract_slice kernel     writes floats to h_slice (host-mapped pinned)
//   → k_reduce_metrics kernel    writes GpuBlockMetrics to h_metrics (host-mapped)
//   After advance() returns:
//   LiveStreamer::gpu_snapshot()  reads h_slice + h_metas to build FrameBuffer
//   NSSolver builds MetricsSnapshot by summing h_metrics across leaves

#include <cstdint>
#include <cstdlib>  // size_t

// ── Per-leaf metadata stored on host and uploaded to device ──────────────────
// Built by GpuGraphSolver::_upload_snap_metas() from download_pairs + BlockTree.
struct SnapLeafMeta {
    const double* d_Q;  // device flat SoA pointer [NVAR * NCELL]
    float  ox, oy, oz;  // physical block origin
    float  h;           // cell size (h³ = h*h*h for volume integrals)
    int    level;       // AMR level (for FrameBuffer descriptor)
    int    _pad;        // alignment
};

// ── Per-block metric accumulator (written by GPU, read by CPU) ────────────────
struct GpuBlockMetrics {
    double mass    = 0.0;    // Σ ρ·h³ over interior cells
    double ke      = 0.0;    // Σ ½ρ|u|²·h³
    double px      = 0.0;    // Σ ρu·h³
    double py      = 0.0;    // Σ ρv·h³
    double pz      = 0.0;    // Σ ρw·h³
    double etot    = 0.0;    // Σ E·h³
    double rho_min = 1e300;
    double rho_max = 0.0;
};

// ── GpuSnapshotBuffer ─────────────────────────────────────────────────────────
struct GpuSnapshotBuffer {
    // ── Host-side output (pinned, host-mapped) ──────────────────────────────
    // h_slice[li * NB*NB + row*NB + col] : float scalar value for leaf li.
    // Non-intersecting leaves write 0.0f; LiveStreamer checks intersection
    // using h_metas[li].ox/oy/oz/h on CPU (no extra device read needed).
    float*           h_slice   = nullptr;   // [max_leaves * NB * NB]

    // h_metrics[li] : per-leaf conserved-variable integrals.
    GpuBlockMetrics* h_metrics = nullptr;   // [max_leaves]

    // ── CPU-side leaf metadata mirror ──────────────────────────────────────
    // Set by GpuGraphSolver::_upload_snap_metas() after each build().
    // Used by LiveStreamer::gpu_snapshot() to build block descriptors.
    SnapLeafMeta*    h_metas   = nullptr;   // [max_leaves]
    int              n_leaves  = 0;         // valid entries in h_metas

    // ── Current slice config — updated by NSSolver before each advance() ──
    int      var_id   = 0;    // StreamVar enum value (0=RHO … 11=SCHLIEREN)
    int      axis     = 2;    // 0=X, 1=Y, 2=Z
    float    norm_pos = 0.5f; // normalised slice position ∈ [0,1]
    float    domain_L = 1.0f; // physical domain side length

    int  max_leaves = 0;
    void* impl_    = nullptr; // SnapImpl* — opaque CUDA handles (stream, device ptrs)

    // ── 3-D volume buffer (Option B) ─────────────────────────────────────────
    // h_volume is pinned host-mapped; the GPU k_build_volume kernel writes to
    // the device-side mapping; after stream sync the CPU reads floats directly.
    // Allocation size is always VOL_MAX_N³ = 128³ = 2 M floats = 8 MB.
    float* h_volume   = nullptr; // [VOL_MAX_N^3] pinned host-mapped
    int    volume_N   = 32;      // active grid resolution, clamped to [4, 128]
    bool   vol_active = false;   // launch k_build_volume when true

    GpuSnapshotBuffer();
    ~GpuSnapshotBuffer();
    GpuSnapshotBuffer(const GpuSnapshotBuffer&)            = delete;
    GpuSnapshotBuffer& operator=(const GpuSnapshotBuffer&) = delete;

    // Allocate pinned host + device buffers for up to n_leaves_max leaves.
    // Also allocates h_metas array. Safe to call again after regrid.
    void alloc(int n_leaves_max);
};
