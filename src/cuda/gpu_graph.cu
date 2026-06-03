// gpu_graph.cu — P8.6: CUDA Graph re-capture on regrid
//
// Four RK3-update kernels (one CUDA block per leaf, 256 threads each):
//   k_save_qn          : d_Qn ← d_Q
//   k_rk3s1            : d_Q = d_Qn + (*d_dt) * d_RHS   (stage 1)
//   k_rk3s23           : d_Q = α*d_Qn + β*(d_Q + (*d_dt)*d_RHS)  (stages 2 & 3)
//   k_positivity_floor : clamp ρ≥EPS and p≥EPS on interior cells (P16.1)
//
// Three per-stage CUDA sub-graphs are captured via _capture_graphs(), which
// records each stage's (ghost fill + prim_duc + rhs_conv + rhs_visc + rk3_update
// + positivity_floor) WITHOUT any d_rhs_pool zeroing inside the graphs.
//
// In advance() replay mode, cudaMemsetAsync(d_rhs_pool, 0, ...) is issued on
// stream before each sub-graph launch so the zeroing is a plain stream op —
// never a captured graph node. This sidesteps the CUDA 13.x memset-in-graph
// reliability issue observed with cudaStreamCaptureModeGlobal.

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_acdi.cuh"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "cuda/gpu_amr.cuh"
#include "gpu_pool.hpp"
#include "gpu_snapshot.hpp"
#include "mpi/mpi_comm.hpp"
#include "mesh/block_tree.hpp"
#include "metrics/metrics_bus.hpp"
#include "metrics/residual_monitor.hpp"
#include "metrics/surface_monitor.hpp"
#include "metrics/probe_monitor.hpp"
#include "fsi/rigid_body.hpp"
#include <cstring>
#include <vector>

// ── Forward declarations for symbols defined in gpu_snapshot.cu ──────────────
// SnapImpl is the CUDA-private implementation of the opaque GpuSnapshotBuffer::impl_.
// Must stay in sync with the definition in gpu_snapshot.cu.
struct SnapImpl {
    cudaStream_t     stream   = nullptr;
    SnapLeafMeta*    d_metas  = nullptr;
    float*           d_slice  = nullptr;  // device-mapped pinned mirror of h_slice
    GpuBlockMetrics* d_hmet   = nullptr;  // device-mapped pinned mirror of h_metrics
    float*           d_volume = nullptr;  // device-mapped pinned mirror of h_volume
    int              max_leaves = 0;
};

// Kernels declared here; compiled and linked from gpu_snapshot.cu.
__global__ void k_extract_slice(const SnapLeafMeta* metas, float* d_out,
                                 int var_id, int axis, float slice_phys);
__global__ void k_reduce_metrics(const SnapLeafMeta* metas, GpuBlockMetrics* d_out);
__global__ void k_build_volume(const SnapLeafMeta* metas, float* d_volume,
                                int n_leaves, int N, int var_id, float domain_L);

// Verify GPU constants match CPU constants (both headers available here)
static_assert(GPU_NB   == NB,   "GPU_NB mismatch with NB in cell_block.hpp");
static_assert(GPU_NG   == NG,   "GPU_NG mismatch with NG in cell_block.hpp");
static_assert(GPU_NB2  == NB2,  "GPU_NB2 mismatch with NB2 in cell_block.hpp");
static_assert(GPU_NCELL == NCELL, "GPU_NCELL mismatch with NCELL in cell_block.hpp");
static_assert(GPU_NVAR == NVAR, "GPU_NVAR mismatch with NVAR in cell_block.hpp");

// RK3 update kernels (k_save_qn, k_rk3s1, k_rk3s23, k_positivity_floor)
#include "cuda/gpu_graph_kernels.cuh"

GpuGraphSolver::GpuGraphSolver() {
    CUDA_CHECK(cudaStreamCreate(&stream));
}

GpuGraphSolver::~GpuGraphSolver() {
    _destroy_graphs();
    if (d_rk3_metas)       { cudaFree(d_rk3_metas);       d_rk3_metas = nullptr; }
    if (d_Qn_pool)         { cudaFree(d_Qn_pool);         d_Qn_pool   = nullptr; }
    // d_wrench_ and d_rhs_leaf_metas_ free themselves via GpuArray dtor.
    if (h_wrench_pinned_)  { cudaFreeHost(h_wrench_pinned_); h_wrench_pinned_ = nullptr; }
    if (wrench_ready_)     { cudaEventDestroy(wrench_ready_); wrench_ready_ = nullptr; }
    if (stream)             { cudaStreamDestroy(stream);   stream      = nullptr; }
}

void GpuGraphSolver::_destroy_graphs() {
    if (graph_valid) {
        cudaGraphExecDestroy(graph_s1); graph_s1 = nullptr;
        cudaGraphExecDestroy(graph_s2); graph_s2 = nullptr;
        cudaGraphExecDestroy(graph_s3); graph_s3 = nullptr;
        graph_valid = false;
    }
}

void GpuGraphSolver::build(const BlockTree& tree, const GpuPool& pool, int bc_type) {
    bc_types_.fill(bc_type);
    _destroy_graphs();

    ghost_list.build(tree, pool, bc_type, mpi_part_);
    rhs_list.duc_p_thr_     = duc_p_thr_;
    rhs_list.duc_blend_inv_ = duc_blend_inv_;
    rhs_list.force_x_ = force_x_;
    rhs_list.force_y_ = force_y_;
    rhs_list.force_z_ = force_z_;
    rhs_list.build(tree, pool);
    cfl_list.build(tree, pool);
    cf_list.build(tree, pool, rhs_list.d_rhs_pool, rhs_list.d_scratch_pool);
    if (sgs_enabled)     sgs_list.build(tree, pool, sgs_Cs_, sgs_Pr_t_);
    if (dyn_sgs_enabled_) dyn_sgs_list_.build(tree, pool, dyn_sgs_Pr_t_);
    if (mpi_part_) mpi_halo_.build(tree, pool, mpi_part_);

    // G1: ACDI phi transport — alloc/upload phi pool, rebuild acdi_list_
    if (acdi_enabled_) {
        for (int idx : tree.leaf_indices()) {
            const CellBlock* blk = tree.nodes[idx].block.get();
            if (!blk) continue;
            if (!phi_pool_.d_phi(blk)) {
                phi_pool_.alloc(blk);
                phi_pool_.upload(blk);
            }
        }
        acdi_list_.build(tree, pool, phi_pool_, acdi_ceps_, bc_type);
    }

    if (ibm_enabled_ && ibm_bvh_ptr_ && ibm_bvh_ptr_->ready())
        ibm_list_.build(tree, pool, *ibm_bvh_ptr_);

    // D7: WMLES — register only wall-adjacent leaves on the specified wall axis.
    if (wmles_enabled_) {
        wmles_lo_.metas.clear();
        wmles_hi_.metas.clear();
        const int face_lo = wmles_wall_ax_ * 2;      // XMINUS/YMINUS/ZMINUS
        const int face_hi = wmles_wall_ax_ * 2 + 1;  // XPLUS/YPLUS/ZPLUS
        for (int idx : tree.leaf_indices()) {
            if (!tree.nodes[idx].has_block()) continue;
            const BlockNode& nd = tree.nodes[idx];
            double* dptr = pool.d_Q(nd.block.get());
            // Use wall-normal cell size (not isotropic h) so y_m = h_wn/2 is correct
            // for non-cubic domains (e.g. y-walls need hy, not hx).
            const double h_wn = (wmles_wall_ax_ == 0) ? nd.block->h
                              : (wmles_wall_ax_ == 1) ? nd.block->hy
                              :                         nd.block->hz;
            if (nd.neighbours[face_lo] == -1)
                wmles_lo_.add_leaf(dptr, h_wn, wmles_wall_ax_, 0);
            if (nd.neighbours[face_hi] == -1)
                wmles_hi_.add_leaf(dptr, h_wn, wmles_wall_ax_, 1);
        }
    }

    // Only process local leaves (those with an allocated block; remote MPI leaves have null).
    std::vector<int> local;
    for (int idx : tree.leaf_indices())
        if (tree.nodes[idx].has_block()) local.push_back(idx);
    n_leaves = (int)local.size();
    download_pairs.clear();

    if (n_leaves == 0) return;

    // Qn pool: one GPU_NVAR*GPU_NCELL double buffer per local leaf
    if (d_Qn_pool) { cudaFree(d_Qn_pool); d_Qn_pool = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_Qn_pool,
        (size_t)GPU_NVAR * GPU_NCELL * n_leaves * sizeof(double)));

    // Per-leaf RK3 metadata (host → device)
    std::vector<GpuRk3LeafMeta> h_metas(n_leaves);
    download_pairs.reserve(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        const BlockNode& nd = tree.nodes[local[li]];
        double* dptr = pool.d_Q(nd.block.get());
        h_metas[li].d_Q  = dptr;
        h_metas[li].d_Qn = d_Qn_pool + (size_t)li * GPU_NVAR * GPU_NCELL;
        h_metas[li].d_RHS = rhs_list.d_rhs_pool + (size_t)li * GPU_NVAR * GPU_NCELL;
        download_pairs.emplace_back(nd.block.get(), dptr);
    }
    gpu_upload_meta(d_rk3_metas, h_metas);

    // FSI-1: allocate wrench buffer and leaf-RHS meta array for surface forces.
    // Allocated when either a rigid body is attached (full FSI coupling) or
    // prescribed-motion mode is on (force-extraction only, e.g. t53 F3).
    if ((rigid_body_ || rigid_prescribed_) && ibm_enabled_) {
        d_wrench_.alloc(N_WRENCH);
        // Lazy-init pinned host mirror + event (re-used across builds).
        if (!h_wrench_pinned_)
            CUDA_CHECK(cudaMallocHost(&h_wrench_pinned_, N_WRENCH * sizeof(double)));
        if (!wrench_ready_)
            CUDA_CHECK(cudaEventCreateWithFlags(&wrench_ready_, cudaEventDisableTiming));

        // Build host array pointing into ibm_list_ pools and rhs_list scratch.
        std::vector<GpuIbmForceMeta> h_rlm(n_leaves);
        for (int li = 0; li < n_leaves; ++li) {
            const BlockNode& nd = tree.nodes[local[li]];
            h_rlm[li].d_scratch   = rhs_list.d_scratch_pool + (size_t)li * SCRATCH_NCOMP * GPU_NCELL;
            h_rlm[li].d_cell_type = ibm_list_.d_cell_type_pool + (size_t)li * GPU_NCELL;
            h_rlm[li].d_sdf       = ibm_list_.d_sdf_pool + (size_t)li * GPU_NCELL;
            h_rlm[li].d_wnx       = ibm_list_.d_wnorm_pool + (size_t)(li * GPU_NCELL * 3 + 0 * GPU_NCELL);
            h_rlm[li].d_wny       = ibm_list_.d_wnorm_pool + (size_t)(li * GPU_NCELL * 3 + 1 * GPU_NCELL);
            h_rlm[li].d_wnz       = ibm_list_.d_wnorm_pool + (size_t)(li * GPU_NCELL * 3 + 2 * GPU_NCELL);
            h_rlm[li].ox = (float)nd.block->ox;
            h_rlm[li].oy = (float)nd.block->oy;
            h_rlm[li].oz = (float)nd.block->oz;
            h_rlm[li].hx = (float)nd.block->h;
            h_rlm[li].hy = (float)nd.block->hy;
            h_rlm[li].hz = (float)nd.block->hz;
        }
        d_rhs_leaf_metas_.upload(h_rlm);
    }

    // Option A/C: upload snapshot leaf metadata whenever topology changes.
    if (snap_buf_) _upload_snap_metas(tree);
}

// ── Option A/C: snapshot metadata upload ────────────────────────────────────
// Assembles SnapLeafMeta from download_pairs + tree, uploads to device,
// and mirrors to snap_buf_->h_metas for CPU use in LiveStreamer::gpu_snapshot().
void GpuGraphSolver::_upload_snap_metas(const BlockTree& tree)
{
    if (!snap_buf_ || n_leaves == 0) return;
    if (n_leaves > snap_buf_->max_leaves) {
        snap_buf_->alloc(n_leaves);  // grow buffer if needed
    }

    auto* impl = static_cast<SnapImpl*>(snap_buf_->impl_);
    if (!impl) return;

    std::vector<SnapLeafMeta> host_metas(n_leaves);
    const auto& leaves = tree.leaf_indices();
    int li = 0;
    for (int leaf_idx : leaves) {
        if (li >= n_leaves) break;
        const BlockNode& nd = tree.nodes[leaf_idx];
        if (!nd.has_block()) continue;
        SnapLeafMeta& m = host_metas[li];
        m.d_Q   = download_pairs[li].second;
        m.ox    = static_cast<float>(nd.ox);
        m.oy    = static_cast<float>(nd.oy);
        m.oz    = static_cast<float>(nd.oz);
        m.h     = static_cast<float>(nd.block->h);
        m.hy    = static_cast<float>(nd.block->hy);
        m.hz    = static_cast<float>(nd.block->hz);
        m.level = nd.level;
        // Copy to CPU mirror
        snap_buf_->h_metas[li] = m;
        ++li;
    }
    snap_buf_->n_leaves = li;

    CUDA_CHECK(cudaMemcpyAsync(impl->d_metas, host_metas.data(),
                               (size_t)li * sizeof(SnapLeafMeta),
                               cudaMemcpyHostToDevice, stream));
}

// ── Option A/C: launch snapshot kernels on main stream ───────────────────────
// Called just before cudaStreamSynchronize in advance() / _advance_amr().
// k_extract_slice and k_reduce_metrics both write to host-mapped pinned memory
// (impl->d_slice / impl->d_hmet), so results are visible in h_slice / h_metrics
// after the stream sync that follows this call.
static void _do_launch_snapshot(GpuSnapshotBuffer* snap_buf, int n_leaves,
                                 cudaStream_t s)
{
    if (!snap_buf || n_leaves == 0) return;
    auto* impl = static_cast<SnapImpl*>(snap_buf->impl_);
    if (!impl || !impl->d_metas) return;

    const float slice_phys = snap_buf->norm_pos * snap_buf->domain_L;

    k_extract_slice<<<n_leaves, GPU_NB * GPU_NB, 0, s>>>(
        impl->d_metas, impl->d_slice,
        snap_buf->var_id, snap_buf->axis, slice_phys);

    k_reduce_metrics<<<n_leaves, 64, 0, s>>>(
        impl->d_metas, impl->d_hmet);

    // Option B: 3-D volume — only when a viewer client has connected /volume-stream.
    if (snap_buf->vol_active && impl->d_volume) {
        const int N = max(4, min(128, snap_buf->volume_N));
        CUDA_CHECK(cudaMemsetAsync(impl->d_volume, 0,
                                   (size_t)N * N * N * sizeof(float), s));
        k_build_volume<<<n_leaves, 64, 0, s>>>(
            impl->d_metas, impl->d_volume,
            n_leaves, N, snap_buf->var_id, snap_buf->domain_L);
    }
}

void GpuGraphSolver::build_faces(const BlockTree& tree, const GpuPool& pool,
                                   const std::array<int,6>& bc_types) {
    // Delegate all setup to build() (RK3 metas, Qn pool, rhs/cfl/cf lists, etc.),
    // then override ghost_list with the per-face BC types.
    build(tree, pool, bc_types[0]);
    ghost_list.build(tree, pool, bc_types, mpi_part_);
    bc_types_ = bc_types;
}

void GpuGraphSolver::build_faces(const BlockTree& tree, const GpuPool& pool,
                                   const FaceBCArray& face_bcs) {
    std::array<int,6> bc_types{};
    for (int d = 0; d < NFACES; ++d) bc_types[d] = bc_to_int(face_bcs[d]);
    build(tree, pool, bc_types[0]);
    ghost_list.build(tree, pool, face_bcs, mpi_part_);
    bc_types_ = bc_types;
}

void GpuGraphSolver::set_snapshot_buffer(GpuSnapshotBuffer* buf)
{
    snap_buf_ = buf;
    // Metadata upload happens in build() when snap_buf_ is non-null.
    // Contract: call set_snapshot_buffer() before build() (or call build()
    // again after) to ensure device metadata is current.
}

// One full SSP-RK3 step executed explicitly (not via graphs).
// d_rhs_pool is zeroed via cudaMemsetAsync before each stage on stream s so
// that this path is identical to the replay path (same zero+launch order).
// When MPI is active, mpi_halo_.exchange() syncs the stream, downloads real
// cell planes to CPU, does MPI exchange, and uploads ghost cells back to GPU.

// FSI-1: download the per-stage surface-force wrench {Fx,Fy,Fz,Tx,Ty,Tz}.
// Reflects the value from the last RK3 sub-stage in the most recent advance().
// When no rigid body / prescribed motion is active, d_wrench_ is empty and
// out[] is zero-filled.
void GpuGraphSolver::read_wrench(double out[N_WRENCH]) const {
    for (int i = 0; i < N_WRENCH; ++i) out[i] = 0.0;
    if (!d_wrench_) return;
    CUDA_CHECK(cudaMemcpy(out, d_wrench_.get(), N_WRENCH * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

// FSI-1: accumulate surface forces, optionally step rigid body, update IBM rigid state.
// The rigid-body ODE is advanced ONLY on the final SSP-RK3 sub-stage (stage_idx==2)
// using the full step dt (= 3 * dt_stage); earlier stages still launch
// k_surface_forces_ibm so the wrench is always fresh, but skip the ODE advance.
// Uses pinned host buffer + event sync to avoid cudaStreamSynchronize on the
// advance stream (CLAUDE.md rule 11).
void GpuGraphSolver::_fsi_stage_update(cudaStream_t s, double dt_stage, int stage_idx) {
    if (!ibm_enabled_ || !d_wrench_ || !d_rhs_leaf_metas_) return;
    if (!rigid_body_ && !rigid_prescribed_) return;

    constexpr int TPB = 256;
    // Pivot for torque integration: use rigid body CG when available, else the
    // currently-prescribed IBM rigid x_cm (set by the caller each step).
    double xp = rigid_body_ ? rigid_body_->x[0] : ibm_list_.rigid.x_cm[0];
    double yp = rigid_body_ ? rigid_body_->x[1] : ibm_list_.rigid.x_cm[1];
    double zp = rigid_body_ ? rigid_body_->x[2] : ibm_list_.rigid.x_cm[2];

    // Zero wrench, launch accumulation, async copy to pinned host, event-wait.
    CUDA_CHECK(cudaMemsetAsync(d_wrench_.get(), 0, N_WRENCH * sizeof(double), s));
    k_surface_forces_ibm<<<n_leaves, TPB, 0, s>>>(
        d_rhs_leaf_metas_.get(), n_leaves, xp, yp, zp, d_wrench_.get());
    CUDA_CHECK(cudaMemcpyAsync(h_wrench_pinned_, d_wrench_.get(),
                               N_WRENCH * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaEventRecord(wrench_ready_, s));
    CUDA_CHECK(cudaEventSynchronize(wrench_ready_));

    // Prescribed-motion mode: caller drives kinematics each step.
    if (rigid_prescribed_) return;

    // Advance the 6-DOF ODE only on the last RK3 sub-stage with the FULL step dt.
    // SSP-RK3 sub-stage dt equals the step dt (Shu-Osher weighted), so the full
    // step covers three sub-stages → step the body once per RK3 step.
    if (stage_idx == 2) {
        rigid_body_->step(h_wrench_pinned_, dt_stage);
        IbmRigidState rs;
        for (int i = 0; i < 3; ++i) rs.v_cm[i]  = rigid_body_->v[i];
        for (int i = 0; i < 3; ++i) rs.omega[i] = rigid_body_->w[i];
        for (int i = 0; i < 3; ++i) rs.x_cm[i]  = rigid_body_->x[i];
        ibm_list_.update_rigid(rs);
    }
    // NOTE: BVH geometry rebuild (moving the surface) is deferred to the caller's
    // advance loop if the geometry itself moves (e.g. per-step regrid). The
    // ghost-cell wall velocities are updated via k_apply_moving_wall inside
    // ibm_list_.exec() on the next stage.
}

void GpuGraphSolver::_run_rk3_explicit(cudaStream_t s) {
    constexpr int TPB = 256;
    const double* d_dt = cfl_list.d_dt;
    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * n_leaves * sizeof(double);

    // FSI-1: fetch host dt for the rigid-body ODE step (advanced once per RK3 step
    // on stage_idx==2 inside _fsi_stage_update).  cfl_list.exec already wrote d_dt.
    double h_dt = 0.0;
    if ((rigid_body_ || rigid_prescribed_) && ibm_enabled_)
        CUDA_CHECK(cudaMemcpy(&h_dt, cfl_list.d_dt, sizeof(double), cudaMemcpyDeviceToHost));

    if (acdi_enabled_) acdi_list_.save_phin(s);
    k_save_qn<<<n_leaves, TPB, 0, s>>>(d_rk3_metas);

    auto stage = [&](int stage_idx, double a, double b) {
        const bool s1 = (stage_idx == 0);
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, s));
        mpi_halo_.exchange(s);
        ghost_list.exec(s);
        if (wmles_enabled_) {
            wmles_lo_.exec_apply(wmles_nu_, wmles_cfg_, s);
            wmles_hi_.exec_apply(wmles_nu_, wmles_cfg_, s);
        }
        if (ibm_enabled_) ibm_list_.exec(s);
        rhs_list.exec(s, false);
        // FSI-1: accumulate surface forces (every stage); step rigid body only on stage 2.
        if ((rigid_body_ || rigid_prescribed_) && ibm_enabled_)
            _fsi_stage_update(s, h_dt, stage_idx);
        if (s1) k_rk3s1 <<<n_leaves, TPB, 0, s>>>(d_rk3_metas, d_dt);
        else    k_rk3s23<<<n_leaves, TPB, 0, s>>>(d_rk3_metas, d_dt, a, b);
        k_positivity_floor<<<n_leaves, TPB, 0, s>>>(d_rk3_metas);
        if (acdi_enabled_) {
            acdi_list_.zero_rhs(s);
            acdi_list_.fill_ghosts(s);
            acdi_list_.rhs_advect(s);
            acdi_list_.rhs_compress(s);
            acdi_list_.update_phi(d_dt, a, s1, s);
        }
    };
    stage(0, 0.0,     0.0   );
    stage(1, 0.75,    0.25  );
    stage(2, 1.0/3.0, 2.0/3.0);
}

// Capture three per-stage sub-graphs.  Each captures (ghost fill + prim_duc +
// rhs_conv + rhs_visc + rk3_update) WITHOUT zeroing d_rhs_pool so the graphs
// contain no memset nodes.  The caller zeroes d_rhs_pool on stream via
// cudaMemsetAsync BEFORE each cudaGraphLaunch.
void GpuGraphSolver::_capture_graphs() {
    constexpr int TPB = 256;
    const double* d_dt = cfl_list.d_dt;

    auto capture_one = [&](cudaGraphExec_t& exec_out, auto body) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(stream,
                                          cudaStreamCaptureModeRelaxed));
        body();
        CUDA_CHECK(cudaStreamEndCapture(stream, &g));
        CUDA_CHECK(cudaGraphInstantiate(&exec_out, g, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(g));
    };

    // Sub-graph 1: k_save_qn + ghost fill + RHS(no zero) + k_rk3s1 + floor
    capture_one(graph_s1, [&]() {
        k_save_qn<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);
        ghost_list.exec(stream);
        rhs_list.exec(stream, /*zero_rhs=*/false);
        k_rk3s1<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt);
        k_positivity_floor<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);
    });

    // Sub-graph 2: ghost fill + RHS(no zero) + k_rk3s23(0.75, 0.25) + floor
    capture_one(graph_s2, [&]() {
        ghost_list.exec(stream);
        rhs_list.exec(stream, /*zero_rhs=*/false);
        k_rk3s23<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt, 0.75, 0.25);
        k_positivity_floor<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);
    });

    // Sub-graph 3: ghost fill + RHS(no zero) + k_rk3s23(1/3, 2/3) + floor
    capture_one(graph_s3, [&]() {
        ghost_list.exec(stream);
        rhs_list.exec(stream, /*zero_rhs=*/false);
        k_rk3s23<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt, 1.0/3.0, 2.0/3.0);
        k_positivity_floor<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);
    });

    graph_valid = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// _advance_amr: P14.4 — explicit kernel sequence with Berger-Colella correction.
// Called when the tree has C/F interfaces (cf_list.n_coarse > 0).
// Does NOT use CUDA graphs (CF correction requires per-stage atomics into d_reg).
// ─────────────────────────────────────────────────────────────────────────────
double GpuGraphSolver::_advance_amr(double cfl) {
    // Invalidate any captured flat-tree graphs — topology changed.
    _destroy_graphs();

    constexpr int TPB = 256;
    const double dt_local = cfl_list.exec(cfl, stream);
    const double dt       = mpi_allreduce_min(dt_local, mpi_part_);
    if (mpi_halo_.active()) {
        CUDA_CHECK(cudaMemcpy(cfl_list.d_dt, &dt,
                              sizeof(double), cudaMemcpyHostToDevice));
    }
    const double* d_dt = cfl_list.d_dt;
    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * n_leaves * sizeof(double);

    // Zero flux registers once before stage 1 (accumulate across all 3 stages).
    cf_list.zero_regs(stream);

    if (acdi_enabled_) acdi_list_.save_phin(stream);

    auto stage = [&](int stage_idx, double cf_wt, double a, double b) {
        const bool save_qn = (stage_idx == 0);
        const bool s1      = (stage_idx == 0);
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
        if (save_qn) k_save_qn<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);
        mpi_halo_.exchange(stream);
        ghost_list.exec(stream);
        if (wmles_enabled_) {
            wmles_lo_.exec_apply(wmles_nu_, wmles_cfg_, stream);
            wmles_hi_.exec_apply(wmles_nu_, wmles_cfg_, stream);
        }
        if (ibm_enabled_) ibm_list_.exec(stream);
        rhs_list.exec(stream, false);
        // FSI-1: accumulate surface forces (every stage); step rigid body only on stage 2.
        if ((rigid_body_ || rigid_prescribed_) && ibm_enabled_)
            _fsi_stage_update(stream, dt, stage_idx);
        cf_list.undo_coarse_flux(stream);
        cf_list.accum_fine_flux(stream, cf_wt);
        if (s1) k_rk3s1 <<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt);
        else    k_rk3s23<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt, a, b);
        k_positivity_floor<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);
        if (acdi_enabled_) {
            acdi_list_.zero_rhs(stream);
            acdi_list_.fill_ghosts(stream);
            acdi_list_.rhs_advect(stream);
            acdi_list_.rhs_compress(stream);
            acdi_list_.update_phi(d_dt, a, s1, stream);
        }
    };
    stage(0, 1.0/6.0, 0.0,     0.0   );   // Stage 1 — weight 1/6
    stage(1, 1.0/6.0, 0.75,    0.25  );   // Stage 2 — weight 1/6
    stage(2, 2.0/3.0, 1.0/3.0, 2.0/3.0); // Stage 3 — weight 2/3

    // Apply Berger-Colella correction to coarse Q (once, after all 3 stages).
    cf_list.apply_correction(stream, dt);

    // SGS operator-split: refresh ghosts with Q^{n+1}, then apply SGS model.
    if (sgs_enabled || dyn_sgs_enabled_) {
        ghost_list.exec(stream);
        if (ibm_enabled_) ibm_list_.exec(stream);
        if (sgs_enabled)      sgs_list.exec(cfl_list.d_dt, stream);
        if (dyn_sgs_enabled_) dyn_sgs_list_.exec(cfl_list.d_dt, stream);
    }

    // Option A/C: GPU slice + metric kernels (before final sync so they're covered).
    if (metrics_bus_)
        metrics_bus_->launch(rhs_list, snap_buf_ ? snap_buf_->h_metas : nullptr,
                             n_leaves, metrics_step_, stream);
    _do_launch_snapshot(snap_buf_, n_leaves, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (metrics_bus_) metrics_bus_->collect(metrics_step_, metrics_t_, dt);

    // G1: copy device phi back to CPU CellBlocks.
    if (acdi_enabled_) {
        for (auto& [blk, _] : download_pairs)
            if (blk && phi_pool_.d_phi(blk)) phi_pool_.download(blk);
    }

    return dt;
}

double GpuGraphSolver::advance(const BlockTree& tree, double cfl) {
    // build() must be called before advance() and after every regrid, even
    // if the leaf count is unchanged (same-count regrid reallocates d_Q
    // pointers; a stale captured graph would dereference freed memory).
    if (n_leaves == 0) return 1.0e300;

    // P14.4: AMR path — explicit kernel sequence with Berger-Colella correction.
    if (cf_list.n_coarse > 0)
        return _advance_amr(cfl);

    // CFL on stream — writes d_dt to device, syncs, returns host value.
    // P-MPI-GPU: allreduce across ranks so every rank uses the same dt,
    // then write the global value back to d_dt for the RK3 update kernels.
    const double dt_local = cfl_list.exec(cfl, stream);
    const double dt       = mpi_allreduce_min(dt_local, mpi_part_);
    if (mpi_halo_.active()) {
        CUDA_CHECK(cudaMemcpy(cfl_list.d_dt, &dt,
                              sizeof(double), cudaMemcpyHostToDevice));
    }

    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * n_leaves * sizeof(double);

    // P-MPI-GPU: when MPI is active, CUDA graphs cannot be used (they cannot
    // capture CPU MPI calls).  Always run the explicit kernel sequence.
    const bool use_explicit = !graph_valid || mpi_halo_.active();

    if (use_explicit) {
        _run_rk3_explicit(stream);
        if (sgs_enabled || dyn_sgs_enabled_) {
            ghost_list.exec(stream);
            if (ibm_enabled_) ibm_list_.exec(stream);
            if (sgs_enabled)      sgs_list.exec(cfl_list.d_dt, stream);
            if (dyn_sgs_enabled_) dyn_sgs_list_.exec(cfl_list.d_dt, stream);
        }
        // Option A/C: GPU slice + metric kernels (before sync so they're covered).
        if (metrics_bus_)
            metrics_bus_->launch(rhs_list, snap_buf_ ? snap_buf_->h_metas : nullptr,
                                 n_leaves, metrics_step_, stream);
        _do_launch_snapshot(snap_buf_, n_leaves, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (metrics_bus_) metrics_bus_->collect(metrics_step_, metrics_t_, dt);
        // Capture graphs only for single-rank runs without ACDI or dynamic SGS
        // (both use dynamic state that cannot be captured in a static graph).
        // IBM and WMLES use dynamic pointers rebuilt on regrid — skip graph capture
        if (!mpi_halo_.active() && !acdi_enabled_ && !dyn_sgs_enabled_
            && !ibm_enabled_ && !wmles_enabled_) _capture_graphs();
    } else {
        // Graph replay: zero RHS on stream before each sub-graph launch
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graph_s1, stream));
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graph_s2, stream));
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graph_s3, stream));
        if (sgs_enabled || dyn_sgs_enabled_) {
            ghost_list.exec(stream);
            if (ibm_enabled_) ibm_list_.exec(stream);
            if (sgs_enabled)      sgs_list.exec(cfl_list.d_dt, stream);
            if (dyn_sgs_enabled_) dyn_sgs_list_.exec(cfl_list.d_dt, stream);
        }
        // Option A/C: GPU slice + metric kernels (before sync so they're covered).
        if (metrics_bus_)
            metrics_bus_->launch(rhs_list, snap_buf_ ? snap_buf_->h_metas : nullptr,
                                 n_leaves, metrics_step_, stream);
        _do_launch_snapshot(snap_buf_, n_leaves, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (metrics_bus_) metrics_bus_->collect(metrics_step_, metrics_t_, dt);
    }

    // G1: copy device phi back to CPU CellBlocks so host tests can read phi.
    if (acdi_enabled_) {
        for (const auto& li : tree.leaf_indices()) {
            CellBlock* blk = tree.nodes[li].block.get();
            if (blk && phi_pool_.d_phi(blk)) phi_pool_.download(blk);
        }
    }

    return dt;
}

void GpuGraphSolver::download_q(const BlockTree& /*tree*/) const {
    static thread_local double h_buf[NVAR * NCELL];
    for (const auto& [blk, dptr] : download_pairs) {
        if (!blk || !dptr) continue;
        CUDA_CHECK(cudaMemcpy(h_buf, dptr, NVAR * NCELL * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int v = 0; v < NVAR; ++v)
            blk->Q[v].assign_from_flat(h_buf + v * NCELL);
    }
}

// D1: GPU-native AMR regrid.
// ─────────────────────────────────────────────────────────────────────────────
bool GpuGraphSolver::gpu_regrid(BlockTree& tree, GpuPool& pool, int bc_type,
                                int cfg_max_level,
                                float refine_thr, float coarsen_thr)
{
    const auto& leaves = tree.leaf_indices();
    const int n = (int)leaves.size();
    if (n == 0) return false;

    // ── Step 1: evaluate refinement sensor on GPU ─────────────────────────────
    // Collect device Q pointers and cell sizes for all leaves.
    std::vector<const double*> d_Q_ptrs(n);
    std::vector<float>          h_vals(n);
    for (int i = 0; i < n; ++i) {
        const BlockNode& nd = tree.nodes[leaves[i]];
        d_Q_ptrs[i] = pool.d_Q(nd.block.get());
        h_vals[i]   = (float)nd.block->h;
    }

    float* d_sensor = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sensor, n * sizeof(float)));

    gpu_eval_refine_sensor(d_Q_ptrs.data(), h_vals.data(), n, d_sensor, stream);

    // Download sensor values (tiny: n floats, not Q arrays).
    std::vector<float> h_sensor(n);
    CUDA_CHECK(cudaMemcpy(h_sensor.data(), d_sensor, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sensor));

    // ── Step 2: decide which leaves to refine / coarsen ──────────────────────
    const float coarsen_grad = coarsen_thr * 0.5f;  // matches CPU should_coarsen

    std::vector<int> to_refine, to_coarsen;
    for (int i = 0; i < n; ++i) {
        const BlockNode& nd = tree.nodes[leaves[i]];
        if (nd.level < cfg_max_level && h_sensor[i] > refine_thr)
            to_refine.push_back(leaves[i]);
    }

    // Coarsen: 8-sibling groups all below threshold.
    // Build set of refine candidates for fast lookup.
    for (int i = 0; i < n; ++i) {
        int p = tree.nodes[leaves[i]].parent;
        if (p < 0) continue;
        int fc = tree.nodes[p].first_child;
        if (fc < 0) continue;
        bool all_leaf   = true;
        bool all_coarse = true;
        for (int oct = 0; oct < 8; ++oct) {
            int ci = fc + oct;
            if (!tree.nodes[ci].is_leaf()) { all_leaf = false; break; }
            // Use the sensor value: coarsen only if all siblings are smooth.
            // Find this sibling's index in our leaves list.
            bool found = false;
            for (int j = 0; j < n; ++j) {
                if (leaves[j] == ci) {
                    if (h_sensor[j] > coarsen_grad) all_coarse = false;
                    found = true;
                    break;
                }
            }
            if (!found) { all_coarse = false; }
        }
        if (all_leaf && all_coarse) {
            bool dup = false;
            for (int pc : to_coarsen) if (pc == p) { dup = true; break; }
            if (!dup) to_coarsen.push_back(p);
        }
    }

    if (to_refine.empty() && to_coarsen.empty()) return false;

    // ── Step 3: wire GPU AMR callbacks and update tree topology ──────────────
    // on_gpu_prolong_: alloc 8 child GPU buffers, D2D k_prolong, free parent GPU.
    tree.set_gpu_amr_callbacks(
        [&](CellBlock* parent, CellBlock* const children[8]) {
            double* d_parent = pool.d_Q(parent);
            std::vector<GpuProlongMeta> ops(8);
            for (int oct = 0; oct < 8; ++oct) {
                pool.alloc(children[oct]);
                ops[oct].d_coarse_Q = d_parent;
                ops[oct].d_fine_Q   = pool.d_Q(children[oct]);
                ops[oct].oct        = oct;
                ops[oct]._pad       = 0;
            }
            GpuAmrList amr;
            amr.build_prolong(ops);
            amr.exec_prolong(stream);
            // Now safe to free parent GPU buffer (kernel is queued on stream).
            CUDA_CHECK(cudaStreamSynchronize(stream));
            pool.free(parent);
        },
        [&](CellBlock* parent, CellBlock* const children[8]) {
            pool.alloc(parent);
            GpuRestrictMeta meta;
            meta.d_coarse_Q = pool.d_Q(parent);
            for (int oct = 0; oct < 8; ++oct)
                meta.d_children_Q[oct] = pool.d_Q(children[oct]);
            GpuAmrList amr;
            amr.build_restrict({meta});
            amr.exec_restrict(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // Free children GPU buffers after restriction completes.
            for (int oct = 0; oct < 8; ++oct)
                pool.free(children[oct]);
        }
    );

    // Refine pass — each refine() calls rebuild_neighbours internally.
    for (int li : to_refine) {
        if (!tree.nodes[li].is_leaf()) continue;
        tree.refine(li);
    }

    // Coarsen pass — sensors collected from pre-refine leaf list (correct).
    for (int p : to_coarsen) {
        int fc = tree.nodes[p].first_child;
        if (fc < 0) continue;
        bool all_leaf = true;
        for (int oct = 0; oct < 8; ++oct)
            if (!tree.nodes[fc + oct].is_leaf()) { all_leaf = false; break; }
        if (!all_leaf) continue;
        tree.coarsen(p);
    }

    // Enforce 2:1 balance — may trigger additional GPU-native refines (callbacks active).
    tree.balance();

    // Clear GPU AMR callbacks after ALL topology changes (refine + coarsen + balance).
    tree.set_gpu_amr_callbacks(nullptr, nullptr);

    // Rebuild neighbour pointers after all changes.
    // CPU Q ghost cells are intentionally stale (GPU is authoritative).
    // GPU ghost fill will run at the start of the next advance().
    tree.rebuild_neighbours();

    // ── Step 4: rebuild GPU lists with new topology ───────────────────────────
    build(tree, pool, bc_type);

    return true;
}

// P11.8: CPU → GPU re-upload (reverse of download_q).
// Called when the previous step used the CPU AMR path (gpu_q_stale_).
void GpuGraphSolver::upload_q() const {
    static thread_local double h_buf[NVAR * NCELL];
    for (const auto& [blk, dptr] : download_pairs) {
        if (!blk || !dptr) continue;
        for (int v = 0; v < NVAR; ++v)
            blk->Q[v].copy_to_flat(h_buf + v * NCELL);
        CUDA_CHECK(cudaMemcpy(dptr, h_buf, NVAR * NCELL * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
}

// G7: build MetricsBus using internal rhs_list/ibm_list, then wire it.
void GpuGraphSolver::build_metrics(MetricsBus* bus, const SolverConfig::MetricsConfig& cfg,
                                    const SnapLeafMeta* snap_metas) noexcept {
    if (!bus) return;
    bus->build(cfg, n_leaves, &rhs_list, snap_metas,
               ibm_enabled_ ? &ibm_list_ : nullptr);
    metrics_bus_  = bus;
    metrics_step_ = 0;
    metrics_t_    = 0.0;
}

void GpuGraphSolver::write_metrics(int step, double t, double dt) noexcept {
    if (metrics_bus_) metrics_bus_->write(step, t, dt);
}
