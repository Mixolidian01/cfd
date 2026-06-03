#pragma once
// P8.6 / D1: CUDA Graph re-capture on regrid; GPU-native AMR.
//
// GpuGraphSolver wraps the full SSP-RK3 loop (ghost fill + WENO5-Z RHS +
// RK3 update) into three per-stage CUDA sub-graphs (s1, s2, s3) that are:
//   • Captured once (after the first explicit step) for each topology.
//   • Replayed every subsequent step with explicit cudaMemsetAsync on
//     stream_ between stages — zeroing d_rhs_pool outside the graphs so
//     the captured nodes never include a memset node (which proved
//     unreliable across repeated replays in CUDA 13.x with Global mode).
//   • Invalidated and re-captured whenever build() is called (regrid).
//
// Per-stage graph content (no RHS zeroing inside):
//   graph_s1_: k_save_qn + ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s1
//   graph_s2_: ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s23(0.75, 0.25)
//   graph_s3_: ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s23(1/3, 2/3)
//
// Replay sequence in advance():
//   cfl_list_.exec()                   (async on stream_, updates d_dt)
//   cudaMemsetAsync(d_rhs_pool, 0, stream_)
//   cudaGraphLaunch(graph_s1_, stream_)
//   cudaMemsetAsync(d_rhs_pool, 0, stream_)
//   cudaGraphLaunch(graph_s2_, stream_)
//   cudaMemsetAsync(d_rhs_pool, 0, stream_)
//   cudaGraphLaunch(graph_s3_, stream_)
//   cudaStreamSynchronize(stream_)

#include "solver/ns_solver.hpp"     // IGpuSolver interface (pure C++)
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "mpi/mpi_comm.hpp"
#include "gpu_ghost_fill.cuh"
#include "gpu_rhs.cuh"
#include "gpu_cfl.cuh"
#include "gpu_cf.cuh"
#include "gpu_sgs.cuh"
#include "gpu_mpi_halo.cuh"
#include "gpu_pool.hpp"
#include "gpu_amr.cuh"
#include "gpu_snapshot.hpp"
#include "cuda/gpu_acdi.cuh"
#include "cuda/gpu_array.cuh"
#include "cuda/gpu_ibm.cuh"
#include "cuda/gpu_wmles.cuh"
#include "models/wall_model.hpp"
#include "fsi/rigid_body.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// ── Per-leaf RK3 update metadata (see gpu_rk3_meta.cuh) ─────────────────────
#include "cuda/gpu_rk3_meta.cuh"

// ── CUDA Graph solver ─────────────────────────────────────────────────────────
struct GpuGraphSolver : IGpuSolver {
    // Component lists (rebuilt on each build())
    GpuGhostFillList ghost_list;
    GpuRhsList       rhs_list;
    GpuCflList       cfl_list;
    GpuCfList        cf_list;      // P14.4: Berger-Colella CF correction
    GpuSgsList       sgs_list;     // P-SGS-GPU: Smagorinsky operator split
    GpuMpiHaloList   mpi_halo_;    // P-MPI-GPU: D2H→MPI→H2D per RK3 stage

    // G1: GPU ACDI phi transport
    GpuAcdiList  acdi_list_;
    GpuPhiPool   phi_pool_;
    bool         acdi_enabled_ = false;
    double       acdi_ceps_    = 0.0;

    // GPU ghost-cell IBM (STL geometry, built once in build(), exec'd per RK3 stage)
    GpuIbmList   ibm_list_;
    GpuBvh*      ibm_bvh_ptr_ = nullptr;  // non-owning; caller manages BVH lifetime
    bool         ibm_enabled_ = false;

    // FSI-1: optional rigid body (non-owning; caller manages lifetime).
    // When non-null, surface forces are accumulated each RK3 stage and fed
    // to rigid_body_->step(); the IBM wall velocity is updated accordingly.
    // FSI is always run in explicit mode (no CUDA graph capture).
    RigidBody6DOF* rigid_body_ = nullptr;
    // When true, surface-force wrench is accumulated each stage but the rigid-body
    // ODE step is skipped and ibm_list_.rigid is left untouched.  Used by validation
    // tests with prescribed kinematics (e.g. t53 F3 Theodorsen pitching airfoil).
    bool           rigid_prescribed_ = false;
    // FSI-1 device buffers — RAII via GpuArray (rule 6).
    GpuArray<double>          d_wrench_{};         // [N_WRENCH]: {Fx,Fy,Fz,Tx,Ty,Tz}
    GpuArray<GpuIbmForceMeta> d_rhs_leaf_metas_{}; // [n_leaves]
    // Pinned host mirror of d_wrench_ + event to avoid a stream-wide sync in
    // _fsi_stage_update (CLAUDE.md rule 11).  Both lazy-init in build().
    double*        h_wrench_pinned_ = nullptr;     // [N_WRENCH]
    cudaEvent_t    wrench_ready_    = nullptr;

    // D7 WMLES: algebraic Reichardt wall model applied after ghost fill each stage.
    // Two lists: wmles_lo_ for low-y wall (side=0), wmles_hi_ for high-y wall (side=1).
    // Only wall-adjacent leaves (neighbours[YMINUS]==-1 or neighbours[YPLUS]==-1) are registered.
    GpuWmlesList wmles_lo_;   // bottom wall leaves (side=0)
    GpuWmlesList wmles_hi_;   // top wall leaves    (side=1)
    bool         wmles_enabled_ = false;
    double       wmles_nu_      = 0.0;
    WallModelCfg wmles_cfg_{};
    int          wmles_wall_ax_ = 1;  // axis perpendicular to wall (1=y for channel)

    // Static Smagorinsky SGS
    bool   sgs_enabled = false;
    double sgs_Cs_     = 0.16;
    double sgs_Pr_t_   = 0.9;

    // G3: dynamic Smagorinsky (Germano + Lilly)
    GpuDynSgsList dyn_sgs_list_;
    bool          dyn_sgs_enabled_ = false;
    double        dyn_sgs_Pr_t_    = 0.9;

    // Body force — set via set_body_force() before build().
    double force_x_ = 0.0;
    double force_y_ = 0.0;
    double force_z_ = 0.0;

    // Ducros sensor config — set via set_ducros() before build().
    double duc_p_thr_     = 0.1;
    double duc_blend_inv_ = 10.0;

    // MPI partition — set via set_mpi() before build().
    MpiPartition* mpi_part_ = nullptr;

    // Option A/C: GPU snapshot buffer — set via IGpuSolver::set_snapshot_buffer()
    // before or after build(). When non-null, advance() launches slice + metric
    // kernels on stream before the final sync; results are in snap_buf_->h_slice
    // and snap_buf_->h_metrics when advance() returns.
    GpuSnapshotBuffer* snap_buf_ = nullptr;

    // BC types last passed to build() — indexed by FaceDir; used by advance_imex().
    std::array<int,6> bc_types_ = {0,0,0,0,0,0};

    // Per-leaf RK3 metadata and Qn pool
    GpuRk3LeafMeta* d_rk3_metas = nullptr;
    double*          d_Qn_pool   = nullptr;
    int              n_leaves    = 0;

    // Host-side (blk, d_Q) pairs kept for download_q() — pool not needed after build().
    std::vector<std::pair<CellBlock*, double*>> download_pairs;

    // CUDA Graph state — three per-stage sub-graphs
    cudaStream_t    stream      = nullptr;
    cudaGraphExec_t graph_s1    = nullptr;
    cudaGraphExec_t graph_s2    = nullptr;
    cudaGraphExec_t graph_s3    = nullptr;
    bool            graph_valid = false;

    GpuGraphSolver();
    GpuGraphSolver(const GpuGraphSolver&) = delete;
    GpuGraphSolver& operator=(const GpuGraphSolver&) = delete;
    ~GpuGraphSolver();

    void set_gpu_sgs(double Cs, double Pr_t) override {
        sgs_Cs_ = Cs; sgs_Pr_t_ = Pr_t; sgs_enabled = true;
    }

    void set_gpu_dyn_sgs(double Pr_t) override {
        dyn_sgs_Pr_t_ = Pr_t; dyn_sgs_enabled_ = true;
    }

    void set_gpu_acdi(double ceps) override {
        acdi_enabled_ = true;
        acdi_ceps_    = ceps;
    }

    void set_gpu_ibm(GpuBvh* bvh, uint8_t bc, float uw, float vw, float ww, float Tw) override {
        ibm_enabled_      = true;
        ibm_bvh_ptr_      = bvh;
        ibm_list_.wall_bc = bc;
        ibm_list_.u_wall  = uw;
        ibm_list_.v_wall  = vw;
        ibm_list_.w_wall  = ww;
        ibm_list_.T_wall  = Tw;
    }

    // Set flat-surface resolution target for IBM curvature AMR sensor.
    // Call after set_gpu_ibm() and before build(). 0 = disabled.
    void set_ibm_surf_h(float h) { ibm_list_.h_ibm_surf = h; }

    // FSI-1: attach a rigid body for moving-wall BC + 6-DOF ODE integration.
    // Call before build().  non-null → IBM must also be enabled.
    void set_rigid_body(RigidBody6DOF* rb) { rigid_body_ = rb; }

    // FSI-1: copy the current surface-force wrench {Fx,Fy,Fz,Tx,Ty,Tz} from
    // device to host (last RK3 sub-stage value).  No-op when FSI not active.
    // Used by validation tests (e.g. t53 F3 Theodorsen Cl) that need the
    // integrated surface force per step even when no rigid body is attached.
    void read_wrench(double out[6]) const;

    // FSI-1: directly set the IBM rigid-body kinematic state for prescribed
    // motion tests.  Bypasses the 6-DOF ODE — caller controls v_cm/omega/x_cm.
    // No effect when IBM is disabled.
    void set_rigid_state(const IbmRigidState& s) { ibm_list_.update_rigid(s); }

    // FSI-1: enable prescribed-motion mode.  When true, the surface-force kernel
    // is launched each stage (so read_wrench() returns valid data) but no rigid
    // body ODE step is performed and ibm_list_.rigid is not overwritten — the
    // caller controls the kinematic state via set_rigid_state() each step.
    // Requires IBM to be enabled.  Call before build().
    void set_rigid_prescribed(bool on) { rigid_prescribed_ = on; }

    void set_body_force(double fx, double fy, double fz) override {
        force_x_ = fx; force_y_ = fy; force_z_ = fz;
    }

    // D7: enable GPU WMLES Reichardt wall model.
    // wall_ax: axis perpendicular to wall (0=x, 1=y, 2=z); nu: kinematic viscosity.
    // Only leaves at domain-boundary wall faces on that axis are registered.
    void set_gpu_wmles(int wall_ax, double nu, const WallModelCfg& cfg = {}) {
        wmles_enabled_ = true;
        wmles_wall_ax_ = wall_ax;
        wmles_nu_      = nu;
        wmles_cfg_     = cfg;
    }

    // Propagate Ducros sensor config to rhs_list for subsequent build() calls.
    void set_ducros(double p_thr, double blend_inv) override {
        duc_p_thr_ = p_thr; duc_blend_inv_ = blend_inv;
    }

    // P-MPI-GPU: wire MPI partition for subsequent build() calls.
    void set_mpi(MpiPartition* p) override { mpi_part_ = p; }

    // Option A/C: wire GPU snapshot buffer.
    // Uploads metadata to device immediately if build() was already called.
    void set_snapshot_buffer(GpuSnapshotBuffer* buf) override;
    // Re-upload snap metadata after regrid (called from build()).
    void _upload_snap_metas(const BlockTree& tree);

    // G7: MetricsBus hooks — set/step are called by NSSolver each advance().
    void set_metrics_bus(MetricsBus* b) noexcept override { metrics_bus_ = b; }
    void set_metrics_step(int s, double t) noexcept override {
        metrics_step_ = s; metrics_t_ = t;
    }
    void write_metrics(int step, double t, double dt) noexcept override;
    // Builds bus with internal rhs_list/ibm_list pointers, then wires it.
    void build_metrics(MetricsBus* bus, const SolverConfig::MetricsConfig& cfg,
                       const SnapLeafMeta* snap_metas) noexcept;

    // Rebuild all component lists from the tree; invalidates any captured graphs.
    // bc_type: 0=periodic, 1=wall, 2=open (all faces same).
    void build(const BlockTree& tree, const GpuPool& pool, int bc_type = 0) override;

    // Per-face variant: bc_types[d] gives BC for face d.
    void build_faces(const BlockTree& tree, const GpuPool& pool,
                     const std::array<int,6>& bc_types) override;

    // Per-face variant: passes FaceBCArray to ghost_list.build (which extracts p_inf from NscbcBC).
    void build_faces(const BlockTree& tree, const GpuPool& pool,
                     const FaceBCArray& face_bcs);

    // Run one SSP-RK3 step.  Returns the CFL-limited dt.
    // PRECONDITION: build() must be called before the first advance() and
    // after every regrid — even when the leaf count is unchanged (a same-count
    // regrid reallocates d_Q pointers; replaying a stale graph is UB).
    double advance(const BlockTree& tree, double cfl) override;

    // Copy device Q back to CPU CellBlocks for all leaves.
    void download_q(const BlockTree& tree) const override;

    // P11.8: Copy CPU CellBlock Q → device (reverse of download_q).
    // Called before GPU advance() when CPU path ran the previous step (AMR fallback).
    void upload_q() const override;

    // D4: IMEX-Euler — SSP-RK3 explicit step then per-leaf implicit Helmholtz
    // viscous correction via GPU GMRES.  mu: constant dynamic viscosity.
    // Defined in gpu_imex.cu (not in _GPU_NS; only linked for t32).
    double advance_imex(const BlockTree& tree, double cfl, double mu);

    // D1: GPU-native AMR regrid.
    // Evaluates the refinement sensor on GPU, updates tree topology on CPU, and
    // moves Q data exclusively via D2D GPU kernels (no large D2H memcpy for Q).
    // Returns true if the topology changed (build() must be called after).
    // cfg_max_level: max refinement level from NSSolverConfig.
    // refine_thr / coarsen_thr: normalised gradient thresholds (defaults match CPU path).
    bool gpu_regrid(BlockTree& tree, GpuPool& pool, int bc_type,
                    int cfg_max_level,
                    float refine_thr = 0.05f, float coarsen_thr = 0.01f);

    // G7: metrics bus (optional; null = disabled).
    MetricsBus* metrics_bus_  = nullptr;
    int         metrics_step_ = 0;
    double      metrics_t_    = 0.0;

private:
    void _run_rk3_explicit(cudaStream_t s, double h_dt);
    void _capture_graphs();
    void _destroy_graphs();
    // P14.4: explicit per-stage kernel sequence with Berger-Colella CF correction.
    // Used when tree has C/F interfaces (cf_list.n_coarse > 0).
    double _advance_amr(double cfl);
    // FSI-1: accumulate surface forces, step rigid body, update IBM rigid state.
    // stage_index ∈ {0,1,2} = SSP-RK3 sub-stage.  The rigid-body ODE is advanced
    // ONLY on the final stage (stage_index == 2) so the body sees one step per
    // RK3 step (not three).  Earlier stages still launch k_surface_forces_ibm
    // (so the wrench is fresh) but skip the ODE step.
    void _fsi_stage_update(cudaStream_t s, double dt_stage, int stage_index);
};
