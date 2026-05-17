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
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "cuda/gpu_amr.cuh"
#include "gpu_pool.hpp"
#include "mpi/mpi_comm.hpp"
#include "mesh/block_tree.hpp"
#include <vector>

// Verify GPU constants match CPU constants (both headers available here)
static_assert(GPU_NB   == NB,   "GPU_NB mismatch with NB in cell_block.hpp");
static_assert(GPU_NG   == NG,   "GPU_NG mismatch with NG in cell_block.hpp");
static_assert(GPU_NB2  == NB2,  "GPU_NB2 mismatch with NB2 in cell_block.hpp");
static_assert(GPU_NCELL == NCELL, "GPU_NCELL mismatch with NCELL in cell_block.hpp");
static_assert(GPU_NVAR == NVAR, "GPU_NVAR mismatch with NVAR in cell_block.hpp");

// ─────────────────────────────────────────────────────────────────────────────
// k_save_qn: d_Qn ← d_Q  (all GPU_NVAR * GPU_NCELL scalars per leaf)
// gridDim.x = n_leaves,  blockDim.x = 256
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_save_qn(const GpuRk3LeafMeta* __restrict__ metas) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    constexpr int total = GPU_NVAR * GPU_NCELL;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        m.d_Qn[i] = m.d_Q[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rk3s1: stage 1 — Q = Qn + dt * RHS
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rk3s1(const GpuRk3LeafMeta* __restrict__ metas,
             const double* __restrict__ d_dt) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    const double dt = *d_dt;
    constexpr int total = GPU_NVAR * GPU_NCELL;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        m.d_Q[i] = m.d_Qn[i] + dt * m.d_RHS[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_rk3s23: stages 2 & 3 — Q = α*Qn + β*(Q + dt*RHS)
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_rk3s23(const GpuRk3LeafMeta* __restrict__ metas,
              const double* __restrict__ d_dt,
              double alpha, double beta) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    const double dt = *d_dt;
    constexpr int total = GPU_NVAR * GPU_NCELL;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        m.d_Q[i] = alpha * m.d_Qn[i] + beta * (m.d_Q[i] + dt * m.d_RHS[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_positivity_floor: mirrors CPU apply_positivity_floor (P16.1).
// Clamps ρ≥EPS_POS and p≥EPS_POS on interior cells after each RK3 stage.
// gridDim.x = n_leaves,  blockDim.x = 256
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_positivity_floor(const GpuRk3LeafMeta* __restrict__ metas) {
    const GpuRk3LeafMeta& m = metas[blockIdx.x];
    constexpr int NB2   = GPU_NB2;
    constexpr int NCELL = GPU_NCELL;
    constexpr double EPS_POS = 1.0e-12;

    const int n_int = GPU_NB * GPU_NB * GPU_NB;  // 512 interior cells
    for (int idx = threadIdx.x; idx < n_int; idx += blockDim.x) {
        const int ii = idx % GPU_NB + GPU_NG;
        const int jj = (idx / GPU_NB) % GPU_NB + GPU_NG;
        const int kk = idx / (GPU_NB * GPU_NB) + GPU_NG;
        const int c  = ii + NB2 * (jj + NB2 * kk);

        double rho  = m.d_Q[0 * NCELL + c];
        double rhou = m.d_Q[1 * NCELL + c];
        double rhov = m.d_Q[2 * NCELL + c];
        double rhow = m.d_Q[3 * NCELL + c];
        double E    = m.d_Q[4 * NCELL + c];

        if (rho < EPS_POS) {
            m.d_Q[0 * NCELL + c] = rho = EPS_POS;
        }
        const double ke = 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
        if ((GPU_GAMMA - 1.0) * (E - ke) < EPS_POS) {
            m.d_Q[4 * NCELL + c] = ke + EPS_POS / (GPU_GAMMA - 1.0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuGraphSolver
// ─────────────────────────────────────────────────────────────────────────────

GpuGraphSolver::GpuGraphSolver() {
    CUDA_CHECK(cudaStreamCreate(&stream));
}

GpuGraphSolver::~GpuGraphSolver() {
    _destroy_graphs();
    if (d_rk3_metas) { cudaFree(d_rk3_metas); d_rk3_metas = nullptr; }
    if (d_Qn_pool)   { cudaFree(d_Qn_pool);   d_Qn_pool   = nullptr; }
    if (stream)       { cudaStreamDestroy(stream); stream = nullptr; }
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
    _destroy_graphs();

    ghost_list.build(tree, pool, bc_type, mpi_part_);
    rhs_list.build(tree, pool);
    cfl_list.build(tree, pool);
    cf_list.build(tree, pool, rhs_list.d_rhs_pool, rhs_list.d_scratch_pool);
    if (sgs_enabled) sgs_list.build(tree, pool, sgs_Cs_, sgs_Pr_t_);
    if (mpi_part_) mpi_halo_.build(tree, pool, mpi_part_);

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
}

// One full SSP-RK3 step executed explicitly (not via graphs).
// d_rhs_pool is zeroed via cudaMemsetAsync before each stage on stream s so
// that this path is identical to the replay path (same zero+launch order).
// When MPI is active, mpi_halo_.exchange() syncs the stream, downloads real
// cell planes to CPU, does MPI exchange, and uploads ghost cells back to GPU.
void GpuGraphSolver::_run_rk3_explicit(cudaStream_t s) {
    constexpr int TPB = 256;
    const double* d_dt = cfl_list.d_dt;
    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * n_leaves * sizeof(double);

    k_save_qn<<<n_leaves, TPB, 0, s>>>(d_rk3_metas);

    // Stage 1
    CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, s));
    mpi_halo_.exchange(s);   // no-op for single-rank
    ghost_list.exec(s);
    rhs_list.exec(s, /*zero_rhs=*/false);
    k_rk3s1<<<n_leaves, TPB, 0, s>>>(d_rk3_metas, d_dt);
    k_positivity_floor<<<n_leaves, TPB, 0, s>>>(d_rk3_metas);

    // Stage 2
    CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, s));
    mpi_halo_.exchange(s);
    ghost_list.exec(s);
    rhs_list.exec(s, /*zero_rhs=*/false);
    k_rk3s23<<<n_leaves, TPB, 0, s>>>(d_rk3_metas, d_dt, 0.75, 0.25);
    k_positivity_floor<<<n_leaves, TPB, 0, s>>>(d_rk3_metas);

    // Stage 3
    CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, s));
    mpi_halo_.exchange(s);
    ghost_list.exec(s);
    rhs_list.exec(s, /*zero_rhs=*/false);
    k_rk3s23<<<n_leaves, TPB, 0, s>>>(d_rk3_metas, d_dt, 1.0/3.0, 2.0/3.0);
    k_positivity_floor<<<n_leaves, TPB, 0, s>>>(d_rk3_metas);
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

    // Stage 1  — weight 1/6
    CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
    k_save_qn<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);
    mpi_halo_.exchange(stream);
    ghost_list.exec(stream);
    rhs_list.exec(stream, /*zero_rhs=*/false);
    cf_list.undo_coarse_flux(stream);
    cf_list.accum_fine_flux(stream, 1.0/6.0);
    k_rk3s1<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt);
    k_positivity_floor<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);

    // Stage 2  — weight 1/6
    CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
    mpi_halo_.exchange(stream);
    ghost_list.exec(stream);
    rhs_list.exec(stream, /*zero_rhs=*/false);
    cf_list.undo_coarse_flux(stream);
    cf_list.accum_fine_flux(stream, 1.0/6.0);
    k_rk3s23<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt, 0.75, 0.25);
    k_positivity_floor<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);

    // Stage 3  — weight 2/3
    CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
    mpi_halo_.exchange(stream);
    ghost_list.exec(stream);
    rhs_list.exec(stream, /*zero_rhs=*/false);
    cf_list.undo_coarse_flux(stream);
    cf_list.accum_fine_flux(stream, 2.0/3.0);
    k_rk3s23<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas, d_dt, 1.0/3.0, 2.0/3.0);
    k_positivity_floor<<<n_leaves, TPB, 0, stream>>>(d_rk3_metas);

    // Apply Berger-Colella correction to coarse Q (once, after all 3 stages).
    cf_list.apply_correction(stream, dt);

    // SGS operator-split: refresh ghosts with Q^{n+1} then apply Smagorinsky.
    if (sgs_enabled) {
        ghost_list.exec(stream);
        sgs_list.exec(cfl_list.d_dt, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
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
        if (sgs_enabled) {
            ghost_list.exec(stream);
            sgs_list.exec(cfl_list.d_dt, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Capture graphs only for single-rank runs (MPI cannot be captured).
        if (!mpi_halo_.active()) _capture_graphs();
    } else {
        // Graph replay: zero RHS on stream before each sub-graph launch
        // (memset is a plain stream op, never a captured graph node)
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graph_s1, stream));
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graph_s2, stream));
        CUDA_CHECK(cudaMemsetAsync(rhs_list.d_rhs_pool, 0, rhs_bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(graph_s3, stream));
        if (sgs_enabled) {
            ghost_list.exec(stream);
            sgs_list.exec(cfl_list.d_dt, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
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
