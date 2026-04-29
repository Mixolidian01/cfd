// gpu_graph.cu — P8.6: CUDA Graph re-capture on regrid
//
// Three RK3-update kernels (one CUDA block per leaf, 256 threads each):
//   k_save_qn   : d_Qn ← d_Q
//   k_rk3s1     : d_Q = d_Qn + (*d_dt) * d_RHS   (stage 1)
//   k_rk3s23    : d_Q = α*d_Qn + β*(d_Q + (*d_dt)*d_RHS)  (stages 2 & 3)
//
// Three per-stage CUDA sub-graphs are captured via _capture_graphs(), which
// records each stage's (ghost fill + prim_duc + rhs_conv + rhs_visc + rk3_update)
// WITHOUT any d_rhs_pool zeroing inside the graphs.
//
// In advance() replay mode, cudaMemsetAsync(d_rhs_pool, 0, ...) is issued on
// stream_ before each sub-graph launch so the zeroing is a plain stream op —
// never a captured graph node. This sidesteps the CUDA 13.x memset-in-graph
// reliability issue observed with cudaStreamCaptureModeGlobal.

#include "../../include/cuda/gpu_graph.cuh"
#include "../../include/cuda/gpu_constants.cuh"
#include "../../include/cuda/gpu_check.cuh"
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
// GpuGraphSolver
// ─────────────────────────────────────────────────────────────────────────────

GpuGraphSolver::GpuGraphSolver() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

GpuGraphSolver::~GpuGraphSolver() {
    _destroy_graphs();
    if (d_rk3_metas_) { cudaFree(d_rk3_metas_); d_rk3_metas_ = nullptr; }
    if (d_Qn_pool_)   { cudaFree(d_Qn_pool_);   d_Qn_pool_   = nullptr; }
    if (stream_)       { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

void GpuGraphSolver::_destroy_graphs() {
    if (graph_valid_) {
        cudaGraphExecDestroy(graph_s1_); graph_s1_ = nullptr;
        cudaGraphExecDestroy(graph_s2_); graph_s2_ = nullptr;
        cudaGraphExecDestroy(graph_s3_); graph_s3_ = nullptr;
        graph_valid_ = false;
    }
}

void GpuGraphSolver::build(const BlockTree& tree, int bc_type) {
    _destroy_graphs();

    ghost_list_.build(tree, bc_type);
    rhs_list_.build(tree);
    cfl_list_.build(tree);

    const auto& leaves = tree.leaf_indices();
    n_leaves_ = (int)leaves.size();
    if (n_leaves_ == 0) return;

    // Qn pool: one GPU_NVAR*GPU_NCELL double buffer per leaf
    if (d_Qn_pool_) { cudaFree(d_Qn_pool_); d_Qn_pool_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_Qn_pool_,
        (size_t)GPU_NVAR * GPU_NCELL * n_leaves_ * sizeof(double)));

    // Per-leaf RK3 metadata (host → device)
    std::vector<GpuRk3LeafMeta> h_metas(n_leaves_);
    for (int li = 0; li < n_leaves_; ++li) {
        const BlockNode& nd = tree.nodes[leaves[li]];
        h_metas[li].d_Q  = nd.block->d_Q;
        h_metas[li].d_Qn = d_Qn_pool_ + (size_t)li * GPU_NVAR * GPU_NCELL;
        h_metas[li].d_RHS = rhs_list_.d_rhs_pool + (size_t)li * GPU_NVAR * GPU_NCELL;
    }
    if (d_rk3_metas_) { cudaFree(d_rk3_metas_); d_rk3_metas_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_rk3_metas_, n_leaves_ * sizeof(GpuRk3LeafMeta)));
    CUDA_CHECK(cudaMemcpy(d_rk3_metas_, h_metas.data(),
                          n_leaves_ * sizeof(GpuRk3LeafMeta),
                          cudaMemcpyHostToDevice));
}

// One full SSP-RK3 step executed explicitly (not via graphs).
// d_rhs_pool is zeroed via cudaMemsetAsync before each stage on stream s so
// that this path is identical to the replay path (same zero+launch order).
void GpuGraphSolver::_run_rk3_explicit(cudaStream_t s) const {
    constexpr int TPB = 256;
    const double* d_dt = cfl_list_.d_dt;
    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * n_leaves_ * sizeof(double);

    k_save_qn<<<n_leaves_, TPB, 0, s>>>(d_rk3_metas_);

    // Stage 1
    CUDA_CHECK(cudaMemsetAsync(rhs_list_.d_rhs_pool, 0, rhs_bytes, s));
    ghost_list_.exec(s);
    rhs_list_.exec(s, /*zero_rhs=*/false);
    k_rk3s1<<<n_leaves_, TPB, 0, s>>>(d_rk3_metas_, d_dt);

    // Stage 2
    CUDA_CHECK(cudaMemsetAsync(rhs_list_.d_rhs_pool, 0, rhs_bytes, s));
    ghost_list_.exec(s);
    rhs_list_.exec(s, /*zero_rhs=*/false);
    k_rk3s23<<<n_leaves_, TPB, 0, s>>>(d_rk3_metas_, d_dt, 0.75, 0.25);

    // Stage 3
    CUDA_CHECK(cudaMemsetAsync(rhs_list_.d_rhs_pool, 0, rhs_bytes, s));
    ghost_list_.exec(s);
    rhs_list_.exec(s, /*zero_rhs=*/false);
    k_rk3s23<<<n_leaves_, TPB, 0, s>>>(d_rk3_metas_, d_dt, 1.0/3.0, 2.0/3.0);
}

// Capture three per-stage sub-graphs.  Each captures (ghost fill + prim_duc +
// rhs_conv + rhs_visc + rk3_update) WITHOUT zeroing d_rhs_pool so the graphs
// contain no memset nodes.  The caller zeroes d_rhs_pool on stream_ via
// cudaMemsetAsync BEFORE each cudaGraphLaunch.
void GpuGraphSolver::_capture_graphs() {
    constexpr int TPB = 256;
    const double* d_dt = cfl_list_.d_dt;

    auto capture_one = [&](cudaGraphExec_t& exec_out, auto body) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(stream_,
                                          cudaStreamCaptureModeRelaxed));
        body();
        CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
        CUDA_CHECK(cudaGraphInstantiate(&exec_out, g, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(g));
    };

    // Sub-graph 1: k_save_qn + ghost fill + RHS(no zero) + k_rk3s1
    capture_one(graph_s1_, [&]() {
        k_save_qn<<<n_leaves_, TPB, 0, stream_>>>(d_rk3_metas_);
        ghost_list_.exec(stream_);
        rhs_list_.exec(stream_, /*zero_rhs=*/false);
        k_rk3s1<<<n_leaves_, TPB, 0, stream_>>>(d_rk3_metas_, d_dt);
    });

    // Sub-graph 2: ghost fill + RHS(no zero) + k_rk3s23(0.75, 0.25)
    capture_one(graph_s2_, [&]() {
        ghost_list_.exec(stream_);
        rhs_list_.exec(stream_, /*zero_rhs=*/false);
        k_rk3s23<<<n_leaves_, TPB, 0, stream_>>>(d_rk3_metas_, d_dt, 0.75, 0.25);
    });

    // Sub-graph 3: ghost fill + RHS(no zero) + k_rk3s23(1/3, 2/3)
    capture_one(graph_s3_, [&]() {
        ghost_list_.exec(stream_);
        rhs_list_.exec(stream_, /*zero_rhs=*/false);
        k_rk3s23<<<n_leaves_, TPB, 0, stream_>>>(d_rk3_metas_, d_dt, 1.0/3.0, 2.0/3.0);
    });

    graph_valid_ = true;
}

double GpuGraphSolver::advance(const BlockTree& tree, double cfl) {
    // build() must be called before advance() and after every regrid, even
    // if the leaf count is unchanged (same-count regrid reallocates d_Q
    // pointers; a stale captured graph would dereference freed memory).
    if (n_leaves_ == 0) return 1.0e300;

    // CFL on stream_ — writes d_dt to device, syncs, returns host value
    const double dt = cfl_list_.exec(cfl, stream_);

    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * n_leaves_ * sizeof(double);

    if (!graph_valid_) {
        // First step after build: run explicit then capture sub-graphs
        _run_rk3_explicit(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        _capture_graphs();
    } else {
        // Graph replay: zero RHS on stream before each sub-graph launch
        // (memset is a plain stream op, never a captured graph node)
        CUDA_CHECK(cudaMemsetAsync(rhs_list_.d_rhs_pool, 0, rhs_bytes, stream_));
        CUDA_CHECK(cudaGraphLaunch(graph_s1_, stream_));
        CUDA_CHECK(cudaMemsetAsync(rhs_list_.d_rhs_pool, 0, rhs_bytes, stream_));
        CUDA_CHECK(cudaGraphLaunch(graph_s2_, stream_));
        CUDA_CHECK(cudaMemsetAsync(rhs_list_.d_rhs_pool, 0, rhs_bytes, stream_));
        CUDA_CHECK(cudaGraphLaunch(graph_s3_, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    return dt;
}

void GpuGraphSolver::download_q(const BlockTree& tree) const {
    static thread_local double h_buf[NVAR * NCELL];
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk || !blk->d_Q) continue;
        CUDA_CHECK(cudaMemcpy(h_buf, blk->d_Q, NVAR * NCELL * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int v = 0; v < NVAR; ++v)
            blk->Q[v].assign_from_flat(h_buf + v * NCELL);
    }
}
