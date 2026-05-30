// gpu_lts.cu — G5: GPU Berger-Oliger LTS integrator.
//
// Implements the Berger-Oliger scheme:
//   r fine sub-steps (dt_f = dt_c/r) → 1 coarse step (dt_c) → BC flux correction.
//
// Ghost fill and RHS run over ALL leaves each stage; only the RK3 update
// kernels are level-filtered via separate fine/coarse GpuRk3LeafMeta arrays.
//
// Berger-Colella correction protocol:
//   zero_regs() once before sub-steps
//   Fine sub-steps: accum_fine_flux(w) at each stage (NO undo_coarse_flux)
//   Coarse step:    undo_coarse_flux + NO accum (reg already has fine flux)
//   apply_correction(dt_c) once after coarse step

#include "cuda/gpu_lts.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include <algorithm>
#include <vector>
#include <map>

// Forward declarations of kernels defined in gpu_graph.cu
// (same nvcc invocation — device-linked at link time).
__global__ void k_save_qn(const GpuRk3LeafMeta* __restrict__ metas);
__global__ void k_rk3s1  (const GpuRk3LeafMeta* __restrict__ metas,
                           const double* __restrict__ d_dt);
__global__ void k_rk3s23 (const GpuRk3LeafMeta* __restrict__ metas,
                           const double* __restrict__ d_dt,
                           double alpha, double beta);
__global__ void k_positivity_floor(const GpuRk3LeafMeta* __restrict__ metas);

// ─────────────────────────────────────────────────────────────────────────────
GpuLtsIntegrator::GpuLtsIntegrator() {
    CUDA_CHECK(cudaStreamCreate(&stream));
}

GpuLtsIntegrator::~GpuLtsIntegrator() {
    if (d_fine_metas)   { cudaFree(d_fine_metas);   d_fine_metas   = nullptr; }
    if (d_coarse_metas) { cudaFree(d_coarse_metas); d_coarse_metas = nullptr; }
    if (d_Qn_fine)      { cudaFree(d_Qn_fine);      d_Qn_fine      = nullptr; }
    if (d_Qn_coarse)    { cudaFree(d_Qn_coarse);    d_Qn_coarse    = nullptr; }
    if (stream)         { cudaStreamDestroy(stream); stream = nullptr; }
}

void GpuLtsIntegrator::build(const BlockTree& tree, const GpuPool& pool,
                              int bc_type, int r)
{
    lts_r = r;

    const int L_max = tree.max_leaf_level();
    const int L_min = tree.min_leaf_level();

    // ── Shared lists (all leaves) ─────────────────────────────────────────────
    rhs_all.build(tree, pool);
    ghost_all.build(tree, pool, bc_type);
    cf.build(tree, pool, rhs_all.d_rhs_pool, rhs_all.d_scratch_pool);

    // ── Per-level CFL ──────────────────────────────────────────────────────────
    cfl_fine.build(tree, pool, L_max);
    cfl_coarse.build(tree, pool, L_min);

    // ── Level-filtered RK3 metadata ───────────────────────────────────────────
    // Build leaf → rhs_pool_slot map (matches order in rhs_all.build()).
    const auto& leaves = tree.leaf_indices();
    std::map<int, int> leaf_to_slot;
    {
        int slot = 0;
        for (int idx : leaves) {
            if (!tree.nodes[idx].has_block()) continue;
            leaf_to_slot[idx] = slot++;
        }
    }

    std::vector<int> fine_leaves, coarse_leaves;
    for (int idx : leaves) {
        if (!tree.nodes[idx].has_block()) continue;
        if (tree.nodes[idx].level == L_max) fine_leaves.push_back(idx);
        else if (tree.nodes[idx].level == L_min) coarse_leaves.push_back(idx);
    }
    n_fine   = (int)fine_leaves.size();
    n_coarse = (int)coarse_leaves.size();

    // Qn buffers
    if (d_Qn_fine)   { cudaFree(d_Qn_fine);   d_Qn_fine   = nullptr; }
    if (d_Qn_coarse) { cudaFree(d_Qn_coarse); d_Qn_coarse = nullptr; }
    if (n_fine   > 0) CUDA_CHECK(cudaMalloc(&d_Qn_fine,
        (size_t)n_fine   * GPU_NVAR * GPU_NCELL * sizeof(double)));
    if (n_coarse > 0) CUDA_CHECK(cudaMalloc(&d_Qn_coarse,
        (size_t)n_coarse * GPU_NVAR * GPU_NCELL * sizeof(double)));

    // Fine metas
    {
        std::vector<GpuRk3LeafMeta> h(n_fine);
        for (int i = 0; i < n_fine; ++i) {
            int idx  = fine_leaves[i];
            int slot = leaf_to_slot.at(idx);
            h[i].d_Q  = pool.d_Q(tree.nodes[idx].block.get());
            h[i].d_Qn = d_Qn_fine + (size_t)i * GPU_NVAR * GPU_NCELL;
            h[i].d_RHS = rhs_all.d_rhs_pool + (size_t)slot * GPU_NVAR * GPU_NCELL;
        }
        gpu_upload_meta(d_fine_metas, h);
    }

    // Coarse metas
    {
        std::vector<GpuRk3LeafMeta> h(n_coarse);
        for (int i = 0; i < n_coarse; ++i) {
            int idx  = coarse_leaves[i];
            int slot = leaf_to_slot.at(idx);
            h[i].d_Q  = pool.d_Q(tree.nodes[idx].block.get());
            h[i].d_Qn = d_Qn_coarse + (size_t)i * GPU_NVAR * GPU_NCELL;
            h[i].d_RHS = rhs_all.d_rhs_pool + (size_t)slot * GPU_NVAR * GPU_NCELL;
        }
        gpu_upload_meta(d_coarse_metas, h);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step: one Berger-Oliger LTS step.  Returns dt_c.
//
// Fine sub-steps: accum CF fine flux (weights sum to 1 over r sub-steps).
// Coarse step:    undo wrong coarse CF flux from d_RHS; no accum (d_reg set).
// apply_correction(dt_c): add d_reg × dt_c to coarse Q at C/F faces.
// ─────────────────────────────────────────────────────────────────────────────
double GpuLtsIntegrator::step(double cfl) {
    if (n_fine == 0 || n_coarse == 0) return 1.0e300;

    constexpr int TPB = 256;
    cudaStream_t s = stream;

    // CFL per level
    const double dt_f_cfl = cfl_fine.exec(cfl, s);
    const double dt_c_cfl = cfl_coarse.exec(cfl, s);
    const double dt_c = std::min(dt_c_cfl, (double)lts_r * dt_f_cfl);
    const double dt_f = dt_c / lts_r;

    // Write dt values to device memory
    CUDA_CHECK(cudaMemcpyAsync(cfl_fine.d_dt,   &dt_f, sizeof(double),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(cfl_coarse.d_dt, &dt_c, sizeof(double),
                               cudaMemcpyHostToDevice, s));

    const size_t rhs_bytes = (size_t)GPU_NVAR * GPU_NCELL * rhs_all.n_leaves
                             * sizeof(double);
    const double r_inv = 1.0 / (double)lts_r;

    // ── Zero flux registers once before all sub-steps ─────────────────────────
    cf.zero_regs(s);

    // ── r fine sub-steps ──────────────────────────────────────────────────────
    for (int k = 0; k < lts_r; ++k) {
        k_save_qn<<<n_fine, TPB, 0, s>>>(d_fine_metas);

        // Stage 1: Q_f = Qn_f + dt_f * RHS_f ; accum fine flux × (1/(6r))
        CUDA_CHECK(cudaMemsetAsync(rhs_all.d_rhs_pool, 0, rhs_bytes, s));
        ghost_all.exec(s);
        rhs_all.exec(s, false);
        cf.accum_fine_flux(s, r_inv / 6.0);
        k_rk3s1<<<n_fine, TPB, 0, s>>>(d_fine_metas, cfl_fine.d_dt);
        k_positivity_floor<<<n_fine, TPB, 0, s>>>(d_fine_metas);

        // Stage 2: Q_f = 3/4*Qn + 1/4*(Q + dt_f*RHS) ; accum fine flux × (1/(6r))
        CUDA_CHECK(cudaMemsetAsync(rhs_all.d_rhs_pool, 0, rhs_bytes, s));
        ghost_all.exec(s);
        rhs_all.exec(s, false);
        cf.accum_fine_flux(s, r_inv / 6.0);
        k_rk3s23<<<n_fine, TPB, 0, s>>>(d_fine_metas, cfl_fine.d_dt, 0.75, 0.25);
        k_positivity_floor<<<n_fine, TPB, 0, s>>>(d_fine_metas);

        // Stage 3: Q_f = 1/3*Qn + 2/3*(Q + dt_f*RHS) ; accum fine flux × (2/(3r))
        CUDA_CHECK(cudaMemsetAsync(rhs_all.d_rhs_pool, 0, rhs_bytes, s));
        ghost_all.exec(s);
        rhs_all.exec(s, false);
        cf.accum_fine_flux(s, r_inv * (2.0/3.0));
        k_rk3s23<<<n_fine, TPB, 0, s>>>(d_fine_metas, cfl_fine.d_dt, 1.0/3.0, 2.0/3.0);
        k_positivity_floor<<<n_fine, TPB, 0, s>>>(d_fine_metas);
    }

    // ── 1 coarse step ─────────────────────────────────────────────────────────
    // Correct Berger-Colella: run coarse RK3 unmodified; accumulate NEGATIVE
    // coarse flux into d_reg so d_reg = avg(F_fine) − F_coarse after the step.
    // apply_correction then adds sign * dt * ih_c * d_reg to coarse Q cells.
    k_save_qn<<<n_coarse, TPB, 0, s>>>(d_coarse_metas);

    // Stage 1
    CUDA_CHECK(cudaMemsetAsync(rhs_all.d_rhs_pool, 0, rhs_bytes, s));
    ghost_all.exec(s);
    rhs_all.exec(s, false);
    cf.accum_coarse_neg_flux(s, 1.0/6.0);
    k_rk3s1<<<n_coarse, TPB, 0, s>>>(d_coarse_metas, cfl_coarse.d_dt);
    k_positivity_floor<<<n_coarse, TPB, 0, s>>>(d_coarse_metas);

    // Stage 2
    CUDA_CHECK(cudaMemsetAsync(rhs_all.d_rhs_pool, 0, rhs_bytes, s));
    ghost_all.exec(s);
    rhs_all.exec(s, false);
    cf.accum_coarse_neg_flux(s, 1.0/6.0);
    k_rk3s23<<<n_coarse, TPB, 0, s>>>(d_coarse_metas, cfl_coarse.d_dt, 0.75, 0.25);
    k_positivity_floor<<<n_coarse, TPB, 0, s>>>(d_coarse_metas);

    // Stage 3
    CUDA_CHECK(cudaMemsetAsync(rhs_all.d_rhs_pool, 0, rhs_bytes, s));
    ghost_all.exec(s);
    rhs_all.exec(s, false);
    cf.accum_coarse_neg_flux(s, 2.0/3.0);
    k_rk3s23<<<n_coarse, TPB, 0, s>>>(d_coarse_metas, cfl_coarse.d_dt,
                                       1.0/3.0, 2.0/3.0);
    k_positivity_floor<<<n_coarse, TPB, 0, s>>>(d_coarse_metas);

    // ── Berger-Colella correction ─────────────────────────────────────────────
    cf.apply_correction(s, dt_c);

    CUDA_CHECK(cudaStreamSynchronize(s));
    return dt_c;
}
