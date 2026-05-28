// gpu_cf.cu — P14.4: GPU Berger-Colella flux-register correction.
//
// Three kernels that mirror the CPU operators.cpp / block_tree.cpp sequence:
//   k_cf_undo  — removes wrong PCM+HLLC-ES CF flux from coarse d_RHS
//   k_cf_accum — accumulates fine-face HLLC-ES flux into d_reg (× weight × area_ratio)
//   k_cf_apply — applies dt × d_reg / h_coarse correction to coarse d_Q
//
// Grid dimensions:  one CUDA block per CF pair (coarse or fine meta entry).
// Block dimensions: dim3(GPU_NB, GPU_NB) = 64 threads — one per boundary face cell.
//
// Sign conventions follow A05-fix3 in block_tree.cpp:
//   +face (side=1, XPLUS/YPLUS/ZPLUS): apply sign = -1  (flux leaves coarse cell)
//   -face (side=0, XMINUS/YMINUS/ZMINUS): apply sign = +1  (flux enters coarse cell)

#include "cuda/gpu_cf.cuh"
#include "cuda/gpu_hllc.cuh"
#include "cuda/gpu_check.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "mesh/block_tree.hpp"
#include <unordered_map>
#include <vector>
#include <utility>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

// Load a GPrim from d_scratch at a given flat cell index.
__device__ __forceinline__
static void load_prim_scratch(const double* __restrict__ sp, int flat, GPrim& q) {
    q.rho = sp[0*GPU_NCELL + flat];
    q.u   = sp[1*GPU_NCELL + flat];
    q.v   = sp[2*GPU_NCELL + flat];
    q.w   = sp[3*GPU_NCELL + flat];
    q.p   = sp[4*GPU_NCELL + flat];
    q.T   = sp[5*GPU_NCELL + flat];
    q.c   = sp[6*GPU_NCELL + flat];
}

// Compute interior and ghost flat indices for a CF boundary face.
// face_dir: 0-5 (XMINUS..ZPLUS).  a, b: transverse thread indices [0, NB-1].
// Returns: bound (boundary interior coord), delta (+1 or -1 toward ghost).
__device__ __forceinline__
static void cf_face_cells(int face_dir, int a, int b,
                           int& flat_int, int& flat_ghost,
                           int& flat_bound_ij) {
    const int axis  = face_dir >> 1;
    const int side  = face_dir & 1;
    const int delta = (side == 1) ? +1 : -1;
    const int bound = (side == 1) ? (GPU_NG + GPU_NB - 1) : GPU_NG;
    const int ga    = GPU_NG + a;
    const int gb    = GPU_NG + b;

    int ci, cj, ck, gi, gj, gk;
    if (axis == 0) {
        ci=bound; cj=ga; ck=gb;  gi=bound+delta; gj=ga; gk=gb;
        flat_bound_ij = gpu_cell_idx(bound, ga, gb);
    } else if (axis == 1) {
        ci=ga; cj=bound; ck=gb;  gi=ga; gj=bound+delta; gk=gb;
        flat_bound_ij = gpu_cell_idx(ga, bound, gb);
    } else {
        ci=ga; cj=gb; ck=bound;  gi=ga; gj=gb; gk=bound+delta;
        flat_bound_ij = gpu_cell_idx(ga, gb, bound);
    }
    flat_int   = gpu_cell_idx(ci, cj, ck);
    flat_ghost = gpu_cell_idx(gi, gj, gk);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_cf_undo
// Grid:  (n_coarse, 1, 1)   Block: (GPU_NB, GPU_NB, 1) = 64 threads
//
// For each coarse CF face, undoes the PCM+HLLC-ES boundary flux that k_rhs_conv
// added, so that apply_correction can replace it with the fine-side average.
// Exact cancellation: same HLLC-ES(pL, pR) formula as k_rhs_conv for is_bnd.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_cf_undo(const GpuCfCoarseMeta* __restrict__ metas) {
    const GpuCfCoarseMeta& m = metas[blockIdx.x];
    const int a = threadIdx.x;  // [0, GPU_NB-1]
    const int b = threadIdx.y;

    const int axis  = m.face_dir >> 1;
    const int side  = m.face_dir & 1;
    const int delta = (side == 1) ? +1 : -1;
    const double sign = (double)delta;
    const double ih   = 1.0 / m.h_coarse;

    int flat_I, flat_G, flat_rhs;
    cf_face_cells(m.face_dir, a, b, flat_I, flat_G, flat_rhs);

    GPrim pI, pG;
    load_prim_scratch(m.d_scratch, flat_I, pI);
    load_prim_scratch(m.d_scratch, flat_G, pG);

    // Same formula as k_rhs_conv for is_bnd (PCM+HLLC-ES):
    // delta>0 → right face → pL=interior, pR=ghost
    // delta<0 → left face  → pL=ghost,    pR=interior
    const GPrim& pL = (delta > 0) ? pI : pG;
    const GPrim& pR = (delta > 0) ? pG : pI;
    double F[GPU_NVAR];
    gpu_hllc_es_flux(pL, pR, axis, F);

    // k_rhs_conv added:
    //   right face (bL=true): rhs[interior] -= ih*F  →  undo adds +ih*F  (sign=+1)
    //   left  face (bR=true): rhs[interior] += ih*F  →  undo adds -ih*F  (sign=-1)
    for (int v = 0; v < GPU_NVAR; ++v)
        atomicAdd(&m.d_RHS[v * GPU_NCELL + flat_rhs], sign * ih * F[v]);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_cf_accum
// Grid:  (n_fine, 1, 1)   Block: (GPU_NB, GPU_NB, 1) = 64 threads
//
// Accumulates fine-face HLLC-ES flux × stage_weight × area_ratio (0.25) into
// the shared coarse flux register d_reg.  Four fine cells (a,b) map to each
// coarse register entry (jc, ic) via integer division; atomicAdd ensures
// correctness when multiple blocks hit the same entry.
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_cf_accum(const GpuCfFineMeta* __restrict__ metas, double stage_weight) {
    const GpuCfFineMeta& m = metas[blockIdx.x];
    const int a = threadIdx.x;
    const int b = threadIdx.y;

    const int axis  = m.face_dir >> 1;
    const int side  = m.face_dir & 1;
    const int delta = (side == 1) ? +1 : -1;

    int flat_I, flat_G, flat_unused;
    cf_face_cells(m.face_dir, a, b, flat_I, flat_G, flat_unused);

    GPrim pI, pG;
    load_prim_scratch(m.d_scratch, flat_I, pI);
    load_prim_scratch(m.d_scratch, flat_G, pG);

    const GPrim& pL = (delta > 0) ? pI : pG;
    const GPrim& pR = (delta > 0) ? pG : pI;
    double F[GPU_NVAR];
    gpu_hllc_es_flux(pL, pR, axis, F);

    // Map (a, b) → coarse register (jc, ic) using octant offsets.
    // Matches CPU cf_accum_one_face index arithmetic (A05-fix7):
    //   axis=X: jc from off1 + a/2,  ic from off2 + b/2
    //   axis=Y/Z: jc from off1 + b/2, ic from off2 + a/2
    constexpr int HALF = GPU_NB / 2;
    int jc, ic;
    if (axis == 0) {
        jc = (int)m.off1 * HALF + a / 2;
        ic = (int)m.off2 * HALF + b / 2;
    } else {
        jc = (int)m.off1 * HALF + b / 2;
        ic = (int)m.off2 * HALF + a / 2;
    }

    // area_ratio = 0.25: fine face area = (h/2)² = h²/4; divide by coarse h².
    for (int v = 0; v < GPU_NVAR; ++v)
        atomicAdd(&m.d_reg[v * GPU_NB * GPU_NB + jc * GPU_NB + ic],
                  stage_weight * 0.25 * F[v]);
}

// ─────────────────────────────────────────────────────────────────────────────
// k_cf_apply
// Grid:  (n_coarse, 1, 1)   Block: (GPU_NB, GPU_NB, 1) = 64 threads
//
// Applies the accumulated flux correction to the coarse d_Q.
// Mirrors apply_flux_correction() in block_tree.cpp (A05-fix3 sign).
// ─────────────────────────────────────────────────────────────────────────────
__global__
void k_cf_apply(const GpuCfCoarseMeta* __restrict__ metas, double dt) {
    const GpuCfCoarseMeta& m = metas[blockIdx.x];
    const int jc = threadIdx.x;  // [0, NB-1]
    const int ic = threadIdx.y;

    const int axis = m.face_dir >> 1;
    const int side = m.face_dir & 1;
    // A05-fix3: +face (side=1) subtracts (flux leaves cell → sign=-1)
    //           -face (side=0) adds      (flux enters cell → sign=+1)
    const double sign  = (side == 1) ? -1.0 : +1.0;
    const int    g     = (side == 0) ? GPU_NG : (GPU_NG + GPU_NB - 1);
    const double ih_c  = 1.0 / m.h_coarse;

    // Physical cell index on coarse boundary face.
    // axis=0 (X): ci=g, cj=ilo+jc, ck=ilo+ic
    // axis=1 (Y): ci=ilo+ic, cj=g, ck=ilo+jc
    // axis=2 (Z): ci=ilo+ic, cj=ilo+jc, ck=g
    int flat;
    if (axis == 0)
        flat = gpu_cell_idx(g, GPU_NG+jc, GPU_NG+ic);
    else if (axis == 1)
        flat = gpu_cell_idx(GPU_NG+ic, g, GPU_NG+jc);
    else
        flat = gpu_cell_idx(GPU_NG+ic, GPU_NG+jc, g);

    for (int v = 0; v < GPU_NVAR; ++v) {
        double corr = sign * dt * ih_c * m.d_reg[v * GPU_NB * GPU_NB + jc * GPU_NB + ic];
        atomicAdd(&m.d_Q[v * GPU_NCELL + flat], corr);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuCfList methods
// ─────────────────────────────────────────────────────────────────────────────

GpuCfList::~GpuCfList() {
    cudaFree(d_coarse);   d_coarse   = nullptr;
    cudaFree(d_fine);     d_fine     = nullptr;
    cudaFree(d_reg_pool); d_reg_pool = nullptr;
}

void GpuCfList::build(const BlockTree& tree, const GpuPool& pool,
                      double* d_rhs_pool, double* d_scratch_pool) {
    cudaFree(d_coarse);   d_coarse   = nullptr; n_coarse = 0;
    cudaFree(d_fine);     d_fine     = nullptr; n_fine   = 0;
    cudaFree(d_reg_pool); d_reg_pool = nullptr;

    const auto& leaves = tree.leaf_indices();
    const int n_leaves = (int)leaves.size();
    if (n_leaves == 0) return;

    // Build leaf-index lookup: node_idx → sequential position li
    std::unordered_map<int,int> node_to_li;
    node_to_li.reserve(n_leaves);
    for (int li = 0; li < n_leaves; ++li)
        node_to_li[leaves[li]] = li;

    auto scratch_ptr = [&](int li) -> const double* {
        return d_scratch_pool + (size_t)li * SCRATCH_NCOMP * NCELL;
    };
    auto rhs_ptr = [&](int li) -> double* {
        return d_rhs_pool + (size_t)li * NVAR * NCELL;
    };

    // ── Pass 1: collect coarse CF faces ──────────────────────────────────────
    std::vector<GpuCfCoarseMeta> h_coarse;
    // Map (coarse_node_idx, face_dir_toward_fine) → reg_slot_index
    std::unordered_map<int, int> coarse_key_to_slot;  // key = node_idx*8 + face_dir
    coarse_key_to_slot.reserve(32);

    for (int li = 0; li < n_leaves; ++li) {
        int nidx = leaves[li];
        const BlockNode& nd = tree.nodes[nidx];
        if (!nd.has_block()) continue;

        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            if (ni < 0 || !tree.nodes[ni].has_block()) continue;
            if (tree.nodes[ni].level <= nd.level) continue;  // not finer

            int slot = (int)h_coarse.size();
            coarse_key_to_slot[nidx * 8 + d] = slot;

            GpuCfCoarseMeta m{};
            m.d_RHS     = rhs_ptr(li);
            m.d_Q       = pool.d_Q(nd.block.get());
            m.d_scratch = scratch_ptr(li);
            m.d_reg     = nullptr;  // set after pool alloc
            m.h_coarse  = (float)nd.block->h;
            m.face_dir  = (int8_t)d;
            h_coarse.push_back(m);
        }
    }

    n_coarse = (int)h_coarse.size();
    if (n_coarse == 0) return;

    // Allocate reg pool (zero-initialised via cudaMallocManaged or explicit memset)
    const size_t reg_bytes = (size_t)n_coarse * NVAR * NB * NB * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_reg_pool, reg_bytes));
    CUDA_CHECK(cudaMemset(d_reg_pool, 0, reg_bytes));

    // Patch d_reg pointers
    for (int i = 0; i < n_coarse; ++i)
        h_coarse[i].d_reg = d_reg_pool + (size_t)i * NVAR * NB * NB;

    gpu_upload_meta(d_coarse, h_coarse);

    // ── Pass 2: collect fine CF faces ────────────────────────────────────────
    std::vector<GpuCfFineMeta> h_fine;

    for (int li = 0; li < n_leaves; ++li) {
        int nidx = leaves[li];
        const BlockNode& nd = tree.nodes[nidx];
        if (!nd.has_block()) continue;

        const int parent_idx = nd.parent;
        const int oct = (parent_idx >= 0)
                        ? nidx - tree.nodes[parent_idx].first_child
                        : 0;
        const int o_ix = oct_ix(oct);
        const int o_iy = oct_iy(oct);
        const int o_iz = oct_iz(oct);

        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            if (ni < 0 || !tree.nodes[ni].has_block()) continue;
            if (tree.nodes[ni].level >= nd.level) continue;  // not coarser

            // d = fine→coarse; coarse's face toward this fine = d^1
            int d_c = d ^ 1;
            auto it = coarse_key_to_slot.find(ni * 8 + d_c);
            if (it == coarse_key_to_slot.end()) continue;
            int slot = it->second;

            // Octant offsets: which quadrant of the NB×NB coarse face this fine covers
            const int axis = d >> 1;
            int off1, off2;
            if      (axis == 0) { off1 = o_iy; off2 = o_iz; }
            else if (axis == 1) { off1 = o_iz; off2 = o_ix; }
            else                { off1 = o_iy; off2 = o_ix; }

            GpuCfFineMeta m{};
            m.d_scratch = scratch_ptr(li);
            m.d_reg     = d_reg_pool + (size_t)slot * NVAR * NB * NB;
            m.h_fine    = (float)nd.block->h;
            m.face_dir  = (int8_t)d;
            m.off1      = (int8_t)off1;
            m.off2      = (int8_t)off2;
            h_fine.push_back(m);
        }
    }

    n_fine = (int)h_fine.size();
    gpu_upload_meta(d_fine, h_fine);
}

void GpuCfList::zero_regs(cudaStream_t stream) const {
    if (n_coarse == 0 || !d_reg_pool) return;
    const size_t reg_bytes = (size_t)n_coarse * NVAR * NB * NB * sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(d_reg_pool, 0, reg_bytes, stream));
}

void GpuCfList::undo_coarse_flux(cudaStream_t stream) const {
    if (n_coarse == 0) return;
    k_cf_undo<<<n_coarse, dim3(GPU_NB, GPU_NB), 0, stream>>>(d_coarse);
}

void GpuCfList::accum_fine_flux(cudaStream_t stream, double stage_weight) const {
    if (n_fine == 0) return;
    k_cf_accum<<<n_fine, dim3(GPU_NB, GPU_NB), 0, stream>>>(d_fine, stage_weight);
}

void GpuCfList::apply_correction(cudaStream_t stream, double dt) const {
    if (n_coarse == 0) return;
    k_cf_apply<<<n_coarse, dim3(GPU_NB, GPU_NB), 0, stream>>>(d_coarse, dt);
}
