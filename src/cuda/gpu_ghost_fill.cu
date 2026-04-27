// P8.2: GPU ghost fill kernels
//
// k_fill_faces:        face ghost cells (cross-block communication)
// k_fill_edges_corners: edge/corner ghost cells (self-referencing, reads own interior)
//
// CF fine←coarse: 5th-order Lagrange in normal direction (P7.2 coefficients).
// CF coarse←fine: zero-gradient fallback — upgraded to conservative average in P8.4.
// Edge/corner source: src = (ghost < NG) ? ghost+NB : ghost-NB  (always interior).

#include "../../include/cuda/gpu_ghost_fill.cuh"
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

static void chk(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}

// ── Device helpers ────────────────────────────────────────────────────────────
__device__ __forceinline__ int cidx(int i, int j, int k) {
    return k * NB2 * NB2 + j * NB2 + i;
}

// 5th-order Lagrange coefficients (P7.2 / McCorquodale-Colella 2011)
// Nodes at {-4,-3,-2,-1,0} in units of h_coarse.
// Lp: target at +1/4 h_c (gl=0, side=0)
// Lm: target at -1/4 h_c (gl=1, side=0)
// side=1: reverse order
__device__ static const double kLp[5] = {
     585.0/6144, -3060.0/6144,  6630.0/6144, -7956.0/6144,  9945.0/6144
};
__device__ static const double kLm[5] = {
    -231.0/6144,  1260.0/6144, -2970.0/6144,  4620.0/6144,  3465.0/6144
};

__device__ __forceinline__ int loclamp(int x) {
    return (x >= NG && x < NG + NB) ? (x - NG) : (x < NG ? 0 : NB - 1);
}

// ── CF fine←coarse: fill one ghost cell for variable v ───────────────────────
__device__ double cf_fine_from_coarse(
    const double* __restrict__ d_coarse,
    int v, int axis, int side, int gl,
    int a, int b,   // transverse fine coords in [0, NB2-1]
    int cf_oct
) {
    const double* Lc = (gl == 0) ? kLp : kLm;
    const int ix = cf_oct & 1;
    const int iy = (cf_oct >> 1) & 1;
    const int iz = (cf_oct >> 2) & 1;
    double val = 0.0;

    if (axis == 0) {
        int cj = NG + iy * (NB / 2) + loclamp(a) / 2;
        int ck = NG + iz * (NB / 2) + loclamp(b) / 2;
        if (side == 0) {
            int i0 = NG + NB - 1;
            for (int k = 0; k < 5; ++k)
                val += Lc[k] * d_coarse[v * NCELL + cidx(i0 - 4 + k, cj, ck)];
        } else {
            int i0 = NG;
            for (int k = 0; k < 5; ++k)
                val += Lc[4 - k] * d_coarse[v * NCELL + cidx(i0 + k, cj, ck)];
        }
    } else if (axis == 1) {
        int ci = NG + ix * (NB / 2) + loclamp(a) / 2;
        int ck = NG + iz * (NB / 2) + loclamp(b) / 2;
        if (side == 0) {
            int j0 = NG + NB - 1;
            for (int k = 0; k < 5; ++k)
                val += Lc[k] * d_coarse[v * NCELL + cidx(ci, j0 - 4 + k, ck)];
        } else {
            int j0 = NG;
            for (int k = 0; k < 5; ++k)
                val += Lc[4 - k] * d_coarse[v * NCELL + cidx(ci, j0 + k, ck)];
        }
    } else {
        int ci = NG + ix * (NB / 2) + loclamp(a) / 2;
        int cj = NG + iy * (NB / 2) + loclamp(b) / 2;
        if (side == 0) {
            int k0 = NG + NB - 1;
            for (int k = 0; k < 5; ++k)
                val += Lc[k] * d_coarse[v * NCELL + cidx(ci, cj, k0 - 4 + k)];
        } else {
            int k0 = NG;
            for (int k = 0; k < 5; ++k)
                val += Lc[4 - k] * d_coarse[v * NCELL + cidx(ci, cj, k0 + k)];
        }
    }
    return val;
}

// =============================================================================
// k_fill_faces — face ghost fill kernel
// Grid: (n_leaves, NFACES)   Block: 256
// =============================================================================
__global__ void k_fill_faces(const GpuLeafGhostMeta* metas) {
    const GpuLeafGhostMeta& m = metas[blockIdx.x];
    const int face = blockIdx.y;
    const int axis = face >> 1;
    const int side = face & 1;

    double*       d_dst = m.d_Q;
    const double* d_src = m.d_nb[face];
    const int     lrel  = m.level_rel[face];

    // Total cells per face: NB2 × NB2 × NG
    // For same-level: only a,b ∈ [NG, NG+NB-1] are meaningful (128 cells).
    // For CF fine←coarse: all a,b ∈ [0, NB2-1] (288 cells, fills edge/corner
    // ghosts at the CF face too).
    static constexpr int TOTAL = NB2 * NB2 * NG;  // 288

    for (int c = threadIdx.x; c < TOTAL; c += blockDim.x) {
        const int gl = c / (NB2 * NB2);   // ghost layer [0, NG-1]
        const int ab = c % (NB2 * NB2);
        const int a  = ab / NB2;           // transverse dim-1 in [0, NB2-1]
        const int b  = ab % NB2;           // transverse dim-2 in [0, NB2-1]

        // Destination ghost index along axis
        int dst_ax = (side == 0) ? (NG - 1 - gl) : (NB + NG + gl);
        int dst_i, dst_j, dst_k;
        if      (axis == 0) { dst_i = dst_ax; dst_j = a;      dst_k = b;      }
        else if (axis == 1) { dst_i = a;      dst_j = dst_ax; dst_k = b;      }
        else                { dst_i = a;      dst_j = b;      dst_k = dst_ax; }
        const int dst_flat = cidx(dst_i, dst_j, dst_k);

        // ── Same-level neighbor copy ─────────────────────────────────────────
        if (d_src != nullptr && lrel == 0) {
            if (a < NG || a >= NG + NB || b < NG || b >= NG + NB) continue;
            int src_ax = (side == 0) ? (NB + NG - 1 - gl) : (NG + gl);
            int src_i, src_j, src_k;
            if      (axis == 0) { src_i = src_ax; src_j = a;      src_k = b;      }
            else if (axis == 1) { src_i = a;      src_j = src_ax; src_k = b;      }
            else                { src_i = a;      src_j = b;      src_k = src_ax; }
            const int src_flat = cidx(src_i, src_j, src_k);
            for (int v = 0; v < NVAR; ++v)
                d_dst[v * NCELL + dst_flat] = d_src[v * NCELL + src_flat];

        // ── CF fine←coarse: 5th-order Lagrange ──────────────────────────────
        } else if (d_src != nullptr && lrel == -1) {
            for (int v = 0; v < NVAR; ++v)
                d_dst[v * NCELL + dst_flat] =
                    cf_fine_from_coarse(d_src, v, axis, side, gl,
                                        a, b, m.cf_oct);

        // ── CF coarse←fine: zero-gradient fallback (P8.2; average in P8.4) ──
        } else if (d_src != nullptr && lrel == +1) {
            if (a < NG || a >= NG + NB || b < NG || b >= NG + NB) continue;
            int int_ax = (side == 0) ? NG : (NB + NG - 1);
            int int_i, int_j, int_k;
            if      (axis == 0) { int_i = int_ax; int_j = a;      int_k = b;      }
            else if (axis == 1) { int_i = a;      int_j = int_ax; int_k = b;      }
            else                { int_i = a;      int_j = b;      int_k = int_ax; }
            const int int_flat = cidx(int_i, int_j, int_k);
            for (int v = 0; v < NVAR; ++v)
                d_dst[v * NCELL + dst_flat] = d_dst[v * NCELL + int_flat];

        // ── Domain boundary ─────────────────────────────────────────────────
        } else if (d_src == nullptr) {
            if (a < NG || a >= NG + NB || b < NG || b >= NG + NB) continue;

            if (m.bc_type == 1) {
                // Wall: all momenta negated (no-slip adiabatic)
                int mir_ax = (side == 0) ? (NG + gl) : (NB + NG - 1 - gl);
                int mir_i, mir_j, mir_k;
                if      (axis == 0) { mir_i = mir_ax; mir_j = a;      mir_k = b;      }
                else if (axis == 1) { mir_i = a;      mir_j = mir_ax; mir_k = b;      }
                else                { mir_i = a;      mir_j = b;      mir_k = mir_ax; }
                const int mir_flat = cidx(mir_i, mir_j, mir_k);
                for (int v = 0; v < NVAR; ++v) {
                    double val = d_dst[v * NCELL + mir_flat];
                    // Negate all momentum components (v=1,2,3)
                    d_dst[v * NCELL + dst_flat] = (v >= 1 && v <= 3) ? -val : val;
                }

            } else if (m.bc_type == 2) {
                // Open: zero gradient (ghost = nearest interior cell)
                int int_ax = (side == 0) ? NG : (NB + NG - 1);
                int int_i, int_j, int_k;
                if      (axis == 0) { int_i = int_ax; int_j = a;      int_k = b;      }
                else if (axis == 1) { int_i = a;      int_j = int_ax; int_k = b;      }
                else                { int_i = a;      int_j = b;      int_k = int_ax; }
                const int int_flat = cidx(int_i, int_j, int_k);
                for (int v = 0; v < NVAR; ++v)
                    d_dst[v * NCELL + dst_flat] = d_dst[v * NCELL + int_flat];

            } else {
                // Periodic self-wrap (root block in single-block periodic domain)
                int src_ax = (side == 0) ? (NB + NG - 1 - gl) : (NG + gl);
                int src_i, src_j, src_k;
                if      (axis == 0) { src_i = src_ax; src_j = a;      src_k = b;      }
                else if (axis == 1) { src_i = a;      src_j = src_ax; src_k = b;      }
                else                { src_i = a;      src_j = b;      src_k = src_ax; }
                const int src_flat = cidx(src_i, src_j, src_k);
                for (int v = 0; v < NVAR; ++v)
                    d_dst[v * NCELL + dst_flat] = d_dst[v * NCELL + src_flat];
            }
        }
    }
}

// =============================================================================
// k_fill_edges_corners — edge/corner ghost fill
// Grid: n_leaves   Block: 256
// Reads from own interior: src = (ghost < NG) ? ghost+NB : ghost-NB
// No cross-block dependency — can run concurrently with k_fill_faces.
// =============================================================================
__global__ void k_fill_edges_corners(const GpuLeafGhostMeta* metas) {
    double* d_Q = metas[blockIdx.x].d_Q;

    for (int c = threadIdx.x; c < NCELL; c += blockDim.x) {
        const int i = c % NB2;
        const int j = (c / NB2) % NB2;
        const int k = c / (NB2 * NB2);

        const bool gx = (i < NG || i >= NG + NB);
        const bool gy = (j < NG || j >= NG + NB);
        const bool gz = (k < NG || k >= NG + NB);
        const int  ng = (gx ? 1 : 0) + (gy ? 1 : 0) + (gz ? 1 : 0);
        if (ng < 2) continue;  // interior (0) or face ghost (1) — skip

        // Map each ghost coordinate back to its interior mirror
        const int si = gx ? ((i < NG) ? i + NB : i - NB) : i;
        const int sj = gy ? ((j < NG) ? j + NB : j - NB) : j;
        const int sk = gz ? ((k < NG) ? k + NB : k - NB) : k;

        const int dst = c;
        const int src = cidx(si, sj, sk);
        for (int v = 0; v < NVAR; ++v)
            d_Q[v * NCELL + dst] = d_Q[v * NCELL + src];
    }
}

// =============================================================================
// C-linkage launchers (callable from non-CUDA TUs if needed)
// =============================================================================
extern "C" {

void gpu_ghost_fill_faces_launch(const GpuLeafGhostMeta* d_metas,
                                  int n_leaves, cudaStream_t stream) {
    if (n_leaves == 0) return;
    dim3 grid(n_leaves, NFACES);
    k_fill_faces<<<grid, 256, 0, stream>>>(d_metas);
}

void gpu_ghost_fill_ec_launch(const GpuLeafGhostMeta* d_metas,
                               int n_leaves, cudaStream_t stream) {
    if (n_leaves == 0) return;
    k_fill_edges_corners<<<n_leaves, 256, 0, stream>>>(d_metas);
}

} // extern "C"

// =============================================================================
// GpuGhostFillList implementation
// =============================================================================
GpuGhostFillList::~GpuGhostFillList() {
    if (d_metas) { cudaFree(d_metas); d_metas = nullptr; }
}

void GpuGhostFillList::build(const BlockTree& tree, int bc_type) {
    const auto& leaves = tree.leaf_indices();
    const int n = static_cast<int>(leaves.size());

    std::vector<GpuLeafGhostMeta> h(n);
    for (int ii = 0; ii < n; ++ii) {
        const int li = leaves[ii];
        const BlockNode& nd = tree.nodes[li];
        GpuLeafGhostMeta& m = h[ii];

        m.d_Q     = nd.block ? nd.block->d_Q : nullptr;
        m.bc_type = static_cast<int8_t>(bc_type);
        m.cf_oct  = 0;
        if (nd.parent >= 0) {
            int fc   = tree.nodes[nd.parent].first_child;
            m.cf_oct = static_cast<int8_t>(li - fc);
        }

        for (int d = 0; d < NFACES; ++d) {
            const int ni = nd.neighbours[d];
            if (ni >= 0 && tree.nodes[ni].has_block() &&
                tree.nodes[ni].block->d_Q != nullptr) {
                m.d_nb[d]    = tree.nodes[ni].block->d_Q;
                m.level_rel[d] =
                    static_cast<int8_t>(tree.nodes[ni].level - nd.level);
            } else {
                m.d_nb[d]    = nullptr;
                m.level_rel[d] = 0;
            }
        }
    }

    n_leaves = n;
    if (d_metas) { chk(cudaFree(d_metas), "GpuGhostFillList free old"); d_metas = nullptr; }
    if (n > 0) {
        chk(cudaMalloc(&d_metas, n * sizeof(GpuLeafGhostMeta)),
            "GpuGhostFillList cudaMalloc");
        chk(cudaMemcpy(d_metas, h.data(), n * sizeof(GpuLeafGhostMeta),
                       cudaMemcpyHostToDevice), "GpuGhostFillList upload meta");
    }
}

void GpuGhostFillList::exec(cudaStream_t stream) const {
    if (n_leaves == 0 || !d_metas) return;
    // Face fill and edge/corner fill are independent (EC reads interior, not face ghosts).
    dim3 grid_face(n_leaves, NFACES);
    k_fill_faces        <<<grid_face,  256, 0, stream>>>(d_metas);
    k_fill_edges_corners<<<n_leaves,   256, 0, stream>>>(d_metas);
}
