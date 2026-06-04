// P8.2: GPU ghost fill kernels
//
// k_fill_faces:        face ghost cells (cross-block communication)
// k_fill_edges_corners: edge/corner ghost cells (self-referencing, reads own interior)
//
// CF fine←coarse: 5th-order Lagrange in normal direction (P7.2 coefficients).
// CF coarse←fine: zero-gradient fallback — upgraded to conservative average in P8.4.
// Edge/corner source: src = (ghost < NG) ? ghost+NB : ghost-NB  (always interior).

#include "cuda/gpu_ghost_fill.cuh"
#include "cuda/gpu_meta_buffer.cuh"
#include "mpi/mpi_comm.hpp"
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

// ── Device helpers ────────────────────────────────────────────────────────────
__device__ __forceinline__ int cidx(int i, int j, int k) {
    return k * NB2 * NB2 + j * NB2 + i;
}

// Axis-dispatch flat index: axis dimension gets ax_val; transverse dims get a and b.
__device__ __forceinline__ int cidx_axis(int axis, int ax_val, int a, int b) noexcept {
    if      (axis == 0) return cidx(ax_val, a,      b     );
    else if (axis == 1) return cidx(a,      ax_val, b     );
    else                return cidx(a,      b,      ax_val);
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

// ── Face ghost fill helpers ───────────────────────────────────────────────────

// Copy NVAR values from d_src at the interior-side cell into d_dst ghost.
// Used by same-level neighbor fill and periodic self-wrap (pass d_dst as d_src).
__device__ __forceinline__ static void fill_copy(
    double* d_dst, const double* d_src,
    int axis, int side, int gl, int a, int b, int dst_flat)
{
    const int src_ax   = (side == 0) ? (NB + NG - 1 - gl) : (NG + gl);
    const int src_flat = cidx_axis(axis, src_ax, a, b);
    for (int v = 0; v < NVAR; ++v)
        d_dst[v * NCELL + dst_flat] = d_src[v * NCELL + src_flat];
}

// Zero-gradient fill: ghost = nearest interior cell along normal.
// Used by CF coarse←fine and open domain BC (identical formula).
__device__ __forceinline__ static void fill_zero_grad(
    double* d_dst, int axis, int side, int a, int b, int dst_flat)
{
    const int int_ax   = (side == 0) ? NG : (NB + NG - 1);
    const int int_flat = cidx_axis(axis, int_ax, a, b);
    for (int v = 0; v < NVAR; ++v)
        d_dst[v * NCELL + dst_flat] = d_dst[v * NCELL + int_flat];
}

// NSCBC characteristic ghost (Thompson 1987 / Poinsot-Lele 1992).
// Subsonic outflow: pin p_ghost = p_inf (isentropic ρ, Riemann-invariant v_n).
// Supersonic or inflow (v_n ≤ 0 or v_n ≥ c): zero-gradient fallback.
__device__ __forceinline__ static void fill_nscbc(
    double* d_Q, int axis, int side, int a, int b, int dst_flat, float p_inf_f)
{
    // Read nearest interior cell
    const int int_ax   = (side == 0) ? NG : (NB + NG - 1);
    const int int_flat = cidx_axis(axis, int_ax, a, b);

    const double rho  = d_Q[0 * NCELL + int_flat];
    const double rhou = d_Q[1 * NCELL + int_flat];
    const double rhov = d_Q[2 * NCELL + int_flat];
    const double rhow = d_Q[3 * NCELL + int_flat];
    const double E    = d_Q[4 * NCELL + int_flat];

    const double u  = rhou / rho;
    const double v  = rhov / rho;
    const double w  = rhow / rho;
    const double ke = 0.5 * (u*u + v*v + w*w);
    const double p  = (GAMMA - 1.0) * (E - rho * ke);
    // Guard: pathological state (pre-positivity-floor) — fall through to zero-gradient
    if (!(p > 0.0) || !(rho > 0.0)) {
        for (int v = 0; v < NVAR; ++v)
            d_Q[v * NCELL + dst_flat] = d_Q[v * NCELL + int_flat];
        return;
    }
    const double c  = sqrt(GAMMA * p / rho);

    // Outward-pointing normal velocity (positive = leaving domain)
    const double s   = (side == 0) ? -1.0 : 1.0;
    const double v_n = s * (axis == 0 ? u : axis == 1 ? v : w);

    double rho_g, ug, vg, wg, pg;
    const double p_inf = (double)p_inf_f;

    if (p_inf > 0.0 && v_n > 0.0 && v_n < c) {
        // Subsonic outflow: isentropic ρ, Riemann-invariant normal velocity
        const double ratio = p_inf / p;
        const double c_g   = c * pow(ratio, (GAMMA - 1.0) / (2.0 * GAMMA));
        const double du_n  = s * 2.0 * (c_g - c) / (GAMMA - 1.0);
        pg    = p_inf;
        rho_g = rho * pow(ratio, 1.0 / GAMMA);
        if (axis == 0) { ug = u + du_n; vg = v;         wg = w; }
        else if (axis == 1) { ug = u;   vg = v + du_n;  wg = w; }
        else                { ug = u;   vg = v;          wg = w + du_n; }
    } else {
        // Zero-gradient fallback: inflow, supersonic, or p_inf not set
        rho_g = rho; ug = u; vg = v; wg = w; pg = p;
    }

    const double ke_g = 0.5 * (ug*ug + vg*vg + wg*wg);
    d_Q[0 * NCELL + dst_flat] = rho_g;
    d_Q[1 * NCELL + dst_flat] = rho_g * ug;
    d_Q[2 * NCELL + dst_flat] = rho_g * vg;
    d_Q[3 * NCELL + dst_flat] = rho_g * wg;
    d_Q[4 * NCELL + dst_flat] = pg / (GAMMA - 1.0) + rho_g * ke_g;
}

// Wall BC: reflect momenta, copy scalars (no-slip adiabatic).
__device__ __forceinline__ static void fill_wall(
    double* d_dst, int axis, int side, int gl, int a, int b, int dst_flat)
{
    const int mir_ax   = (side == 0) ? (NG + gl) : (NB + NG - 1 - gl);
    const int mir_flat = cidx_axis(axis, mir_ax, a, b);
    for (int v = 0; v < NVAR; ++v) {
        const double val = d_dst[v * NCELL + mir_flat];
        d_dst[v * NCELL + dst_flat] = (v >= 1 && v <= 3) ? -val : val;
    }
}

// Inviscid slip wall: only the wall-normal momentum component (v == axis+1) is negated.
__device__ __forceinline__ static void fill_slip_wall(
    double* d_dst, int axis, int side, int gl, int a, int b, int dst_flat)
{
    const int mir_ax   = (side == 0) ? (NG + gl) : (NB + NG - 1 - gl);
    const int mir_flat = cidx_axis(axis, mir_ax, a, b);
    for (int v = 0; v < NVAR; ++v) {
        const double val = d_dst[v * NCELL + mir_flat];
        d_dst[v * NCELL + dst_flat] = (v == axis + 1) ? -val : val;
    }
}

// =============================================================================
// k_fill_faces — face ghost fill kernel
// Grid: (n_leaves, NFACES)   Block: 256
// Four dispatch paths: same-level copy, CF fine←coarse (5th-order Lagrange),
// CF coarse←fine (zero-gradient), domain BC (wall/open/periodic self-wrap).
// =============================================================================
__global__ void k_fill_faces(const GpuLeafGhostMeta* metas) {
    const GpuLeafGhostMeta& m = metas[blockIdx.x];
    const int face = blockIdx.y;
    // P-MPI-GPU: ghost cells on remote-rank faces are pre-filled via MPI exchange.
    if (m.is_mpi_face[face]) return;
    const int axis = face >> 1;
    const int side = face & 1;

    double*       d_dst = m.d_Q;
    const double* d_src = m.d_nb[face];
    const int     lrel  = m.level_rel[face];

    // Total ghost cells per face: NB2 × NB2 × NG = 288.
    // CF fine←coarse fills all (a,b) including edges; other branches guard to
    // interior-only (a,b ∈ [NG, NG+NB-1]).
    static constexpr int TOTAL = NB2 * NB2 * NG;

    for (int c = threadIdx.x; c < TOTAL; c += blockDim.x) {
        const int gl = c / (NB2 * NB2);
        const int ab = c % (NB2 * NB2);
        const int a  = ab / NB2;
        const int b  = ab % NB2;

        const int dst_ax   = (side == 0) ? (NG - 1 - gl) : (NB + NG + gl);
        const int dst_flat = cidx_axis(axis, dst_ax, a, b);

        if (d_src != nullptr && lrel == 0) {
            // Same-level neighbor copy
            if (a < NG || a >= NG + NB || b < NG || b >= NG + NB) continue;
            fill_copy(d_dst, d_src, axis, side, gl, a, b, dst_flat);

        } else if (d_src != nullptr && lrel == -1) {
            // CF fine←coarse: 5th-order Lagrange (fills all a,b)
            for (int v = 0; v < NVAR; ++v)
                d_dst[v * NCELL + dst_flat] =
                    cf_fine_from_coarse(d_src, v, axis, side, gl, a, b, m.cf_oct);

        } else if (d_src != nullptr && lrel == +1) {
            // CF coarse←fine: zero-gradient fallback
            if (a < NG || a >= NG + NB || b < NG || b >= NG + NB) continue;
            fill_zero_grad(d_dst, axis, side, a, b, dst_flat);

        } else if (d_src == nullptr) {
            // Domain boundary
            if (a < NG || a >= NG + NB || b < NG || b >= NG + NB) continue;
            const int bc = m.bc_type[face];
            if      (bc == 1) fill_wall(d_dst, axis, side, gl, a, b, dst_flat);
            else if (bc == 2) fill_zero_grad(d_dst, axis, side, a, b, dst_flat);
            else if (bc == 3) fill_nscbc(d_dst, axis, side, a, b, dst_flat,
                                         m.open_p_inf[face]);
            else if (bc == 4) fill_slip_wall(d_dst, axis, side, gl, a, b, dst_flat);
            else              fill_copy(d_dst, d_dst, axis, side, gl, a, b, dst_flat);
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

void GpuGhostFillList::build(const BlockTree& tree, const GpuPool& pool, int bc_type,
                              const MpiPartition* mpi_part) {
    std::array<int,6>   bc_types;  bc_types.fill(bc_type);
    std::array<float,6> p_infs;    p_infs.fill(0.0f);
    build(tree, pool, bc_types, p_infs, mpi_part);
}

void GpuGhostFillList::build(const BlockTree& tree, const GpuPool& pool,
                              const std::array<int,6>& bc_types,
                              const MpiPartition* mpi_part) {
    std::array<float,6> p_infs;  p_infs.fill(0.0f);
    build(tree, pool, bc_types, p_infs, mpi_part);
}

void GpuGhostFillList::build(const BlockTree& tree, const GpuPool& pool,
                              const std::array<int,6>& bc_types,
                              const std::array<float,6>& p_infs,
                              const MpiPartition* mpi_part) {
    std::vector<int> local;
    for (int idx : tree.leaf_indices())
        if (tree.nodes[idx].has_block()) local.push_back(idx);
    const int n = static_cast<int>(local.size());

    std::vector<GpuLeafGhostMeta> h(n);
    for (int ii = 0; ii < n; ++ii) {
        const int li = local[ii];
        const BlockNode& nd = tree.nodes[li];
        GpuLeafGhostMeta& m = h[ii];

        m.d_Q    = pool.d_Q(nd.block.get());
        m.cf_oct = 0;
        if (nd.parent >= 0) {
            int fc   = tree.nodes[nd.parent].first_child;
            m.cf_oct = static_cast<int8_t>(li - fc);
        }

        for (int d = 0; d < NFACES; ++d) {
            m.bc_type[d]    = static_cast<int8_t>(bc_types[d]);
            m.open_p_inf[d] = p_infs[d];
            const int ni = nd.neighbours[d];
            m.is_mpi_face[d] = mpi_is_remote(mpi_part, ni) ? 1 : 0;
            if (ni >= 0 && tree.nodes[ni].has_block() &&
                pool.has_device(tree.nodes[ni].block.get())) {
                m.d_nb[d]      = pool.d_Q(tree.nodes[ni].block.get());
                m.level_rel[d] = static_cast<int8_t>(tree.nodes[ni].level - nd.level);
            } else {
                m.d_nb[d]      = nullptr;
                m.level_rel[d] = 0;
            }
        }
    }
    n_leaves = n;
    gpu_upload_meta(d_metas, h);
}

void GpuGhostFillList::build(const BlockTree& tree, const GpuPool& pool,
                              const FaceBCArray& face_bcs,
                              const MpiPartition* mpi_part) {
    std::array<int,6>   bc_types{};
    std::array<float,6> p_infs{};
    for (int d = 0; d < NFACES; ++d) {
        bc_types[d] = bc_to_int(face_bcs[d]);
        if (std::holds_alternative<NscbcBC>(face_bcs[d]))
            p_infs[d] = static_cast<float>(std::get<NscbcBC>(face_bcs[d]).p_inf);
        // OpenBC (bc_type==2) uses fill_zero_grad on GPU; p_infs[d] is unused but
        // populated here in case OpenBC is later upgraded to bc_type==3.
        else if (std::holds_alternative<OpenBC>(face_bcs[d]))
            p_infs[d] = static_cast<float>(
                std::get<OpenBC>(face_bcs[d]).far_field_pressure);
        // else p_infs[d] stays 0.0f
    }
    build(tree, pool, bc_types, p_infs, mpi_part);
}

void GpuGhostFillList::exec(cudaStream_t stream) const {
    if (n_leaves == 0 || !d_metas) return;
    // Face fill and edge/corner fill are independent (EC reads interior, not face ghosts).
    dim3 grid_face(n_leaves, NFACES);
    k_fill_faces        <<<grid_face,  256, 0, stream>>>(d_metas);
    k_fill_edges_corners<<<n_leaves,   256, 0, stream>>>(d_metas);
}
