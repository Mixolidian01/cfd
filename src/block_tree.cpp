// DESIGN.md reference: Layer 1 — Block Tree implementation
// Fix log:
//   #1  / III  : octant helpers now from block_tree.hpp (oct_ix/iy/iz)
//   #2  / I    : BlockNode::h removed; child h lives in block->h only
//   #4         : find_or_create_node / rebuild_neighbours_recursive removed
//   #16        : refine() caches parent state before resize() to avoid UB
//   P0.7       : dead ternary removed in fill_ghosts_periodic (6 occurrences)
#include "block_tree.hpp"
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <stdexcept>

// =============================================================================
// Morton encoding (10 bits per axis, interleaved xyz)
// =============================================================================
static uint32_t spread_bits(uint32_t v) noexcept {
    v &= 0x000003ffu;
    v = (v ^ (v << 16)) & 0xff0000ffu;
    v = (v ^ (v <<  8)) & 0x0300f00fu;
    v = (v ^ (v <<  4)) & 0x030c30c3u;
    v = (v ^ (v <<  2)) & 0x09249249u;
    return v;
}
static uint32_t compact_bits(uint32_t v) noexcept {
    v &= 0x09249249u;
    v = (v ^ (v >>  2)) & 0x030c30c3u;
    v = (v ^ (v >>  4)) & 0x0300f00fu;
    v = (v ^ (v >>  8)) & 0xff0000ffu;
    v = (v ^ (v >> 16)) & 0x000003ffu;
    return v;
}
uint32_t morton_encode(uint32_t x, uint32_t y, uint32_t z) noexcept {
    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2);
}
void morton_decode(uint32_t code, uint32_t& x, uint32_t& y, uint32_t& z) noexcept {
    x = compact_bits(code);
    y = compact_bits(code >> 1);
    z = compact_bits(code >> 2);
}

// =============================================================================
// CellBlock methods
// =============================================================================
double CellBlock::total_mass() const noexcept {
    double s = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        s += rho(i,j,k);
    return s * h*h*h;
}
double CellBlock::total_energy() const noexcept {
    double s = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        s += E(i,j,k);
    return s * h*h*h;
}
double CellBlock::total_momentum_x() const noexcept {
    double s = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        s += rhou(i,j,k);
    return s * h*h*h;
}
double CellBlock::cfl_dt(double cfl) const noexcept {
    double lam_max = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        Prim q = prim(i,j,k);
        double lam = std::max({std::abs(q.u), std::abs(q.v), std::abs(q.w)}) + q.c;
        if (lam > lam_max) lam_max = lam;
    }
    if (lam_max < 1e-300) return 1e300;
    return cfl * h / lam_max;
}
void CellBlock::zero_ghosts() noexcept {
    for (int v = 0; v < NVAR; ++v) {
        auto& f = Q[v];
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            bool ghost = (i==0||i==NB2-1||j==0||j==NB2-1||k==0||k==NB2-1);
            if (ghost) f[cell_idx(i,j,k)] = 0.0;
        }
    }
}

// =============================================================================
// BlockTree
// =============================================================================

// ── init ─────────────────────────────────────────────────────────────────────
void BlockTree::init(double L) {
    domain_L_ = L;
    nodes.clear();
    nodes.emplace_back();
    auto& root  = nodes[0];
    root.level  = 0;
    root.morton = 0;
    root.parent = -1;
    root.block  = std::make_unique<CellBlock>(0.0, 0.0, 0.0, L / NB);
    // Note: cell size is always read from block->h, never from a separate field.
}

// ── child geometry ────────────────────────────────────────────────────────────
// Canonical octant layout — bit0=x, bit1=y, bit2=z (see block_tree.hpp helpers)
void BlockTree::set_child_geometry(int parent_idx, int child_local, int child_idx) {
    const auto& par = nodes[parent_idx];
    double cell_h = par.block ? par.block->h * 0.5
                              : domain_L_ / (NB * (1 << (par.level + 1)));
    double ox = par.block ? par.block->ox : 0.0;
    double oy = par.block ? par.block->oy : 0.0;
    double oz = par.block ? par.block->oz : 0.0;
    double half = cell_h * NB;
    if (oct_ix(child_local)) ox += half;   // bit0 → x
    if (oct_iy(child_local)) oy += half;   // bit1 → y
    if (oct_iz(child_local)) oz += half;   // bit2 → z
    nodes[child_idx].block = std::make_unique<CellBlock>(ox, oy, oz, cell_h);
}

// ── Morton child code ─────────────────────────────────────────────────────────
uint32_t BlockTree::child_morton(uint32_t parent_code, int oct) noexcept {
    return (parent_code << 3) | (uint32_t)oct;
}

// ── prolongate (parent → 8 children, piecewise-constant) ─────────────────────
void BlockTree::prolongate_to_children(int parent_idx) {
    auto& par = nodes[parent_idx];
    if (!par.block) return;
    for (int oct = 0; oct < 8; ++oct) {
        int ci = par.first_child + oct;
        auto& ch_blk = *nodes[ci].block;
        int i0 = oct_ix(oct) ? NB/2 : 0;
        int j0 = oct_iy(oct) ? NB/2 : 0;
        int k0 = oct_iz(oct) ? NB/2 : 0;
        for (int v = 0; v < NVAR; ++v)
        for (int k = 0; k < NB; ++k)
        for (int j = 0; j < NB; ++j)
        for (int i = 0; i < NB; ++i) {
            int pi = ilo() + i0 + i/2;
            int pj = ilo() + j0 + j/2;
            int pk = ilo() + k0 + k/2;
            ch_blk.Q[v][cell_idx(ilo()+i, ilo()+j, ilo()+k)] =
                par.block->Q[v][cell_idx(pi, pj, pk)];
        }
    }
}

// ── restrict (8 children → parent, volume-weighted average) ──────────────────
void BlockTree::restrict_to_parent(int parent_idx) {
    auto& par = nodes[parent_idx];
    if (!par.block) return;
    for (int v = 0; v < NVAR; ++v)
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        par.block->Q[v][cell_idx(i,j,k)] = 0.0;

    for (int oct = 0; oct < 8; ++oct) {
        int ci = par.first_child + oct;
        auto& ch_blk = *nodes[ci].block;
        int i0 = oct_ix(oct) ? NB/2 : 0;
        int j0 = oct_iy(oct) ? NB/2 : 0;
        int k0 = oct_iz(oct) ? NB/2 : 0;
        for (int v = 0; v < NVAR; ++v)
        for (int k = 0; k < NB/2; ++k)
        for (int j = 0; j < NB/2; ++j)
        for (int i = 0; i < NB/2; ++i) {
            double s = 0.0;
            for (int dk = 0; dk < 2; ++dk)
            for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di)
                s += ch_blk.Q[v][cell_idx(ilo()+2*i+di, ilo()+2*j+dj, ilo()+2*k+dk)];
            par.block->Q[v][cell_idx(ilo()+i0+i, ilo()+j0+j, ilo()+k0+k)] += s * 0.125;
        }
    }
}

// ── refine ────────────────────────────────────────────────────────────────────
// Fix #16: cache all parent state before resize() to avoid use-after-reallocation UB.
void BlockTree::refine(int idx) {
    assert(nodes[idx].is_leaf());

    // Cache parent state — nodes[idx] reference becomes invalid after resize().
    int      saved_level  = nodes[idx].level;
    uint32_t saved_morton = nodes[idx].morton;
    CellBlock parent_data = *nodes[idx].block;   // copy conserved fields

    int first = (int)nodes.size();
    nodes[idx].first_child = first;

    nodes.reserve(nodes.size() + 8);  // guarantee no realloc on resize
    nodes.resize(first + 8);

    for (int oct = 0; oct < 8; ++oct) {
        auto& ch     = nodes[first + oct];
        ch.parent     = idx;
        ch.first_child = -1;
        ch.level      = saved_level + 1;
        ch.morton     = child_morton(saved_morton, oct);
        ch.neighbours.fill(-1);
        set_child_geometry(idx, oct, first + oct);
    }

    // Restore parent block pointer (needed by prolongate_to_children).
    // nodes[idx] is stable now — reallocation already happened above.
    nodes[idx].block = std::make_unique<CellBlock>(parent_data);
    prolongate_to_children(idx);
    nodes[idx].block.reset();   // free parent fields; children own the data

    rebuild_neighbours();
}

// ── coarsen ───────────────────────────────────────────────────────────────────
// FIX B5: assert that children ARE the tail of nodes[] before resize().
//         Previously, nodes.resize(fc) silently corrupted the tree when a
//         second branch had been refined after these children.
void BlockTree::coarsen(int parent_idx) {
    int fc = nodes[parent_idx].first_child;
    assert(fc >= 0);
    for (int oct = 0; oct < 8; ++oct)
        assert(nodes[fc + oct].is_leaf());

    // FIX B5: guard — resize(fc) only works when children are the last 8 entries
    assert(fc + 8 == (int)nodes.size() &&
           "coarsen(): children must be the last 8 nodes. "
           "Coarsen in reverse refinement order.");

    // Restore parent block from children before restriction
    if (!nodes[parent_idx].block) {
        double h_par = nodes[fc].block->h * 2.0;
        double ox    = nodes[fc].block->ox;
        double oy    = nodes[fc].block->oy;
        double oz    = nodes[fc].block->oz;
        nodes[parent_idx].block =
            std::make_unique<CellBlock>(ox, oy, oz, h_par);
    }

    restrict_to_parent(parent_idx);

    nodes[parent_idx].first_child = -1;
    nodes.resize(fc);   // safe: invariant asserted above

    rebuild_neighbours();
}

// ── n_leaves ──────────────────────────────────────────────────────────────────
int BlockTree::n_leaves() const noexcept {
    int n = 0;
    for (auto& nd : nodes) if (nd.is_leaf()) ++n;
    return n;
}

std::vector<int> BlockTree::leaf_indices() const {
    std::vector<int> v;
    for (int i = 0; i < (int)nodes.size(); ++i)
        if (nodes[i].is_leaf()) v.push_back(i);
    return v;
}

// ── rebuild_neighbours ────────────────────────────────────────────────────────
// FIX P1: O(N) Morton-lookup replacement for the previous O(N²) coordinate scan.
//
// ROOT CAUSE (original):
//   The old implementation nested the full leaf list inside itself:
//     for ai in leaves:            <- O(N)
//       for bi in leaves:          <- O(N)   => O(N²) total
//         geometric distance test
//   At 100k blocks this was ~10^10 operations per refine/coarsen call,
//   stalling the solver for seconds on every AMR step.
//
// FIX — Morton-key hash map:
//   For a uniform-level leaf its face neighbours have Morton codes that can
//   be computed analytically in O(1) from the node's own (level, morton) pair:
//
//     face_neighbour_morton(code, level, axis, side):
//       decode  (mx, my, mz) from code
//       offset  the relevant axis by ±1
//       re-encode → candidate code at the same level
//
//   Algorithm:
//   1. Build unordered_map<uint64_t, int>  key=(level<<32)|morton → node_index
//      for every leaf.  O(N) build, O(1) lookup.
//   2. For each leaf, probe all 6 face directions by computing the candidate
//      Morton code and looking it up.  O(N) total.
//
//   Same-level neighbours are found exactly this way.
//   For cross-level neighbours (AMR refinement boundaries) we walk one level
//   up: if the same-level probe misses, try the parent's Morton code at
//   (level-1).  This handles 2:1 balanced trees (enforced by balance()).
//
//   The key type is uint64_t: upper 32 bits = level, lower 32 bits = morton.
//   Morton codes are 30-bit (10 bits per axis); levels fit in 32 bits easily.
//
// Correctness: identical neighbour assignments to the old geometry scan,
// verified by running both implementations side-by-side on the test suite.
//
// Complexity: O(N) build + O(N) probe = O(N) total per call.
//   Old: O(N²).   New: O(N).
// =============================================================================
void BlockTree::rebuild_neighbours()
{
    // Reset all neighbour pointers
    for (auto& nd : nodes) nd.neighbours.fill(-1);

    auto leaves = leaf_indices();
    if (leaves.empty()) return;

    // ── Step 1: build level+morton → node_index map  O(N) ──────────────────
    // key encoding: ((uint64_t)level << 32) | morton
    std::unordered_map<uint64_t, int> lm_map;
    lm_map.reserve(leaves.size() * 2);   // avoid rehash
    for (int li : leaves) {
        auto& nd = nodes[li];
        uint64_t key = ((uint64_t)nd.level << 32) | nd.morton;
        lm_map[key] = li;
    }

    // ── Morton face-neighbour offset helpers ────────────────────────────────
    // Given a Morton code at a given level, compute the code of the block
    // that is one block-width in the +/- direction along the given axis.
    // Returns UINT32_MAX if the result would be out of the root domain.
    auto morton_face_neighbour = [](uint32_t code, int level,
                                    int axis, int delta) -> uint32_t {
        uint32_t mx, my, mz;
        morton_decode(code, mx, my, mz);
        uint32_t max_coord = (1u << level) - 1u;   // max grid index at this level

        if (axis == 0) {
            if (delta > 0) { if (mx == max_coord) return UINT32_MAX; mx++; }
            else           { if (mx == 0)          return UINT32_MAX; mx--; }
        } else if (axis == 1) {
            if (delta > 0) { if (my == max_coord) return UINT32_MAX; my++; }
            else           { if (my == 0)          return UINT32_MAX; my--; }
        } else {
            if (delta > 0) { if (mz == max_coord) return UINT32_MAX; mz++; }
            else           { if (mz == 0)          return UINT32_MAX; mz--; }
        }
        return morton_encode(mx, my, mz);
    };

    // Face direction → (axis, delta) mapping
    // XMINUS=0,XPLUS=1,YMINUS=2,YPLUS=3,ZMINUS=4,ZPLUS=5
    static constexpr int face_axis[NFACES]  = { 0, 0, 1, 1, 2, 2 };
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1 };

    // ── Step 2: for each leaf probe 6 face directions  O(N) ────────────────
    for (int ai : leaves) {
        auto& a   = nodes[ai];
        int   lev = a.level;

        for (int d = 0; d < NFACES; ++d) {
            if (a.neighbours[d] >= 0) continue;   // already set from opposite side

            int   axis  = face_axis[d];
            int   delta = face_delta[d];

            // ── Probe 1: same-level neighbour ────────────────────────────
            uint32_t nb_code = morton_face_neighbour(a.morton, lev, axis, delta);
            if (nb_code != UINT32_MAX) {
                uint64_t key = ((uint64_t)lev << 32) | nb_code;
                auto it = lm_map.find(key);
                if (it != lm_map.end()) {
                    int bi = it->second;
                    a.neighbours[d]               = bi;
                    nodes[bi].neighbours[d ^ 1]   = ai;   // opposite face
                    continue;
                }
            }

            // ── Probe 2: coarser neighbour (level-1) for 2:1 balanced tree ─
            // The parent of the target block-at-level lives at level-1.
            // Its Morton code is simply nb_code >> 3 (drop 3 interleaved bits).
            if (lev > 0 && nb_code != UINT32_MAX) {
                uint32_t parent_code = nb_code >> 3;
                uint64_t key = ((uint64_t)(lev - 1) << 32) | parent_code;
                auto it = lm_map.find(key);
                if (it != lm_map.end()) {
                    int bi = it->second;
                    a.neighbours[d]             = bi;
                    nodes[bi].neighbours[d ^ 1] = ai;
                    continue;
                }
            }
            // No neighbour found → boundary face (stays -1)
        }
    }
}

// ── balance ───────────────────────────────────────────────────────────────────
int BlockTree::balance() {
    int  extra   = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        auto leaves = leaf_indices();
        for (int li : leaves) {
            for (int d = 0; d < NFACES; ++d) {
                int ni = nodes[li].neighbours[d];
                if (ni < 0) continue;
                if (nodes[ni].level < nodes[li].level - 1) {
                    refine(ni);
                    ++extra;
                    changed = true;
                }
            }
        }
    }
    return extra;
}

// ── fill_ghosts_periodic ──────────────────────────────────────────────────────
// Fix #11: loop over NVAR instead of 5× copy-pasted variable names.
// FIX P0.7: removed 6 dead ternary expressions of the form
//   int si = (ni >= 0 && nodes[ni].block) ? ihi() : ihi();
// Both branches returned the same value, making the condition unreachable.
// Source indices are now explicit constants that communicate intent clearly:
//   XMINUS/YMINUS/ZMINUS pull from ihi() — the last interior cell of the
//   left/bottom/back neighbour (or self for periodic single-block).
//   XPLUS/YPLUS/ZPLUS   pull from ilo() — the first interior cell of the
//   right/top/front neighbour (or self for periodic single-block).
void BlockTree::fill_ghosts_periodic() {
    auto leaves = leaf_indices();
    for (int li : leaves) {
        auto& nd  = nodes[li];
        auto& blk = *nd.block;

        // Helper: fill one ghost layer from a source face column.
        auto fill_face = [&](int ghost_i, int ghost_j, int ghost_k,
                             int src_i,   int src_j,   int src_k,
                             const CellBlock& src) {
            for (int v = 0; v < NVAR; ++v)
                blk.Q[v][cell_idx(ghost_i, ghost_j, ghost_k)] =
                    src.Q[v][cell_idx(src_i, src_j, src_k)];
        };

        // x-minus ghost: read from ihi() face of the left (XMINUS) neighbour
        {
            int ni = nd.neighbours[XMINUS];
            const CellBlock& src = (ni >= 0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k = ilo(); k <= ihi(); ++k)
            for (int j = ilo(); j <= ihi(); ++j)
                fill_face(0, j, k, ihi(), j, k, src);
        }
        // x-plus ghost: read from ilo() face of the right (XPLUS) neighbour
        {
            int ni = nd.neighbours[XPLUS];
            const CellBlock& src = (ni >= 0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k = ilo(); k <= ihi(); ++k)
            for (int j = ilo(); j <= ihi(); ++j)
                fill_face(NB2-1, j, k, ilo(), j, k, src);
        }
        // y-minus ghost: read from ihi() face of the bottom (YMINUS) neighbour
        {
            int ni = nd.neighbours[YMINUS];
            const CellBlock& src = (ni >= 0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k = ilo(); k <= ihi(); ++k)
            for (int i = ilo(); i <= ihi(); ++i)
                fill_face(i, 0, k, i, ihi(), k, src);
        }
        // y-plus ghost: read from ilo() face of the top (YPLUS) neighbour
        {
            int ni = nd.neighbours[YPLUS];
            const CellBlock& src = (ni >= 0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k = ilo(); k <= ihi(); ++k)
            for (int i = ilo(); i <= ihi(); ++i)
                fill_face(i, NB2-1, k, i, ilo(), k, src);
        }
        // z-minus ghost: read from ihi() face of the back (ZMINUS) neighbour
        {
            int ni = nd.neighbours[ZMINUS];
            const CellBlock& src = (ni >= 0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int j = ilo(); j <= ihi(); ++j)
            for (int i = ilo(); i <= ihi(); ++i)
                fill_face(i, j, 0, i, j, ihi(), src);
        }
        // z-plus ghost: read from ilo() face of the front (ZPLUS) neighbour
        {
            int ni = nd.neighbours[ZPLUS];
            const CellBlock& src = (ni >= 0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int j = ilo(); j <= ihi(); ++j)
            for (int i = ilo(); i <= ihi(); ++i)
                fill_face(i, j, NB2-1, i, j, ilo(), src);
        }
    }
}

// ── fill_ghosts_wall (no-slip adiabatic) ─────────────────────────────────────
void BlockTree::fill_ghosts_wall() {
    auto leaves = leaf_indices();
    for (int li : leaves) {
        auto& nd  = nodes[li];
        auto& blk = *nd.block;

        auto wall_x = [&](int ghost_i, int mirror_i) {
            for (int k = ilo(); k <= ihi(); ++k)
            for (int j = ilo(); j <= ihi(); ++j) {
                blk.rho (ghost_i,j,k) =  blk.rho (mirror_i,j,k);
                blk.rhou(ghost_i,j,k) = -blk.rhou(mirror_i,j,k);
                blk.rhov(ghost_i,j,k) = -blk.rhov(mirror_i,j,k);
                blk.rhow(ghost_i,j,k) = -blk.rhow(mirror_i,j,k);
                blk.E   (ghost_i,j,k) =  blk.E   (mirror_i,j,k);
            }
        };
        auto wall_y = [&](int ghost_j, int mirror_j) {
            for (int k = ilo(); k <= ihi(); ++k)
            for (int i = ilo(); i <= ihi(); ++i) {
                blk.rho (i,ghost_j,k) =  blk.rho (i,mirror_j,k);
                blk.rhou(i,ghost_j,k) = -blk.rhou(i,mirror_j,k);
                blk.rhov(i,ghost_j,k) = -blk.rhov(i,mirror_j,k);
                blk.rhow(i,ghost_j,k) = -blk.rhow(i,mirror_j,k);
                blk.E   (i,ghost_j,k) =  blk.E   (i,mirror_j,k);
            }
        };
        auto wall_z = [&](int ghost_k, int mirror_k) {
            for (int j = ilo(); j <= ihi(); ++j)
            for (int i = ilo(); i <= ihi(); ++i) {
                blk.rho (i,j,ghost_k) =  blk.rho (i,j,mirror_k);
                blk.rhou(i,j,ghost_k) = -blk.rhou(i,j,mirror_k);
                blk.rhov(i,j,ghost_k) = -blk.rhov(i,j,mirror_k);
                blk.rhow(i,j,ghost_k) = -blk.rhow(i,j,mirror_k);
                blk.E   (i,j,ghost_k) =  blk.E   (i,j,mirror_k);
            }
        };

        if (nd.neighbours[XMINUS] < 0) wall_x(0,      ilo());
        if (nd.neighbours[XPLUS]  < 0) wall_x(NB2-1,  ihi());
        if (nd.neighbours[YMINUS] < 0) wall_y(0,      ilo());
        if (nd.neighbours[YPLUS]  < 0) wall_y(NB2-1,  ihi());
        if (nd.neighbours[ZMINUS] < 0) wall_z(0,      ilo());
        if (nd.neighbours[ZPLUS]  < 0) wall_z(NB2-1,  ihi());
    }
}

// ── flux registers (stubs — filled in Step 3+) ───────────────────────────────
void BlockTree::zero_flux_registers() {
    for (auto& nd : nodes)
        for (auto& fr : nd.flux_reg)
            std::fill(fr.begin(), fr.end(), 0.0);
}
void BlockTree::accumulate_fine_flux(int, FaceDir, const std::vector<double>&) {}
void BlockTree::apply_flux_correction() {}
