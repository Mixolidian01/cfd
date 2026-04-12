// DESIGN.md reference: Layer 1 — Block Tree implementation
// Fix log:
//   #1  / III  : octant helpers now from block_tree.hpp (oct_ix/iy/iz)
//   #2  / I    : BlockNode::h removed; child h lives in block->h only
//   #4         : find_or_create_node / rebuild_neighbours_recursive removed
//   #16        : refine() caches parent state before resize() to avoid UB
//   P0.7       : dead ternary removed in fill_ghosts_periodic (6 occurrences)
//   A04-fix    : edge+corner ghost fill added to fill_ghosts_periodic and
//                fill_ghosts_wall so that viscous cross-partial stencil
//                (FIX C1) never reads uninitialised ghost-of-ghost cells.
//   T11c-fix   : all three momentum components negated in wall_x/y/z lambdas
//                (no-slip image method requires u_ghost = -u_interior for
//                all velocity components, not just the wall-normal one).
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
}

// ── child geometry ────────────────────────────────────────────────────────────
void BlockTree::set_child_geometry(int parent_idx, int child_local, int child_idx) {
    const auto& par = nodes[parent_idx];
    double cell_h = par.block ? par.block->h * 0.5
                              : domain_L_ / (NB * (1 << (par.level + 1)));
    double ox = par.block ? par.block->ox : 0.0;
    double oy = par.block ? par.block->oy : 0.0;
    double oz = par.block ? par.block->oz : 0.0;
    double half = cell_h * NB;
    if (oct_ix(child_local)) ox += half;
    if (oct_iy(child_local)) oy += half;
    if (oct_iz(child_local)) oz += half;
    nodes[child_idx].block = std::make_unique<CellBlock>(ox, oy, oz, cell_h);
}

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
void BlockTree::refine(int idx) {
    assert(nodes[idx].is_leaf());

    int      saved_level  = nodes[idx].level;
    uint32_t saved_morton = nodes[idx].morton;
    CellBlock parent_data = *nodes[idx].block;

    int first = (int)nodes.size();
    nodes[idx].first_child = first;

    nodes.reserve(nodes.size() + 8);
    nodes.resize(first + 8);

    for (int oct = 0; oct < 8; ++oct) {
        auto& ch      = nodes[first + oct];
        ch.parent      = idx;
        ch.first_child = -1;
        ch.level       = saved_level + 1;
        ch.morton      = child_morton(saved_morton, oct);
        ch.neighbours.fill(-1);
        set_child_geometry(idx, oct, first + oct);
    }

    nodes[idx].block = std::make_unique<CellBlock>(parent_data);
    prolongate_to_children(idx);
    nodes[idx].block.reset();

    rebuild_neighbours();
}

// ── coarsen ───────────────────────────────────────────────────────────────────
void BlockTree::coarsen(int parent_idx) {
    int fc = nodes[parent_idx].first_child;
    assert(fc >= 0);
    for (int oct = 0; oct < 8; ++oct)
        assert(nodes[fc + oct].is_leaf());
    assert(fc + 8 == (int)nodes.size() &&
           "coarsen(): children must be the last 8 nodes.");

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
    nodes.resize(fc);
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
void BlockTree::rebuild_neighbours()
{
    for (auto& nd : nodes) nd.neighbours.fill(-1);

    auto leaves = leaf_indices();
    if (leaves.empty()) return;

    std::unordered_map<uint64_t, int> lm_map;
    lm_map.reserve(leaves.size() * 2);
    for (int li : leaves) {
        auto& nd = nodes[li];
        uint64_t key = ((uint64_t)nd.level << 32) | nd.morton;
        lm_map[key] = li;
    }

    auto morton_face_neighbour = [](uint32_t code, int level,
                                    int axis, int delta) -> uint32_t {
        uint32_t mx, my, mz;
        morton_decode(code, mx, my, mz);
        uint32_t max_coord = (1u << level) - 1u;
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

    static constexpr int face_axis[NFACES]  = { 0, 0, 1, 1, 2, 2 };
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1 };

    for (int ai : leaves) {
        auto& a   = nodes[ai];
        int   lev = a.level;
        for (int d = 0; d < NFACES; ++d) {
            if (a.neighbours[d] >= 0) continue;
            int   axis  = face_axis[d];
            int   delta = face_delta[d];
            uint32_t nb_code = morton_face_neighbour(a.morton, lev, axis, delta);
            if (nb_code != UINT32_MAX) {
                uint64_t key = ((uint64_t)lev << 32) | nb_code;
                auto it = lm_map.find(key);
                if (it != lm_map.end()) {
                    int bi = it->second;
                    a.neighbours[d]             = bi;
                    nodes[bi].neighbours[d ^ 1] = ai;
                    continue;
                }
            }
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

// =============================================================================
// Ghost fill helpers
// =============================================================================
//
// The viscous operator (FIX C1) uses 12 edge-diagonal neighbours such as
// prim(i+1,j+1,k), requiring a complete NG=1 halo over all 26 neighbours:
//   6 face ghosts  +  12 edge ghosts  +  8 corner ghosts.
//
// Fill order:
//   1. Faces    (read interior cells only)
//   2. Edges    (read face ghosts already filled in step 1)
//   3. Corners  (read face/edge ghosts already filled in steps 1–2)
// =============================================================================

// ── fill_ghosts_periodic ──────────────────────────────────────────────────────
void BlockTree::fill_ghosts_periodic() {
    auto leaves = leaf_indices();
    for (int li : leaves) {
        auto& nd  = nodes[li];
        auto& blk = *nd.block;

        auto copy_cell = [&](int gi, int gj, int gk,
                             int si, int sj, int sk,
                             const CellBlock& src) noexcept {
            for (int v = 0; v < NVAR; ++v)
                blk.Q[v][cell_idx(gi,gj,gk)] = src.Q[v][cell_idx(si,sj,sk)];
        };

        // ── 1. Face ghosts ───────────────────────────────────────────────
        {
            int ni = nd.neighbours[XMINUS];
            const CellBlock& src = (ni>=0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k=ilo();k<=ihi();++k)
            for (int j=ilo();j<=ihi();++j)
                copy_cell(0,j,k, ihi(),j,k, src);
        }
        {
            int ni = nd.neighbours[XPLUS];
            const CellBlock& src = (ni>=0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k=ilo();k<=ihi();++k)
            for (int j=ilo();j<=ihi();++j)
                copy_cell(NB2-1,j,k, ilo(),j,k, src);
        }
        {
            int ni = nd.neighbours[YMINUS];
            const CellBlock& src = (ni>=0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k=ilo();k<=ihi();++k)
            for (int i=ilo();i<=ihi();++i)
                copy_cell(i,0,k, i,ihi(),k, src);
        }
        {
            int ni = nd.neighbours[YPLUS];
            const CellBlock& src = (ni>=0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int k=ilo();k<=ihi();++k)
            for (int i=ilo();i<=ihi();++i)
                copy_cell(i,NB2-1,k, i,ilo(),k, src);
        }
        {
            int ni = nd.neighbours[ZMINUS];
            const CellBlock& src = (ni>=0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int j=ilo();j<=ihi();++j)
            for (int i=ilo();i<=ihi();++i)
                copy_cell(i,j,0, i,j,ihi(), src);
        }
        {
            int ni = nd.neighbours[ZPLUS];
            const CellBlock& src = (ni>=0 && nodes[ni].block) ? *nodes[ni].block : blk;
            for (int j=ilo();j<=ihi();++j)
            for (int i=ilo();i<=ihi();++i)
                copy_cell(i,j,NB2-1, i,j,ilo(), src);
        }

        // ── 2. Edge ghosts ─────────────────────────────────────────────
        for (int k=ilo();k<=ihi();++k) {
            copy_cell(0,    0,    k,  ihi(),ihi(),k, blk);
            copy_cell(NB2-1,0,    k,  ilo(),ihi(),k, blk);
            copy_cell(0,    NB2-1,k,  ihi(),ilo(),k, blk);
            copy_cell(NB2-1,NB2-1,k,  ilo(),ilo(),k, blk);
        }
        for (int j=ilo();j<=ihi();++j) {
            copy_cell(0,    j,0,     ihi(),j,ihi(), blk);
            copy_cell(NB2-1,j,0,     ilo(),j,ihi(), blk);
            copy_cell(0,    j,NB2-1, ihi(),j,ilo(), blk);
            copy_cell(NB2-1,j,NB2-1, ilo(),j,ilo(), blk);
        }
        for (int i=ilo();i<=ihi();++i) {
            copy_cell(i,0,    0,     i,ihi(),ihi(), blk);
            copy_cell(i,NB2-1,0,     i,ilo(),ihi(), blk);
            copy_cell(i,0,    NB2-1, i,ihi(),ilo(), blk);
            copy_cell(i,NB2-1,NB2-1, i,ilo(),ilo(), blk);
        }

        // ── 3. Corner ghosts ──────────────────────────────────────────
        copy_cell(0,    0,    0,     ihi(),ihi(),ihi(), blk);
        copy_cell(NB2-1,0,    0,     ilo(),ihi(),ihi(), blk);
        copy_cell(0,    NB2-1,0,     ihi(),ilo(),ihi(), blk);
        copy_cell(NB2-1,NB2-1,0,     ilo(),ilo(),ihi(), blk);
        copy_cell(0,    0,    NB2-1, ihi(),ihi(),ilo(), blk);
        copy_cell(NB2-1,0,    NB2-1, ilo(),ihi(),ilo(), blk);
        copy_cell(0,    NB2-1,NB2-1, ihi(),ilo(),ilo(), blk);
        copy_cell(NB2-1,NB2-1,NB2-1, ilo(),ilo(),ilo(), blk);
    }
}

// ── fill_ghosts_wall (no-slip adiabatic) ─────────────────────────────────────
//
// No-slip image method: u_ghost = -u_interior for ALL velocity components,
// so that u_wall = (u_ghost + u_interior)/2 = 0.
// Adiabatic: rho and E are copied (zero normal gradient).
void BlockTree::fill_ghosts_wall() {
    auto leaves = leaf_indices();
    for (int li : leaves) {
        auto& nd  = nodes[li];
        auto& blk = *nd.block;

        // All three momentum components negated for no-slip (T11c fix).
        auto wall_x = [&](int gi, int mi) noexcept {
            for (int k=ilo();k<=ihi();++k)
            for (int j=ilo();j<=ihi();++j) {
                blk.rho (gi,j,k) =  blk.rho (mi,j,k);
                blk.rhou(gi,j,k) = -blk.rhou(mi,j,k);
                blk.rhov(gi,j,k) = -blk.rhov(mi,j,k);
                blk.rhow(gi,j,k) = -blk.rhow(mi,j,k);
                blk.E   (gi,j,k) =  blk.E   (mi,j,k);
            }
        };
        auto wall_y = [&](int gj, int mj) noexcept {
            for (int k=ilo();k<=ihi();++k)
            for (int i=ilo();i<=ihi();++i) {
                blk.rho (i,gj,k) =  blk.rho (i,mj,k);
                blk.rhou(i,gj,k) = -blk.rhou(i,mj,k);
                blk.rhov(i,gj,k) = -blk.rhov(i,mj,k);
                blk.rhow(i,gj,k) = -blk.rhow(i,mj,k);
                blk.E   (i,gj,k) =  blk.E   (i,mj,k);
            }
        };
        auto wall_z = [&](int gk, int mk) noexcept {
            for (int j=ilo();j<=ihi();++j)
            for (int i=ilo();i<=ihi();++i) {
                blk.rho (i,j,gk) =  blk.rho (i,j,mk);
                blk.rhou(i,j,gk) = -blk.rhou(i,j,mk);
                blk.rhov(i,j,gk) = -blk.rhov(i,j,mk);
                blk.rhow(i,j,gk) = -blk.rhow(i,j,mk);
                blk.E   (i,j,gk) =  blk.E   (i,j,mk);
            }
        };

        const bool xm = (nd.neighbours[XMINUS] < 0);
        const bool xp = (nd.neighbours[XPLUS]  < 0);
        const bool ym = (nd.neighbours[YMINUS] < 0);
        const bool yp = (nd.neighbours[YPLUS]  < 0);
        const bool zm = (nd.neighbours[ZMINUS] < 0);
        const bool zp = (nd.neighbours[ZPLUS]  < 0);

        if (xm) wall_x(0,     ilo());
        if (xp) wall_x(NB2-1, ihi());
        if (ym) wall_y(0,     ilo());
        if (yp) wall_y(NB2-1, ihi());
        if (zm) wall_z(0,     ilo());
        if (zp) wall_z(NB2-1, ihi());

        // ── Edge ghosts ────────────────────────────────────────────────
        for (int k=ilo();k<=ihi();++k) {
            if (xm||ym) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    0,    k)] = blk.Q[v][cell_idx(0,    ilo(),k)];
            if (xp||ym) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,0,    k)] = blk.Q[v][cell_idx(NB2-1,ilo(),k)];
            if (xm||yp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    NB2-1,k)] = blk.Q[v][cell_idx(0,    ihi(),k)];
            if (xp||yp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,NB2-1,k)] = blk.Q[v][cell_idx(NB2-1,ihi(),k)];
        }
        for (int j=ilo();j<=ihi();++j) {
            if (xm||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    j,0    )] = blk.Q[v][cell_idx(0,    j,ilo())];
            if (xp||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,j,0    )] = blk.Q[v][cell_idx(NB2-1,j,ilo())];
            if (xm||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    j,NB2-1)] = blk.Q[v][cell_idx(0,    j,ihi())];
            if (xp||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,j,NB2-1)] = blk.Q[v][cell_idx(NB2-1,j,ihi())];
        }
        for (int i=ilo();i<=ihi();++i) {
            if (ym||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(i,0,    0    )] = blk.Q[v][cell_idx(i,0,    ilo())];
            if (yp||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(i,NB2-1,0    )] = blk.Q[v][cell_idx(i,NB2-1,ilo())];
            if (ym||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(i,0,    NB2-1)] = blk.Q[v][cell_idx(i,0,    ihi())];
            if (yp||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(i,NB2-1,NB2-1)] = blk.Q[v][cell_idx(i,NB2-1,ihi())];
        }

        // ── Corner ghosts ────────────────────────────────────────────────
        if (xm||ym||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    0,    0    )] = blk.Q[v][cell_idx(0,    0,    ilo())];
        if (xp||ym||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,0,    0    )] = blk.Q[v][cell_idx(NB2-1,0,    ilo())];
        if (xm||yp||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    NB2-1,0    )] = blk.Q[v][cell_idx(0,    NB2-1,ilo())];
        if (xp||yp||zm) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,NB2-1,0    )] = blk.Q[v][cell_idx(NB2-1,NB2-1,ilo())];
        if (xm||ym||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    0,    NB2-1)] = blk.Q[v][cell_idx(0,    0,    ihi())];
        if (xp||ym||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,0,    NB2-1)] = blk.Q[v][cell_idx(NB2-1,0,    ihi())];
        if (xm||yp||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(0,    NB2-1,NB2-1)] = blk.Q[v][cell_idx(0,    NB2-1,ihi())];
        if (xp||yp||zp) for (int v=0;v<NVAR;++v) blk.Q[v][cell_idx(NB2-1,NB2-1,NB2-1)] = blk.Q[v][cell_idx(NB2-1,NB2-1,ihi())];
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
