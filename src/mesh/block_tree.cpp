// DESIGN.md reference: Layer 1 — Block Tree implementation
// Fix log:
//   #1  / III  : octant helpers now from block_tree.hpp (oct_ix/iy/iz)
//   #16        : refine() caches parent state before resize() to avoid UB
//   P0.7       : dead ternary removed in fill_ghosts_periodic
//   A04-fix    : edge+corner ghost fill added to fill_ghosts_periodic and
//                fill_ghosts_wall (viscous cross-partial stencil safety)
//   T11c-fix   : all three momentum components negated in wall lambdas
//   P1.1       : coarsen() now uses free-list; alloc_node/free_node added
//   P1.2       : balance() uses work-queue (std::deque) — no stale snapshots
//   P1.3       : fill_ghosts_periodic / _wall dispatch fill_cf_ghosts for CF faces
//   P1.4       : accumulate_fine_flux / apply_flux_correction implemented
//   P1.6       : leaf_indices() returns cached vector; dirty flag set by
//                refine(), coarsen(), rebuild_neighbours()
//   build-fix  : #include paths corrected to ../include/ prefix
//   A05-fix    : alloc_node_group(8) guarantees contiguous child allocation
//                so first_child+oct is always valid after free-list reuse
//   A05-fix2   : averaged coarse ghost fill when ni is finer than nd;
//                apply_flux_correction axis=0/1 ck/ci indexing corrected
//   A05-fix3   : apply_flux_correction sign corrected: +face subtracts fine
//                flux, -face adds it, matching dQ=(dt/h)*(F_left-F_right)
//   A05-fix4   : fill_coarse_ghost_from_fine: removed ix_fixed/iy_fixed/iz_fixed
//                -1 sentinel.  -1 & 1 == 1 in C++, silently selecting the wrong
//                fine block on y-face and z-face CF interfaces → mass leak in A05.
//                Now uses `side` directly in each axis branch.
//   A05-fix5   : fill_coarse_ghost_from_fine: guard against stale `ni` pointer.
//                rebuild_neighbours() only stores the LAST fine leaf that sets
//                nodes[coarse].neighbours[d^1], so after a regrid `ni` may belong
//                to a different refinement patch (different parent → wrong
//                first_child → wrong fine blocks read for all NB×NB ghost positions).
//                Fix: check that nodes[first_child].block->h == coarse_blk.h/2
//                before proceeding; skip silently if sizes disagree.
//   A05-fix6   : accumulate_fine_flux: store at nodes[ni].flux_reg[d^1]
//                (= opposite(d)).  d is the FINE→COARSE direction; apply_flux_correction
//                reads coarse.flux_reg[d_coarse_to_fine] = coarse.flux_reg[d^1].
//                The prior A05-fix6 incorrectly used flux_reg[d] (fine→coarse slot),
//                which apply_flux_correction never reads, silently discarding all
//                fine fluxes and leaving the coarse budget uncorrected (~2.69e-8 A05).
#include "mesh/block_tree.hpp"
#include "mpi/mpi_comm.hpp"
#include "mesh/amr_operators.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <deque>

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
template<class Fn>
static double interior_sum(const CellBlock& b, Fn get) noexcept {
    double s = 0.0;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        s += get(i, j, k);
    return s * b.h * b.hy * b.hz;
}
double CellBlock::total_mass()       const noexcept { return interior_sum(*this, [&](int i,int j,int k){ return rho (i,j,k); }); }
double CellBlock::total_energy()     const noexcept { return interior_sum(*this, [&](int i,int j,int k){ return E   (i,j,k); }); }
double CellBlock::total_momentum_x() const noexcept { return interior_sum(*this, [&](int i,int j,int k){ return rhou(i,j,k); }); }
double CellBlock::cfl_dt(double cfl) const noexcept {
    // Convective CFL stability: dt_conv = cfl * h / max(|u|+c)
    // Viscous  CFL stability:  dt_visc = h² / (2 * C_visc * max(µ/ρ))
    //   C_visc = max(4/3, γ/Pr)  — momentum vs. thermal diffusivity
    // Take the minimum of both constraints.
    static constexpr double C_VISC = (GAMMA / PR > 4.0/3.0) ? GAMMA / PR : 4.0/3.0;
    double lam_max = 0.0;
    double nu_max  = 0.0;  // µ/ρ
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        Prim q = prim(i,j,k);
        double lam = std::max({std::abs(q.u), std::abs(q.v), std::abs(q.w)}) + q.c;
        if (lam > lam_max) lam_max = lam;
        double nu = sutherland(q.T) / q.rho;
        if (nu > nu_max) nu_max = nu;
    }
    const double h_min = std::min({h, hy, hz});
    double dt_conv = (lam_max > 1e-300) ? cfl * h_min / lam_max : 1e300;
    double dt_visc = (nu_max  > 1e-300) ? h_min * h_min / (2.0 * C_VISC * nu_max) : 1e300;
    return std::min(dt_conv, dt_visc);
}
void CellBlock::zero_ghosts() noexcept {
    for (int v = 0; v < NVAR; ++v) {
        auto& f = Q[v];
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            bool ghost = (i<NG||i>=NB2-NG||j<NG||j>=NB2-NG||k<NG||k>=NB2-NG);
            if (ghost) f[cell_idx(i,j,k)] = 0.0;
        }
    }
}

// =============================================================================
// BlockTree — P1.1 free-list allocator
// =============================================================================
int BlockTree::alloc_node() {
    if (!free_list_.empty()) {
        int idx = free_list_.back();
        free_list_.pop_back();
        nodes[idx].reset();
        return idx;
    }
    nodes.emplace_back();
    return (int)nodes.size() - 1;
}

// Allocate `n` consecutive node indices, guaranteeing nodes[first..first+n-1]
// are a valid, contiguous, ascending run.  Used by refine() to keep the
// first_child+oct indexing invariant valid even after free-list reuse.
//
// Strategy:
//   1. Sort the free-list and look for a run of n consecutive indices.
//   2. If found, remove them from the free_list_ and return the first.
//   3. Otherwise append n new nodes to the end of nodes[] and return the first.
//
// This is O(F log F + F) where F = free_list_.size(), which is acceptable
// because refine() is rare compared to per-cell flux computations.
int BlockTree::alloc_node_group(int n) {
    // Try to find n consecutive slots in the free list
    if ((int)free_list_.size() >= n) {
        std::sort(free_list_.begin(), free_list_.end());
        for (int k = 0; k <= (int)free_list_.size() - n; ++k) {
            bool run = true;
            for (int j = 1; j < n; ++j) {
                if (free_list_[k+j] != free_list_[k] + j) { run = false; break; }
            }
            if (run) {
                int first = free_list_[k];
                free_list_.erase(free_list_.begin() + k,
                                 free_list_.begin() + k + n);
                for (int j = 0; j < n; ++j) nodes[first + j].reset();
                return first;
            }
        }
    }
    // No contiguous run found: append n new nodes
    int first = (int)nodes.size();
    nodes.resize(first + n);
    return first;
}

void BlockTree::free_node(int idx) {
    nodes[idx].block.reset();
    nodes[idx].parent      = NODE_DEAD;
    nodes[idx].first_child = -1;
    for (auto& fr : nodes[idx].flux_reg) fr.clear();
    free_list_.push_back(idx);
    invalidate_leaf_cache();
}

// =============================================================================
// BlockTree — P1.6 leaf cache
// =============================================================================
const std::vector<int>& BlockTree::leaf_indices() const {
    if (!leaf_dirty_) return leaf_cache_;
    leaf_cache_.clear();
    for (int i = 0; i < (int)nodes.size(); ++i)
        if (nodes[i].is_leaf() && nodes[i].has_block())
            leaf_cache_.push_back(i);
    leaf_dirty_ = false;
    return leaf_cache_;
}

const std::vector<int>& BlockTree::morton_leaf_indices() const {
    const auto& leaves = leaf_indices();   // rebuilds leaf_cache_ if dirty
    if (morton_leaf_cache_.size() == leaves.size())
        return morton_leaf_cache_;
    morton_leaf_cache_ = leaves;
    std::sort(morton_leaf_cache_.begin(), morton_leaf_cache_.end(),
              [this](int a, int b) { return nodes[a].morton < nodes[b].morton; });
    return morton_leaf_cache_;
}

int BlockTree::max_leaf_level() const noexcept {
    int lmax = 0;
    for (int li : leaf_indices())
        lmax = std::max(lmax, nodes[li].level);
    return lmax;
}

int BlockTree::min_leaf_level() const noexcept {
    int lmin = INT_MAX;
    for (int li : leaf_indices())
        lmin = std::min(lmin, nodes[li].level);
    return (lmin == INT_MAX) ? 0 : lmin;
}

// =============================================================================
// init
// =============================================================================
void BlockTree::init(double Lx, double Ly, double Lz) {
    nx_roots_ = 1; ny_roots_ = 1; nz_roots_ = 1;
    domain_L_  = Lx;
    domain_Ly_ = Ly;
    domain_Lz_ = Lz;
    nodes.clear();
    free_list_.clear();
    leaf_dirty_ = true;
    [[maybe_unused]] int root_idx = alloc_node();
    assert(root_idx == 0);
    auto& root  = nodes[0];
    root.level  = 0;
    root.morton = 0;
    root.parent = -1;
    root.ox = 0.0; root.oy = 0.0; root.oz = 0.0;
    root.block  = std::make_unique<CellBlock>(0.0, 0.0, 0.0,
                                              Lx / NB, Ly / NB, Lz / NB);
}

void BlockTree::init(double L) {
    init(L, L, L);   // cubic shorthand — nx_roots_=ny_roots_=nz_roots_=1 set inside
}

void BlockTree::init(double Lx, double Ly, double Lz, int NX, int NY, int NZ) {
    assert(NX >= 1 && NY >= 1 && NZ >= 1);
    domain_L_  = Lx;
    domain_Ly_ = Ly;
    domain_Lz_ = Lz;
    nx_roots_ = NX;
    ny_roots_ = NY;
    nz_roots_ = NZ;
    nodes.clear();
    free_list_.clear();
    leaf_dirty_ = true;

    const double cell_hx = Lx / (NX * NB);
    const double cell_hy = Ly / (NY * NB);
    const double cell_hz = Lz / (NZ * NB);
    const double root_Lx = Lx / NX;
    const double root_Ly = Ly / NY;
    const double root_Lz = Lz / NZ;

    nodes.resize(NX * NY * NZ);
    for (int iz = 0; iz < NZ; ++iz)
    for (int iy = 0; iy < NY; ++iy)
    for (int ix = 0; ix < NX; ++ix) {
        int ridx = iz * (NX * NY) + iy * NX + ix;
        auto& nd  = nodes[ridx];
        nd.reset();
        nd.parent      = -1;
        nd.first_child = -1;
        nd.level       = 0;
        nd.morton      = 0;
        nd.ox = ix * root_Lx;
        nd.oy = iy * root_Ly;
        nd.oz = iz * root_Lz;
        nd.block = std::make_unique<CellBlock>(nd.ox, nd.oy, nd.oz,
                                               cell_hx, cell_hy, cell_hz);
    }
    rebuild_neighbours();
}

// =============================================================================
// child geometry
// =============================================================================
void BlockTree::set_child_geometry(int parent_idx, int child_local, int child_idx) {
    const auto& par = nodes[parent_idx];
    double cell_hx = par.block ? par.block->h  * 0.5
                               : domain_L_  / (NB * (1 << (par.level + 1)));
    double cell_hy = par.block ? par.block->hy * 0.5
                               : domain_Ly_ / (NB * (1 << (par.level + 1)));
    double cell_hz = par.block ? par.block->hz * 0.5
                               : domain_Lz_ / (NB * (1 << (par.level + 1)));
    // Use node-level ox/oy/oz (valid even when block is null after internal refine).
    double ox = par.ox;
    double oy = par.oy;
    double oz = par.oz;
    if (oct_ix(child_local)) ox += cell_hx * NB;
    if (oct_iy(child_local)) oy += cell_hy * NB;
    if (oct_iz(child_local)) oz += cell_hz * NB;
    nodes[child_idx].ox = ox;
    nodes[child_idx].oy = oy;
    nodes[child_idx].oz = oz;
    nodes[child_idx].block = std::make_unique<CellBlock>(ox, oy, oz, cell_hx, cell_hy, cell_hz);
}

uint32_t BlockTree::child_morton(uint32_t parent_code, int oct) noexcept {
    return (parent_code << 3) | (uint32_t)oct;
}

// =============================================================================
// prolongate / restrict
// =============================================================================
void BlockTree::prolongate_to_children(int parent_idx) {
    auto& par = nodes[parent_idx];
    if (!par.block) return;
    for (int oct = 0; oct < 8; ++oct) {
        int ci = par.first_child + oct;   // guaranteed contiguous by alloc_node_group
        auto& ch_blk = *nodes[ci].block;
        int i0 = oct_ix(oct) ? NB/2 : 0;
        int j0 = oct_iy(oct) ? NB/2 : 0;
        int k0 = oct_iz(oct) ? NB/2 : 0;
        for (int k = 0; k < NB; ++k)
        for (int j = 0; j < NB; ++j)
        for (int i = 0; i < NB; ++i) {
            int pi = ilo() + i0 + i/2;
            int pj = ilo() + j0 + j/2;
            int pk = ilo() + k0 + k/2;
            const int dst = cell_idx(ilo()+i, ilo()+j, ilo()+k);
            const int src = cell_idx(pi, pj, pk);
            for (int v = 0; v < NVAR; ++v)
                ch_blk.Q[v][dst] = par.block->Q[v][src];
            ch_blk.phi_data_[dst] = par.block->phi_data_[src];  // P14.1
        }
    }
}

void BlockTree::restrict_to_parent(int parent_idx) {
    auto& par = nodes[parent_idx];
    if (!par.block) return;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i) {
        const int idx = cell_idx(i,j,k);
        for (int v = 0; v < NVAR; ++v) par.block->Q[v][idx] = 0.0;
        par.block->phi_data_[idx] = 0.0;  // P14.1
    }

    for (int oct = 0; oct < 8; ++oct) {
        int ci = par.first_child + oct;   // guaranteed contiguous by alloc_node_group
        auto& ch_blk = *nodes[ci].block;
        int i0 = oct_ix(oct) ? NB/2 : 0;
        int j0 = oct_iy(oct) ? NB/2 : 0;
        int k0 = oct_iz(oct) ? NB/2 : 0;
        for (int k = 0; k < NB/2; ++k)
        for (int j = 0; j < NB/2; ++j)
        for (int i = 0; i < NB/2; ++i) {
            double phi_s = 0.0;
            double qs[NVAR] = {};
            for (int dk = 0; dk < 2; ++dk)
            for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di) {
                const int src = cell_idx(ilo()+2*i+di, ilo()+2*j+dj, ilo()+2*k+dk);
                for (int v = 0; v < NVAR; ++v) qs[v] += ch_blk.Q[v][src];
                phi_s += ch_blk.phi_data_[src];  // P14.1
            }
            const int dst = cell_idx(ilo()+i0+i, ilo()+j0+j, ilo()+k0+k);
            for (int v = 0; v < NVAR; ++v)
                par.block->Q[v][dst] += qs[v] * 0.125;
            par.block->phi_data_[dst] += phi_s * 0.125;  // P14.1
        }
    }
}

// =============================================================================
// refine
// =============================================================================
void BlockTree::refine(int idx) {
    assert(nodes[idx].is_leaf());

    int      saved_level  = nodes[idx].level;
    uint32_t saved_morton = nodes[idx].morton;
    CellBlock parent_data = *nodes[idx].block;  // cache geometry

    // A05-fix: allocate 8 children as a CONTIGUOUS group.
    int first = alloc_node_group(8);
    nodes[idx].first_child = first;

    for (int oct = 0; oct < 8; ++oct) {
        int ci = first + oct;
        auto& ch      = nodes[ci];
        ch.parent      = idx;
        ch.first_child = -1;
        ch.level       = saved_level + 1;
        ch.morton      = child_morton(saved_morton, oct);
        ch.neighbours.fill(-1);
        set_child_geometry(idx, oct, ci);  // creates child CellBlock with geometry
    }

    if (on_gpu_prolong_) {
        // D1 GPU-native path: original block still in pool (not freed yet).
        // Callback owns: alloc children GPU, D2D prolong parent→children, free parent GPU.
        // CPU Q in children is uninitialized — GPU is authoritative.
        CellBlock* children[8];
        for (int oct = 0; oct < 8; ++oct)
            children[oct] = nodes[first + oct].block.get();
        on_gpu_prolong_(nodes[idx].block.get(), children);
        // block is freed by callback; reset() here only nulls the unique_ptr.
    } else {
        // Original CPU path: free parent GPU (while original block still alive),
        // then replace with parent_data copy for prolongation, then upload children.
        if (on_block_free_) on_block_free_(nodes[idx].block.get());
        nodes[idx].block = std::make_unique<CellBlock>(parent_data);
        prolongate_to_children(idx);
        if (on_block_alloc_) {
            for (int oct = 0; oct < 8; ++oct)
                on_block_alloc_(nodes[first + oct].block.get());
        }
    }

    nodes[idx].block.reset();

    invalidate_leaf_cache();
    rebuild_neighbours();
}

// =============================================================================
// coarsen — P1.1: free-list, no tail assumption
// =============================================================================
void BlockTree::coarsen(int parent_idx) {
    int fc = nodes[parent_idx].first_child;
    assert(fc >= 0);
    // Children are always contiguous by alloc_node_group invariant.
    for (int oct = 0; oct < 8; ++oct)
        assert(nodes[fc + oct].is_leaf());

    if (!nodes[parent_idx].block) {
        double hx_par = nodes[fc].block->h  * 2.0;
        double hy_par = nodes[fc].block->hy * 2.0;
        double hz_par = nodes[fc].block->hz * 2.0;
        double ox     = nodes[fc].block->ox;
        double oy     = nodes[fc].block->oy;
        double oz     = nodes[fc].block->oz;
        nodes[parent_idx].block =
            std::make_unique<CellBlock>(ox, oy, oz, hx_par, hy_par, hz_par);
    }

    if (on_gpu_coarsen_) {
        // D1 GPU-native path: callback owns all device-buffer management
        // (alloc parent GPU, D2D restrict children→parent, free children GPU).
        CellBlock* children[8];
        for (int oct = 0; oct < 8; ++oct)
            children[oct] = nodes[fc + oct].block.get();
        on_gpu_coarsen_(nodes[parent_idx].block.get(), children);
    } else {
        restrict_to_parent(parent_idx);
        if (on_block_alloc_) on_block_alloc_(nodes[parent_idx].block.get());
        if (on_block_free_) {
            for (int oct = 0; oct < 8; ++oct)
                on_block_free_(nodes[fc + oct].block.get());
        }
    }

    nodes[parent_idx].first_child = -1;

    for (int oct = 0; oct < 8; ++oct)
        free_node(fc + oct);

    invalidate_leaf_cache();
    rebuild_neighbours();
}

// =============================================================================
// n_leaves
// =============================================================================
int BlockTree::n_leaves() const noexcept {
    return (int)leaf_indices().size();
}

// =============================================================================
// rebuild_neighbours
// =============================================================================
void BlockTree::rebuild_neighbours() {
    for (auto& nd : nodes) nd.neighbours.fill(-1);

    const auto& leaves = leaf_indices();
    if (leaves.empty()) return;

    // For each leaf, find which level-0 root it belongs to by climbing parent links.
    auto get_root_id = [&](int li) -> int {
        int cur = li;
        while (nodes[cur].level > 0) cur = nodes[cur].parent;
        return cur;  // index of the level-0 root node (0..nx*ny*nz-1)
    };

    // Key: bits[63:35]=root_id (29 bits), bits[34:30]=level (5 bits),
    //      bits[29:0]=morton (30 bits).  Supports up to 2^29 roots — plenty.
    std::unordered_map<uint64_t, int> lm_map;
    lm_map.reserve(leaves.size() * 2);
    for (int li : leaves) {
        int root_id = get_root_id(li);
        auto& nd    = nodes[li];
        uint64_t key = ((uint64_t)root_id << 35)
                     | ((uint64_t)nd.level << 30)
                     | (uint64_t)nd.morton;
        lm_map[key] = li;
    }

    auto morton_face_neighbour = [](uint32_t code, int level,
                                    int axis, int delta) -> uint32_t {
        uint32_t mx, my, mz;
        morton_decode(code, mx, my, mz);
        uint32_t max_coord = (1u << level) - 1u;
        uint32_t& mc = (axis == 0) ? mx : (axis == 1) ? my : mz;
        if (delta > 0) { if (mc == max_coord) return UINT32_MAX; mc++; }
        else           { if (mc == 0)          return UINT32_MAX; mc--; }
        return morton_encode(mx, my, mz);
    };

    // Mirror the boundary axis coordinate to the far end of the adjacent root.
    auto cross_morton = [](uint32_t code, int level, int axis, int delta) -> uint32_t {
        uint32_t mx, my, mz;
        morton_decode(code, mx, my, mz);
        uint32_t max_coord = (1u << level) - 1u;
        uint32_t& mc = (axis == 0) ? mx : (axis == 1) ? my : mz;
        mc = (delta > 0) ? 0u : max_coord;
        return morton_encode(mx, my, mz);
    };

    static constexpr int face_axis[NFACES]  = { 0, 0, 1, 1, 2, 2 };
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1 };

    for (int ai : leaves) {
        auto& a      = nodes[ai];
        int   lev    = a.level;
        int   root_a = get_root_id(ai);

        for (int d = 0; d < NFACES; ++d) {
            if (a.neighbours[d] >= 0) continue;
            int axis  = face_axis[d];
            int delta = face_delta[d];

            uint32_t nb_code = morton_face_neighbour(a.morton, lev, axis, delta);

            if (nb_code != UINT32_MAX) {
                // ── Intra-root: same level ──────────────────────────────────────
                uint64_t key = ((uint64_t)root_a << 35)
                             | ((uint64_t)lev << 30)
                             | (uint64_t)nb_code;
                auto it = lm_map.find(key);
                if (it != lm_map.end()) {
                    int bi = it->second;
                    a.neighbours[d]             = bi;
                    nodes[bi].neighbours[d ^ 1] = ai;
                    continue;
                }
                // ── Intra-root: one level coarser (C/F interface) ───────────────
                if (lev > 0) {
                    uint64_t key2 = ((uint64_t)root_a << 35)
                                  | ((uint64_t)(lev - 1) << 30)
                                  | (uint64_t)(nb_code >> 3);
                    auto it2 = lm_map.find(key2);
                    if (it2 != lm_map.end()) {
                        int bi = it2->second;
                        a.neighbours[d]             = bi;
                        nodes[bi].neighbours[d ^ 1] = ai;
                        continue;
                    }
                }
            } else {
                // ── At root boundary — cross-root or domain edge ────────────────
                int rx = root_a % nx_roots_;
                int ry = (root_a / nx_roots_) % ny_roots_;
                int rz = root_a / (nx_roots_ * ny_roots_);

                int rx2 = rx + (axis == 0 ? delta : 0);
                int ry2 = ry + (axis == 1 ? delta : 0);
                int rz2 = rz + (axis == 2 ? delta : 0);

                if (rx2 < 0 || rx2 >= nx_roots_) {
                    if (!periodic_axis_[0]) continue;
                    rx2 = (rx2 + nx_roots_) % nx_roots_;
                }
                if (ry2 < 0 || ry2 >= ny_roots_) {
                    if (!periodic_axis_[1]) continue;
                    ry2 = (ry2 + ny_roots_) % ny_roots_;
                }
                if (rz2 < 0 || rz2 >= nz_roots_) {
                    if (!periodic_axis_[2]) continue;
                    rz2 = (rz2 + nz_roots_) % nz_roots_;
                }

                int root_b = rz2 * (nx_roots_ * ny_roots_) + ry2 * nx_roots_ + rx2;

                // Single-root periodic: avoid linking root to itself at lev==0
                if (root_b == root_a && lev == 0) continue;

                uint32_t xcode = cross_morton(a.morton, lev, axis, delta);

                // ── Cross-root: same level ──────────────────────────────────────
                uint64_t key = ((uint64_t)root_b << 35)
                             | ((uint64_t)lev << 30)
                             | (uint64_t)xcode;
                auto it = lm_map.find(key);
                if (it != lm_map.end()) {
                    int bi = it->second;
                    a.neighbours[d]             = bi;
                    nodes[bi].neighbours[d ^ 1] = ai;
                    continue;
                }
                // ── Cross-root: one level coarser (C/F at root boundary) ─────────
                if (lev > 0) {
                    uint64_t key2 = ((uint64_t)root_b << 35)
                                  | ((uint64_t)(lev - 1) << 30)
                                  | (uint64_t)(xcode >> 3);
                    auto it2 = lm_map.find(key2);
                    if (it2 != lm_map.end()) {
                        int bi = it2->second;
                        a.neighbours[d]             = bi;
                        nodes[bi].neighbours[d ^ 1] = ai;
                        continue;
                    }
                }
            }
        }
    }
    invalidate_leaf_cache();
    leaf_dirty_ = false;
}

// =============================================================================
// balance — P1.2: work-queue model
// =============================================================================
int BlockTree::balance() {
    int extra = 0;
    std::deque<int> queue;
    for (int li : leaf_indices()) queue.push_back(li);

    while (!queue.empty()) {
        int li = queue.front(); queue.pop_front();
        if (!nodes[li].is_leaf()) continue;  // may have been refined since enqueue
        for (int d = 0; d < NFACES; ++d) {
            int ni = nodes[li].neighbours[d];
            if (ni < 0 || !nodes[ni].is_leaf()) continue;
            if (nodes[ni].level < nodes[li].level - 1) {
                refine(ni);
                ++extra;
                // Enqueue the 8 new children (contiguous by alloc_node_group)
                int fc = nodes[ni].first_child;
                for (int oct = 0; oct < 8; ++oct)
                    queue.push_back(fc + oct);
            }
        }
    }
    return extra;
}

// =============================================================================
// Ghost fill helpers
// =============================================================================
// P1.3: per-face level dispatch
//   same level  → direct copy
//   fine (li) adjacent to coarse (ni) → fill_cf_ghosts(fine, coarse, oct, axis, side)
//   coarse (li) adjacent to fine (ni) → A05-fix2: average the 2×2 fine interior
//     cells covering each coarse ghost position (conservative averaged ghost fill)
//
// Fill order: 1→faces, 2→edges (read face ghosts), 3→corners (read edge ghosts)
// =============================================================================

// Helper: axis and side from FaceDir
static inline int fd_axis(int d) { return d >> 1; }  // 0→x, 2→y, 4→z
static inline int fd_side(int d) { return d & 1;  }  // 0→minus, 1→plus

// Periodic wrap lookup: find the source block at the wrapped Morton position.
// Returns nullptr when the domain is a single root block (level==0).
struct PeriodicSrc { const CellBlock* blk; int level_rel; };
static PeriodicSrc periodic_src_lookup(
    const std::vector<BlockNode>& nodes,
    const std::unordered_map<uint64_t,int>& lm_map,
    const BlockNode& nd, int d) noexcept
{
    static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
    static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};
    const int axis = face_axis[d], delta = face_delta[d], lev = nd.level;
    if (lev == 0) return {nullptr, 0};
    uint32_t mx, my, mz;
    morton_decode(nd.morton, mx, my, mz);
    const uint32_t max_coord = (1u << lev) - 1u;
    uint32_t& mc = (axis == 0) ? mx : (axis == 1) ? my : mz;
    mc = (delta > 0) ? 0 : max_coord;
    const uint64_t key = ((uint64_t)lev << 32) | morton_encode(mx, my, mz);
    auto it = lm_map.find(key);
    if (it != lm_map.end() && nodes[it->second].has_block())
        return {nodes[it->second].block.get(), 0};
    if (lev > 1) {
        const uint32_t pc = morton_encode(mx, my, mz) >> 3;
        const uint64_t pk = ((uint64_t)(lev-1) << 32) | pc;
        auto it2 = lm_map.find(pk);
        if (it2 != lm_map.end() && nodes[it2->second].has_block())
            return {nodes[it2->second].block.get(), -1};
    }
    return {nullptr, 0};
}

// Compute characteristic ghost Prim for one open-boundary cell.
// axis: 0/1/2 = x/y/z; outward_sign: +1 for + faces, -1 for - faces.
// Returns zero-gradient state when open_bc_p == 0 or supersonic outflow.
static Prim open_char_ghost(const Prim& p, int axis, double outward_sign,
                            double open_bc_p) noexcept {
    // Normal velocity component (positive = flow exits the domain)
    const double u_n = outward_sign * (axis==0 ? p.u : axis==1 ? p.v : p.w);

    // Zero-gradient: no far-field pressure, supersonic outflow, or inflow
    if (open_bc_p <= 0.0 || u_n >= p.c || u_n < 0.0) return p;

    // Subsonic outflow: set p_ghost = p_∞, isentropic ρ, Riemann-invariant u_n
    // ρ_g = ρ·(p∞/p)^(1/γ),  c_g = c·(p∞/p)^((γ-1)/(2γ))
    // Incoming Riemann char: u_n_g - outward_sign·2c_g/(γ-1) = u_n_i - outward_sign·2c_i/(γ-1)
    //   → δu_n_g = outward_sign·2·(c_g - c_i)/(γ-1)  (both + and - face use same formula
    //               because outward_sign cancels from both sides of the invariant eq.)
    const double ratio = open_bc_p / p.p;
    const double c_g   = p.c * std::pow(ratio, (GAMMA-1.0)/(2.0*GAMMA));
    const double delta_u_n = outward_sign * 2.0*(c_g - p.c)/(GAMMA-1.0);

    Prim g = p;
    g.p   = open_bc_p;
    g.rho = p.rho * std::pow(ratio, 1.0/GAMMA);
    ((axis==0)?g.u:(axis==1)?g.v:g.w) = ((axis==0)?p.u:(axis==1)?p.v:p.w) + delta_u_n;
    g.T = g.p / (g.rho * R_GAS);
    g.c = c_g;
    return g;
}

// Helper: octant of child li relative to its parent (needed for fill_cf_ghosts)
// Children are contiguous by alloc_node_group invariant, so li - first_child
// gives the correct octant index.
static int child_octant_of(const std::vector<BlockNode>& nodes, int li) {
    int p = nodes[li].parent;
    if (p < 0) return 0;
    int fc = nodes[p].first_child;
    return li - fc;
}

// =============================================================================
// A05-fix2: fill_coarse_ghost_from_fine
//
// Called when coarse leaf `nd` (node index `li`) has a fine neighbour `ni`
// on face direction `d`.  For each coarse ghost slot, average the 2×2 fine
// interior cells that cover it.
//
// A05-fix5: guard against stale `ni` pointer.
//   rebuild_neighbours() writes nodes[coarse].neighbours[d^1] = ai for every fine
//   leaf ai that faces the coarse block on direction d.  With 4 fine blocks per
//   coarse face, only the LAST one processed survives.  After a regrid the
//   surviving ni may belong to a completely different refinement patch, so
//   nodes[ni].parent → first_child gives the wrong set of 8 children.
//   We detect this by checking that the fine children's cell size equals
//   coarse_blk.h / 2.  If the sizes disagree the pointer is stale and we
//   skip the ghost fill rather than corrupt mass conservation.
// =============================================================================
static void fill_coarse_ghost_from_fine(
    CellBlock& coarse_blk,
    const std::vector<BlockNode>& nodes,
    int ni,      // one of the fine block indices on this face (for parent lookup)
    int d        // face direction
) {
    const int axis = fd_axis(d);
    const int side = fd_side(d);
    const int half = NB / 2;

    // Parent of fine block ni holds all 8 fine children.
    int fine_parent = nodes[ni].parent;
    if (fine_parent < 0) return;  // ni is root level, no averaging possible
    int first_child = nodes[fine_parent].first_child;
    if (first_child < 0) return;

    // A05-fix5: verify that the fine children have the expected cell size on all axes.
    if (!nodes[first_child].has_block()) return;
    {
        const auto* fc = nodes[first_child].block.get();
        if (!fc) return;
        if (std::fabs(fc->h  - coarse_blk.h  * 0.5) > 1e-12 * coarse_blk.h  ||
            std::fabs(fc->hy - coarse_blk.hy * 0.5) > 1e-12 * coarse_blk.hy ||
            std::fabs(fc->hz - coarse_blk.hz * 0.5) > 1e-12 * coarse_blk.hz)
            return;  // stale pointer — skip rather than corrupt
    }

    // Fill NG ghost layers.  For each layer gl:
    //   side=0: ghost index NG-1-gl, fine row ihi()-gl (going inward)
    //   side=1: ghost index NB2-NG+gl, fine row ilo()+gl (going inward)
    for (int gl = 0; gl < NG; ++gl) {
        const int g      = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
        const int face_i = (side == 0) ? (ihi() - gl)  : (ilo() + gl);

        for (int a = ilo(); a <= ihi(); ++a)
        for (int b = ilo(); b <= ihi(); ++b) {
            int a_local = a - ilo();
            int b_local = b - ilo();

            int ia_blk = a_local / half;
            int ib_blk = b_local / half;

            // A05-fix4: use `side` directly — no -1 sentinel
            const int oix = (axis == 0) ? side   : ia_blk;
            const int oiy = (axis == 1) ? side   : (axis == 0) ? ia_blk : ib_blk;
            const int oiz = (axis == 2) ? side   : ib_blk;
            int fine_oct = oct_from_xyz(oix, oiy, oiz);
            int fi = first_child + fine_oct;

            if (fi < 0 || fi >= (int)nodes.size()) continue;
            if (!nodes[fi].has_block()) continue;
            const CellBlock& fsrc = *nodes[fi].block;

            int fa_start = NG + 2 * (a_local % half);
            int fb_start = NG + 2 * (b_local % half);

            const int gi = (axis == 0) ? g : a;
            const int gj = (axis == 1) ? g : (axis == 0) ? a : b;
            const int gk = (axis == 2) ? g : b;

            double avg[NVAR] = {};
            double phi_avg = 0.0;
            for (int da = 0; da < 2; ++da)
            for (int db = 0; db < 2; ++db) {
                const int fa   = fa_start + da;
                const int fb   = fb_start + db;
                const int ci   = (axis == 0) ? face_i : fa;
                const int cj   = (axis == 1) ? face_i : (axis == 0) ? fa : fb;
                const int ck   = (axis == 2) ? face_i : fb;
                const int flat = cell_idx(ci, cj, ck);
                for (int v = 0; v < NVAR; ++v) avg[v] += fsrc.Q[v][flat];
                phi_avg += fsrc.phi_data_[flat];
            }
            const int gdst = cell_idx(gi, gj, gk);
            for (int v = 0; v < NVAR; ++v)
                coarse_blk.Q[v][gdst] = avg[v] * 0.25;
            coarse_blk.phi_data_[gdst] = phi_avg * 0.25;  // P14.1
        }
    }
}

// ── fill_coarse_ghost_zero_grad ───────────────────────────────────────────────
// LTS coarse-step zero-gradient ghost fill.  For each ghost layer of face d on
// coarse_blk, sets ghost = adjacent interior cell (∂Q/∂n = 0 at C/F boundary).
// Viscous stress and heat flux are then exactly zero at the C/F face, so that
// the total-energy conservation identity holds after Berger-Colella correction.
static void fill_coarse_ghost_zero_grad(CellBlock& coarse_blk, int d) noexcept
{
    const int axis = fd_axis(d);
    const int side = fd_side(d);
    for (int gl = 0; gl < NG; ++gl) {
        const int g     = (side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
        const int inner = (side == 0) ? (ilo() + gl)  : (ihi() - gl);
        // Only fill face-normal ghost cells (ilo..ihi range in transverse dirs),
        // matching the range that fill_coarse_ghost_from_fine uses.
        for (int a = ilo(); a <= ihi(); ++a)
        for (int b = ilo(); b <= ihi(); ++b) {
            const int gi = (axis==0)?g:a,     gj = (axis==1)?g:(axis==0)?a:b,     gk = (axis==2)?g:b;
            const int ii = (axis==0)?inner:a, ij = (axis==1)?inner:(axis==0)?a:b, ik = (axis==2)?inner:b;
            const int dst = cell_idx(gi,gj,gk), src2 = cell_idx(ii,ij,ik);
            for (int v = 0; v < NVAR; ++v) coarse_blk.Q[v][dst] = coarse_blk.Q[v][src2];
            coarse_blk.phi_data_[dst] = coarse_blk.phi_data_[src2];
        }
    }
}

// ── fill_ghosts_periodic ──────────────────────────────────────────────────────
void BlockTree::fill_ghosts_periodic(bool cf_zero_grad) {
    const auto& leaves = leaf_indices();

    // Build (level, morton) → leaf index map for periodic boundary lookup.
    // When rebuild_neighbours() finds no same-level neighbor (domain edge),
    // it leaves neighbours[d]=-1.  On a periodic multi-block domain the correct
    // source is the leaf at the wrapped Morton code, not `this` block.
    std::unordered_map<uint64_t, int> lm_map;
    lm_map.reserve(leaves.size() * 2);
    for (int li : leaves) {
        auto& nd_tmp = nodes[li];
        lm_map[((uint64_t)nd_tmp.level << 32) | nd_tmp.morton] = li;
    }

    auto periodic_src = [&](const BlockNode& nd, int d) -> PeriodicSrc {
        return periodic_src_lookup(nodes, lm_map, nd, d);
    };

    for (int li : leaves) {
        auto& nd  = nodes[li];
        if (!nd.has_block()) continue;  // P7.1: remote leaf (no local data)
        auto& blk = *nd.block;

        auto copy_cell = [&](int gi, int gj, int gk,
                             int si, int sj, int sk,
                             const CellBlock& src) noexcept {
            const int dst_flat = cell_idx(gi,gj,gk);
            const int src_flat = cell_idx(si,sj,sk);
            for (int v = 0; v < NVAR; ++v)
                blk.Q[v][dst_flat] = src.Q[v][src_flat];
            blk.phi_data_[dst_flat] = src.phi_data_[src_flat];  // P14.1
        };

        // ── 1. Face ghosts ────────────────────────────────────────────────────
        struct FaceSpec { int ghost_g, mirror_s, axis, side; };
        static const FaceSpec specs[NFACES] = {
            {0,      ihi(), 0, 0},  // XMINUS
            {NB2-1,  ilo(), 0, 1},  // XPLUS
            {0,      ihi(), 1, 0},  // YMINUS
            {NB2-1,  ilo(), 1, 1},  // YPLUS
            {0,      ihi(), 2, 0},  // ZMINUS
            {NB2-1,  ilo(), 2, 1},  // ZPLUS
        };

        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            const FaceSpec& sp = specs[d];

            // P7.1: remote face already filled by mpi_exchange_halos()
            if (ni >= 0 && mpi_is_remote(mpi_, ni)) continue;

            if (ni >= 0 && nodes[ni].has_block()) {
                if (nodes[ni].level < nd.level) {
                    // P1.3: fine leaf adjacent to coarse neighbour → CF ghost fill
                    int oct = child_octant_of(nodes, li);
                    fill_cf_ghosts(blk, *nodes[ni].block, oct, sp.axis, sp.side);
                    continue;
                }
                if (nodes[ni].level > nd.level) {
                    // A05-fix2: coarse leaf adjacent to fine neighbour →
                    // averaged ghost fill OR zero-gradient (LTS coarse step).
                    if (cf_zero_grad)
                        fill_coarse_ghost_zero_grad(blk, d);
                    else
                        fill_coarse_ghost_from_fine(blk, nodes, ni, d);
                    continue;
                }
            }

            // Same level neighbour, or domain boundary → resolve periodic source.
            // When ni==-1 (domain boundary), look up the periodically-wrapped leaf.
            PeriodicSrc psrc = (ni < 0) ? periodic_src(nd, d) : PeriodicSrc{nullptr, 0};
            if (ni < 0 && psrc.blk && psrc.level_rel < 0) {
                // Periodic wrap reached a coarser block: use CF ghost fill, not 1:1 copy.
                int oct = child_octant_of(nodes, li);
                fill_cf_ghosts(blk, *psrc.blk, oct, sp.axis, sp.side);
                continue;
            }
            const CellBlock& src = (ni>=0 && nodes[ni].has_block())
                                   ? *nodes[ni].block
                                   : (psrc.blk ? *psrc.blk : blk);
            // Fill all NG ghost layers.
            // side=0: ghost NG-1-gl from src[ihi()-gl]; side=1: ghost NB2-NG+gl from src[ilo()+gl]
            for (int gl = 0; gl < NG; ++gl) {
                const int g_idx   = (sp.side == 0) ? (NG - 1 - gl) : (NB2 - NG + gl);
                const int src_idx = (sp.side == 0) ? (ihi() - gl)  : (ilo() + gl);
                for (int a = ilo(); a <= ihi(); ++a)
                for (int b = ilo(); b <= ihi(); ++b) {
                    const int gi=(sp.axis==0)?g_idx:a,   gj=(sp.axis==1)?g_idx:(sp.axis==0)?a:b,   gk=(sp.axis==2)?g_idx:b;
                    const int si=(sp.axis==0)?src_idx:a, sj=(sp.axis==1)?src_idx:(sp.axis==0)?a:b, sk=(sp.axis==2)?src_idx:b;
                    copy_cell(gi, gj, gk, si, sj, sk, src);
                }
            }
        }

        // ── 2. Edge ghosts ────────────────────────────────────────────────────
        // Periodic map: left ghost gl ↔ source ihi()-(NG-1)+gl
        //               right ghost gl ↔ source ilo()+gl
        // XY edges (k interior, i and j both in ghost range)
        for (int k=ilo();k<=ihi();++k)
        for (int glx=0; glx<NG; ++glx)
        for (int gly=0; gly<NG; ++gly) {
            const int gx_lo=glx,        sx_lo=ihi()-(NG-1)+glx;
            const int gx_hi=NB2-NG+glx, sx_hi=ilo()+glx;
            const int gy_lo=gly,        sy_lo=ihi()-(NG-1)+gly;
            const int gy_hi=NB2-NG+gly, sy_hi=ilo()+gly;
            copy_cell(gx_lo, gy_lo, k,  sx_lo, sy_lo, k, blk);
            copy_cell(gx_hi, gy_lo, k,  sx_hi, sy_lo, k, blk);
            copy_cell(gx_lo, gy_hi, k,  sx_lo, sy_hi, k, blk);
            copy_cell(gx_hi, gy_hi, k,  sx_hi, sy_hi, k, blk);
        }
        // XZ edges (j interior)
        for (int j=ilo();j<=ihi();++j)
        for (int glx=0; glx<NG; ++glx)
        for (int glz=0; glz<NG; ++glz) {
            const int gx_lo=glx,        sx_lo=ihi()-(NG-1)+glx;
            const int gx_hi=NB2-NG+glx, sx_hi=ilo()+glx;
            const int gz_lo=glz,        sz_lo=ihi()-(NG-1)+glz;
            const int gz_hi=NB2-NG+glz, sz_hi=ilo()+glz;
            copy_cell(gx_lo, j, gz_lo,  sx_lo, j, sz_lo, blk);
            copy_cell(gx_hi, j, gz_lo,  sx_hi, j, sz_lo, blk);
            copy_cell(gx_lo, j, gz_hi,  sx_lo, j, sz_hi, blk);
            copy_cell(gx_hi, j, gz_hi,  sx_hi, j, sz_hi, blk);
        }
        // YZ edges (i interior)
        for (int i=ilo();i<=ihi();++i)
        for (int gly=0; gly<NG; ++gly)
        for (int glz=0; glz<NG; ++glz) {
            const int gy_lo=gly,        sy_lo=ihi()-(NG-1)+gly;
            const int gy_hi=NB2-NG+gly, sy_hi=ilo()+gly;
            const int gz_lo=glz,        sz_lo=ihi()-(NG-1)+glz;
            const int gz_hi=NB2-NG+glz, sz_hi=ilo()+glz;
            copy_cell(i, gy_lo, gz_lo,  i, sy_lo, sz_lo, blk);
            copy_cell(i, gy_hi, gz_lo,  i, sy_hi, sz_lo, blk);
            copy_cell(i, gy_lo, gz_hi,  i, sy_lo, sz_hi, blk);
            copy_cell(i, gy_hi, gz_hi,  i, sy_hi, sz_hi, blk);
        }

        // ── 3. Corner ghosts ───────────────────────────────────────────────────
        for (int glx=0; glx<NG; ++glx)
        for (int gly=0; gly<NG; ++gly)
        for (int glz=0; glz<NG; ++glz) {
            const int gx_lo=glx, sx_lo=ihi()-(NG-1)+glx, gx_hi=NB2-NG+glx, sx_hi=ilo()+glx;
            const int gy_lo=gly, sy_lo=ihi()-(NG-1)+gly, gy_hi=NB2-NG+gly, sy_hi=ilo()+gly;
            const int gz_lo=glz, sz_lo=ihi()-(NG-1)+glz, gz_hi=NB2-NG+glz, sz_hi=ilo()+glz;
            copy_cell(gx_lo, gy_lo, gz_lo,  sx_lo, sy_lo, sz_lo, blk);
            copy_cell(gx_hi, gy_lo, gz_lo,  sx_hi, sy_lo, sz_lo, blk);
            copy_cell(gx_lo, gy_hi, gz_lo,  sx_lo, sy_hi, sz_lo, blk);
            copy_cell(gx_hi, gy_hi, gz_lo,  sx_hi, sy_hi, sz_lo, blk);
            copy_cell(gx_lo, gy_lo, gz_hi,  sx_lo, sy_lo, sz_hi, blk);
            copy_cell(gx_hi, gy_lo, gz_hi,  sx_hi, sy_lo, sz_hi, blk);
            copy_cell(gx_lo, gy_hi, gz_hi,  sx_lo, sy_hi, sz_hi, blk);
            copy_cell(gx_hi, gy_hi, gz_hi,  sx_hi, sy_hi, sz_hi, blk);
        }
    }
}

// ── fill_ghosts_wall (no-slip adiabatic) ────────────────────────────────────
void BlockTree::fill_ghosts_wall(bool cf_zero_grad) {
    FaceBCArray bcs; bcs.fill(WallBC{});
    fill_ghosts_per_face(bcs, cf_zero_grad);
}

// ── fill_ghosts_open (P13.3: characteristic open BC with optional p∞) ────────
// When bc_cfg.open_bc_p == 0: zero-gradient transmissive (legacy behaviour).
// When bc_cfg.open_bc_p >  0: subsonic outflow uses isentropic ghost + Riemann-
//   invariant velocity; HLLC-ES then adds entropy dissipation at the face.
//   Supersonic outflow and inflow fall back to zero-gradient.
void BlockTree::fill_ghosts_open(bool cf_zero_grad) {
    FaceBCArray bcs; bcs.fill(OpenBC{});
    fill_ghosts_per_face(bcs, cf_zero_grad);
}

// ── fill_ghosts_per_face ─────────────────────────────────────────────────────
// Per-face ghost fill: each of the six domain faces independently uses one of
// Periodic / Wall / Open / ContactAngleBC.  Interior and C/F faces are handled
// identically to the uniform variants; only domain-boundary faces differ.
// wall_T and open_bc_p are read from bc_cfg (shared across all faces).
void BlockTree::fill_ghosts_per_face(const FaceBCArray& bcs, bool cf_zero_grad) {
    const auto& leaves = leaf_indices();

    bool any_periodic = false;
    for (int d = 0; d < NFACES; ++d)
        if (bc_is_periodic(bcs[d])) { any_periodic = true; break; }

    std::unordered_map<uint64_t, int> lm_map;
    if (any_periodic) {
        lm_map.reserve(leaves.size() * 2);
        for (int li : leaves) {
            const auto& nd_tmp = nodes[li];
            lm_map[((uint64_t)nd_tmp.level << 32) | nd_tmp.morton] = li;
        }
    }

    auto periodic_src = [&](const BlockNode& nd, int d) -> PeriodicSrc {
        return periodic_src_lookup(nodes, lm_map, nd, d);
    };

    struct FaceSpec { int axis, side; };
    static const FaceSpec specs[NFACES] = {
        {0,0},{0,1},{1,0},{1,1},{2,0},{2,1},
    };

    for (int li : leaves) {
        auto& nd  = nodes[li];
        if (!nd.has_block()) continue;
        auto& blk = *nd.block;

        const double wall_T    = bc_cfg.wall_T;
        const double ca_cos    = bc_cfg.wall_ca_cos;
        const double ca_ceps   = bc_cfg.wall_ca_ceps;
        const double open_bc_p = bc_cfg.open_bc_p;

        auto wall_E = [wall_T](const CellBlock& b, int mi, int mj, int mk) noexcept {
            if (wall_T <= 0.0) return b.E(mi, mj, mk);
            const Prim p = b.prim(mi, mj, mk);
            const double KE = 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
            return p.rho * (R_GAS/(GAMMA-1.0)) * (2.0*wall_T - p.T) + KE;
        };
        auto phi_wall_ghost = [ca_cos, ca_ceps](double phi_ref, int dist) noexcept -> double {
            if (ca_ceps <= 0.0) return phi_ref;
            const double g_prime = 0.5 * phi_ref * (1.0 - phi_ref) * (1.0 - 2.0 * phi_ref);
            const double phi_g = phi_ref - dist * ca_cos / ca_ceps * g_prime;
            return (phi_g < 0.0) ? 0.0 : (phi_g > 1.0) ? 1.0 : phi_g;
        };
        auto write_ghost = [&](int gi, int gj, int gk, const Prim& g) noexcept {
            blk.rho (gi,gj,gk) = g.rho;
            blk.rhou(gi,gj,gk) = g.rho * g.u;
            blk.rhov(gi,gj,gk) = g.rho * g.v;
            blk.rhow(gi,gj,gk) = g.rho * g.w;
            blk.E   (gi,gj,gk) = g.p/(GAMMA-1.0) + 0.5*g.rho*(g.u*g.u+g.v*g.v+g.w*g.w);
        };
        auto copy_cell_f = [&](int gi, int gj, int gk, int si, int sj, int sk,
                                const CellBlock& src) noexcept {
            const int dst_flat = cell_idx(gi,gj,gk);
            const int src_flat = cell_idx(si,sj,sk);
            for (int v = 0; v < NVAR; ++v)
                blk.Q[v][dst_flat] = src.Q[v][src_flat];
            blk.phi_data_[dst_flat] = src.phi_data_[src_flat];
        };
        auto copy_strip = [&](int g_idx, int src_idx, int ax, const CellBlock& src) noexcept {
            for (int a = ilo(); a <= ihi(); ++a)
            for (int b = ilo(); b <= ihi(); ++b) {
                const int gi = (ax==0)?g_idx:a,   gj = (ax==1)?g_idx:(ax==0)?a:b,   gk = (ax==2)?g_idx:b;
                const int si = (ax==0)?src_idx:a, sj = (ax==1)?src_idx:(ax==0)?a:b, sk = (ax==2)?src_idx:b;
                copy_cell_f(gi,gj,gk, si,sj,sk, src);
            }
        };

        bool bnd[NFACES];
        for (int d = 0; d < NFACES; ++d) bnd[d] = (nd.neighbours[d] < 0);

        for (int d = 0; d < NFACES; ++d) {
            const int ni   = nd.neighbours[d];
            const int axis = specs[d].axis;
            const int side = specs[d].side;

            if (ni >= 0 && mpi_is_remote(mpi_, ni)) continue;

            if (ni >= 0 && nodes[ni].has_block()) {
                if (nodes[ni].level < nd.level) {
                    fill_cf_ghosts(blk, *nodes[ni].block,
                                   child_octant_of(nodes, li), axis, side);
                    continue;
                }
                if (nodes[ni].level > nd.level) {
                    if (cf_zero_grad) fill_coarse_ghost_zero_grad(blk, d);
                    else              fill_coarse_ghost_from_fine(blk, nodes, ni, d);
                    continue;
                }
                // same-level: direct copy
                const CellBlock& src = *nodes[ni].block;
                for (int gl = 0; gl < NG; ++gl) {
                    copy_strip((side==0)?(NG-1-gl):(NB2-NG+gl),
                               (side==0)?(ihi()-gl):(ilo()+gl), axis, src);
                }
                continue;
            }

            // Domain boundary: dispatch on face BC type
            if (bc_is_periodic(bcs[d])) {
                PeriodicSrc psrc = periodic_src(nd, d);
                if (psrc.blk && psrc.level_rel < 0) {
                    fill_cf_ghosts(blk, *psrc.blk, child_octant_of(nodes, li), axis, side);
                    continue;
                }
                const CellBlock& src = psrc.blk ? *psrc.blk : blk;
                for (int gl = 0; gl < NG; ++gl) {
                    copy_strip((side==0)?(NG-1-gl):(NB2-NG+gl),
                               (side==0)?(ihi()-gl):(ilo()+gl), axis, src);
                }
            } else if (std::holds_alternative<WallBC>(bcs[d]) ||
                       std::holds_alternative<ContactAngleBC>(bcs[d])) {
                for (int gl = 0; gl < NG; ++gl) {
                    const int ghost = (side==0) ? (NG-1-gl)   : (NB2-NG+gl);
                    const int mirr  = (side==0) ? (ilo()+gl)  : (ihi()-gl);
                    const int ref   = (side==0) ? ilo()       : ihi();
                    const int dist  = (side==0) ? (ilo()-ghost) : (ghost-ihi());
                    for (int a = ilo(); a <= ihi(); ++a)
                    for (int b = ilo(); b <= ihi(); ++b) {
                        const int gi=(axis==0)?ghost:a, gj=(axis==1)?ghost:(axis==0)?a:b, gk=(axis==2)?ghost:b;
                        const int mi=(axis==0)?mirr:a,  mj=(axis==1)?mirr:(axis==0)?a:b,  mk=(axis==2)?mirr:b;
                        const int ri=(axis==0)?ref:a,   rj=(axis==1)?ref:(axis==0)?a:b,   rk=(axis==2)?ref:b;
                        blk.rho (gi,gj,gk) =  blk.rho (mi,mj,mk);
                        blk.rhou(gi,gj,gk) = -blk.rhou(mi,mj,mk);
                        blk.rhov(gi,gj,gk) = -blk.rhov(mi,mj,mk);
                        blk.rhow(gi,gj,gk) = -blk.rhow(mi,mj,mk);
                        blk.E   (gi,gj,gk) =  wall_E(blk,mi,mj,mk);
                        blk.phi (gi,gj,gk) =  phi_wall_ghost(blk.phi(ri,rj,rk), dist);
                    }
                }
            } else {
                // OpenBC / NscbcBC: characteristic transmissive ghost
                const double p_ref  = std::holds_alternative<NscbcBC>(bcs[d])
                                    ? std::get<NscbcBC>(bcs[d]).p_inf : open_bc_p;
                const double outward = (side == 0) ? -1.0 : +1.0;
                for (int gl = 0; gl < NG; ++gl) {
                    const int ghost = (side==0) ? (NG-1-gl) : (NB2-NG+gl);
                    const int int_r = (side==0) ? ilo()     : ihi();
                    for (int a = ilo(); a <= ihi(); ++a)
                    for (int b = ilo(); b <= ihi(); ++b) {
                        const int gi=(axis==0)?ghost:a, gj=(axis==1)?ghost:(axis==0)?a:b, gk=(axis==2)?ghost:b;
                        const int ri=(axis==0)?int_r:a, rj=(axis==1)?int_r:(axis==0)?a:b, rk=(axis==2)?int_r:b;
                        write_ghost(gi,gj,gk, open_char_ghost(blk.prim(ri,rj,rk), axis, outward, p_ref));
                        blk.phi(gi,gj,gk) = blk.phi(ri,rj,rk);
                    }
                }
            }
        }

        // Edge/corner ghosts — copy_flat from already-filled face ghosts
        auto copy_flat = [&](int d, int s) noexcept {
            for (int v=0;v<NVAR;++v) blk.Q[v][d] = blk.Q[v][s];
            blk.phi_data_[d] = blk.phi_data_[s];
        };
        const bool xm=bnd[XMINUS], xp=bnd[XPLUS], ym=bnd[YMINUS];
        const bool yp=bnd[YPLUS],  zm=bnd[ZMINUS], zp=bnd[ZPLUS];
        for (int k=ilo();k<=ihi();++k)
        for (int glx=0; glx<NG; ++glx)
        for (int gly=0; gly<NG; ++gly) {
            if (xm||ym) copy_flat(cell_idx(glx,        gly,        k), cell_idx(glx,        ilo()+gly, k));
            if (xp||ym) copy_flat(cell_idx(NB2-NG+glx, gly,        k), cell_idx(NB2-NG+glx, ilo()+gly, k));
            if (xm||yp) copy_flat(cell_idx(glx,        NB2-NG+gly, k), cell_idx(glx,        ihi()-gly, k));
            if (xp||yp) copy_flat(cell_idx(NB2-NG+glx, NB2-NG+gly, k), cell_idx(NB2-NG+glx, ihi()-gly, k));
        }
        for (int j=ilo();j<=ihi();++j)
        for (int glx=0; glx<NG; ++glx)
        for (int glz=0; glz<NG; ++glz) {
            if (xm||zm) copy_flat(cell_idx(glx,        j, glz       ), cell_idx(glx,        j, ilo()+glz));
            if (xp||zm) copy_flat(cell_idx(NB2-NG+glx, j, glz       ), cell_idx(NB2-NG+glx, j, ilo()+glz));
            if (xm||zp) copy_flat(cell_idx(glx,        j, NB2-NG+glz), cell_idx(glx,        j, ihi()-glz));
            if (xp||zp) copy_flat(cell_idx(NB2-NG+glx, j, NB2-NG+glz), cell_idx(NB2-NG+glx, j, ihi()-glz));
        }
        for (int i=ilo();i<=ihi();++i)
        for (int gly=0; gly<NG; ++gly)
        for (int glz=0; glz<NG; ++glz) {
            if (ym||zm) copy_flat(cell_idx(i, gly,        glz       ), cell_idx(i, gly,        ilo()+glz));
            if (yp||zm) copy_flat(cell_idx(i, NB2-NG+gly, glz       ), cell_idx(i, NB2-NG+gly, ilo()+glz));
            if (ym||zp) copy_flat(cell_idx(i, gly,        NB2-NG+glz), cell_idx(i, gly,        ihi()-glz));
            if (yp||zp) copy_flat(cell_idx(i, NB2-NG+gly, NB2-NG+glz), cell_idx(i, NB2-NG+gly, ihi()-glz));
        }
        for (int glx=0; glx<NG; ++glx)
        for (int gly=0; gly<NG; ++gly)
        for (int glz=0; glz<NG; ++glz) {
            if (xm||ym||zm) copy_flat(cell_idx(glx,        gly,        glz       ), cell_idx(glx,        gly,        ilo()+glz));
            if (xp||ym||zm) copy_flat(cell_idx(NB2-NG+glx, gly,        glz       ), cell_idx(NB2-NG+glx, gly,        ilo()+glz));
            if (xm||yp||zm) copy_flat(cell_idx(glx,        NB2-NG+gly, glz       ), cell_idx(glx,        NB2-NG+gly, ilo()+glz));
            if (xp||yp||zm) copy_flat(cell_idx(NB2-NG+glx, NB2-NG+gly, glz       ), cell_idx(NB2-NG+glx, NB2-NG+gly, ilo()+glz));
            if (xm||ym||zp) copy_flat(cell_idx(glx,        gly,        NB2-NG+glz), cell_idx(glx,        gly,        ihi()-glz));
            if (xp||ym||zp) copy_flat(cell_idx(NB2-NG+glx, gly,        NB2-NG+glz), cell_idx(NB2-NG+glx, gly,        ihi()-glz));
            if (xm||yp||zp) copy_flat(cell_idx(glx,        NB2-NG+gly, NB2-NG+glz), cell_idx(glx,        NB2-NG+gly, ihi()-glz));
            if (xp||yp||zp) copy_flat(cell_idx(NB2-NG+glx, NB2-NG+gly, NB2-NG+glz), cell_idx(NB2-NG+glx, NB2-NG+gly, ihi()-glz));
        }
    }
}

// =============================================================================
// P1.4 — Flux register (Berger & Colella 1989)
// =============================================================================
void BlockTree::zero_flux_registers() {
    for (auto& nd : nodes) {
        if (!nd.is_leaf()) continue;
        for (auto& fr : nd.flux_reg)
            std::fill(fr.begin(), fr.end(), 0.0);
    }
}

// =============================================================================
// accumulate_fine_flux — direction convention:
//   d           = direction from FINE leaf to COARSE neighbour (fine's view)
//   d^1         = opposite = direction from COARSE to FINE (coarse's view)
//
// apply_flux_correction iterates over coarse face directions d_c and reads
//   coarse.flux_reg[d_c] where d_c = coarse-to-fine direction.
// Since d = fine-to-coarse, d_c = d^1 = opposite(d).
// =============================================================================
void BlockTree::accumulate_fine_flux(int fine_leaf, FaceDir d,
                                     const std::vector<double>& flux) {
    int ni = nodes[fine_leaf].neighbours[d];
    if (ni < 0) return;
    if (nodes[ni].level >= nodes[fine_leaf].level) return;

    // Correct direction: d is fine→coarse; apply_flux_correction reads
    // coarse.flux_reg[d_coarse_to_fine] = coarse.flux_reg[d^1] = opposite(d).
    auto& reg = nodes[ni].flux_reg[d ^ 1];
    const int face_size = NVAR * NB * NB;
    if (reg.size() != (size_t)face_size)
        reg.assign(face_size, 0.0);

    const double area_ratio = 0.25;
    for (int idx = 0; idx < face_size; ++idx)
        reg[idx] += flux[idx] * area_ratio;
}

// =============================================================================
// A05-fix3: apply_flux_correction — sign corrected per face direction
//
// Conservative update: dQ/dt = (1/h)*(F_left - F_right)
//   -face (side=0, XMINUS/YMINUS/ZMINUS): F enters the cell  → ADD
//   +face (side=1, XPLUS/YPLUS/ZPLUS):    F leaves the cell  → SUBTRACT
//
// flux_reg layout: reg[v*NB*NB + jc*NB + ic]
//   axis=0 (x-face, YZ plane): jc→y, ic→z
//   axis=1 (y-face, XZ plane): jc→z, ic→x
//   axis=2 (z-face, XY plane): jc→y, ic→x
// =============================================================================
void BlockTree::apply_flux_correction(double dt) {
    for (int li : leaf_indices()) {
        auto& nd  = nodes[li];
        if (!nd.has_block()) continue;  // P7.1: remote leaf
        auto& blk = *nd.block;

        for (int d = 0; d < NFACES; ++d) {
            auto& reg = nd.flux_reg[d];
            if (reg.empty()) continue;
            int ni = nd.neighbours[d];
            if (ni < 0 || !nodes[ni].has_block()) continue;
            if (nodes[ni].level <= nd.level) continue;

            const int axis = fd_axis(d);
            // Use axis-specific cell size for the dt/h correction factor.
            const double h_axis = (axis == 0) ? blk.h : (axis == 1) ? blk.hy : blk.hz;
            // +face subtracts (flux leaves cell), -face adds (flux enters cell)
            const double sign = (fd_side(d) == 1) ? -1.0 : +1.0;
            int g = (fd_side(d) == 0) ? ilo() : ihi();

            for (int jc = 0; jc < NB; ++jc)
            for (int ic = 0; ic < NB; ++ic) {
                const int flat = (axis == 0) ? cell_idx(g,        ilo()+jc, ilo()+ic) :
                                 (axis == 1) ? cell_idx(ilo()+ic, g,        ilo()+jc) :
                                               cell_idx(ilo()+ic, ilo()+jc, g       );
                const double k = sign * (dt / h_axis);
                for (int v = 0; v < NVAR; ++v)
                    blk.Q[v][flat] += k * reg[v*NB*NB + jc*NB + ic];
            }
        }
    }
}
