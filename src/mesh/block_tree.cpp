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
    double dt_conv = (lam_max > 1e-300) ? cfl * h / lam_max : 1e300;
    double dt_visc = (nu_max  > 1e-300) ? h * h / (2.0 * C_VISC * nu_max) : 1e300;
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
void BlockTree::init(double L) {
    domain_L_ = L;
    nodes.clear();
    free_list_.clear();
    leaf_dirty_ = true;
    int root_idx = alloc_node();
    assert(root_idx == 0);
    auto& root  = nodes[0];
    root.level  = 0;
    root.morton = 0;
    root.parent = -1;
    root.ox = 0.0; root.oy = 0.0; root.oz = 0.0;
    root.block  = std::make_unique<CellBlock>(0.0, 0.0, 0.0, L / NB);
}

// =============================================================================
// child geometry
// =============================================================================
void BlockTree::set_child_geometry(int parent_idx, int child_local, int child_idx) {
    const auto& par = nodes[parent_idx];
    double cell_h = par.block ? par.block->h * 0.5
                              : domain_L_ / (NB * (1 << (par.level + 1)));
    // Use node-level ox/oy/oz (valid even when block is null after internal refine).
    double ox = par.ox;
    double oy = par.oy;
    double oz = par.oz;
    double half = cell_h * NB;
    if (oct_ix(child_local)) ox += half;
    if (oct_iy(child_local)) oy += half;
    if (oct_iz(child_local)) oz += half;
    nodes[child_idx].ox = ox;
    nodes[child_idx].oy = oy;
    nodes[child_idx].oz = oz;
    nodes[child_idx].block = std::make_unique<CellBlock>(ox, oy, oz, cell_h);
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
    CellBlock parent_data = *nodes[idx].block;  // cache before any resize

    // P8.1: free GPU buffer for parent before replacing the block.
    if (on_block_free_) on_block_free_(nodes[idx].block.get());

    // A05-fix: allocate 8 children as a CONTIGUOUS group so that
    // first_child + oct is always valid regardless of free-list state.
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
        set_child_geometry(idx, oct, ci);
    }

    nodes[idx].block = std::make_unique<CellBlock>(parent_data);
    prolongate_to_children(idx);

    // P8.1: alloc GPU buffers for children after prolongation (CPU data ready).
    if (on_block_alloc_) {
        for (int oct = 0; oct < 8; ++oct)
            on_block_alloc_(nodes[first + oct].block.get());
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
        double h_par = nodes[fc].block->h * 2.0;
        double ox    = nodes[fc].block->ox;
        double oy    = nodes[fc].block->oy;
        double oz    = nodes[fc].block->oz;
        nodes[parent_idx].block =
            std::make_unique<CellBlock>(ox, oy, oz, h_par);
    }
    restrict_to_parent(parent_idx);

    // P8.1: alloc GPU buffer for parent after restriction (CPU data ready).
    if (on_block_alloc_) on_block_alloc_(nodes[parent_idx].block.get());

    nodes[parent_idx].first_child = -1;

    // P8.1: free GPU buffers for children before free_node destroys the blocks.
    if (on_block_free_) {
        for (int oct = 0; oct < 8; ++oct)
            on_block_free_(nodes[fc + oct].block.get());
    }

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

            // P4.1-fix: periodic wrapping — when at a domain boundary and the
            // solver uses periodic BC, wrap the coordinate so that C/F interfaces
            // at the domain edge get proper neighbour links.  Without this,
            // accumulate_cf_fine_fluxes and undo_cf_face_flux both skip ni<0,
            // and the flux register correction is never applied at periodic C/F
            // faces, breaking mass conservation in AMR+periodic configurations.
            if (nb_code == UINT32_MAX && periodic_bc_ && lev > 0) {
                uint32_t mx, my, mz;
                morton_decode(a.morton, mx, my, mz);
                uint32_t max_coord = (1u << lev) - 1u;
                if      (axis == 0) mx = (delta > 0) ? 0u : max_coord;
                else if (axis == 1) my = (delta > 0) ? 0u : max_coord;
                else                mz = (delta > 0) ? 0u : max_coord;
                nb_code = morton_encode(mx, my, mz);
            }

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
    invalidate_leaf_cache();  // topology changed — must rebuild cache next call
    leaf_dirty_ = false;      // but we just rebuilt it above, so mark clean
    // (leaf_cache_ is already populated by the leaf_indices() call at top)
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
    if (axis==0) g.u = p.u + delta_u_n;
    else if (axis==1) g.v = p.v + delta_u_n;
    else g.w = p.w + delta_u_n;
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

    // A05-fix5: verify that the fine children have the expected cell size.
    if (!nodes[first_child].has_block()) return;
    {
        const double h_fine_expected = coarse_blk.h * 0.5;
        const double h_fine_actual   = nodes[first_child].block->h;
        if (std::fabs(h_fine_actual - h_fine_expected) > 1e-12 * coarse_blk.h)
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
            int oix, oiy, oiz;
            if (axis == 0) {
                oix = side;     oiy = ia_blk;  oiz = ib_blk;
            } else if (axis == 1) {
                oix = ia_blk;   oiy = side;    oiz = ib_blk;
            } else {
                oix = ia_blk;   oiy = ib_blk;  oiz = side;
            }
            int fine_oct = oct_from_xyz(oix, oiy, oiz);
            int fi = first_child + fine_oct;

            if (fi < 0 || fi >= (int)nodes.size()) continue;
            if (!nodes[fi].has_block()) continue;
            const CellBlock& fsrc = *nodes[fi].block;

            int fa_start = NG + 2 * (a_local % half);
            int fb_start = NG + 2 * (b_local % half);

            int gi, gj, gk;
            if (axis == 0) { gi=g; gj=a; gk=b; }
            else if (axis==1) { gi=a; gj=g; gk=b; }
            else              { gi=a; gj=b; gk=g; }

            for (int v = 0; v < NVAR; ++v) {
                double avg = 0.0;
                for (int da = 0; da < 2; ++da)
                for (int db = 0; db < 2; ++db) {
                    int fa = fa_start + da;
                    int fb = fb_start + db;
                    int ci, cj, ck;
                    if (axis == 0) { ci=face_i; cj=fa; ck=fb; }
                    else if (axis==1) { ci=fa; cj=face_i; ck=fb; }
                    else              { ci=fa; cj=fb; ck=face_i; }
                    avg += fsrc.Q[v][cell_idx(ci, cj, ck)];
                }
                coarse_blk.Q[v][cell_idx(gi, gj, gk)] = avg * 0.25;
            }

            // P14.1: phi — same 2×2 cell average
            {
                double phi_avg = 0.0;
                for (int da = 0; da < 2; ++da)
                for (int db = 0; db < 2; ++db) {
                    int fa = fa_start + da;
                    int fb = fb_start + db;
                    int ci, cj, ck;
                    if (axis == 0) { ci=face_i; cj=fa; ck=fb; }
                    else if (axis==1) { ci=fa; cj=face_i; ck=fb; }
                    else              { ci=fa; cj=fb; ck=face_i; }
                    phi_avg += fsrc.phi_data_[cell_idx(ci, cj, ck)];
                }
                coarse_blk.phi_data_[cell_idx(gi, gj, gk)] = phi_avg * 0.25;
            }
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
            int gi, gj, gk, ii, ij, ik;
            if (axis == 0) { gi=g; gj=a; gk=b;  ii=inner; ij=a; ik=b; }
            else if(axis==1){ gi=a; gj=g; gk=b;  ii=a; ij=inner; ik=b; }
            else             { gi=a; gj=b; gk=g;  ii=a; ij=b; ik=inner; }
            for (int v = 0; v < NVAR; ++v)
                coarse_blk.Q[v][cell_idx(gi,gj,gk)] =
                    coarse_blk.Q[v][cell_idx(ii,ij,ik)];
            coarse_blk.phi_data_[cell_idx(gi,gj,gk)] =   // P14.1: ∂φ/∂n=0
                coarse_blk.phi_data_[cell_idx(ii,ij,ik)];
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

    // Helper: find periodic-wrap block for face d of node nd when ni==-1.
    // Returns {block pointer, neighbor_level - nd.level}; pointer is nullptr if
    // not found (caller falls back to self-copy / zero-gradient).
    struct PeriodicSrc { const CellBlock* blk; int level_rel; };
    auto periodic_src = [&](const BlockNode& nd, int d) -> PeriodicSrc {
        static constexpr int face_axis[NFACES]  = {0,0,1,1,2,2};
        static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1};
        int axis  = face_axis[d];
        int delta = face_delta[d];
        int lev   = nd.level;
        if (lev == 0) return {nullptr, 0};  // single-block root → wrap to self
        uint32_t mx, my, mz;
        morton_decode(nd.morton, mx, my, mz);
        uint32_t max_coord = (1u << lev) - 1u;
        if (axis == 0) mx = (delta > 0) ? 0 : max_coord;
        else if (axis == 1) my = (delta > 0) ? 0 : max_coord;
        else               mz = (delta > 0) ? 0 : max_coord;
        uint64_t key = ((uint64_t)lev << 32) | morton_encode(mx, my, mz);
        auto it = lm_map.find(key);
        if (it != lm_map.end() && nodes[it->second].has_block())
            return {nodes[it->second].block.get(), 0};
        // Coarser periodic neighbor (2:1 balance): check level-1
        if (lev > 1) {
            uint32_t pc = morton_encode(mx, my, mz) >> 3;
            uint64_t pk = ((uint64_t)(lev-1) << 32) | pc;
            auto it2 = lm_map.find(pk);
            if (it2 != lm_map.end() && nodes[it2->second].has_block())
                return {nodes[it2->second].block.get(), -1};
        }
        return {nullptr, 0};
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
                if (sp.axis == 0) {
                    for (int k=ilo();k<=ihi();++k)
                    for (int j=ilo();j<=ihi();++j)
                        copy_cell(g_idx,j,k, src_idx,j,k, src);
                } else if (sp.axis == 1) {
                    for (int k=ilo();k<=ihi();++k)
                    for (int i=ilo();i<=ihi();++i)
                        copy_cell(i,g_idx,k, i,src_idx,k, src);
                } else {
                    for (int j=ilo();j<=ihi();++j)
                    for (int i=ilo();i<=ihi();++i)
                        copy_cell(i,j,g_idx, i,j,src_idx, src);
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
    const auto& leaves = leaf_indices();
    for (int li : leaves) {
        auto& nd  = nodes[li];
        if (!nd.has_block()) continue;  // P7.1: remote leaf
        auto& blk = *nd.block;

        // P13.4: isothermal ghost energy: E_g = ρ·Cv·(2Tw−Ti) + ½ρ|u|²
        // Adiabatic (bc_cfg.wall_T == 0): copy E directly (∂T/∂n = 0 at wall).
        const double wall_T    = bc_cfg.wall_T;
        const double ca_cos    = bc_cfg.wall_ca_cos;
        const double ca_ceps   = bc_cfg.wall_ca_ceps;
        auto wall_E = [wall_T](const CellBlock& b, int mi, int mj, int mk) noexcept {
            if (wall_T <= 0.0) return b.E(mi, mj, mk);
            const Prim p = b.prim(mi, mj, mk);
            const double T_ghost = 2.0*wall_T - p.T;
            const double KE = 0.5*p.rho*(p.u*p.u + p.v*p.v + p.w*p.w);
            return p.rho * (R_GAS/(GAMMA-1.0)) * T_ghost + KE;
        };

        // P14.2: contact angle ghost phi — phi_ghost = phi_ref - dist*cos(θ)/(ceps)*g'(phi_ref)
        // g'(phi) = phi*(1-phi)*(1-2*phi)/2  (double-well derivative)
        // dist: ghost-cell distance from wall in cell spacings (1 for nearest, 2 for outermost).
        // With gi<ilo(): XMINUS wall, ref = ilo(); gi>ihi(): XPLUS wall, ref = ihi().
        auto phi_wall_ghost = [ca_cos, ca_ceps](double phi_ref, int dist) noexcept -> double {
            if (ca_ceps <= 0.0) return phi_ref;  // disabled → Neumann
            const double g_prime = 0.5 * phi_ref * (1.0 - phi_ref) * (1.0 - 2.0 * phi_ref);
            const double phi_g = phi_ref - dist * ca_cos / ca_ceps * g_prime;
            return (phi_g < 0.0) ? 0.0 : (phi_g > 1.0) ? 1.0 : phi_g;
        };

        auto wall_x = [&](int gi, int mi) noexcept {
            const int ref_i = (gi < ilo()) ? ilo() : ihi();
            const int dist  = (gi < ilo()) ? (ilo() - gi) : (gi - ihi());
            for (int k=ilo();k<=ihi();++k)
            for (int j=ilo();j<=ihi();++j) {
                blk.rho (gi,j,k) =  blk.rho (mi,j,k);
                blk.rhou(gi,j,k) = -blk.rhou(mi,j,k);
                blk.rhov(gi,j,k) = -blk.rhov(mi,j,k);
                blk.rhow(gi,j,k) = -blk.rhow(mi,j,k);
                blk.E   (gi,j,k) =  wall_E(blk, mi,j,k);
                blk.phi(gi,j,k)  =  phi_wall_ghost(blk.phi(ref_i,j,k), dist);
            }
        };
        auto wall_y = [&](int gj, int mj) noexcept {
            const int ref_j = (gj < ilo()) ? ilo() : ihi();
            const int dist  = (gj < ilo()) ? (ilo() - gj) : (gj - ihi());
            for (int k=ilo();k<=ihi();++k)
            for (int i=ilo();i<=ihi();++i) {
                blk.rho (i,gj,k) =  blk.rho (i,mj,k);
                blk.rhou(i,gj,k) = -blk.rhou(i,mj,k);
                blk.rhov(i,gj,k) = -blk.rhov(i,mj,k);
                blk.rhow(i,gj,k) = -blk.rhow(i,mj,k);
                blk.E   (i,gj,k) =  wall_E(blk, i,mj,k);
                blk.phi(i,gj,k)  =  phi_wall_ghost(blk.phi(i,ref_j,k), dist);
            }
        };
        auto wall_z = [&](int gk, int mk) noexcept {
            const int ref_k = (gk < ilo()) ? ilo() : ihi();
            const int dist  = (gk < ilo()) ? (ilo() - gk) : (gk - ihi());
            for (int j=ilo();j<=ihi();++j)
            for (int i=ilo();i<=ihi();++i) {
                blk.rho (i,j,gk) =  blk.rho (i,j,mk);
                blk.rhou(i,j,gk) = -blk.rhou(i,j,mk);
                blk.rhov(i,j,gk) = -blk.rhov(i,j,mk);
                blk.rhow(i,j,gk) = -blk.rhow(i,j,mk);
                blk.E   (i,j,gk) =  wall_E(blk, i,j,mk);
                blk.phi(i,j,gk)  =  phi_wall_ghost(blk.phi(i,j,ref_k), dist);
            }
        };

        const bool xm = (nd.neighbours[XMINUS] < 0);
        const bool xp = (nd.neighbours[XPLUS]  < 0);
        const bool ym = (nd.neighbours[YMINUS] < 0);
        const bool yp = (nd.neighbours[YPLUS]  < 0);
        const bool zm = (nd.neighbours[ZMINUS] < 0);
        const bool zp = (nd.neighbours[ZPLUS]  < 0);

        // P1.3 / A05-fix2 / B2: CF and same-level ghost fill on non-wall faces
        struct FaceSpec { int ghost_g, mirror_s, axis, side; };
        static const FaceSpec specs[NFACES] = {
            {0,      ihi(), 0, 0},  // XMINUS
            {NB2-1,  ilo(), 0, 1},  // XPLUS
            {0,      ihi(), 1, 0},  // YMINUS
            {NB2-1,  ilo(), 1, 1},  // YPLUS
            {0,      ihi(), 2, 0},  // ZMINUS
            {NB2-1,  ilo(), 2, 1},  // ZPLUS
        };

        auto copy_cell_wall = [&](int gi, int gj, int gk,
                                  int si, int sj, int sk,
                                  const CellBlock& src) noexcept {
            const int dst_flat = cell_idx(gi,gj,gk);
            const int src_flat = cell_idx(si,sj,sk);
            for (int v = 0; v < NVAR; ++v)
                blk.Q[v][dst_flat] = src.Q[v][src_flat];
            blk.phi_data_[dst_flat] = src.phi_data_[src_flat];  // P14.1
        };

        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            if (ni < 0 || !nodes[ni].has_block()) continue;
            if (mpi_is_remote(mpi_, ni)) continue;  // P7.1: filled by mpi_exchange_halos()
            const FaceSpec& sp = specs[d];
            if (nodes[ni].level < nd.level) {
                int oct = child_octant_of(nodes, li);
                fill_cf_ghosts(blk, *nodes[ni].block, oct, sp.axis, sp.side);
            } else if (nodes[ni].level > nd.level) {
                if (cf_zero_grad)
                    fill_coarse_ghost_zero_grad(blk, d);
                else
                    fill_coarse_ghost_from_fine(blk, nodes, ni, d);
            } else {
                // B2: same-level interior neighbour — direct face copy (all NG layers)
                const CellBlock& src = *nodes[ni].block;
                for (int gl = 0; gl < NG; ++gl) {
                    const int g_idx   = (sp.side==0) ? (NG-1-gl)    : (NB2-NG+gl);
                    const int src_idx = (sp.side==0) ? (ihi()-gl)    : (ilo()+gl);
                    if (sp.axis == 0) {
                        for (int k=ilo();k<=ihi();++k)
                        for (int j=ilo();j<=ihi();++j)
                            copy_cell_wall(g_idx,j,k, src_idx,j,k, src);
                    } else if (sp.axis == 1) {
                        for (int k=ilo();k<=ihi();++k)
                        for (int i=ilo();i<=ihi();++i)
                            copy_cell_wall(i,g_idx,k, i,src_idx,k, src);
                    } else {
                        for (int j=ilo();j<=ihi();++j)
                        for (int i=ilo();i<=ihi();++i)
                            copy_cell_wall(i,j,g_idx, i,j,src_idx, src);
                    }
                }
            }
        }

        // Wall ghost fill: all NG ghost layers (anti-symmetric momentum, symmetric rho/E)
        if (xm) { for (int gl=0;gl<NG;++gl) wall_x(NG-1-gl,   ilo()+gl); }
        if (xp) { for (int gl=0;gl<NG;++gl) wall_x(NB2-NG+gl, ihi()-gl); }
        if (ym) { for (int gl=0;gl<NG;++gl) wall_y(NG-1-gl,   ilo()+gl); }
        if (yp) { for (int gl=0;gl<NG;++gl) wall_y(NB2-NG+gl, ihi()-gl); }
        if (zm) { for (int gl=0;gl<NG;++gl) wall_z(NG-1-gl,   ilo()+gl); }
        if (zp) { for (int gl=0;gl<NG;++gl) wall_z(NB2-NG+gl, ihi()-gl); }

        // ── Edge ghosts ─────────────────────────────────────────────────────
        // Copy from the already-wall/neighbour-filled face ghost in the other direction.
        // Side=0 ghost gl → reads face ghost at ilo()+gl; side=1 ghost gl → reads at ihi()-gl.
        // P14.1: copy_flat propagates phi alongside Q in all edge/corner copies.
        auto copy_flat = [&](int d, int s) noexcept {
            for (int v=0;v<NVAR;++v) blk.Q[v][d] = blk.Q[v][s];
            blk.phi_data_[d] = blk.phi_data_[s];
        };
        // XY edges (k interior)
        for (int k=ilo();k<=ihi();++k)
        for (int glx=0; glx<NG; ++glx)
        for (int gly=0; gly<NG; ++gly) {
            if (xm||ym) copy_flat(cell_idx(glx,        gly,        k), cell_idx(glx,        ilo()+gly, k));
            if (xp||ym) copy_flat(cell_idx(NB2-NG+glx, gly,        k), cell_idx(NB2-NG+glx, ilo()+gly, k));
            if (xm||yp) copy_flat(cell_idx(glx,        NB2-NG+gly, k), cell_idx(glx,        ihi()-gly, k));
            if (xp||yp) copy_flat(cell_idx(NB2-NG+glx, NB2-NG+gly, k), cell_idx(NB2-NG+glx, ihi()-gly, k));
        }
        // XZ edges (j interior)
        for (int j=ilo();j<=ihi();++j)
        for (int glx=0; glx<NG; ++glx)
        for (int glz=0; glz<NG; ++glz) {
            if (xm||zm) copy_flat(cell_idx(glx,        j, glz       ), cell_idx(glx,        j, ilo()+glz));
            if (xp||zm) copy_flat(cell_idx(NB2-NG+glx, j, glz       ), cell_idx(NB2-NG+glx, j, ilo()+glz));
            if (xm||zp) copy_flat(cell_idx(glx,        j, NB2-NG+glz), cell_idx(glx,        j, ihi()-glz));
            if (xp||zp) copy_flat(cell_idx(NB2-NG+glx, j, NB2-NG+glz), cell_idx(NB2-NG+glx, j, ihi()-glz));
        }
        // YZ edges (i interior)
        for (int i=ilo();i<=ihi();++i)
        for (int gly=0; gly<NG; ++gly)
        for (int glz=0; glz<NG; ++glz) {
            if (ym||zm) copy_flat(cell_idx(i, gly,        glz       ), cell_idx(i, gly,        ilo()+glz));
            if (yp||zm) copy_flat(cell_idx(i, NB2-NG+gly, glz       ), cell_idx(i, NB2-NG+gly, ilo()+glz));
            if (ym||zp) copy_flat(cell_idx(i, gly,        NB2-NG+glz), cell_idx(i, gly,        ihi()-glz));
            if (yp||zp) copy_flat(cell_idx(i, NB2-NG+gly, NB2-NG+glz), cell_idx(i, NB2-NG+gly, ihi()-glz));
        }

        // ── Corner ghosts ─────────────────────────────────────────────────────
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

// ── fill_ghosts_open (P13.3: characteristic open BC with optional p∞) ────────
// When bc_cfg.open_bc_p == 0: zero-gradient transmissive (legacy behaviour).
// When bc_cfg.open_bc_p >  0: subsonic outflow uses isentropic ghost + Riemann-
//   invariant velocity; HLLC-ES then adds entropy dissipation at the face.
//   Supersonic outflow and inflow fall back to zero-gradient.
void BlockTree::fill_ghosts_open(bool cf_zero_grad) {
    const auto& leaves = leaf_indices();
    for (int li : leaves) {
        auto& nd  = nodes[li];
        if (!nd.has_block()) continue;  // P7.1: remote leaf
        auto& blk = *nd.block;

        // Write a ghost Prim back into conserved Q of blk at index (gi,gj,gk)
        auto write_ghost = [&](int gi, int gj, int gk, const Prim& g) noexcept {
            blk.rho (gi,gj,gk) = g.rho;
            blk.rhou(gi,gj,gk) = g.rho * g.u;
            blk.rhov(gi,gj,gk) = g.rho * g.v;
            blk.rhow(gi,gj,gk) = g.rho * g.w;
            blk.E   (gi,gj,gk) = g.p/(GAMMA-1.0)
                                + 0.5*g.rho*(g.u*g.u+g.v*g.v+g.w*g.w);
        };

        const double open_bc_p = bc_cfg.open_bc_p;
        auto open_x = [&](int gi, int mi, double outward_sign) noexcept {
            for (int k=ilo();k<=ihi();++k)
            for (int j=ilo();j<=ihi();++j) {
                write_ghost(gi, j, k, open_char_ghost(blk.prim(mi,j,k), 0, outward_sign, open_bc_p));
                blk.phi(gi,j,k) = blk.phi(mi,j,k);  // P14.1: zero-gradient
            }
        };
        auto open_y = [&](int gj, int mj, double outward_sign) noexcept {
            for (int k=ilo();k<=ihi();++k)
            for (int i=ilo();i<=ihi();++i) {
                write_ghost(i, gj, k, open_char_ghost(blk.prim(i,mj,k), 1, outward_sign, open_bc_p));
                blk.phi(i,gj,k) = blk.phi(i,mj,k);  // P14.1: zero-gradient
            }
        };
        auto open_z = [&](int gk, int mk, double outward_sign) noexcept {
            for (int j=ilo();j<=ihi();++j)
            for (int i=ilo();i<=ihi();++i) {
                write_ghost(i, j, gk, open_char_ghost(blk.prim(i,j,mk), 2, outward_sign, open_bc_p));
                blk.phi(i,j,gk) = blk.phi(i,j,mk);  // P14.1: zero-gradient
            }
        };

        const bool xm = (nd.neighbours[XMINUS] < 0);
        const bool xp = (nd.neighbours[XPLUS]  < 0);
        const bool ym = (nd.neighbours[YMINUS] < 0);
        const bool yp = (nd.neighbours[YPLUS]  < 0);
        const bool zm = (nd.neighbours[ZMINUS] < 0);
        const bool zp = (nd.neighbours[ZPLUS]  < 0);

        // Interior/CF face fill — identical to wall version
        struct FaceSpec { int ghost_g, mirror_s, axis, side; };
        static const FaceSpec specs[NFACES] = {
            {0,      ihi(), 0, 0},
            {NB2-1,  ilo(), 0, 1},
            {0,      ihi(), 1, 0},
            {NB2-1,  ilo(), 1, 1},
            {0,      ihi(), 2, 0},
            {NB2-1,  ilo(), 2, 1},
        };
        auto copy_cell = [&](int gi, int gj, int gk, int si, int sj, int sk,
                              const CellBlock& src) noexcept {
            const int dst_flat = cell_idx(gi,gj,gk);
            const int src_flat = cell_idx(si,sj,sk);
            for (int v = 0; v < NVAR; ++v)
                blk.Q[v][dst_flat] = src.Q[v][src_flat];
            blk.phi_data_[dst_flat] = src.phi_data_[src_flat];  // P14.1
        };
        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            if (ni < 0 || !nodes[ni].has_block()) continue;
            if (mpi_is_remote(mpi_, ni)) continue;  // P7.1: filled by mpi_exchange_halos()
            const FaceSpec& sp = specs[d];
            if (nodes[ni].level < nd.level) {
                int oct = child_octant_of(nodes, li);
                fill_cf_ghosts(blk, *nodes[ni].block, oct, sp.axis, sp.side);
            } else if (nodes[ni].level > nd.level) {
                if (cf_zero_grad)
                    fill_coarse_ghost_zero_grad(blk, d);
                else
                    fill_coarse_ghost_from_fine(blk, nodes, ni, d);
            } else {
                const CellBlock& src = *nodes[ni].block;
                for (int gl = 0; gl < NG; ++gl) {
                    const int g_idx   = (sp.side==0) ? (NG-1-gl)    : (NB2-NG+gl);
                    const int src_idx = (sp.side==0) ? (ihi()-gl)    : (ilo()+gl);
                    if (sp.axis == 0) {
                        for (int k=ilo();k<=ihi();++k)
                        for (int j=ilo();j<=ihi();++j)
                            copy_cell(g_idx,j,k, src_idx,j,k, src);
                    } else if (sp.axis == 1) {
                        for (int k=ilo();k<=ihi();++k)
                        for (int i=ilo();i<=ihi();++i)
                            copy_cell(i,g_idx,k, i,src_idx,k, src);
                    } else {
                        for (int j=ilo();j<=ihi();++j)
                        for (int i=ilo();i<=ihi();++i)
                            copy_cell(i,j,g_idx, i,j,src_idx, src);
                    }
                }
            }
        }

        // Open boundary ghost fill (P13.3: characteristic or zero-gradient)
        if (xm) { for (int gl=0;gl<NG;++gl) open_x(NG-1-gl,   ilo(), -1.0); }
        if (xp) { for (int gl=0;gl<NG;++gl) open_x(NB2-NG+gl, ihi(), +1.0); }
        if (ym) { for (int gl=0;gl<NG;++gl) open_y(NG-1-gl,   ilo(), -1.0); }
        if (yp) { for (int gl=0;gl<NG;++gl) open_y(NB2-NG+gl, ihi(), +1.0); }
        if (zm) { for (int gl=0;gl<NG;++gl) open_z(NG-1-gl,   ilo(), -1.0); }
        if (zp) { for (int gl=0;gl<NG;++gl) open_z(NB2-NG+gl, ihi(), +1.0); }

        // Edge ghosts (same as wall version — copy from face-filled ghost)
        // P14.1: copy_flat also propagates phi.
        auto copy_flat_o = [&](int d, int s) noexcept {
            for (int v=0;v<NVAR;++v) blk.Q[v][d] = blk.Q[v][s];
            blk.phi_data_[d] = blk.phi_data_[s];
        };
        for (int k=ilo();k<=ihi();++k)
        for (int glx=0; glx<NG; ++glx)
        for (int gly=0; gly<NG; ++gly) {
            if (xm||ym) copy_flat_o(cell_idx(glx,        gly,        k), cell_idx(glx,        ilo()+gly, k));
            if (xp||ym) copy_flat_o(cell_idx(NB2-NG+glx, gly,        k), cell_idx(NB2-NG+glx, ilo()+gly, k));
            if (xm||yp) copy_flat_o(cell_idx(glx,        NB2-NG+gly, k), cell_idx(glx,        ihi()-gly, k));
            if (xp||yp) copy_flat_o(cell_idx(NB2-NG+glx, NB2-NG+gly, k), cell_idx(NB2-NG+glx, ihi()-gly, k));
        }
        for (int j=ilo();j<=ihi();++j)
        for (int glx=0; glx<NG; ++glx)
        for (int glz=0; glz<NG; ++glz) {
            if (xm||zm) copy_flat_o(cell_idx(glx,        j, glz       ), cell_idx(glx,        j, ilo()+glz));
            if (xp||zm) copy_flat_o(cell_idx(NB2-NG+glx, j, glz       ), cell_idx(NB2-NG+glx, j, ilo()+glz));
            if (xm||zp) copy_flat_o(cell_idx(glx,        j, NB2-NG+glz), cell_idx(glx,        j, ihi()-glz));
            if (xp||zp) copy_flat_o(cell_idx(NB2-NG+glx, j, NB2-NG+glz), cell_idx(NB2-NG+glx, j, ihi()-glz));
        }
        for (int i=ilo();i<=ihi();++i)
        for (int gly=0; gly<NG; ++gly)
        for (int glz=0; glz<NG; ++glz) {
            if (ym||zm) copy_flat_o(cell_idx(i, gly,        glz       ), cell_idx(i, gly,        ilo()+glz));
            if (yp||zm) copy_flat_o(cell_idx(i, NB2-NG+gly, glz       ), cell_idx(i, NB2-NG+gly, ilo()+glz));
            if (ym||zp) copy_flat_o(cell_idx(i, gly,        NB2-NG+glz), cell_idx(i, gly,        ihi()-glz));
            if (yp||zp) copy_flat_o(cell_idx(i, NB2-NG+gly, NB2-NG+glz), cell_idx(i, NB2-NG+gly, ihi()-glz));
        }
        // Corner ghosts
        for (int glx=0; glx<NG; ++glx)
        for (int gly=0; gly<NG; ++gly)
        for (int glz=0; glz<NG; ++glz) {
            if (xm||ym||zm) copy_flat_o(cell_idx(glx,        gly,        glz       ), cell_idx(glx,        gly,        ilo()+glz));
            if (xp||ym||zm) copy_flat_o(cell_idx(NB2-NG+glx, gly,        glz       ), cell_idx(NB2-NG+glx, gly,        ilo()+glz));
            if (xm||yp||zm) copy_flat_o(cell_idx(glx,        NB2-NG+gly, glz       ), cell_idx(glx,        NB2-NG+gly, ilo()+glz));
            if (xp||yp||zm) copy_flat_o(cell_idx(NB2-NG+glx, NB2-NG+gly, glz       ), cell_idx(NB2-NG+glx, NB2-NG+gly, ilo()+glz));
            if (xm||ym||zp) copy_flat_o(cell_idx(glx,        gly,        NB2-NG+glz), cell_idx(glx,        gly,        ihi()-glz));
            if (xp||ym||zp) copy_flat_o(cell_idx(NB2-NG+glx, gly,        NB2-NG+glz), cell_idx(NB2-NG+glx, gly,        ihi()-glz));
            if (xm||yp||zp) copy_flat_o(cell_idx(glx,        NB2-NG+gly, NB2-NG+glz), cell_idx(glx,        NB2-NG+gly, ihi()-glz));
            if (xp||yp||zp) copy_flat_o(cell_idx(NB2-NG+glx, NB2-NG+gly, NB2-NG+glz), cell_idx(NB2-NG+glx, NB2-NG+gly, ihi()-glz));
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
        double h_c = blk.h;

        for (int d = 0; d < NFACES; ++d) {
            auto& reg = nd.flux_reg[d];
            if (reg.empty()) continue;
            int ni = nd.neighbours[d];
            if (ni < 0 || !nodes[ni].has_block()) continue;
            if (nodes[ni].level <= nd.level) continue;

            const int axis = fd_axis(d);
            // +face subtracts (flux leaves cell), -face adds (flux enters cell)
            const double sign = (fd_side(d) == 1) ? -1.0 : +1.0;
            int g = (fd_side(d) == 0) ? ilo() : ihi();

            for (int v = 0; v < NVAR; ++v)
            for (int jc = 0; jc < NB; ++jc)
            for (int ic = 0; ic < NB; ++ic) {
                double corr = sign * (dt / h_c) * reg[v*NB*NB + jc*NB + ic];
                int ci, cj, ck;
                if      (axis == 0) { ci = g;         cj = ilo()+jc; ck = ilo()+ic; }
                else if (axis == 1) { ci = ilo()+ic;  cj = g;        ck = ilo()+jc; }
                else                { ci = ilo()+ic;  cj = ilo()+jc; ck = g;        }
                blk.Q[v][cell_idx(ci,cj,ck)] += corr;
            }
        }
    }
}
