#pragma once
// DESIGN.md reference: Layer 1 — Block Tree (AMR octree)
//
// BlockTree manages an octree of CellBlocks.
// - Each node is either a LEAF (holds a CellBlock) or INTERNAL (fields freed,
//   8 children allocated).
// - Morton codes: 30-bit z-curve key, 10 bits per axis (max 1024^3 at finest).
// - 2:1 balance: no two adjacent leaf nodes differ by more than 1 refinement level.
// - Flux registers: at each coarse-fine interface, fine fluxes are accumulated
//   and used to correct the coarse update (exact conservation).
//
// Neighbour convention (6 face directions):
//   XMINUS=0  XPLUS=1  YMINUS=2  YPLUS=3  ZMINUS=4  ZPLUS=5
//
// P1 fix log:
//   P1.1 : free-list node allocator (alloc_node / free_node)
//   P1.2 : balance() work-queue model
//   P1.3 : fill_cf_ghosts dispatched from fill_ghosts_periodic/_wall
//   P1.4 : flux register implemented (accumulate_fine_flux / apply_flux_correction)
//   P1.5 : regrid() full Berger-Colella protocol
//   P1.6 : leaf cache with dirty flag (leaf_cache_ / leaf_dirty_)

#include "cell_block.hpp"
#include <vector>
#include <array>
#include <memory>
#include <cassert>
#include <cstdint>
#include <deque>

// ── Morton encoding (10 bits per axis) ───────────────────────────────────────
uint32_t morton_encode(uint32_t x, uint32_t y, uint32_t z) noexcept;
void     morton_decode(uint32_t code, uint32_t& x, uint32_t& y, uint32_t& z) noexcept;

// ── Face direction enum ───────────────────────────────────────────────────────
enum FaceDir : int {
    XMINUS = 0, XPLUS  = 1,
    YMINUS = 2, YPLUS  = 3,
    ZMINUS = 4, ZPLUS  = 5,
    NFACES = 6
};
inline FaceDir opposite(FaceDir d) noexcept {
    return static_cast<FaceDir>(d ^ 1);   // XM<->XP, YM<->YP, ZM<->ZP
}

// ── Canonical octant bit layout (fix #1 / improvement III) ──────────────────
// bit 0 = x-high, bit 1 = y-high, bit 2 = z-high
// All octant arithmetic in block_tree.cpp AND amr_operators.cpp must use these.
inline int  oct_from_xyz(int ix, int iy, int iz) noexcept { return ix | (iy<<1) | (iz<<2); }
inline int  oct_ix(int oct) noexcept { return  oct     & 1; }
inline int  oct_iy(int oct) noexcept { return (oct>>1) & 1; }
inline int  oct_iz(int oct) noexcept { return (oct>>2) & 1; }

// ── Sentinel for dead (freed) nodes ──────────────────────────────────────────
static constexpr int NODE_DEAD = -2;

// ── Tree node ─────────────────────────────────────────────────────────────────
struct BlockNode {
    // Tree topology
    int parent      = -1;   // index in BlockTree::nodes (-1 = root)
    int first_child = -1;   // index of first of 8 children; -1 = leaf
    std::array<int, NFACES> neighbours;  // leaf neighbour indices (-1 = boundary)

    // AMR metadata
    int      level  = 0;    // refinement level (0 = coarsest)
    uint32_t morton = 0;    // Morton code at this level

    // Field storage (null for internal nodes after refine()).
    // Cell size h is always read from block->h — never from a separate field.
    std::unique_ptr<CellBlock> block;

    // Flux registers: one per face, size = NVAR*NB*NB.
    // Populated only on leaves adjacent to a finer-level neighbour.
    // Layout: flux_reg[face][var*NB*NB + jf*NB + if_]
    std::array<std::vector<double>, NFACES> flux_reg;

    bool is_leaf()   const noexcept { return first_child == -1 && parent != NODE_DEAD; }
    bool has_block() const noexcept { return block != nullptr; }
    bool is_dead()   const noexcept { return parent == NODE_DEAD; }

    BlockNode() { neighbours.fill(-1); }

    // Reset to default-constructed state (used by free-list reuse).
    void reset() noexcept {
        parent = -1; first_child = -1; level = 0; morton = 0;
        neighbours.fill(-1);
        block.reset();
        for (auto& fr : flux_reg) fr.clear();
    }
};

// ── BlockTree ─────────────────────────────────────────────────────────────────
struct BlockTree {
    std::vector<BlockNode> nodes;

    // ── Construction ─────────────────────────────────────────────────────
    void init(double L);

    // ── Refinement / coarsening ───────────────────────────────────────────────
    // refine: leaf → 8 children (prolongates Q piecewise-constant)
    void refine(int idx);
    // coarsen: 8 siblings → parent (restricts Q volume-averaged)
    // P1.1: no tail assumption — uses free-list
    void coarsen(int parent_idx);

    // ── 2:1 balance ───────────────────────────────────────────────────────
    // P1.2: work-queue model — correct after any topology change
    int balance();

    // ── Accessors ─────────────────────────────────────────────────────────
    int  n_leaves() const noexcept;
    int  root()     const noexcept { return 0; }
    bool valid()    const noexcept { return !nodes.empty(); }

    // P1.6: cached leaf index list; invalidated by refine/coarsen/rebuild.
    const std::vector<int>& leaf_indices() const;

    // ── Neighbour topology ────────────────────────────────────────────────
    void rebuild_neighbours();

    // ── Ghost fill ────────────────────────────────────────────────────────
    // P1.3: dispatches fill_cf_ghosts for coarse-fine faces
    void fill_ghosts_periodic();
    void fill_ghosts_wall();

    // ── Flux register management (P1.4) ───────────────────────────────────
    void zero_flux_registers();
    void accumulate_fine_flux(int fine_leaf, FaceDir d,
                              const std::vector<double>& flux);
    void apply_flux_correction(double dt);

    // ── Morton utilities ──────────────────────────────────────────────────
    static uint32_t child_morton(uint32_t parent_code, int oct) noexcept;

    double domain_L() const noexcept { return domain_L_; }

private:
    double domain_L_ = 1.0;

    // P1.1: free-list allocator
    std::vector<int> free_list_;
    int  alloc_node();
    void free_node(int idx);

    // P1.6: leaf cache
    mutable std::vector<int> leaf_cache_;
    mutable bool             leaf_dirty_ = true;
    void invalidate_leaf_cache() const noexcept { leaf_dirty_ = true; }

    void set_child_geometry(int parent_idx, int child_local, int child_idx);
    void prolongate_to_children(int parent_idx);
    void restrict_to_parent(int parent_idx);
};
