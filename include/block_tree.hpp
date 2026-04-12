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

#include "cell_block.hpp"
#include <vector>
#include <array>
#include <memory>
#include <cassert>
#include <cstdint>

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

    // Flux registers: one per face, size = NB*NB per variable.
    // Allocated only on leaves adjacent to a finer-level leaf.
    std::array<std::vector<double>, NFACES> flux_reg;  // [face][var*NB*NB]

    bool is_leaf()    const noexcept { return first_child == -1; }
    bool has_block()  const noexcept { return block != nullptr;  }

    BlockNode() { neighbours.fill(-1); }
};

// ── BlockTree ─────────────────────────────────────────────────────────────────
struct BlockTree {
    std::vector<BlockNode> nodes;

    // ── Construction ─────────────────────────────────────────────────────
    // Create a single-block root covering [0,L]^3 at level 0.
    void init(double L);

    // ── Refinement / coarsening ───────────────────────────────────────────────
    // Refine leaf at index `idx` → 8 children.
    // Prolongates Q from parent to children (piecewise-constant).
    // Fix #16: parent state is cached before resize() to avoid UB.
    void refine(int idx);

    // Coarsen 8 siblings (children of `parent_idx`) → parent leaf.
    // Restricts Q from children to parent (volume-weighted average).
    // Precondition: all 8 children must be leaves.
    void coarsen(int parent_idx);

    // ── 2:1 balance ───────────────────────────────────────────────────────
    // Ensure no two adjacent leaf nodes differ by more than 1 level.
    // Returns number of extra refinements performed.
    int balance();

    // ── Accessors ─────────────────────────────────────────────────────────
    int  n_leaves() const noexcept;
    int  root()     const noexcept { return 0; }
    bool valid()    const noexcept { return !nodes.empty(); }

    // Returns leaf node indices.  O(n) scan; result is a fresh vector.
    std::vector<int> leaf_indices() const;

    // ── Neighbour topology ────────────────────────────────────────────────
    // Rebuild neighbour pointers for all leaves after refine/coarsen.
    void rebuild_neighbours();

    // ── Ghost fill ────────────────────────────────────────────────────────
    // Copy interior data from neighbour blocks into ghost cells.
    void fill_ghosts_periodic();   // periodic in all directions
    void fill_ghosts_wall();       // no-slip wall on all boundaries

    // ── Flux register management ──────────────────────────────────────────
    void zero_flux_registers();
    void accumulate_fine_flux(int fine_leaf, FaceDir d,
                              const std::vector<double>& flux);
    void apply_flux_correction();  // correct coarse cells at all CF interfaces

    // ── Morton utilities ──────────────────────────────────────────────────
    static uint32_t child_morton(uint32_t parent_code, int oct) noexcept;
    
    double domain_L() const noexcept { return domain_L_; }

private:
    double domain_L_ = 1.0;

    void set_child_geometry(int parent_idx, int child_local, int child_idx);
    void prolongate_to_children(int parent_idx);
    void restrict_to_parent(int parent_idx);
};
