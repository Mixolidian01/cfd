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
//   A05  : alloc_node_group(n) — contiguous group allocation for refine()

#include "mesh/cell_block.hpp"
#include "mesh/bc_types.hpp"
#include <vector>
#include <array>
#include <memory>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>

// Forward-declare to avoid circular include (mpi_comm.hpp includes block_tree.hpp)
struct MpiPartition;

// ── Morton encoding (10 bits per axis) ────────────────────────────────────────────
uint32_t morton_encode(uint32_t x, uint32_t y, uint32_t z) noexcept;
void     morton_decode(uint32_t code, uint32_t& x, uint32_t& y, uint32_t& z) noexcept;

// Forward declaration so FaceBCArray can be used in BCRuntimeConfig.
using FaceBCArray = std::array<BCVariant, 6>;  // defined after FaceDir, but BCVariant is already visible

// ── BC runtime configuration (R9-A2: replaces file-scope statics) ──────────────────
struct BCRuntimeConfig {
    double wall_T       = 0.0;   // > 0 → isothermal wall; 0 → adiabatic
    double open_bc_p    = 0.0;   // > 0 → subsonic outflow pressure; 0 → transmissive
    double wall_ca_cos  = 0.0;   // cos(θ_w) for ACDI contact angle
    double wall_ca_ceps = 0.0;   // acdi_ceps at wall (0 → disabled)

    // When set, GhostFiller::fill_all() dispatches per-face instead of using a
    // single BCVariant for all six domain faces.  wall_T and open_bc_p are shared.
    std::optional<FaceBCArray> face_bc;
};

// ── Face direction enum ─────────────────────────────────────────────────────────────
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

// ── Sentinel for dead (freed) nodes ────────────────────────────────────────────
static constexpr int NODE_DEAD = -2;

// ── Tree node ───────────────────────────────────────────────────────────────────
struct BlockNode {
    // Tree topology
    int parent      = -1;   // index in BlockTree::nodes (-1 = root)
    int first_child = -1;   // index of first of 8 children; -1 = leaf
    std::array<int, NFACES> neighbours;  // leaf neighbour indices (-1 = boundary)

    // AMR metadata
    int      level  = 0;    // refinement level (0 = coarsest)
    uint32_t morton = 0;    // Morton code at this level
    double   ox = 0.0, oy = 0.0, oz = 0.0;  // block origin — valid even when block is null

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

// ── BlockTree ────────────────────────────────────────────────────────────────────
struct BlockTree {
    std::vector<BlockNode> nodes;

    // ── Construction ────────────────────────────────────────────────────
    void init(double L);

    // ── Refinement / coarsening ─────────────────────────────────────────────────
    // refine: leaf → 8 children (prolongates Q piecewise-constant)
    void refine(int idx);
    // coarsen: 8 siblings → parent (restricts Q volume-averaged)
    // P1.1: no tail assumption — uses free-list
    void coarsen(int parent_idx);

    // ── 2:1 balance ──────────────────────────────────────────────────────────
    // P1.2: work-queue model — correct after any topology change
    int balance();

    // ── Accessors ──────────────────────────────────────────────────────────
    int  n_leaves() const noexcept;
    int  root()     const noexcept { return 0; }
    bool valid()    const noexcept { return !nodes.empty(); }

    // P1.6: cached leaf index list; invalidated by refine/coarsen/rebuild.
    const std::vector<int>& leaf_indices() const;

    // Morton-sorted leaf indices — cached alongside leaf_cache_.
    // Returns slot indices (positions in leaf_indices()) sorted by Morton code.
    // O(1) after first call; invalidated when leaf_dirty_ is set.
    const std::vector<int>& morton_leaf_indices() const;

    // P4.1: max/min refinement level that has at least one leaf.
    int max_leaf_level() const noexcept;
    int min_leaf_level() const noexcept;

    // ── Neighbour topology ────────────────────────────────────────────────────
    void rebuild_neighbours();

    // R9-A2: BC runtime configuration — assign fields directly before advancing.
    BCRuntimeConfig bc_cfg;

    // ── Ghost fill ─────────────────────────────────────────────────────────────
    // P1.3: dispatches fill_cf_ghosts for coarse-fine faces
    // cf_zero_grad=false (default): coarse C/F ghosts filled from fine cell averages.
    // cf_zero_grad=true  (LTS coarse step): coarse C/F ghosts use zero-gradient
    //   extrapolation (ghost = interior).  This makes viscous flux at C/F exactly
    //   zero, so total-energy conservation holds after the Berger-Colella correction.
    void fill_ghosts_periodic(bool cf_zero_grad = false);
    void fill_ghosts_wall    (bool cf_zero_grad = false);
    void fill_ghosts_open    (bool cf_zero_grad = false); // zero-gradient transmissive

    // Per-face ghost fill: each of the 6 domain faces can independently be
    // Periodic, Wall, Open, or ContactAngleBC.  Interior / C/F faces are handled
    // identically to the uniform variants; only domain-boundary faces differ.
    // wall_T and open_bc_p come from bc_cfg (shared across faces).
    void fill_ghosts_per_face(const FaceBCArray& bcs, bool cf_zero_grad = false);

    // ── Flux register management (P1.4) ───────────────────────────────────────
    void zero_flux_registers();
    void accumulate_fine_flux(int fine_leaf, FaceDir d,
                              const std::vector<double>& flux);
    void apply_flux_correction(double dt);

    // ── Morton utilities ──────────────────────────────────────────────────────────
    static uint32_t child_morton(uint32_t parent_code, int oct) noexcept;

    double domain_L() const noexcept { return domain_L_; }

    // P4.1-fix: periodic BC flags — set by NSSolver::init() before any refine/balance.
    // When an axis is true, rebuild_neighbours() wraps domain-boundary faces on that
    // axis so that periodic C/F interfaces get correct neighbour links.
    bool periodic_axis_[3] = {false, false, false};
    void set_periodic(bool p) noexcept {
        periodic_axis_[0] = p; periodic_axis_[1] = p; periodic_axis_[2] = p;
    }
    void set_periodic_axes(bool x, bool y, bool z) noexcept {
        periodic_axis_[0] = x; periodic_axis_[1] = y; periodic_axis_[2] = z;
    }

    // P7.1: optional MPI partition.  When set, fill_ghosts_* skips remote faces
    // (they are already filled by mpi_exchange_halos() before the ghost fill).
    MpiPartition* mpi_ = nullptr;
    void set_mpi(MpiPartition* p) noexcept { mpi_ = p; }

    // P8.1: GPU lifecycle hooks.
    // on_block_alloc_: called after a leaf CellBlock's CPU data is ready
    //   (after prolongation in refine; after restriction in coarsen).
    //   Typical use: GpuPool::alloc + upload.
    // on_block_free_: called before a CellBlock is destroyed
    //   (parent in refine; children in coarsen).
    //   Typical use: GpuPool::free.
    // Both default to nullptr (disabled).
    void set_gpu_callbacks(std::function<void(CellBlock*)> alloc_cb,
                           std::function<void(CellBlock*)> free_cb) noexcept {
        on_block_alloc_ = std::move(alloc_cb);
        on_block_free_  = std::move(free_cb);
    }

    // D1: GPU-native AMR callbacks — when set, refine()/coarsen() bypass CPU
    // prolongation/restriction and leave all Q data movement to these callbacks.
    //
    // on_gpu_prolong_: called after child nodes are created but before the parent
    //   block is destroyed.  The callback must:
    //   (a) allocate GPU buffers for each child, (b) launch k_prolong D2D from
    //   parent GPU → children GPU, (c) free the parent GPU buffer.
    //   Signature: (parent_blk*, [child_blk* × 8])
    //
    // on_gpu_coarsen_: called with the parent and its 8 children still alive.
    //   The callback must:
    //   (a) allocate a GPU buffer for the parent, (b) launch k_restrict D2D from
    //   children GPU → parent GPU, (c) free all 8 child GPU buffers.
    //   Signature: (parent_blk*, [child_blk* × 8])
    //
    // When set, the CPU on_block_alloc_/on_block_free_ callbacks are ignored for
    // refine/coarsen (but may still be called for initial alloc at startup).
    void set_gpu_amr_callbacks(
        std::function<void(CellBlock*, CellBlock* const[8])> prolong_cb,
        std::function<void(CellBlock*, CellBlock* const[8])> coarsen_cb) noexcept {
        on_gpu_prolong_  = std::move(prolong_cb);
        on_gpu_coarsen_  = std::move(coarsen_cb);
    }

private:
    double domain_L_ = 1.0;

    // P8.1: GPU lifecycle callbacks (see set_gpu_callbacks above)
    std::function<void(CellBlock*)> on_block_alloc_;
    std::function<void(CellBlock*)> on_block_free_;

    // D1: GPU-native prolong/restrict callbacks (see set_gpu_amr_callbacks above)
    std::function<void(CellBlock*, CellBlock* const[8])> on_gpu_prolong_;
    std::function<void(CellBlock*, CellBlock* const[8])> on_gpu_coarsen_;

    // P1.1: free-list allocator
    // alloc_node()       — pop one slot (or append)
    // alloc_node_group(n)— pop/append n CONTIGUOUS slots; required by refine()
    //                       so that first_child+oct is always valid.
    std::vector<int> free_list_;
    int  alloc_node();
    int  alloc_node_group(int n);   // A05-fix: contiguous group allocation
    void free_node(int idx);

    // P1.6: leaf cache
    mutable std::vector<int> leaf_cache_;
    mutable bool             leaf_dirty_ = true;
    void invalidate_leaf_cache() const noexcept { leaf_dirty_ = true; }

    mutable std::vector<int> morton_leaf_cache_;

    void set_child_geometry(int parent_idx, int child_local, int child_idx);
    void prolongate_to_children(int parent_idx);
    void restrict_to_parent(int parent_idx);
};
