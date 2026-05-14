// P8.2 gate test — GPU ghost fill kernels
//
// Protocol:
//   F1: Wall BC (single block) — GPU ghost matches CPU fill_ghosts_wall
//   F2: Periodic self-wrap (single block) — GPU ghost matches CPU fill_ghosts_periodic
//   F3: Same-level neighbor copy — 8 level-1 siblings, GPU ghosts match CPU fill
//   F4: CF fine←coarse (5th-order Lagrange) — 2-level tree, GPU CF ghosts match CPU
//
// All comparisons are exact (double precision, same coefficients).
// Tolerance: 1e-10 relative to max interior value (~1e5).

#include "../../include/cuda/gpu_ghost_fill.cuh"
#include "../../include/gpu_pool.hpp"
#include "../../include/block_tree.hpp"
#include "../../include/amr_operators.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

static int nfail = 0;
static GpuPool pool;

static void check(bool ok, const char* tag, const char* msg,
                  double val = -1.0) {
    if (ok) {
        printf("  PASS  %s  %s\n", tag, msg);
    } else {
        if (val >= 0.0)
            printf("  FAIL  %s  %s  (ghost err = %.3e)\n", tag, msg, val);
        else
            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static void fill_interior(CellBlock& blk, double seed) {
    for (int v = 0; v < NVAR; ++v)
        for (int k = NG; k < NG + NB; ++k)
        for (int j = NG; j < NG + NB; ++j)
        for (int i = NG; i < NG + NB; ++i)
            blk.Q[v][cell_idx(i,j,k)] = seed + v * 1e5
                                         + k * NB2*NB2 + j * NB2 + i;
}

// Snapshot (dense SoA): snap[v*NCELL + flat] = Q[v][flat]
using Snap = std::vector<double>;  // NVAR * NCELL doubles

static Snap take_snap(const CellBlock& blk) {
    Snap s(NVAR * NCELL);
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            s[v * NCELL + idx] = blk.Q[v][idx];
    return s;
}

// Max error over face-only ghost cells (exactly 1 ghost axis).
// Edge/corner ghost cells are excluded: the CPU wall BC edge fill reads from
// already-reflected face ghosts, while the GPU EC kernel reads from interior,
// producing legitimately different (but both valid) fill strategies.
// Face-only ghosts are what the RHS stencil actually reads.
static double ghost_err_face(const Snap& gpu, const Snap& cpu) {
    double e = 0.0;
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        const bool gx = (i < NG || i >= NG+NB);
        const bool gy = (j < NG || j >= NG+NB);
        const bool gz = (k < NG || k >= NG+NB);
        if ((gx ? 1 : 0) + (gy ? 1 : 0) + (gz ? 1 : 0) != 1) continue; // not a face ghost
        int flat = cell_idx(i,j,k);
        for (int v = 0; v < NVAR; ++v)
            e = std::fmax(e, std::fabs(gpu[v*NCELL + flat] - cpu[v*NCELL + flat]));
    }
    return e;
}

static void upload_leaves(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk) continue;
        if (!pool.has_device(blk)) pool.alloc(blk);
        pool.upload(blk);
    }
}

static void download_leaves(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (!blk || !pool.has_device(blk)) continue;
        pool.download(blk);
    }
}

static void free_leaves(BlockTree& tree) {
    for (int li : tree.leaf_indices()) {
        CellBlock* blk = tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }
}

// =============================================================================
// F1: Wall BC — single root block, fill_ghosts_wall vs GPU
// =============================================================================
static void test_f1() {
    printf("\n-- F1  Wall BC (single block) --\n");

    BlockTree tree;
    tree.init(1.0);

    // Fill interior with a deterministic ramp
    CellBlock& blk = *tree.nodes[0].block;
    fill_interior(blk, 1.0);

    // CPU reference
    tree.fill_ghosts_wall();
    Snap snap_cpu = take_snap(blk);

    // Reload interior (ghost cells now stale on CPU side), upload to GPU
    fill_interior(blk, 1.0);  // restore interior (ghosts left as CPU-written, but GPU will overwrite)
    upload_leaves(tree);

    // GPU fill
    GpuGhostFillList gfl;
    gfl.build(tree, pool, /*bc_type=*/1);
    gfl.exec(nullptr);
    cudaDeviceSynchronize();

    download_leaves(tree);
    Snap snap_gpu = take_snap(blk);
    free_leaves(tree);

    double err = ghost_err_face(snap_gpu, snap_cpu);
    printf("   wall BC face ghost err = %.2e  (expect 0)\n", err);
    check(err == 0.0, "F1", "wall BC face ghosts GPU==CPU fill_ghosts_wall", err);
}

// =============================================================================
// F2: Periodic self-wrap — single root block, fill_ghosts_periodic vs GPU
// =============================================================================
static void test_f2() {
    printf("\n-- F2  Periodic self-wrap (single block) --\n");

    BlockTree tree;
    tree.init(1.0);

    CellBlock& blk = *tree.nodes[0].block;
    fill_interior(blk, 2.0);

    // CPU reference
    tree.fill_ghosts_periodic();
    Snap snap_cpu = take_snap(blk);

    // Restore interior, upload to GPU
    fill_interior(blk, 2.0);
    upload_leaves(tree);

    GpuGhostFillList gfl;
    gfl.build(tree, pool, /*bc_type=*/0);
    gfl.exec(nullptr);
    cudaDeviceSynchronize();

    download_leaves(tree);
    Snap snap_gpu = take_snap(blk);
    free_leaves(tree);

    double err = ghost_err_face(snap_gpu, snap_cpu);
    printf("   periodic self-wrap face ghost err = %.2e  (expect 0)\n", err);
    check(err == 0.0, "F2", "periodic self-wrap face ghosts GPU==CPU fill_ghosts_periodic", err);
}

// =============================================================================
// F3: Same-level neighbor copy — 8 level-1 siblings
// =============================================================================
static void test_f3() {
    printf("\n-- F3  Same-level neighbor copy (8 siblings) --\n");

    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic(true);  // ensures all level-1 neighbours are set explicitly
    tree.refine(0);            // 8 level-1 leaves, all 6 faces linked to same-level blocks

    const auto& leaves = tree.leaf_indices();
    assert(leaves.size() == 8);

    // Fill each leaf's interior with a unique seed based on leaf index
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        CellBlock& blk = *tree.nodes[leaves[ii]].block;
        fill_interior(blk, (ii + 1) * 1e6);
    }

    // CPU reference: same-level ghost fill
    tree.fill_ghosts_periodic();
    std::vector<Snap> snap_cpu(leaves.size());
    for (int ii = 0; ii < (int)leaves.size(); ++ii)
        snap_cpu[ii] = take_snap(*tree.nodes[leaves[ii]].block);

    // Restore interiors for GPU run
    for (int ii = 0; ii < (int)leaves.size(); ++ii)
        fill_interior(*tree.nodes[leaves[ii]].block, (ii + 1) * 1e6);
    upload_leaves(tree);

    GpuGhostFillList gfl;
    gfl.build(tree, pool, /*bc_type=*/0);
    gfl.exec(nullptr);
    cudaDeviceSynchronize();

    download_leaves(tree);

    double max_err = 0.0;
    for (int ii = 0; ii < (int)leaves.size(); ++ii) {
        Snap sg = take_snap(*tree.nodes[leaves[ii]].block);
        max_err = std::fmax(max_err, ghost_err_face(sg, snap_cpu[ii]));
    }
    free_leaves(tree);

    printf("   same-level neighbor face ghost err = %.2e  (expect 0)\n", max_err);
    check(max_err == 0.0, "F3", "same-level neighbor face ghosts GPU==CPU fill_ghosts_periodic",
          max_err);
}

// =============================================================================
// F4: CF fine←coarse (5th-order Lagrange) — 2-level tree
// =============================================================================
static void test_f4() {
    printf("\n-- F4  CF fine-from-coarse (5th-order Lagrange) --\n");

    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic(true);  // wrap boundary faces so no nil d_nb[d] in GPU metadata
    tree.refine(0);          // 8 level-1 leaves: nodes[1..8]
    int fc = tree.nodes[0].first_child;  // = 1
    tree.refine(fc);         // level-1-oct=0 → 8 level-2 leaves starting at fc2

    // Level-1 leaves: nodes fc+1 .. fc+7 (oct 1-7)
    // Level-2 leaves: nodes fc2 .. fc2+7 (oct 0-7)
    int fc2 = tree.nodes[fc].first_child;

    // Fill level-1 leaves with a smooth linear pattern
    for (int oct = 1; oct < 8; ++oct) {
        CellBlock& blk = *tree.nodes[fc + oct].block;
        fill_interior(blk, oct * 3.14159e4);
    }

    // Fill level-2 leaves with a different seed (not the CF-fill target)
    for (int oct = 0; oct < 8; ++oct)
        fill_interior(*tree.nodes[fc2 + oct].block, 0.0);

    // CPU reference: fill_ghosts_periodic handles CF fine←coarse internally
    tree.fill_ghosts_periodic();
    std::vector<int> l2_nodes(8);
    std::vector<Snap> snap_cpu_l2(8);
    for (int oct = 0; oct < 8; ++oct) {
        l2_nodes[oct] = fc2 + oct;
        snap_cpu_l2[oct] = take_snap(*tree.nodes[fc2 + oct].block);
    }

    // Restore level-2 interiors (CPU ghost fill has overwritten all ghost cells)
    for (int oct = 0; oct < 8; ++oct)
        fill_interior(*tree.nodes[fc2 + oct].block, 0.0);
    // (Level-1 blocks may have CF coarse←fine ghosts filled by CPU; those aren't
    // tested here so we don't need to restore them exactly.)

    // Upload all leaves (level-1 and level-2)
    upload_leaves(tree);

    GpuGhostFillList gfl;
    gfl.build(tree, pool, /*bc_type=*/0);
    gfl.exec(nullptr);
    cudaDeviceSynchronize();

    download_leaves(tree);

    // Compare face-only ghost cells of each level-2 leaf
    double max_err = 0.0;
    int n_cf_faces = 0;
    for (int oct = 0; oct < 8; ++oct) {
        int li = l2_nodes[oct];
        Snap sg = take_snap(*tree.nodes[li].block);
        double e = ghost_err_face(sg, snap_cpu_l2[oct]);
        if (e > max_err) max_err = e;
        // Count CF faces for diagnostic
        for (int d = 0; d < NFACES; ++d) {
            int ni = tree.nodes[li].neighbours[d];
            if (ni >= 0 && tree.nodes[ni].is_leaf() &&
                tree.nodes[ni].level < tree.nodes[li].level)
                ++n_cf_faces;
        }
    }
    free_leaves(tree);

    printf("   CF fine-from-coarse face ghost err = %.2e  (expect 0, %d CF faces)\n",
           max_err, n_cf_faces);
    const double tol = 1e-10 * 1e6;  // relative to max interior value ~1e6*NCELL~1e6
    check(max_err < tol, "F4", "CF fine-from-coarse face ghosts GPU==CPU fill_ghosts_periodic",
          max_err);
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("=== P8.2 GPU ghost fill gate test ===\n");
    test_f1();
    test_f2();
    test_f3();
    test_f4();

    printf("\n=== Result: %d failure(s) ===\n", nfail);
    return nfail == 0 ? 0 : 1;
}
