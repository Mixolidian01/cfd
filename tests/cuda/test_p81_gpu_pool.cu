// P8.1 gate test — GPU memory pool for AMR leaf CellBlocks
//
// Protocol:
//   G1a: GpuPool::alloc registers a non-null device pointer (pool.has_device)
//   G1b: upload → corrupt CPU → download → CPU matches original (exact round-trip)
//   G1c: GpuPool::free removes device mapping; slot re-used on next alloc
//   G2a: BlockTree::refine triggers on_block_alloc_ × 8 (children)
//   G2b: BlockTree::refine triggers on_block_free_  × 1 (parent)
//   G2c: BlockTree::coarsen triggers on_block_alloc_ × 1 (parent)
//   G2d: BlockTree::coarsen triggers on_block_free_  × 8 (children)

#include "gpu_pool.hpp"
#include "mesh/block_tree.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>

static int nfail = 0;

static void check(bool ok, const char* tag, const char* msg,
                  double val = -1.0, double ref = -1.0) {
    if (ok) {
        printf("  PASS  %s  %s\n", tag, msg);
    } else {
        if (val >= 0.0)
            printf("  FAIL  %s  %s  (got %.4e  ref %.4e)\n", tag, msg, val, ref);
        else
            printf("  FAIL  %s  %s\n", tag, msg);
        ++nfail;
    }
}

// =============================================================================
// Fill a CellBlock with a deterministic pattern: Q[v][flat] = v*1000.0 + flat
// =============================================================================
static void fill_pattern(CellBlock& blk) {
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            blk.Q[v][idx] = static_cast<double>(v * 1000 + idx);
}

static double max_err(const CellBlock& a, const CellBlock& b) {
    double e = 0.0;
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            e = std::fmax(e, std::fabs(a.Q[v][idx] - b.Q[v][idx]));
    return e;
}

// =============================================================================
// G1: alloc / upload / download / free
// =============================================================================
static void test_g1() {
    printf("\n-- G1  GpuPool alloc/upload/download/free --\n");

    CellBlock blk(0.0, 0.0, 0.0, 1.0/NB);
    fill_pattern(blk);

    GpuPool pool;

    // G1a: alloc → pool reports device allocated
    pool.alloc(&blk);
    check(pool.has_device(&blk), "G1a", "alloc: pool.has_device(&blk) is true");

    // G1b: upload → corrupt CPU → download → exact match
    pool.upload(&blk);
    CellBlock orig = blk;    // save original
    // Corrupt all CPU data
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            blk.Q[v][idx] = -999.0;
    pool.download(&blk);
    double err = max_err(blk, orig);
    printf("   upload→download max error = %.2e  (expect 0)\n", err);
    check(err == 0.0, "G1b", "upload/download round-trip exact", err, 0.0);

    // G1c: free removes device mapping; slot is re-used on next alloc
    void* first_ptr = pool.d_Q(&blk);
    pool.free(&blk);
    check(!pool.has_device(&blk), "G1c", "free: pool.has_device(&blk) is false");
    check(pool.n_active() == 0, "G1c2", "n_active == 0 after free");

    CellBlock blk2(0.0, 0.0, 0.0, 1.0/NB);
    pool.alloc(&blk2);
    check(pool.d_Q(&blk2) == first_ptr, "G1c3", "freed slot is reused");
    pool.free(&blk2);
}

// =============================================================================
// G2: BlockTree lifecycle callbacks — counting only (no CUDA calls per event)
// =============================================================================
static void test_g2() {
    printf("\n-- G2  BlockTree refine/coarsen callback counts --\n");

    int n_alloc = 0, n_free = 0;

    BlockTree tree;
    tree.set_gpu_callbacks(
        [&](CellBlock*) { ++n_alloc; },
        [&](CellBlock*) { ++n_free;  }
    );
    tree.init(1.0);  // root block; init() does NOT call callbacks (data not ready yet)

    // Counts start at 0 (init doesn't fire callbacks)
    check(n_alloc == 0, "G2a0", "init does not trigger on_block_alloc_");
    check(n_free  == 0, "G2b0", "init does not trigger on_block_free_");

    n_alloc = n_free = 0;
    tree.refine(0);  // root → 8 children
    check(n_alloc == 8, "G2a", "refine triggers on_block_alloc_ × 8",
          static_cast<double>(n_alloc), 8.0);
    check(n_free  == 1, "G2b", "refine triggers on_block_free_ × 1 (parent)",
          static_cast<double>(n_free), 1.0);

    n_alloc = n_free = 0;
    tree.coarsen(0);  // 8 children → parent
    check(n_alloc == 1, "G2c", "coarsen triggers on_block_alloc_ × 1 (parent)",
          static_cast<double>(n_alloc), 1.0);
    check(n_free  == 8, "G2d", "coarsen triggers on_block_free_ × 8 (children)",
          static_cast<double>(n_free), 8.0);

    // After round-trip, exactly 1 leaf (root) exists
    check(tree.n_leaves() == 1, "G2e", "1 leaf after refine+coarsen round-trip");
}

// =============================================================================
// G3: GpuPool lifecycle with BlockTree callbacks (full integration)
// =============================================================================
static void test_g3() {
    printf("\n-- G3  GpuPool + BlockTree lifecycle integration --\n");

    GpuPool pool;

    // Manually alloc root block (simulates NSSolver::init())
    BlockTree tree;
    tree.init(1.0);
    auto* root_blk = tree.nodes[0].block.get();
    fill_pattern(*root_blk);
    pool.alloc(root_blk);
    pool.upload(root_blk);
    check(pool.n_active() == 1, "G3a", "1 active slot after root alloc");

    // Wire callbacks: alloc+upload / free
    tree.set_gpu_callbacks(
        [&](CellBlock* b) { pool.alloc(b); pool.upload(b); },
        [&](CellBlock* b) { pool.free(b); }
    );

    // Refine: root freed (1 active → 0), 8 children allocated (→ 8)
    tree.refine(0);
    check(pool.n_active() == 8, "G3b", "8 active slots after refine",
          static_cast<double>(pool.n_active()), 8.0);

    // All child blocks should have a live device buffer in the pool
    bool all_set = true;
    for (int li : tree.leaf_indices())
        if (!pool.has_device(tree.nodes[li].block.get())) all_set = false;
    check(all_set, "G3c", "all child d_Q pointers are non-null");

    // Coarsen: 8 children freed (→ 0 active), 1 parent allocated (→ 1)
    tree.coarsen(0);
    check(pool.n_active() == 1, "G3d", "1 active slot after coarsen",
          static_cast<double>(pool.n_active()), 1.0);
    check(pool.has_device(tree.nodes[0].block.get()), "G3e",
          "root d_Q non-null after coarsen");

    // Download parent and verify restriction produced finite values
    CellBlock result;
    result = *tree.nodes[0].block;   // CPU data copy
    pool.download(tree.nodes[0].block.get());
    // Download overwrites CPU with what's on GPU (the uploaded child data)
    // Just check no NaN
    bool no_nan = true;
    for (int v = 0; v < NVAR; ++v)
        for (int idx = 0; idx < NCELL; ++idx)
            if (!std::isfinite(tree.nodes[0].block->Q[v][idx])) no_nan = false;
    check(no_nan, "G3f", "downloaded parent data is finite (no NaN)");

    // Cleanup
    pool.free(tree.nodes[0].block.get());
    check(pool.n_active() == 0, "G3g", "0 active after cleanup");
}

// =============================================================================
int main() {
    printf("=== P8.1 gate: GPU memory pool for AMR leaf CellBlocks ===\n");

    int n_gpu = 0;
    cudaGetDeviceCount(&n_gpu);
    if (n_gpu == 0) {
        printf("SKIP  no CUDA device found — GPU tests skipped\n");
        printf("Results: 0 passed, 0 failed  (skipped)\n");
        return 0;
    }
    printf("GPU device count: %d\n", n_gpu);

    test_g1();
    test_g2();
    test_g3();

    const int ntotal = 14;
    const int npass  = ntotal - nfail;
    printf("\nResults: %d passed, %d failed\n", npass, nfail);
    if (nfail == 0)
        printf("==> PASS  P8.1 gate cleared — GPU memory pool operational\n");
    return nfail > 0 ? 1 : 0;
}
