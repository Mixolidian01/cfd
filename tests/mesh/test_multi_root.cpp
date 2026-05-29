// test_multi_root.cpp — forest-of-octrees BlockTree gate test
// Tests: geometry, cross-root neighbour links, C/F at root boundary,
//        periodic root-grid wrapping, mass conservation over 10 steps.

#include "mesh/block_tree.hpp"
#include "solver/ns_solver.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { if (cond) { ++g_pass; } else { ++g_fail; \
    std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while(0)

// T1: 2×1×1 root grid — cell sizes and origins
static void test_geometry() {
    BlockTree tree;
    tree.init(2.0, 1.0, 1.0, 2, 1, 1);
    CHECK(tree.n_roots() == 2);
    const auto& r0 = tree.nodes[0];
    const auto& r1 = tree.nodes[1];
    CHECK(r0.level == 0 && r1.level == 0);
    CHECK(std::fabs(r0.ox - 0.0) < 1e-12);
    CHECK(std::fabs(r1.ox - 1.0) < 1e-12);
    CHECK(std::fabs(r0.block->h   - 1.0 / 8) < 1e-12);
    CHECK(std::fabs(r1.block->h   - 1.0 / 8) < 1e-12);
    CHECK(std::fabs(r0.block->hy  - 1.0 / 8) < 1e-12);
    CHECK(std::fabs(r0.block->hz  - 1.0 / 8) < 1e-12);
    // Single-root init still works
    BlockTree t2;
    t2.init(1.0);
    CHECK(t2.n_roots() == 1);
    CHECK(t2.nx_roots() == 1 && t2.ny_roots() == 1 && t2.nz_roots() == 1);
}

// T2: 2×1×1 — cross-root neighbor links at X interface (no periodicity)
static void test_cross_root_neighbors() {
    BlockTree tree;
    tree.init(2.0, 1.0, 1.0, 2, 1, 1);
    // Root 0 XPLUS neighbor is Root 1
    CHECK(tree.nodes[0].neighbours[XPLUS]  == 1);
    // Root 1 XMINUS neighbor is Root 0
    CHECK(tree.nodes[1].neighbours[XMINUS] == 0);
    // Domain boundary directions must be -1
    CHECK(tree.nodes[0].neighbours[XMINUS] == -1);
    CHECK(tree.nodes[1].neighbours[XPLUS]  == -1);
    CHECK(tree.nodes[0].neighbours[YMINUS] == -1);
    CHECK(tree.nodes[0].neighbours[YPLUS]  == -1);
}

// T3: 2×1×1 periodic — XMINUS of root 0 wraps to root 1 and vice versa
static void test_cross_root_periodic() {
    BlockTree tree;
    tree.init(2.0, 1.0, 1.0, 2, 1, 1);
    tree.set_periodic(true);
    tree.rebuild_neighbours();
    CHECK(tree.nodes[0].neighbours[XMINUS] == 1);
    CHECK(tree.nodes[1].neighbours[XPLUS]  == 0);
}

// T4: single-root periodic still works after the algorithm change
static void test_single_root_periodic() {
    BlockTree tree;
    tree.init(1.0);
    tree.set_periodic(true);
    tree.refine(0);
    tree.rebuild_neighbours();
    // All level-1 leaves should have 6 neighbours (periodic wrap or siblings)
    const auto& leaves = tree.leaf_indices();
    CHECK((int)leaves.size() == 8);
    for (int li : leaves) {
        for (int d = 0; d < NFACES; ++d)
            CHECK(tree.nodes[li].neighbours[d] >= 0);
    }
}

// T5: refine root 0 — level-1 children at XPLUS boundary link to unrefined root 1 (C/F)
static void test_cf_cross_root() {
    BlockTree tree;
    tree.init(2.0, 1.0, 1.0, 2, 1, 1);
    tree.refine(0);   // root 0 → 8 level-1 children
    const int fc = tree.nodes[0].first_child;
    bool found_cf = false;
    for (int oct = 0; oct < 8; ++oct) {
        if (oct_ix(oct) == 1) {  // child at X-high boundary of root 0
            int ni = tree.nodes[fc + oct].neighbours[XPLUS];
            if (ni == 1) found_cf = true;  // links to root 1 (coarser, level 0)
        }
    }
    CHECK(found_cf);
    // Root 1 must also link back to one of those children
    int ni_r1 = tree.nodes[1].neighbours[XMINUS];
    CHECK(ni_r1 >= fc && ni_r1 < fc + 8);
}

// T6: 2×2×1 root grid — check all 4 internal cross-root links
static void test_2x2_grid() {
    BlockTree tree;
    tree.init(2.0, 2.0, 1.0, 2, 2, 1);
    CHECK(tree.n_roots() == 4);
    // Layout: root(ix,iy,0) = iy*2 + ix
    //   0=(0,0)  1=(1,0)
    //   2=(0,1)  3=(1,1)
    CHECK(tree.nodes[0].neighbours[XPLUS]  == 1);  // (0,0)→(1,0)
    CHECK(tree.nodes[1].neighbours[XMINUS] == 0);  // (1,0)→(0,0)
    CHECK(tree.nodes[2].neighbours[XPLUS]  == 3);  // (0,1)→(1,1)
    CHECK(tree.nodes[0].neighbours[YPLUS]  == 2);  // (0,0)→(0,1)
    CHECK(tree.nodes[2].neighbours[YMINUS] == 0);  // (0,1)→(0,0)
}

// T7: mass conservation over 10 steps in a 2×1×1 forest
static void test_mass_conservation() {
    auto ic = [](double x, double /*y*/, double /*z*/) -> Prim {
        Prim q{};
        q.rho = 1.0 + 0.1 * std::sin(2.0 * M_PI * x / 2.0);
        q.u   = 0.1;
        q.v   = 0.0;
        q.w   = 0.0;
        q.p   = 1.0;
        return q;
    };
    NSSolver solver;
    solver.init(2.0, 1.0, 1.0, 2, 1, 1, ic);
    const double m0 = solver.compute_diag().mass;
    for (int step = 0; step < 10; ++step) solver.advance();
    const double m1 = solver.compute_diag().mass;
    const double err = std::fabs(m1 - m0) / (m0 + 1e-300);
    if (err > 1e-10)
        std::fprintf(stderr, "  mass error = %.3e (m0=%.10f m1=%.10f)\n", err, m0, m1);
    CHECK(err < 1e-10);
}

int main() {
    test_geometry();
    test_cross_root_neighbors();
    test_cross_root_periodic();
    test_single_root_periodic();
    test_cf_cross_root();
    test_2x2_grid();
    test_mass_conservation();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
