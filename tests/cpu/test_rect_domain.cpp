#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

static bool nearly_eq(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) < tol;
}

int main() {
    // T1: CellBlock 6-arg constructor propagates all sizes
    {
        CellBlock blk(0.0, 0.0, 0.0, 0.1, 0.2, 0.3);
        assert(nearly_eq(blk.h,  0.1));
        assert(nearly_eq(blk.hy, 0.2));
        assert(nearly_eq(blk.hz, 0.3));
    }

    // T2: BlockTree::init(Lx,Ly,Lz) sets root cell sizes
    {
        BlockTree tree;
        tree.init(2.0, 1.0, 0.5);
        [[maybe_unused]] const CellBlock& rb = *tree.nodes[0].block;
        assert(tree.nodes[0].has_block());
        assert(nearly_eq(rb.h,  2.0 / NB));
        assert(nearly_eq(rb.hy, 1.0 / NB));
        assert(nearly_eq(rb.hz, 0.5 / NB));
    }

    // T3: 1-arg init is still cubic
    {
        BlockTree tree;
        tree.init(1.0);
        [[maybe_unused]] const CellBlock& rb = *tree.nodes[0].block;
        assert(nearly_eq(rb.h,  1.0 / NB));
        assert(nearly_eq(rb.hy, 1.0 / NB));
        assert(nearly_eq(rb.hz, 1.0 / NB));
    }

    // T4: child cell after one refinement is half the parent size
    {
        BlockTree tree;
        tree.init(2.0, 1.0, 0.5);
        tree.refine(0);
        int fc = tree.nodes[0].first_child;
        assert(fc >= 0);
        for (int oct = 0; oct < 8; ++oct) {
            int ci = fc + oct;
            const auto& child = tree.nodes[ci];
            if (!child.has_block()) continue;
            assert(nearly_eq(child.block->h,  (2.0 / NB) * 0.5));
            assert(nearly_eq(child.block->hy, (1.0 / NB) * 0.5));
            assert(nearly_eq(child.block->hz, (0.5 / NB) * 0.5));
        }
    }

    std::puts("PASS test_rect_domain");
    return 0;
}
