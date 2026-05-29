#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "schemes/operators.hpp"
#include "solver/ns_solver.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

[[maybe_unused]] static bool nearly_eq(double a, double b, double tol = 1e-12) {
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

    // T5: convective RHS of uniform state is zero on a 2×1×0.5 rectangular domain
    {
        // Cell sizes for a single-block (NB=8) covering a 2×1×0.5 domain
        const double hx = 2.0 / NB;
        const double hy = 1.0 / NB;
        const double hz = 0.5 / NB;
        CellBlock blk(0.0, 0.0, 0.0, hx, hy, hz);
        // Uniform state: rho=1, u=v=w=0, p=1  (ideal gas, E = p/(γ-1))
        const double rho0 = 1.0;
        const double p0   = 1.0;
        const double E0   = p0 / (GAMMA - 1.0);
        // Fill ALL cells including ghost layers with the uniform state
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            int idx = cell_idx(i, j, k);
            blk.Q[0][idx] = rho0;
            blk.Q[1][idx] = 0.0;
            blk.Q[2][idx] = 0.0;
            blk.Q[3][idx] = 0.0;
            blk.Q[4][idx] = E0;
        }
        CellBlock rhs(0.0, 0.0, 0.0, hx, hy, hz);
        compute_rhs(blk, rhs);
        // For a uniform state all fluxes cancel; RHS must be exactly zero
        double max_rhs = 0.0;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            int idx = cell_idx(i, j, k);
            for (int v = 0; v < NVAR; ++v)
                max_rhs = std::max(max_rhs, std::fabs(rhs.Q[v][idx]));
        }
        assert(max_rhs < 1e-12);
    }

    // T6: viscous RHS of uniform-velocity state is zero on 2×1×0.5 domain
    {
        const double hx = 2.0 / NB;
        const double hy = 1.0 / NB;
        const double hz = 0.5 / NB;
        CellBlock blk(0.0, 0.0, 0.0, hx, hy, hz);
        // Constant-velocity state: rho=1.2, u=0.3, v=0, w=0, p=1
        const double rho = 1.2, u = 0.3, p = 1.0;
        const double E = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int i = 0; i < NB2; ++i) {
            int idx = cell_idx(i, j, k);
            blk.Q[0][idx] = rho;
            blk.Q[1][idx] = rho * u;
            blk.Q[2][idx] = 0.0;
            blk.Q[3][idx] = 0.0;
            blk.Q[4][idx] = E;
        }
        CellBlock rhs(0.0, 0.0, 0.0, hx, hy, hz);
        // compute_rhs includes viscous; for uniform flow, viscous terms vanish
        compute_rhs(blk, rhs);
        double max_visc = 0.0;
        for (int k = ilo(); k <= ihi(); ++k)
        for (int j = ilo(); j <= ihi(); ++j)
        for (int i = ilo(); i <= ihi(); ++i) {
            int idx = cell_idx(i, j, k);
            for (int v = 0; v < NVAR; ++v)
                max_visc = std::max(max_visc, std::fabs(rhs.Q[v][idx]));
        }
        assert(max_visc < 1e-10);
    }

    // T7: AMR CF correction mass conservation (cubic domain — verifies no regress)
    //
    // Uses a cubic domain so that the test is self-contained without Task 7's
    // 3-arg NSSolver::init() overload.  The purpose is to verify that the
    // axis-correct ih/h_axis changes in undo_cf_one_face and apply_flux_correction
    // do not break the Berger-Colella flux-register invariant on the existing
    // cubic-domain code path.
    {
        NSSolver solver;
        SolverConfig cfg;
        cfg.amr.max_level = 1;
        cfg.amr.regrid_interval = 5;
        cfg.time.cfl = 0.5;
        cfg.bc.variant = PeriodicBC{};
        cfg.io.verbose = false;
        solver.cfg = cfg;
        solver.init(1.0, [](double x, double /*y*/, double /*z*/) -> Prim {
            const double pi = std::acos(-1.0);
            Prim q;
            q.rho = 1.0 + 0.1 * std::sin(2.0 * pi * x);
            q.u = 0; q.v = 0; q.w = 0; q.p = 1.0;
            q.T = q.p / (q.rho * R_GAS);
            q.c = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        });
        solver.regrid();
        const double m0 = solver.compute_diag().mass;
        for (int step = 0; step < 10; ++step)
            solver.advance();
        const double m1 = solver.compute_diag().mass;
        [[maybe_unused]] const double rel_err = std::fabs((m1 - m0) / m0);
        assert(rel_err < 1e-10);
    }

    std::puts("PASS test_rect_domain");
    return 0;
}
