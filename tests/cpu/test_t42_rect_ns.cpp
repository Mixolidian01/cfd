#include "solver/ns_solver.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

static Prim uniform_prim(double rho = 1.0, double p = 1.0) {
    Prim q{};
    q.rho     = rho;
    q.u       = 0.0; q.v = 0.0; q.w = 0.0;
    q.p       = p;
    q.gamma_m = GAMMA;
    q.p_inf_m = 0.0;
    q.T       = q.p / (q.rho * R_GAS);
    q.c       = std::sqrt(GAMMA * q.p / q.rho);
    return q;
}

int main() {
    // T1: leaf cell sizes match requested domain
    {
        NSSolver solver;
        solver.cfg.bc.variant = PeriodicBC{};
        solver.init(2.0, 1.0, 0.5, [](double, double, double) -> Prim {
            return uniform_prim();
        });
        for (int ni : solver.tree.leaf_indices()) {
            const auto& blk = *solver.tree.nodes[ni].block;
            [[maybe_unused]] double eh  = std::fabs(blk.h  - 2.0/NB);
            [[maybe_unused]] double ehy = std::fabs(blk.hy - 1.0/NB);
            [[maybe_unused]] double ehz = std::fabs(blk.hz - 0.5/NB);
            assert(eh  < 1e-14);
            assert(ehy < 1e-14);
            assert(ehz < 1e-14);
        }
    }

    // T2: IC coordinates are scaled to the correct domain extents
    {
        double x_max = -1e99, y_max = -1e99, z_max = -1e99;
        NSSolver solver;
        solver.cfg.bc.variant = PeriodicBC{};
        solver.init(2.0, 1.0, 0.5, [&](double x, double y, double z) -> Prim {
            x_max = std::max(x_max, x);
            y_max = std::max(y_max, y);
            z_max = std::max(z_max, z);
            return uniform_prim();
        });
        // Cell centres reach up to L - h/2 (last interior cell centre)
        assert(x_max > 1.9 && x_max < 2.0);
        assert(y_max > 0.9 && y_max < 1.0);
        assert(z_max > 0.45 && z_max < 0.5);
    }

    // T3: total mass matches rho * Lx * Ly * Lz
    {
        NSSolver solver;
        solver.cfg.bc.variant = PeriodicBC{};
        solver.init(2.0, 1.0, 0.5, [](double, double, double) -> Prim {
            return uniform_prim(1.0, 1.0);
        });
        double mass = solver.compute_diag().mass;
        // volume = 2.0 * 1.0 * 0.5 = 1.0; rho = 1 → mass = 1.0
        [[maybe_unused]] double err = std::fabs(mass - 1.0);
        assert(err < 1e-10);
    }

    // T4: mass conserved to 1e-10 over 10 steps
    {
        NSSolver solver;
        solver.cfg.bc.variant = PeriodicBC{};
        solver.cfg.time.cfl   = 0.4;
        solver.init(2.0, 1.0, 0.5, [](double x, double, double) -> Prim {
            return uniform_prim(1.0 + 0.01 * std::sin(2 * M_PI * x / 2.0), 1.0);
        });
        double m0 = solver.compute_diag().mass;
        for (int i = 0; i < 10; ++i) solver.advance();
        double m1 = solver.compute_diag().mass;
        [[maybe_unused]] double rel_err = std::fabs((m1 - m0) / m0);
        assert(rel_err < 1e-10);
    }

    std::puts("PASS t42_rect_ns");
    return 0;
}
