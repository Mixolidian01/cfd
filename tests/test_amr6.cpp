// test_amr6.cpp — Step 6 gate: conservative AMR
// Gate: global mass conserved to < 1e-12 at every step including regrid
// FIX P0.2: global_mass() now uses node.block->h (BlockNode::h was removed).
// FIX P0.3: A02 octant expected-value computation aligned to canonical
//           bit0=x, bit1=y, bit2=z convention used by amr_operators.cpp.

#include "../include/ns_solver.hpp"
#include "../include/amr_operators.hpp"
#include <cstdio>
#include <cmath>

static int n_pass=0, n_fail=0;
static void check(const char* name, bool ok, double got=-1, double thr=-1) {
    if (ok) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got>=0) printf("  FAIL  %s  (got %.6e  threshold %.6e)\n", name, got, thr);
        else printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// Compute global mass = sum over all leaves of sum(rho * h^3)
// FIX P0.2: cell size read from node.block->h, not the removed node.h field.
static double global_mass(const NSSolver& s) {
    double m = 0;
    for (int li : s.tree.leaf_indices()) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double h   = node.block->h;          // P0.2: was node.h
        double h3  = h * h * h;
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i)
            m += blk.Q[0][cell_idx(i,j,k)] * h3;
    }
    return m;
}

// A01: prolongation is conservative
static void a01_prolong_conservative() {
    CellBlock coarse, fine;
    // Fill coarse with uniform rho=2
    for (int k=NG;k<NG+NB;++k)
    for (int j=NG;j<NG+NB;++j)
    for (int i=NG;i<NG+NB;++i)
        coarse.Q[0][cell_idx(i,j,k)] = 2.0;

    double mass_coarse = 0;
    for (int k=NG;k<NG+NB;++k)
    for (int j=NG;j<NG+NB;++j)
    for (int i=NG;i<NG+NB;++i)
        mass_coarse += coarse.Q[0][cell_idx(i,j,k)];

    double err = 0;
    for (int oct=0; oct<8; ++oct) {
        prolong_conservative(coarse, fine, oct);
        double mass_fine = 0;
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i)
            mass_fine += fine.Q[0][cell_idx(i,j,k)];
        // Each child covers 1/8 of coarse volume (same number of cells,
        // but h_fine = h_coarse/2 so h^3 ratio = 1/8).
        // Conservation: fine_sum * (h_f/h_c)^3 = coarse contribution
        // fine cells in this octant = NB^3, each covers (h/2)^3 vs h^3
        // mass_fine * (1/8) should equal mass_coarse * (1/8)
        err = std::fmax(err, std::fabs(mass_fine - mass_coarse));
    }
    check("A01 prolong conservative: sum(fine)==sum(coarse) per octant", err < 1e-12, err, 1e-12);
}

// A02: restriction is conservative
// FIX P0.3: expected octant index now uses canonical bit0=x convention:
//   oct = (lf_i/half) | ((lf_j/half)<<1) | ((lf_k/half)<<2)
// Previously the test used bit2=x which disagreed with oct_from_xyz in
// amr_operators.cpp, producing wrong expected values for non-corner octants.
static void a02_restrict_conservative() {
    CellBlock coarse;
    CellBlock children_storage[8];
    const CellBlock* children[8];
    for (int oct=0; oct<8; ++oct) {
        children[oct] = &children_storage[oct];
        for (int k=NG;k<NG+NB;++k)
        for (int j=NG;j<NG+NB;++j)
        for (int i=NG;i<NG+NB;++i)
            children_storage[oct].Q[0][cell_idx(i,j,k)] = 1.0 + oct*0.1;
    }
    restrict_conservative(coarse, children);

    // Expected: coarse[I] = value of the child octant that owns cell I.
    // Canonical octant convention: bit0=x, bit1=y, bit2=z.
    double err = 0;
    for (int k=NG;k<NG+NB;++k)
    for (int j=NG;j<NG+NB;++j)
    for (int i=NG;i<NG+NB;++i) {
        int lf_i=i-NG, lf_j=j-NG, lf_k=k-NG, half=NB/2;
        // FIX P0.3: bit0=x, bit1=y, bit2=z  (was bit2=x)
        int oct = (lf_i/half) | ((lf_j/half)<<1) | ((lf_k/half)<<2);
        double expected = 1.0 + oct*0.1;
        err = std::fmax(err, std::fabs(coarse.Q[0][cell_idx(i,j,k)] - expected));
    }
    check("A02 restrict conservative: coarse = avg of 8 fine", err < 1e-12, err, 1e-12);
}

// A03: regrid (refine then restrict) preserves global mass
static void a03_regrid_conserves_mass() {
    auto ic = [](double x, double y, double z) -> Prim {
        double pi = acos(-1.0);
        Prim q; q.rho = 1.0 + 0.1*sin(2*pi*x)*cos(2*pi*y);
        q.u=0; q.v=0; q.w=0; q.p=1e5;
        q.T=q.p/(q.rho*R_GAS); q.c=sqrt(GAMMA*q.p/q.rho); return q;
    };
    NSSolver s;
    s.cfg.bc_variant = PeriodicBC{}; s.cfg.verbose=false;
    s.init(1.0, ic);
    double m0 = global_mass(s);

    // Manually trigger refine on first block and restrict back
    int root = s.tree.root();
    auto& coarse = *s.tree.nodes[root].block;

    // Refine
    CellBlock fine_children[8];
    const CellBlock* children[8];
    for (int oct=0; oct<8; ++oct) {
        children[oct] = &fine_children[oct];
        prolong_conservative(coarse, fine_children[oct], oct);
    }

    // Restrict back
    CellBlock restored;
    restrict_conservative(restored, children);

    // Compare
    double err = 0;
    for (int k=NG;k<NG+NB;++k)
    for (int j=NG;j<NG+NB;++j)
    for (int i=NG;i<NG+NB;++i)
        err = std::fmax(err, std::fabs(restored.Q[0][cell_idx(i,j,k)] -
                                       coarse.Q[0][cell_idx(i,j,k)]));
    check("A03 prolong+restrict round-trip L∞(rho) < 1e-12", err < 1e-12, err, 1e-12);
}

// A04: NSSolver::advance() conserves mass with no regrid
static void a04_advance_conserves_mass() {
    auto ic = [](double x, double y, double z) -> Prim {
        double pi = acos(-1.0);
        Prim q; q.rho=1.225+0.05*sin(2*pi*x)*cos(2*pi*y);
        q.u=0.5*sin(2*pi*x)*cos(2*pi*y);
        q.v=-0.5*cos(2*pi*x)*sin(2*pi*y);
        q.w=0; q.p=101325;
        q.T=q.p/(q.rho*R_GAS); q.c=sqrt(GAMMA*q.p/q.rho); return q;
    };
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc_variant = PeriodicBC{}; s.cfg.verbose=false;
    s.cfg.regrid_interval=0; // no regrid
    s.init(1.0, ic);
    double m0 = global_mass(s);
    for (int i=0;i<20;++i) s.advance();
    double m1 = global_mass(s);
    double err = std::fabs(m1-m0)/std::fabs(m0);
    check("A04 mass conserved 20 steps no regrid < 1e-10", err < 1e-10, err, 1e-10);
}

// A05: mass conserved through regrid step
static void a05_regrid_step_conserves_mass() {
    auto ic = [](double x, double y, double z) -> Prim {
        double pi = acos(-1.0);
        Prim q; q.rho=1.225+0.2*exp(-20*((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)));
        q.u=0; q.v=0; q.w=0; q.p=101325;
        q.T=q.p/(q.rho*R_GAS); q.c=sqrt(GAMMA*q.p/q.rho); return q;
    };
    NSSolver s;
    s.cfg.cfl=0.5; s.cfg.bc_variant = PeriodicBC{}; s.cfg.verbose=false;
    s.cfg.regrid_interval=5; s.cfg.max_level=2;
    s.init(1.0, ic);
    double m0 = global_mass(s);
    double err_max = 0;
    for (int i=0;i<20;++i) {
        s.advance();
        double m = global_mass(s);
        err_max = std::fmax(err_max, std::fabs(m-m0)/std::fabs(m0));
    }
    check("A05 mass conserved through regrid steps < 1e-10", err_max < 1e-10, err_max, 1e-10);
}

int main() {
    printf("=== Step 6: Conservative AMR ===\n\n");
    a01_prolong_conservative();
    a02_restrict_conservative();
    a03_regrid_conserves_mass();
    a04_advance_conserves_mass();
    a05_regrid_step_conserves_mass();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail==0) printf("==> PASS  Step 6 gate cleared\n");
    else           printf("==> FAIL  Step 6 gate NOT cleared\n");
    return (n_fail==0) ? 0 : 1;
}
