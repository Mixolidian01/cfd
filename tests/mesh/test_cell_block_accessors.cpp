// R10-T1: flat-index typed field accessors on CellBlock
#include "mesh/cell_block.hpp"
#include <cstdio>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool cond) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else       { printf("  FAIL  %s\n", name); ++n_fail; }
}

static void t01_flat_read() {
    CellBlock b;
    for (int f = 0; f < NCELL; ++f) {
        b.Q[0][f] = 1.0+f; b.Q[1][f] = 2.0+f; b.Q[2][f] = 3.0+f;
        b.Q[3][f] = 4.0+f; b.Q[4][f] = 5.0+f;
    }
    bool ok = true;
    for (int f = 0; f < NCELL; ++f)
        ok &= (b.rho(f)==1.0+f) && (b.rhou(f)==2.0+f) && (b.rhov(f)==3.0+f)
           && (b.rhow(f)==4.0+f) && (b.E(f)==5.0+f);
    check("T01 flat read", ok);
}

static void t02_flat_write() {
    CellBlock b;
    for (int f = 0; f < NCELL; ++f) {
        b.rho(f) = 10.0+f; b.rhou(f) = 20.0+f; b.rhov(f) = 30.0+f;
        b.rhow(f) = 40.0+f; b.E(f) = 50.0+f;
    }
    bool ok = true;
    for (int f = 0; f < NCELL; ++f)
        ok &= (b.Q[0][f]==10.0+f) && (b.Q[1][f]==20.0+f) && (b.Q[2][f]==30.0+f)
           && (b.Q[3][f]==40.0+f) && (b.Q[4][f]==50.0+f);
    check("T02 flat write", ok);
}

static void t03_flat_ijk_consistent() {
    CellBlock b;
    for (int f = 0; f < NCELL; ++f) b.rho(f) = (double)f;
    bool ok = true;
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i)
        ok &= (b.rho(i,j,k) == b.rho(cell_idx(i,j,k)));
    check("T03 flat/ijk consistency", ok);
}

int main() {
    t01_flat_read(); t02_flat_write(); t03_flat_ijk_consistent();
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
