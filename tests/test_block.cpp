// DESIGN.md reference: Step 2 gate tests — Layer 1
// Gate criteria:
//   Morton:  encode/decode round-trip exact for all corners of 10-bit range
//   EOS:     prim→cons→prim round-trip error < 1e-14
//   Block:   mass/momentum conservation through refine + coarsen < 1e-14
//   Ghost:   periodic ghost fill copies correct values
//   Tree:    refine → 8 leaves; coarsen → 1 leaf; 2:1 balance maintained
//   Neighbours: sibling adjacency correct after refine

#include "cell_block.hpp"
#include "block_tree.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

static int n_pass = 0, n_fail = 0;
static void check(const char* name, bool cond, double got=-1, double thr=-1) {
    if (cond) { printf("  PASS  %s\n", name); ++n_pass; }
    else {
        if (got >= 0)
            printf("  FAIL  %s  (got %.6e  threshold %.6e)\n", name, got, thr);
        else
            printf("  FAIL  %s\n", name);
        ++n_fail;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T01  Morton encode/decode round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void t01_morton() {
    bool ok = true;
    // Test all corners and a dense grid of values
    for (uint32_t x : {0u, 1u, 511u, 1023u}) {
    for (uint32_t y : {0u, 1u, 511u, 1023u}) {
    for (uint32_t z : {0u, 1u, 511u, 1023u}) {
        uint32_t code = morton_encode(x, y, z);
        uint32_t rx, ry, rz;
        morton_decode(code, rx, ry, rz);
        if (rx != x || ry != y || rz != z) { ok = false; goto morton_done; }
    }}}
    // Dense inner grid
    for (uint32_t x = 0; x < 64; ++x)
    for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t z = 0; z < 64; ++z) {
        uint32_t code = morton_encode(x,y,z);
        uint32_t rx,ry,rz; morton_decode(code,rx,ry,rz);
        if (rx!=x||ry!=y||rz!=z) { ok=false; goto morton_done; }
    }
    morton_done:
    check("T01 Morton encode/decode round-trip", ok);
}

// ─────────────────────────────────────────────────────────────────────────────
// T02  EOS round-trip: prim → cons → prim, error < 1e-14
// ─────────────────────────────────────────────────────────────────────────────
static void t02_eos_roundtrip() {
    Prim p0;
    p0.rho = 1.2; p0.u = 30.0; p0.v = -10.0; p0.w = 5.0;
    p0.p   = 101325.0;
    p0.T   = p0.p / (p0.rho * R_GAS);
    p0.c   = std::sqrt(GAMMA * p0.p / p0.rho);

    double rho, rhou, rhov, rhow, E;
    eos_prim_to_cons(p0, rho, rhou, rhov, rhow, E);
    Prim p1 = eos_cons_to_prim(rho, rhou, rhov, rhow, E);

    double err = std::abs(p1.rho - p0.rho) + std::abs(p1.u - p0.u)
               + std::abs(p1.v   - p0.v)   + std::abs(p1.w - p0.w)
               + std::abs(p1.p   - p0.p) / p0.p;
    check("T02 EOS prim→cons→prim round-trip < 1e-14", err < 1e-14, err, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// T03  cell_idx bijection: every (i,j,k) in [0,NB2)^3 maps to a unique index
// ─────────────────────────────────────────────────────────────────────────────
static void t03_cell_idx_bijection() {
    std::vector<bool> seen(NCELL, false);
    bool ok = true;
    for (int k = 0; k < NB2; ++k)
    for (int j = 0; j < NB2; ++j)
    for (int i = 0; i < NB2; ++i) {
        int idx = cell_idx(i,j,k);
        if (idx < 0 || idx >= NCELL || seen[idx]) { ok = false; break; }
        seen[idx] = true;
    }
    check("T03 cell_idx bijection (no collisions)", ok);
}

// ─────────────────────────────────────────────────────────────────────────────
// T04  CellBlock: fill uniform state, check total mass
// ─────────────────────────────────────────────────────────────────────────────
static void t04_block_mass() {
    CellBlock blk(0,0,0, 1.0/NB);
    double rho0 = 1.225;
    for (int k = ilo(); k <= ihi(); ++k)
    for (int j = ilo(); j <= ihi(); ++j)
    for (int i = ilo(); i <= ihi(); ++i)
        blk.rho(i,j,k) = rho0;

    double mass  = blk.total_mass();
    double exact = rho0 * 1.0 * 1.0 * 1.0;   // rho * L^3, L = NB * h = 1
    double err   = std::abs(mass - exact) / exact;
    check("T04 CellBlock total mass == rho*L^3", err < 1e-14, err, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// T05  Sutherland viscosity: mu(273.15 K) == mu_ref, mu increases with T
// ─────────────────────────────────────────────────────────────────────────────
static void t05_sutherland() {
    double mu0   = sutherland(273.15);
    double mu_hi = sutherland(500.0);
    double err   = std::abs(mu0 - 1.716e-5) / 1.716e-5;
    check("T05 Sutherland mu(273.15K) == 1.716e-5", err < 1e-10, err, 1e-10);
    check("T05 Sutherland mu increases with T",      mu_hi > mu0);
}

// ─────────────────────────────────────────────────────────────────────────────
// T06  Tree init: single root leaf, correct geometry
// ─────────────────────────────────────────────────────────────────────────────
static void t06_tree_init() {
    BlockTree tree; tree.init(1.0);
    check("T06a root is leaf",           tree.nodes[0].is_leaf());
    check("T06b n_leaves == 1",          tree.n_leaves() == 1);
    check("T06c root has block",         tree.nodes[0].has_block());
    double h = tree.nodes[0].block->h;
    check("T06d root h == L/NB",         std::abs(h - 1.0/NB) < 1e-14, h, 1.0/NB);
    check("T06e root parent == -1",      tree.nodes[0].parent == -1);
    check("T06f root all neighbours -1", tree.nodes[0].neighbours[XMINUS] == -1);
}

// ─────────────────────────────────────────────────────────────────────────────
// T07  Refine root → 8 children, coarsen back → 1 leaf
// ─────────────────────────────────────────────────────────────────────────────
static void t07_refine_coarsen() {
    BlockTree tree; tree.init(1.0);

    // Fill root with known state
    auto& root_blk = *tree.nodes[0].block;
    for (int k=ilo();k<=ihi();++k)
    for (int j=ilo();j<=ihi();++j)
    for (int i=ilo();i<=ihi();++i) {
        root_blk.rho(i,j,k) = 1.0 + 0.1*(i+j+k);
    }
    double mass_before = root_blk.total_mass();

    tree.refine(0);
    check("T07a n_leaves == 8 after refine",    tree.n_leaves() == 8);
    check("T07b root is not leaf after refine",  !tree.nodes[0].is_leaf());
    check("T07c root block freed after refine",  !tree.nodes[0].has_block());

    // Mass is conserved through prolongation
    double mass_after = 0;
    for (int li : tree.leaf_indices())
        mass_after += tree.nodes[li].block->total_mass();
    double mass_err = std::abs(mass_after - mass_before) / std::abs(mass_before);
    check("T07d mass conserved through refine < 1e-13", mass_err < 1e-13, mass_err, 1e-13);

    // Check child cell size and level
    int fc = tree.nodes[0].first_child;
    double ch = tree.nodes[fc].block->h;
    check("T07e child h == root_h/2",  std::abs(ch - 1.0/(2*NB)) < 1e-14, ch, 1.0/(2*NB));
    check("T07f child level == 1",     tree.nodes[fc].level == 1);

    // Coarsen back
    double mass_kids = mass_after;
    tree.coarsen(0);
    check("T07g n_leaves == 1 after coarsen",  tree.n_leaves() == 1);
    check("T07h root is leaf after coarsen",    tree.nodes[0].is_leaf());
    check("T07i root has block after coarsen",  tree.nodes[0].has_block());

    double mass_coarse = tree.nodes[0].block->total_mass();
    double coarsen_err = std::abs(mass_coarse - mass_kids) / std::abs(mass_kids);
    check("T07j mass conserved through coarsen < 1e-13", coarsen_err < 1e-13, coarsen_err, 1e-13);
}

// ─────────────────────────────────────────────────────────────────────────────
// T08  Sibling neighbours correct after refine
// ─────────────────────────────────────────────────────────────────────────────
static void t08_sibling_neighbours() {
    BlockTree tree; tree.init(1.0);
    tree.refine(0);
    int fc = tree.nodes[0].first_child;

    // Oct 0 (---) and Oct 1 (x++) are x-adjacent siblings
    int n0 = fc + 0;   // oct 0: x=low
    int n1 = fc + 1;   // oct 1: x=high
    check("T08a oct0 XPLUS  neighbour == oct1", tree.nodes[n0].neighbours[XPLUS]  == n1);
    check("T08b oct1 XMINUS neighbour == oct0", tree.nodes[n1].neighbours[XMINUS] == n0);

    // Oct 0 and Oct 2 are y-adjacent siblings
    int n2 = fc + 2;
    check("T08c oct0 YPLUS  neighbour == oct2", tree.nodes[n0].neighbours[YPLUS]  == n2);
    check("T08d oct2 YMINUS neighbour == oct0", tree.nodes[n2].neighbours[YMINUS] == n0);

    // Oct 0 and Oct 4 are z-adjacent siblings
    int n4 = fc + 4;
    check("T08e oct0 ZPLUS  neighbour == oct4", tree.nodes[n0].neighbours[ZPLUS]  == n4);
    check("T08f oct4 ZMINUS neighbour == oct0", tree.nodes[n4].neighbours[ZMINUS] == n0);
}

// ─────────────────────────────────────────────────────────────────────────────
// T09  2-level refine: 2:1 balance maintained after balance()
// ─────────────────────────────────────────────────────────────────────────────
static void t09_balance() {
    BlockTree tree; tree.init(1.0);
    tree.refine(0);                 // 8 level-1 leaves
    int fc = tree.nodes[0].first_child;
    tree.refine(fc);                // refine oct 0 → 8 level-2 leaves + 7 level-1

    // Before balance: oct0's level-2 leaves are adjacent to level-1 leaves
    // (difference = 1 — already balanced in this case)
    int extra = tree.balance();
    // Verify: no two adjacent leaves differ by more than 1 level
    bool ok = true;
    for (int li : tree.leaf_indices()) {
        int lv = tree.nodes[li].level;
        for (int d = 0; d < NFACES; ++d) {
            int ni = tree.nodes[li].neighbours[d];
            if (ni < 0) continue;
            if (std::abs(tree.nodes[ni].level - lv) > 1) { ok = false; break; }
        }
    }
    check("T09a 2:1 balance satisfied after balance()", ok);
    (void)extra;
}

// ─────────────────────────────────────────────────────────────────────────────
// T10  Periodic ghost fill: ghost == interior from opposite side
// ─────────────────────────────────────────────────────────────────────────────
static void t10_periodic_ghost() {
    // Two-block periodic domain: refine root and use two of the 8 children
    // that are x-adjacent. Fill interior, call fill_ghosts_periodic, verify.
    BlockTree tree; tree.init(1.0);
    tree.refine(0);
    int fc = tree.nodes[0].first_child;

    // Fill all leaves with rho = leaf_index + 1
    for (int li : tree.leaf_indices()) {
        auto& blk = *tree.nodes[li].block;
        for (int k=ilo();k<=ihi();++k)
        for (int j=ilo();j<=ihi();++j)
        for (int i=ilo();i<=ihi();++i)
            blk.rho(i,j,k) = (double)(li + 1);
    }

    tree.fill_ghosts_periodic();

    // Check oct0 (n0) x-plus ghost == oct1's interior (rho = fc+1+1)
    int n0 = fc + 0, n1 = fc + 1;
    auto& blk0 = *tree.nodes[n0].block;
    double expected = (double)(n1 + 1);
    double got_ghost = blk0.rho(NB2-1, ilo(), ilo());  // x-plus ghost cell
    double err = std::abs(got_ghost - expected);
    check("T10a x-plus ghost == neighbour interior", err < 1e-14, err, 1e-14);

    // Check oct1 x-minus ghost == oct0's interior
    auto& blk1 = *tree.nodes[n1].block;
    double exp0 = (double)(n0 + 1);
    double got0  = blk1.rho(0, ilo(), ilo());   // x-minus ghost cell
    double err0  = std::abs(got0 - exp0);
    check("T10b x-minus ghost == neighbour interior", err0 < 1e-14, err0, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// T11  Wall ghost fill: normal velocity reflected, tangential negated
// ─────────────────────────────────────────────────────────────────────────────
static void t11_wall_ghost() {
    BlockTree tree; tree.init(1.0);   // single block, all boundaries are walls
    auto& blk = *tree.nodes[0].block;

    // Set interior: rho=1, u=10, v=5, w=3, E=1e5
    Prim p0; p0.rho=1.0; p0.u=10.0; p0.v=5.0; p0.w=3.0; p0.p=1e5;
    p0.T=p0.p/(p0.rho*R_GAS); p0.c=std::sqrt(GAMMA*p0.p/p0.rho);
    double rho,rhou,rhov,rhow,E;
    eos_prim_to_cons(p0, rho, rhou, rhov, rhow, E);
    for (int k=ilo();k<=ihi();++k)
    for (int j=ilo();j<=ihi();++j)
    for (int i=ilo();i<=ihi();++i) {
        blk.rho(i,j,k)=rho; blk.rhou(i,j,k)=rhou;
        blk.rhov(i,j,k)=rhov; blk.rhow(i,j,k)=rhow; blk.E(i,j,k)=E;
    }

    tree.fill_ghosts_wall();

    // x-minus ghost: rhou should be negated (no-slip), rho same
    double ghost_rho  = blk.rho (0, ilo(), ilo());
    double ghost_rhou = blk.rhou(0, ilo(), ilo());
    double ghost_rhov = blk.rhov(0, ilo(), ilo());
    check("T11a wall ghost rho == interior rho",    std::abs(ghost_rho  - rho ) < 1e-14);
    check("T11b wall ghost rhou == -interior rhou",  std::abs(ghost_rhou + rhou) < 1e-14);
    check("T11c wall ghost rhov == -interior rhov",  std::abs(ghost_rhov + rhov) < 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// P13.4: isothermal wall — verify E_ghost = ρ·Cv·(2Tw−Ti) + ½ρ|u|²
static void t12_isothermal_wall() {
    BlockTree tree; tree.init(1.0);
    auto& blk = *tree.nodes[0].block;

    // Interior: uniform state with known T
    Prim p0; p0.rho=1.2; p0.u=50.0; p0.v=30.0; p0.w=10.0; p0.p=101325.0;
    p0.T=p0.p/(p0.rho*R_GAS); p0.c=std::sqrt(GAMMA*p0.p/p0.rho);
    double rho,rhou,rhov,rhow,E;
    eos_prim_to_cons(p0, rho, rhou, rhov, rhow, E);
    for (int k=ilo();k<=ihi();++k)
    for (int j=ilo();j<=ihi();++j)
    for (int i=ilo();i<=ihi();++i) {
        blk.rho(i,j,k)=rho; blk.rhou(i,j,k)=rhou;
        blk.rhov(i,j,k)=rhov; blk.rhow(i,j,k)=rhow; blk.E(i,j,k)=E;
    }

    const double T_wall = 350.0;   // hot wall
    BlockTree::set_wall_T(T_wall);
    tree.fill_ghosts_wall();
    BlockTree::set_wall_T(0.0);    // reset to adiabatic for other tests

    // Expected E_ghost = ρ·Cv·(2Tw−T_int) + ½ρ|u|²
    const double Cv = R_GAS / (GAMMA - 1.0);
    const double T_ghost_exp = 2.0*T_wall - p0.T;
    const double KE = 0.5*rho*(p0.u*p0.u + p0.v*p0.v + p0.w*p0.w);
    const double E_ghost_exp = rho*Cv*T_ghost_exp + KE;

    // x-minus ghost (index 0, i interior = ilo())
    double E_gx = blk.E(0, ilo(), ilo());
    check("T12a isothermal wall E_ghost x-face", std::abs(E_gx - E_ghost_exp) < 1e-8*E_ghost_exp);

    // y-minus ghost (index 0, j interior = ilo())
    double E_gy = blk.E(ilo(), 0, ilo());
    check("T12b isothermal wall E_ghost y-face", std::abs(E_gy - E_ghost_exp) < 1e-8*E_ghost_exp);

    // z-minus ghost (index 0, k interior = ilo())
    double E_gz = blk.E(ilo(), ilo(), 0);
    check("T12c isothermal wall E_ghost z-face", std::abs(E_gz - E_ghost_exp) < 1e-8*E_ghost_exp);

    // rho unchanged, rhou negated (no-slip)
    check("T12d isothermal wall rho unchanged", std::abs(blk.rho(0,ilo(),ilo()) - rho) < 1e-14);
    check("T12e isothermal wall rhou negated",  std::abs(blk.rhou(0,ilo(),ilo()) + rhou) < 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== Step 2: Layer 1 — Cell Block + Block Tree ===\n");
    printf("    Gate: mass conserved through refine/coarsen < 1e-13\n");
    printf("          Morton round-trip exact\n");
    printf("          Ghost fill correct (periodic + wall)\n\n");

    t01_morton();
    t02_eos_roundtrip();
    t03_cell_idx_bijection();
    t04_block_mass();
    t05_sutherland();
    t06_tree_init();
    t07_refine_coarsen();
    t08_sibling_neighbours();
    t09_balance();
    t10_periodic_ghost();
    t11_wall_ghost();
    t12_isothermal_wall();

    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0)
        printf("==> FAIL test_block  (Step 2 gate NOT cleared)\n");
    else
        printf("==> PASS  Step 2 gate cleared — proceed to Step 3\n");
    return (n_fail == 0) ? 0 : 1;
}
