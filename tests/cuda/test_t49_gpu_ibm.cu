// tests/cuda/test_t49_gpu_ibm.cu
// IBM GPU gate: I5–I9.
// I5: GPU classify sphere STL matches CPU SphereLevelSet (>90% interior cells agree)
// I6: No-slip wall: all IBM ghost |u| < ambient u=0.3 (stationary wall)
// I7: Adiabatic wall: ghost T error < 2 K in uniform-temperature field
// I8: 10-step SSP-RK3 advance with IBM active → stable (dt > 0, ρ > 0)
// I9: Ghost count invariant across two consecutive build() calls (regrid resilience)

#include "cuda/gpu_graph.cuh"
#include "cuda/gpu_ibm.cuh"
#include "cuda/gpu_bvh.cuh"
#include "models/stl_loader.hpp"
#include "models/ibm.hpp"
#include "mesh/block_tree.hpp"
#include "mesh/cell_block.hpp"
#include "gpu_pool.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg, double v = -1.0) {
    if (ok) printf("  PASS  %-4s  %s\n", tag, msg);
    else  { printf("  FAIL  %-4s  %s  (val=%.3e)\n", tag, msg, v); ++nfail; }
}

// NB, NG, NB2, NCELL, GAMMA, R_GAS, cell_idx() all come from cell_block.hpp

// Generates a binary STL torus centred at (0.5, 0.5, 0.5): major radius R, minor r,
// n_phi x n_theta quad patches → 2*n_phi*n_theta triangles.
static void write_torus_stl(const char* path, float R, float r, int n_phi, int n_theta) {
    FILE* f = fopen(path, "wb");
    char header[80] = {};
    std::snprintf(header, sizeof(header), "torus R=%.2f r=%.2f", R, r);
    fwrite(header, 1, 80, f);
    uint32_t n_tris = 2u * (uint32_t)n_phi * (uint32_t)n_theta;
    fwrite(&n_tris, 4, 1, f);
    // Torus centred at (0.5, 0.5, 0.5) in the unit-cube domain
    auto torus_pt = [&](int ip, int it) -> std::array<float,3> {
        float phi   = 2.0f * 3.14159265f * ip / n_phi;
        float theta = 2.0f * 3.14159265f * it / n_theta;
        float x = (R + r * cosf(theta)) * cosf(phi) + 0.5f;
        float y = (R + r * cosf(theta)) * sinf(phi) + 0.5f;
        float z = r * sinf(theta)                   + 0.5f;
        return {x, y, z};
    };
    float norm[3] = {0, 0, 0};
    uint16_t attr = 0;
    for (int ip = 0; ip < n_phi; ++ip) {
        for (int it = 0; it < n_theta; ++it) {
            auto p00 = torus_pt(ip,             it);
            auto p10 = torus_pt((ip+1) % n_phi, it);
            auto p01 = torus_pt(ip,             (it+1) % n_theta);
            auto p11 = torus_pt((ip+1) % n_phi, (it+1) % n_theta);
            // tri 1: p00, p10, p11
            fwrite(norm,        4, 3, f);
            fwrite(p00.data(),  4, 3, f);
            fwrite(p10.data(),  4, 3, f);
            fwrite(p11.data(),  4, 3, f);
            fwrite(&attr, 2, 1, f);
            // tri 2: p00, p11, p01
            fwrite(norm,        4, 3, f);
            fwrite(p00.data(),  4, 3, f);
            fwrite(p11.data(),  4, 3, f);
            fwrite(p01.data(),  4, 3, f);
            fwrite(&attr, 2, 1, f);
        }
    }
    fclose(f);
}

static GpuPool g_pool;

static void upload_block(CellBlock* blk) {
    if (!g_pool.has_device(blk)) g_pool.alloc(blk);
    g_pool.upload(blk);
}

// Write a lat-lon sphere STL (128 triangles) centred at (cx,cy,cz) radius R
static void write_sphere_stl(float cx, float cy, float cz, float R) {
    FILE* f = fopen("/tmp/sphere_t49.stl", "wb");
    const int NLAT = 8, NLON = 8;
    std::vector<std::array<float,12>> tris;
    for (int la = 0; la < NLAT; ++la) {
        float t0 = (float)M_PI * la / NLAT - (float)M_PI / 2;
        float t1 = (float)M_PI * (la + 1) / NLAT - (float)M_PI / 2;
        for (int lo = 0; lo < NLON; ++lo) {
            float p0 = 2 * (float)M_PI * lo / NLON;
            float p1 = 2 * (float)M_PI * (lo + 1) / NLON;
            auto pt = [&](float t, float p) -> std::array<float, 3> {
                return {cx + R*cosf(t)*cosf(p), cy + R*cosf(t)*sinf(p), cz + R*sinf(t)};
            };
            auto a = pt(t0,p0), b = pt(t0,p1), c = pt(t1,p0), d = pt(t1,p1);
            float na[3] = {cosf(t0)*cosf(p0), cosf(t0)*sinf(p0), sinf(t0)};
            tris.push_back({na[0],na[1],na[2], a[0],a[1],a[2], b[0],b[1],b[2], c[0],c[1],c[2]});
            tris.push_back({na[0],na[1],na[2], b[0],b[1],b[2], d[0],d[1],d[2], c[0],c[1],c[2]});
        }
    }
    char hdr[80] = "sphere"; fwrite(hdr, 1, 80, f);
    uint32_t n = (uint32_t)tris.size(); fwrite(&n, 4, 1, f);
    for (auto& t2 : tris) {
        fwrite(t2.data(), 4, 12, f);
        uint16_t attr = 0; fwrite(&attr, 2, 1, f);
    }
    fclose(f);
}

int main() {
    printf("=== IBM GPU gate (t49) ===\n\n");

    // ── Setup ─────────────────────────────────────────────────────────────────
    write_sphere_stl(0.5f, 0.5f, 0.5f, 0.25f);
    StlMesh mesh = load_stl("/tmp/sphere_t49.stl");
    GpuBvh bvh;
    bvh.build(mesh);

    BlockTree tree; tree.init(1.0);
    CellBlock* blk = tree.nodes[0].block.get();
    upload_block(blk);

    auto reset_ic = [&](double rho, double u, double p) {
        for (int f = 0; f < NCELL; ++f) {
            blk->Q[0][f] = rho;
            blk->Q[1][f] = rho * u;
            blk->Q[2][f] = 0.0;
            blk->Q[3][f] = 0.0;
            blk->Q[4][f] = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
        }
        g_pool.upload(blk);
    };

    // ── I5: GPU classify matches CPU SphereLevelSet ───────────────────────────
    reset_ic(1.0, 0.0, 1.0);
    GpuIbmList ibm; ibm.wall_bc = 0;
    ibm.build(tree, g_pool, bvh);

    std::vector<int8_t> h_ct(NCELL);
    cudaMemcpy(h_ct.data(), ibm.d_cell_type_pool, NCELL, cudaMemcpyDeviceToHost);

    SphereLevelSet ls(0.5, 0.5, 0.5, 0.25);
    std::vector<CellType> cpu_ct(NCELL, CellType::FLUID);
    classify_ibm_cells(*blk, ls, cpu_ct.data());

    int match = 0, total = 0;
    for (int k = NG; k < NG+NB; ++k)
    for (int j = NG; j < NG+NB; ++j)
    for (int i = NG; i < NG+NB; ++i) {
        int f = cell_idx(i,j,k); ++total;
        // CPU uses 0=FLUID, 1=SOLID; GPU uses 0=FLUID, 1=SOLID, 2=IBM_GHOST
        // IBM_GHOST (2) is promoted from FLUID, so compare 0 vs !=1 for solid agreement
        bool cpu_solid = ((int)cpu_ct[f] == 1);
        bool gpu_solid = (h_ct[f] == 1);
        if (cpu_solid == gpu_solid) ++match;
    }
    check((double)match / total > 0.90, "I5",
          "GPU classify matches CPU SphereLevelSet (>90% solid/fluid agree)",
          (double)match / total);

    // ── I6: No-slip ghost velocities ──────────────────────────────────────────
    reset_ic(1.0, 0.3, 1.0);
    ibm.wall_bc = 0; ibm.u_wall = ibm.v_wall = ibm.w_wall = 0.f;
    ibm.build(tree, g_pool, bvh);
    ibm.exec(nullptr);
    cudaDeviceSynchronize();
    g_pool.download(blk);

    std::vector<int8_t> h_ct2(NCELL);
    cudaMemcpy(h_ct2.data(), ibm.d_cell_type_pool, NCELL, cudaMemcpyDeviceToHost);
    int g_ok = 0, g_tot = 0;
    for (int f = 0; f < NCELL; ++f) {
        if (h_ct2[f] != 2) continue;
        ++g_tot;
        double u = blk->Q[1][f] / blk->Q[0][f];
        if (u < -0.05) ++g_ok; // no-slip reflection: u_ghost ≈ −u_image < 0
    }
    check(g_tot > 0, "I6a", "At least one IBM_GHOST cell exists", (double)g_tot);
    check(g_tot == 0 || (double)g_ok / g_tot > 0.95,
          "I6b", "No-slip: ghost u < 0 (velocity reflected, >95% of ghosts)",
          g_tot > 0 ? (double)g_ok / g_tot : 1.0);

    // ── I7: Adiabatic ghost temperature ───────────────────────────────────────
    const double T_amb = 300.0, rho_amb = 1.2;
    reset_ic(rho_amb, 0.0, rho_amb * R_GAS * T_amb);
    ibm.wall_bc = 0; // NoSlip/Adiabatic
    ibm.build(tree, g_pool, bvh);
    ibm.exec(nullptr);
    cudaDeviceSynchronize();
    g_pool.download(blk);

    std::vector<int8_t> h_ct3(NCELL);
    cudaMemcpy(h_ct3.data(), ibm.d_cell_type_pool, NCELL, cudaMemcpyDeviceToHost);
    double max_T_err = 0;
    for (int f = 0; f < NCELL; ++f) {
        if (h_ct3[f] != 2) continue;
        double rho = blk->Q[0][f];
        double u = blk->Q[1][f]/rho, v = blk->Q[2][f]/rho, w = blk->Q[3][f]/rho;
        double p = (GAMMA - 1.0) * (blk->Q[4][f] - 0.5*rho*(u*u+v*v+w*w));
        double T = p / (rho * R_GAS);
        max_T_err = std::max(max_T_err, std::fabs(T - T_amb));
    }
    check(max_T_err < 2.0, "I7",
          "Adiabatic: ghost T ≈ T_ambient (err < 2 K)", max_T_err);

    // ── I8: 10-step SSP-RK3 advance with IBM active ───────────────────────────
    reset_ic(1.0, 0.1, 1.0);
    GpuGraphSolver solver;
    solver.set_gpu_ibm(&bvh, 0, 0.f, 0.f, 0.f, 300.f);
    solver.build(tree, g_pool, 0);

    bool stable = true;
    for (int s = 0; s < 10 && stable; ++s) {
        double dt = solver.advance(tree, 0.5);
        if (!std::isfinite(dt) || dt <= 0.0) { stable = false; break; }
    }
    solver.download_q(tree);
    bool rho_ok = true;
    for (int f = 0; f < NCELL; ++f)
        if (blk->Q[0][f] <= 0.0 || !std::isfinite(blk->Q[0][f])) { rho_ok = false; }
    check(stable && rho_ok, "I8", "10-step SSP-RK3 with IBM: stable and ρ > 0");

    // ── I9: Ghost count invariant across rebuild ───────────────────────────────
    int ng1 = ibm.n_ghosts;
    ibm.build(tree, g_pool, bvh); // rebuild on same topology (simulates regrid)
    int ng2 = ibm.n_ghosts;
    check(ng1 > 0,     "I9a", "n_ghosts > 0 after build()", (double)ng1);
    check(ng1 == ng2,  "I9b", "n_ghosts unchanged after rebuild",
          (double)std::abs(ng1 - ng2));

    // ── W5: winding-number sign on a non-convex torus ─────────────────────────
    // Torus: major radius R=0.30, minor radius r=0.09, centred at (0.5,0.5,0.5).
    // Domain [0,1]^3, 1 leaf block → h = 1/NB = 0.125.
    // Cell(i,j,k) centre: ((i+0.5)*h, (j+0.5)*h, (k+0.5)*h)
    //
    // SOLID cell (inside tube): i=6,j=3,k=3 → centre=(0.8125, 0.4375, 0.4375)
    //   Nearest ring point at phi=0: (0.80, 0.50, 0.50).
    //   Dist ≈ sqrt(0.0125²+0.0625²+0.0625²) ≈ 0.0893 < r=0.09 → inside tube.
    //
    // FLUID cell (in central hole): i=3,j=3,k=3 → centre=(0.4375, 0.4375, 0.4375)
    //   r_xy = sqrt(2*(0.4375-0.5)²) ≈ 0.0884; dist_from_ring ≈ 0.221 > r → outside.
    {
        const char* torus_stl = "/tmp/torus_w5.stl";
        write_torus_stl(torus_stl, 0.30f, 0.09f, 12, 8);
        StlMesh torus_mesh = load_stl(torus_stl);
        GpuBvh  torus_bvh;
        torus_bvh.build(torus_mesh);

        BlockTree torus_tree; torus_tree.init(1.0);
        CellBlock* tblk = torus_tree.nodes[0].block.get();
        g_pool.alloc(tblk);
        // zero-init Q
        for (int fv = 0; fv < 5; ++fv)
            for (int fi = 0; fi < NCELL; ++fi) tblk->Q[fv][fi] = 1.0;
        g_pool.upload(tblk);

        GpuIbmList torus_ibm; torus_ibm.wall_bc = 0;
        torus_ibm.build(torus_tree, g_pool, torus_bvh);

        std::vector<int8_t> h_tct(NCELL);
        cudaMemcpy(h_tct.data(), torus_ibm.d_cell_type_pool, NCELL, cudaMemcpyDeviceToHost);

        // flat index: cell_idx(i,j,k) uses (k*NB2 + j)*NB2 + i with ghost offset
        int f_solid = cell_idx(6+NG, 3+NG, 3+NG);  // interior coords 6,3,3
        int f_fluid = cell_idx(3+NG, 3+NG, 3+NG);  // interior coords 3,3,3

        bool solid_ok = (h_tct[f_solid] == 1);  // IBM_SOLID
        bool fluid_ok = (h_tct[f_fluid] != 1);  // not SOLID (FLUID or IBM_GHOST)

        check(solid_ok, "W5a",
              "Winding-number sign: tube interior cell is SOLID",
              (double)h_tct[f_solid]);
        check(fluid_ok, "W5b",
              "Winding-number sign: torus hole cell is FLUID (not SOLID)",
              (double)h_tct[f_fluid]);

        g_pool.free(tblk);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    g_pool.free(blk);
    printf("\n=== %s  %d gate(s) failed ===\n",
           nfail == 0 ? "PASS" : "FAIL", nfail);
    return nfail;
}
