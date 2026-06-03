#include "models/mesh_loader.hpp"
#include "models/stl_loader.hpp"   // backward-compat shim
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else  { printf("  FAIL  %s  %s\n", tag, msg); ++nfail; }
}

// ── helpers ───────────────────────────────────────────────────────────────────

static void write_binary_stl(const char* path) {
    FILE* f = fopen(path, "wb");
    assert(f);
    char hdr[80] = "test";
    fwrite(hdr, 1, 80, f);
    uint32_t n = 2;
    fwrite(&n, 4, 1, f);
    float tri1[12] = {0.f,0.f,1.f, 0.f,0.f,0.f, 1.f,0.f,0.f, 0.f,1.f,0.f};
    uint16_t attr = 0;
    fwrite(tri1, 4, 12, f); fwrite(&attr, 2, 1, f);
    float tri2[12] = {0.f,0.f,1.f, 1.f,0.f,0.f, 1.f,1.f,0.f, 0.f,1.f,0.f};
    fwrite(tri2, 4, 12, f); fwrite(&attr, 2, 1, f);
    fclose(f);
}

static void write_obj_triangles(const char* path) {
    std::ofstream f(path);
    f << "# unit square, two triangles, normals computed from cross product\n"
      << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
      << "f 1 2 3\nf 1 3 4\n";
}

static void write_obj_with_vn(const char* path) {
    std::ofstream f(path);
    f << "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
      << "vn 0 0 1\n"
      << "f 1//1 2//1 3//1\n";
}

static void write_obj_quad(const char* path) {
    std::ofstream f(path);
    f << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
      << "f 1 2 3 4\n";  // quad → fan → 2 triangles
}

// ── STL gates (backward compat) ───────────────────────────────────────────────

static void test_stl() {
    write_binary_stl("/tmp/ml_test.stl");
    TriangleMesh mesh = load_stl("/tmp/ml_test.stl");
    check(mesh.triangles.size() == 2, "SL1", "binary STL: 2 triangles");

    const auto& t0 = mesh.triangles[0];
    check(std::fabs(t0.normal[2] - 1.0f) < 1e-5f, "SL2", "normal[2] == 1.0");
    check(std::fabs(t0.v1[0]    - 1.0f) < 1e-5f,  "SL3", "v1[0] == 1.0");

    bool threw = false;
    try { load_stl("/tmp/does_not_exist_xyz.stl"); } catch (...) { threw = true; }
    check(threw, "SL4", "missing file throws");
}

// ── OBJ gates ─────────────────────────────────────────────────────────────────

static void test_obj() {
    write_obj_triangles("/tmp/ml_test.obj");
    TriangleMesh mesh = load_obj("/tmp/ml_test.obj");
    check(mesh.triangles.size() == 2, "ML1", "OBJ: 2 triangles from 4 verts");

    const auto& t0 = mesh.triangles[0];
    check(std::fabs(t0.v0[0]) < 1e-5f && std::fabs(t0.v0[1]) < 1e-5f,
          "ML2", "v0 at origin");
    check(std::fabs(t0.normal[2] - 1.0f) < 1e-4f, "ML3", "computed normal[2] == 1.0");

    write_obj_with_vn("/tmp/ml_normals.obj");
    TriangleMesh nm = load_obj("/tmp/ml_normals.obj");
    check(!nm.triangles.empty() && std::fabs(nm.triangles[0].normal[2] - 1.0f) < 1e-4f,
          "ML4", "vn normal[2] == 1.0");

    write_obj_quad("/tmp/ml_quad.obj");
    TriangleMesh qm = load_obj("/tmp/ml_quad.obj");
    check(qm.triangles.size() == 2, "ML5", "quad face → 2 triangles");
}

// ── Dispatcher gates ──────────────────────────────────────────────────────────

static void test_dispatch() {
    TriangleMesh ms = load_mesh("/tmp/ml_test.stl");
    check(ms.triangles.size() == 2, "ML6", "load_mesh dispatches .stl");

    TriangleMesh mo = load_mesh("/tmp/ml_test.obj");
    check(mo.triangles.size() == 2, "ML7", "load_mesh dispatches .obj");

    bool threw = false;
    try { load_mesh("/tmp/foo.iges"); } catch (...) { threw = true; }
    check(threw, "ML8", "unsupported extension throws");

    write_binary_stl("/tmp/ml_upper.STL");
    bool ok = false;
    try { TriangleMesh mu = load_mesh("/tmp/ml_upper.STL"); ok = !mu.triangles.empty(); }
    catch (...) {}
    check(ok, "ML9", "load_mesh handles uppercase .STL");
}

// ── Alias backward-compat ─────────────────────────────────────────────────────

static void test_aliases() {
    StlMesh sm = load_stl("/tmp/ml_test.stl");
    check(sm.triangles.size() == 2, "AL1", "StlMesh alias: load_stl still works");
    const StlTriangle& t = sm.triangles[0];
    check(std::fabs(t.normal[2] - 1.0f) < 1e-5f, "AL2", "StlTriangle alias: fields accessible");
}

int main() {
    test_stl();
    test_obj();
    test_dispatch();
    test_aliases();

    const int total = 4 + 5 + 4 + 2;
    printf("\n%s  %d tests\n", nfail == 0 ? "ALL PASS" : "SOME FAIL", total);
    return nfail;
}
