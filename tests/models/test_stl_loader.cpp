#include "models/stl_loader.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else  { printf("  FAIL  %s  %s\n", tag, msg); ++nfail; }
}

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

int main() {
    // SL1: binary parse
    write_binary_stl("/tmp/test_ibm.stl");
    StlMesh mesh = load_stl("/tmp/test_ibm.stl");
    check(mesh.triangles.size() == 2, "SL1", "binary STL: 2 triangles parsed");

    // SL2: normal direction preserved
    const auto& t0 = mesh.triangles[0];
    check(std::fabs(t0.normal[2] - 1.0f) < 1e-5f, "SL2", "normal[2] == 1.0");

    // SL3: vertex coords correct
    check(std::fabs(t0.v1[0] - 1.0f) < 1e-5f, "SL3", "v1[0] == 1.0");

    // SL4: nonexistent file throws
    bool threw = false;
    try { load_stl("/tmp/does_not_exist_xyz.stl"); }
    catch (...) { threw = true; }
    check(threw, "SL4", "missing file throws runtime_error");

    printf("\n%s  %d tests\n", nfail == 0 ? "ALL PASS" : "SOME FAIL", 4);
    return nfail;
}
