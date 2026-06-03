#include "models/primitive_gen.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else  { printf("  FAIL  %s  %s\n", tag, msg); ++nfail; }
}

int main() {
    // Box: 6 faces × 2 triangles = 12
    auto box = make_box(0.f,0.f,0.f, 1.f,1.f,1.f);
    check(box.triangles.size() == 12, "PG1", "box: 12 triangles");

    // Sphere nlon=16 nlat=8: north-cap 16 + (8-2)*16*2 + south-cap 16 = 224
    auto sph = make_sphere(0.f,0.f,0.f, 1.f, 16, 8);
    check(sph.triangles.size() == 224, "PG2", "sphere: 224 triangles");

    // Normals on sphere should be unit vectors
    bool nok = true;
    for (const auto& t : sph.triangles) {
        float n = t.normal[0]*t.normal[0]+t.normal[1]*t.normal[1]+t.normal[2]*t.normal[2];
        if (std::fabs(n-1.f) > 0.01f) { nok=false; break; }
    }
    check(nok, "PG3", "sphere: unit normals");

    // PG3b: sphere normals point outward
    bool outok = true;
    for (const auto& t : sph.triangles) {
        float cx = (t.v0[0]+t.v1[0]+t.v2[0])/3.f;
        float cy = (t.v0[1]+t.v1[1]+t.v2[1])/3.f;
        float cz = (t.v0[2]+t.v1[2]+t.v2[2])/3.f;
        float dot = t.normal[0]*cx + t.normal[1]*cy + t.normal[2]*cz; // sphere centred at 0
        if (dot < 0.f) { outok=false; break; }
    }
    check(outok, "PG3b", "sphere: outward-facing normals");

    // Cylinder nseg=16: nseg*2 side + nseg top + nseg bottom = 64
    auto cyl = make_cylinder(0.f,0.f, 0.f,1.f, 0.5f, 16);
    check(cyl.triangles.size() == 64, "PG4", "cylinder: 64 triangles");

    // PG4b: cylinder side normals point outward from axis
    bool cok = true;
    float ccx = 0.f, ccy = 0.f; // cylinder centred at (0,0) for the test call
    for (const auto& t : cyl.triangles) {
        float mx = (t.v0[0]+t.v1[0]+t.v2[0])/3.f - ccx;
        float my = (t.v0[1]+t.v1[1]+t.v2[1])/3.f - ccy;
        // Only check side faces: those whose centroid is not at z=0 or z=1 extremes
        float mz = (t.v0[2]+t.v1[2]+t.v2[2])/3.f;
        if (mz < 0.05f || mz > 0.95f) continue; // skip caps
        float dot = t.normal[0]*mx + t.normal[1]*my;
        if (dot < 0.f) { cok=false; break; }
    }
    check(cok, "PG4b", "cylinder: outward side normals");

    // Box normals axis-aligned
    bool bok = true;
    for (const auto& t : box.triangles) {
        int cnt=0;
        for (int i=0;i<3;i++) if (std::fabs(std::fabs(t.normal[i])-1.f)<0.01f) ++cnt;
        if (cnt != 1) { bok=false; break; }
    }
    check(bok, "PG5", "box: axis-aligned normals");

    printf("\n%s  7 tests\n", nfail==0?"ALL PASS":"SOME FAIL");
    return nfail;
}
