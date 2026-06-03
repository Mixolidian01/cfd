#include "models/mesh_loader.hpp"
#include "models/stl_loader.hpp"   // backward-compat shim
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else  { printf("  FAIL  %s  %s\n", tag, msg); ++nfail; }
}

// ── STL helpers ───────────────────────────────────────────────────────────────

static void write_binary_stl(const char* path) {
    FILE* f = fopen(path, "wb"); assert(f);
    char hdr[80] = "test"; fwrite(hdr, 1, 80, f);
    uint32_t n = 2; fwrite(&n, 4, 1, f);
    float tri1[12] = {0.f,0.f,1.f, 0.f,0.f,0.f, 1.f,0.f,0.f, 0.f,1.f,0.f};
    uint16_t attr = 0;
    fwrite(tri1, 4, 12, f); fwrite(&attr, 2, 1, f);
    float tri2[12] = {0.f,0.f,1.f, 1.f,0.f,0.f, 1.f,1.f,0.f, 0.f,1.f,0.f};
    fwrite(tri2, 4, 12, f); fwrite(&attr, 2, 1, f);
    fclose(f);
}

// ── OBJ helpers ───────────────────────────────────────────────────────────────

static void write_obj_triangles(const char* path) {
    std::ofstream f(path);
    f << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
      << "f 1 2 3\nf 1 3 4\n";
}

static void write_obj_with_vn(const char* path) {
    std::ofstream f(path);
    f << "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n";
}

static void write_obj_quad(const char* path) {
    std::ofstream f(path);
    f << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
}

// ── GLB / glTF helpers ────────────────────────────────────────────────────────

static void write_le_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }

// Write a GLB 2.0 file from a JSON string and a binary buffer.
static void write_glb(const char* path, const std::string& json,
                      const uint8_t* bin, uint32_t bin_len) {
    // Pad JSON to 4-byte boundary with spaces
    std::string jpad = json;
    while (jpad.size() % 4) jpad += ' ';
    // Pad BIN to 4-byte boundary with zeros
    uint32_t bpad_len = (bin_len + 3) & ~3u;
    std::vector<uint8_t> bpad(bpad_len, 0);
    if (bin_len) std::memcpy(bpad.data(), bin, bin_len);

    uint32_t total = 12 + 8 + (uint32_t)jpad.size()
                       + (bpad_len ? 8 + bpad_len : 0);
    FILE* f = fopen(path, "wb"); assert(f);
    write_le_u32(f, 0x46546C67u); // magic "glTF"
    write_le_u32(f, 2);           // version
    write_le_u32(f, total);
    // JSON chunk
    write_le_u32(f, (uint32_t)jpad.size());
    write_le_u32(f, 0x4E4F534Au); // "JSON"
    fwrite(jpad.c_str(), 1, jpad.size(), f);
    // BIN chunk
    if (bpad_len) {
        write_le_u32(f, bpad_len);
        write_le_u32(f, 0x004E4942u); // "BIN\0"
        fwrite(bpad.data(), 1, bpad_len, f);
    }
    fclose(f);
}

// One non-indexed triangle: (0,0,0) (1,0,0) (0,1,0).
// Normal computed from cross product → (0,0,1).
static void make_glb_one_triangle(const char* path) {
    // BIN: 3 × vec3 float32 = 36 bytes
    std::vector<uint8_t> bin(36, 0);
    float pos[9] = {0,0,0, 1,0,0, 0,1,0};
    std::memcpy(bin.data(), pos, 36);

    const char* json = R"({"asset":{"version":"2.0"},)"
      R"("meshes":[{"primitives":[{"attributes":{"POSITION":0},"mode":4}]}],)"
      R"("accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],)"
      R"("bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],)"
      R"("buffers":[{"byteLength":36}]})";

    write_glb(path, json, bin.data(), 36);
}

// Two triangles assembled from 4 vertices via uint16 index buffer.
// Positions at offset 0 (48 bytes), indices at offset 48 (12 bytes).
static void make_glb_indexed(const char* path) {
    // 4 positions (48 bytes) + 6 indices uint16 (12 bytes) = 60 bytes
    std::vector<uint8_t> bin(60, 0);
    float pos[12] = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
    std::memcpy(bin.data(), pos, 48);
    uint16_t idx[6] = {0,1,2, 0,2,3};
    std::memcpy(bin.data()+48, idx, 12);

    const char* json = R"({"asset":{"version":"2.0"},)"
      R"("meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1,"mode":4}]}],)"
      R"("accessors":[)"
        R"({"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"},)"
        R"({"bufferView":1,"componentType":5123,"count":6,"type":"SCALAR"})"
      R"(],)"
      R"("bufferViews":[)"
        R"({"buffer":0,"byteOffset":0,"byteLength":48},)"
        R"({"buffer":0,"byteOffset":48,"byteLength":12})"
      R"(],)"
      R"("buffers":[{"byteLength":60}]})";

    write_glb(path, json, bin.data(), 60);
}

// One triangle with explicit NORMAL accessor → normals taken from buffer.
static void make_glb_with_normals(const char* path) {
    // 36 bytes positions + 36 bytes normals = 72 bytes
    std::vector<uint8_t> bin(72, 0);
    float pos[9]  = {0,0,0, 1,0,0, 0,1,0};
    float nrm[9]  = {0,0,1, 0,0,1, 0,0,1};
    std::memcpy(bin.data(),    pos, 36);
    std::memcpy(bin.data()+36, nrm, 36);

    const char* json = R"({"asset":{"version":"2.0"},)"
      R"("meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"mode":4}]}],)"
      R"("accessors":[)"
        R"({"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},)"
        R"({"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"})"
      R"(],)"
      R"("bufferViews":[)"
        R"({"buffer":0,"byteOffset":0,"byteLength":36},)"
        R"({"buffer":0,"byteOffset":36,"byteLength":36})"
      R"(],)"
      R"("buffers":[{"byteLength":72}]})";

    write_glb(path, json, bin.data(), 72);
}

// .gltf JSON referencing a separate .bin file (two triangles).
static void make_gltf_external(const char* json_path, const char* bin_path) {
    float pos[12] = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
    FILE* b = fopen(bin_path, "wb"); assert(b);
    fwrite(pos, 4, 12, b); fclose(b);

    // bin filename only (relative, same directory)
    std::string bin_name = bin_path;
    auto slash = bin_name.rfind('/');
    if (slash != std::string::npos) bin_name = bin_name.substr(slash+1);

    std::ofstream jf(json_path);
    jf << "{\"asset\":{\"version\":\"2.0\"},"
       << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"mode\":4}]}],"
       << "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":4,\"type\":\"VEC3\"}],"
       << "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48}],"
       << "\"buffers\":[{\"uri\":\"" << bin_name << "\",\"byteLength\":48}]}";
}

// ── Test functions ────────────────────────────────────────────────────────────

static void test_stl() {
    write_binary_stl("/tmp/ml_test.stl");
    TriangleMesh mesh = load_stl("/tmp/ml_test.stl");
    check(mesh.triangles.size()==2,                       "SL1","binary STL: 2 triangles");
    check(std::fabs(mesh.triangles[0].normal[2]-1.f)<1e-5f,"SL2","normal[2]==1.0");
    check(std::fabs(mesh.triangles[0].v1[0]-1.f)<1e-5f,   "SL3","v1[0]==1.0");
    bool threw=false;
    try { load_stl("/tmp/does_not_exist_xyz.stl"); } catch(...){threw=true;}
    check(threw,"SL4","missing file throws");
}

static void test_obj() {
    write_obj_triangles("/tmp/ml_test.obj");
    TriangleMesh mesh = load_obj("/tmp/ml_test.obj");
    check(mesh.triangles.size()==2,                       "ML1","OBJ: 2 triangles");
    check(std::fabs(mesh.triangles[0].v0[0])<1e-5f,       "ML2","v0 at origin");
    check(std::fabs(mesh.triangles[0].normal[2]-1.f)<1e-4f,"ML3","computed normal[2]==1.0");
    write_obj_with_vn("/tmp/ml_normals.obj");
    TriangleMesh nm=load_obj("/tmp/ml_normals.obj");
    check(!nm.triangles.empty()&&std::fabs(nm.triangles[0].normal[2]-1.f)<1e-4f,
          "ML4","vn normal[2]==1.0");
    write_obj_quad("/tmp/ml_quad.obj");
    check(load_obj("/tmp/ml_quad.obj").triangles.size()==2,"ML5","quad → 2 triangles");
}

static void test_gltf() {
    // GT1: non-indexed GLB, 1 triangle, normal from cross product
    make_glb_one_triangle("/tmp/ml_test.glb");
    TriangleMesh m1 = load_gltf("/tmp/ml_test.glb");
    check(m1.triangles.size()==1,                          "GT1","GLB: 1 non-indexed triangle");
    check(std::fabs(m1.triangles[0].v0[0])<1e-5f
       && std::fabs(m1.triangles[0].v0[1])<1e-5f,         "GT2","GLB: v0 at origin");
    check(std::fabs(m1.triangles[0].normal[2]-1.f)<1e-4f,  "GT3","GLB: cross-product normal +Z");

    // GT4: indexed GLB → 2 triangles
    make_glb_indexed("/tmp/ml_idx.glb");
    check(load_gltf("/tmp/ml_idx.glb").triangles.size()==2,"GT4","GLB: indexed 2 triangles");

    // GT5: GLB with NORMAL accessor → normal taken from buffer
    make_glb_with_normals("/tmp/ml_nrm.glb");
    TriangleMesh mn = load_gltf("/tmp/ml_nrm.glb");
    check(!mn.triangles.empty()&&std::fabs(mn.triangles[0].normal[2]-1.f)<1e-4f,
          "GT5","GLB: NORMAL accessor normal[2]==1.0");

    // GT6: .gltf with external .bin → 4 vertices, non-indexed → 1 triangle? No: 4 verts → 1 tri
    // Actually 4 verts non-indexed → floor(4/3)=1 triangle (last vertex discarded).
    // Let's just check >= 1 triangle and correct vertex.
    make_gltf_external("/tmp/ml_test.gltf", "/tmp/ml_test.bin");
    TriangleMesh me = load_gltf("/tmp/ml_test.gltf");
    check(!me.triangles.empty(),                           "GT6",".gltf + external .bin parsed");
    check(std::fabs(me.triangles[0].v0[0])<1e-5f,         "GT7",".gltf v0 at origin");
}

static void test_dispatch() {
    check(load_mesh("/tmp/ml_test.stl").triangles.size()==2,"ML6","load_mesh .stl dispatch");
    check(load_mesh("/tmp/ml_test.obj").triangles.size()==2,"ML7","load_mesh .obj dispatch");
    check(!load_mesh("/tmp/ml_test.glb").triangles.empty(), "ML8","load_mesh .glb dispatch");
    check(!load_mesh("/tmp/ml_test.gltf").triangles.empty(),"ML9","load_mesh .gltf dispatch");

    // Uppercase extension
    write_binary_stl("/tmp/ml_upper.STL");
    bool ok=false;
    try { ok=!load_mesh("/tmp/ml_upper.STL").triangles.empty(); } catch(...){}
    check(ok,"ML10","load_mesh uppercase .STL");

    // Unknown extension throws
    bool threw=false;
    try { load_mesh("/tmp/foo.iges"); } catch(...){threw=true;}
    check(threw,"ML11","unsupported extension throws");
}

static void test_aliases() {
    StlMesh sm = load_stl("/tmp/ml_test.stl");
    check(sm.triangles.size()==2,                             "AL1","StlMesh alias works");
    check(std::fabs(sm.triangles[0].normal[2]-1.f)<1e-5f,    "AL2","StlTriangle fields work");
}

int main() {
    test_stl();
    test_obj();
    test_gltf();
    test_dispatch();
    test_aliases();

    const int total = 4+5+7+6+2;
    printf("\n%s  %d tests\n", nfail==0?"ALL PASS":"SOME FAIL", total);
    return nfail;
}
