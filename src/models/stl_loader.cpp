#include "models/stl_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdint>

static bool is_binary_stl(std::ifstream& f) {
    char hdr[5] = {};
    f.read(hdr, 5);
    f.seekg(0);
    return !(hdr[0]=='s' && hdr[1]=='o' && hdr[2]=='l' && hdr[3]=='i' && hdr[4]=='d');
}

static StlMesh parse_binary(std::ifstream& f) {
    char hdr[80];
    f.read(hdr, 80);
    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), 4);
    StlMesh mesh;
    mesh.triangles.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        float buf[12];
        f.read(reinterpret_cast<char*>(buf), 48);
        uint16_t attr;
        f.read(reinterpret_cast<char*>(&attr), 2);
        StlTriangle t;
        t.normal = {buf[0], buf[1], buf[2]};
        t.v0     = {buf[3], buf[4], buf[5]};
        t.v1     = {buf[6], buf[7], buf[8]};
        t.v2     = {buf[9], buf[10], buf[11]};
        mesh.triangles.push_back(t);
    }
    return mesh;
}

static StlMesh parse_ascii(std::ifstream& f) {
    StlMesh mesh;
    std::string tok;
    StlTriangle tri{};
    int vcount = 0;
    while (f >> tok) {
        if (tok == "normal") {
            f >> tri.normal[0] >> tri.normal[1] >> tri.normal[2];
            vcount = 0;
        } else if (tok == "vertex") {
            std::array<float,3> v;
            f >> v[0] >> v[1] >> v[2];
            if      (vcount == 0) tri.v0 = v;
            else if (vcount == 1) tri.v1 = v;
            else                  tri.v2 = v;
            ++vcount;
        } else if (tok == "endfacet") {
            mesh.triangles.push_back(tri);
        }
    }
    return mesh;
}

StlMesh load_stl(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_stl: cannot open " + path);
    if (is_binary_stl(f)) return parse_binary(f);
    f.close();
    std::ifstream fa(path);
    return parse_ascii(fa);
}
