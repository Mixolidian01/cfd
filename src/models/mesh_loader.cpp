#include "models/mesh_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cmath>

// ── OBJ loader ────────────────────────────────────────────────────────────────

static int obj_idx(int raw, int n) {
    // OBJ is 1-based; negative indices count from the current end of the list.
    return raw > 0 ? raw - 1 : n + raw;
}

static void parse_face_vertex(const std::string& tok,
                               int n_verts, int n_normals,
                               int& vi, int& ni) {
    // Formats: "v"  "v/t"  "v//n"  "v/t/n"
    auto slash1 = tok.find('/');
    if (slash1 == std::string::npos) {
        vi = obj_idx(std::stoi(tok), n_verts);
        ni = -1;
        return;
    }
    vi = obj_idx(std::stoi(tok.substr(0, slash1)), n_verts);
    auto slash2 = tok.find('/', slash1 + 1);
    if (slash2 != std::string::npos && slash2 + 1 < tok.size()) {
        ni = obj_idx(std::stoi(tok.substr(slash2 + 1)), n_normals);
    } else {
        ni = -1;
    }
}

TriangleMesh load_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_mesh: cannot open " + path);

    std::vector<std::array<float,3>> verts;
    std::vector<std::array<float,3>> vnormals;
    TriangleMesh mesh;

    std::string line, tok;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        ss >> tok;
        if (tok == "v") {
            float x, y, z; ss >> x >> y >> z;
            verts.push_back({x, y, z});
        } else if (tok == "vn") {
            float x, y, z; ss >> x >> y >> z;
            vnormals.push_back({x, y, z});
        } else if (tok == "f") {
            std::vector<int> vis, nis;
            std::string vtok;
            while (ss >> vtok) {
                int vi, ni;
                parse_face_vertex(vtok, (int)verts.size(), (int)vnormals.size(),
                                   vi, ni);
                vis.push_back(vi);
                nis.push_back(ni);
            }
            // Fan-triangulate: (0,1,2), (0,2,3), …
            for (int i = 1; i + 1 < (int)vis.size(); ++i) {
                Triangle t;
                t.v0 = verts[vis[0]];
                t.v1 = verts[vis[i]];
                t.v2 = verts[vis[i + 1]];

                bool has_n = !vnormals.empty()
                          && nis[0]   >= 0 && nis[0]   < (int)vnormals.size()
                          && nis[i]   >= 0 && nis[i]   < (int)vnormals.size()
                          && nis[i+1] >= 0 && nis[i+1] < (int)vnormals.size();
                if (has_n) {
                    float nx = (vnormals[nis[0]][0] + vnormals[nis[i]][0] + vnormals[nis[i+1]][0]) / 3.f;
                    float ny = (vnormals[nis[0]][1] + vnormals[nis[i]][1] + vnormals[nis[i+1]][1]) / 3.f;
                    float nz = (vnormals[nis[0]][2] + vnormals[nis[i]][2] + vnormals[nis[i+1]][2]) / 3.f;
                    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
                    if (len > 1e-8f) { nx /= len; ny /= len; nz /= len; }
                    t.normal = {nx, ny, nz};
                } else {
                    float ax = t.v1[0]-t.v0[0], ay = t.v1[1]-t.v0[1], az = t.v1[2]-t.v0[2];
                    float bx = t.v2[0]-t.v0[0], by = t.v2[1]-t.v0[1], bz = t.v2[2]-t.v0[2];
                    float nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
                    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
                    if (len > 1e-8f) { nx /= len; ny /= len; nz /= len; }
                    t.normal = {nx, ny, nz};
                }
                mesh.triangles.push_back(t);
            }
        }
    }
    if (mesh.triangles.empty())
        throw std::runtime_error("load_mesh: no triangles found in " + path);
    return mesh;
}

// ── Extension dispatcher ──────────────────────────────────────────────────────

TriangleMesh load_mesh(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos)
        throw std::runtime_error("load_mesh: no file extension in '" + path + "'");
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (ext == ".stl") return load_stl(path);
    if (ext == ".obj") return load_obj(path);
    throw std::runtime_error("load_mesh: unsupported format '" + ext
                             + "' (supported: .stl, .obj)");
}
