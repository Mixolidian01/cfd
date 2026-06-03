#pragma once
#include <array>
#include <vector>
#include <string>

struct Triangle {
    std::array<float,3> normal;
    std::array<float,3> v0, v1, v2;
};

struct TriangleMesh {
    std::vector<Triangle> triangles;
};

// Parse binary or ASCII STL. Throws std::runtime_error on failure.
TriangleMesh load_stl(const std::string& path);

// Parse Wavefront OBJ. Quads/n-gons are fan-triangulated.
// Face normals are taken from 'vn' declarations if present, otherwise
// computed from the cross product. Throws std::runtime_error on failure.
TriangleMesh load_obj(const std::string& path);

// Dispatch to load_stl / load_obj by file extension (.stl, .obj).
// Throws std::runtime_error for unsupported extensions.
TriangleMesh load_mesh(const std::string& path);
