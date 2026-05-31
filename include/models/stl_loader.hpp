#pragma once
#include <array>
#include <vector>
#include <string>

struct StlTriangle {
    std::array<float,3> normal;
    std::array<float,3> v0, v1, v2;
};

struct StlMesh {
    std::vector<StlTriangle> triangles;
};

// Parse binary or ASCII STL. Throws std::runtime_error on failure.
StlMesh load_stl(const std::string& path);
