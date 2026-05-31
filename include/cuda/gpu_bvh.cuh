#pragma once
// gpu_bvh.cuh — Flat AABB BVH built on CPU from an STL mesh, uploaded to GPU.
//
// Layout:
//   Interior node: left >= 0, right >= 0  (child indices into nodes[])
//   Leaf node:     left < 0               (tri_idx = ~left), right unused
//
// The tree is stored in a flat array built by build_recursive() using a
// "reserve slot first, fill children, then fill AABB" order so that the root
// is always at index 0.

#include "models/stl_loader.hpp"
#include <cuda_runtime.h>
#include <vector>

// ── BvhNode ───────────────────────────────────────────────────────────────────

struct BvhNode {
    float aabb_min[3];  // inclusive lower corner (padded by ±1e-5 on build)
    float aabb_max[3];  // inclusive upper corner
    int   left;         // >= 0: child index; < 0: leaf (tri = ~left)
    int   right;        // >= 0: child index; < 0: unused for leaves
};

// ── GpuBvh ───────────────────────────────────────────────────────────────────

struct GpuBvh {
    // Node array
    BvhNode* d_nodes = nullptr;

    // Triangle vertex SoA — 9 arrays of n_tris floats
    float* d_v0x = nullptr;  float* d_v0y = nullptr;  float* d_v0z = nullptr;
    float* d_v1x = nullptr;  float* d_v1y = nullptr;  float* d_v1z = nullptr;
    float* d_v2x = nullptr;  float* d_v2y = nullptr;  float* d_v2z = nullptr;

    // Triangle unit-normal SoA — 3 arrays of n_tris floats
    float* d_nx  = nullptr;  float* d_ny  = nullptr;  float* d_nz  = nullptr;

    int n_nodes = 0;
    int n_tris  = 0;

    GpuBvh() = default;
    GpuBvh(const GpuBvh&) = delete;
    GpuBvh& operator=(const GpuBvh&) = delete;
    ~GpuBvh();

    // Build BVH on CPU from StlMesh, then upload all data to device.
    void build(const StlMesh& mesh);

    bool ready() const noexcept { return n_nodes > 0; }
};

// ── bvh_sdf device function ───────────────────────────────────────────────────
//
// Returns the signed distance from (px, py, pz) to the BVH surface.
//
//   positive  →  query point is on the outward-normal side (fluid / exterior)
//   negative  →  query point is on the inward-normal side  (solid / interior)
//
// Also writes the outward wall normal (from the STL) at the closest surface
// point into (out_nx, out_ny, out_nz).
//
// Traversal: iterative depth-first stack, max depth 64 levels.

__device__ float bvh_sdf(
    const BvhNode* __restrict__ nodes, int n_nodes,
    const float* __restrict__ v0x, const float* __restrict__ v0y,
    const float* __restrict__ v0z,
    const float* __restrict__ v1x, const float* __restrict__ v1y,
    const float* __restrict__ v1z,
    const float* __restrict__ v2x, const float* __restrict__ v2y,
    const float* __restrict__ v2z,
    const float* __restrict__ tnx, const float* __restrict__ tny,
    const float* __restrict__ tnz,
    float px, float py, float pz,
    float& out_nx, float& out_ny, float& out_nz);
