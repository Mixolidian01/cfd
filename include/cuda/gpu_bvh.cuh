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
// Sign determination: generalised winding number (Van Oosterom & Strackee 1983).
// Robust for non-convex and non-watertight STL meshes.
//
// Also writes the outward wall normal (from the STL) at the closest surface
// point into (out_nx, out_ny, out_nz).
//
// Traversal: iterative depth-first stack, max depth 64 levels.
//
// Implementation is header-inline so that any .cu TU that includes this header
// gets the device code inlined (no -rdc / separate compilation required).

// ── Device helper: AABB squared distance ─────────────────────────────────────
__device__ __forceinline__ float bvh_aabb_sq_dist(
    const float* __restrict__ bmin,
    const float* __restrict__ bmax,
    float px, float py, float pz) noexcept
{
    const float coords[3] = { px, py, pz };
    float d2 = 0.0f;
    for (int a = 0; a < 3; ++a) {
        const float d = fmaxf(bmin[a] - coords[a], 0.0f)
                      + fmaxf(coords[a] - bmax[a], 0.0f);
        d2 += d * d;
    }
    return d2;
}

// ── Device helper: closest point on triangle (Ericson §5.1.5) ────────────────
__device__ __forceinline__ void bvh_closest_on_triangle(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz,
    float px, float py, float pz,
    float& rx, float& ry, float& rz) noexcept
{
    const float abx = bx-ax, aby = by-ay, abz = bz-az;
    const float acx = cx-ax, acy = cy-ay, acz = cz-az;
    const float apx = px-ax, apy = py-ay, apz = pz-az;

    const float d1 = abx*apx + aby*apy + abz*apz;
    const float d2 = acx*apx + acy*apy + acz*apz;

    if (d1 <= 0.0f && d2 <= 0.0f) { rx=ax; ry=ay; rz=az; return; }

    const float bpx = px-bx, bpy = py-by, bpz = pz-bz;
    const float d3  = abx*bpx + aby*bpy + abz*bpz;
    const float d4  = acx*bpx + acy*bpy + acz*bpz;

    if (d3 >= 0.0f && d4 <= d3) { rx=bx; ry=by; rz=bz; return; }

    const float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        rx = ax + v*abx;  ry = ay + v*aby;  rz = az + v*abz;
        return;
    }

    const float cpx = px-cx, cpy = py-cy, cpz = pz-cz;
    const float d5  = abx*cpx + aby*cpy + abz*cpz;
    const float d6  = acx*cpx + acy*cpy + acz*cpz;

    if (d6 >= 0.0f && d5 <= d6) { rx=cx; ry=cy; rz=cz; return; }

    const float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        rx = ax + w*acx;  ry = ay + w*acy;  rz = az + w*acz;
        return;
    }

    const float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4-d3) >= 0.0f && (d5-d6) >= 0.0f) {
        const float w = (d4-d3) / ((d4-d3) + (d5-d6));
        rx = bx + w*(cx-bx);  ry = by + w*(cy-by);  rz = bz + w*(cz-bz);
        return;
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    rx = ax + v*abx + w*acx;
    ry = ay + v*aby + w*acy;
    rz = az + v*abz + w*acz;
}

__device__ __forceinline__ float bvh_winding_number(
    int n_tris,
    const float* __restrict__ v0x, const float* __restrict__ v0y, const float* __restrict__ v0z,
    const float* __restrict__ v1x, const float* __restrict__ v1y, const float* __restrict__ v1z,
    const float* __restrict__ v2x, const float* __restrict__ v2y, const float* __restrict__ v2z,
    float px, float py, float pz) noexcept
{
    float w = 0.0f;
    // O(n_tris): suitable for small meshes (< ~2k tris); no BVH acceleration here
    for (int ti = 0; ti < n_tris; ++ti) {
        float ax=v0x[ti]-px, ay=v0y[ti]-py, az=v0z[ti]-pz;
        float bx=v1x[ti]-px, by=v1y[ti]-py, bz=v1z[ti]-pz;
        float cx=v2x[ti]-px, cy=v2y[ti]-py, cz=v2z[ti]-pz;
        const float ra=sqrtf(ax*ax+ay*ay+az*az), rb=sqrtf(bx*bx+by*by+bz*bz), rc=sqrtf(cx*cx+cy*cy+cz*cz);
        if (ra<1e-8f||rb<1e-8f||rc<1e-8f) continue;
        ax/=ra; ay/=ra; az/=ra; bx/=rb; by/=rb; bz/=rb; cx/=rc; cy/=rc; cz/=rc;
        const float num = ax*(by*cz-bz*cy)+ay*(bz*cx-bx*cz)+az*(bx*cy-by*cx);
        const float den = 1.0f+ax*bx+ay*by+az*bz+bx*cx+by*cy+bz*cz+cx*ax+cy*ay+cz*az;
        w += 2.0f * atan2f(num, den);
    }
    return w * (1.0f / (4.0f * 3.14159265358979323846f));
}

__device__ __forceinline__ float bvh_sdf(
    const BvhNode* __restrict__ nodes, int /*n_nodes*/, int n_tris,
    const float* __restrict__ v0x, const float* __restrict__ v0y,
    const float* __restrict__ v0z,
    const float* __restrict__ v1x, const float* __restrict__ v1y,
    const float* __restrict__ v1z,
    const float* __restrict__ v2x, const float* __restrict__ v2y,
    const float* __restrict__ v2z,
    const float* __restrict__ tnx, const float* __restrict__ tny,
    const float* __restrict__ tnz,
    float px, float py, float pz,
    float& out_nx, float& out_ny, float& out_nz)
{
    float best_dist2 = 1e30f;
    float best_cx = px, best_cy = py, best_cz = pz;
    int   best_tri = 0;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int idx = stack[--sp];
        const BvhNode& node = nodes[idx];

        const float ad2 = bvh_aabb_sq_dist(node.aabb_min, node.aabb_max, px, py, pz);
        if (ad2 >= best_dist2) continue;

        if (node.left < 0) {
            const int ti = ~node.left;
            float rx, ry, rz;
            bvh_closest_on_triangle(
                v0x[ti], v0y[ti], v0z[ti],
                v1x[ti], v1y[ti], v1z[ti],
                v2x[ti], v2y[ti], v2z[ti],
                px, py, pz,
                rx, ry, rz);
            const float dx = px-rx, dy = py-ry, dz = pz-rz;
            const float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_cx = rx;  best_cy = ry;  best_cz = rz;
                best_tri = ti;
            }
        } else {
            const float d2l = bvh_aabb_sq_dist(
                nodes[node.left].aabb_min, nodes[node.left].aabb_max, px, py, pz);
            const float d2r = bvh_aabb_sq_dist(
                nodes[node.right].aabb_min, nodes[node.right].aabb_max, px, py, pz);

            if (d2l <= d2r) {
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            } else {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }

    const float wn = bvh_winding_number(n_tris, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z, px,py,pz);
    const float sign = (wn > 0.5f) ? -1.0f : 1.0f;

    out_nx = tnx[best_tri];
    out_ny = tny[best_tri];
    out_nz = tnz[best_tri];

    return sign * sqrtf(best_dist2);
}
