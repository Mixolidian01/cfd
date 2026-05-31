// gpu_bvh.cu — CPU AABB BVH build + device upload; bvh_sdf device function.
//
// Build strategy: recursive median split on the longest centroid-AABB axis.
//   Leaf when span == 1 triangle.  Left < 0 encodes leaf (tri_idx = ~left).
//   The root node occupies index 0.
//
// Device traversal: iterative DFS with a 64-entry stack; AABB early-out.

#include "cuda/gpu_bvh.cuh"
#include "cuda/gpu_check.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// ── AABB padding ─────────────────────────────────────────────────────────────
static constexpr float kPad = 1e-5f;

// ── CPU-side node (same layout as BvhNode) ───────────────────────────────────
using CpuBvhNode = BvhNode;

// ── AABB helpers ─────────────────────────────────────────────────────────────

struct AABB {
    float lo[3] = { std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max() };
    float hi[3] = { -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max() };

    void expand(float x, float y, float z) noexcept {
        if (x < lo[0]) lo[0] = x;  if (x > hi[0]) hi[0] = x;
        if (y < lo[1]) lo[1] = y;  if (y > hi[1]) hi[1] = y;
        if (z < lo[2]) lo[2] = z;  if (z > hi[2]) hi[2] = z;
    }

    void expand(const AABB& o) noexcept {
        for (int a = 0; a < 3; ++a) {
            if (o.lo[a] < lo[a]) lo[a] = o.lo[a];
            if (o.hi[a] > hi[a]) hi[a] = o.hi[a];
        }
    }
};

static AABB tri_aabb(const StlTriangle& t) noexcept {
    AABB b;
    b.expand(t.v0[0], t.v0[1], t.v0[2]);
    b.expand(t.v1[0], t.v1[1], t.v1[2]);
    b.expand(t.v2[0], t.v2[1], t.v2[2]);
    return b;
}

static float tri_centroid(const StlTriangle& t, int axis) noexcept {
    return (t.v0[axis] + t.v1[axis] + t.v2[axis]) * (1.0f / 3.0f);
}

// ── Recursive BVH build ───────────────────────────────────────────────────────
//
// Appends nodes to `nodes`; returns the index of the newly created node.
// [lo, hi) is a half-open range into `indices`.

static int build_recursive(
    const std::vector<StlTriangle>& tris,
    std::vector<int>& indices,
    int lo, int hi,
    std::vector<CpuBvhNode>& nodes)
{
    assert(lo < hi);

    const int my_idx = static_cast<int>(nodes.size());
    nodes.push_back(CpuBvhNode{});

    if (hi - lo == 1) {
        const int tri_idx = indices[lo];
        const AABB b = tri_aabb(tris[tri_idx]);
        CpuBvhNode& n = nodes[my_idx];
        n.aabb_min[0] = b.lo[0] - kPad;
        n.aabb_min[1] = b.lo[1] - kPad;
        n.aabb_min[2] = b.lo[2] - kPad;
        n.aabb_max[0] = b.hi[0] + kPad;
        n.aabb_max[1] = b.hi[1] + kPad;
        n.aabb_max[2] = b.hi[2] + kPad;
        n.left  = ~tri_idx;   // encode: negative, recover with ~left
        n.right = -1;
        return my_idx;
    }

    AABB centroid_aabb;
    for (int i = lo; i < hi; ++i) {
        const int t = indices[i];
        centroid_aabb.expand(
            tri_centroid(tris[t], 0),
            tri_centroid(tris[t], 1),
            tri_centroid(tris[t], 2));
    }
    float span[3] = {
        centroid_aabb.hi[0] - centroid_aabb.lo[0],
        centroid_aabb.hi[1] - centroid_aabb.lo[1],
        centroid_aabb.hi[2] - centroid_aabb.lo[2]
    };
    int axis = 0;
    if (span[1] > span[0]) axis = 1;
    if (span[2] > span[axis]) axis = 2;

    const int mid = (lo + hi) / 2;
    std::nth_element(
        indices.begin() + lo,
        indices.begin() + mid,
        indices.begin() + hi,
        [&](int a, int b) {
            return tri_centroid(tris[a], axis) < tri_centroid(tris[b], axis);
        });

    const int left_idx  = build_recursive(tris, indices, lo,  mid, nodes);
    const int right_idx = build_recursive(tris, indices, mid, hi,  nodes);

    CpuBvhNode& n  = nodes[my_idx];
    const CpuBvhNode& lc = nodes[left_idx];
    const CpuBvhNode& rc = nodes[right_idx];
    for (int a = 0; a < 3; ++a) {
        n.aabb_min[a] = (lc.aabb_min[a] < rc.aabb_min[a]) ? lc.aabb_min[a] : rc.aabb_min[a];
        n.aabb_max[a] = (lc.aabb_max[a] > rc.aabb_max[a]) ? lc.aabb_max[a] : rc.aabb_max[a];
    }
    n.left  = left_idx;
    n.right = right_idx;

    return my_idx;
}

// ── GpuBvh destructor ─────────────────────────────────────────────────────────

GpuBvh::~GpuBvh() {
    if (d_nodes) { cudaFree(d_nodes); d_nodes = nullptr; }
    if (d_v0x)   { cudaFree(d_v0x);   d_v0x   = nullptr; }
    if (d_v0y)   { cudaFree(d_v0y);   d_v0y   = nullptr; }
    if (d_v0z)   { cudaFree(d_v0z);   d_v0z   = nullptr; }
    if (d_v1x)   { cudaFree(d_v1x);   d_v1x   = nullptr; }
    if (d_v1y)   { cudaFree(d_v1y);   d_v1y   = nullptr; }
    if (d_v1z)   { cudaFree(d_v1z);   d_v1z   = nullptr; }
    if (d_v2x)   { cudaFree(d_v2x);   d_v2x   = nullptr; }
    if (d_v2y)   { cudaFree(d_v2y);   d_v2y   = nullptr; }
    if (d_v2z)   { cudaFree(d_v2z);   d_v2z   = nullptr; }
    if (d_nx)    { cudaFree(d_nx);     d_nx    = nullptr; }
    if (d_ny)    { cudaFree(d_ny);     d_ny    = nullptr; }
    if (d_nz)    { cudaFree(d_nz);     d_nz    = nullptr; }
    n_nodes = 0;
    n_tris  = 0;
}

// ── GpuBvh::build ─────────────────────────────────────────────────────────────

void GpuBvh::build(const StlMesh& mesh) {
    const int nt = static_cast<int>(mesh.triangles.size());
    if (nt == 0) return;

    // Free any prior allocation so build() is safe to call more than once.
    this->~GpuBvh();
    new (this) GpuBvh();

    // ── 1. Build CPU BVH ──────────────────────────────────────────────────────
    std::vector<int> indices(nt);
    for (int i = 0; i < nt; ++i) indices[i] = i;

    std::vector<CpuBvhNode> cpu_nodes;
    cpu_nodes.reserve(2 * nt);
    build_recursive(mesh.triangles, indices, 0, nt, cpu_nodes);

    const int nn = static_cast<int>(cpu_nodes.size());

    // ── 2. Build SoA triangle arrays ─────────────────────────────────────────
    std::vector<float> h_v0x(nt), h_v0y(nt), h_v0z(nt);
    std::vector<float> h_v1x(nt), h_v1y(nt), h_v1z(nt);
    std::vector<float> h_v2x(nt), h_v2y(nt), h_v2z(nt);
    std::vector<float> h_nx(nt),  h_ny(nt),  h_nz(nt);

    for (int i = 0; i < nt; ++i) {
        const StlTriangle& t = mesh.triangles[i];
        h_v0x[i] = t.v0[0];  h_v0y[i] = t.v0[1];  h_v0z[i] = t.v0[2];
        h_v1x[i] = t.v1[0];  h_v1y[i] = t.v1[1];  h_v1z[i] = t.v1[2];
        h_v2x[i] = t.v2[0];  h_v2y[i] = t.v2[1];  h_v2z[i] = t.v2[2];

        // Use STL-stored normal; normalise defensively.
        float nx = t.normal[0], ny = t.normal[1], nz = t.normal[2];
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12f) { nx /= len; ny /= len; nz /= len; }
        h_nx[i] = nx;  h_ny[i] = ny;  h_nz[i] = nz;
    }

    // ── 3. Upload to device ───────────────────────────────────────────────────
    auto alloc_and_copy_f = [&](float** d, const std::vector<float>& h) {
        CUDA_CHECK(cudaMalloc(d, nt * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(*d, h.data(), nt * sizeof(float),
                              cudaMemcpyHostToDevice));
    };

    CUDA_CHECK(cudaMalloc(&d_nodes, nn * sizeof(BvhNode)));
    CUDA_CHECK(cudaMemcpy(d_nodes, cpu_nodes.data(), nn * sizeof(BvhNode),
                          cudaMemcpyHostToDevice));

    alloc_and_copy_f(&d_v0x, h_v0x);
    alloc_and_copy_f(&d_v0y, h_v0y);
    alloc_and_copy_f(&d_v0z, h_v0z);
    alloc_and_copy_f(&d_v1x, h_v1x);
    alloc_and_copy_f(&d_v1y, h_v1y);
    alloc_and_copy_f(&d_v1z, h_v1z);
    alloc_and_copy_f(&d_v2x, h_v2x);
    alloc_and_copy_f(&d_v2y, h_v2y);
    alloc_and_copy_f(&d_v2z, h_v2z);
    alloc_and_copy_f(&d_nx,  h_nx);
    alloc_and_copy_f(&d_ny,  h_ny);
    alloc_and_copy_f(&d_nz,  h_nz);

    n_nodes = nn;
    n_tris  = nt;
}

// ── Device: closest point on triangle (Ericson §5.1.5) ───────────────────────

__device__ static void closest_on_triangle(
    float ax, float ay, float az,   // vertex A
    float bx, float by, float bz,   // vertex B
    float cx, float cy, float cz,   // vertex C
    float px, float py, float pz,   // query point
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

    // Interior of triangle.
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    rx = ax + v*abx + w*acx;
    ry = ay + v*aby + w*acy;
    rz = az + v*abz + w*acz;
}

// ── Device: AABB squared distance ─────────────────────────────────────────────
// Returns 0 if point is inside the box.

__device__ __forceinline__ static float aabb_sq_dist(
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

// ── bvh_sdf ───────────────────────────────────────────────────────────────────

__device__ float bvh_sdf(
    const BvhNode* __restrict__ nodes, int /*n_nodes*/,
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

        // AABB early-out: skip if closest possible point on this AABB
        // is already farther than current best.
        const float ad2 = aabb_sq_dist(node.aabb_min, node.aabb_max, px, py, pz);
        if (ad2 >= best_dist2) continue;

        if (node.left < 0) {
            const int ti = ~node.left;
            float rx, ry, rz;
            closest_on_triangle(
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
            const float d2l = aabb_sq_dist(
                nodes[node.left].aabb_min, nodes[node.left].aabb_max, px, py, pz);
            const float d2r = aabb_sq_dist(
                nodes[node.right].aabb_min, nodes[node.right].aabb_max, px, py, pz);

            // Push farther first so closer is on top of stack.
            if (d2l <= d2r) {
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            } else {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }

    // Sign: dot(query - closest_pt, tri_normal) >= 0 => exterior (positive).
    const float dx = px - best_cx;
    const float dy = py - best_cy;
    const float dz = pz - best_cz;
    const float sign = (dx*tnx[best_tri] + dy*tny[best_tri] + dz*tnz[best_tri] >= 0.0f)
                       ? 1.0f : -1.0f;

    out_nx = tnx[best_tri];
    out_ny = tny[best_tri];
    out_nz = tnz[best_tri];

    return sign * sqrtf(best_dist2);
}
