# How to Add a New GPU Physics Subsystem (the "List" Pattern)

Every GPU physics subsystem in this codebase follows the same four-file protocol.
Copy this guide and substitute `Xxx` / `xxx` with your subsystem name.

---

## 1. Header — `include/cuda/gpu_xxx.cuh`

```cpp
#pragma once
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Per-leaf metadata uploaded to device once in build().
// Keep under 128 B; pad to a power-of-two size with static_assert.
struct GpuXxxLeafMeta {
    double* d_Q;          // base pointer into GpuPool (stride NCELL between vars)
    float   ox, oy, oz;   // block origin (copy from CellBlock)
    float   hx, hy, hz;   // cell sizes
    // ... subsystem-specific fields
    uint8_t _pad[N];      // pad to 64 or 128 bytes
};
static_assert(sizeof(GpuXxxLeafMeta) == 64, "");

struct GpuXxxList {
    GpuXxxLeafMeta* d_metas = nullptr;
    // add other device arrays here (e.g. double* d_scratch = nullptr)
    int n_leaves = 0;

    GpuXxxList() = default;
    GpuXxxList(const GpuXxxList&) = delete;
    GpuXxxList& operator=(const GpuXxxList&) = delete;
    ~GpuXxxList();

    void build(const BlockTree& tree, const GpuPool& pool /*, extra params */);
    void exec(cudaStream_t stream = nullptr) const;
};
```

**Rules:**
- One `d_metas` array of per-leaf structs is the minimum; add more device arrays only when the kernel needs data that doesn't fit in the meta struct.
- Never store host-side data after `build()` returns — everything the kernel needs must be on device.
- `exec()` is `const` — it only launches kernels, never reallocates.

---

## 2. Source — `src/cuda/gpu_xxx.cu`

```cpp
#include "cuda/gpu_xxx.cuh"
#include "cuda/gpu_constants.cuh"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// ── Kernel ────────────────────────────────────────────────────────────────────

__global__ void k_xxx(const GpuXxxLeafMeta* __restrict__ metas, int n_leaves)
{
    const int li = blockIdx.x;
    if (li >= n_leaves) return;
    const GpuXxxLeafMeta& m = metas[li];
    const int flat = threadIdx.x;  // one thread per cell (use 256 TPB → 7 blocks per leaf)
    if (flat >= GPU_NCELL) return;

    double* Q = m.d_Q;
    // read:  Q[v * GPU_NCELL + flat]
    // write: Q[v * GPU_NCELL + flat] = ...
}

// ── build() ───────────────────────────────────────────────────────────────────

void GpuXxxList::build(const BlockTree& tree, const GpuPool& pool)
{
    // Free previous allocation
    if (d_metas) { cudaFree(d_metas); d_metas = nullptr; }

    const auto& leaves = tree.local_leaves();
    n_leaves = static_cast<int>(leaves.size());
    if (n_leaves == 0) return;

    // Fill host-side meta array
    std::vector<GpuXxxLeafMeta> h_metas(n_leaves);
    for (int li = 0; li < n_leaves; ++li) {
        CellBlock* blk = leaves[li]->block.get();
        h_metas[li].d_Q = pool.d_ptr(blk);
        h_metas[li].ox  = (float)blk->ox;
        h_metas[li].oy  = (float)blk->oy;
        h_metas[li].oz  = (float)blk->oz;
        h_metas[li].hx  = (float)blk->h;
        h_metas[li].hy  = (float)blk->hy;
        h_metas[li].hz  = (float)blk->hz;
        // fill subsystem-specific fields
    }

    // Upload
    const size_t bytes = n_leaves * sizeof(GpuXxxLeafMeta);
    cudaMalloc(&d_metas, bytes);
    cudaMemcpy(d_metas, h_metas.data(), bytes, cudaMemcpyHostToDevice);
}

// ── exec() ────────────────────────────────────────────────────────────────────

void GpuXxxList::exec(cudaStream_t stream) const
{
    if (n_leaves == 0 || !d_metas) return;
    constexpr int TPB = 256;
    // Launch one block per leaf; TPB threads cover GPU_NCELL=1728 cells in 7 passes
    k_xxx<<<n_leaves, TPB, 0, stream>>>(d_metas, n_leaves);
}

// ── destructor ────────────────────────────────────────────────────────────────

GpuXxxList::~GpuXxxList()
{
    if (d_metas) { cudaFree(d_metas); d_metas = nullptr; }
    // cudaFree any other device arrays
}
```

**TPB / grid sizing guide:**
- `GPU_NCELL = 1728 = 12³` — use 256 threads → 7 iterations per thread (loop or split into blocks).
- For face-only kernels: `GPU_NB * GPU_NB = 64` threads, one block per leaf per face.
- For warp-level reductions: 32 threads, use cooperative groups `cg::tiled_partition<32>`.

---

## 3. Wire into GpuGraphSolver

### 3a. Add to `include/cuda/gpu_graph.cuh`

```cpp
#include "cuda/gpu_xxx.cuh"   // add to includes

struct GpuGraphSolver : IGpuSolver {
    // ... existing fields ...
    GpuXxxList  xxx_list_;
    bool        xxx_enabled_ = false;

    void set_gpu_xxx(/* params */) override {
        xxx_enabled_ = true;
        // store params into xxx_list_ fields
    }
```

### 3b. Add to `include/solver/ns_solver.hpp` (IGpuSolver interface)

```cpp
virtual void set_gpu_xxx(/* params */) {}   // default no-op
```

### 3c. Call `build()` in `GpuGraphSolver::build()` (`src/cuda/gpu_graph.cu`)

```cpp
if (xxx_enabled_)
    xxx_list_.build(tree, pool /*, params */);
```

### 3d. Call `exec()` in every advance path

There are **three** places in `gpu_graph.cu` that need `exec()`:

```cpp
// 1. _run_rk3_explicit — after ghost_list.exec(s), before rhs_list.exec(s)
if (xxx_enabled_) xxx_list_.exec(s);

// 2. _advance_amr — same position inside the stage loop
if (xxx_enabled_) xxx_list_.exec(stream);

// 3. Graph capture guard — if xxx changes pointers on regrid, block capture:
if (!mpi_halo_.active() && !acdi_enabled_ && !dyn_sgs_enabled_
    && !ibm_enabled_ && !xxx_enabled_)
    _capture_graphs();
```

If your subsystem is **operator-split** (runs after the full RK3 step, not per-stage), add it after the `cudaStreamSynchronize` in `advance()` instead.

---

## 4. CMakeLists.txt

```cmake
# Add source to _GPU_NS (search for the existing list around gpu_ibm.cu)
${_S}/cuda/gpu_xxx.cu

# Add gate target
add_nvcc_gate(TARGET tNN BIN tNN_xxx_gpu
    COMMENT "XxxPhysics gate"
    SRCS ${_T}/cuda/test_tNN_xxx_gpu.cu ${_GPU_NS})
```

---

## 5. Gate test skeleton — `tests/cuda/test_tNN_xxx_gpu.cu`

```cpp
// tNN — GpuXxxList: X1 <what>, X2 <what>
#include "cuda/gpu_xxx.cuh"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>

static bool nearly_eq(double a, double b, double tol)
{ return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

// ── X1: <description> ────────────────────────────────────────────────────────
static bool test_x1()
{
    // minimal single-block tree + pool setup
    BlockTree tree; tree.init(1.0, 1.0, 1.0); tree.refine_all(1);
    GpuPool pool; pool.alloc(tree);

    GpuXxxList list;
    list.build(tree, pool);
    list.exec(nullptr);  // nullptr = default stream
    cudaDeviceSynchronize();

    // download result and check
    // ...

    bool ok = nearly_eq(result, expected, 1e-10);
    printf("  X1: result=%.3e expected=%.3e  %s\n",
           result, expected, ok ? "PASS" : "FAIL");
    return ok;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main()
{
    bool all = true;
    all &= test_x1();
    // all &= test_x2();
    return all ? 0 : 1;
}
```

---

## Checklist before committing

- [ ] `static_assert(sizeof(GpuXxxLeafMeta) == N, "")` passes
- [ ] `build()` handles `n_leaves == 0` gracefully (early return)
- [ ] `exec()` guards `n_leaves == 0 || !d_metas`
- [ ] Destructor frees every `cudaMalloc`'d pointer
- [ ] `exec()` called in all three advance paths (or blocked by graph capture guard)
- [ ] Gate test passes: `cmake --build build -t tNN`
- [ ] `ba` CPU suite still passes: `cmake --build build -t ba`
- [ ] Commit message: `Xxx: <title>; tNN pass`
