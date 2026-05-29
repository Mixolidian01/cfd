# Multi-Root BlockTree (Forest of Octrees) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-root BlockTree with a forest of NX×NY×NZ independent octree roots so that the user can request multiple blocks per axis direction.

**Architecture:** Each of the NX×NY×NZ root cells lives at `nodes[0..NX*NY*NZ-1]` (level 0, flat index `iz*(NX*NY)+iy*NX+ix`). AMR refines within each subtree independently. `rebuild_neighbours()` links cross-root faces by climbing to the level-0 ancestor to identify which root a leaf belongs to, then computing the Morton-mirrored address in the adjacent root. Single-root backward-compatibility is preserved: `init(L)` and `init(Lx,Ly,Lz)` set `nx=ny=nz=1`, and the neighbour algorithm degenerates to the existing behaviour.

**Tech Stack:** C++20, existing BlockTree/CellBlock/NSSolver; new test in `tests/mesh/test_multi_root.cpp`.

---

## File Map

| File | Change |
|---|---|
| `include/mesh/block_tree.hpp` | Add `nx_roots_,ny_roots_,nz_roots_`; declare `init(Lx,Ly,Lz,NX,NY,NZ)` and `n_roots()` |
| `src/mesh/block_tree.cpp` | Implement new `init()`; rewrite `rebuild_neighbours()` |
| `include/solver/ns_solver.hpp` | Add `init(Lx,Ly,Lz,NX,NY,NZ,ic,phi_ic=nullptr)` declaration |
| `src/solver/ns_solver.cpp` | Implement the new `NSSolver::init()` overload |
| `tests/mesh/test_multi_root.cpp` | New test: geometry, cross-root links, C/F AMR, mass conservation |
| `CMakeLists.txt` | Register `test_multi_root`; add to `ba` target |

---

## Task 1: BlockTree header — root-grid fields and new `init` overload

**Files:**
- Modify: `include/mesh/block_tree.hpp`

- [ ] **Step 1.1: Add public API to BlockTree**

  Inside `struct BlockTree {` after the existing `init` declarations, add:

  ```cpp
  // Forest of octrees: NX×NY×NZ root blocks (cubic when Lx/NX==Ly/NY==Lz/NZ).
  // init(L) and init(Lx,Ly,Lz) remain single-root (NX=NY=NZ=1).
  void init(double Lx, double Ly, double Lz, int NX, int NY, int NZ);

  int n_roots()    const noexcept { return nx_roots_ * ny_roots_ * nz_roots_; }
  int nx_roots()   const noexcept { return nx_roots_; }
  int ny_roots()   const noexcept { return ny_roots_; }
  int nz_roots()   const noexcept { return nz_roots_; }
  ```

- [ ] **Step 1.2: Add private root-grid fields**

  Inside the `private:` section, after `domain_Lz_`:

  ```cpp
  int nx_roots_ = 1;
  int ny_roots_ = 1;
  int nz_roots_ = 1;
  ```

- [ ] **Step 1.3: Build**

  ```bash
  cmake --build /home/dkoffibi/dev/git_cfd/cfd/build --target block -- -j$(nproc) 2>&1 | tail -5
  ```

  Expected: compiles cleanly (no test yet).

---

## Task 2: Implement `init(Lx, Ly, Lz, NX, NY, NZ)` in block_tree.cpp

**Files:**
- Modify: `src/mesh/block_tree.cpp`

- [ ] **Step 2.1: Write the failing test (geometry only)**

  In `tests/mesh/test_multi_root.cpp` (create file):

  ```cpp
  #include "mesh/block_tree.hpp"
  #include <cassert>
  #include <cmath>
  #include <cstdio>

  static int g_pass = 0, g_fail = 0;
  #define CHECK(cond) do { if (cond) { ++g_pass; } else { ++g_fail; \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while(0)

  // T1: 2×1×1 root grid — cell sizes and origins
  static void test_geometry() {
      BlockTree tree;
      tree.init(2.0, 1.0, 1.0, 2, 1, 1);
      CHECK(tree.n_roots() == 2);
      // Root 0 is at ix=0,iy=0,iz=0 — covers [0,1]×[0,1]×[0,1]
      // Root 1 is at ix=1,iy=0,iz=0 — covers [1,2]×[0,1]×[0,1]
      const auto& r0 = tree.nodes[0];
      const auto& r1 = tree.nodes[1];
      CHECK(r0.level == 0 && r1.level == 0);
      CHECK(std::fabs(r0.ox - 0.0) < 1e-12);
      CHECK(std::fabs(r1.ox - 1.0) < 1e-12);
      CHECK(std::fabs(r0.block->h  - 1.0/8) < 1e-12);
      CHECK(std::fabs(r1.block->h  - 1.0/8) < 1e-12);
      CHECK(std::fabs(r0.block->hy - 1.0/8) < 1e-12);
      CHECK(std::fabs(r0.block->hz - 1.0/8) < 1e-12);
  }

  // T2: 2×1×1 — cross-root neighbor links at X interface
  static void test_cross_root_neighbors() {
      BlockTree tree;
      tree.init(2.0, 1.0, 1.0, 2, 1, 1);
      tree.rebuild_neighbours();
      // Root 0 XPLUS neighbor is Root 1
      CHECK(tree.nodes[0].neighbours[XPLUS]  == 1);
      // Root 1 XMINUS neighbor is Root 0
      CHECK(tree.nodes[1].neighbours[XMINUS] == 0);
      // All other domain-boundary directions should be -1 (no periodicity set)
      CHECK(tree.nodes[0].neighbours[XMINUS] == -1);
      CHECK(tree.nodes[1].neighbours[XPLUS]  == -1);
  }

  // T3: 2×1×1 periodic — cross-root wrapping
  static void test_cross_root_periodic() {
      BlockTree tree;
      tree.init(2.0, 1.0, 1.0, 2, 1, 1);
      tree.set_periodic(true);
      tree.rebuild_neighbours();
      // Root 0 XMINUS wraps to Root 1
      CHECK(tree.nodes[0].neighbours[XMINUS] == 1);
      // Root 1 XPLUS wraps to Root 0
      CHECK(tree.nodes[1].neighbours[XPLUS]  == 0);
  }

  // T4: refine root 0 — level-1 children link to unrefined root 1 (C/F)
  static void test_cf_cross_root() {
      BlockTree tree;
      tree.init(2.0, 1.0, 1.0, 2, 1, 1);
      tree.refine(0);   // root 0 → 8 level-1 children
      // After refine, root 1 (level 0) and root 0's children at XPLUS face
      // should be linked as a C/F interface.
      // The level-1 children with ix-high octant (oct_ix==1) are at XPLUS boundary
      // of root 0; their XPLUS neighbor should be root 1 (which is coarser).
      const int fc = tree.nodes[0].first_child;
      bool found_cf = false;
      for (int oct = 0; oct < 8; ++oct) {
          if (oct_ix(oct) == 1) {  // child at X-high boundary of root 0
              int ni = tree.nodes[fc + oct].neighbours[XPLUS];
              if (ni == 1) found_cf = true;  // links to root 1 (level 0)
          }
      }
      CHECK(found_cf);
  }

  int main() {
      test_geometry();
      test_cross_root_neighbors();
      test_cross_root_periodic();
      test_cf_cross_root();
      std::printf("%d passed, %d failed\n", g_pass, g_fail);
      return g_fail ? 1 : 0;
  }
  ```

- [ ] **Step 2.2: Register test in CMakeLists.txt**

  After the `test_rect_domain` block (around line 642), add:

  ```cmake
  add_executable(test_multi_root tests/mesh/test_multi_root.cpp)
  target_link_libraries(test_multi_root PRIVATE ns_solver m)
  add_test(NAME multi_root COMMAND test_multi_root)
  add_custom_target(t_multi_root
      DEPENDS test_multi_root
      COMMAND $<TARGET_FILE:test_multi_root>
      COMMENT "Forest of octrees: multi-root BlockTree"
      USES_TERMINAL
  )
  ```

  Also add `test_multi_root` to both the `ba` DEPENDS list and the final DEPENDS list above it:

  Find:
  ```cmake
  DEPENDS simulate test_linalg ... test_rect_domain test_t42_rect_ns
  ```
  Append `test_multi_root` to both DEPENDS lines.

- [ ] **Step 2.3: Verify the test fails to compile (init overload not yet implemented)**

  ```bash
  cmake --build /home/dkoffibi/dev/git_cfd/cfd/build --target test_multi_root 2>&1 | tail -10
  ```

  Expected: linker error or compile error (undefined `init(Lx,Ly,Lz,NX,NY,NZ)`).

- [ ] **Step 2.4: Implement `init(Lx, Ly, Lz, NX, NY, NZ)` in block_tree.cpp**

  After the existing `void BlockTree::init(double Lx, double Ly, double Lz)` (around line 226), add:

  ```cpp
  void BlockTree::init(double Lx, double Ly, double Lz, int NX, int NY, int NZ) {
      assert(NX >= 1 && NY >= 1 && NZ >= 1);
      domain_L_  = Lx;
      domain_Ly_ = Ly;
      domain_Lz_ = Lz;
      nx_roots_ = NX;
      ny_roots_ = NY;
      nz_roots_ = NZ;
      nodes.clear();
      free_list_.clear();
      leaf_dirty_ = true;

      const double cell_hx = Lx / (NX * NB);
      const double cell_hy = Ly / (NY * NB);
      const double cell_hz = Lz / (NZ * NB);
      const double root_Lx = Lx / NX;   // span of one root in X
      const double root_Ly = Ly / NY;
      const double root_Lz = Lz / NZ;

      // Pre-allocate all NX*NY*NZ root nodes contiguously.
      nodes.resize(NX * NY * NZ);

      for (int iz = 0; iz < NZ; ++iz)
      for (int iy = 0; iy < NY; ++iy)
      for (int ix = 0; ix < NX; ++ix) {
          int ridx = iz * (NX * NY) + iy * NX + ix;
          auto& nd  = nodes[ridx];
          nd.reset();
          nd.parent      = -1;   // root: no parent
          nd.first_child = -1;   // leaf
          nd.level       = 0;
          nd.morton      = 0;    // each subtree root has morton=0 at level 0
          nd.ox = ix * root_Lx;
          nd.oy = iy * root_Ly;
          nd.oz = iz * root_Lz;
          nd.block = std::make_unique<CellBlock>(nd.ox, nd.oy, nd.oz,
                                                 cell_hx, cell_hy, cell_hz);
      }
      rebuild_neighbours();
  }
  ```

  Also update the existing two-argument `init` overloads to reset the root-grid fields:

  In `void BlockTree::init(double Lx, double Ly, double Lz)` (existing, ~line 226):
  ```cpp
  void BlockTree::init(double Lx, double Ly, double Lz) {
      nx_roots_ = 1;
      ny_roots_ = 1;
      nz_roots_ = 1;
      // ... rest unchanged
  ```

  In `void BlockTree::init(double L)` (existing, ~line 244):
  ```cpp
  void BlockTree::init(double L) {
      init(L, L, L);   // cubic shorthand — nx_roots_=ny_roots_=nz_roots_=1 set inside
  }
  ```

- [ ] **Step 2.5: Build and run T1 (geometry)**

  ```bash
  cmake --build /home/dkoffibi/dev/git_cfd/cfd/build --target test_multi_root -- -j$(nproc) 2>&1 | tail -5 && /home/dkoffibi/dev/git_cfd/cfd/build/test_multi_root
  ```

  Expected: T1 (test_geometry) passes; T2/T3/T4 may fail (rebuild_neighbours not yet updated).

---

## Task 3: Update `rebuild_neighbours()` for cross-root links

**Files:**
- Modify: `src/mesh/block_tree.cpp`

This is the core change. The algorithm adds cross-root adjacency to the existing Morton-code lookup.

**Design:**
- Each leaf's root is found by climbing parent links to level 0 (O(depth) per leaf, called once per `rebuild_neighbours()`).
- Map key: `uint64_t key = ((uint64_t)root_id << 35) | ((uint64_t)lev << 30) | (uint64_t)morton` — safe up to 32 roots (5 bits), level ≤ 31 (5 bits), morton ≤ 30 bits.
- UINT32_MAX case: replaced with cross-root lookup that also handles single-root periodic wrapping correctly.

- [ ] **Step 3.1: Write the updated `rebuild_neighbours()` — replace the existing function in block_tree.cpp**

  Replace the entire `void BlockTree::rebuild_neighbours()` function (from `// ===rebuild` header to closing `}`):

  ```cpp
  // =============================================================================
  // rebuild_neighbours
  // =============================================================================
  void BlockTree::rebuild_neighbours() {
      for (auto& nd : nodes) nd.neighbours.fill(-1);

      const auto& leaves = leaf_indices();
      if (leaves.empty()) return;

      // For each leaf, find which level-0 root it belongs to by climbing parent links.
      // root_of[i] = index in nodes[] of the level-0 ancestor of leaves[i].
      auto get_root_id = [&](int li) -> int {
          int cur = li;
          while (nodes[cur].level > 0) cur = nodes[cur].parent;
          return cur;  // root index is in 0..nx_roots_*ny_roots_*nz_roots_-1
      };

      // Build map: (root_id, level, morton) -> node_idx
      // Key: bits[63:35]=root_id (up to 2^29 roots), bits[34:30]=level, bits[29:0]=morton
      // For any realistic forest (< 1024 roots) the 29-bit root field is sufficient.
      std::unordered_map<uint64_t, int> lm_map;
      lm_map.reserve(leaves.size() * 2);
      for (int li : leaves) {
          int   root_id = get_root_id(li);
          auto& nd      = nodes[li];
          uint64_t key  = ((uint64_t)root_id << 35)
                        | ((uint64_t)nd.level << 30)
                        | (uint64_t)nd.morton;
          lm_map[key] = li;
      }

      // Compute the face-neighbour Morton code within the same root, or UINT32_MAX
      // if the face exits the root boundary.
      auto morton_face_neighbour = [](uint32_t code, int level,
                                      int axis, int delta) -> uint32_t {
          uint32_t mx, my, mz;
          morton_decode(code, mx, my, mz);
          uint32_t max_coord = (1u << level) - 1u;
          uint32_t& mc = (axis == 0) ? mx : (axis == 1) ? my : mz;
          if (delta > 0) { if (mc == max_coord) return UINT32_MAX; mc++; }
          else           { if (mc == 0)          return UINT32_MAX; mc--; }
          return morton_encode(mx, my, mz);
      };

      // Given a root_id, return (rx, ry, rz) in the root grid.
      auto root_to_grid = [&](int rid, int& rx, int& ry, int& rz) {
          rx = rid % nx_roots_;
          ry = (rid / nx_roots_) % ny_roots_;
          rz = rid / (nx_roots_ * ny_roots_);
      };

      // Compute the cross-root Morton code: mirror the boundary axis to the far end.
      auto cross_morton = [](uint32_t code, int level, int axis, int delta) -> uint32_t {
          uint32_t mx, my, mz;
          morton_decode(code, mx, my, mz);
          uint32_t max_coord = (1u << level) - 1u;
          uint32_t& mc = (axis == 0) ? mx : (axis == 1) ? my : mz;
          mc = (delta > 0) ? 0u : max_coord;
          return morton_encode(mx, my, mz);
      };

      static constexpr int face_axis[NFACES]  = { 0, 0, 1, 1, 2, 2 };
      static constexpr int face_delta[NFACES] = {-1,+1,-1,+1,-1,+1 };

      for (int ai : leaves) {
          auto& a   = nodes[ai];
          int   lev = a.level;
          int   root_a = get_root_id(ai);

          for (int d = 0; d < NFACES; ++d) {
              if (a.neighbours[d] >= 0) continue;
              int axis  = face_axis[d];
              int delta = face_delta[d];

              uint32_t nb_code = morton_face_neighbour(a.morton, lev, axis, delta);

              if (nb_code != UINT32_MAX) {
                  // ── Same-root intra-tree lookup ──────────────────────────────
                  uint64_t key = ((uint64_t)root_a << 35) | ((uint64_t)lev << 30) | nb_code;
                  auto it = lm_map.find(key);
                  if (it != lm_map.end()) {
                      int bi = it->second;
                      a.neighbours[d]             = bi;
                      nodes[bi].neighbours[d ^ 1] = ai;
                      continue;
                  }
                  // Coarser same-root neighbour (C/F interface)
                  if (lev > 0) {
                      uint64_t key2 = ((uint64_t)root_a << 35)
                                    | ((uint64_t)(lev - 1) << 30)
                                    | (uint64_t)(nb_code >> 3);
                      auto it2 = lm_map.find(key2);
                      if (it2 != lm_map.end()) {
                          int bi = it2->second;
                          a.neighbours[d]             = bi;
                          nodes[bi].neighbours[d ^ 1] = ai;
                          continue;
                      }
                  }
              } else {
                  // ── At root boundary — cross-root or domain edge ────────────
                  int rx, ry, rz;
                  root_to_grid(root_a, rx, ry, rz);

                  int rx2 = rx + (axis == 0 ? delta : 0);
                  int ry2 = ry + (axis == 1 ? delta : 0);
                  int rz2 = rz + (axis == 2 ? delta : 0);

                  // Periodic wrapping of the root grid
                  if (rx2 < 0 || rx2 >= nx_roots_) {
                      if (!periodic_axis_[0]) continue;
                      rx2 = (rx2 + nx_roots_) % nx_roots_;
                  }
                  if (ry2 < 0 || ry2 >= ny_roots_) {
                      if (!periodic_axis_[1]) continue;
                      ry2 = (ry2 + ny_roots_) % ny_roots_;
                  }
                  if (rz2 < 0 || rz2 >= nz_roots_) {
                      if (!periodic_axis_[2]) continue;
                      rz2 = (rz2 + nz_roots_) % nz_roots_;
                  }

                  int root_b = rz2 * (nx_roots_ * ny_roots_) + ry2 * nx_roots_ + rx2;

                  // Self-link check: single-root periodic (NX=NY=NZ=1) with lev==0
                  // would wrap root to itself — skip to avoid root self-loop.
                  if (root_b == root_a && lev == 0) continue;

                  uint32_t xcode = cross_morton(a.morton, lev, axis, delta);

                  // Same-level in adjacent root
                  uint64_t key = ((uint64_t)root_b << 35) | ((uint64_t)lev << 30) | xcode;
                  auto it = lm_map.find(key);
                  if (it != lm_map.end()) {
                      int bi = it->second;
                      a.neighbours[d]             = bi;
                      nodes[bi].neighbours[d ^ 1] = ai;
                      continue;
                  }
                  // Coarser in adjacent root (C/F at root boundary)
                  if (lev > 0) {
                      uint64_t key2 = ((uint64_t)root_b << 35)
                                    | ((uint64_t)(lev - 1) << 30)
                                    | (uint64_t)(xcode >> 3);
                      auto it2 = lm_map.find(key2);
                      if (it2 != lm_map.end()) {
                          int bi = it2->second;
                          a.neighbours[d]             = bi;
                          nodes[bi].neighbours[d ^ 1] = ai;
                          continue;
                      }
                  }
              }
          }
      }
      invalidate_leaf_cache();
      leaf_dirty_ = false;
  }
  ```

- [ ] **Step 3.2: Build**

  ```bash
  cmake --build /home/dkoffibi/dev/git_cfd/cfd/build --target test_multi_root -- -j$(nproc) 2>&1 | tail -5
  ```

  Expected: clean compile.

- [ ] **Step 3.3: Run test_multi_root — T1 through T4 all pass**

  ```bash
  /home/dkoffibi/dev/git_cfd/cfd/build/test_multi_root
  ```

  Expected: `N passed, 0 failed` (all T1–T4 pass).

  Debug guidance if failures:
  - T2 fail: check `rebuild_neighbours()` is called after `init()`; confirm cross-root key computation; add `std::fprintf` tracing.
  - T3 fail: check `periodic_axis_` handling; `set_periodic(true)` must be called before `rebuild_neighbours()`.
  - T4 fail: check the `xcode >> 3` coarser fallback for level-1 → level-0 root cross-root C/F link.

---

## Task 4: Mass conservation test with multi-root

**Files:**
- Modify: `tests/mesh/test_multi_root.cpp`

- [ ] **Step 4.1: Add T5 mass conservation test to test_multi_root.cpp**

  Add after `test_cf_cross_root()`:

  ```cpp
  #include "solver/ns_solver.hpp"

  // T5: 10-step advance on 2×1×1 forest (2 roots) — mass conserved to 1e-10
  static void test_mass_conservation() {
      auto ic = [](double x, double /*y*/, double /*z*/) -> Prim {
          // Smooth density wave spanning both roots
          Prim q{};
          q.rho = 1.0 + 0.1 * std::sin(2.0 * M_PI * x / 2.0);
          q.u   = 0.1;
          q.v   = 0.0;
          q.w   = 0.0;
          q.p   = 1.0;
          return q;
      };
      NSSolver solver;
      solver.init(2.0, 1.0, 1.0, 2, 1, 1, ic);
      const double m0 = solver.compute_diag().mass;
      for (int step = 0; step < 10; ++step) solver.advance();
      const double m1 = solver.compute_diag().mass;
      const double err = std::fabs(m1 - m0) / (m0 + 1e-300);
      if (err > 1e-10)
          std::fprintf(stderr, "mass error = %.3e\n", err);
      CHECK(err < 1e-10);
  }
  ```

  And add `test_mass_conservation();` to `main()`.

- [ ] **Step 4.2: Build and run**

  ```bash
  cmake --build /home/dkoffibi/dev/git_cfd/cfd/build --target test_multi_root -- -j$(nproc) && /home/dkoffibi/dev/git_cfd/cfd/build/test_multi_root
  ```

  Expected: all 5 sub-tests pass (T5 mass error < 1e-10).

  Debug guidance if T5 fails:
  - Print `m0, m1` to identify direction of error.
  - If mass leaks at cross-root face: the `accumulate_fine_flux` / `apply_flux_correction` path needs to handle cross-root flux registers. Check that `nodes[ni].flux_reg[d^1]` is populated correctly by verifying the `ni` neighbor is valid at the C/F cross-root face.

---

## Task 5: NSSolver overload + ba wiring

**Files:**
- Modify: `include/solver/ns_solver.hpp`
- Modify: `src/solver/ns_solver.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 5.1: Declare `NSSolver::init(Lx,Ly,Lz,NX,NY,NZ,...)` in ns_solver.hpp**

  After the existing two-parameter init declarations (around line 227 in `include/solver/ns_solver.hpp`), add:

  ```cpp
  // Forest of octrees: NX×NY×NZ root blocks on [0,Lx]×[0,Ly]×[0,Lz].
  void init(double Lx, double Ly, double Lz, int NX, int NY, int NZ,
            const std::function<Prim(double,double,double)>& ic,
            const std::function<double(double,double,double)>* phi_ic = nullptr);
  ```

- [ ] **Step 5.2: Implement in ns_solver.cpp**

  In `src/solver/ns_solver.cpp`, find where `void NSSolver::init(double Lx, double Ly, double Lz, ...)` ends and add the new overload immediately after:

  ```cpp
  void NSSolver::init(double Lx, double Ly, double Lz, int NX, int NY, int NZ,
                      const std::function<Prim(double,double,double)>& ic,
                      const std::function<double(double,double,double)>* phi_ic) {
      // Set periodic flags before init so rebuild_neighbours wraps correctly.
      for (int ax = 0; ax < 3; ++ax)
          tree.periodic_axis_[ax] = periodic_axis_[ax];
      tree.init(Lx, Ly, Lz, NX, NY, NZ);
      _apply_ic(ic, phi_ic);
  }
  ```

  Note: `_apply_ic` is the internal helper used by the existing rect `init` overload. Look up its exact name by searching:
  ```bash
  grep -n "_apply_ic\|apply_ic\|init_ic\|set_ic" /home/dkoffibi/dev/git_cfd/cfd/src/solver/ns_solver.cpp | head -10
  ```
  Use whatever internal IC application pattern the existing `init(Lx,Ly,Lz,ic)` uses — copy it verbatim.

- [ ] **Step 5.3: Add `test_multi_root` to `ba` target in CMakeLists.txt**

  In `CMakeLists.txt`, find the `ba` target DEPENDS (around line 630–636):
  ```cmake
  add_custom_target(ba
      DEPENDS test_linalg ... test_rect_domain test_t42_rect_ns
  ```
  Append `test_multi_root` to both DEPENDS lists (the simulate target line and the ba target line).

- [ ] **Step 5.4: Full ba suite — verify no regressions**

  ```bash
  cmake --build /home/dkoffibi/dev/git_cfd/cfd/build -t ba -- -j$(nproc) 2>&1 | tail -20
  ```

  Expected: all previously passing tests still pass; `test_multi_root` now also passes.
  Pre-existing failures (bench_b8) are acceptable.

- [ ] **Step 5.5: Commit**

  ```bash
  cd /home/dkoffibi/dev/git_cfd/cfd
  git add include/mesh/block_tree.hpp src/mesh/block_tree.cpp \
          include/solver/ns_solver.hpp src/solver/ns_solver.cpp \
          tests/mesh/test_multi_root.cpp CMakeLists.txt
  git commit -m "$(cat <<'EOF'
  feat: forest-of-octrees — BlockTree::init(Lx,Ly,Lz,NX,NY,NZ)

  Add multi-root BlockTree support: NX×NY×NZ independent octree subtrees
  sharing a flat node pool.  rebuild_neighbours() extended with cross-root
  Morton-key lookup (same level + one-level-coarser C/F fallback); periodic
  wrapping applied at root-grid boundaries.  Backward-compatible: single-root
  init(L) and init(Lx,Ly,Lz) unchanged.  Gate: test_multi_root all pass,
  ba suite unchanged.

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Multiple root blocks per axis direction | Task 2: `init(Lx,Ly,Lz,NX,NY,NZ)` |
| Cross-root neighbor links | Task 3: updated `rebuild_neighbours()` |
| AMR refine across root boundary (C/F) | Task 3: coarser fallback + T4 test |
| Periodic wrapping across root grid | Task 3: root-grid wrapping + T3 test |
| Mass conservation | Task 4: T5 test |
| NSSolver API | Task 5 |
| ba gate | Task 5, Step 5.4 |

### Potential issues

1. **`_apply_ic` name**: Task 5.2 requires verifying the internal IC-application function name in `ns_solver.cpp`. The plan says "copy verbatim from the existing `init(Lx,Ly,Lz,ic)` pattern" to avoid guessing.

2. **`set_child_geometry()` with multi-root**: When `refine()` is called on a root in the forest, `set_child_geometry()` uses `domain_L_/(NB*(1<<(level+1)))` for the fallback cell size when `par.block` is null. For multi-root, the per-root cell size is `Lx/(NX*NB)` not `Lx/(NB)`. However, this fallback is only used when `par.block == nullptr`, which only happens for internal (already-refined) nodes. For a freshly-init'd forest, `par.block` is always non-null for level-0 roots, so `set_child_geometry()` reads `par.block->h * 0.5` — correct. The fallback is only reached for internal nodes at level > 0, where the cell size has already been halved at each refinement — also correct.

3. **64-bit key overflow**: With `nx_roots_*ny_roots_*nz_roots_ <= 512` (e.g. 8×8×8 forest), the root_id fits in 9 bits. Key bits: 29(root) + 5(level) + 30(morton) = 64 bits — exactly 64. The current split is 29+5+30. Any forest up to ~500M roots is fine; practical limit is probably 16×16×16=4096 roots (12 bits), leaving 17 bits of headroom.
