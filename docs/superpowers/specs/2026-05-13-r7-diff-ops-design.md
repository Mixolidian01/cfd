# R7 — Differential Operator Abstraction

**Date:** 2026-05-13
**Branch:** `to_refactor`
**Prerequisite:** R0–R5 complete and stable (all gates green except pre-existing BN04).

---

## 1. Motivation

Three functions in `operators.cpp` — `fill_ducros_cache`, `phi_compression_rhs`, and
`viscous_rhs_impl` — each hardcode 2nd-order central-difference stencils inline with
no shared abstraction. This creates three problems:

1. **Duplication**: the same `(f_{i+1} − f_{i−1}) / (2h)` pattern appears in all three
   functions; an off-by-one error or sign bug must be fixed in every copy independently.
2. **Untestability**: the stencil arithmetic is buried inside large loops and cannot be
   exercised in isolation.
3. **Order lock-in**: upgrading `viscous_rhs_impl` from O(h²) to O(h⁴) viscosity requires
   touching ~80 lines of index arithmetic with no mechanical guarantee of correctness.

R7 resolves all three by introducing a two-level functor abstraction in Layer P and a
corresponding concept in Layer C.

---

## 2. Goals

- Extract scalar and tensor differential operators into named, testable functors.
- Make stencil order a compile-time template parameter (`Order=2` default, `Order=4`
  available without API change).
- Guarantee GPU safety via `is_trivially_copyable` constraint on every functor.
- Leave `gpu_rhs.cu` untouched — GPU migration is a separate future step.
- All existing gates (t1–t26, minus pre-existing BN04) must remain green.

---

## 3. Architecture

R7 is purely additive to Layer P and Layer C. Layer E and the solver are untouched.

```
LAYER P — include/physics/diff_ops.hpp        NEW (header-only)
           CellGrad<Axis DIR, int Order=2>
           FaceGrad<Axis DIR, int Order=2>
           CellLaplacian<int Order=2>
           CellDiv<int Order=2>
           VelocityGradComponents             (plain aggregate)
           VelocityGradAtFace<Axis DIR, int Order=2>

LAYER C — include/concepts.hpp               ADD three concepts
           ScalarCellOperator
           ScalarFaceOperator
           TensorFaceOperator

MODIFIED — src/operators.cpp
           fill_ducros_cache                 → uses CellGrad<DIR, 2>
           phi_compression_rhs               → uses CellGrad + CellDiv
           viscous_rhs_impl                  → uses VelocityGradAtFace<DIR, 2>

TESTS    — tests/test_operators.cpp          ADD unit + regression tests
```

No new `.cpp` files. No new CMake targets. No changes to the CMakeLists.

---

## 4. Scalar Primitives (`include/physics/diff_ops.hpp`)

All functors are empty structs. All methods are `inline`, `noexcept`,
`__host__ __device__`. The `Field` parameter is any callable `double(int,int,int)`.

### 4.1 `CellGrad<Axis DIR, int Order = 2>`

Cell-centred `∂f/∂x_DIR` at `(i,j,k)`.

```
Order=2: (f_{i+1} − f_{i−1}) / (2h)
Order=4: (−f_{i+2} + 8f_{i+1} − 8f_{i−1} + f_{i−2}) / (12h)
```

Ghost layer requirement: `NG≥1` for Order=2; `NG≥2` for Order=4. Both satisfied by
the current `NG=2`.

Used by: `fill_ducros_cache` (all 9 velocity gradient components),
`phi_compression_rhs` pass 1 (gradient of φ).

### 4.2 `FaceGrad<Axis DIR, int Order = 2>`

Scalar gradient at the face between cell `(i,j,k)` and `(i+1,j,k)` (for `DIR=X`).
Two methods:

```cpp
double normal(Field f, int i, int j, int k, double h);
// ∂f/∂x_DIR at the face

template<Axis T>  // T ≠ DIR
double tangential(Field f, int i, int j, int k, double h);
// ∂f/∂x_T at the face — averaged from both sides
```

Stencils (DIR=X, face at i+½):

```
normal Order=2:     (f(i+1,j,k) − f(i,j,k)) / h
normal Order=4:     (−f(i+2,j,k) + 27f(i+1,j,k) − 27f(i,j,k) + f(i−1,j,k)) / (24h)

tangential<Y> O=2:  [(f(i+1,j+1,k)−f(i+1,j−1,k)) + (f(i,j+1,k)−f(i,j−1,k))] / (4h)
tangential<Y> O=4:  [4th-order 4-point average along Y from both face sides] / (12h)
```

The Order=4 normal stencil uses `f(i−1,j,k)` — one ghost cell on the upstream side.
With `NG=2` this is always in bounds for all interior faces.

Used by: `VelocityGradAtFace` (see §4.5).

### 4.3 `CellLaplacian<int Order = 2>`

`∇²f` at `(i,j,k)`, summed over all three axes.

```
Order=2: Σ_dir (f_{+1} − 2f_0 + f_{−1}) / h²
```

Not used by any current function. Provided as a building block for future implicit
diffusion or hyperviscosity operators.

### 4.4 `CellDiv<int Order = 2>`

Conservative `∇·F` from cell-centred scalar flux arrays `Fx`, `Fy`, `Fz`.
The accessor takes a flat cell index.

```
Order=2: (Fx(i+1,j,k)−Fx(i−1,j,k))/(2h)
        +(Fy(i,j+1,k)−Fy(i,j−1,k))/(2h)
        +(Fz(i,j,k+1)−Fz(i,j,k−1))/(2h)
```

Used by: `phi_compression_rhs` pass 2.

### 4.5 `VelocityGradComponents` (aggregate)

Plain struct returned by `VelocityGradAtFace`. Fields are named relative to face
orientation (normal/tangential) so the same struct is returned for all three axes.

```cpp
struct VelocityGradComponents {
    double dun_dxn;    // ∂u_n/∂x_n   (τ_nn diagonal)
    double dut1_dxn;   // ∂u_t1/∂x_n
    double dut2_dxn;   // ∂u_t2/∂x_n
    double dun_dxt1;   // ∂u_n/∂x_t1
    double dun_dxt2;   // ∂u_n/∂x_t2
    double dut1_dxt1;  // ∂u_t1/∂x_t1  (contributes to div u)
    double dut2_dxt2;  // ∂u_t2/∂x_t2  (contributes to div u)

    double divu() const noexcept { return dun_dxn + dut1_dxt1 + dut2_dxt2; }
};
```

These seven components cover exactly what `viscous_rhs_impl` requires per face.

### 4.6 `VelocityGradAtFace<Axis DIR, int Order = 2>`

Composite functor. Calls `FaceGrad<DIR, Order>` for each of the seven components
and returns `VelocityGradComponents`. Two methods:

```cpp
template<typename UF, typename VF, typename WF>
VelocityGradComponents plus (UF u, VF v, WF w, int i, int j, int k, double h);
// face at (i+½, j, k) for DIR=X

template<typename UF, typename VF, typename WF>
VelocityGradComponents minus(UF u, VF v, WF w, int i, int j, int k, double h);
// face at (i−½, j, k) for DIR=X  — shifts index by −1 along DIR, then calls plus
```

`minus` is implemented by shifting the index one step in the `−DIR` direction and
delegating to the same `FaceGrad` calls as `plus`. Stencil logic lives in exactly
one place.

---

## 5. Layer C Concepts (`include/concepts.hpp`)

A function-pointer probe type avoids requiring a concrete `Field` in the concept:

```cpp
using _FieldProbe = double(*)(int,int,int);

template<typename Op>
concept ScalarCellOperator = requires(Op op, _FieldProbe f,
                                      int i, int j, int k, double h) {
    { op(f, i, j, k, h) } -> std::convertible_to<double>;
    requires std::is_trivially_copyable_v<Op>;
};

template<typename Op>
concept ScalarFaceOperator = requires(Op op, _FieldProbe f,
                                      int i, int j, int k, double h) {
    { op.normal(f, i, j, k, h) } -> std::convertible_to<double>;
    requires std::is_trivially_copyable_v<Op>;
};

template<typename Op>
concept TensorFaceOperator = requires(Op op,
                                       _FieldProbe u, _FieldProbe v, _FieldProbe w,
                                       int i, int j, int k, double h) {
    { op.plus (u, v, w, i, j, k, h) } -> std::same_as<VelocityGradComponents>;
    { op.minus(u, v, w, i, j, k, h) } -> std::same_as<VelocityGradComponents>;
    requires std::is_trivially_copyable_v<Op>;
};
```

`is_trivially_copyable` is the GPU-safety invariant. All current functors satisfy it
automatically (empty structs with no user-defined constructors).

`static_assert` checks added at the top of `operators.cpp`:

```cpp
static_assert(ScalarCellOperator<CellGrad<Axis::X>>);
static_assert(ScalarFaceOperator<FaceGrad<Axis::X>>);
static_assert(TensorFaceOperator<VelocityGradAtFace<Axis::X>>);
```

---

## 6. Refactoring Sites (`src/operators.cpp`)

No behavioral change. Only index arithmetic is replaced with functor calls.

### `fill_ducros_cache`

The 9-line block of `ih2*(pc[i±1]−pc[i∓1])` expressions is replaced with 9 symmetric
`CellGrad<DIR, 2>` calls. The `eps_duc` guard and Ducros sensor formula are unchanged.

### `phi_compression_rhs`

- Pass 1: three inline gradient expressions replaced with `CellGrad<X,2>`,
  `CellGrad<Y,2>`, `CellGrad<Z,2>` calls.
- Pass 2: the three-term `divF` expression replaced with a single `CellDiv<2>` call.
- The two-pass structure is preserved — pass 1 fills `Fx/Fy/Fz`, pass 2 differentiates
  them. This ordering is load-bearing for correctness.

### `viscous_rhs_impl`

The ~80-line block of `dudx_xp`, `dvdx_xp`, … variables per cell is replaced with
six `VelocityGradAtFace` calls:

```cpp
VelocityGradAtFace<Axis::X, 2> VGX;
VelocityGradAtFace<Axis::Y, 2> VGY;
VelocityGradAtFace<Axis::Z, 2> VGZ;

auto gxp = VGX.plus (U, V, W, i, j, k, h);
auto gxm = VGX.minus(U, V, W, i, j, k, h);
auto gyp = VGY.plus (U, V, W, i, j, k, h);
auto gym = VGY.minus(U, V, W, i, j, k, h);
auto gzp = VGZ.plus (U, V, W, i, j, k, h);
auto gzm = VGZ.minus(U, V, W, i, j, k, h);
```

The stress tensor assembly block (`txx_xp = mu_xp*(2.0*gxp.dun_dxn − …)`) is
unchanged in structure. It reads named fields from `VelocityGradComponents` instead
of local scalar variables.

`gpu_rhs.cu` is explicitly out of scope. It has its own gradient stencils and will
be migrated in a separate step once CPU functors are validated.

---

## 7. Testing (`tests/test_operators.cpp`)

### 7.1 Unit tests — manufactured solutions

| Functor | Field | Check |
|---|---|---|
| `CellGrad<DIR,2>` | `sin(2πx/L)` | error `O(h²)`, ratio ≥ 3.9 |
| `CellGrad<DIR,4>` | `sin(2πx/L)` | error `O(h⁴)`, ratio ≥ 15 |
| `FaceGrad<DIR,2>` normal | `sin(2πx/L)` | error `O(h²)` |
| `FaceGrad<DIR,2>` tangential | `sin(2πy/L)` | error `O(h²)` |
| `CellLaplacian<2>` | `x²+y²+z²` | result = `6.0` to machine precision |
| `CellDiv<2>` | `(cos x, cos y, cos z)` | error `O(h²)` |
| `VelocityGradAtFace<DIR,2>` | linear `u=ax+by+cz` | exact to machine precision |

### 7.2 Order=4 smoke test

Instantiate `CellGrad<Axis::X,4>`, `FaceGrad<Axis::X,4>`, `VelocityGradAtFace<Axis::X,4>`.
Verify error on manufactured solution is visibly smaller than `Order=2` at the same
resolution. Confirms the 4th-order path compiles and produces the correct answer;
does not require a full convergence study.

### 7.3 Symmetry test

Sod shock tube run with `viscous_rhs_impl` refactored, in X, Y, Z directions.
Results must be bit-identical across axes (structurally guaranteed by
`template<Axis DIR>` construction, verified numerically).

### 7.4 Regression gates

All existing gates must remain green:

```bash
cmake --build build -t ba    # full suite (t1–t26 + benchmarks)
cmake --build build -t t3    # T08 convergence rate ≥ 1.8
cmake --build build -t t4    # 28 sub-tests including mass conservation
```

---

## 8. Non-goals

- **GPU migration**: `gpu_rhs.cu` stencils are unchanged. A future R7-GPU step will
  port the functors into device kernels once CPU validation is complete.
- **SBP-SAT compatible stencils**: the `Order` parameter covers central-difference
  families only. Custom stencil policies (Padé, SBP) are deferred.
- **Implicit viscosity**: `CellLaplacian` is provided as a building block but no
  implicit time integration is introduced.
- **Higher ghost layers**: `Order=4` is designed to work within the current `NG=2`
  constraint. No change to `NB`, `NG`, or `NB2`.

---

## 9. Upgrade path summary

To upgrade `viscous_rhs_impl` from O(h²) to O(h⁴) after R7 is merged:

```cpp
// Before:
VelocityGradAtFace<Axis::X, 2> VGX;
// After:
VelocityGradAtFace<Axis::X, 4> VGX;
```

One line per axis (3 lines total). The T08 convergence test will verify the upgrade
does not degrade below rate 1.8 (and should improve it).
