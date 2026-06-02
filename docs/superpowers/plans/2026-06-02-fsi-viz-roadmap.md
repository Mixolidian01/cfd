# FSI + Visualization Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Live visualization via VTK XML binary writer wired into MetricsBus. (2) Extend IBM to moving walls and rigid 6-DOF dynamics. (3) Close the FSI loop with a corotational beam FEM structural solver and Aitken partitioned coupling.

**Architecture:**
- **C1 Viz:** Upgrade existing `vtk_writer.cpp` (ASCII, CPU-side) to a binary VTK XML (`.vts` per block + `.pvts` multi-block collection) writer. Wire it into `MetricsBus` as a `VtkChannel` alongside the existing `BinDumper`. No new library dependencies — raw binary VTK XML is self-contained. ParaView reads `.pvts` natively.
- **FSI-1:** Ghost-cell BC extended with `v_wall`; per-step BVH rigid transform; explicit RK3 6-DOF ODE integrator driven by integrated surface forces from `k_surface_forces_ibm`. All GPU-resident except the ODE state (host-side scalar).
- **FSI-2+3:** Corotational Euler-Bernoulli beam FEM in Python/JAX (auto-diff ready for adjoint). Partitioned Aitken Δ² relaxation: 3–5 sub-iterations per RK3 stage. Load integration from IBM ghost-cell pressure/shear to beam nodes via BVH nearest-triangle projection.

**Tech Stack:** CUDA C++20, VTK XML binary (no library — write raw bytes), Python 3 + JAX + numpy for FEM, pybind11 (D11 existing).

---

## File Map

| File | Change |
|------|--------|
| `src/io/vtk_writer.cpp` + `include/io/vtk_writer.hpp` | Replace ASCII writer with `vtk_write_binary()` + `vtk_write_pvts()` |
| `include/metrics/metrics_bus.hpp` | Add `VtkChannel` monitor type |
| `src/metrics/metrics_bus.cu` | Instantiate `VtkChannel`; call `vtk_write_binary` per `write()` |
| `include/metrics/imonitor.hpp` | No change if `VtkChannel` is an `IMonitor` subclass |
| `include/cuda/gpu_ibm.cuh` | Add `v_wall` field to ghost-cell BC struct |
| `src/cuda/gpu_ibm.cu` | `k_surface_forces_ibm` kernel; moving-wall ghost-cell reconstruction |
| `include/fsi/rigid_body.hpp` | `RigidBody6DOF` struct: state, mass, inertia tensor, RK3 integrator |
| `src/fsi/rigid_body.cpp` | `RigidBody6DOF::step()` implementation |
| `scripts/fsi/beam_fem.py` | Corotational E-B beam in JAX; Newmark-β time integrator |
| `scripts/fsi/fsi_coupler.py` | Aitken Δ² partitioned coupling loop; calls both fluid (pybind11) and beam |
| `tests/cuda/test_t52_fsi_rigid.cu` | Prescribed pitching NACA0012; compare Cl with Theodorsen theory |
| `tests/fsi/test_t53_fsi_elastic.py` | Dowell 2D flat-plate flutter speed sweep |
| `CMakeLists.txt` | Add t52 target; add t53 Python test; add `fsi/` src dir |
| `docs/dev_phases.md` | Add C1, FSI1, FSI2 phases |
| `CLAUDE.md` | Add t52, t53, t54 gate commands |

---

## Task 1 — VTK XML binary writer + MetricsBus VtkChannel

**Files:**
- Modify: `src/io/vtk_writer.cpp`, `include/io/vtk_writer.hpp`
- Modify: `include/metrics/metrics_bus.hpp`, `src/metrics/metrics_bus.cu`
- Modify: `CMakeLists.txt` (add t52_vtk compile test)

### Background

The existing `vtk_writer.cpp` writes ASCII VTK legacy format from CPU `blk.prim`. It is too slow for interactive use and not wired into `MetricsBus`. The GPU fields live in `d_Q` (conserved) or `d_scratch` (primitive, post-prim kernel). The `BinDumper` already downloads GPU fields to CPU. The plan:
1. Add `vtk_write_binary(path_prefix, step, leaves[])` that writes one `.vts` file per leaf block in VTK XML DataSet format (raw binary appended data — no VTK library needed).
2. Add `vtk_write_pvts(path_prefix, step, n_leaves)` that writes the `.pvts` parallel collection file that ParaView uses to open all blocks at once.
3. Add a `VtkChannel` to `MetricsBus` that calls these every `cfg.vtk_interval` steps if `cfg.vtk_prefix` is non-empty.

### VTK XML binary format primer (no library needed)

A `.vts` file (VTK XML StructuredGrid) for a single `NB×NB×NB` block:
```xml
<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt64">
  <StructuredGrid WholeExtent="0 NB-1 0 NB-1 0 NB-1">
    <Piece Extent="0 NB-1 0 NB-1 0 NB-1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="0"/>
      </Points>
      <PointData>
        <DataArray type="Float64" Name="rho" format="appended" offset="OFFSET_rho"/>
        ...
      </PointData>
    </Piece>
  </StructuredGrid>
  <AppendedData encoding="raw">
    _[8-byte length][raw double data][8-byte length][raw double data]...
  </AppendedData>
</VTKFile>
```

Each appended array is preceded by a `uint64_t` byte-count. All data is native (little-endian) doubles.

- [ ] **Step 1: Write `vtk_write_binary` for a single leaf block**

Replace the body of `vtk_writer.cpp` with the new binary writer. Keep the old `vtk_write` signature for backwards compatibility; add the new one:

```cpp
// New declaration in vtk_writer.hpp:
void vtk_write_binary(const std::string& path,
                      int step, int blk_id,
                      double ox, double oy, double oz, double h,
                      const double* Q,   // conserved [NVAR × NCELL], device pointer
                      int nvar);

void vtk_write_pvts(const std::string& path_prefix,
                    int step, int n_blocks);
```

For the implementation, `Q` is a device pointer. Use `cudaMemcpy` to pull the data to a host buffer first:
```cpp
std::vector<double> host(nvar * NCELL);
cudaMemcpy(host.data(), Q, host.size()*sizeof(double), cudaMemcpyDeviceToHost);
```

Then write the VTK XML manually:
1. Open file, write XML header
2. Write `<StructuredGrid WholeExtent="0 7 0 7 0 7">` (NB=8 cells, indices 0..7)
3. Write `<AppendedData encoding="raw">_`
4. For each array (Points + rho,u,v,w,p — 5 scalars from Q using cons_to_prim inline):
   - Write `uint64_t nbytes = NB*NB*NB * sizeof(double)`
   - Write the raw double array
5. Close `</AppendedData></VTKFile>`

For `vtk_write_pvts`, write the `.pvts` file that references all per-block `.vts` files:
```xml
<?xml version="1.0"?>
<VTKFile type="PStructuredGrid" ...>
  <PStructuredGrid WholeExtent="..." GhostLevel="0">
    <PPointData>
      <PDataArray type="Float64" Name="rho"/>
      ...
    </PPointData>
    <Piece Extent="0 7 0 7 0 7" Source="prefix_step000001_blk000.vts"/>
    <Piece .../>
  </PStructuredGrid>
</VTKFile>
```

- [ ] **Step 2: Build-test the new writer**

```bash
cmake --build build -t cfd_lib 2>&1 | tail -5
```

Expected: `[100%] Built target cfd_lib` — no errors.

- [ ] **Step 3: Add `VtkChannel` to MetricsBus**

In `metrics_bus.hpp`, add to `MetricsConfig`:
```cpp
std::string vtk_prefix   = "";   // empty = VTK output disabled
int         vtk_interval = 100;  // write every N steps
```

In `metrics_bus.cu`, in `MetricsBus::write()`, add after the existing monitor writes:
```cpp
if (!cfg_.vtk_prefix.empty() && step % cfg_.vtk_interval == 0) {
    // iterate snap_metas_ leaves, call vtk_write_binary per leaf
    for (int li = 0; li < n_leaves_; ++li) {
        const auto& m = snap_metas_[li];
        vtk_write_binary(cfg_.vtk_prefix, step, li,
                         m.ox, m.oy, m.oz, m.h,
                         m.d_Q, GPU_NVAR);
    }
    vtk_write_pvts(cfg_.vtk_prefix, step, n_leaves_);
}
```

- [ ] **Step 4: Add a compile test `t52_vtk`**

In `CMakeLists.txt`, add a minimal compile+run test that:
1. Creates an `NSSolver` with a TGV initial condition (reuse existing t25 setup)
2. Sets `cfg.vtk_prefix = "/tmp/test_vtk"`, `cfg.vtk_interval = 1`
3. Advances 2 steps
4. Checks that `/tmp/test_vtk_step000001_blk000.vts` exists and is a valid XML file (grep for `<VTKFile`)

```bash
cmake --build build -t t52_vtk 2>&1 | tail -10
```

Expected: `PASS  V1  VTK file written and valid XML`

- [ ] **Step 5: Commit**

```bash
git add src/io/vtk_writer.cpp include/io/vtk_writer.hpp \
        include/metrics/metrics_bus.hpp src/metrics/metrics_bus.cu \
        CMakeLists.txt
git commit -m "C1: VTK XML binary writer + MetricsBus VtkChannel; t52_vtk pass"
```

---

## Task 2 — FSI-1: Moving-wall IBM + rigid 6-DOF ODE

**Files:**
- Modify: `include/cuda/gpu_ibm.cuh`, `src/cuda/gpu_ibm.cu`
- Create: `include/fsi/rigid_body.hpp`, `src/fsi/rigid_body.cpp`
- Modify: `CMakeLists.txt` (add `src/fsi/rigid_body.cpp`; add t53 target)

### Background

The existing IBM (t49) imposes a stationary no-slip wall via ghost-cell reconstruction. Ghost cells are identified by the BVH signed-distance query; their velocity is mirrored about the wall to impose `u_wall = 0`. To support moving walls:
1. The ghost-cell BC needs to mirror about `v_wall` (not zero).
2. The BVH must be updated each RHS call after the rigid-body ODE advances.
3. A 6-DOF ODE integrator accumulates surface forces → translational/angular acceleration → velocity/position.

### Ghost-cell moving-wall BC

Current ghost-cell velocity imposition (conceptual):
```cpp
u_ghost = -u_interior;   // reflection about u=0
```
New:
```cpp
// Reflect about v_wall: u_ghost + u_interior = 2 * v_wall
u_ghost = 2.0 * v_wall.x - u_interior;
v_ghost = 2.0 * v_wall.y - v_interior;
w_ghost = 2.0 * v_wall.z - w_interior;
```

The wall velocity `v_wall` at each ghost cell is the rigid-body velocity at the nearest surface point:
```cpp
v_wall = v_cm + omega × (x_surface - x_cm);
```
where `(v_cm, omega)` is the current rigid-body state.

- [ ] **Step 1: Add `v_wall` support to ghost-cell kernel**

In `gpu_ibm.cuh`, add to the ghost-cell parameter struct:
```cpp
double3 v_cm    = {0,0,0};   // centre-of-mass velocity
double3 omega   = {0,0,0};   // angular velocity
double3 x_cm    = {0,0,0};   // centre-of-mass position
```

In `gpu_ibm.cu`, in the ghost-cell reconstruction kernel (wherever `u_ghost = -u_interior` appears), replace with the `v_wall` reflection above. Compute `x_surface` as the BVH nearest-point already stored in the ghost-cell meta struct.

- [ ] **Step 2: Add `k_surface_forces_ibm` kernel**

New kernel that walks all ghost cells, reconstructs the pressure and viscous stress at the nearest surface point, and atomically accumulates force and torque:

```cpp
__global__
void k_surface_forces_ibm(
    const GpuLeafRhsMeta* metas,
    const GpuIbmGhostMeta* ghost_metas,  // existing ghost-cell metadata
    int n_ghost,
    double3 x_cm,
    double* d_F_out,   // [6]: Fx,Fy,Fz,Tx,Ty,Tz (atomic accumulation)
    double dA)         // face area = h^2
{
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= n_ghost) return;
    const auto& g = ghost_metas[gi];
    // pressure at surface ≈ average of ghost and interior
    double p_surf = 0.5 * (g.p_ghost + g.p_interior);
    // normal force: F = -p * n * dA
    double3 F = {-p_surf * g.nx * dA,
                 -p_surf * g.ny * dA,
                 -p_surf * g.nz * dA};
    double3 r = {g.x_surf - x_cm.x,
                 g.y_surf - x_cm.y,
                 g.z_surf - x_cm.z};
    double3 T = cross(r, F);
    atomicAdd(&d_F_out[0], F.x); atomicAdd(&d_F_out[1], F.y); atomicAdd(&d_F_out[2], F.z);
    atomicAdd(&d_F_out[3], T.x); atomicAdd(&d_F_out[4], T.y); atomicAdd(&d_F_out[5], T.z);
}
```

- [ ] **Step 3: Implement `RigidBody6DOF`**

Create `include/fsi/rigid_body.hpp`:
```cpp
#pragma once
#include <array>

struct RigidBody6DOF {
    double mass;
    double I[9];          // inertia tensor (body frame, row-major)
    double x[3] = {};     // position (world frame)
    double q[4] = {1,0,0,0};  // orientation quaternion (w,x,y,z)
    double v[3] = {};     // translational velocity
    double omega[3] = {}; // angular velocity (world frame)

    // Advance state by dt using explicit RK3 (matching fluid integrator).
    // F[6] = {Fx,Fy,Fz,Tx,Ty,Tz} are forces+torques in world frame.
    void step(const double F[6], double dt);

    // Return velocity at world point x_p (for ghost-cell wall velocity).
    void wall_velocity(const double x_p[3], double v_out[3]) const;
};
```

Create `src/fsi/rigid_body.cpp` with `step()` implementing standard explicit Euler (upgrade to RK3 later):
- Translational: `v += (F/mass)*dt`, `x += v*dt`
- Rotational: integrate `omega` via `I*α = T - omega×(I*omega)`, then update quaternion

- [ ] **Step 4: Wire into `GpuGraphSolver::advance()`**

After each fluid RHS call (before positivity floor), call `k_surface_forces_ibm`, download `d_F_out` to host, call `rigid_body.step(F, dt)`, update BVH with the new rigid-body transform, and upload the new `v_cm/omega/x_cm` to the IBM ghost-cell parameter struct.

- [ ] **Step 5: Add gate `t53` — prescribed pitching NACA0012**

Write `tests/cuda/test_t53_fsi_rigid.cu`:
- NACA0012 at Re=1000, prescribed sinusoidal pitch `α(t) = α₀ sin(2πkt)` (k=0.25 reduced frequency, α₀=5°)
- Run 2 periods; integrate lift coefficient `Cl(t)`
- Theodorsen theory: `|Cl_amplitude| ≈ 2π(α₀ + α̇/(2k))` (small amplitude, Theodorsen C(k)≈0.8 for k=0.25)
- Gate: `|Cl_amplitude - Cl_theodorsen| / Cl_theodorsen < 0.15` (15% tolerance — coarse grid)

```bash
cmake --build build -t t53 2>&1 | tail -10
```

Expected: `PASS  F1  Cl amplitude within 15% of Theodorsen theory`

- [ ] **Step 6: Commit**

```bash
git add include/cuda/gpu_ibm.cuh src/cuda/gpu_ibm.cu \
        include/fsi/rigid_body.hpp src/fsi/rigid_body.cpp \
        CMakeLists.txt tests/cuda/test_t53_fsi_rigid.cu
git commit -m "FSI-1: moving-wall IBM + k_surface_forces_ibm + RigidBody6DOF; t53 pass"
```

---

## Task 3 — FSI-2+3: Elastic coupling + load integration + Aitken relaxation

**Files:**
- Create: `scripts/fsi/beam_fem.py` (corotational beam in JAX)
- Create: `scripts/fsi/fsi_coupler.py` (Aitken partitioned coupling)
- Create: `tests/fsi/test_t54_fsi_elastic.py` (Dowell flutter gate)
- Modify: `CMakeLists.txt` (add t54 Python test)

### Background

The Aitken Δ² acceleration for partitioned FSI works as follows per time step:
1. Predict structural displacement `x_s^{(0)}` (extrapolation from previous step).
2. Update IBM BVH from predicted displacement.
3. Advance fluid one RK3 step → surface loads `F_fluid`.
4. Advance structural solver one step with `F_fluid` → new displacement `x_s^{(1)}`.
5. Compute interface residual `r = x_s^{(1)} - x_s^{(0)}`.
6. If `‖r‖ < tol`, accept; otherwise update relaxation ω via Aitken formula and repeat from step 2.

Typically 3–5 sub-iterations suffice for wing-in-air (low added-mass ratio).

- [ ] **Step 1: Corotational beam FEM in JAX**

Create `scripts/fsi/beam_fem.py`:

```python
import jax
import jax.numpy as jnp

def local_stiffness(E, A, Iy, Iz, J, L):
    """12×12 Euler-Bernoulli beam local stiffness matrix."""
    ...  # standard textbook expressions

def corotational_step(nodes, u, F_ext, dt, E, A, Iy, Iz, J, rho_s):
    """
    One Newmark-β implicit step (β=0.25, γ=0.5 → unconditionally stable).
    nodes: (N_nodes, 3) reference positions
    u:     (N_nodes, 6) current dof state [dx,dy,dz,θx,θy,θz]
    F_ext: (N_nodes, 6) external nodal forces/moments
    Returns updated u, u_dot
    """
    # Assemble global K, M via corotational transformation
    # Solve (M/dt² + K) Δu = F_ext - K u (linearized)
    ...
```

Use JAX's `jit` for the assembly loop; `jnp.linalg.solve` for the linear solve. The JAX auto-diff will give structural adjoint for free later.

- [ ] **Step 2: Load integration (FSI-3)**

In `scripts/fsi/fsi_coupler.py`, implement `integrate_loads(surface_forces, bvh, beam_nodes)`:
- For each beam node, find the nearest IBM surface triangle via BVH (already Python-accessible via pybind11)
- Project distributed traction field onto beam cross-sections: `F_node_i = Σ_{triangles near node_i} p_surf * n * dA`
- Return `F_ext` array shaped `(N_nodes, 6)`

- [ ] **Step 3: Aitken partitioned coupling loop**

In `scripts/fsi/fsi_coupler.py`, implement the main FSI time-advance:

```python
def fsi_advance(solver, beam, dt, tol=1e-4, max_iter=10):
    """One coupled FSI time step with Aitken Δ² relaxation."""
    x_pred = beam.u.copy()                    # predictor
    omega_aitken = 0.1                         # initial relaxation
    r_prev = None

    for it in range(max_iter):
        solver.set_ibm_displacement(x_pred)    # update BVH
        solver.advance(dt)                     # fluid step
        F_ext = integrate_loads(solver, beam)  # FSI-3
        x_new = beam.step(F_ext, dt)           # structural step
        r = x_new - x_pred
        if jnp.linalg.norm(r) < tol:
            break
        # Aitken update
        if r_prev is not None:
            dr = r - r_prev
            omega_aitken = -omega_aitken * jnp.dot(r_prev.ravel(), dr.ravel()) \
                           / (jnp.linalg.norm(dr)**2 + 1e-300)
            omega_aitken = jnp.clip(omega_aitken, 0.1, 1.0)
        x_pred = x_pred + omega_aitken * r
        r_prev = r

    beam.u = x_pred
    return it + 1
```

- [ ] **Step 4: Gate `t54` — Dowell 2D flat-plate flutter**

Create `tests/fsi/test_t54_fsi_elastic.py`:
- Flat plate of span L=0.5 m, chord c=0.1 m, thickness t=0.001 m, E=70 GPa (aluminium)
- Uniform flow at M=0.3; sweep reduced velocity U* = U/(b*ω_α) from 1 to 10
- Detect onset of flutter (diverging amplitude) by checking if `max|Δz|` grows >10× over 2 periods
- Analytical flutter speed (Theodorsen): U*_flutter ≈ 6.28 for this aspect ratio and mass ratio
- Gate: detected U*_flutter ∈ [5.3, 7.3] (±15% of theory)

```bash
cmake --build build -t t54 2>&1 | tail -10
```

Expected: `PASS  F2  Flutter onset U* ∈ [5.3, 7.3]`

- [ ] **Step 5: Commit**

```bash
git add scripts/fsi/ tests/fsi/ CMakeLists.txt
git commit -m "FSI-2+3: corotational beam FEM (JAX) + Aitken FSI coupling + load integration; t54 pass"
```

---

## Self-check before merge

```bash
cmake --build build -t ba 2>&1 | grep -E "Result:|PASS|FAIL|failure"
```

Expected: all `PASS`, `0 failure(s)` for every gate.
