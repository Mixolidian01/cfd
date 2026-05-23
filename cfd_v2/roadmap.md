# CFD v2 – Architecture and Boundary-Condition Roadmap

This document captures a proposed design for a fresh implementation of the solver ("cfd_v2") that preserves the current philosophy: high‑order explicit finite volumes, block‑structured AMR, GPU‑first, strong layering, and modern boundary conditions.

---

## 1. Solver architecture

### 1.1 Mesh and AMR layer

* Physics: compressible Euler/Navier–Stokes (plus optional multiphase) on rectangular domains, potentially with embedded boundaries (EB) in the future.
* Maths: decompose the domain into a hierarchy of rectangular patches (blocks) \(B_{\ell,k}\), each with a fixed interior size (e.g. \(N_x\times N_y\times N_z\)) and NG ghost layers. Each level \(\ell\) has uniform spacings \((h_x^{(\ell)}, h_y^{(\ell)}, h_z^{(\ell)})\) with 2:1 refinement between levels.
* Numerics:
  - `Patch` (successor to `CellBlock`): owns conserved variables and auxiliary fields on a single patch, plus geometry (origin, spacings, level index).
  - `Level`: a vector of patches at a given refinement level.
  - `Hierarchy`: a vector of levels plus metadata for neighbor relationships, coarse/fine interfaces, and flux registers for Berger–Colella reflux.
  - Interior ghost cells are filled from neighboring patches (same level) and from coarser levels via interpolation; domain ghost cells are handled by the BC system.

### 1.2 Discrete operators (RHS)

* Physics: convective and viscous transport of mass, momentum, and energy, plus scalar / phase‑field transport (e.g. ACDI).
* Maths: finite‑volume update
  \[
    U_i^{n+1} = U_i^n - \frac{\Delta t}{V_i} \sum_{f\in\partial V_i} F_f\cdot n_f\,A_f + \Delta t\,S_i,
  \]
  where fluxes \(F_f\) include convective and viscous contributions, and \(S_i\) includes source terms and SAT penalties.
* Numerics:
  - Convective operator: characteristic WENO5/TENO5 reconstruction, HLLC or entropy‑stable HLLC‑ES fluxes, KE‑preserving central flux blended via a sensor (e.g. Ducros) to reduce dissipation in smooth vortical regions.
  - Viscous operator: face‑based divergence of \(\tau\) and \(q\), using face‑centered gradients (initially 2nd‑order central) and face‑averaged viscosity/thermal conductivity.
  - Scalar / phase‑field (\(\phi\)): conservative advection plus a compression/diffusion term to maintain sharp but grid‑resolved interfaces.
  - SAT penalties: terms added to the RHS at coarse/fine interfaces (and later at physical boundaries) to enforce continuity and stability.

### 1.3 Time integration and AMR in time

* Physics: explicit time stepping, subject to convective and viscous CFL constraints; refined regions suggest local time stepping.
* Maths: SSP‑RK3 (or higher SSP) applied to the semi‑discrete system \(\partial_t U = R(U)\), with multirate / level‑subcycling: finer levels take smaller timesteps than coarser levels.
* Numerics:
  - Per level: compute \(\Delta t_\ell\) from CFL and diffusive constraints.
  - Subcycle fine levels so all levels reach the same physical time at synchronization points.
  - At each substep: ghost fill → RHS evaluation (convective, viscous, scalars, SAT) → update conservative state.
  - At coarse steps: apply Berger–Colella reflux using flux registers; apply SAT corrections at coarse/fine interfaces.

### 1.4 Parallelism and GPU execution

* Maths: decompose the hierarchy over MPI ranks; on each rank, patches are processed in parallel on the GPU (or CPU in fallback). Load balancing is based on patch counts and possibly per‑patch cost models.
* Numerics:
  - GPU kernels with "one patch per thread‑block" as the default granularity for RHS, ghost fill, SAT, and CFL estimation.
  - Use CUDA Graphs (or similar) to capture the SSP‑RK sequence to reduce kernel launch overhead.
  - MPI halo exchanges operate on patch faces; with GPU‑direct where available or host staging otherwise.

---

## 2. Boundary conditions (BCs)

### 2.1 Physics

Common BC types for compressible Navier–Stokes:

* Periodic tiling (homogeneous turbulence / canonical flows).
* Subsonic and supersonic inflow/outflow, often treated with characteristic (NSCBC‑type) formulations.
* Solid walls: inviscid slip and viscous no‑slip walls, with thermal conditions (adiabatic or isothermal).
* Symmetry planes.
* Phase interfaces (handled by \(\phi\) and EOS rather than as true domain boundaries).

### 2.2 Ghost‑cell framework

* Maths: ghost cells outside the physical domain are filled so that the numerical flux at the boundary face respects the desired physical BC. The Riemann/reconstruction machinery is unchanged and sees a regular grid.
* Numerics (per step or RK stage):
  1. Fill interior ghosts from neighboring patches and coarser levels.
  2. For each patch that touches the physical domain, fill ghost cells outside the domain according to BC type and face direction.
  3. Compute fluxes using the same operators as for interior faces.

### 2.3 Canonical BC recipes

These are the standard ghost‑cell transformations used in many finite‑volume codes:

* Periodic: copy interior values from the opposite side of the domain.
* Outflow / extrapolation: copy the nearest interior state in the normal direction (zero‑gradient approximation); optionally enhanced with NSCBC for reduced reflections.
* Inviscid slip wall / symmetry: odd reflection for normal velocity, even reflection for tangential velocities and scalars, to enforce zero normal velocity and zero normal gradients of scalars.
* Viscous no‑slip wall: reflect all velocity components (or momenta) with sign change; treat density and energy as even or extrapolated; set temperature in ghosts to achieve desired wall temperature (isothermal) or zero normal gradient (adiabatic).
* Subsonic inflow/outflow (NSCBC): work in characteristic variables along the boundary normal; impose incoming characteristics according to prescribed pressure/temperature/velocity and let outgoing ones be determined by the interior state.

### 2.4 Edges and corners (adjacent faces with different BCs)

* Continuous mathematics does not uniquely determine a single BC at points belonging to multiple boundary segments; corners are often singular (e.g. lid‑driven cavity).
* Numerics must adopt a consistent policy:
  - Simple precedence: define an ordering (e.g. wall > symmetry > inflow/outflow > extrap) and let a dedicated corner routine enforce it.
  - Face segmentation: associate each ghost cell with a specific physical surface (using coordinates) and apply that surface's BC logic, avoiding ambiguous corners in most cases.
  - SAT perspective: when using SBP‑SAT, corners and edges receive penalty contributions from each adjacent boundary; the total penalty is the sum, and stability follows from the energy estimate. This reduces dependence on ad‑hoc precedence rules.

### 2.5 SAT at physical boundaries

* Instead of (or in addition to) overwriting ghost cells, SBP‑SAT methods add penalty terms to the semi‑discrete equations:
  \[
    \partial_t U = R(U) + \tau (U - g) \delta_{\partial\Omega},
  \]
  where \(g\) encodes the target boundary data and \(\tau\) is chosen to preserve stability.
* For a finite‑volume code, a practical hybrid approach is:
  - Use ghost cells to provide states for Riemann fluxes at faces.
  - Use SAT‑like terms at the discrete boundary to enforce BCs weakly in an energy‑stable way (especially useful at AMR interfaces and for high‑order operators).

---

## 3. Integrated implementation roadmap

### Phase 1 – Mesh and core architecture skeleton

Goal: establish the core types (`Patch`, `Level`, `Hierarchy`) and geometry so that later operators and BCs have a clean, physically consistent foundation.

1. Implement `PatchGeometry` with origin `(x0,y0,z0)`, spacings `(hx,hy,hz)`, and `level` index. This ensures directional spacing is available everywhere (CFL, operators, BCs).
2. Implement `Patch`:
   - Constants for interior cell counts and ghost width (e.g. `NI,NJ,NK,NG`).
   - Derived totals (`NI_TOT = NI + 2*NG`, etc.) and `NCELL`.
   - Conserved variables stored in a contiguous array (SoA/AoSoA to be refined later) plus auxiliary fields like `phi`.
   - Indexing helpers: flatten `(i,j,k)` to a flat index, interior index ranges `[NG, NG+NI-1]` etc., and physical coordinate helpers for cell centers.
3. Implement `Level` and `Hierarchy`:
   - `Level` contains a vector of patches and its level index.
   - `Hierarchy` contains a vector of levels and the minimal metadata needed for neighbor lookups and flux registers (placeholders at first, with carefully documented invariants).

At the end of Phase 1, you should be able to:

* Construct a simple hierarchy with a single level and a few patches that tessellate a rectangular domain.
* Iterate over all patches and their interior cells with correct indexing and geometry.
* Reason about CFL and AMR logic using the available (x0,hx,level) metadata, even before implementing full operators.

Later phases will add: high‑order operators (Phase 2), full BC machinery (Phase 3), time integration and multilevel AMR (Phase 4), GPU/MPI integration (Phase 5), and verification & BC‑focused tests (Phase 6).
