# CFD Solver — Technical Documentation

> Updated: 2026-05-16
> Covers: physics, mathematics, numerical schemes, and developer extension API for every component.
> Reflects the `to_refactor` branch (diverges from `to_debug` at commit 2f16dae; R0–R6 complete).

---

## 0. Architecture Overview

The solver is organised as a **four-layer runtime stack** plus **three orthogonal refactor layers** added on top.

### 0.1 Runtime Stack

```
Layer 0  include/linalg.hpp / src/linalg/linalg.cpp
         Kahan-summation BLAS-1, CG solver, geometric multigrid.
         No knowledge of blocks, physics, or time integration.

Layer 1  include/mesh/cell_block.hpp      — CellBlock (AoSoA, NB=8, NG=2)
         include/mesh/block_tree.hpp      — BlockTree octree, AMR topology
         src/mesh/amr_operators.cpp       — prolong / restrict / fill_cf_ghosts
         (amr_operators.cpp must stay in libblock; it is called by tree_rhs)

Layer 2  include/schemes/operators.hpp   — compute_rhs, tree_rhs, typed overloads
         src/schemes/operators.cpp        — main dispatch + undo_cf_one_face
         src/schemes/convective_rhs.cpp   — accumulate_face<DIR> hybrid loop
         src/schemes/viscous_rhs.cpp      — viscous_rhs_impl, undo_cf_viscous_energy

Layer 3  include/solver/ns_solver.hpp    — NSSolver, SolverConfig
         src/solver/ns_solver.cpp         — advance(), regrid(), flux correction
         src/cuda/gpu_graph.cuh/cu        — CUDA Graph RK3 (GPU path)
         src/cuda/gpu_rhs.cuh/cu          — WENO5-Z GPU RHS
         src/cuda/gpu_ghost_fill.cuh/cu   — GPU ghost fill
```

Key compile-time constants — never change without updating both CPU and GPU headers:

| Constant | Value | Where |
|---|---|---|
| `NB` | 8 | interior cells per axis |
| `NG` | 2 | ghost layers per face |
| `NB2` | 12 | total cells per axis (NB + 2·NG) |
| `NCELL` | 1728 | cells per block (NB2³) |
| `NVAR` | 5 | conserved fields (ρ, ρu, ρv, ρw, E) |
| `GAMMA` | 1.4 | default ratio of specific heats (overridable via `SolverConfig`) |

### 0.2 Refactor Layers (R0–R6, all complete)

Three orthogonal layers added on top of the runtime stack without replacing it:

**LAYER P — Physics functors** (`include/physics/`)
Structs with `operator()`. Marked `__host__ __device__`. Templated on `Axis DIR`.
Carry only physics state (γ, p∞, μ…). Zero execution knowledge.
Files: `hllc_flux.hpp`, `weno5_recon.hpp`, `ideal_gas_eos.hpp`, `stiffened_gas_eos.hpp`, `diff_ops.hpp`, `face_interp.hpp`.

**LAYER C — Concept contracts** (`include/schemes/concepts.hpp`)
C++20 concepts: `RiemannFlux`, `SpatialReconstruction`, `EquationOfState`, `BoundaryCondition`.
Property flags: `is_entropy_stable<F>`, `is_conservative<F>`, `is_skew_symmetric<F>`.
Applied at every template boundary via `requires` clauses. Never inside `__global__` kernels.

**LAYER E — Execution backend** (`SolverConfig::ExecConfig`)
`ExecutionBackend::CPU` / `ExecutionBackend::GPU` tags. `SolverConfig::FluxScheme` enum.
`compute_rhs_typed<Flux,Recon,EOS>` instantiation matrix in `src/schemes/operators.cpp`.
Factory-style: `NSSolver::init()` reads `cfg.exec` and selects the pre-compiled specialisation.

**R6 — mdspan axis views** (`include/mesh/cell_block.hpp`, `include/vendor/mdspan.hpp`)
`AoSoAAccessor` + `md::mdspan` give zero-copy axis rotation for `template<Axis DIR>` kernels.
`CellBlock::axis_view<DIR>(v)` returns a 3D view that maps `(n,a,b)` to the correct AoSoA index
without any branch on `DIR`. Free function `cell_idx_axis<DIR>(n,a,b)` provides the flat equivalent.

---

## 1. Governing Equations

The solver integrates the **3D compressible Navier-Stokes equations** in conservation form:

$$
\frac{\partial \mathbf{Q}}{\partial t} + \frac{\partial \mathbf{F}_i}{\partial x_i} = \frac{\partial \mathbf{G}_i}{\partial x_i} + \mathbf{S}
$$

where $\mathbf{Q}$ is the vector of conserved variables, $\mathbf{F}_i$ are the inviscid (convective) flux components, $\mathbf{G}_i$ are the viscous flux components, and $\mathbf{S}$ is the source term (chemistry, SGS model).

### 1.1 Conserved Variables

$$
\mathbf{Q} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho w \\ E \end{pmatrix}
\quad\text{(NVAR = 5)}
$$

where $\rho$ is density [kg/m³], $\rho u_i$ is momentum per unit volume [kg/m²·s], and $E$ is total energy per unit volume [J/m³]:

$$
E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho|\mathbf{u}|^2
$$

### 1.2 Inviscid Flux

The inviscid flux in direction $x_i$ is:

$$
\mathbf{F}_i = \begin{pmatrix} \rho u_i \\ \rho u u_i + p\delta_{1i} \\ \rho v u_i + p\delta_{2i} \\ \rho w u_i + p\delta_{3i} \\ (E+p)u_i \end{pmatrix}
$$

This flux vector is **isotropic under permutation of axes**: $\mathbf{F}_x$, $\mathbf{F}_y$, $\mathbf{F}_z$ are the same function with arguments permuted. This is the mathematical justification for `template<Axis DIR>` — one implementation covers all three directions.

### 1.3 Viscous Flux

$$
\mathbf{G}_i = \begin{pmatrix} 0 \\ \tau_{i1} \\ \tau_{i2} \\ \tau_{i3} \\ \tau_{ij}u_j + q_i \end{pmatrix}
$$

where $\tau_{ij} = \mu\!\left(\partial u_i/\partial x_j + \partial u_j/\partial x_i - \frac{2}{3}\delta_{ij}\nabla\cdot\mathbf{u}\right)$ and $q_i = -\kappa\partial T/\partial x_i$ (Pr = 0.72).

---

## 2. Equation of State and Transport Properties

### 2.1 Perfect Gas EOS (`include/physics/ideal_gas_eos.hpp`)

```cpp
struct IdealGasEOS {
    double gamma = GAMMA;
    Prim cons_to_prim(double rho, double rhou, double rhov, double rhow, double E) const noexcept;
};
```

$$
p = (\gamma - 1)\!\left(E - \tfrac{1}{2}\rho|\mathbf{u}|^2\right), \quad T = \frac{p}{\rho R}, \quad c = \sqrt{\gamma p/\rho}
$$

with $R = 287.058$ J/(kg·K). The EOS satisfies `EquationOfState<IdealGasEOS>`.

### 2.2 Stiffened Gas EOS (`include/physics/stiffened_gas_eos.hpp`)

```cpp
struct StiffenedGasEOS {
    double gamma = GAMMA;
    double p_inf = 0.0;   // [Pa]; 0 → reduces to IdealGasEOS
    Prim cons_to_prim(double rho, double rhou, double rhov, double rhow, double E) const noexcept;
};
```

$$
p = (\gamma-1)\!\left(E - \tfrac{1}{2}\rho|\mathbf{u}|^2\right) - \gamma p_\infty
$$

Enables liquid-like fluids (water: γ ≈ 6.12, p∞ ≈ 3.43×10⁸ Pa). The Allaire mixture rule in `SolverConfig::AcdiConfig` blends two stiffened-gas EOS instances across a diffuse interface.

### 2.3 Sutherland Viscosity Law (`cell_block.hpp::sutherland`)

$$
\mu(T) = \mu_\text{ref}\!\left(\frac{T}{T_\text{ref}}\right)^{3/2}\frac{T_\text{ref}+S}{T+S}
$$

with $\mu_\text{ref} = 1.716\times10^{-5}$ Pa·s, $T_\text{ref} = 273.15$ K, $S = 110.4$ K.
Computed as `ratio * sqrt(ratio)` rather than `pow` (~10× faster). No temperature floor — if $T < 0$ (transient near strong rarefactions) `sqrt` returns NaN and poisons the block.

### 2.4 Extension API — adding a new EOS

1. Create `include/physics/my_eos.hpp`:
   ```cpp
   struct MyEOS {
       // any runtime parameters
       Prim cons_to_prim(double rho, double rhou, double rhov, double rhow, double E) const noexcept;
   };
   static_assert(EquationOfState<MyEOS>);
   ```
2. Add an explicit instantiation row in `src/schemes/operators.cpp`:
   ```cpp
   template void compute_rhs_typed<HllcEsFlux, Weno5Recon, MyEOS>(
       const CellBlock&, CellBlock&, const DucrosConfig&) noexcept;
   ```
3. Add the corresponding `tree_rhs_typed` instantiation and a `FluxScheme` enum entry if runtime dispatch is needed.

---

## 3. Spatial Discretization

### 3.1 Block Structure and Memory Layout (`include/mesh/cell_block.hpp`)

| Parameter | Value | Meaning |
|---|---|---|
| `NB` | 8 | interior cells per axis |
| `NG` | 2 | ghost layers per face |
| `NB2` | 12 | total cells per axis |
| `NCELL` | 1728 | total cells per block (NB2³) |
| `NTILE` | 216 | AoSoA tiles per block (NCELL/8) |
| `W` | 8 | AVX-512 double lanes per tile |

**AoSoA memory layout:**
```
data_[tile * NVAR * W + v * W + lane]
  tile = flat >> 3        (flat / 8)
  lane = flat & 7         (flat % 8)
```
All `tile_ptr(v, t)` pointers are 64-byte aligned — enables AVX-512 autovectorisation of EOS loops. GPU SoA layout on device (`d_Q[v * NCELL + flat]`) is separate and requires explicit upload/download.

**Cell index convention:**
$$\text{flat}(i,j,k) = k\cdot\text{NB2}^2 + j\cdot\text{NB2} + i$$

Interior range: $i,j,k \in [2, 9]$. Ghosts: $[0,1]$ and $[10,11]$.

### 3.2 R6 mdspan Axis Views

`CellBlock::axis_view<DIR>(int v)` returns a `BlockView` (an `md::mdspan` over 3D extents `[NB2][NB2][NB2]`) with an `AoSoAAccessor` that translates the logical flat index from the layout into the AoSoA physical address:

```cpp
blk.axis_view<Axis::X>(v)(i, j, k)  // maps to data_[tile*NVAR*W + v*W + lane]
blk.axis_view<Axis::Y>(v)(j, i, k)  // same cell as cell_idx(i,j,k), Y-normal stride
blk.axis_view<Axis::Z>(v)(k, i, j)  // Z-normal stride
```

The **layout strides** encode the axis rotation:

| Axis | stride[0] (normal n) | stride[1] (tangential a) | stride[2] (tangential b) |
|---|---|---|---|
| X | 1 | NB2 | NB2² |
| Y | NB2 | 1 | NB2² |
| Z | NB2² | 1 | NB2 |

`cell_idx_axis<DIR>(n, a, b)` provides the flat equivalent without constructing a view — used for reading `pc[]` and `duc[]` arrays that are not owned by `CellBlock`.

**Portability:** `include/vendor/mdspan.hpp` provides `namespace md` pointing to `Kokkos::` (P0009 reference impl, C++20 fallback) or `std::` (C++23 stdlib when `__cpp_lib_mdspan` is defined).

### 3.3 Ghost Fill Protocols

**Same-level faces:** Direct copy from the neighbouring block's interior cells via `fill_ghosts_periodic` / `fill_ghosts_wall` / `fill_ghosts_open` in `BlockTree`.

**Coarse-fine (C/F) faces:** `fill_cf_ghosts()` in `amr_operators.cpp` applies 5th-order Lagrange interpolation in the normal direction and piecewise-constant in the transverse directions.

**Unified dispatch (R3):** `GhostFiller::fill_all(tree, bc_variant, cf_zero_grad)` accepts a `BCVariant` and calls `std::visit` to dispatch to the correct fill strategy. Use this at every ghost fill site; do not call individual `fill_ghosts_*` methods directly.

#### C/F Ghost Fill: 5th-Order Lagrange

5-cell stencil at offsets $\{-4,-3,-2,-1,0\}$ (relative to the last interior coarse cell):

$$
q_f^{(\text{gl}=0)} = \sum_{k=0}^{4} L_k^+ \, q_c^{(i_0-4+k)}, \quad
q_f^{(\text{gl}=1)} = \sum_{k=0}^{4} L_k^- \, q_c^{(i_0-4+k)}
$$

Coefficients (`amr_operators.cpp`):
$$L^+ = \tfrac{1}{6144}\{585,-3060,6630,-7956,9945\}, \quad L^- = \tfrac{1}{6144}\{-231,1260,-2970,4620,3465\}$$

Requires `NB ≥ 5` so both stencils fit within the coarse interior.

---

## 4. Convective Operator

### 4.1 Face-Centred Flux Loop (`src/schemes/convective_rhs.cpp`)

For each Cartesian direction: $(N_B+1)\times N_B\times N_B = 576$ faces per axis, 1728 total per block. Each face evaluated **once**. Sign convention: flux $F$ leaves the left cell.

$$\text{rhs}[\text{left}] \mathrel{-}= h^{-1}F, \quad \text{rhs}[\text{right}] \mathrel{+}= h^{-1}F$$

The accumulation now goes through `rhs.axis_view<DIR>(v)(n,a,b)` — zero per-axis branching (R6).

### 4.2 Hybrid Scheme: Ducros Sensor + KEP/WENO5-ES Blend

$$\mathbf{F} = (1-\theta)\,\mathbf{F}_\text{KEP} + \theta\,\mathbf{F}_\text{WENO5-ES}$$

$\theta = \max(\Phi_L, \Phi_R)$ from the combined Ducros-pressure sensor.
Fast path: when $\theta < 10^{-8}$ only $\mathbf{F}_\text{KEP}$ is computed; WENO5 reconstruction is skipped.

### 4.3 Ducros + Pressure-Ratio Shock Sensor

**Velocity-based (Ducros 1999):**
$$\Phi_\text{vel} = \frac{(\nabla\cdot\mathbf{u})^2}{(\nabla\cdot\mathbf{u})^2 + |\nabla\times\mathbf{u}|^2 + \varepsilon}$$

**Pressure-ratio sensor** (catches stationary shocks where $\mathbf{u}=0$ at $t=0$):
$$\Phi_p = \max_d\frac{|p_{i\pm 1} - p_i|}{p_i + \varepsilon}$$
Linear ramp: $\Phi_p < \tau \to 0$; $\Phi_p > \tau+w \to 1$. Configurable via `SolverConfig::NumericsConfig`:
```cpp
cfg.numerics.ducros_p_threshold = 0.1;   // τ: ramp lower bound
cfg.numerics.ducros_blend_width  = 0.1;   // w: ramp width
```
Raise `ducros_p_threshold` to ≥ 0.5 for DNS/LES without shocks to stay in the KEP-only path.

### 4.4 KE-Preserving Flux — Pirozzoli (2011) (`include/physics/hllc_flux.hpp`)

```cpp
template<Axis DIR> struct HllcFlux {
    std::array<double,NVAR> operator()(const Prim& L, const Prim& R) const noexcept;
};
static_assert(is_conservative_v<HllcFlux<Axis::X>>);
```

Arithmetic-mean primitive variables: $\bar\rho$, $\bar{u}_i$, $\bar{p}$, $\bar{H}$.
$$F_\text{KEP}[0]=\bar\rho\bar{u}_n,\quad F_\text{KEP}[\text{mom}_i]=\bar\rho\bar{u}_n\bar{u}_i+\bar{p}\delta_{ni},\quad F_\text{KEP}[E]=\bar\rho\bar{u}_n\bar{H}$$

Conserves discrete KE to machine precision on smooth flows; no numerical dissipation added.

### 4.5 WENO5-Z Reconstruction with Characteristic Decomposition (`include/physics/weno5_recon.hpp`)

```cpp
template<Axis DIR> struct Weno5Recon {
    void operator()(const Prim* pc, int i, int j, int k,
                    Prim& qL, Prim& qR) const noexcept;
};
static_assert(SpatialReconstruction<Weno5Recon<Axis::X>>);
```

**Scalar WENO-Z** (Borges et al. 2008): three 3rd-order candidates + global smoothness indicator $\tau_5 = |\beta_0 - \beta_2|$.
$$\alpha_k = d_k\!\left(1+\frac{\tau_5^2}{(\beta_k+\varepsilon)^2}\right), \quad \varepsilon=10^{-36}, \quad (d_0,d_1,d_2)=(0.1,0.6,0.3)$$

**Characteristic decomposition:** Roe-averaged state → project 6-cell stencil to characteristic space → apply WENO-Z independently on each field → back-project. Prevents cross-contamination of acoustic, entropy, and shear waves at strong shocks.

**Safe fallback:** If reconstructed ρ ≤ 0 or p ≤ 0, fall back to cell-center primitive to prevent NaN propagating through `log_mean`.

### 4.6 Entropy-Stable HLLC-ES Flux — Chandrashekar (2013) (`include/physics/hllc_flux.hpp`)

```cpp
template<Axis DIR> struct HllcEsFlux {
    std::array<double,NVAR> operator()(const Prim& L, const Prim& R) const noexcept;
};
static_assert(is_entropy_stable_v<HllcEsFlux<Axis::X>>);
static_assert(is_conservative_v<HllcEsFlux<Axis::X>>);
```

Entropy-conservative base flux uses **log-mean** density and temperature:
$$\hat\rho = \text{logmean}(\rho_L,\rho_R), \quad \hat\beta = \text{logmean}(\beta_L,\beta_R), \quad \beta = \rho/(2p)$$

Entropy-STABLE flux adds scalar LF dissipation:
$$\mathbf{F}^{ES} = \mathbf{F}^{EC} - \tfrac{\lambda_\text{max}}{2}\Delta\mathbf{Q}, \quad \lambda_\text{max} = \max(|u_{n,L}|+c_L, |u_{n,R}|+c_R)$$

**Numerically stable log-mean:** For $|f|^2 < 10^{-4}$ uses Taylor expansion $F = 1 + f^2/3 + f^4/5 + f^6/7$ to avoid catastrophic cancellation.

**Wall face detection:** `is_wall_ghost(pL, pR)` detects the anti-symmetric momentum pattern (exact $p_L = p_R$, $|u_L + u_R| \ll \varepsilon$) and substitutes $\mathbf{F}_\text{KEP}$ to prevent the LF dissipation term from draining tangential momentum at no-slip walls.

### 4.7 Extension API — adding a new Riemann flux

1. Create `include/physics/my_flux.hpp`:
   ```cpp
   template<Axis DIR>
   struct MyFlux {
       std::array<double,NVAR> operator()(const Prim& L, const Prim& R) const noexcept {
           // ... physics ...
       }
   };
   template<Axis DIR> struct is_conservative<MyFlux<DIR>> : std::true_type {};
   static_assert(RiemannFlux<MyFlux<Axis::X>>);
   ```
2. Add explicit instantiations in `src/schemes/operators.cpp`:
   ```cpp
   #include "physics/my_flux.hpp"
   template void compute_rhs_typed<MyFlux, Weno5Recon, IdealGasEOS>(...) noexcept;
   template void tree_rhs_typed   <MyFlux, Weno5Recon, IdealGasEOS>(...) noexcept;
   ```
3. Add `MyFlux` to `SolverConfig::FluxScheme` enum and the dispatch in `cpu_rk3.cpp`.
4. Gate test: add a row to `operators_t.cpp` verifying the flux conserves mass on a uniform state.

### 4.8 Extension API — changing scheme at runtime

```cpp
NSSolver solver;
solver.cfg.exec.flux_scheme = SolverConfig::FluxScheme::HLLC;   // plain HLLC
// or
solver.cfg.exec.flux_scheme = SolverConfig::FluxScheme::HLLC_ES; // entropy-stable (default)
solver.init(L, ic);
```
The `CpuRk3Integrator::step()` dispatches to the matching `tree_rhs_typed` specialisation based on `cfg.exec.flux_scheme`.

---

## 5. Viscous Operator

### 5.1 Conservative Divergence Form (`src/schemes/viscous_rhs.cpp::viscous_rhs_impl`)

Face-averaged viscosity and conservative stress divergence:
$$\mu_{i+\frac{1}{2}} = \tfrac{1}{2}(\mu_i + \mu_{i+1})$$
$$\left(\frac{\partial(\rho u_x)}{\partial t}\right)_\text{visc} = h^{-1}\!\left[(\tau_{xx}|_{i+1/2} - \tau_{xx}|_{i-1/2}) + \ldots\right]$$

All nine velocity gradient components at faces are computed by `VelocityGradAtFace<Axis, Order>` from `include/physics/diff_ops.hpp`:
```cpp
constexpr VelocityGradAtFace<Axis::X, 2> VGX;
auto g = VGX.plus(U, V, W, i, j, k, h);
// g.dun_dxn, g.dun_dxt1, g.dut1_dxn, ... g.divu()
```

### 5.2 Energy Equation Viscous Term

Conservative face-flux form:
$$\left(\frac{\partial E}{\partial t}\right)_\text{visc} = h^{-1}\!\left[(F_e|_{x+1/2} - F_e|_{x-1/2}) + \ldots\right]$$
$$F_e|_{x+1/2} = \tau_{xx}\bar{u} + \tau_{xy}\bar{v} + \tau_{xz}\bar{w} + \kappa\,h^{-1}(T_{i+1}-T_i)$$

**Known GPU/CPU inconsistency:** `k_rhs_visc` uses the face-flux form; the CPU `viscous_rhs_impl` uses a cell-centre form for the energy viscous work term. This causes an O(h²) difference in viscous energy dissipation that shows up as ~7.6% error in `t26` A1 after 20 steps.

### 5.3 C/F Viscous Energy Reflux (`viscous_rhs.cpp::undo_cf_viscous_energy`)

At C/F interfaces, the coarse block used a ghost-based viscous energy flux while the fine block used interior-based values. `undo_cf_viscous_energy` replaces the coarse flux with the fine-based estimate via `cf_visc_energy_flux<AX>`, applying the same Berger-Colella correction principle as the convective reflux.

### 5.4 Extension API — modifying the viscous scheme

- **Higher-order gradients:** `VelocityGradAtFace<Axis, 4>` provides 4th-order normal derivatives. Substitute `constexpr VelocityGradAtFace<Axis::X, 4> VGX;` in `viscous_rhs_impl`.
- **IMEX (implicit viscous):** Enable via `cfg.physics.use_imex = true`. This routes the viscous operator through the implicit-explicit ARK integrator using a multigrid preconditioner (see §6.3).
- **Variable Prandtl number:** `CPU_CP` and `PR` are compile-time constants in `cell_block.hpp`. To make Pr a runtime parameter, add it to `SolverConfig::PhysicsConfig` and thread it into `viscous_rhs_impl`.

---

## 6. Time Integration

### 6.1 SSP-RK3: Shu-Osher Form (`src/solver/cpu_rk3.cpp` / `ns_solver.cpp`)

$$\mathbf{Q}^{(1)} = \mathbf{Q}^n + \Delta t\,\mathcal{L}(\mathbf{Q}^n)$$
$$\mathbf{Q}^{(2)} = \tfrac{3}{4}\mathbf{Q}^n + \tfrac{1}{4}\mathbf{Q}^{(1)} + \tfrac{1}{4}\Delta t\,\mathcal{L}(\mathbf{Q}^{(1)})$$
$$\mathbf{Q}^{n+1} = \tfrac{1}{3}\mathbf{Q}^n + \tfrac{2}{3}\mathbf{Q}^{(2)} + \tfrac{2}{3}\Delta t\,\mathcal{L}(\mathbf{Q}^{(2)})$$

Equivalent Butcher weights: $w_1 = 1/6$, $w_2 = 1/6$, $w_3 = 2/3$ (sum = 1). These are the stage weights passed to `tree_rhs_typed(..., stage_weight)` for flux register accumulation.

**SSP property:** If forward Euler preserves a monotone/TVD/positivity property, the 3-stage scheme preserves it too — essential for $\rho > 0$, $p > 0$ with WENO5-Z near shocks.

**Regrid ordering:** `regrid()` runs at the **top** of `advance()`, on $\mathbf{Q}^n$, before zeroing flux registers. Regridding after `apply_flux_correction()` would allow `coarsen()` to overwrite flux-corrected cells (verified ~2.69e-8 mass leak if violated).

### 6.2 CFL Condition

$$\Delta t = \text{CFL} \cdot \min_\text{leaves} \frac{h}{|u_n| + c}$$

Acoustic CFL only — the viscous stability condition $\Delta t_\text{visc} = \rho h^2/(2\mu\max(4/3, \gamma/\text{Pr}))$ is not enforced. At high AMR refinement levels or near hot-wall gradients, viscous CFL may dominate; verify manually when $\Delta x \ll 1$.

```cpp
cfg.time.cfl   = 0.4;   // gate tests
cfg.time.cfl   = 0.8;   // production LES
cfg.time.t_end = 10.0;
```

### 6.3 IMEX-ARK: Implicit Viscous Solve (`src/solver/lts_integrator.cpp`)

Enabled via `cfg.physics.use_imex = true`. Uses a 3rd-order additive Runge-Kutta scheme (Kennedy & Carpenter 2003):

- **Explicit part:** convective + SGS RHS, computed via `tree_rhs_typed`.
- **Implicit part:** viscous operator, solved by CG+multigrid at each stage.
- **Multigrid levels:** `cfg.physics.mg_levels` (default 3). Ensure $N_B \ge 2^{mg\_levels}$.

Use when $\Delta t_\text{visc} \ll \Delta t_\text{conv}$ (e.g. low-Mach wall-bounded flows with fine near-wall spacing).

### 6.4 Local Time Stepping (`src/solver/lts_integrator.cpp`)

Enabled via `cfg.amr.use_lts = true`. Each refinement level advances with its own `dt_\ell = dt_0 / 2^\ell`, using the Berger-Oliger sub-cycling approach. Fine levels take multiple RK3 steps per coarse step.

```cpp
cfg.amr.use_lts   = true;
cfg.amr.lts_ratio = 2;   // must match the tree's geometric refinement ratio
```

### 6.5 Non-Throwing Interface (`include/solver/solver_result.hpp`)

`NSSolver::advance_result()` wraps `advance()` in a try/catch and returns `SolverResult<double>`:

```cpp
auto r = solver.advance_result();
if (!r.ok()) { std::cerr << r.error().message << "\n"; return 1; }
double dt = r.value();
```

`SolverResult<T>` is a C++20 polyfill for `std::expected<T, SolverError>` using `std::variant<T, SolverError>`. Error code 1 = non-positive `dt`; code 2 = exception; code 3 = unknown exception.

### 6.6 Extension API — adding a new integrator

1. Subclass `IRk3Integrator` (if the interface exists) or directly replace the `step()` call in `ns_solver.cpp`.
2. Keep the ghost fill + `tree_rhs_typed` + `apply_flux_correction` structure — the Berger-Colella register accumulation and topology freeze between `zero_regs` and `apply_flux_correction` are non-negotiable invariants.
3. New stage weights must sum to 1 for exact time integration and be passed to `tree_rhs_typed` to maintain flux register conservation.

---

## 7. Adaptive Mesh Refinement (AMR)

### 7.1 Octree Block Structure (`include/mesh/block_tree.hpp`)

Complete octree of `BlockNode`s. Only leaves own `CellBlock` storage. Internal nodes act as aggregators.

**Octant convention:** children at `first_child + oct`, where `oct = ix | (iy << 1) | (iz << 2)`.
**Leaf cache:** `leaf_indices()` returns a cached vector, invalidated on `refine()`/`coarsen()`/`rebuild_neighbours()`. `morton_leaf_indices()` additionally returns the leaf indices sorted by Morton code (cached, O(1) after first call; used by `tree_rhs_typed` for cache-friendly block ordering).

### 7.2 Regrid Protocol

```
advance() top:
  1. should_refine()  → tag based on |∇ρ|h/ρ > threshold (2nd-order centred)
  2. tree.refine(li)  → allocate 8 children, piecewise-constant prolong
  3. tree.balance()   → enforce 2:1 refinement ratio
  4. tree.rebuild_neighbours()
  5. GhostFiller::fill_all(tree, cfg.bc, cf_zero_grad=false)
  6. should_coarsen() → tag if |∇ρ|h/ρ < threshold/2
  7. tree.coarsen(li) → restrict_conservative + free children
```

```cpp
cfg.amr.max_level       = 2;    // 0 = flat (no AMR)
cfg.amr.regrid_interval = 5;    // steps between regrid calls (0 = disabled)
```

### 7.3 Prolongation: Coarse → Fine (`amr_operators.cpp::prolong_conservative`)

Piecewise-constant (0th-order): each fine cell takes its enclosing coarse cell value.
$$Q_f(i_f,j_f,k_f) = Q_c\!\left(\lfloor(i_f-\text{NG})/(N_B/2)\rfloor + \text{NG},\;\ldots\right)$$
Conserves mass/momentum/energy exactly (integral preserved). Only $O(h)$ — introduces discontinuity at the interface; higher-order prolongation is possible via the existing 5th-order C/F ghost fill infrastructure.

### 7.4 Restriction: Fine → Coarse (`amr_operators.cpp::restrict_conservative`)

Volume-weighted average of 8 fine cells:
$$Q_c(i_c,j_c,k_c) = \frac{1}{8}\sum_{d_x=0}^{1}\sum_{d_y=0}^{1}\sum_{d_z=0}^{1} Q_f(f_x+d_x, f_y+d_y, f_z+d_z)$$
The $1/8 = (h_f/h_c)^3$ factor is the volume ratio — preserves the integral of conserved quantities.

### 7.5 Berger-Colella Flux Correction

During SSP-RK3, each fine-leaf RHS evaluation accumulates into the flux register with the stage quadrature weight $w_s$:
$$R_f \mathrel{+}= w_s \cdot F_f^{(s)} \cdot \left(\frac{h_f}{h_c}\right)^2 \cdot \Delta t$$

After stage 3, the coarse cell correction is:
$$\Delta Q_c = \frac{\Delta t}{h_c}\!\left(\sum_{f\in\text{coarse face}} F_f^\text{fine} - F_c^\text{coarse}\right)$$

**Invariant:** Tree topology must not change between `zero_flux_registers()` and `apply_flux_correction()`. Enforced by `regrid()` running at the top of `advance()`.

### 7.6 Extension API — modifying AMR

- **Change refinement criterion:** Replace the gradient threshold in `should_refine()` (`amr_operators.cpp`) with any block-level scalar (e.g. vorticity, pressure gradient, level-set distance).
- **Higher-order prolongation:** Override `prolong_conservative()` with a slope-limited linear or WENO5 stencil. Must preserve the integral exactly (conservatism).
- **Refinement threshold at runtime:** Pass through `SolverConfig::AmrConfig`; `should_refine` reads `cfg.amr.refine_threshold` (add this field to `AmrConfig` if needed).

---

## 8. SGS Models

Both models use **operator-split**: inviscid + viscous RHS goes through SSP-RK3; SGS stress divergence is applied as an explicit correction after `apply_flux_correction()`.

### 8.1 SGS Interface (`include/models/sgs.hpp`)

```cpp
struct SGSModel {
    virtual void apply(CellBlock& blk, double h, double dt) const = 0;
    virtual ~SGSModel() = default;
};
cfg.physics.sgs = std::make_shared<SmagorinskyModel>(Cs);
cfg.physics.sgs = make_neural_sgs();       // Vreman fallback or ONNX neural
cfg.physics.sgs = nullptr;                 // no SGS (DNS or no-model LES)
```

### 8.2 Smagorinsky Model

$$\mu_t = \rho(C_s\Delta)^2|\bar{S}|, \quad \Delta = h, \quad C_s \approx 0.1$$
$$|\bar{S}| = \sqrt{2(S_{xx}^2+S_{yy}^2+S_{zz}^2+2S_{xy}^2+2S_{xz}^2+2S_{yz}^2)}$$

Applied via conservative face-centred divergence with face-averaged $\mu_t$ (telescoping → global momentum conservation). Wall-face $\mu_t = 0$ (Dirichlet) to avoid inflated SGS viscosity at no-slip walls.

### 8.3 Dynamic Smagorinsky Model (Germano 1991, Lilly 1992)

Test-filter (box, $\hat\Delta = 2\Delta$) + Germano identity + Lilly least-squares:
$$C_s^2 = \frac{\mathcal{L}_{ij}M_{ij}}{M_{ij}M_{ij}}, \quad C_s^2 \ge 0 \text{ (clipped)}$$
Test filter uses 3×3×3 box average — requires `NG ≥ 2` ghost layers.

### 8.4 Neural SGS Model (`include/models/neural_sgs.hpp`)

```cpp
std::shared_ptr<SGSModel> make_neural_sgs(const std::string& onnx_path = "",
                                           const std::string& fallback  = "vreman");
```

Wraps an ONNX Runtime inference session when `libonnxruntime-dev` is available and `USE_ONNXRUNTIME=ON` at CMake time. Falls back to the Vreman algebraic model otherwise. The Vreman model correctly predicts $\nu_t = 0$ in pure solid-body rotation (unlike Smagorinsky).

### 8.5 Extension API — adding a new SGS model

```cpp
struct MySGSModel : SGSModel {
    void apply(CellBlock& blk, double h, double dt) const override {
        // compute eddy viscosity from blk.prim(i,j,k)
        // accumulate stress divergence into blk.Q[1..3][idx]
    }
};
cfg.physics.sgs = std::make_shared<MySGSModel>(...);
```

The `apply()` call receives the block state after SSP-RK3 + flux correction. It should use the same conservative face-centred divergence form as `SmagorinskyModel` to ensure global momentum conservation on periodic domains.

---

## 9. Boundary Conditions

### 9.1 BCVariant Dispatch (`include/mesh/bc_types.hpp`)

```cpp
using BCVariant = std::variant<PeriodicBC, WallBC, OpenBC, ContactAngleBC>;

struct PeriodicBC { void fill_ghost(CellBlock& blk, int axis, int side); };
struct WallBC     { void fill_ghost(CellBlock& blk, int axis, int side); };
struct OpenBC     { void fill_ghost(CellBlock& blk, int axis, int side); };
struct ContactAngleBC {
    double theta_deg = 90.0;  // static contact angle [degrees]
    void fill_ghost(CellBlock& blk, int axis, int side);
};
```

Ghost fill is dispatched via `GhostFiller::fill_all(tree, cfg.bc.variant)`:
```cpp
std::visit([&](auto& bc){ bc.fill_ghost(blk, axis, side); }, cfg.bc.variant);
```

### 9.2 Periodic

Direct copy with domain wrap. Periodic neighbour determined by Morton index arithmetic in `rebuild_neighbours()`.

### 9.3 Wall (No-Slip, Adiabatic / Isothermal)

Anti-symmetric reflection of momentum; symmetric density and energy:
$$\rho_g = \rho_i, \quad (\rho\mathbf{u}_g)_n = -(\rho\mathbf{u}_i)_n, \quad (\rho\mathbf{u}_g)_t = -(\rho\mathbf{u}_i)_t, \quad E_g = E_i$$

Gives $\bar{u}_t^{(f)} = 0$ (no-slip) and $T_g = T_i$ (adiabatic).

**Isothermal wall:** Set `cfg.bc.wall_T > 0`. When set, the ghost energy is overridden so $T_g = 2 T_w - T_i$, enforcing $T_\text{face} = T_w$.

### 9.4 Open (Zero-Gradient)

$Q_g = Q_{i_\text{last interior}}$. Piecewise-constant extrapolation. Appropriate for subsonic outflows.

**GPU limitation:** GPU ghost fill kernels (`gpu_ghost_fill.cu`) currently implement periodic wrapping only. Wall and open BCs on GPU must be added before `cfg.exec.use_gpu = true` is compatible with those BC types.

### 9.5 Contact Angle BC (`ContactAngleBC`)

Used with ACDI multiphase (`cfg.acdi.use_acdi = true`). Sets the phase-field ghost so the interface meets the wall at the prescribed static contact angle $\theta_w$:
```cpp
cfg.bc.variant = ContactAngleBC{.theta_deg = 45.0};  // 45° partial wetting
cfg.acdi.acdi_ceps = 0.5;
```

### 9.6 Extension API — adding a new boundary condition

1. Create a new BC struct satisfying `BoundaryCondition`:
   ```cpp
   struct InletBC {
       Prim inlet_state;
       void fill_ghost(CellBlock& blk, int axis, int side) noexcept {
           // set ghost cells to inlet_state (or supersonic inlet extrapolation)
       }
   };
   static_assert(BoundaryCondition<InletBC>);
   ```
2. Add to `BCVariant`:
   ```cpp
   using BCVariant = std::variant<PeriodicBC, WallBC, OpenBC, ContactAngleBC, InletBC>;
   ```
3. Add `bc_to_int` mapping in `bc_types.hpp` for GPU dispatch.
4. Implement `fill_ghost` equivalent in `gpu_ghost_fill.cu` if GPU support is needed.

---

## 10. NSSolver: Full API Reference

### 10.1 Minimal Usage

```cpp
NSSolver solver;
solver.cfg.time.cfl   = 0.4;
solver.cfg.time.t_end = 1.0;
solver.cfg.bc.variant = WallBC{};
solver.cfg.physics.gamma = 1.4;

solver.init(L, [](double x, double y, double z) -> Prim {
    return Prim{1.0, 0.1, 0.0, 0.0, 2.5, 0.0, 0.0};  // ρ, u, v, w, p, T, (unused)
});

while (!solver.done()) {
    auto r = solver.advance_result();
    if (!r.ok()) { /* handle error */ break; }
}
```

### 10.2 SolverConfig Reference

| Field | Default | Effect |
|---|---|---|
| `exec.flux_scheme` | `HLLC_ES` | `HLLC` or `HLLC_ES`; selects instantiation from the explicit matrix |
| `exec.use_gpu` | `false` | Enables GpuPool + GpuGraphSolver path |
| `time.cfl` | `0.8` | Acoustic CFL; gate tests use 0.4 |
| `time.t_end` | `1.0` | Stop time |
| `bc.variant` | `PeriodicBC{}` | Any `BCVariant` alternative |
| `bc.wall_T` | `0.0` | > 0 → isothermal wall temperature [K] |
| `amr.max_level` | `2` | 0 = uniform (no AMR) |
| `amr.regrid_interval` | `0` | 0 = disabled |
| `amr.use_lts` | `false` | Berger-Oliger local time stepping |
| `physics.sgs` | `nullptr` | Shared pointer to SGSModel |
| `physics.use_imex` | `false` | Implicit viscous solve |
| `physics.gamma` | `1.4` | Overrides compile-time GAMMA |
| `numerics.ducros_p_threshold` | `0.1` | Raise to 0.5+ for shock-free flows |
| `numerics.sat_tau` | `0.0` | SBP-SAT penalty coefficient at C/F interfaces |
| `acdi.use_acdi` | `false` | ACDI compressible multiphase |
| `acdi.acdi_ceps` | `0.0` | Interface sharpening coefficient |
| `acdi.gamma_a/b` | `1.4` | Per-fluid γ for Allaire mixture EOS |
| `acdi.p_inf_a/b` | `0.0` | Per-fluid p∞ (0 = ideal gas) |

### 10.3 GPU Path

```cpp
#include "cuda/gpu_graph.cuh"
GpuGraphSolver gpu_solver;
solver.cfg.exec.use_gpu = true;
gpu_solver.build(solver.tree, bc_to_int(solver.cfg.bc.variant));
solver.set_gpu_solver(&gpu_solver);
solver.init(L, ic);
// advance() now routes entirely through CUDA Graph replay
```

---

## 11. GPU Implementation

### 11.1 Memory Model

**`GpuPool`** (`include/gpu_pool.hpp`): CUDA free-list allocator. Per-block allocation:
$$\text{NVAR}\times\text{NCELL}\times 8 = 5\times 1728\times 8 = 69120\text{ bytes} \approx 68\text{ KB}$$
Flat **SoA** on device: `d_Q[v * NCELL + flat]`. Different from host AoSoA; explicit up/download required.

`BlockTree` calls `on_block_alloc_` / `on_block_free_` callbacks that trigger `GpuPool::alloc(blk)` / `GpuPool::free(blk)` automatically during regrid.

### 11.2 GPU Kernel Pipeline (P8.2–P8.5)

Per leaf block per stage:
1. `k_prim_duc` — EOS inversion + Ducros sensor
2. `k_rhs_conv` — WENO5-Z + HLLC-ES hybrid, `atomicAdd` into `d_RHS`
3. `k_rhs_visc` — face-averaged stress tensor divergence + energy work

Stage updates: `k_rk3s1` (after stage 1), `k_rk3s23` (stages 2 & 3 with Shu-Osher coefficients).

**GPU CFL:** `k_cfl_reduce` uses warp-shuffle reduction with `atomicMin` on bit-reinterpreted `uint64` doubles — 8 bytes device-to-host per step.

### 11.3 CUDA Graph Capture/Replay (P8.6)

Three per-stage sub-graphs captured once and replayed for all subsequent steps:
- `graph_s1_`: `k_save_qn + ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s1`
- `graph_s2_`: `ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s23(3/4, 1/4)`
- `graph_s3_`: `ghost_fill + prim_duc + rhs_conv + rhs_visc + k_rk3s23(1/3, 2/3)`

`cudaMemsetAsync(d_rhs_pool, 0)` is done **outside** the graphs between stages (captured memset nodes proved unreliable across repeated replays in CUDA 13.x with Global capture mode).

**Topology change:** Any regrid destroys the three `cudaGraphExec_t` handles and triggers a rebuild on the next `advance()` call via `GpuGraphSolver::build()`.

### 11.4 Extension API — adding a new GPU kernel

1. Add the kernel in `src/cuda/gpu_rhs.cu`.
2. Register it as a node in `GpuRhsList::build()` in `gpu_graph.cu`.
3. Add a corresponding `cudaGraph` node before capture.
4. Invalidate and rebuild the graph on topology change.
5. Gate test: verify CPU and GPU results agree to `<1e-10` on a uniform state (follow the pattern of `t25`).

---

## 12. LiveStreamer

### 12.1 Overview (`include/live_streamer.hpp`)

In-situ browser streaming plugin. Attach by setting `solver.streamer_` before `advance()`. Runs a minimal HTTP server inside the solver process.

```cpp
#include "live_streamer.hpp"
auto streamer = std::make_shared<LiveStreamer>(18082);  // port
solver.streamer_ = streamer;
solver.init(L, ic);
// solver.advance() calls streamer->snapshot(tree) automatically
```

Open `http://localhost:18082` in a browser to see a live 2D cross-section. A WebSocket client can subscribe to the 3D volume stream.

### 12.2 Threading Model

Three background threads:
- `accept_thread_` — TCP accept loop; hands connections to stream threads.
- `stream_thread_` — 2D slice writer; wakes when `front_` is swapped.
- `stream3d_thread_` — 3D volume writer; activates only when a volume client connects.

**Double-buffer protocol:** Solver writes into `back_` with no lock during `snapshot()`. After writing: `swap_mtx_` is held for one O(1) `std::swap(back_, front_)`. Stream thread swaps `front_` into `work_` under `swap_mtx_`. No lock contention on the main loop.

### 12.3 Wire Formats

**2D frame:**
```
[4-byte LE length][32-byte header][n_blocks × 16-byte BlockDesc2D][n_blocks × NB² × float32]
```
Header: magic `0xCFD00001`, n_blocks, vmin, vmax, variable_id, timestamp.

**3D frame:**
```
[4-byte LE length][header][N³ × r32float volume]
```
LZ4-compressed when built with `HAVE_LZ4=1`.

**Streamed variables:** ρ, p, T, |u|, ρu, ρv, ρw, E — derived at snapshot time via `CellBlock::prim()`.

### 12.4 Extension API — adding a new streamed variable

1. Add an enum entry to the variable list in `live_streamer.hpp`.
2. In `LiveStreamer::write_frame()`, add a case that fills the float32 buffer with the new derived quantity.
3. Update the JavaScript client variable selector if a browser UI is in use.
4. No threading changes needed — the double-buffer protocol is variable-agnostic.

---

## 13. Baer-Nunziato Two-Phase Solver (BNSolver)

### 13.1 Overview (`include/models/bn_solver.hpp`)

The `BNSolver` integrates the **Baer-Nunziato (1986) five-equation model** for compressible two-phase flows with independent phase pressures and velocities. It uses an **external parallel vector pattern**: the BlockTree is borrowed from a parent NSSolver and `BNCellBlock` arrays shadow each leaf slot.

```
NVAR_BN = 7: (α₁ρ₁, α₂ρ₂, ρu, ρv, ρw, E, α₁)
NVAR_BN_CONS = 6: conservative part only (α₁ excludes from flux)
```

### 13.2 BNSolver Usage

```cpp
BNSolver bn;
bn.eos = BNEosParams{.gamma1=1.4, .gamma2=4.4, .p_inf2=6e8};
bn.init(tree, [](double x,double y,double z) -> std::array<double,7> {
    // return [α₁ρ₁, α₂ρ₂, ρu, ρv, ρw, E, α₁]
});

while (!done) {
    double dt = cfl_dt(tree, bn.Q);
    bn.advance(tree, dt, bc_variant);
}
```

### 13.3 SSP-RK3 + Berger-Colella for BN

`BNSolver::advance()` follows the same SSP-RK3 structure as NSSolver:
1. For each stage: call `bn_rhs(tree, Q, rhs, eos)` → BN Riemann flux + source terms.
2. Accumulate `bn_accumulate_cf_correction_fluxes(tree, Qs_, regs_, stage_weight, eos)`.
3. After all stages: call `bn_apply_flux_correction(tree, Q, regs_, dt)`.

**BN11 Berger-Colella:** The correction flux is `sw × (F_fine − F_coarse_ghost)` accumulated per stage, applied once after RK3. The coarse block RHS is **never modified** during stages — only the post-RK3 correction touches it.

### 13.4 Ghost Fill and AMR

- Same-level: `bn_fill_ghosts(tree, blocks, bc)` — direct copy from neighbour's interior.
- C/F: piecewise-constant fine←coarse, 2×2 average coarse←fine (BN08).
- Prolong/restrict: `bn_prolong` / `bn_restrict` — exact mass conservation (BN09, BN10).

### 13.5 Extension API

- **New EOS pair:** Add fields to `BNEosParams` and update `bn_prim()` to compute partial pressures.
- **New BN flux:** Follow the same `template<int NVAR_BN>` + accumulate_face pattern in `bn_operators.cpp`.
- **Coupling to NSSolver:** BNSolver shares the `BlockTree` reference. For coupled problems, advance BNSolver and NSSolver with the same `dt` on the same tree.

---

## 14. Multi-Species Chemistry (`include/models/chemistry.hpp`)

### 14.1 Model

Operator-split finite-rate chemistry (Poinsot & Veynante 2005). Species mass fractions $Y_k$ are stored in a `SpeciesBlock` that parallels `CellBlock`. Hydrodynamics (NVAR=5) advances first; chemistry updates $Y_k$ and internal energy in a separate sub-step.

**Reaction rate (modified Arrhenius):**
$$k_f = A\,T^n\,\exp(-E_a/(R_u T)), \quad R_u = 8314.46\;\text{J/(kmol·K)}$$

**Source term:**
$$\dot\omega_k = \sum_r (\nu_{k,r}'' - \nu_{k,r}')\,W_k\,k_{f,r}\prod_j [X_j]^{\nu_{j,r}'}$$

### 14.2 Usage

```cpp
#include "models/chemistry.hpp"
ChemMechanism mech;
mech.add_species({"H2",  2.016,  0.0,       14307.0});
mech.add_species({"O2",  32.0,   0.0,        918.0});
mech.add_species({"H2O", 18.015, -2.418e7,  1864.0});
mech.add_reaction(Reaction{.A=1e12, .n=0, .Ea=1.7e8,
                            .spec={0,1,2}, .nu_r={1,0.5,0}, .nu_p={0,0,1}});

// In time loop, after ns_solver.advance():
apply_chemistry_block(blk, sblk, mech, dt);
```

### 14.3 IMEX Sub-step

`apply_chemistry_block` uses implicit Euler with one Newton iteration for stiff reactions:
$$Y^{n+1} - Y^n = dt\,\dot\omega(Y^{n+1}, T^{n+1})$$
Linearise once around $Y^n$, solve the $(N_\text{spec}+1)\times(N_\text{spec}+1)$ system. Good for moderately stiff H₂/O₂ chemistry; for highly stiff mechanisms (detailed CNOX, large hydrocarbons) use VODE or an exponential integrator.

---

## 15. Ghost-Cell Immersed Boundary Method (`include/models/ibm.hpp`)

### 15.1 Method (Mittal & Iaccarino 2005)

Cells classified as `FLUID`, `SOLID`, or `IBM_GHOST` using a signed-distance level-set $\phi > 0$ in fluid.

For each `IBM_GHOST` cell G:
1. Find image point $I = G - 2\phi_G \mathbf{n}$ (mirror across surface).
2. Trilinear interpolation of $Q$ at $I$ from surrounding fluid cells.
3. Apply wall BC: $u_\text{ghost} = 2u_\text{wall} - u_\text{image}$.

### 15.2 Usage

```cpp
#include "models/ibm.hpp"

// Define geometry via level-set
SphereLevelSet sphere({0.5, 0.5, 0.5}, 0.2);  // centre + radius

// Classify cells (once, or after regrid)
classify_cells(blk, sphere, blk.x0, blk.h);

IBMConfig cfg;
cfg.wall_bc  = IBMWallBC::NoSlip;    // or Adiabatic, Isothermal
cfg.wall_T   = 300.0;                // only for Isothermal

// In time loop, before tree_rhs:
apply_ibm(blk, sphere, cfg);
```

### 15.3 Adding a new geometry

Subclass `LevelSet` and implement `phi(x,y,z)`. Override `normal()` for exact normals (avoids finite-difference approximation errors near the surface):

```cpp
class CylinderLevelSet : public LevelSet {
    double cx, cy, R;
public:
    double phi(double x, double y, double z) const noexcept override {
        return std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)) - R;
    }
};
```

---

## 16. Wall-Modelled LES (`include/models/wall_model.hpp`)

### 16.1 Model Options

`WallModelCfg` selects between two sub-models:

```cpp
struct WallModelCfg {
    double nu;        // kinematic viscosity [m²/s]
    double y_m;       // wall-normal height of matching point [m]
    bool   use_ode = false;  // false → algebraic Reichardt; true → ODE TBLE
    int    ode_pts = 128;    // integration points for ODE model
};
```

**Algebraic (default):** Reichardt composite law-of-the-wall. Newton solve for $u_\tau$:
$$u^+ = \frac{1}{\kappa}\ln(1 + \kappa y^+) + C_1\!\left(1 - e^{-y^+/C_2} - \frac{y^+}{C_2}e^{-y^+/C_3}\right)$$

**ODE mixing-length (TBLE):** Integrates the thin-boundary-layer equation with equilibrium mixing-length closure. More accurate for $y^+ \gg 100$ but ~128× more expensive per wall cell.

### 16.2 Usage

```cpp
WallModelCfg wmcfg{.nu = 1.5e-5, .y_m = 0.005};

// In time loop, replace wall ghost fill with WMLES ghost fill:
wm_apply_wall(blk, wall_axis, wall_side, nu, wmcfg);
```

`wm_apply_wall` sets the ghost velocity such that the face-averaged stress equals $\tau_w = \rho u_\tau^2$ in the direction of the wall-parallel velocity. The wall-normal ghost velocity is set to zero (no penetration).

### 16.3 Extension API

- **Non-equilibrium effects:** Replace `wm_log_law` with a custom function accepting pressure gradient and streamwise history; the rest of `wm_apply_wall` is independent of the wall-stress model.
- **Rough walls:** Add a roughness length $z_0$ to `WallModelCfg` and modify the log-law: $u^+ = \kappa^{-1}\ln(y/z_0)$.

---

## 17. SBP-SAT Penalty at AMR Interfaces

### 17.1 Overview

An optional SBP-SAT penalty can be added at C/F interfaces alongside the Berger-Colella flux correction. The penalty weakly enforces the coarse-fine solution continuity:

$$\sigma_c = -\frac{\tau}{h_f}(Q_\text{coarse,face} - Q_\text{fine,face})$$

applied per RK3 stage. Configurable via:
```cpp
cfg.numerics.sat_tau = 0.5;  // 0.0 = disabled (Berger-Colella only)
```
A value of $\tau = 0.5$ is the minimum energy-stable penalty for a standard 2nd-order SBP operator. Set $\tau = 0$ (default) for pure Berger-Colella correction. Gate test: `t9` (`test_sat`) verifies the penalty is globally conservative for all five variables.

---

## 18. ACDI Compressible Multiphase

### 18.1 Overview (P14.1)

The **Accurate Conservative Diffuse Interface** (ACDI) method advects a phase indicator $\phi \in [0,1]$ alongside the conservative variables. $\phi = 1$ is fluid A, $\phi = 0$ is fluid B.

### 18.2 Usage

```cpp
cfg.acdi.use_acdi  = true;
cfg.acdi.acdi_ceps = 0.5;     // interface sharpening (ε = Cε·h)
cfg.acdi.gamma_a   = 6.12;    // liquid water
cfg.acdi.p_inf_a   = 3.43e8;
cfg.acdi.gamma_b   = 1.4;     // air
cfg.acdi.p_inf_b   = 0.0;

// IC: initialise φ field
solver.init(L, [](double x, double y, double z) -> Prim { return ...; },
               [](double x, double y, double z) -> double {
                   return (x < 0.5) ? 1.0 : 0.0;  // phase-field IC
               });
```

The Allaire mixture EOS automatically weights $\gamma_m$ and $p_{\infty,m}$ by $\phi$.

### 18.3 Contact Angle

```cpp
cfg.bc.variant = ContactAngleBC{.theta_deg = 60.0};
```

The contact angle BC sets ghost phase-field values so the interface meets the wall at $\theta_w$. Requires `use_acdi = true` and `acdi_ceps > 0`.

---

## 19. Checkpointing and I/O

### 19.1 Checkpoint round-trip

```cpp
solver.save_checkpoint("state.cfd");   // writes binary header + NB2³ × NVAR × float64
solver.load_checkpoint("state.cfd");   // restores Q, t, step
```

### 19.2 ZFP and Float32 Compression (`tests/test_checkpoint_zfp.cpp`)

Lossy compression modes enabled at compile time:
- **Float32 (F32):** ~2× size reduction, relative error ≤ 1.5×10⁻⁷.
- **ZFP rate=16:** ~4× size reduction, relative error ≤ 10⁻⁴.

Load from file auto-detects the compression mode from the header (no explicit mode argument needed at load time — BZ04).

### 19.3 VTK Output

```cpp
solver.write_vtk("frame_0000.vts");  // structured VTK XML, readable by ParaView/VisIt
```

---

## References

| Citation | Title | Used in |
|---|---|---|
| Borges et al. (2008) | An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws | WENO5-Z (§4.5) |
| Chandrashekar (2013) | Kinetic energy preserving and entropy stable finite volume schemes | HLLC-ES (§4.6) |
| Pirozzoli (2011) | Numerical methods for high-speed flows. Annu. Rev. Fluid Mech. | KEP flux (§4.4) |
| Ducros et al. (1999) | Large-eddy simulation of the shock/turbulence interaction | Ducros sensor (§4.3) |
| Toro (2009) | Riemann Solvers and Numerical Methods for Fluid Dynamics, §10.4 | HLLC solver (§4.7) |
| Jiang & Shu (1996) | Efficient implementation of weighted ENO schemes | WENO-JS indicators (§4.5) |
| Shu & Osher (1988) | Efficient implementation of essentially non-oscillatory shock-capturing schemes | SSP-RK3 (§6.1) |
| Kennedy & Carpenter (2003) | Additive Runge-Kutta schemes for convection-diffusion-reaction equations | IMEX-ARK (§6.3) |
| Berger & Colella (1989) | Local adaptive mesh refinement for shock hydrodynamics | Flux registers (§7.5) |
| Berger & Oliger (1984) | Adaptive mesh refinement for hyperbolic PDEs | LTS sub-cycling (§6.4) |
| McCorquodale & Colella (2011) | High-order finite-volume method for conservation laws on locally refined grids | C/F 5th-order fill (§3.3) |
| Germano et al. (1991) | A dynamic subgrid-scale eddy viscosity model | Dynamic Smagorinsky (§8.3) |
| Lilly (1992) | A proposed modification of the Germano subgrid-scale closure method | Lilly LS contraction (§8.3) |
| Batten et al. (1997) | On the choice of wavespeeds for the HLLC Riemann solver | HLLC wave speeds (§4.7) |
| Ismail & Roe (2009) | Affordable, entropy-consistent Euler flux functions II | Log-mean numerics (§4.6) |
| Mittal & Iaccarino (2005) | Immersed boundary methods. Annu. Rev. Fluid Mech. 37:239 | IBM (§15) |
| Poinsot & Veynante (2005) | Theoretical and Numerical Combustion, 2nd ed. | Chemistry (§14) |
| Baer & Nunziato (1986) | A two-phase mixture theory for the deflagration-to-detonation transition | BN solver (§13) |
| Allaire et al. (2002) | A five-equation model for the simulation of interfaces between compressible fluids | ACDI EOS (§18) |
| Kokkos mdspan (P0009) | Reference implementation, github.com/kokkos/mdspan (Apache-2.0) | Vendor mdspan (§3.2) |
| Svärd & Gjesteland (2025) | Entropy-stable boundary conditions for compressible NS | SAT penalty (§17) |
| Zhang & Shu (2010) | Positivity-preserving high order DG schemes for compressible Euler | Positivity floor |
