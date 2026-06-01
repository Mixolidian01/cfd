# CFD Solver — Technical Reference

> Updated: 2026-06-01  
> Covers: governing equations, numerics, GPU architecture, and developer APIs for every component.  
> For phase status and commit history, see `docs/dev_phases.md` and `docs/dev_log.md`.

---

## 1. Governing Equations

The solver integrates the **3D compressible Navier-Stokes equations** in conservation form:

$$
\frac{\partial \mathbf{Q}}{\partial t} + \frac{\partial \mathbf{F}_i}{\partial x_i} = \frac{\partial \mathbf{G}_i}{\partial x_i} + \mathbf{S}
$$

where $\mathbf{Q}$ is the vector of conserved variables, $\mathbf{F}_i$ are the inviscid (convective) flux components, $\mathbf{G}_i$ are the viscous flux components, and $\mathbf{S}$ is the source term vector (body force, chemistry, radiation, SGS).

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

This flux vector is **isotropic under permutation of axes** — the mathematical justification for `template<Axis DIR>`.

### 1.3 Viscous Flux

$$
\mathbf{G}_i = \begin{pmatrix} 0 \\ \tau_{i1} \\ \tau_{i2} \\ \tau_{i3} \\ \tau_{ij}u_j + q_i \end{pmatrix}
$$

where $\tau_{ij} = \mu\!\left(\partial u_i/\partial x_j + \partial u_j/\partial x_i - \frac{2}{3}\delta_{ij}\nabla\cdot\mathbf{u}\right)$ and $q_i = -\kappa\partial T/\partial x_i$ (Pr = 0.72).

### 1.4 Body Force

A constant body acceleration $\mathbf{f} = (f_x, f_y, f_z)$ adds to momentum and energy:

$$
\mathbf{S}_\text{body} = \begin{pmatrix} 0 \\ f_x\rho \\ f_y\rho \\ f_z\rho \\ f_x(\rho u) + f_y(\rho v) + f_z(\rho w) \end{pmatrix}
$$

Configured via `SolverConfig::PhysicsConfig::body_force[3]` (default `{0,0,0}`). On the GPU path, `set_body_force(fx, fy, fz)` must be called before `build()`. For body-force-driven channel flow, see §16.3.

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
Computed as `ratio * sqrt(ratio)` (~10× faster than `pow`). Temperature floor: `T = max(T, 1.0)` before `sqrt` in both CPU and GPU paths (prevents NaN near rarefactions).

### 2.4 Extension API — adding a new EOS

1. Create `include/physics/my_eos.hpp`:
   ```cpp
   struct MyEOS {
       Prim cons_to_prim(double rho, double rhou, double rhov, double rhow, double E) const noexcept;
   };
   static_assert(EquationOfState<MyEOS>);
   ```
2. Add an explicit instantiation row in `src/schemes/operators.cpp`.
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

**AoSoA memory layout (CPU):**
```
data_[tile * NVAR * W + v * W + lane]
  tile = flat >> 3        (flat / 8)
  lane = flat & 7         (flat % 8)
```
All `tile_ptr(v, t)` pointers are 64-byte aligned — enables AVX-512 autovectorisation.

**GPU SoA layout on device:** `d_Q[v * NCELL + flat]` — separate from host AoSoA; requires explicit upload/download.

**Cell index convention:**
$$\text{flat}(i,j,k) = k\cdot\text{NB2}^2 + j\cdot\text{NB2} + i$$

Interior range: $i,j,k \in [2, 9]$. Ghosts: $[0,1]$ and $[10,11]$.

**Rectangular domains:** `BlockTree::init(Lx, Ly, Lz, NX, NY, NZ)` creates a NX×NY×NZ forest of root blocks. Each `CellBlock` carries per-axis cell sizes `(h, hy, hz)`.

### 3.2 R6 mdspan Axis Views

`CellBlock::axis_view<DIR>(int v)` returns an `md::mdspan` over 3D extents `[NB2][NB2][NB2]` with an `AoSoAAccessor`. The layout strides encode axis rotation:

| Axis | stride[0] (normal n) | stride[1] (tangential a) | stride[2] (tangential b) |
|---|---|---|---|
| X | 1 | NB2 | NB2² |
| Y | NB2 | 1 | NB2² |
| Z | NB2² | 1 | NB2 |

### 3.3 Ghost Fill Protocols

**Same-level faces:** Direct copy from the neighbouring block's interior cells.

**Coarse-fine (C/F) faces:** `fill_cf_ghosts()` in `amr_operators.cpp` applies 5th-order Lagrange interpolation in the normal direction.

**Unified dispatch:** `GhostFiller::fill_all(tree, bc_variant, cf_zero_grad)` accepts a `BCVariant` and calls `std::visit`. Use this at every ghost fill site.

#### C/F Ghost Fill: 5th-Order Lagrange

5-cell stencil at offsets $\{-4,-3,-2,-1,0\}$:

$$
q_f^{(\text{gl}=0)} = \sum_{k=0}^{4} L_k^+ \, q_c^{(i_0-4+k)}, \quad
q_f^{(\text{gl}=1)} = \sum_{k=0}^{4} L_k^- \, q_c^{(i_0-4+k)}
$$

Coefficients: $L^+ = \tfrac{1}{6144}\{585,-3060,6630,-7956,9945\}$, $L^- = \tfrac{1}{6144}\{-231,1260,-2970,4620,3465\}$

---

## 4. Convective Operator

### 4.1 Face-Centred Flux Loop (`src/schemes/convective_rhs.cpp`)

$(N_B+1)\times N_B\times N_B = 576$ faces per axis, 1728 total per block. Each face evaluated **once**. Sign convention: flux $F$ leaves the left cell.

$$\text{rhs}[\text{left}] \mathrel{-}= h^{-1}F, \quad \text{rhs}[\text{right}] \mathrel{+}= h^{-1}F$$

### 4.2 Hybrid Scheme: Ducros Sensor + KEP/WENO5-ES Blend

$$\mathbf{F} = (1-\theta)\,\mathbf{F}_\text{KEP} + \theta\,\mathbf{F}_\text{WENO5-ES}$$

$\theta = \max(\Phi_L, \Phi_R)$ from the combined Ducros-pressure sensor.
Fast path: when $\theta < 10^{-8}$ only $\mathbf{F}_\text{KEP}$ is computed; WENO5 reconstruction is skipped.

### 4.3 Ducros + Pressure-Ratio Shock Sensor

**Velocity-based (Ducros 1999):**
$$\Phi_\text{vel} = \frac{(\nabla\cdot\mathbf{u})^2}{(\nabla\cdot\mathbf{u})^2 + |\nabla\times\mathbf{u}|^2 + \varepsilon}$$

**Pressure-ratio sensor:**
$$\Phi_p = \max_d\frac{|p_{i\pm 1} - p_i|}{p_i + \varepsilon}$$

Linear ramp: $\Phi_p < \tau \to 0$; $\Phi_p > \tau+w \to 1$. Configurable via `SolverConfig::NumericsConfig`:
```cpp
cfg.numerics.ducros_p_threshold = 0.1;   // τ: ramp lower bound
cfg.numerics.ducros_blend_width  = 0.1;   // w: ramp width
```
Raise `ducros_p_threshold` to ≥ 0.5 for DNS/LES without shocks.

### 4.4 KE-Preserving Flux — Pirozzoli (2011)

Arithmetic-mean primitive variables: $\bar\rho$, $\bar{u}_i$, $\bar{p}$, $\bar{H}$.
$$F_\text{KEP}[0]=\bar\rho\bar{u}_n,\quad F_\text{KEP}[\text{mom}_i]=\bar\rho\bar{u}_n\bar{u}_i+\bar{p}\delta_{ni},\quad F_\text{KEP}[E]=\bar\rho\bar{u}_n\bar{H}$$

Mass flux uses Subbareddy & Candler (2009) form: $\bar\rho\bar{u}_n = \tfrac{1}{2}(\rho_L u_{n,L} + \rho_R u_{n,R})$.

### 4.5 WENO5-Z Reconstruction with Characteristic Decomposition

**Scalar WENO-Z** (Borges et al. 2008): three 3rd-order candidates + global smoothness indicator $\tau_5 = |\beta_0 - \beta_2|$.
$$\alpha_k = d_k\!\left(1+\frac{\tau_5^2}{(\beta_k+\varepsilon)^2}\right), \quad \varepsilon=10^{-36}, \quad (d_0,d_1,d_2)=(0.1,0.6,0.3)$$

**Characteristic decomposition:** Roe-averaged state → project 6-cell stencil to characteristic space → apply WENO-Z independently → back-project.

**Safe fallback:** If reconstructed ρ ≤ 0 or p ≤ 0, fall back to cell-center primitive.

*Note: WENO5-Z is the CPU default. On GPU, TENO7-A is the default — see §4.6.*

### 4.6 GPU Reconstruction Schemes

The GPU path (`GpuRhsList`, `gpu_rhs.cu`) dispatches reconstruction via `GpuReconScheme` enum:

| Scheme | Enum | Order | Notes |
|--------|------|-------|-------|
| TENO7-A | `TENO7A` | 7th | **GPU default**; Fu et al. (2019); FP64-compute-bound on RTX 3070 |
| TENO5-A | `TENO5A` | 5th | Lower FLOP count; faster on low-FP64 hardware |
| WENO5-Z | `WENO5Z` | 5th | CPU=GPU parity gate (t25 uses this) |

```cpp
solver.rhs_list.scheme = GpuReconScheme::TENO7A;  // set before build()
```

The CPU path always uses WENO5-Z + HLLC-ES hybrid (§4.2). See `docs/tech_debt.md` §"k_rhs_conv roofline gap" for performance notes on RTX 3070 vs A100/H100.

### 4.7 Entropy-Stable HLLC-ES Flux — Chandrashekar (2013)

Entropy-conservative base flux uses **log-mean** density and temperature:
$$\hat\rho = \text{logmean}(\rho_L,\rho_R), \quad \hat\beta = \text{logmean}(\beta_L,\beta_R), \quad \beta = \rho/(2p)$$

Entropy-stable flux adds scalar LF dissipation:
$$\mathbf{F}^{ES} = \mathbf{F}^{EC} - \tfrac{\lambda_\text{max}}{2}\Delta\mathbf{Q}, \quad \lambda_\text{max} = \max(|u_{n,L}|+c_L, |u_{n,R}|+c_R)$$

**Numerically stable log-mean:** For $|f|^2 < 10^{-4}$ uses Taylor expansion to avoid cancellation.

**Wall face detection:** `is_wall_ghost(pL, pR)` detects anti-symmetric momentum and substitutes $\mathbf{F}_\text{KEP}$ to prevent LF dissipation from draining tangential momentum at no-slip walls.

### 4.8 Extension API — adding a new Riemann flux

1. Create `include/physics/my_flux.hpp` satisfying concept `RiemannFlux`.
2. Add explicit instantiations in `src/schemes/operators.cpp`.
3. Add `MyFlux` to `SolverConfig::FluxScheme` enum and the dispatch in `cpu_rk3.cpp`.

---

## 5. Viscous Operator

### 5.1 Conservative Divergence Form (`src/schemes/viscous_rhs.cpp`)

Face-averaged viscosity:
$$\mu_{i+\frac{1}{2}} = \tfrac{1}{2}(\mu_i + \mu_{i+1})$$

All nine velocity gradient components at faces computed by `VelocityGradAtFace<Axis, Order>` from `include/physics/diff_ops.hpp`.

### 5.2 Energy Equation Viscous Term

Conservative face-flux form:
$$F_e|_{x+1/2} = \tau_{xx}\bar{u} + \tau_{xy}\bar{v} + \tau_{xz}\bar{w} + \kappa\,h^{-1}(T_{i+1}-T_i)$$

### 5.3 C/F Viscous Energy Reflux

At C/F interfaces, `undo_cf_viscous_energy` replaces the coarse flux with the fine-based estimate via `cf_visc_energy_flux<AX>`, applying the same Berger-Colella correction as the convective reflux.

---

## 6. Time Integration

### 6.1 SSP-RK3: Shu-Osher Form

$$\mathbf{Q}^{(1)} = \mathbf{Q}^n + \Delta t\,\mathcal{L}(\mathbf{Q}^n)$$
$$\mathbf{Q}^{(2)} = \tfrac{3}{4}\mathbf{Q}^n + \tfrac{1}{4}\mathbf{Q}^{(1)} + \tfrac{1}{4}\Delta t\,\mathcal{L}(\mathbf{Q}^{(1)})$$
$$\mathbf{Q}^{n+1} = \tfrac{1}{3}\mathbf{Q}^n + \tfrac{2}{3}\mathbf{Q}^{(2)} + \tfrac{2}{3}\Delta t\,\mathcal{L}(\mathbf{Q}^{(2)})$$

Equivalent Butcher weights: $w_1 = 1/6$, $w_2 = 1/6$, $w_3 = 2/3$.

**Regrid ordering:** `regrid()` runs at the **top** of `advance()`, on $\mathbf{Q}^n$, before zeroing flux registers. Regridding after `apply_flux_correction()` allows `coarsen()` to overwrite flux-corrected cells (~2.69e-8 mass leak if violated).

### 6.2 CFL Condition

$$\Delta t = \text{CFL} \cdot \min_\text{leaves} \frac{h}{|u_n| + c}$$

Both acoustic and viscous CFL conditions are enforced: $\Delta t_\text{visc} = h^2 / (2 C_\text{visc} \max_i(\mu_i/\rho_i))$ where $C_\text{visc} = \max(4/3, \gamma/\text{Pr})$.

```cpp
cfg.time.cfl   = 0.4;   // gate tests; use 0.03 for WMLES channel
cfg.time.t_end = 10.0;
```

### 6.3 IMEX-ARK: Implicit Viscous Solve

Enabled via `cfg.physics.use_imex = true`. Explicit convective RHS stays on GPU stream; implicit viscous Helmholtz solve uses GPU-resident GMRES (D4, gate t32).

### 6.4 Local Time Stepping

Enabled via `cfg.amr.use_lts = true`. Berger-Oliger sub-cycling; fine levels take multiple RK3 steps per coarse step. GPU-native via `GpuLtsList` (G5, gate t47).

### 6.5 Adjoint: Reversed SSP-RK3

`NSSolver::adjoint_step(lambda_arrays)` reverses the three Shu-Osher stages in order 3→2→1 using stored checkpoints (`Qs0_`, `Qs1_`, `Qs2_`). Exposed to Python via pybind11 and to JAX via `custom_vjp` (D10/D11, gates t38/t39/t41).

---

## 7. Adaptive Mesh Refinement (AMR)

### 7.1 Octree Block Structure

Complete octree of `BlockNode`s. Only leaves own `CellBlock` storage.

**Octant convention:** children at `first_child + oct`, where `oct = ix | (iy << 1) | (iz << 2)`.

**Leaf cache:** `leaf_indices()` — dirty-flag cache; `morton_leaf_indices()` returns Morton-sorted order (cache-friendly block ordering).

### 7.2 Regrid Protocol

```
advance() top:
  1. should_refine()  → tag based on |∇ρ|h/ρ > threshold
  2. tree.refine(li)  → allocate 8 children, piecewise-constant prolong
  3. tree.balance()   → enforce 2:1 refinement ratio
  4. tree.rebuild_neighbours()
  5. GhostFiller::fill_all(tree, cfg.bc, cf_zero_grad=false)
  6. should_coarsen() → tag if |∇ρ|h/ρ < threshold/2
  7. tree.coarsen(li) → restrict_conservative + free children
```

### 7.3 Prolongation: Coarse → Fine

Piecewise-constant (0th-order): conserves mass/momentum/energy exactly.

### 7.4 Restriction: Fine → Coarse

Volume-weighted average of 8 fine cells:
$$Q_c(i_c,j_c,k_c) = \frac{1}{8}\sum_{d_x=0}^{1}\sum_{d_y=0}^{1}\sum_{d_z=0}^{1} Q_f(f_x+d_x, f_y+d_y, f_z+d_z)$$

### 7.5 Berger-Colella Flux Correction

During SSP-RK3, fine-leaf RHS evaluation accumulates into flux register with stage weight $w_s$:
$$R_f \mathrel{+}= w_s \cdot F_f^{(s)} \cdot \left(\frac{h_f}{h_c}\right)^2 \cdot \Delta t$$

After stage 3:
$$\Delta Q_c = \frac{\Delta t}{h_c}\!\left(\sum_{f\in\text{coarse face}} F_f^\text{fine} - F_c^\text{coarse}\right)$$

**Invariant:** Tree topology must not change between `zero_flux_registers()` and `apply_flux_correction()`.

---

## 8. SGS Models

Both models use **operator-split**: SGS stress divergence applied after `apply_flux_correction()`.

### 8.1 SGS Interface

```cpp
cfg.physics.sgs = std::make_shared<SmagorinskyModel>(Cs);
cfg.physics.sgs = nullptr;   // DNS or no-model LES
```

### 8.2 Smagorinsky Model

$$\mu_t = \rho(C_s\Delta)^2|\bar{S}|, \quad \Delta = h, \quad C_s \approx 0.1$$

Wall-face $\mu_t = 0$ to avoid inflated SGS viscosity at no-slip walls.

### 8.3 Dynamic Smagorinsky Model (Germano 1991, Lilly 1992)

Test-filter (box, $\hat\Delta = 2\Delta$) + Germano identity + Lilly least-squares:
$$C_s^2 = \frac{\mathcal{L}_{ij}M_{ij}}{M_{ij}M_{ij}}, \quad C_s^2 \ge 0 \text{ (clipped)}$$

Test filter uses 3×3×3 box average. GPU-native via `GpuDynSgsList` (G3, gate t45).

### 8.4 Neural SGS Model

```cpp
std::shared_ptr<SGSModel> make_neural_sgs(const std::string& onnx_path = "",
                                           const std::string& fallback  = "vreman");
```

ONNX Runtime when available; falls back to Vreman algebraic model (correctly predicts $\nu_t = 0$ in solid-body rotation).

---

## 9. Boundary Conditions

### 9.1 BCVariant Dispatch

```cpp
using BCVariant = std::variant<PeriodicBC, WallBC, OpenBC, NscbcBC, ContactAngleBC>;
```

Dispatched via `GhostFiller::fill_all(tree, cfg.bc.variant)`:
```cpp
std::visit([&](auto& bc){ bc.fill_ghost(blk, axis, side); }, cfg.bc.variant);
```

### 9.2 Periodic

Direct copy with domain wrap via Morton index arithmetic.

### 9.3 Wall (No-Slip, Adiabatic / Isothermal)

Anti-symmetric reflection of momentum; symmetric density and energy.
**Isothermal wall:** Set `cfg.bc.wall_T > 0`. Ghost energy overridden so $T_\text{face} = T_w$.

### 9.4 Open (Zero-Gradient)

$Q_g = Q_{i_\text{last interior}}$. Piecewise-constant extrapolation. Use `NscbcBC` (§9.5) for non-reflecting outflows.

### 9.5 NSCBC: Navier-Stokes Characteristic Boundary Conditions (D9)

```cpp
struct NscbcBC {
    double p_inf  = 1.0;   // far-field static pressure
    double L_ref  = 1.0;   // domain length scale for relaxation
    double sigma  = 0.28;  // incoming-wave relaxation (Poinsot & Lele 1992)
    void fill_ghost(CellBlock& blk, int axis, int side);
};
cfg.bc.variant = NscbcBC{.p_inf = 1.0, .L_ref = 1.0, .sigma = 0.28};
```

Decomposes the boundary state into characteristic waves (Thompson 1987; Poinsot & Lele 1992). Subsonic outflow: prescribes `p_inf`, extrapolates density and velocity, damps only the incoming acoustic wave with relaxation factor $\sigma(1-M^2)c/L_\text{ref}$. Supersonic faces: zero-gradient (all waves outgoing).

Gate: `t40` — reflected amplitude ≤ 1% of incident (vs ~20% for `OpenBC`).

GPU: `k_fill_faces` handles `bc_type=3` in the same device dispatch as other BC types.

### 9.6 Contact Angle BC

Used with ACDI multiphase. Sets phase-field ghost so the interface meets the wall at prescribed static contact angle $\theta_w$:
```cpp
cfg.bc.variant = ContactAngleBC{.theta_deg = 45.0};
```

### 9.7 Extension API

1. Create a BC struct satisfying `BoundaryCondition` concept.
2. Add to `BCVariant` in `bc_types.hpp`.
3. Add `bc_to_int` mapping for GPU dispatch.
4. Implement `fill_ghost` equivalent in `gpu_ghost_fill.cu`.

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
    return Prim{1.0, 0.1, 0.0, 0.0, 2.5, 0.0, 0.0};
});

while (!solver.done()) {
    auto r = solver.advance_result();
    if (!r.ok()) { /* handle error */ break; }
}
```

### 10.2 SolverConfig Reference

| Field | Default | Effect |
|---|---|---|
| `exec.flux_scheme` | `HLLC_ES` | `HLLC` or `HLLC_ES` |
| `exec.use_gpu` | `false` | Routes `advance()` through GpuGraphSolver |
| `time.cfl` | `0.8` | Acoustic CFL; use 0.03–0.4 for WMLES |
| `time.t_end` | `1.0` | Stop time |
| `bc.variant` | `PeriodicBC{}` | Any `BCVariant` alternative |
| `bc.wall_T` | `0.0` | > 0 → isothermal wall temperature [K] |
| `amr.max_level` | `2` | 0 = uniform (no AMR) |
| `amr.regrid_interval` | `0` | 0 = disabled |
| `amr.use_lts` | `false` | Berger-Oliger local time stepping |
| `physics.sgs` | `nullptr` | Shared pointer to SGSModel |
| `physics.use_imex` | `false` | Implicit viscous solve (D4) |
| `physics.gamma` | `1.4` | Overrides compile-time GAMMA |
| `physics.body_force[3]` | `{0,0,0}` | Body force [fx,fy,fz]; call `set_body_force()` on GPU path |
| `physics.wmles_enabled` | `false` | Wall-modelled LES; configure via `physics.wmles_cfg` |
| `physics.combustion_enabled` | `false` | Single-step Arrhenius (D5); `physics.arrhenius` holds `ArrheniusParams` |
| `physics.radiation.enabled` | `false` | P1 diffusion radiation (D6); configure `RadiationParams` |
| `numerics.ducros_p_threshold` | `0.1` | Raise to 0.5+ for shock-free flows |
| `numerics.sat_tau` | `0.0` | SBP-SAT penalty at C/F interfaces |
| `acdi.use_acdi` | `false` | ACDI compressible multiphase |
| `acdi.acdi_ceps` | `0.0` | Interface sharpening coefficient |
| `acdi.gamma_a/b` | `1.4` | Per-fluid γ for Allaire mixture EOS |
| `acdi.p_inf_a/b` | `0.0` | Per-fluid p∞ (0 = ideal gas) |

For the complete list of fields, see `include/solver/ns_solver.hpp`.

### 10.3 GPU Path

```cpp
#include "cuda/gpu_graph.cuh"
GpuGraphSolver gpu_solver;
solver.cfg.exec.use_gpu = true;
gpu_solver.set_body_force(fx, fy, fz);   // if body force needed
gpu_solver.build(solver.tree, bc_to_int(solver.cfg.bc.variant));
solver.set_gpu_solver(&gpu_solver);
solver.init(L, ic);
// advance() now routes entirely through CUDA Graph replay
```

---

## 12. LiveStreamer

### 12.1 Overview (`include/live_streamer.hpp`)

In-situ browser streaming plugin.

```cpp
#include "live_streamer.hpp"
auto streamer = std::make_shared<LiveStreamer>(18082);  // port
solver.streamer_ = streamer;
solver.init(L, ic);
```

Open `http://localhost:18082`. WebSocket client can subscribe to 3D volume stream.

### 12.2 Threading Model

Three background threads: `accept_thread_`, `stream_thread_` (2D slice), `stream3d_thread_` (3D volume). Double-buffer protocol: solver writes `back_` without lock; `swap_mtx_` held for O(1) `std::swap(back_, front_)`.

### 12.3 Wire Formats

**2D frame:** `[4-byte LE length][32-byte header][n_blocks × 16-byte BlockDesc2D][n_blocks × NB² × float32]`

**3D frame:** `[4-byte LE length][header][N³ × r32float volume]` — LZ4-compressed when built with `HAVE_LZ4=1`.

**Streamed variables:** ρ, p, T, |u|, ρu, ρv, ρw, E, Mach, vorticity, Q-criterion, schlieren.

---

## 13. Baer-Nunziato Two-Phase Solver (BNSolver)

### 13.1 Overview

`BNSolver` integrates the Baer-Nunziato (1986) model for compressible two-phase flows.

```
NVAR_BN = 7: (α₁ρ₁, α₂ρ₂, ρu, ρv, ρw, E, α₁)
```

### 13.2 Usage

```cpp
BNSolver bn;
bn.eos = BNEosParams{.gamma1=1.4, .gamma2=4.4, .p_inf2=6e8};
bn.init(tree, [](double x,double y,double z) -> std::array<double,7> { ... });

while (!done) {
    double dt = cfl_dt(tree, bn.Q);
    bn.advance(tree, dt, bc_variant);
}
```

GPU-native via `GpuBnList` (G6, gate t48).

### 13.3 SSP-RK3 + Berger-Colella for BN

Same SSP-RK3 structure as NSSolver. BN Berger-Colella (BN11): correction flux `sw × (F_fine − F_coarse_ghost)` accumulated per stage, applied once after RK3.

---

## 14. Multi-Species Chemistry (`include/models/chemistry.hpp`)

### 14.1 Model

Operator-split finite-rate chemistry. Reaction rate (modified Arrhenius):
$$k_f = A\,T^n\,\exp(-E_a/(R_u T)), \quad R_u = 8314.46\;\text{J/(kmol·K)}$$

### 14.2 IMEX Sub-step

`apply_chemistry_block` uses implicit Euler with one Newton iteration for stiff reactions.

---

## 15. Ghost-Cell Immersed Boundary Method

### 15.1 Method (Mittal & Iaccarino 2005)

Cells classified as `FLUID`, `SOLID`, or `IBM_GHOST`.

**CPU path:** LevelSet subclass provides $\phi(x,y,z)$ (positive = fluid). Sign at cell centres comes directly from the user-supplied analytic function.

**GPU path:** `bvh_sdf()` in `include/cuda/gpu_bvh.cuh` uses BVH traversal for closest-point distance, with **generalised winding number** (Van Oosterom & Strackee 1983) for sign determination — robust for non-convex and non-watertight STL meshes. Sign rule: winding number > 0.5 → inside → SOLID (sdf < 0). See `docs/tech_debt.md` for performance notes.

For each `IBM_GHOST` cell G:
1. Find image point $I = G - 2\phi_G \mathbf{n}$ (mirror across surface).
2. Trilinear interpolation of $Q$ at $I$ from surrounding fluid cells.
3. Apply wall BC: $u_\text{ghost} = 2u_\text{wall} - u_\text{image}$.

### 15.2 GPU Usage (STL meshes)

```cpp
#include "models/stl_loader.hpp"
#include "cuda/gpu_bvh.cuh"

StlMesh mesh = load_stl("geometry.stl");
GpuBvh bvh; bvh.build(mesh);
gpu_solver.set_gpu_ibm(&bvh, /*bc=*/0, /*uw,vw,ww,Tw=*/0,0,0,0);
gpu_solver.build(tree, bc_int);
```

Gate t49 (I5–I9) and winding-number gate W5 (non-convex torus geometry).

### 15.3 CPU Usage (analytic level-set)

```cpp
SphereLevelSet sphere({0.5, 0.5, 0.5}, 0.2);
classify_cells(blk, sphere, blk.x0, blk.h);
IBMConfig cfg; cfg.wall_bc = IBMWallBC::NoSlip;
apply_ibm(blk, sphere, cfg);
```

### 15.4 Adding a New Geometry (CPU)

Subclass `LevelSet` and implement `phi(x,y,z)`. Override `normal()` for exact normals.

---

## 16. Wall-Modelled LES

### 16.1 Model Options

```cpp
struct WallModelCfg {
    double nu;        // kinematic viscosity [m²/s]
    double y_m;       // wall-normal height of matching point [m]
    bool   use_ode = false;  // false → algebraic Reichardt; true → ODE TBLE
    int    ode_pts = 128;
};
```

**Algebraic (default):** Reichardt composite law-of-the-wall. Newton solve for $u_\tau$:
$$u^+ = \frac{1}{\kappa}\ln(1 + \kappa y^+) + C_1\!\left(1 - e^{-y^+/C_2} - \frac{y^+}{C_2}e^{-y^+/C_3}\right)$$

**ODE mixing-length (TBLE):** Van Driest damping + Picard iteration for u_τ. GPU-native via extended `GpuWmlesList` (G4, gate t46).

### 16.2 Usage

```cpp
WallModelCfg wmcfg{.nu = 1.5e-5, .y_m = 0.005};
wm_apply_wall(blk, wall_axis, wall_side, nu, wmcfg);
```

`wm_apply_wall` sets ghost velocity so face-averaged stress equals $\tau_w = \rho u_\tau^2$.

### 16.3 Body-Force-Driven Channel Flow

For turbulent channel flow driven by a body force:

```cpp
cfg.physics.body_force[0] = 1.0;   // f_x = u_τ²/h = 1.0 for u_τ=1, h=1
cfg.physics.wmles_enabled  = true;
cfg.time.cfl               = 0.03; // acoustic CFL ≤ 0.03 for WMLES stability
```

Gate t50 (C50): Re_τ=395, Lx=2π, Ly=2, Lz=4π/3, 64 leaves; 500-step spin-up + 500-step statistics; log-law intercept B ∈ [4.9, 6.2] at y⁺ ∈ [30, 200] (measured B ≈ 5.6, consistent with Reichardt composite law). DNS reference: Lee & Moser (2015) Re_τ=395.

---

## 17. SBP-SAT Penalty at AMR Interfaces

An optional SBP-SAT penalty weakly enforces coarse-fine solution continuity:

$$\sigma_c = -\frac{\tau}{h_f}(Q_\text{coarse,face} - Q_\text{fine,face})$$

```cpp
cfg.numerics.sat_tau = 0.5;  // 0.0 = disabled (Berger-Colella only)
```

$\tau = 0.5$ is the minimum energy-stable penalty for a standard 2nd-order SBP operator. Gate: `t9` verifies global conservation for all five variables.

---

## 18. ACDI Compressible Multiphase

The **Accurate Conservative Diffuse Interface** (ACDI) method advects a phase indicator $\phi \in [0,1]$ alongside the conservative variables. GPU-native via `GpuAcdiList` (G1, gate t43).

```cpp
cfg.acdi.use_acdi  = true;
cfg.acdi.acdi_ceps = 0.5;     // interface sharpening (ε = Cε·h)
cfg.acdi.gamma_a   = 6.12;    // liquid water
cfg.acdi.p_inf_a   = 3.43e8;
cfg.acdi.gamma_b   = 1.4;     // air
cfg.bc.variant = ContactAngleBC{.theta_deg = 60.0};  // optional wall BC
```

---

## 19. Checkpointing and I/O

### 19.1 Checkpoint round-trip

```cpp
solver.save_checkpoint("state.cfd");   // binary header + NB2³ × NVAR × float64
solver.load_checkpoint("state.cfd");   // restores Q, t, step
```

### 19.2 Compression

- **Float32:** ~2× size reduction, relative error ≤ 1.5×10⁻⁷.
- **ZFP rate=16:** ~4× size reduction, relative error ≤ 10⁻⁴.

Load auto-detects compression mode from header.

### 19.3 VTK Output

```cpp
solver.write_vtk("frame_0000.vts");  // structured VTK XML for ParaView/VisIt
```

---

## GPU Subsystem Architecture

> Scope: GPU subsystem only. The CPU layer (linalg → cell_block → operators → ns_solver) is documented above.

---

### G1. Solver Layers

```
┌─────────────────────────────────────────────────────────────┐
│  apps/simulate_gpu.cu   apps/simulate.cpp                   │  Entry points
├─────────────────────────────────────────────────────────────┤
│  IGpuSolver  (include/solver/ns_solver.hpp)                 │  Interface
│  └─ GpuGraphSolver  (include/cuda/gpu_graph.cuh)            │  Concrete impl
├───────────────────────┬─────────────────────────────────────┤
│  GpuPool              │  GpuSnapshotBuffer                  │  Device memory
│  (gpu_pool.hpp/.cu)   │  (gpu_snapshot.hpp/.cu)             │
├───────────────────────┴─────────────────────────────────────┤
│  "List" objects — one per physics subsystem (see §G3)       │  Physics lists
├─────────────────────────────────────────────────────────────┤
│  __global__ kernels  (src/cuda/*.cu)                        │  CUDA kernels
├─────────────────────────────────────────────────────────────┤
│  Device helpers  (include/cuda/gpu_constants.cuh, gpu_bvh.cuh, ...) │
├─────────────────────────────────────────────────────────────┤
│  CellBlock SoA  (include/mesh/cell_block.hpp)               │  Data layout
│  BlockTree      (include/mesh/block_tree.hpp)               │
└─────────────────────────────────────────────────────────────┘
```

---

### G2. CellBlock Memory Layout

```
NB=8  NG=2  NB2=12  NCELL=1728(=12³)  NVAR=5  GAMMA=1.4  R_GAS=287.058

data_[NTILE * NVAR * W]          // 216 * 5 * 8 = 8640 doubles = 69 120 B
  NTILE = NCELL/W = 216
  W = 8 (AVX-512 lanes)
  index: tile*(NVAR*W) + var*W + lane
         tile = flat >> 3,  lane = flat & 7
  vars:  [0]=ρ  [1]=ρu  [2]=ρv  [3]=ρw  [4]=E

phi_data_[NCELL]                 // flat double array, separate from Q

Interior range per axis:  ilo()=2 … ihi()=9  (ghost layers at 0,1 and 10,11)
Flat index: k*NB2² + j*NB2 + i   (i fastest)
```

On device, Q is stored as: `d_Q[v * NCELL + flat]` — same layout pointer into `GpuPool`.

---

### G3. The "List" Pattern

Every physics subsystem follows the same three-phase protocol:

```
build(tree, pool, …)   — called from GpuGraphSolver::build(); allocates device
                          arrays, classifies topology, uploads metadata
exec(stream)           — called every RK3 stage; launches kernels
~destructor            — cudaFree all device arrays
```

| List | Header | Source | Owns (device) | Notes |
|------|--------|--------|---------------|-------|
| `GpuGhostFillList` | gpu_ghost_fill.cuh | gpu_ghost_fill.cu | `GpuLeafGhostMeta* d_metas` | Same-level + C/F ghost fill; 4 build() overloads for BC/MPI variants |
| `GpuRhsList` | gpu_rhs.cuh | gpu_rhs.cu | `GpuLeafRhsMeta* d_metas`, `double* d_scratch_pool`, `double* d_rhs_pool` | k_prim_duc → k_rhs_conv_teno (TENO7-A default) → k_rhs_visc → k_body_force; `exec(s, zero_rhs)` |
| `GpuCflList` | gpu_cfl.cuh | gpu_cfl.cu | `GpuLeafCflMeta* d_metas`, `ull* d_dt_bits`, `double* d_dt` | Returns dt via warp-shuffle reduction |
| `GpuCfList` | gpu_cf.cuh | gpu_cf.cu | `GpuCfCoarseMeta* d_coarse`, `GpuCfFineMeta* d_fine`, `double* d_reg_pool` | Berger-Colella flux registers |
| `GpuSgsList` | gpu_sgs.cuh | gpu_sgs.cu | `GpuSgsMeta* d_metas` | Static Smagorinsky operator-split |
| `GpuDynSgsList` | gpu_sgs.cuh | gpu_sgs.cu | `GpuDynSgsMeta* d_metas`, `double* d_scratch` | Germano+Lilly dynamic SGS |
| `GpuMpiHaloList` | gpu_mpi_halo.cuh | gpu_mpi_halo.cu | `vector<FaceEntry>`, `vector<RankBuf>` | Face-pack D2H→MPI→H2D per stage |
| `GpuAcdiList` + `GpuPhiPool` | gpu_acdi.cuh | gpu_acdi.cu | `GpuAcdiLeafMeta* d_metas`, phi pools | ACDI phi transport |
| `GpuIbmList` | gpu_ibm.cuh | gpu_ibm.cu | `GpuIbmMeta* d_metas`, `int8_t* d_cell_type_pool`, `float* d_sdf/wnorm_pool`, `GhostEntry* d_ghosts/d_solid_fills` | STL ghost-cell IBM; winding-number BVH sign |
| `GpuAmrList` | gpu_amr.cuh | gpu_amr.cu | prolong/restrict meta arrays | Built transiently inside gpu_regrid() |

---

### G4. GpuGraphSolver Anatomy

**Defined in:** `include/cuda/gpu_graph.cuh`, `src/cuda/gpu_graph.cu`

#### 4.1 Fields (selected)

```cpp
// ── Physics lists ────────────────────────────────────────────────────────────
GpuGhostFillList ghost_list;
GpuRhsList       rhs_list;          // force_x_/y_/z_ fields for body force
GpuCflList       cfl_list;
GpuCfList        cf_list;
GpuSgsList       sgs_list;
GpuDynSgsList    dyn_sgs_list_;
GpuMpiHaloList   mpi_halo_;
GpuAcdiList      acdi_list_;  GpuPhiPool phi_pool_;
GpuIbmList       ibm_list_;

// ── Feature flags ─────────────────────────────────────────────────────────────
bool   acdi_enabled_ = false;   double acdi_ceps_ = 0.0;
bool   ibm_enabled_  = false;   GpuBvh* ibm_bvh_ptr_ = nullptr;
bool   sgs_enabled   = false;   double sgs_Cs_ = 0.16;
bool   dyn_sgs_enabled_ = false;
double force_x_ = 0.0;  double force_y_ = 0.0;  double force_z_ = 0.0;

// ── Per-leaf RK3 state ────────────────────────────────────────────────────────
GpuRk3LeafMeta* d_rk3_metas = nullptr;
double*          d_Qn_pool   = nullptr;
cudaStream_t    stream      = nullptr;
cudaGraphExec_t graph_s1/s2/s3 = nullptr;
bool            graph_valid = false;
```

#### 4.2 Lifecycle

```
Constructor → create stream
set_gpu_*()  → configure feature flags (call BEFORE build())
set_body_force(fx, fy, fz) → propagates to rhs_list fields
build()      → rebuild all lists, allocate RK3 state, invalidate graphs
advance()    → choose path (§G5), return dt
download_q() → D2H copy after advance() if CPU access needed
```

---

### G5. Advance Loop — Three Paths

#### 5.1 Graph replay path (fast; no AMR C/F faces, no MPI, no ACDI, no IBM)

```
advance()
├─ cfl_list.exec(cfl, stream) → dt
├─ [Stage 1]
│   cudaMemsetAsync(d_rhs_pool, 0, stream)
│   cudaGraphLaunch(graph_s1, stream)   ← k_save_qn + ghost_fill + rhs + k_rk3s1 + positivity
├─ [Stage 2/3]  (same pattern with Shu-Osher weights)
├─ [Optional] SGS operator-split
└─ cudaStreamSynchronize(stream)
```

Graphs captured once after first explicit step. Graph capture **skipped** when:
`mpi_halo_.active() || acdi_enabled_ || dyn_sgs_enabled_ || ibm_enabled_`

#### 5.2 Explicit path (first step, or any feature blocking capture)

```
_run_rk3_explicit(stream):
  for stage in {s1, s2, s3}:
    cudaMemsetAsync(d_rhs_pool, 0, stream)
    mpi_halo_.exchange(stream)
    ghost_list.exec(stream)
    ibm_list_.exec(stream)       [IBM only]
    rhs_list.exec(stream, false) [includes k_body_force if force ≠ 0]
    k_rk3s1 / k_rk3s23
    k_positivity_floor
```

#### 5.3 AMR C/F path (`_advance_amr`; active when `cf_list.n_coarse > 0`)

```
_advance_amr(cfl):
  cfl_list.exec → dt
  cf_list.zero_regs(stream)
  for stage in {s1, s2, s3}:
    [ghost + rhs + cf_list.undo/accum + rk3 + positivity]
  cf_list.apply_correction(stream, dt)
  [Optional] SGS operator-split
```

#### 5.4 Decision tree

```
advance()
  └─ n_leaves == 0?  → return 1e300
  └─ cf_list.n_coarse > 0?  → _advance_amr()
  └─ graph_valid && !mpi_halo_.active()?  → graph replay
  └─ else  → _run_rk3_explicit(); [capture graphs if conditions allow]
```

---

### G6. IBM Subsystem Detail

```
GpuBvh  (include/cuda/gpu_bvh.cuh + src/cuda/gpu_bvh.cu)
  build(StlMesh)  — CPU median-split BVH, upload nodes + triangle SoA to device
  d_nodes         — BvhNode flat array (root at index 0)
  d_v{0,1,2}{x,y,z}, d_n{x,y,z}  — triangle geometry SoA

  __device__ bvh_sdf(nodes, v0x..v2z, tnx..tnz, n_tris, px,py,pz, &nx,&ny,&nz) → float
    Iterative DFS, stack[64]; returns signed distance + outward wall normal.
    Sign determined by generalised winding number (Van Oosterom & Strackee 1983):
      sum solid angles over all n_tris triangles; wn > 0.5 → inside (SOLID, sdf < 0).
    Robust for non-convex and non-watertight STL meshes.

GpuIbmList  (include/cuda/gpu_ibm.cuh + src/cuda/gpu_ibm.cu)
  build(tree, pool, bvh)
    1. k_ibm_classify: bvh_sdf per cell → d_cell_type_pool (FLUID/SOLID/IBM_GHOST)
    2. Pass 1: GhostEntry for each IBM_GHOST; image point I = G − 2·sdf·n_outward
    3. Pass 2: GhostEntry for each SOLID; sdf_eff = max(sdf, −1.5h) (depth clamp)

  exec(stream)
    1. k_ghost_fill_ibm(d_solid_fills) — SolidFill first
    2. k_ghost_fill_ibm(d_ghosts)      — ghost fill second

GhostEntry:
  ghost_ptr    — base of ghost cell in d_Q
  stencil[8]   — trilinear stencil cell bases
  w[8]         — trilinear weights
  wall_bc      — 0=NoSlip+Adiab, 2=Isotherm, 3=SolidFill
  u/v/w/T_wall
```

---

### G7. File Index (GPU subsystem)

#### Headers (`include/cuda/`)

| File | Defines | Notes |
|------|---------|-------|
| `gpu_constants.cuh` | `GPU_NB/NG/NCELL/NVAR/GAMMA`; `gpu_cell_idx()`, `gpu_ilo/ihi()`, `gpu_sutherland()` | Include in every kernel file |
| `gpu_pool.hpp` | `GpuPool` | Device memory arena; maps CellBlock*→double* |
| `gpu_ghost_fill.cuh` | `GpuLeafGhostMeta`, `GpuGhostFillList` | Same-level + C/F BC; bc_types 0/1/2/3 |
| `gpu_rhs.cuh` | `GpuLeafRhsMeta`, `GpuRhsList`, `GpuReconScheme` enum | WENO5-Z / TENO5-A / TENO7-A dispatch; body force fields |
| `gpu_cfl.cuh` | `GpuLeafCflMeta`, `GpuCflList` | CFL + viscous CFL reduction |
| `gpu_cf.cuh` | `GpuCfCoarseMeta`, `GpuCfFineMeta`, `GpuCfList` | Berger-Colella registers |
| `gpu_sgs.cuh` | `GpuSgsMeta`, `GpuSgsList`, `GpuDynSgsMeta`, `GpuDynSgsList` | Static + dynamic Smagorinsky |
| `gpu_mpi_halo.cuh` | `GpuMpiHaloList`, `FaceEntry`, `RankBuf` | Face-pack GPU-buffer MPI halos |
| `gpu_acdi.cuh` | `GpuPhiPool`, `GpuAcdiLeafMeta`, `GpuAcdiList` | ACDI phi transport |
| `gpu_bvh.cuh` | `BvhNode`, `GpuBvh`; `bvh_sdf()` (device inline) | STL BVH; winding-number sign |
| `gpu_ibm.cuh` | `GpuIbmMeta`, `GhostEntry`, `GpuIbmList` | IBM ghost-cell lists |
| `gpu_amr.cuh` | `GpuProlongMeta`, `GpuRestrictMeta`, `GpuSensorMeta`, `GpuAmrList` | Prolongation/restriction kernels |
| `gpu_snapshot.hpp` | `GpuSnapshotBuffer`, `GpuBlockMetrics`, `SnapLeafMeta` | Slice extraction + metrics |
| `gpu_graph.cuh` | `GpuRk3LeafMeta`, `GpuGraphSolver` | Master solver; includes all above |
| `gpu_adjoint_rhs.cuh` | adjoint RHS kernel decl | D10 |
| `gpu_bn.cuh` | Baer-Nunziato two-phase | G6 |
| `gpu_lts.cuh` | Berger-Oliger LTS | G5 |
| `gpu_gmres.cuh` | GPU GMRES | D4 |
| `gpu_p1.cuh` | P1 radiation | D6 |
| `gpu_source.cuh` | Arrhenius + species | D5 |
| `gpu_wmles.cuh` | WMLES algebraic + ODE wall model | D7, G4 |

#### Sources (`src/cuda/`)

Each `.cu` implements its matching `.cuh`. Notable extras:

| File | Notes |
|------|-------|
| `gpu_graph.cu` | `GpuGraphSolver` — build, advance, _capture_graphs, _advance_amr |
| `gpu_pool.cu` | `GpuPool` alloc/free/upload/download |
| `gpu_snapshot.cu` | Slice + metric reduction kernels |
| `gpu_imex.cu` | `advance_imex()` — IMEX-Euler (linked only for t32) |

---

### G8. Key Structs Quick Reference

```cpp
// ── GPU per-leaf metadata ─────────────────────────────────────────────────────
// GpuRk3LeafMeta (gpu_graph.cuh)
  double* d_Q;    // current state (in/out)
  double* d_Qn;   // Q^n checkpoint
  const double* d_RHS;

// GpuIbmMeta (gpu_ibm.cuh)
  double* d_Q;
  int8_t* d_cell_type;  // [1728]  0=FLUID 1=SOLID 2=IBM_GHOST
  float*  d_sdf;        // [1728]  signed distance (+=fluid)
  float*  d_wnx/y/z;   // [1728]  outward wall normals
  float   ox,oy,oz, hx,hy,hz;

// GhostEntry (gpu_ibm.cuh)
  double*  ghost_ptr;     // base of ghost cell's d_Q
  double*  stencil[8];    // trilinear stencil cell bases
  float    w[8];          // trilinear weights
  uint8_t  wall_bc;       // 0=NoSlip+Adiab, 2=Isotherm, 3=SolidFill
  float    u/v/w/T_wall;

// BvhNode (gpu_bvh.cuh)
  float aabb_min[3], aabb_max[3];
  int   left;   // >=0: child idx; <0: leaf, tri=~left
  int   right;
```

---

### G9. IGpuSolver Interface

```cpp
struct IGpuSolver : TimeIntegrator {
    // Core
    virtual double advance(const BlockTree&, double cfl) = 0;
    virtual void   build(const BlockTree&, const GpuPool&, int bc_type=0) = 0;
    virtual void   download_q(const BlockTree&) const = 0;
    virtual void   upload_q() const = 0;

    // Physics (all default no-op)
    virtual void set_gpu_sgs(double Cs, double Pr_t) {}
    virtual void set_gpu_dyn_sgs(double Pr_t) {}
    virtual void set_gpu_acdi(double ceps) {}
    virtual void set_gpu_ibm(GpuBvh*, uint8_t bc,
                             float uw, float vw, float ww, float Tw) {}
    virtual void set_ducros(double p_thr, double blend_inv) {}
    virtual void set_mpi(MpiPartition*) {}
    virtual void set_snapshot_buffer(GpuSnapshotBuffer*) {}
    virtual void set_body_force(double fx, double fy, double fz) {}
};
```

`set_gpu_*()` and `set_body_force()` must be called **before** `build()`.

---

### G10. Gate Test Map (GPU)

| Target | Gate | What it tests |
|--------|------|---------------|
| t24 | P8.6 | CUDA Graph re-capture on regrid |
| t25 | P9.1 | GPU vs CPU solution correctness |
| t26 | P10-A3 | NSSolver GPU dispatch |
| t27 | P-SGS-GPU | Static Smagorinsky |
| t28 | P-MPI-GPU | MPI halo exchange |
| t29 | D1 | GPU-native AMR (no D2H for Q) |
| t30 | D2 | CUDA-aware MPI halos |
| t31 | D3a | TENO5-A reconstruction |
| t32 | D4 | GPU GMRES (IMEX) |
| t33 | D5 | Arrhenius reactive flow |
| t34 | D6 | P1 radiation |
| t35 | D7 | WMLES algebraic wall model |
| t36 | — | GPU snapshot (slice + metrics) |
| t37 | D3b | TENO7-A reconstruction |
| t38 | D10 | Adjoint dot-product test |
| t39 | D11 | Python NSSolver bindings |
| t40 | D9 | NSCBC outflow/inflow BC |
| t41 | D11 | JAX custom_vjp wiring |
| t42 | D11 | Adjoint RK3 + rectangular NS |
| t43 | G1 | ACDI phi transport |
| t44 | G2 | Adjoint convective RHS (GPU) |
| t45 | G3 | Dynamic Smagorinsky (Germano+Lilly) |
| t46 | G4 | ODE mixing-length wall model |
| t47 | G5 | Berger-Oliger LTS |
| t48 | G6 | Baer-Nunziato two-phase |
| t49 | I5–I9, W5 | IBM STL import + ghost-cell BC (winding-number sign) |
| t50 | C50 | Turbulent channel WMLES (Re_τ=395, body force) |

Note: t28, t30 are MPI-gated (`HAVE_MPI`). GPU targets are **not** in the `ba` target.

---

### G11. Build System Sketch

```cmake
# _GPU_NS = full GPU solver source set
set(_GPU_NS
  src/cuda/gpu_graph.cu  src/cuda/gpu_ghost_fill.cu  src/cuda/gpu_rhs.cu
  src/cuda/gpu_cfl.cu    src/cuda/gpu_pool.cu         src/cuda/gpu_cf.cu
  src/cuda/gpu_sgs.cu    src/cuda/gpu_mpi_halo.cu     src/cuda/gpu_amr.cu
  src/cuda/gpu_snapshot.cu  src/cuda/gpu_acdi.cu
  src/cuda/gpu_bvh.cu    src/cuda/gpu_ibm.cu          src/models/stl_loader.cpp
  src/solver/ns_solver.cpp  src/mesh/block_tree.cpp  ...
)
# add_nvcc_gate(TARGET tNN BIN tNN_xxx_gpu SRCS test_tNN.cu ${_GPU_NS})
# simulate_gpu links _GPU_NS
```

---

## References

| Citation | Title | Used in |
|---|---|---|
| Borges et al. (2008) | An improved weighted essentially non-oscillatory scheme | WENO5-Z (§4.5) |
| Fu et al. (2019) | A family of high-order targeted ENO schemes | TENO7-A / TENO5-A (§4.6) |
| Chandrashekar (2013) | Kinetic energy preserving and entropy stable finite volume schemes | HLLC-ES (§4.7) |
| Pirozzoli (2011) | Numerical methods for high-speed flows | KEP flux (§4.4) |
| Subbareddy & Candler (2009) | A fully discrete, kinetic energy consistent finite-volume scheme | Mass flux (§4.4) |
| Ducros et al. (1999) | Large-eddy simulation of the shock/turbulence interaction | Ducros sensor (§4.3) |
| Shu & Osher (1988) | Efficient implementation of essentially non-oscillatory schemes | SSP-RK3 (§6.1) |
| Kennedy & Carpenter (2003) | Additive Runge-Kutta schemes | IMEX-ARK (§6.3) |
| Berger & Colella (1989) | Local adaptive mesh refinement for shock hydrodynamics | Flux registers (§7.5) |
| Berger & Oliger (1984) | Adaptive mesh refinement for hyperbolic PDEs | LTS sub-cycling (§6.4) |
| McCorquodale & Colella (2011) | High-order finite-volume method for conservation laws | C/F 5th-order fill (§3.3) |
| Germano et al. (1991) | A dynamic subgrid-scale eddy viscosity model | Dynamic Smagorinsky (§8.3) |
| Lilly (1992) | A proposed modification of the Germano subgrid-scale closure | Lilly LS contraction (§8.3) |
| Thompson (1987) | Time-dependent boundary conditions for hyperbolic systems | NSCBC (§9.5) |
| Poinsot & Lele (1992) | Boundary conditions for direct simulations of compressible flows | NSCBC (§9.5) |
| Mittal & Iaccarino (2005) | Immersed boundary methods. Annu. Rev. Fluid Mech. | IBM (§15) |
| Van Oosterom & Strackee (1983) | The solid angle of a plane triangle. IEEE Trans. Biomed. Eng. | IBM winding number sign (§G6) |
| Baer & Nunziato (1986) | A two-phase mixture theory for the deflagration-to-detonation transition | BN solver (§13) |
| Allaire et al. (2002) | A five-equation model for the simulation of interfaces | ACDI EOS (§18) |
| Poinsot & Veynante (2005) | Theoretical and Numerical Combustion, 2nd ed. | Chemistry (§14) |
| Ismail & Roe (2009) | Affordable, entropy-consistent Euler flux functions II | Log-mean numerics (§4.7) |
| Svärd & Gjesteland (2025) | Entropy-stable boundary conditions for compressible NS | SAT penalty (§17) |
| Zhang & Shu (2010) | Positivity-preserving high order DG schemes | Positivity floor |
| Lee & Moser (2015) | Direct numerical simulation of turbulent channel flow up to Reτ ≈ 5200 | Channel WMLES validation (§16.3) |
| Kokkos mdspan (P0009) | Reference implementation (Apache-2.0) | Vendor mdspan (§3.2) |
