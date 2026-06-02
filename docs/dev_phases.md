# Development Phase Status

Quick-reference status for every planned development phase.
Detailed narratives are in `docs/dev_log.md`.

Legend: ✅ DONE · 🚫 DROPPED · ⬜ TODO

---

## Baseline

| Phase | Title | Status | Gate | Commit | Date |
|-------|-------|--------|------|--------|------|
| D0 | Baseline verification | ✅ DONE | t1–t28 | 28f1f88 | 2026-05-16 |
| D0.5 | Shared-memory tiling for k_rhs_conv | ✅ DONE | t24–t28 | ca08faf / 029ae79 | 2026-05-17/19 |

> D0.5 note: tiled kernel implemented and verified correct; exec() reverted to original after profiling showed 1.5× regression (latency-bound, not BW-bound at 4.2% peak). `k_rhs_conv_tiled` retained in `gpu_rhs.cu` as reference.

---

## Core GPU Infrastructure

| Phase | Title | Status | Gate | Commit | Date |
|-------|-------|--------|------|--------|------|
| D1 | GPU-native AMR | ✅ DONE | t29 | 0d5fc2c | 2026-05-17 |
| D2 | CUDA-aware MPI halo exchange | ✅ DONE | t28 / t30 | e32fa9f / c0b8d0c | 2026-05-17/19 |
| D3 | TENO7-A reconstruction | ✅ DONE | t31 / t37 | 13b817f / bc188b1 | 2026-05-17/25 |
| D4 | GPU-resident GMRES (IMEX viscous) | ✅ DONE | t32 | dda0d77 | 2026-05-17 |

> D2 note: CUDA-aware MPI path compiled (`#ifdef MPIX_CUDA_AWARE_SUPPORT`) but inactive on WSL2/OpenMPI; CPU-staging fallback with 6× smaller face-only buffers (1440 vs 8640 doubles/face).

---

## Advanced Physics

| Phase | Title | Status | Gate | Commit | Date |
|-------|-------|--------|------|--------|------|
| D5 | Reactive flows (single-step Arrhenius) | ✅ DONE | t33 | c723756 | 2026-05-17 |
| D6 | P1 radiation transport | ✅ DONE | t34 | 3f0749c | 2026-05-18 |
| D7 | Wall-modelled LES (algebraic) | ✅ DONE | t35 | 9335d7d | 2026-05-18 |
| D8 | H100 TMA + thread-block clusters | 🚫 DROPPED | — | — | — |

> D8 permanently dropped. Never implement.

---

## Numerics & Adjoints

| Phase | Title | Status | Gate | Commit | Date |
|-------|-------|--------|------|--------|------|
| D9 | NSCBC outflow/inflow BC | ✅ DONE | t40 | e460b97 | 2026-05-25 |
| D10 | Discrete adjoint of SSP-RK3 + HLLC-ES | ✅ DONE | t38 | f1f5638 | 2026-05-26 |
| D11 | Python bindings (pybind11) + JAX wiring | ✅ DONE | t39 / t41 | f62cf95 / 86bb47a | 2026-05-26/28 |

> D10 includes: adjoint_hllc, adjoint_teno7, adjoint_rhs (CPU), dot-product test to 1e-10.
> D11 includes: NSSolver pybind11 (t39), adjoint_step binding, JAX custom_vjp (t41), RK3 checkpoint adjoint (t42_adjoint_rk3).

---

## GPU Gaps (backfill of CPU-only subsystems)

These were not in the original CLAUDE.md phase list; implemented 2026-05-30.

| Phase | Title | Status | Gate | Commit | Date |
|-------|-------|--------|------|--------|------|
| G1 | GPU ACDI phi transport | ✅ DONE | t43 | 0bac1ea | 2026-05-30 |
| G2 | GPU adjoint convective RHS | ✅ DONE | t44 | d8d066b | 2026-05-30 |
| G3 | GPU dynamic Smagorinsky (Germano+Lilly) | ✅ DONE | t45 | 7543b4b | 2026-05-30 |
| G4 | GPU ODE mixing-length wall model (van Driest + Picard) | ✅ DONE | t46 | 6730c32 | 2026-05-30 |
| G5 | GPU Berger-Oliger LTS integrator | ✅ DONE | t47 | 55a701c | 2026-05-30 |
| G6 | GPU Baer-Nunziato two-phase solver | ✅ DONE | t48 | e5bc6f6 | 2026-05-30 |

---

## Extra Work (not in original phase list)

| Work item | Title | Status | Gate | Commit | Date |
|-----------|-------|--------|------|--------|------|
| IBM | STL import + GPU ghost-cell IBM | ✅ DONE | t49 (I5–I9, W5) | 70d2157 / 596223b | 2026-05-31/06-01 |
| Rect | Rectangular domain (Lx/Ly/Lz/Nx/Ny/Nz) | ✅ DONE | t42_rect | 06e42ac–5a0faac | 2026-05-29 |
| Sim | simulate + template.json sync | ✅ DONE | — | 46576f0–f02b424 | 2026-05-31 |
| BF | Body-force source term (momentum + energy) | ✅ DONE | — | e7b4085–dc26e2e | 2026-06-01 |
| C50 | Turbulent channel WMLES Re_τ=395 DNS gate | ✅ DONE | t50 (C50) | 233ca24 | 2026-06-01 |
| M | Metrics & monitoring system | ✅ DONE | t51 (M0–M4) | 77223f2 | 2026-06-02 |

---

## FSI + Visualization Roadmap

| Phase | Title | Status | Gate | Commit | Date |
|-------|-------|--------|------|--------|------|
| C1 | VTK XML binary writer + MetricsBus VtkChannel | ⬜ TODO | t52 | — | — |
| FSI-1 | Moving-wall IBM + rigid 6-DOF ODE | ⬜ TODO | t53 | — | — |
| FSI-2 | Corotational beam FEM (JAX) + Aitken partitioned coupling | ⬜ TODO | t54 | — | — |

> Plan: `docs/superpowers/plans/2026-06-02-fsi-viz-roadmap.md`

---

## Gate Map Summary

| Gate | Phase | Description |
|------|-------|-------------|
| t1–t23 | CPU baseline | CPU correctness suite (`ba` target) |
| t24 | P8.6 | CUDA Graph re-capture on regrid |
| t25 | P9.1 | GPU vs CPU solution correctness |
| t26 | P10-A3 | NSSolver GPU dispatch |
| t27 | P-SGS | Static Smagorinsky |
| t28 | D2 | MPI halo (MPI-gated) |
| t29 | D1 | GPU-native AMR |
| t30 | D2 | CUDA-aware MPI (MPI-gated) |
| t31 | D3 | TENO5-A |
| t32 | D4 | GPU GMRES (IMEX) |
| t33 | D5 | Arrhenius reactive flow |
| t34 | D6 | P1 radiation |
| t35 | D7 | WMLES algebraic wall model |
| t36 | — | GPU snapshot (slice + metrics) |
| t37 | D3 | TENO7-A |
| t38 | D10 | Adjoint dot-product test |
| t39 | D11 | Python NSSolver bindings |
| t40 | D9 | NSCBC outflow/inflow BC |
| t41 | D11 | JAX custom_vjp wiring |
| t42 | D11 | Adjoint RK3 + rectangular NS |
| t43 | G1 | ACDI phi transport |
| t44 | G2 | Adjoint convective RHS (GPU) |
| t45 | G3 | Dynamic Smagorinsky (GPU) |
| t46 | G4 | ODE wall model (GPU) |
| t47 | G5 | Berger-Oliger LTS (GPU) |
| t48 | G6 | Baer-Nunziato two-phase (GPU) |
| t49 | IBM | STL import + ghost-cell IBM (I5–I9, W5 winding-number sign) |
| t50 | C50 | Turbulent channel WMLES Re_τ=395, B∈[4.9,6.2] |
| t51 | M | Metrics & monitoring (residual, surface forces, probes, field dump) |
| t52 | C1 | VTK XML binary writer — file written and valid XML |
| t53 | FSI-1 | Prescribed pitching NACA0012, Cl amplitude within 15% of Theodorsen |
| t54 | FSI-2 | Dowell 2D flat-plate flutter onset U* ∈ [5.3, 7.3] (±15% of theory) |

---

*Last updated: 2026-06-02*
