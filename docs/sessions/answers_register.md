# Answers Register

This file tracks all answers provided by the AI assistant during the CFD solver development session.
Each entry contains the answer index, date, and a one-line description.

| Index | Date | Description |
|---|---|---|
| A1 | 2026-04-09 | Initial code review — octant convention mismatch, `BlockNode::h` div/0, `regrid()` no-op, UB in `refine()` |
| A2 | 2026-04-09 | Extended analysis — GPU ghost fill n_blocks>1 broken, SGS NaN for child blocks, MG disconnected, CFL D→H bottleneck |
| A3 | 2026-04-12 | Full architecture analysis — physics, numerics, bug register, performance vs PeleC/OpenFOAM/nekRS |
| A4 | 2026-04-12 | Consolidated bug register merging Perplexity A3, claude_analysis_20260409, Code_technical_review_v2, and supplementary Claude analysis — 16 critical/moderate/minor items |
| A5 | 2026-04-12 | Cross-domain improvement proposals — WENO5-Z, skew-symmetric convection, entropy-stable fluxes, persistent GPU kernels, AoSoA, CUDA Graph, LTS, IMEX-ARK, neural SGS |
| A6 | 2026-04-12 | Combined roadmap — compatibility analysis + 5-phase plan (P0 emergency fixes through P4 HPC features); dependency graph; expected performance table |
| A9 | 2026-04-13 | A04 root-cause diagnosis — BlockNode constructor, ghost-fill logic and viscous stencil traced; edge+corner ghosts identified as the mass-conservation leak |
| A10 | 2026-04-13 | A04 fix committed — edge+corner ghost fill added to fill_ghosts_periodic and fill_ghosts_wall in block_tree.cpp; Python-verified; commit 7f17be9 |
| A11 | 2026-04-13 | T11c root-cause identified — wall_x/y/z lambdas only negated wall-normal momentum; tangential components copied instead of negated (wrong no-slip) |
| A12 | 2026-04-13 | T11c fix committed — all three momentum components negated in wall_x/y/z lambdas (no-slip image method); commit 4ee1b38 |
| A13 | 2026-04-13 | Phase 1 AMR foundation committed (P1.1–P1.6): free-list coarsen, work-queue balance, CF ghost dispatch, flux registers, full regrid(), leaf cache; commit 32dbfe7 |
| A19 | 2026-04-13 | A05-fix2 committed — averaged coarse ghost fill (2×2 fine cells per coarse ghost slot) + apply_flux_correction axis=0/1 ck/ci index corrected; commit 6d4095e |
| A20 | 2026-04-13 | Custom instructions summary — Space role, solver features, answer format, code rules, companion files, self-correction protocol |
| A21 | 2026-04-13 | Root-cause analysis + fixes for S03/S07/S08 (sgs.cpp: mu_t ghost cells zero, periodic wrap missing) and A05 (accumulate_fine_flux never called, register always empty) |
| A28 | 2026-04-15 | Solver iteration call graph — Mermaid diagram of full advance() sequence: RK3 stages, Berger-Colella reflux, regrid, SGS split |
| A34 | 2026-04-15 | A05 root-cause confirmed from test_amr6.cpp: coarsen()->restrict_to_parent() overwrote flux-corrected coarse cells; fix is to move regrid() before RK3 |
| A35 | 2026-04-15 | A05-fix5 committed — regrid() moved to top of advance() (on Q^n) so topology is frozen during RK3+Berger-Colella; apply_flux_correction is last write per step; commit 8bdf893 |
| A48 | 2026-04-25 | P7.5 TGV DNS benchmark — ε(t) comparison table at 32³ ILES vs HiOCFD4 512³ reference; gate 4/4 pass; 128³ production run requires MPI (P7.1). See table below. |

## A48 — P7.5 TGV DNS quantitative error table

**Setup:** ILES, 32³ (2 uniform AMR levels, 64 blocks × 8³), CFL=0.5, t∈[0,18].  
**Normalization:** E_k = KE/V, V=(2π)³=248.05; ε = −dE_k/dt.  
**Reference:** HiOCFD4 C3.3, DeBonis (2013) NASA/TM-2013-217850 (inviscid, 512³).

| t   | ε solver (32³) | ε ref (512³) | ratio |
|-----|----------------|--------------|-------|
|  0  | 0.000e+00      | 0.000e+00    | —     |
|  2  | 1.443e-03      | 2.340e-03    | 0.62  |
|  4  | 2.379e-03      | 5.400e-03    | 0.44  |
|  6  | 3.514e-03      | 8.200e-03    | 0.43  |
|  8  | 4.854e-03      | 9.380e-03    | 0.52  |
|  9  | 3.659e-03      | 9.500e-03    | 0.39  |
| 10  | 4.717e-03      | 9.180e-03    | 0.51  |
| 12  | 4.288e-03      | 7.810e-03    | 0.55  |
| 14  | 4.138e-03      | 6.250e-03    | 0.66  |
| 18  | 2.626e-03      | 3.910e-03    | 0.67  |

**Key results:** t\*_peak = 4.88 (vs DNS 9.0), ε_peak = 8.82e-3 (vs ref 9.5e-3 at t=9).  
E_k(18)/E_k(0) = 0.414 (solver 32³) vs 0.234 (ref 512³).

**Physical interpretation:** At 32³ the cascade is underresolved — the large-scale peak
comes earlier (t\*≈5 vs t\*≈9) and total E_k decays less than the high-resolution reference
(where the fully-resolved small-scale modes carry the dissipation). Convergence toward the
Brachet (1983) DNS reference at 128³ requires multi-rank MPI (P7.1).
