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
| A9 | 2026-04-13 | A04 root-cause diagnosis — BlockNode constructor, ghost-fill logic and viscous stencil traced; edge/corner ghosts identified as the mass-conservation leak |
| A10 | 2026-04-13 | A04 fix committed — edge+corner ghost fill added to fill_ghosts_periodic and fill_ghosts_wall in block_tree.cpp; Python-verified; commit 7f17be9 |
| A11 | 2026-04-13 | T11c root-cause identified — wall_x/y/z lambdas only negated wall-normal momentum; tangential components copied instead of negated (wrong no-slip) |
| A12 | 2026-04-13 | T11c fix committed — all three momentum components negated in wall_x/y/z lambdas (no-slip image method); commit 4ee1b38 |
| A13 | 2026-04-13 | Phase 1 AMR foundation committed (P1.1–P1.6): free-list coarsen, work-queue balance, CF ghost dispatch, flux registers, full regrid(), leaf cache; commit 32dbfe7 |
| A19 | 2026-04-13 | A05-fix2 committed — averaged coarse ghost fill (2×2 fine cells per coarse ghost slot) + apply_flux_correction axis=0/1 ck/ci index corrected; commit 6d4095e |
| A20 | 2026-04-13 | Custom instructions summary — Space role, solver features, answer format, code rules, companion files, self-correction protocol |
| A21 | 2026-04-13 | Root-cause analysis + fixes for S03/S07/S08 (sgs.cpp: mu_t ghost cells zero, periodic wrap missing) and A05 (accumulate_fine_flux never called, register always empty) |
