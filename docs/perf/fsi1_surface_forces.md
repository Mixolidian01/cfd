# FSI-1 Surface Force Kernels — NCU Baseline Placeholder

Kernels added in commit 608ca66 / 49b38fc:
- `k_surface_forces_ibm` — one thread per leaf × GPU_NCELL cells, atomicAdd to d_wrench[6]
- `k_apply_moving_wall` — one thread per ghost entry, updates u_wall/v_wall/w_wall

## Roofline characteristics
- `k_surface_forces_ibm`: memory-bound (reads d_scratch, d_cell_type, d_sdf, d_wnorm; writes d_wrench via atomicAdd). Expected: low arithmetic intensity.
- `k_apply_moving_wall`: compute-bound (cross-product per ghost entry). Expected: ~1 FLOP/byte.

## Baseline to collect
Run on RTX 3070 Laptop or A100 with:
```
ncu --set full --export docs/perf/fsi1_surface_forces_<date> \
    ./build/t53_fsi_rigid
```
Then update this file with DRAM BW%, roofline position.
