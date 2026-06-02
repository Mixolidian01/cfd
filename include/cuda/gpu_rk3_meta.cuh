#pragma once
// Per-leaf RK3 state pointers — thin header included by both gpu_graph.cuh
// and gpu_graph_kernels.cuh so neither needs to pull in the other.
struct GpuRk3LeafMeta {
    double*       d_Q;    // current state (in/out per stage)
    double*       d_Qn;   // saved Q^n (written by k_save_qn, read by update kernels)
    const double* d_RHS;  // from GpuRhsList.d_rhs_pool (read-only)
};
