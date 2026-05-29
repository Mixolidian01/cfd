#pragma once
// G1: GPU ACDI — phi advection + interface-compression per RK3 stage.
// Mirrors phi_rhs() and phi_compression_rhs() in src/schemes/rhs_sensors.cpp.

#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"
#include "gpu_pool.hpp"
#include "cuda/gpu_constants.cuh"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

// ── Separate device pool for phi scalars (NCELL doubles per leaf) ─────────────
struct GpuPhiPool {
    ~GpuPhiPool();
    void alloc(const CellBlock* blk);
    void free(const CellBlock* blk);
    double* d_phi(const CellBlock* blk) const noexcept;
    void upload(const CellBlock* blk);    // blk->phi_data_ → device
    void download(CellBlock* blk) const; // device → blk->phi_data_
    static constexpr size_t slot_bytes() noexcept { return GPU_NCELL * sizeof(double); }
private:
    std::unordered_map<const CellBlock*, double*> ptrs_;
    std::vector<double*> free_list_;
};

// ── Per-leaf ACDI metadata (one entry per leaf, uploaded to device) ───────────
struct alignas(64) GpuAcdiLeafMeta {
    double*       d_Q;              // conserved [GPU_NVAR*GPU_NCELL], read-only
    double*       d_phi;            // current phi [GPU_NCELL]
    double*       d_phin;           // phi^n snapshot [GPU_NCELL]
    double*       d_phi_rhs;        // RHS scratch [GPU_NCELL]
    const double* d_phi_nb[NFACES]; // neighbour phi ptrs; nullptr = domain BC
    double        hx, hy, hz;
    double        ceps;             // compression coefficient (0 = skip)
    int8_t        level_rel[NFACES];// +2 = domain BC
    int8_t        bc_type[NFACES];  // 0=periodic,1=wall,2=open
    int8_t        cf_oct;
};

// ── ACDI list ─────────────────────────────────────────────────────────────────
struct GpuAcdiList {
    GpuAcdiLeafMeta* d_metas       = nullptr;
    double*          d_phin_pool   = nullptr; // n_leaves*GPU_NCELL
    double*          d_rhs_pool    = nullptr; // n_leaves*GPU_NCELL
    int              n_leaves      = 0;
    double           ceps_         = 0.0;

    GpuAcdiList() = default;
    GpuAcdiList(const GpuAcdiList&) = delete;
    GpuAcdiList& operator=(const GpuAcdiList&) = delete;
    ~GpuAcdiList();

    void build(const BlockTree& tree, const GpuPool& q_pool,
               const GpuPhiPool& phi_pool, double ceps, int bc_type);

    void save_phin   (cudaStream_t s) const;
    void zero_rhs    (cudaStream_t s) const;
    void fill_ghosts (cudaStream_t s) const;
    void rhs_advect  (cudaStream_t s) const;
    void rhs_compress(cudaStream_t s) const;   // no-op when ceps==0
    // Stage 1: stage1=true, alpha ignored → phi = phin + dt*rhs
    // Stages 2/3: phi = alpha*phin + (1-alpha)*(phi + dt*rhs)
    void update_phi(const double* d_dt, double alpha,
                    bool stage1, cudaStream_t s) const;
};
