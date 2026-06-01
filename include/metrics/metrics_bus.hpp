#pragma once
#include "solver/ns_solver.hpp"
#include "field_dumper.hpp"
#include "metrics/imonitor.hpp"
#include <memory>
#include <vector>
#include <string>
#ifdef __CUDACC__
#  include <cuda_runtime.h>
#else
typedef struct CUstream_st* cudaStream_t;
#endif

// Forward declarations
struct GpuRhsList;
struct SnapLeafMeta;
struct GpuIbmList;
struct ResidualMonitor;
struct SurfaceMonitor;
struct ProbeMonitor;

// ── MetricsBus ────────────────────────────────────────────────────────────────
struct MetricsBus {
    MetricsBus() = default;
    MetricsBus(const MetricsBus&) = delete;
    MetricsBus& operator=(const MetricsBus&) = delete;

    void build(const MetricsConfig& cfg,
               int n_leaves,
               const GpuRhsList* rhs_list,
               const SnapLeafMeta* snap_metas,
               const GpuIbmList*  ibm_list);

    void launch(const GpuRhsList& rhs, const SnapLeafMeta* h_metas,
                int n_leaves, int step, cudaStream_t s);

    void collect(int step, double t, double dt);
    void write(int step, double t, double dt);
    void rebuild(int n_leaves, const GpuRhsList* rhs_list,
                 const SnapLeafMeta* snap_metas, const GpuIbmList* ibm_list);

    [[nodiscard]] bool active() const { return !monitors_.empty(); }

private:
    MetricsConfig                          cfg_;
    std::vector<std::unique_ptr<IMonitor>> monitors_;
    ResidualMonitor*                       residual_mon_ = nullptr;
    std::vector<SurfaceMonitor*>           surface_mons_;
    std::vector<ProbeMonitor*>             probe_mons_;
    const GpuRhsList*                      rhs_list_     = nullptr;
    const SnapLeafMeta*                    snap_metas_   = nullptr;
    int                                    n_leaves_     = 0;
    int                                    collect_step_ = 0;
    double                                 collect_t_    = 0.0;
    double                                 collect_dt_   = 0.0;
};
