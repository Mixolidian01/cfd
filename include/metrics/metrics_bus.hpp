#pragma once
#include "solver/ns_solver.hpp"
#include "field_dumper.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// Forward declarations
struct GpuRhsList;
struct SnapLeafMeta;
struct GpuIbmList;

// ── IMonitor — interface for all monitor categories ──────────────────────────
struct IMonitor {
    virtual ~IMonitor() = default;
    virtual void launch([[maybe_unused]] cudaStream_t s) {}
    virtual void collect([[maybe_unused]] int step, [[maybe_unused]] double t,
                         [[maybe_unused]] double dt) {}
};

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
    MetricsConfig cfg_;
    std::vector<std::unique_ptr<IMonitor>> monitors_;
};
