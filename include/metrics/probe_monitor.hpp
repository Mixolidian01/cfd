#pragma once
#include "field_dumper.hpp"
#include "cuda/gpu_metrics.cuh"
#include "solver/ns_solver.hpp"
#include "metrics/imonitor.hpp"
#include <memory>
#include <string>
#include <vector>

struct ProbeMonitor : IMonitor {
    GpuProbeList                   gpu_list;
    std::unique_ptr<CsvWriter>     csv;
    int                            interval = 0;
    SolverConfig::ProbeConfig::Type type = SolverConfig::ProbeConfig::Type::POINT;
    std::string                    name;

    void build(int n_leaves, const SnapLeafMeta* h_metas,
               const SolverConfig::ProbeConfig& cfg,
               int interval_steps, const std::string& output_dir);

    void exec(const SnapLeafMeta* d_metas, int n_leaves, cudaStream_t s);
    void write(int step, double t);
};

inline void ProbeMonitor::build(int /*n_leaves*/, const SnapLeafMeta* /*h_metas*/,
                                 const SolverConfig::ProbeConfig& cfg,
                                 int interval_steps, const std::string& output_dir) {
    interval = interval_steps;
    type     = cfg.type;
    name     = cfg.name;
    if (cfg.type == SolverConfig::ProbeConfig::Type::POINT) {
        gpu_list.build_point((float)cfg.p0[0], (float)cfg.p0[1], (float)cfg.p0[2], 0);
        csv = std::make_unique<CsvWriter>(
            output_dir + "/" + cfg.name + "_probe.csv", "step,t," + cfg.quantity);
    } else if (cfg.type == SolverConfig::ProbeConfig::Type::PLANE_AVG) {
        const int ax = cfg.axis;
        gpu_list.build_plane(cfg.n_slabs, ax,
                             (float)cfg.p0[ax], (float)cfg.p1[ax], 0);
        csv = std::make_unique<CsvWriter>(
            output_dir + "/" + cfg.name + "_plane.csv", "step,t,slab,mean");
    }
}

inline void ProbeMonitor::exec(const SnapLeafMeta* d_metas, int n_leaves, cudaStream_t s) {
    if (type == SolverConfig::ProbeConfig::Type::POINT)
        gpu_list.exec(d_metas, n_leaves, s);
    else if (type == SolverConfig::ProbeConfig::Type::PLANE_AVG)
        gpu_list.exec_plane(d_metas, n_leaves, s);
}

inline void ProbeMonitor::write(int step, double t) {
    if (type == SolverConfig::ProbeConfig::Type::POINT) {
        for (int i = 0; i < gpu_list.n_probes; ++i)
            csv->append(step, t, gpu_list.h_results[i]);
    } else if (type == SolverConfig::ProbeConfig::Type::PLANE_AVG) {
        for (int s = 0; s < gpu_list.n_slabs; ++s) {
            const double mean = gpu_list.h_slab_cnt[s] > 0
                              ? gpu_list.h_slab_sum[s] / gpu_list.h_slab_cnt[s] : 0.0;
            csv->append(step, t, s, mean);
        }
    }
}
