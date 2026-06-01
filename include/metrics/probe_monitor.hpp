#pragma once
#include "field_dumper.hpp"
#include "cuda/gpu_metrics.cuh"
#include "solver/ns_solver.hpp"
#include <memory>
#include <string>
#include <vector>

struct ProbeMonitor {
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
