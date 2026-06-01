#pragma once
#include "field_dumper.hpp"
#include "cuda/gpu_metrics.cuh"
#include "solver/ns_solver.hpp"
#include <memory>
#include <string>

struct SurfaceMonitor {
    GpuSurfaceList                 gpu_list;
    std::unique_ptr<CsvWriter>     csv;
    int                            interval = 0;
    float                          mu = 1.8e-5f;

    void build(const GpuIbmList& ibm, const SolverConfig::SurfaceConfig& cfg,
               float dynamic_viscosity, int interval_steps,
               const std::string& output_dir);
    void exec(cudaStream_t s) const { gpu_list.exec(s, mu); }
    void write(int step, double t);
};
