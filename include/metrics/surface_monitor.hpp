#pragma once
#include "field_dumper.hpp"
#include "cuda/gpu_metrics.cuh"
#include "solver/ns_solver.hpp"
#include "metrics/imonitor.hpp"
#include <memory>
#include <string>

struct SurfaceMonitor : IMonitor {
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

inline void SurfaceMonitor::build(const GpuIbmList& ibm,
                                   const SolverConfig::SurfaceConfig& cfg,
                                   float dynamic_viscosity, int interval_steps,
                                   const std::string& output_dir) {
    interval = interval_steps;
    mu = dynamic_viscosity;
    gpu_list.build(ibm, cfg);
    csv = std::make_unique<CsvWriter>(output_dir + "/" + cfg.name + "_forces.csv",
                                      "step,t,Fx,Fy,Fz,Mx,My,Mz");
}

inline void SurfaceMonitor::write(int step, double t) {
    const double* r = gpu_list.results();
    csv->append(step, t, r[0], r[1], r[2], r[3], r[4], r[5]);
}
