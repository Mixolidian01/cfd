#pragma once
#include "field_dumper.hpp"
#include "cuda/gpu_metrics.cuh"
#include "metrics/imonitor.hpp"
#include <string>
#include <array>
#include <memory>

struct ResidualMonitor : IMonitor {
    int  interval   = 0;
    int  step_count = 0;

    GpuResidualList  gpu_list;
    std::array<double, GPU_NVAR> l2_ref = {};  // norms at step 0

    void build(int n_leaves, int interval_steps, const std::string& output_dir);
    void launch(const GpuLeafRhsMeta* d_metas, int step, cudaStream_t s);
    void collect_and_write(int step, double t, double dt);

private:
    std::string csv_path_;
    std::unique_ptr<CsvWriter> csv_;
};

inline void ResidualMonitor::build(int n_leaves, int interval_steps,
                                    const std::string& output_dir) {
    interval = interval_steps;
    gpu_list.build(n_leaves);
    csv_ = std::make_unique<CsvWriter>(output_dir + "/residuals.csv",
                                        "step,t,dt,rho,rhou,rhov,rhow,E");
}

inline void ResidualMonitor::launch(const GpuLeafRhsMeta* d_metas,
                                     int step, cudaStream_t s) {
    if (interval <= 0 || step % interval != 0) return;
    gpu_list.exec(d_metas, s);
}

inline void ResidualMonitor::collect_and_write(int step, double t, double dt) {
    if (interval <= 0 || step % interval != 0) return;
    double l2[GPU_NVAR];
    gpu_list.fold(l2);
    csv_->append(step, t, dt, l2[0], l2[1], l2[2], l2[3], l2[4]);
}
