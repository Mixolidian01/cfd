#pragma once
#include "field_dumper.hpp"
#include "cuda/gpu_metrics.cuh"
#include <string>
#include <array>
#include <memory>

struct ResidualMonitor {
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
