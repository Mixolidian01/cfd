#include "metrics/metrics_bus.hpp"
#include "cuda/gpu_rhs.cuh"
#include "gpu_snapshot.hpp"
#include "cuda/gpu_ibm.cuh"

void MetricsBus::build(const MetricsConfig& cfg, int n_leaves,
                       const GpuRhsList* rhs_list,
                       const SnapLeafMeta* snap_metas,
                       const GpuIbmList* ibm_list)
{
    cfg_ = cfg;
    monitors_.clear();
    (void)n_leaves; (void)rhs_list; (void)snap_metas; (void)ibm_list;
}

void MetricsBus::launch(const GpuRhsList&, const SnapLeafMeta*, int, int, cudaStream_t) {}
void MetricsBus::collect(int, double, double) {}
void MetricsBus::write(int, double, double) {}
void MetricsBus::rebuild(int, const GpuRhsList*, const SnapLeafMeta*, const GpuIbmList*) {}
