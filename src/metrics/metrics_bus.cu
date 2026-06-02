#include "metrics/metrics_bus.hpp"
#include "metrics/residual_monitor.hpp"
#include "metrics/surface_monitor.hpp"
#include "metrics/probe_monitor.hpp"
#include "cuda/gpu_rhs.cuh"
#include "gpu_snapshot.hpp"
#include "cuda/gpu_ibm.cuh"
#include "io/vtk_writer.hpp"
#include <filesystem>

void MetricsBus::build(const MetricsConfig& cfg, int n_leaves,
                       const GpuRhsList* rhs_list,
                       const SnapLeafMeta* snap_metas,
                       const GpuIbmList*  ibm_list)
{
    cfg_ = cfg;
    monitors_.clear();
    residual_mon_ = nullptr;
    surface_mons_.clear();
    probe_mons_.clear();

    std::filesystem::create_directories(cfg.output_dir);

    if (cfg.residual_interval > 0 && rhs_list && n_leaves > 0) {
        auto m = std::make_unique<ResidualMonitor>();
        m->build(n_leaves, cfg.residual_interval, cfg.output_dir);
        residual_mon_ = m.get();
        monitors_.push_back(std::move(m));
    }

    if (cfg.surface_interval > 0 && ibm_list) {
        for (const auto& sc : cfg.surfaces) {
            auto m = std::make_unique<SurfaceMonitor>();
            m->build(*ibm_list, sc, 1.8e-5f, cfg.surface_interval, cfg.output_dir);
            surface_mons_.push_back(m.get());
            monitors_.push_back(std::move(m));
        }
    }

    if (cfg.probe_interval > 0 && snap_metas) {
        for (const auto& pc : cfg.probes) {
            auto m = std::make_unique<ProbeMonitor>();
            m->build(n_leaves, snap_metas, pc, cfg.probe_interval, cfg.output_dir);
            probe_mons_.push_back(m.get());
            monitors_.push_back(std::move(m));
        }
    }

    n_leaves_   = n_leaves;
    rhs_list_   = rhs_list;
    snap_metas_ = snap_metas;
}

void MetricsBus::launch(const GpuRhsList& rhs, const SnapLeafMeta* h_metas,
                        int n_leaves, int step, cudaStream_t s)
{
    if (residual_mon_)
        residual_mon_->launch(rhs.d_metas, step, s);
    for (auto* sm : surface_mons_)
        if (sm->interval > 0 && step % sm->interval == 0)
            sm->exec(s);
    for (auto* pm : probe_mons_)
        if (pm->interval > 0 && step % pm->interval == 0)
            pm->exec(h_metas, n_leaves, s);
}

void MetricsBus::collect(int step, double t, double dt) {
    collect_step_ = step;
    collect_t_    = t;
    collect_dt_   = dt;
}

void MetricsBus::write(int step, double t, double dt) {
    if (residual_mon_) residual_mon_->collect_and_write(step, t, dt);
    for (auto* sm : surface_mons_)
        if (sm->interval > 0 && step % sm->interval == 0)
            sm->write(step, t);
    for (auto* pm : probe_mons_)
        if (pm->interval > 0 && step % pm->interval == 0)
            pm->write(step, t);
    // VTK XML binary output
    if (!cfg_.vtk_prefix.empty() && cfg_.vtk_interval > 0
        && step % cfg_.vtk_interval == 0 && snap_metas_ && n_leaves_ > 0) {
        for (int li = 0; li < n_leaves_; ++li) {
            const auto& m = snap_metas_[li];
            vtk_write_vts(cfg_.vtk_prefix, step, li,
                          (double)m.ox, (double)m.oy, (double)m.oz, (double)m.h,
                          m.d_Q);
        }
        vtk_write_pvts(cfg_.vtk_prefix, step, n_leaves_);
    }
}

void MetricsBus::rebuild(int n_leaves, const GpuRhsList* rhs_list,
                         const SnapLeafMeta* snap_metas, const GpuIbmList* ibm_list) {
    build(cfg_, n_leaves, rhs_list, snap_metas, ibm_list);
}
