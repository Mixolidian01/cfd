// apps/simulate_gpu.cu — GPU-native command-line simulation runner
//
// Identical JSON config as apps/simulate.cpp.  The GPU backend is always
// active: GpuGraphSolver is injected after NSSolver::init(), routing all
// advance() calls through the CUDA Graph SSP-RK3 path.
//
// Build:  cmake --build build -t simulate_gpu
// Run:    ./build/simulate_gpu config.json
//
// Use scripts/launch.sh --backend gpu instead of calling this directly.

#include "solver/ns_solver.hpp"
#include "cuda/gpu_graph.cuh"
#include "gpu_pool.hpp"
#include "mesh/bc_types.hpp"        // bc_to_int()
#include "io/live_streamer.hpp"
#include "io/checkpoint.hpp"
#include "models/sgs.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <cuda_runtime.h>

// =============================================================================
// Minimal flat-JSON reader (identical to simulate.cpp)
// =============================================================================

struct Config {
    std::map<std::string, std::string> m_;

    static Config from_file(const char* path) {
        std::ifstream f(path);
        if (!f) {
            fprintf(stderr, "simulate_gpu: cannot open config '%s'\n", path);
            exit(1);
        }
        std::string src((std::istreambuf_iterator<char>(f)), {});

        Config c;
        size_t n = src.size(), i = 0;

        auto skip = [&]() {
            while (i < n) {
                if (std::isspace((unsigned char)src[i])) { ++i; continue; }
                if (i+1 < n && src[i] == '/' && src[i+1] == '/') {
                    while (i < n && src[i] != '\n') ++i;
                    continue;
                }
                break;
            }
        };

        while (i < n) {
            skip();
            if (i >= n) break;
            if (src[i] != '"') { ++i; continue; }
            ++i;
            std::string key;
            while (i < n && src[i] != '"') key += src[i++];
            if (i < n) ++i;
            skip();
            if (i >= n || src[i] != ':') continue;
            ++i; skip();
            if (i >= n) break;
            std::string val;
            if (src[i] == '"') {
                ++i;
                while (i < n && src[i] != '"') {
                    if (src[i] == '\\' && i+1 < n) { ++i; val += src[i]; }
                    else val += src[i];
                    ++i;
                }
                if (i < n) ++i;
            } else {
                while (i < n && src[i] != ',' && src[i] != '}'
                       && src[i] != '\n' && src[i] != '\r') {
                    if (i+1 < n && src[i] == '/' && src[i+1] == '/') break;
                    val += src[i++];
                }
                while (!val.empty() && std::isspace((unsigned char)val.back()))
                    val.pop_back();
            }
            if (!key.empty()) c.m_[key] = val;
        }
        return c;
    }

    std::string str(const char* k, const char* def = "") const {
        auto it = m_.find(k); return it != m_.end() ? it->second : def;
    }
    double d(const char* k, double def = 0.0) const {
        auto it = m_.find(k); if (it == m_.end()) return def;
        try { return std::stod(it->second); } catch(...) { return def; }
    }
    int i(const char* k, int def = 0) const {
        auto it = m_.find(k); if (it == m_.end()) return def;
        try { return std::stoi(it->second); } catch(...) { return def; }
    }
    bool b(const char* k, bool def = false) const {
        auto it = m_.find(k); if (it == m_.end()) return def;
        return it->second == "true";
    }
    bool has(const char* k) const { return m_.count(k) > 0; }
    void print_all() const {
        printf("  %-28s  %s\n", "key", "value");
        for (auto& [k, v] : m_) printf("  %-28s  %s\n", k.c_str(), v.c_str());
    }
};

#include "initial_conditions.hpp"

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    const char* config_path = (argc >= 2) ? argv[1] : "sim.json";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("simulate_gpu: GPU = %s\n", prop.name);
    printf("simulate_gpu: loading config from '%s'\n", config_path);

    Config cfg = Config::from_file(config_path);

    printf("simulate_gpu: configuration:\n");
    cfg.print_all();
    printf("\n");

    // ── Solver config ──────────────────────────────────────────────────────────
    NSSolver solver;
    auto& sc = solver.cfg;

    sc.time.cfl             = cfg.d("cfl",             0.8);
    sc.time.t_end           = cfg.d("t_end",           1.0);
    sc.time.max_steps       = cfg.i("max_steps",       1000000);
    sc.io.diag_interval     = cfg.i("diag_interval",   10);
    sc.io.verbose           = cfg.b("verbose",         true);
    sc.amr.regrid_interval  = cfg.i("regrid_interval", 0);
    sc.amr.max_level        = cfg.i("max_level",       2);
    sc.amr.use_lts          = cfg.b("use_lts",         false);
    sc.amr.lts_ratio        = cfg.i("lts_ratio",       2);
    sc.physics.use_imex     = cfg.b("use_imex",        false);

    {
        std::string bc_str = cfg.str("bc", "Periodic");
        if      (bc_str == "Wall") sc.bc.variant = WallBC{};
        else if (bc_str == "Open") sc.bc.variant = OpenBC{};
        else                       sc.bc.variant = PeriodicBC{};
    }

    double sgs_cs  = cfg.d("sgs_cs",  0.16);
    double sgs_prt = cfg.d("sgs_prt", 0.9);
    {
        std::string sgs_str = cfg.str("sgs", "none");
        if (sgs_str == "Smagorinsky")
            sc.physics.sgs = std::make_shared<SmagorinskyModel>(sgs_cs, sgs_prt);
        else if (sgs_str == "Dynamic")
            sc.physics.sgs = std::make_shared<DynamicSmagorinskyModel>(sgs_prt);
        else
            sc.physics.sgs = nullptr;
    }

    double domain_L      = cfg.d("domain_L",      1.0);
    int    refine_levels = cfg.i("refine_levels",  0);

    std::string ckpt_load  = cfg.str("checkpoint_load",  "");
    std::string ckpt_save  = cfg.str("checkpoint_save",  "");
    int         ckpt_intvl = cfg.i("checkpoint_interval", 0);

    // ── Build IC and initialise ────────────────────────────────────────────────
    auto ic = build_ic(cfg);

    printf("simulate_gpu: initialising solver (domain_L=%.4g, ic=%s, bc=%s)\n",
           domain_L, cfg.str("ic", "uniform").c_str(), cfg.str("bc", "Periodic").c_str());

    if (!ckpt_load.empty()) {
        solver.init(domain_L, ic);
        printf("simulate_gpu: loading checkpoint from '%s'\n", ckpt_load.c_str());
        checkpoint_load(solver, ckpt_load);
    } else {
        solver.init(domain_L, ic);

        if (refine_levels > 0) {
            printf("simulate_gpu: applying %d extra uniform refinement pass(es)\n",
                   refine_levels);
            for (int lvl = 0; lvl < refine_levels; ++lvl) {
                auto leaves = solver.tree.leaf_indices();
                for (int li : leaves) solver.tree.refine(li);
                solver.tree.rebuild_neighbours();
            }
            fill_leaves(solver, ic);
            solver.alloc_scratch();
        }
    }

    // ── GPU solver injection ───────────────────────────────────────────────────
    GpuPool pool;
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (!blk) continue;
        pool.alloc(blk);
        pool.upload(blk);
    }

    GpuGraphSolver graph_solver;
    if (sc.physics.sgs) {
        if (auto* sm = dynamic_cast<SmagorinskyModel*>(sc.physics.sgs.get()))
            graph_solver.set_gpu_sgs(sm->Cs, sm->Pr_t);
    }
    graph_solver.set_ducros(sc.numerics.ducros_p_threshold,
                            1.0 / sc.numerics.ducros_blend_width);
    graph_solver.build(solver.tree, pool, bc_to_int(sc.bc.variant));

    solver.set_gpu_pool(&pool);
    solver.set_gpu_solver(&graph_solver);

    printf("simulate_gpu: GPU solver active  leaves=%d\n",
           (int)solver.tree.leaf_indices().size());

    // ── Live streamer + GPU snapshot buffer (optional) ────────────────────────
    std::unique_ptr<LiveStreamer>        streamer;
    std::unique_ptr<GpuSnapshotBuffer>  snap_buf;
    int stream_port = cfg.i("stream_port", 0);
    if (stream_port > 0) {
        StreamConfig scfg;
        scfg.port        = stream_port;
        scfg.axis        = static_cast<uint8_t>(cfg.i("stream_axis",   2));
        scfg.pos         = cfg.d("stream_pos",         0.5);
        scfg.stride      = cfg.i("stream_stride",      1);
        scfg.volume_size = cfg.i("volume_size",        32);

        std::string sv = cfg.str("stream_var", "rho");
        if      (sv == "press") scfg.var = StreamVar::PRESS;
        else if (sv == "temp")  scfg.var = StreamVar::TEMP;
        else if (sv == "umag")  scfg.var = StreamVar::UMAG;
        else if (sv == "rhou")  scfg.var = StreamVar::RHOU;
        else if (sv == "rhov")  scfg.var = StreamVar::RHOV;
        else if (sv == "rhow")  scfg.var = StreamVar::RHOW;
        else if (sv == "etot")  scfg.var = StreamVar::ETOT;
        else                    scfg.var = StreamVar::RHO;

        streamer = std::make_unique<LiveStreamer>(scfg);
        solver.set_streamer(streamer.get());

        // Option A/C: allocate GPU snapshot buffer — zero-copy slice + GPU metrics.
        const int n_leaves_max = static_cast<int>(solver.tree.leaf_indices().size());
        snap_buf = std::make_unique<GpuSnapshotBuffer>();
        snap_buf->alloc(std::max(n_leaves_max, 64));  // reserve some headroom for AMR
        snap_buf->var_id   = static_cast<int>(scfg.var);
        snap_buf->axis     = static_cast<int>(scfg.axis);
        snap_buf->norm_pos = static_cast<float>(scfg.pos);
        snap_buf->domain_L = static_cast<float>(domain_L);
        solver.set_gpu_snapshot(snap_buf.get());
        // Re-build with snapshot buffer set so _upload_snap_metas() runs.
        graph_solver.build(solver.tree, pool, bc_to_int(sc.bc.variant));

        printf("simulate_gpu: live feed enabled on http://localhost:%d  (var=%s axis=%d)  [GPU snap]\n",
               stream_port, sv.c_str(), (int)scfg.axis);
    }

    // ── Time integration ───────────────────────────────────────────────────────
    printf("simulate_gpu: running  t_end=%.4g  max_steps=%d  cfl=%.3g  sgs=%s\n",
           sc.time.t_end, sc.time.max_steps, sc.time.cfl,
           sc.physics.sgs ? sc.physics.sgs->name() : "none");

    if (ckpt_intvl > 0 && !ckpt_save.empty()) {
        while (solver.t < sc.time.t_end && solver.step < sc.time.max_steps) {
            solver.advance();
            if (solver.step % ckpt_intvl == 0) {
                std::string path = ckpt_save + "." + std::to_string(solver.step);
                checkpoint_save(solver, path);
                printf("simulate_gpu: checkpoint → %s\n", path.c_str());
            }
        }
    } else {
        solver.run();
    }

    // ── Final checkpoint ───────────────────────────────────────────────────────
    if (!ckpt_save.empty()) {
        checkpoint_save(solver, ckpt_save);
        printf("simulate_gpu: final checkpoint → %s\n", ckpt_save.c_str());
    }

    // ── Cleanup ────────────────────────────────────────────────────────────────
    // GPU pool must be freed before GpuGraphSolver is destroyed.
    for (int li : solver.tree.leaf_indices()) {
        CellBlock* blk = solver.tree.nodes[li].block.get();
        if (blk && pool.has_device(blk)) pool.free(blk);
    }

    // ── Final diagnostics ──────────────────────────────────────────────────────
    auto diag = solver.compute_diag();
    printf("\nsimulate_gpu: final state\n");
    printf("  step            = %d\n",    diag.step);
    printf("  t               = %.6e\n",  diag.t);
    printf("  mass            = %.6e\n",  diag.mass);
    printf("  kinetic energy  = %.6e\n",  diag.kinetic_energy);
    printf("  total energy    = %.6e\n",  diag.total_energy);
    printf("simulate_gpu: done.\n");
    return 0;
}
