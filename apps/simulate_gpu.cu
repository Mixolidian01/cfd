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
#include "cuda/gpu_ibm.cuh"
#include "cuda/gpu_bvh.cuh"
#include "models/mesh_loader.hpp"
#include "gpu_pool.hpp"
#include "mesh/bc_types.hpp"        // bc_to_int()
#include "io/live_streamer.hpp"
#include "io/checkpoint.hpp"
#include "models/sgs.hpp"
#include "metrics/metrics_bus.hpp"
#include "fsi/rigid_body.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
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
    // ── Argument parsing ──────────────────────────────────────────────────────
    bool        launcher_mode = false;
    int         launcher_port = 8080;
    const char* config_path   = nullptr;
    std::string tmpconfig_path;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--launcher") == 0) { launcher_mode = true; }
        else if (std::strcmp(argv[i], "--port") == 0 && i+1 < argc)
                { launcher_port = std::atoi(argv[++i]); }
        else    { config_path = argv[i]; }
    }
    if (!config_path && !launcher_mode) config_path = "sim.json";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("simulate_gpu: GPU = %s\n", prop.name);

    // ── Phase 1: launcher mode — start HTTP server, wait for POST /launch ────
    std::unique_ptr<LiveStreamer>       streamer;
    std::unique_ptr<GpuSnapshotBuffer> snap_buf;

    if (launcher_mode) {
        StreamConfig lscfg;
        lscfg.port = launcher_port;
        streamer = std::make_unique<LiveStreamer>(lscfg);
        printf("simulate_gpu: launcher mode — open http://localhost:%d to configure\n",
               launcher_port);
        std::string launch_json;
        while (!streamer->pop_launch(launch_json))
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        char tmpbuf[64];
        std::snprintf(tmpbuf, sizeof(tmpbuf), "/tmp/sgpu_launch_%d.json", (int)getpid());
        tmpconfig_path = tmpbuf;
        { std::ofstream tf(tmpconfig_path); tf << launch_json; }
        config_path = tmpconfig_path.c_str();
        printf("simulate_gpu: config received — building solver...\n");
    }

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

    // === Model selection (informational; GPU binary always uses NSSolver) ===
    {
        const std::string model = cfg.str("model", "ns");
        if (model == "bn")
            fprintf(stderr, "[WARN] simulate_gpu: model=bn not yet wired for GPU — use simulate for BN\n");
    }

    // Boundary conditions — per-face keys take precedence over global "bc"
    {
        auto parse_bc_str = [&](const std::string& s) -> BCVariant {
            if (s == "Wall")  return WallBC{};
            if (s == "Open")  return OpenBC{};
            if (s == "NSCBC") return NscbcBC{ cfg.d("nscbc_p_inf", 1.0) };
            return PeriodicBC{};
        };
        static const char* face_keys[6] = {
            "bc_xlo", "bc_xhi", "bc_ylo", "bc_yhi", "bc_zlo", "bc_zhi"
        };
        bool any_face = false;
        for (int d = 0; d < 6; ++d)
            if (cfg.has(face_keys[d])) { any_face = true; break; }
        if (any_face) {
            const std::string dflt = cfg.str("bc", "Periodic");
            FaceBCArray faces;
            for (int d = 0; d < 6; ++d)
                faces[d] = parse_bc_str(cfg.str(face_keys[d], dflt.c_str()));
            sc.bc.faces   = faces;
            sc.bc.variant = faces[0];
        } else {
            sc.bc.variant = parse_bc_str(cfg.str("bc", "Periodic"));
        }
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

    // === Scheme selection ===
    {
        const std::string sch = cfg.str("scheme", "weno5z");
        if (sch == "teno5a")
            sc.exec.recon = SolverConfig::ReconScheme::TENO5A;
        else if (sch == "teno7a")
            sc.exec.recon = SolverConfig::ReconScheme::TENO7A;
        else if (sch != "weno5z")
            fprintf(stderr, "[WARN] simulate_gpu: unknown scheme='%s', falling back to weno5z\n", sch.c_str());
    }

    // === Viscosity ===
    {
        const double mu         = cfg.d("mu",         0.0);
        const bool   sutherland = cfg.b("sutherland", false);
        if (mu != 0.0 || sutherland)
            fprintf(stderr, "[WARN] simulate_gpu: mu/sutherland not yet wired in simulate_gpu.cu step loop\n");
        (void)mu; (void)sutherland;
    }

    // === ACDI phase field ===
    sc.acdi.use_acdi  = cfg.b("acdi",         false);
    sc.acdi.acdi_ceps = cfg.d("acdi_ceps",    0.0);
    sc.acdi.gamma_a   = cfg.d("acdi_gamma_a", GAMMA);
    sc.acdi.gamma_b   = cfg.d("acdi_gamma_b", GAMMA);
    sc.acdi.p_inf_a   = cfg.d("acdi_pinf_a",  0.0);
    sc.acdi.p_inf_b   = cfg.d("acdi_pinf_b",  0.0);

    // === IBM ===
    sc.ibm.enabled  = cfg.b("ibm_enabled",     false);
    sc.ibm.stl_path = cfg.str("ibm_stl_path",  "");
    sc.ibm.wall_bc  = cfg.str("ibm_wall_bc",   "noslip");
    sc.ibm.u_wall   = cfg.d("ibm_u_wall",   0.0);
    sc.ibm.v_wall   = cfg.d("ibm_v_wall",   0.0);
    sc.ibm.w_wall   = cfg.d("ibm_w_wall",   0.0);
    sc.ibm.T_wall   = cfg.d("ibm_T_wall",   300.0);

    // === FSI rigid body ===
    const double fsi_mass        = cfg.d("fsi_mass",       0.0);
    const double fsi_Ix          = cfg.d("fsi_Ix",         1.0);
    const double fsi_Iy          = cfg.d("fsi_Iy",         1.0);
    const double fsi_Iz          = cfg.d("fsi_Iz",         1.0);
    const bool   fsi_prescribed  = cfg.b("fsi_prescribed", false);

    // === VTK / Metrics ===
    sc.metrics.vtk_prefix   = cfg.str("vtk_prefix",    "");
    sc.metrics.vtk_interval = cfg.i("vtk_interval",    100);
    sc.metrics.output_dir   = cfg.str("metrics_dir",   ".");
    sc.metrics.global_interval   = cfg.i("metrics_interval", 10);
    sc.metrics.residual_interval = cfg.i("metrics_residual", 0);

    // === Combustion / Arrhenius ===
    sc.physics.combustion_enabled = cfg.b("combustion",      false);
    sc.physics.arrhenius.A        = cfg.d("combustion_A",    1e4);
    sc.physics.arrhenius.T_act    = cfg.d("combustion_Tact", 10.0);
    sc.physics.arrhenius.q_heat   = cfg.d("combustion_Q",    10.0);
    sc.physics.arrhenius.n_sub    = cfg.i("combustion_nsub", 8);
    if (sc.physics.combustion_enabled)
        fprintf(stderr, "[WARN] simulate_gpu: combustion wired via gpu_source.cu — not yet in simulate_gpu.cu step loop\n");

    // === Radiation / P1 ===
    sc.physics.radiation_enabled  = cfg.b("radiation",       false);
    sc.physics.radiation.kappa    = cfg.d("radiation_kappa", 1.0);
    sc.physics.radiation.a_rad    = cfg.d("radiation_arad",  1.0);
    if (sc.physics.radiation_enabled)
        fprintf(stderr, "[WARN] simulate_gpu: radiation wired via gpu_p1.cu — not yet in simulate_gpu.cu step loop\n");

    // === WMLES ===
    sc.physics.wmles_enabled        = cfg.b("wmles",         false);
    sc.physics.wall_model.use_ode   = (cfg.str("wmles_model","reichardt") == "ode");
    if (sc.physics.wmles_enabled)
        fprintf(stderr, "[WARN] simulate_gpu: wmles wired via gpu_wmles.cu — not yet in simulate_gpu.cu step loop\n");

    // === Body force ===
    sc.physics.body_force[0] = cfg.d("body_fx", 0.0);
    sc.physics.body_force[1] = cfg.d("body_fy", 0.0);
    sc.physics.body_force[2] = cfg.d("body_fz", 0.0);

    const double domain_L = cfg.d("domain_L", 1.0);
    const double Lx = cfg.has("domain_Lx") ? cfg.d("domain_Lx", domain_L) : domain_L;
    const double Ly = cfg.has("domain_Ly") ? cfg.d("domain_Ly", domain_L) : domain_L;
    const double Lz = cfg.has("domain_Lz") ? cfg.d("domain_Lz", domain_L) : domain_L;
    const int    NX = cfg.i("domain_Nx", 0);
    const int    NY = cfg.i("domain_Ny", 0);
    const int    NZ = cfg.i("domain_Nz", 0);
    const bool   forest_domain = (NX > 0 && NY > 0 && NZ > 0);
    int    refine_levels = cfg.i("refine_levels",  0);

    std::string ckpt_load  = cfg.str("checkpoint_load",  "");
    std::string ckpt_save  = cfg.str("checkpoint_save",  "");
    int         ckpt_intvl = cfg.i("checkpoint_interval", 0);

    // ── Build IC and initialise ────────────────────────────────────────────────
    auto ic = build_ic(cfg);

    if (forest_domain)
        printf("simulate_gpu: initialising solver (Lx=%.4g Ly=%.4g Lz=%.4g NX=%d NY=%d NZ=%d ic=%s bc=%s)\n",
               Lx, Ly, Lz, NX, NY, NZ, cfg.str("ic","uniform").c_str(), cfg.str("bc","Periodic").c_str());
    else
        printf("simulate_gpu: initialising solver (Lx=%.4g Ly=%.4g Lz=%.4g ic=%s bc=%s)\n",
               Lx, Ly, Lz, cfg.str("ic","uniform").c_str(), cfg.str("bc","Periodic").c_str());

    auto do_solver_init = [&]() {
        if (forest_domain)
            solver.init(Lx, Ly, Lz, NX, NY, NZ, ic);
        else
            solver.init(Lx, Ly, Lz, ic);
    };

    if (!ckpt_load.empty()) {
        do_solver_init();
        printf("simulate_gpu: loading checkpoint from '%s'\n", ckpt_load.c_str());
        checkpoint_load(solver, ckpt_load);
    } else {
        do_solver_init();

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
        else if (auto* dm = dynamic_cast<DynamicSmagorinskyModel*>(sc.physics.sgs.get()))
            graph_solver.set_gpu_dyn_sgs(dm->Pr_t);
    }
    if (sc.acdi.use_acdi)
        graph_solver.set_gpu_acdi(sc.acdi.acdi_ceps);
    std::unique_ptr<GpuBvh> ibm_bvh;
    if (sc.ibm.enabled && !sc.ibm.stl_path.empty()) {
        try {
            uint8_t bc = 0; // NoSlip
            if (sc.ibm.wall_bc == "isothermal") bc = 2;
            ibm_bvh = std::make_unique<GpuBvh>();
            TriangleMesh stl_mesh = load_mesh(sc.ibm.stl_path);
            ibm_bvh->build(stl_mesh);
            graph_solver.set_gpu_ibm(ibm_bvh.get(), bc,
                                     (float)sc.ibm.u_wall, (float)sc.ibm.v_wall,
                                     (float)sc.ibm.w_wall, (float)sc.ibm.T_wall);
            // Flat-surface AMR criterion: refine until h ≤ mean_tri_edge / curvature_k.
            // For curved surfaces the curvature criterion already handles refinement;
            // for flat surfaces (R_c = ∞) this is the only signal.
            float h_surf = (float)cfg.d("ibm_h_surf", 0.0);
            if (h_surf == 0.0f && !stl_mesh.triangles.empty()) {
                double edge_sum = 0.0;
                int n_edges = 0;
                for (const auto& tri : stl_mesh.triangles) {
                    auto elen = [](const std::array<float,3>& a,
                                   const std::array<float,3>& b) {
                        float dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
                        return (double)sqrtf(dx*dx + dy*dy + dz*dz);
                    };
                    edge_sum += elen(tri.v0, tri.v1) + elen(tri.v1, tri.v2)
                              + elen(tri.v2, tri.v0);
                    n_edges += 3;
                }
                h_surf = (float)(edge_sum / n_edges) / 5.0f;  // curvature_k default = 5
            }
            if (h_surf > 0.0f) {
                graph_solver.set_ibm_surf_h(h_surf);
                printf("simulate_gpu: IBM flat-surface h_surf=%.4g\n", (double)h_surf);
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "Warning: IBM STL load failed: %s\n", e.what());
        }
    }
    // === FSI rigid body (requires IBM) ===
    std::unique_ptr<RigidBody6DOF> rigid_body;
    if (sc.ibm.enabled && fsi_mass > 0.0) {
        rigid_body = std::make_unique<RigidBody6DOF>();
        rigid_body->mass = fsi_mass;
        rigid_body->I[0] = fsi_Ix;
        rigid_body->I[1] = fsi_Iy;
        rigid_body->I[2] = fsi_Iz;
        graph_solver.set_rigid_body(rigid_body.get());
        graph_solver.set_rigid_prescribed(fsi_prescribed);
        printf("simulate_gpu: FSI rigid body  mass=%.4g  I=[%.4g,%.4g,%.4g]%s\n",
               fsi_mass, fsi_Ix, fsi_Iy, fsi_Iz,
               fsi_prescribed ? "  [prescribed]" : "");
    }
    graph_solver.set_ducros(sc.numerics.ducros_p_threshold,
                            1.0 / sc.numerics.ducros_blend_width);
    graph_solver.set_body_force(sc.physics.body_force[0],
                                sc.physics.body_force[1],
                                sc.physics.body_force[2]);
    auto gpu_build = [&]() {
        if (sc.bc.faces) {
            std::array<int,6> bt{};
            for (int d = 0; d < 6; ++d) bt[d] = bc_to_int((*sc.bc.faces)[d]);
            graph_solver.build_faces(solver.tree, pool, bt);
        } else {
            graph_solver.build(solver.tree, pool, bc_to_int(sc.bc.variant));
        }
    };
    gpu_build();

    solver.set_gpu_pool(&pool);
    solver.set_gpu_solver(&graph_solver);

    // ── IBM startup pre-refinement ────────────────────────────────────────────
    // Iterate gpu_regrid() until h ≤ R_c/curvature_k on every IBM leaf.
    // Runs when IBM is active and max_level > 0; terminates because depth is capped.
    if (sc.ibm.enabled && ibm_bvh && sc.amr.max_level > 0) {
        graph_solver.upload_q();
        const int bc0 = sc.bc.faces ? bc_to_int((*sc.bc.faces)[0])
                                    : bc_to_int(sc.bc.variant);
        int n_passes = 0;
        while (graph_solver.gpu_regrid(solver.tree, pool, bc0, sc.amr.max_level))
            ++n_passes;
        if (n_passes > 0) {
            gpu_build();
            printf("simulate_gpu: IBM pre-refinement: %d pass(es)  leaves=%d\n",
                   n_passes, (int)solver.tree.leaf_indices().size());
        }
    }

    printf("simulate_gpu: GPU solver active  leaves=%d\n",
           (int)solver.tree.leaf_indices().size());

    // ── Live streamer + GPU snapshot buffer (optional) ────────────────────────
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

        if (!streamer) {
            streamer = std::make_unique<LiveStreamer>(scfg);
            if (!sc.ibm.stl_path.empty()) {
                TriangleMesh gui_geom = load_mesh(sc.ibm.stl_path);
                streamer->set_geometry(gui_geom);
            }
        }
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
        gpu_build();

        printf("simulate_gpu: live feed enabled on http://localhost:%d  (var=%s axis=%d)  [GPU snap]\n",
               stream_port, sv.c_str(), (int)scfg.axis);
    }
    if (launcher_mode && streamer) streamer->set_running();

    // ── VTK snapshot buffer (when VTK requested but no live stream) ──────────
    if (!sc.metrics.vtk_prefix.empty() && !snap_buf) {
        const int n_leaves_max = static_cast<int>(solver.tree.leaf_indices().size());
        snap_buf = std::make_unique<GpuSnapshotBuffer>();
        snap_buf->alloc(std::max(n_leaves_max, 64));
        snap_buf->var_id   = 0;   // rho
        snap_buf->axis     = 2;
        snap_buf->norm_pos = 0.5f;
        snap_buf->domain_L = static_cast<float>(domain_L);
        solver.set_gpu_snapshot(snap_buf.get());
        gpu_build();   // re-build so _upload_snap_metas() populates h_metas
    }

    // ── MetricsBus (VTK, residuals, surface forces, probes) ──────────────────
    MetricsBus metrics_bus;
    if (!sc.metrics.vtk_prefix.empty() || sc.metrics.residual_interval > 0) {
        graph_solver.build_metrics(&metrics_bus, sc.metrics,
                                   snap_buf ? snap_buf->h_metas : nullptr);
        printf("simulate_gpu: metrics active  vtk_prefix='%s'  vtk_interval=%d\n",
               sc.metrics.vtk_prefix.c_str(), sc.metrics.vtk_interval);
    }

    // ── Time integration ───────────────────────────────────────────────────────
    printf("simulate_gpu: running  t_end=%.4g  max_steps=%d  cfl=%.3g  sgs=%s\n",
           sc.time.t_end, sc.time.max_steps, sc.time.cfl,
           sc.physics.sgs ? sc.physics.sgs->name() : "none");

    if (ckpt_intvl > 0 && !ckpt_save.empty()) {
        while (solver.t < sc.time.t_end && solver.step < sc.time.max_steps) {
            while (streamer && streamer->is_paused())
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (streamer && streamer->pop_checkpoint() && !ckpt_save.empty()) {
                std::string path = ckpt_save + "." + std::to_string(solver.step) + ".steer";
                checkpoint_save(solver, path);
                printf("simulate_gpu: steer checkpoint → %s\n", path.c_str());
            }
            double last_dt = solver.advance();
            if (metrics_bus.active())
                metrics_bus.write(solver.step, solver.t, last_dt);
            if (solver.step % ckpt_intvl == 0) {
                std::string path = ckpt_save + "." + std::to_string(solver.step);
                checkpoint_save(solver, path);
                printf("simulate_gpu: checkpoint → %s\n", path.c_str());
            }
        }
    } else if (metrics_bus.active()) {
        while (solver.t < sc.time.t_end && solver.step < sc.time.max_steps) {
            while (streamer && streamer->is_paused())
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (streamer && streamer->pop_checkpoint() && !ckpt_save.empty()) {
                std::string path = ckpt_save + "." + std::to_string(solver.step) + ".steer";
                checkpoint_save(solver, path);
            }
            double last_dt = solver.advance();
            metrics_bus.write(solver.step, solver.t, last_dt);
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
