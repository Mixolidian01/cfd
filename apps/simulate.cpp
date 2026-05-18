// apps/simulate.cpp — command-line simulation runner
//
// Usage:  simulate [config.json]
//         (default: looks for sim.json in the current working directory)
//
// JSON config keys — all optional, defaults shown:
//   Solver:
//     "domain_L"           : 1.0          physical box side length [m]
//     "cfl"                : 0.8
//     "t_end"              : 1.0
//     "max_steps"          : 1000000
//     "diag_interval"      : 10
//     "bc"                 : "Periodic"   Periodic | Wall | Open
//     "verbose"            : true
//     "regrid_interval"    : 0            0 = every step
//     "max_level"          : 2
//     "sgs"                : "none"       none | Smagorinsky | Dynamic
//     "sgs_cs"             : 0.16         Smagorinsky constant (Smagorinsky only)
//     "sgs_prt"            : 0.9          turbulent Prandtl number
//     "use_lts"            : false
//     "lts_ratio"          : 2
//     "use_imex"           : false
//     "refine_levels"      : 0            extra uniform refinement passes after init
//   IC (initial condition):
//     "ic"                 : "uniform"    see below for supported names
//     "ic_rho"             : 1.0          (uniform / blast)
//     "ic_p"               : 1.0          (uniform)
//     "ic_u"               : 0.0          (uniform)
//     "ic_v"               : 0.0
//     "ic_w"               : 0.0
//     "ic_rho_l"           : 1.0          (sod) left state
//     "ic_p_l"             : 1.0
//     "ic_rho_r"           : 0.125        right state
//     "ic_p_r"             : 0.1
//     "ic_x0"              : 0.5          discontinuity position (sod)
//     "ic_ma"              : 0.1          convective Mach (taylor_green)
//     "ic_v0"              : 1.0          velocity scale (taylor_green)
//     "ic_rho0"            : 1.0          density (taylor_green)
//     "ic_du"              : 1.0          velocity jump (kelvin_helmholtz)
//     "ic_eps"             : 0.01         perturbation amplitude (kelvin_helmholtz)
//     "ic_delta"           : 0.025        interface thickness (kelvin_helmholtz)
//     "ic_p0"              : 2.5          background pressure (kelvin_helmholtz)
//     "ic_mach"            : 0.3          vortex Mach number (isentropic_vortex)
//     "ic_rc"              : 0.1          vortex core radius (isentropic_vortex)
//   Checkpointing:
//     "checkpoint_load"    : ""           path to read restart file (empty = skip)
//     "checkpoint_save"    : ""           path to write checkpoint (empty = disable)
//     "checkpoint_interval": 0            write every N steps (0 = at end only)
//   Live feed (P6):
//     "stream_port"        : 0            0 = disabled; >0 = HTTP port
//     "stream_var"         : "rho"        rho | press | temp | umag | rhou | rhov | rhow | etot
//     "stream_axis"        : 2            slice axis 0=X 1=Y 2=Z
//     "stream_pos"         : 0.5          slice position in [0,1] (fraction of domain_L)
//     "stream_stride"      : 1            stream every N steps
//     "volume_size"        : 32           N for the N³ 3D volume grid (GET /volume)
//
// Supported IC names:
//   uniform           — constant state (ic_rho, ic_p, ic_u, ic_v, ic_w)
//   sod               — 1D Riemann: left/right states split at x=ic_x0
//   taylor_green      — 3D TGV (ic_ma, ic_v0, ic_rho0)  domain should be 2π
//   kelvin_helmholtz  — 2D shear layer (ic_du, ic_eps, ic_delta, ic_p0)
//   isentropic_vortex — 2D analytical vortex (ic_mach, ic_rc)  quiet convergence test

#include "solver/ns_solver.hpp"
#include "mpi/mpi_comm.hpp"
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

// =============================================================================
// Minimal flat-JSON reader (no external deps, no nested objects)
// =============================================================================

struct Config {
    std::map<std::string, std::string> m_;

    // Load from file; calls exit(1) on open failure.
    static Config from_file(const char* path) {
        std::ifstream f(path);
        if (!f) {
            fprintf(stderr, "simulate: cannot open config file '%s'\n", path);
            exit(1);
        }
        std::string src((std::istreambuf_iterator<char>(f)), {});

        Config c;
        size_t n = src.size();
        size_t i = 0;

        auto skip = [&]() {
            while (i < n) {
                if (std::isspace((unsigned char)src[i])) { ++i; continue; }
                // strip // comments
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
            if (src[i] != '"') { ++i; continue; }    // skip { , } and other structure chars
            ++i;

            // Parse quoted key
            std::string key;
            while (i < n && src[i] != '"') key += src[i++];
            if (i < n) ++i;   // closing "

            skip();
            if (i >= n || src[i] != ':') continue;   // malformed — skip
            ++i;
            skip();
            if (i >= n) break;

            // Parse value
            std::string val;
            if (src[i] == '"') {
                ++i;
                while (i < n && src[i] != '"') {
                    if (src[i] == '\\' && i+1 < n) { ++i; val += src[i]; }
                    else val += src[i];
                    ++i;
                }
                if (i < n) ++i;   // closing "
            } else {
                // scalar: read until , } or newline or //
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

    // Typed accessors
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
    // RAII: calls MPI_Init (or is a no-op stub when compiled without MPI).
    MpiEnvironment mpi_env(argc, argv);
    const int mpi_rank = mpi_env.rank();
    const int mpi_size = mpi_env.size();

    const char* config_path = (argc >= 2) ? argv[1] : "sim.json";
    if (mpi_rank == 0)
        printf("simulate: loading config from '%s'\n", config_path);

    Config cfg = Config::from_file(config_path);

    if (mpi_rank == 0) {
        printf("simulate: configuration:\n");
        cfg.print_all();
        printf("\n");
    }

    // ── Solver config ──────────────────────────────────────────────────────────
    NSSolver solver;
    auto& sc = solver.cfg;

    sc.time.cfl             = cfg.d("cfl",             0.8);
    sc.time.t_end           = cfg.d("t_end",           1.0);
    sc.time.max_steps       = cfg.i("max_steps",       1000000);
    sc.io.diag_interval   = cfg.i("diag_interval",   10);
    sc.io.verbose         = cfg.b("verbose",         true);
    sc.amr.regrid_interval = cfg.i("regrid_interval", 0);
    sc.amr.max_level       = cfg.i("max_level",       2);
    sc.amr.use_lts         = cfg.b("use_lts",         false);
    sc.amr.lts_ratio       = cfg.i("lts_ratio",       2);
    sc.physics.use_imex        = cfg.b("use_imex",        false);

    // Boundary conditions
    {
        std::string bc_str = cfg.str("bc", "Periodic");
        if      (bc_str == "Wall")     sc.bc.variant = WallBC{};
        else if (bc_str == "Open")     sc.bc.variant = OpenBC{};
        else                           sc.bc.variant = PeriodicBC{};
    }

    // SGS model
    {
        std::string sgs_str = cfg.str("sgs", "none");
        double cs  = cfg.d("sgs_cs",  0.16);
        double prt = cfg.d("sgs_prt", 0.9);
        if (sgs_str == "Smagorinsky")
            sc.physics.sgs = std::make_shared<SmagorinskyModel>(cs, prt);
        else if (sgs_str == "Dynamic")
            sc.physics.sgs = std::make_shared<DynamicSmagorinskyModel>(prt);
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

    if (mpi_rank == 0)
        printf("simulate: initialising solver (domain_L=%.4g, ic=%s, bc=%s, ranks=%d)\n",
               domain_L, cfg.str("ic", "uniform").c_str(),
               cfg.str("bc", "Periodic").c_str(), mpi_size);

    if (!ckpt_load.empty()) {
        // Restart from checkpoint; IC is used only to set tree topology.
        solver.init(domain_L, ic);
        printf("simulate: loading checkpoint from '%s'\n", ckpt_load.c_str());
        checkpoint_load(solver, ckpt_load);
    } else {
        solver.init(domain_L, ic);

        // Optional extra uniform refinement
        if (refine_levels > 0) {
            printf("simulate: applying %d extra uniform refinement pass(es)\n",
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

    // ── MPI domain decomposition (optional) ───────────────────────────────────
    MpiPartition mpi_part;
    if (mpi_size > 1) {
        mpi_part.comm = mpi_env.comm();
        mpi_partition(solver.tree, &mpi_part);
        // Free remote blocks; each rank keeps only its owned leaves in memory.
        mpi_alloc_local_blocks(solver.tree, mpi_part,
                               domain_L / NB);  // h0 = L/NB for root block
        solver.set_mpi(&mpi_part);
        if (mpi_rank == 0)
            printf("simulate: MPI partition  ranks=%d  local_leaves=%d\n",
                   mpi_size, (int)mpi_part.local_leaves.size());
    }

    // ── Live streamer (optional) ───────────────────────────────────────────────
    std::unique_ptr<LiveStreamer> streamer;
    int stream_port = cfg.i("stream_port", 0);
    if (stream_port > 0) {
        StreamConfig scfg;
        scfg.port        = stream_port;
        scfg.axis        = static_cast<uint8_t>(cfg.i("stream_axis",   2));
        scfg.pos         = cfg.d("stream_pos",         0.5);
        scfg.stride      = cfg.i("stream_stride",      1);
        scfg.volume_size = cfg.i("volume_size",        32);  // N³ for 3D viewer

        // Map string var name → StreamVar
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
        printf("simulate: live feed enabled on http://localhost:%d  "
               "(var=%s axis=%d)\n",
               stream_port, sv.c_str(), (int)scfg.axis);
    }

    // ── Time integration ───────────────────────────────────────────────────────
    if (mpi_rank == 0)
        printf("simulate: running  t_end=%.4g  max_steps=%d  cfl=%.3g  sgs=%s\n",
               sc.time.t_end, sc.time.max_steps, sc.time.cfl,
               sc.physics.sgs ? sc.physics.sgs->name() : "none");

    if (ckpt_intvl > 0 && !ckpt_save.empty()) {
        // Manual step loop with periodic checkpoint saves
        while (solver.t < sc.time.t_end && solver.step < sc.time.max_steps) {
            solver.advance();
            if (solver.step % ckpt_intvl == 0) {
                std::string path = ckpt_save;
                // Append step number to distinguish snapshots
                path += "." + std::to_string(solver.step);
                checkpoint_save(solver, path);
                printf("simulate: checkpoint → %s\n", path.c_str());
            }
        }
    } else {
        solver.run();
    }

    // ── Final checkpoint ───────────────────────────────────────────────────────
    if (!ckpt_save.empty()) {
        checkpoint_save(solver, ckpt_save);
        printf("simulate: final checkpoint → %s\n", ckpt_save.c_str());
    }

    // ── Final diagnostics ──────────────────────────────────────────────────────
    if (mpi_rank == 0) {
        auto diag = solver.compute_diag();
        printf("\nsimulate: final state\n");
        printf("  step            = %d\n",    diag.step);
        printf("  t               = %.6e\n",  diag.t);
        printf("  mass            = %.6e\n",  diag.mass);
        printf("  kinetic energy  = %.6e\n",  diag.kinetic_energy);
        printf("  total energy    = %.6e\n",  diag.total_energy);
        printf("simulate: done.\n");
    }
    return 0;
}
