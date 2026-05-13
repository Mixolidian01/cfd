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

#include "../include/ns_solver.hpp"
#include "../include/live_streamer.hpp"
#include "../include/checkpoint.hpp"
#include "../include/sgs.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
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

// =============================================================================
// re-fill all leaf blocks with IC (needed after manual refinement)
// =============================================================================

static void fill_leaves(NSSolver& s,
                        const std::function<Prim(double, double, double)>& ic)
{
    for (int li : s.tree.leaf_indices()) {
        auto& blk = *s.tree.nodes[li].block;
        for (int k = 0; k < NB2; ++k)
        for (int j = 0; j < NB2; ++j)
        for (int ii = 0; ii < NB2; ++ii) {
            double x = blk.ox + (ii - NG + 0.5) * blk.h;
            double y = blk.oy + (j  - NG + 0.5) * blk.h;
            double z = blk.oz + (k  - NG + 0.5) * blk.h;
            Prim   p = ic(x, y, z);
            int    idx = cell_idx(ii, j, k);
            blk.Q[0][idx] = p.rho;
            blk.Q[1][idx] = p.rho * p.u;
            blk.Q[2][idx] = p.rho * p.v;
            blk.Q[3][idx] = p.rho * p.w;
            blk.Q[4][idx] = p.p / (GAMMA - 1.0)
                           + 0.5 * p.rho * (p.u*p.u + p.v*p.v + p.w*p.w);
        }
    }
}

// =============================================================================
// IC builder — returns a lambda matching NSSolver::init() signature
// =============================================================================

static std::function<Prim(double, double, double)>
build_ic(const Config& cfg)
{
    std::string name = cfg.str("ic", "uniform");

    if (name == "uniform") {
        double rho = cfg.d("ic_rho", 1.0);
        double p   = cfg.d("ic_p",   1.0);
        double u   = cfg.d("ic_u",   0.0);
        double v   = cfg.d("ic_v",   0.0);
        double w   = cfg.d("ic_w",   0.0);
        return [=](double, double, double) -> Prim {
            Prim q{};
            q.rho = rho; q.u = u; q.v = v; q.w = w; q.p = p;
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "sod") {
        double rho_l = cfg.d("ic_rho_l", 1.0);
        double p_l   = cfg.d("ic_p_l",   1.0);
        double rho_r = cfg.d("ic_rho_r", 0.125);
        double p_r   = cfg.d("ic_p_r",   0.1);
        double x0    = cfg.d("ic_x0",    0.5);
        return [=](double x, double, double) -> Prim {
            Prim q{};
            bool left = (x < x0);
            q.rho = left ? rho_l : rho_r;
            q.p   = left ? p_l   : p_r;
            q.u = 0.0; q.v = 0.0; q.w = 0.0;
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "taylor_green") {
        double ma   = cfg.d("ic_ma",   0.1);
        double v0   = cfg.d("ic_v0",   1.0);
        double rho0 = cfg.d("ic_rho0", 1.0);
        double p0   = rho0 * v0 * v0 / (GAMMA * ma * ma);
        return [=](double x, double y, double z) -> Prim {
            Prim q{};
            q.rho = rho0;
            q.u   =  v0 * std::sin(x) * std::cos(y) * std::cos(z);
            q.v   = -v0 * std::cos(x) * std::sin(y) * std::cos(z);
            q.w   =  0.0;
            q.p   = p0 + rho0*v0*v0/16.0
                       * (std::cos(2.0*x) + std::cos(2.0*y))
                       * (std::cos(2.0*z) + 2.0);
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "kelvin_helmholtz") {
        double du    = cfg.d("ic_du",    1.0);
        double eps   = cfg.d("ic_eps",   0.01);
        double delta = cfg.d("ic_delta", 0.025);
        double p0    = cfg.d("ic_p0",    2.5);
        double L     = cfg.d("domain_L", 1.0);
        return [=](double x, double y, double) -> Prim {
            Prim q{};
            bool dense = (y >= 0.25*L && y <= 0.75*L);
            double sign = dense ? 1.0 : -1.0;
            double d1 = std::tanh((y - 0.25*L) / delta);
            double d2 = std::tanh((0.75*L - y) / delta);
            double taper = 0.5 * (1.0 + d1) * 0.5 * (1.0 + d2);
            q.rho = dense ? 2.0 : 1.0;
            q.u   = sign * 0.5 * du;
            q.v   = eps * du * std::sin(2.0 * M_PI * x / L) * taper;
            q.w   = 0.0;
            q.p   = p0;
            q.T   = q.p / (q.rho * R_GAS);
            q.c   = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    if (name == "isentropic_vortex") {
        // 2D isentropic vortex: analytical solution to smooth Euler equations.
        // Superimposes a vortex of strength Γ on a uniform stream (u∞, 0, 0).
        // ρ(r) = ρ∞ · (1 − β²(γ−1)/(8π²γ) · e^(1−r²/rc²))^(1/(γ−1))
        double mach = cfg.d("ic_mach", 0.3);
        double rc   = cfg.d("ic_rc",   0.1);
        double L    = cfg.d("domain_L", 1.0);
        // Ambient: uniform flow at ic_mach
        double p_inf  = 1.0 / GAMMA;          // p∞ so that c∞=1 → u∞ = mach
        double rho_inf = 1.0;
        double u_inf   = mach;                 // uniform x-velocity
        // Vortex amplitude: circulation-based, parameterised by ic_mach
        double beta = mach;                    // vortex strength in units of mach
        double xc = 0.5*L, yc = 0.5*L;        // vortex centre
        return [=](double x, double y, double) -> Prim {
            double dx = x - xc, dy = y - yc;
            double r2 = (dx*dx + dy*dy) / (rc*rc);
            double f  = beta * beta * (GAMMA - 1.0) / (8.0 * M_PI * M_PI * GAMMA);
            double fac = std::exp(1.0 - r2);
            double T_ratio = 1.0 - f * fac;
            double rho = rho_inf * std::pow(T_ratio, 1.0/(GAMMA-1.0));
            double p   = p_inf   * std::pow(T_ratio, GAMMA/(GAMMA-1.0));
            double du  = -beta / (2.0 * M_PI * rc) * dy * std::exp(0.5*(1.0 - r2));
            double dv  =  beta / (2.0 * M_PI * rc) * dx * std::exp(0.5*(1.0 - r2));
            Prim q{};
            q.rho = rho; q.p = p;
            q.u = u_inf + du; q.v = dv; q.w = 0.0;
            q.T = q.p / (q.rho * R_GAS);
            q.c = std::sqrt(GAMMA * q.p / q.rho);
            return q;
        };
    }

    fprintf(stderr, "simulate: unknown ic '%s'. "
            "Valid: uniform sod taylor_green kelvin_helmholtz isentropic_vortex\n",
            name.c_str());
    exit(1);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[])
{
    const char* config_path = (argc >= 2) ? argv[1] : "sim.json";
    printf("simulate: loading config from '%s'\n", config_path);

    Config cfg = Config::from_file(config_path);

    printf("simulate: configuration:\n");
    cfg.print_all();
    printf("\n");

    // ── Solver config ──────────────────────────────────────────────────────────
    NSSolver solver;
    auto& sc = solver.cfg;

    sc.cfl             = cfg.d("cfl",             0.8);
    sc.t_end           = cfg.d("t_end",           1.0);
    sc.max_steps       = cfg.i("max_steps",       1000000);
    sc.diag_interval   = cfg.i("diag_interval",   10);
    sc.verbose         = cfg.b("verbose",         true);
    sc.regrid_interval = cfg.i("regrid_interval", 0);
    sc.max_level       = cfg.i("max_level",       2);
    sc.use_lts         = cfg.b("use_lts",         false);
    sc.lts_ratio       = cfg.i("lts_ratio",       2);
    sc.use_imex        = cfg.b("use_imex",        false);

    // Boundary conditions
    {
        std::string bc_str = cfg.str("bc", "Periodic");
        if      (bc_str == "Wall")     sc.bc_variant = WallBC{};
        else if (bc_str == "Open")     sc.bc_variant = OpenBC{};
        else                           sc.bc_variant = PeriodicBC{};
    }

    // SGS model
    {
        std::string sgs_str = cfg.str("sgs", "none");
        double cs  = cfg.d("sgs_cs",  0.16);
        double prt = cfg.d("sgs_prt", 0.9);
        if (sgs_str == "Smagorinsky")
            sc.sgs = std::make_shared<SmagorinskyModel>(cs, prt);
        else if (sgs_str == "Dynamic")
            sc.sgs = std::make_shared<DynamicSmagorinskyModel>(prt);
        else
            sc.sgs = nullptr;
    }

    double domain_L      = cfg.d("domain_L",      1.0);
    int    refine_levels = cfg.i("refine_levels",  0);

    std::string ckpt_load  = cfg.str("checkpoint_load",  "");
    std::string ckpt_save  = cfg.str("checkpoint_save",  "");
    int         ckpt_intvl = cfg.i("checkpoint_interval", 0);

    // ── Build IC and initialise ────────────────────────────────────────────────
    auto ic = build_ic(cfg);

    printf("simulate: initialising solver (domain_L=%.4g, ic=%s, bc=%s)\n",
           domain_L, cfg.str("ic", "uniform").c_str(), cfg.str("bc", "Periodic").c_str());

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
    printf("simulate: running  t_end=%.4g  max_steps=%d  cfl=%.3g  sgs=%s\n",
           sc.t_end, sc.max_steps, sc.cfl, sc.sgs ? sc.sgs->name() : "none");

    if (ckpt_intvl > 0 && !ckpt_save.empty()) {
        // Manual step loop with periodic checkpoint saves
        while (solver.t < sc.t_end && solver.step < sc.max_steps) {
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
    auto diag = solver.compute_diag();
    printf("\nsimulate: final state\n");
    printf("  step            = %d\n",    diag.step);
    printf("  t               = %.6e\n",  diag.t);
    printf("  mass            = %.6e\n",  diag.mass);
    printf("  kinetic energy  = %.6e\n",  diag.kinetic_energy);
    printf("  total energy    = %.6e\n",  diag.total_energy);
    printf("simulate: done.\n");
    return 0;
}
