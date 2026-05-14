#pragma once
// DESIGN.md reference: Layer 3 — Time Loop
// P8.1: GpuPool forward-declared so ns_solver.hpp compiles without CUDA headers.
//
// NSSolver owns the BlockTree and drives the SSP-RK3 time integration.
// It knows nothing about linear algebra (Layer 0) directly.
//
// SSP-RK3 (Shu-Osher):
//   Q^(1) = Q^n + dt * L(Q^n)
//   Q^(2) = 3/4 * Q^n + 1/4 * (Q^(1) + dt * L(Q^(1)))
//   Q^(n+1) = 1/3 * Q^n + 2/3 * (Q^(2) + dt * L(Q^(2)))
//
// Each stage: fill ghosts → compute rhs → update Q.
// dt is recomputed from CFL at the start of each step.

#include "mesh/bc_types.hpp"
#include <memory>
#include <stdexcept>
#include "models/sgs.hpp"
#include "schemes/operators.hpp"
#include "io/live_streamer.hpp"
#include "mpi/mpi_comm.hpp"
#include <functional>
#include <string>

// Forward declaration — full definition in gpu_pool.hpp (CUDA TU only).
struct GpuPool;

// P10-A2: TimeIntegrator — common interface for all SSP-RK3 implementations.
//
// Implementations:
//   CpuRk3Integrator  (src/cpu_rk3.cpp)         — CPU flat-tree SSP-RK3
//   LtsIntegrator     (src/lts_integrator.cpp)  — Berger-Oliger local time stepping
//   GpuGraphSolver    (src/cuda/gpu_graph.cu)   — IGpuSolver : TimeIntegrator
//
// NSSolver::advance() dispatches to integrator_->step(tree, cfg.cfl) after
// handling early-exit paths (IMEX, LTS, GPU flat-tree).
struct TimeIntegrator {
    virtual double step(BlockTree& tree, double cfl) = 0;
    virtual ~TimeIntegrator() = default;
};

// Forward declarations — full definitions in include/cpu_rk3.hpp and include/lts_integrator.hpp.
struct CpuRk3Integrator;
struct LtsIntegrator;

// IGpuSolver extends TimeIntegrator with GPU lifecycle (build, download_q, upload_q).
// Implemented by GpuGraphSolver in src/cuda/gpu_graph.cu (CUDA TU only).
struct IGpuSolver : TimeIntegrator {
    // Bridge: TimeIntegrator::step() delegates to the GPU-named advance().
    double step(BlockTree& tree, double cfl) final { return advance(tree, cfl); }
    virtual double advance(const BlockTree& tree, double cfl)                      = 0;
    virtual void   download_q(const BlockTree& tree)                         const = 0;
    // P11.8: re-upload CPU Q → GPU after CPU fallback steps (AMR path).
    virtual void   upload_q()                                                 const = 0;
    virtual void   build(const BlockTree& tree, const GpuPool& pool, int bc_type) = 0;
};

// ── Diagnostics written every `diag_interval` steps ───────────────────────────────────
struct StepDiag {
    int    step;
    double t;
    double dt;
    double mass;        // total mass (should be conserved)
    double momentum_x;  // total x-momentum
    double kinetic_energy;
    double total_energy;
};

// ── Solver configuration ─────────────────────────────────────────────────────────────────────────────
//
// Fields are decomposed into eight typed sub-structs.  All defaults produce
// a valid single-phase periodic-BC CPU run.  Call validate() after any change.
//
struct SolverConfig {
    // R4: Backend tag dispatch and flux scheme.
    enum class ExecutionBackend { CPU, GPU };
    enum class FluxScheme       { HLLC, HLLC_ES };

    // ── 1. Execution backend ──────────────────────────────────────────────
    struct ExecConfig {
        ExecutionBackend backend     = ExecutionBackend::CPU;
        FluxScheme       flux_scheme = FluxScheme::HLLC_ES;
        bool             use_gpu     = false;  // P8.1: GPU memory pool path
    } exec;

    // ── 2. Time integration ───────────────────────────────────────────────
    struct TimeConfig {
        double cfl       = 0.8;       // CFL number ∈ (0, 1]
        double t_end     = 1.0;       // stop time
        int    max_steps = 1000000;   // hard step cap
    } time;

    // ── 3. Boundary conditions ────────────────────────────────────────────
    struct BcConfig {
        BCVariant variant = PeriodicBC{};

        // P13.4: wall temperature for isothermal no-slip walls (WallBC/ContactAngleBC).
        // 0.0 (default) → adiabatic wall (∂T/∂n = 0).
        // > 0 → isothermal: ghost E enforces T_ghost = 2*wall_T − T_interior.
        double wall_T = 0.0;
    } bc;

    // ── 4. AMR + local time stepping ──────────────────────────────────────
    struct AmrConfig {
        int  max_level       = 2;   // max refinement depth (0 = flat tree)
        int  regrid_interval = 0;   // steps between regrid (0 = disabled)

        // P4.1: Berger-Oliger local time stepping.
        bool use_lts   = false;
        int  lts_ratio = 2;   // refinement ratio (must match tree's geometric ratio)
    } amr;

    // ── 5. Physics: SGS turbulence + IMEX ────────────────────────────────
    struct PhysicsConfig {
        std::shared_ptr<SGSModel> sgs = nullptr;

        // P3.5: IMEX-ARK implicit viscous solve.
        bool use_imex  = false;
        int  mg_levels = 3;
    } physics;

    // ── 6. Numerical sensors ──────────────────────────────────────────────
    struct NumericsConfig {
        // P11.6: Ducros pressure-ratio sensor (controls WENO5 → central switch).
        // ducros_p_threshold: |Δp|/p below this → central scheme.
        // ducros_blend_width: ramp width above threshold (endpoint = thr + width → 1).
        // For DNS/LES without shocks, raise threshold (e.g. 0.5) to avoid HLLC-ES.
        double ducros_p_threshold = 0.1;
        double ducros_blend_width = 0.1;

        // P13.5: SBP-SAT penalty at AMR C/F interfaces.
        // 0.0 (default) → pure Berger-Colella correction.
        // > 0 → add σ = (tau/h_f)·(Q_ghost − Q_interior) penalty each RK3 stage.
        //        Recommended: tau = 0.5 (minimal energy-stable penalty).
        double sat_tau = 0.0;
    } numerics;

    // ── 7. ACDI compressible multiphase ───────────────────────────────────
    struct AcdiConfig {
        // P14.1: Accurate Conservative Diffuse Interface.
        // false → single-phase mode (phi field inactive, zero overhead).
        // true  → φ ∈ [0,1] advected alongside Q; set IC via NSSolver::init().
        bool   use_acdi  = false;

        // P14.1b: compression coefficient Cε; interface thickness ε = Cε·h.
        // 0.0 → pure advection; >0 → adds sharpening source to each RK3 stage.
        double acdi_ceps = 0.0;

        // P14.1c: Stiffened-gas EOS.  Allaire (2002) mixture rule:
        //   1/(γ_m−1) = φ/(γ_A−1) + (1−φ)/(γ_B−1).
        // Defaults reproduce ideal gas (γ=1.4, p∞=0) for identical fluids.
        // Active only when use_acdi=true AND fluids differ.
        double gamma_a = GAMMA;   // γ for fluid A (φ=1), e.g. 6.12 for liquid water
        double gamma_b = GAMMA;   // γ for fluid B (φ=0), e.g. 1.4  for air
        double p_inf_a = 0.0;     // p∞ [Pa] for fluid A, e.g. 3.43e8 for liquid water
        double p_inf_b = 0.0;     // p∞ [Pa] for fluid B (0 = ideal gas)

        // P14.2: static contact angle θ_w [deg] at wall (ContactAngleBC + use_acdi).
        // 90° (default) → ∂φ/∂n=0 (neutral).  0° → fully wetting.  180° → non-wetting.
        // Activate via bc.variant = ContactAngleBC{theta_deg}; requires acdi_ceps > 0.
    } acdi;

    // ── 8. I/O ────────────────────────────────────────────────────────────
    struct IoConfig {
        bool verbose      = true;
        bool verbose_json = false;
        int  diag_interval = 10;
    } io;

    void validate() const;
};

// ── NSSolver ──────────────────────────────────────────────────────────────────────────────────────
struct NSSolver {
    BlockTree    tree;
    SolverConfig cfg;
    double       t    = 0.0;
    int          step = 0;

    std::vector<StepDiag> history;

    // P14.1: phi_ic (optional) sets the initial phase-field φ(x,y,z) ∈ [0,1].
    // Pass nullptr (or omit) for single-phase runs (φ≡0).
    void init(double domain_L,
              const std::function<Prim(double,double,double)>& ic,
              const std::function<double(double,double,double)>* phi_ic = nullptr);
    void run();
    double advance();
    void regrid();

    StepDiag compute_diag() const;
    void     print_diag(const StepDiag& d) const;

    // Phase 6: optional in-situ browser live feed.
    // Set via set_streamer() before run()/advance().  Null = disabled.
    LiveStreamer* streamer_ = nullptr;
    void set_streamer(LiveStreamer* s) noexcept { streamer_ = s; }

    // P7.1: optional MPI partition.  When set, advance() calls mpi_exchange_halos()
    // before each RK stage's ghost fill, and uses MPI_Allreduce for CFL and diagnostics.
    MpiPartition* mpi_ = nullptr;
    void set_mpi(MpiPartition* p) noexcept {
        mpi_ = p;
        tree.set_mpi(p);
    }

    // P8.1: optional GPU memory pool.  When cfg.use_gpu=true, init() creates the
    // pool and wires BlockTree lifecycle callbacks.  All leaf blocks are uploaded
    // to GPU after IC application.  Caller may also inject an external pool via
    // set_gpu_pool() before init() to share a pool across solvers.
    GpuPool* gpu_pool_ = nullptr;
    void set_gpu_pool(GpuPool* p) noexcept { gpu_pool_ = p; }

    // P10-A2: optional GPU time integrator (flat-tree runs).
    // IGpuSolver implements TimeIntegrator::step() via advance() + build() + download_q().
    // Caller creates a GpuGraphSolver (CUDA TU) and injects it here before advance().
    // When set, advance() dispatches the SSP-RK3 loop to the GPU via gpu_solver_->step().
    // NSSolver::regrid() calls build() after every topology change.
    IGpuSolver* gpu_solver_ = nullptr;
    void set_gpu_solver(IGpuSolver* s) noexcept { gpu_solver_ = s; }

    int  scratch_leaf_count_ = -1;  ///< FIX P5: tracks last alloc size
    void alloc_scratch();

private:
    friend struct CpuRk3Integrator;  // P10-A2: needs rhs_, Qn_, Qs_, save_Qn()
    friend struct LtsIntegrator;     // P10-A2: needs rhs_, Qn_, Qs_

    std::unique_ptr<TimeIntegrator> integrator_;      // P10-A2: CPU SSP-RK3 integrator
    std::unique_ptr<TimeIntegrator> lts_integrator_;  // P10-A2: Berger-Oliger LTS integrator

    std::vector<CellBlock> rhs_;
    std::vector<CellBlock> Qn_;
    std::vector<CellBlock> Qs_;

    // FIX P0.6: was a static local inside advance(); promoted to member so
    // that multiple NSSolver instances do not share state, and so that
    // reset on init() is explicit. Sentinel -1.0 → first step residual=0.
    double ke_prev_  = -1.0;
    double last_dt_  = 0.0;   // B4: populated by advance(), exposed via compute_diag()
    // P12.3: step-0 conservation baselines (sentinel -1 = uninitialized)
    double mass0_    = -1.0;
    double mtm0_     = -1.0;  // |total momentum| = sqrt(px²+py²+pz²)
    // P11.8: CPU path was used last step → GPU Q is stale; re-upload before next GPU step.
    bool gpu_q_stale_ = false;
    double energy0_  = -1.0;

    void save_Qn();
    void copy_stage_to_tree(const std::vector<CellBlock>& stage);
    void copy_tree_to_stage(std::vector<CellBlock>& stage);

    // P3.5: IMEX advance — implicit viscous Helmholtz correction after RK3.
    double advance_imex();
};
