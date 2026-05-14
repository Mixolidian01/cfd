#pragma once
// Phase 6 — In-situ browser live feed
//
// LiveStreamer is a plugin attached to NSSolver (via streamer_ pointer, null by default).
// After apply_flux_correction() completes each advance(), NSSolver calls
// streamer_->snapshot(tree, step, t).  snapshot() is non-blocking: it builds the
// frame into back_, then atomically swaps with front_ under swap_mtx_ and notifies
// the stream thread.  The stream thread steals front_ into work_ (O(1) std::swap),
// serializes, and sends over an HTTP chunked binary connection.
//
// Wire protocol:
//   [4 bytes]          uint32 LE frame_len
//   [frame_len bytes]  frame body:
//     Header (32 bytes):
//       magic(4) step(4) time(8) n_blocks(1) axis(1) var_id(1) rsvd(1)
//       vmin(4) vmax(4) domain_L(4) [= 32 bytes]
//     Block descriptors (n_blocks × 16 bytes each):
//       ox2d(4) oy2d(4) h(4) level(1) pad(3)
//     Data:
//       n_blocks × NB × NB × float32 (row-major, dim0=first non-slice axis)
//
// HTTP endpoints (all on cfg.port, localhost):
//   GET /        -> embedded HTML viewer (single-page, no external deps)
//   GET /stream  -> chunked octet-stream of frames (long-lived)
//   POST /config -> JSON {var:int, axis:int, pos:float}  -> 200 OK
//
// Threading model:
//   Solver thread:  snapshot() — writes back_, swaps under swap_mtx_
//   Stream thread:  run_stream() — steals front_ into work_, sends chunks
//   Accept thread:  run_accept() — accepts TCP connections, spawns per-conn threads

#include "mesh/block_tree.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Free function defined in viewer_html.cpp (R9-E1 extraction).
const char* viewer_html();

// ── Variable selector ────────────────────────────────────────────────────────
enum class StreamVar : uint8_t {
    RHO      = 0,   // density ρ
    PRESS    = 1,   // pressure p  (derived via EOS)
    TEMP     = 2,   // temperature T (derived via EOS)
    UMAG     = 3,   // speed |u| = √(u²+v²+w²)
    RHOU     = 4,   // x-momentum ρu
    RHOV     = 5,   // y-momentum ρv
    RHOW     = 6,   // z-momentum ρw
    ETOT     = 7,   // total energy E
    // P12.5 — derived fields (gradient-based, use ghost cells)
    MACH     = 8,   // local Mach = |u|/c,  c=√(γp/ρ)
    VORT     = 9,   // vorticity magnitude |ω| = |∇×u|
    QCRIT    = 10,  // Q-criterion = -½ tr(A·A^T),  A_ij=∂u_i/∂x_j
    SCHLIEREN= 11,  // numerical schlieren |∇ρ|
};

// ── Streamer configuration ───────────────────────────────────────────────────
struct StreamConfig {
    int       port        = 8080;
    StreamVar var         = StreamVar::RHO;
    uint8_t   axis        = 2;    // 0=X, 1=Y, 2=Z
    double    pos         = 0.5;  // normalised slice position ∈ [0,1]
    int       stride      = 1;    // stream every n-th advance() step
    int       volume_size = 32;   // N for the N³ uniform volume grid (P6.6)
};

// ── Block descriptor in the 2-D projected plane ─────────────────────────────
struct BlockDesc2D {
    float   ox2d, oy2d;   // origin in the projected 2-D plane (8 bytes)
    float   h;            // cell size  (4 bytes)
    uint8_t level;        // AMR refinement level (1 byte)
    uint8_t pad[3];       // (3 bytes)
};                        // total: 16 bytes
static_assert(sizeof(BlockDesc2D) == 16, "BlockDesc2D must be 16 bytes");

// ── Frame ready for serialisation ───────────────────────────────────────────
struct FrameBuffer {
    int32_t step      = 0;
    double  sim_time  = 0.0;
    uint8_t n_blocks  = 0;
    uint8_t axis      = 2;
    uint8_t var_id    = 0;
    float   g_vmin    = 0.f;
    float   g_vmax    = 1.f;
    float   domain_L  = 1.f;
    std::vector<BlockDesc2D>           descs;  // n_blocks entries (always sent)
    std::vector<float>                 data;   // n_blocks × NB × NB float32
    std::vector<std::array<float, 8>>  probe;  // P12.2: all 8 vars per cell (same layout as data)
};

// ── 3-D volume frame (P6.6) ──────────────────────────────────────────────────
// Wire format:
//   [4]  uint32 LE frame_len
//   [40] header:
//     magic(4) step(4) time(8) nx(2) ny(2) nz(2) pad(2)
//     g_vmin(4) g_vmax(4) domain_L(4) var_id(1) compressed(1) pad(2)
//   data:
//     compressed=0 : nx*ny*nz float32 (z-major: data[vk*N²+vj*N+vi])
//     compressed=1 : [4] uint32 unc_bytes + LZ4(nx*ny*nz uint16 LE)
struct FrameBuffer3D {
    int32_t  step      = 0;
    double   sim_time  = 0.0;
    uint16_t nx = 32, ny = 32, nz = 32;
    float    g_vmin    = 0.f;
    float    g_vmax    = 1.f;
    float    domain_L  = 1.f;
    uint8_t  var_id    = 0;
    std::vector<float> data;   // nx*ny*nz
};

// ── P12.1: per-step solver diagnostics (exposed via GET /metrics) ────────────
struct MetricsSnapshot {
    int    step              = 0;
    double t                 = 0.0;
    double dt                = 0.0;
    double cfl               = 0.0;
    double ke                = 0.0;
    double mass              = 0.0;
    int    n_leaves          = 0;
    double rho_min           = 0.0;
    double rho_max           = 0.0;
    double wall_time_ms      = 0.0;  // elapsed since LiveStreamer construction
    int    leaves_per_level[8] = {}; // leaf count per AMR level (index 0..7)
    bool   gpu_active        = false;
    // P12.3: relative conservation errors (|val-val0|/|val0|, 0 on step 0)
    double mass_error        = 0.0;
    double momentum_error    = 0.0;
    double energy_error      = 0.0;
};

// ── LiveStreamer ─────────────────────────────────────────────────────────────
class LiveStreamer {
public:
    explicit LiveStreamer(const StreamConfig& cfg);
    ~LiveStreamer();

    LiveStreamer(const LiveStreamer&)            = delete;
    LiveStreamer& operator=(const LiveStreamer&) = delete;

    // Called by NSSolver::advance() after apply_flux_correction(). Never blocks.
    void snapshot(const BlockTree& tree, int step, double t);

    // P12.1: push latest solver diagnostics (called from NSSolver::advance()).
    void push_metrics(const MetricsSnapshot& m);

    void set_var (StreamVar v) noexcept;
    void set_axis(uint8_t  a) noexcept;
    void set_pos (double   p) noexcept;

    int port() const noexcept { return cfg_.port; }

private:
    StreamConfig cfg_;
    mutable std::mutex cfg_mtx_;

    // ── P12.1: latest metrics snapshot ──────────────────────────────────────
    MetricsSnapshot metrics_;
    std::mutex      metrics_mtx_;
    std::chrono::steady_clock::time_point t_start_;

    // ── 2-D slice double-buffer ──────────────────────────────────────────────
    FrameBuffer back_, front_, work_;
    std::mutex              swap_mtx_;
    std::condition_variable swap_cv_;
    bool front_fresh_ = false;
    bool shutdown_    = false;

    // ── 3-D volume double-buffer (P6.6) ──────────────────────────────────────
    FrameBuffer3D back3d_, front3d_, work3d_;
    std::mutex              swap3d_mtx_;
    std::condition_variable swap3d_cv_;
    bool front3d_fresh_ = false;

    // ── Threads ──────────────────────────────────────────────────────────────
    std::thread accept_thread_;
    std::thread stream_thread_;
    std::thread stream3d_thread_;

    std::atomic<int> stream_fd_{-1};     // 2-D slice stream socket
    std::atomic<int> vol_stream_fd_{-1}; // 3-D volume stream socket

    // ── Internal helpers ──────────────────────────────────────────────────────
    void run_accept();
    void run_stream();
    void run_stream3d();   // P6.6

    void handle_connection(int cfd);
    void handle_get_root        (int cfd);
    void handle_get_stream      (int cfd);
    void handle_get_volume      (int cfd);        // P6.6 — WebGPU viewer HTML
    void handle_get_vol_stream  (int cfd);        // P6.6 — 3-D chunked stream
    void handle_post_config     (int cfd, const std::string& req_with_body);
    void handle_get_metrics     (int cfd);
    void handle_post_probe      (int cfd, const std::string& req_with_body);

    void build_frame    (const BlockTree&, int step, double t, FrameBuffer&);
    void serialize_frame(const FrameBuffer&, std::vector<uint8_t>& out);

    void build_volume    (const BlockTree&, int step, double t, FrameBuffer3D&);
    void serialize_volume(const FrameBuffer3D&, std::vector<uint8_t>& out);

    std::string viewer_html_3d();          // P6.6 — WebGPU ray marcher (needs port)
};
