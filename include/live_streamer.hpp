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

#include "../include/block_tree.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ── Variable selector ────────────────────────────────────────────────────────
enum class StreamVar : uint8_t {
    RHO   = 0,   // density ρ
    PRESS = 1,   // pressure p  (derived via EOS)
    TEMP  = 2,   // temperature T (derived via EOS)
    UMAG  = 3,   // speed |u| = √(u²+v²+w²)
    RHOU  = 4,   // x-momentum ρu
    RHOV  = 5,   // y-momentum ρv
    RHOW  = 6,   // z-momentum ρw
    ETOT  = 7,   // total energy E
};

// ── Streamer configuration ───────────────────────────────────────────────────
struct StreamConfig {
    int       port   = 8080;
    StreamVar var    = StreamVar::RHO;
    uint8_t   axis   = 2;     // 0=X, 1=Y, 2=Z
    double    pos    = 0.5;   // normalised slice position ∈ [0,1]
    int       stride = 1;     // stream every n-th advance() step
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
    std::vector<BlockDesc2D> descs;   // n_blocks entries (always sent)
    std::vector<float>       data;    // n_blocks × NB × NB float32
};

// ── LiveStreamer ─────────────────────────────────────────────────────────────
class LiveStreamer {
public:
    explicit LiveStreamer(const StreamConfig& cfg);
    ~LiveStreamer();

    // Non-copyable / non-movable (owns threads + socket)
    LiveStreamer(const LiveStreamer&)            = delete;
    LiveStreamer& operator=(const LiveStreamer&) = delete;

    // Called by NSSolver::advance() after apply_flux_correction().
    // Skips steps not divisible by cfg.stride.  Never blocks.
    void snapshot(const BlockTree& tree, int step, double t);

    // Config setters — may be called from any thread (protected by cfg_mtx_)
    void set_var (StreamVar v) noexcept;
    void set_axis(uint8_t  a) noexcept;
    void set_pos (double   p) noexcept;

    int port() const noexcept { return cfg_.port; }

private:
    StreamConfig cfg_;
    mutable std::mutex cfg_mtx_;   // guards cfg_ reads/writes across threads

    // ── Double buffer ────────────────────────────────────────────────────────
    FrameBuffer back_, front_, work_;
    std::mutex              swap_mtx_;
    std::condition_variable swap_cv_;
    bool front_fresh_ = false;
    bool shutdown_    = false;

    // ── Threads ──────────────────────────────────────────────────────────────
    std::thread accept_thread_;
    std::thread stream_thread_;

    // Active stream socket; -1 = no client connected.
    // Written by accept thread, read by stream thread.
    std::atomic<int> stream_fd_{-1};

    // ── Internal helpers ──────────────────────────────────────────────────────
    void run_accept();
    void run_stream();

    void handle_connection(int cfd);
    void handle_get_root  (int cfd);
    void handle_get_stream(int cfd);
    void handle_post_config(int cfd, const std::string& req_with_body);

    void build_frame    (const BlockTree&, int step, double t, FrameBuffer&);
    void serialize_frame(const FrameBuffer&, std::vector<uint8_t>& out);

    static const char* viewer_html();
};
