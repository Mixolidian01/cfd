// Phase 6 — In-situ browser live feed implementation
//
// See include/live_streamer.hpp for architecture notes.
//
// POSIX socket API — Linux only (AF_INET, MSG_NOSIGNAL, SO_REUSEADDR).

#include "io/live_streamer.hpp"
#include "mesh/cell_block.hpp"
#include "io/http_server.hpp"    // http_safe_send, http_read_request, …
#include "io/frame_packer.hpp"   // serialize_frame, serialize_volume
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

// POSIX sockets
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// =============================================================================
// Embedded HTML viewer (Phase A — 2D slice, viridis colormap, canvas API)
// =============================================================================
// viewer_html() is defined in viewer_html.cpp (R9-E1). Declaration in
// include/live_streamer.hpp. The function has external linkage (not static).

// (function body moved to src/viewer_html.cpp)


// =============================================================================
// P6.6 — 3-D volume helpers
// =============================================================================

// Resample all AMR leaf blocks onto a uniform N×N×N grid.
// Leaves are processed coarse-to-fine so finer data overwrites coarser.
void LiveStreamer::build_volume(const BlockTree& tree, int step, double t,
                                FrameBuffer3D& fb)
{
    StreamConfig cfg_snap;
    {
        std::lock_guard<std::mutex> lk(cfg_mtx_);
        cfg_snap = cfg_;
    }

    const int    N  = std::max(4, std::min(128, cfg_snap.volume_size));
    const double L  = tree.domain_L();
    const StreamVar svar = cfg_snap.var;

    fb.step     = step;
    fb.sim_time = t;
    fb.nx = fb.ny = fb.nz = static_cast<uint16_t>(N);
    fb.domain_L = static_cast<float>(L);
    fb.var_id   = static_cast<uint8_t>(svar);
    fb.data.assign(static_cast<size_t>(N) * N * N, 0.f);

    float g_vmin = std::numeric_limits<float>::max();
    float g_vmax = std::numeric_limits<float>::lowest();

    auto vel_at = [](const CellBlock& blk, int comp, int i, int j, int k) -> double {
        switch (comp) {
            case 0: return blk.rhou(i,j,k) / blk.rho(i,j,k);
            case 1: return blk.rhov(i,j,k) / blk.rho(i,j,k);
            default: return blk.rhow(i,j,k) / blk.rho(i,j,k);
        }
    };

    auto cell_val = [&](const CellBlock& blk, int i, int j, int k) -> float {
        switch (svar) {
            case StreamVar::RHO:   return static_cast<float>(blk.rho (i,j,k));
            case StreamVar::RHOU:  return static_cast<float>(blk.rhou(i,j,k));
            case StreamVar::RHOV:  return static_cast<float>(blk.rhov(i,j,k));
            case StreamVar::RHOW:  return static_cast<float>(blk.rhow(i,j,k));
            case StreamVar::ETOT:  return static_cast<float>(blk.E   (i,j,k));
            default: break;
        }
        Prim q = blk.prim(i, j, k);
        switch (svar) {
            case StreamVar::PRESS: return static_cast<float>(q.p);
            case StreamVar::TEMP:  return static_cast<float>(q.T);
            case StreamVar::UMAG:  return static_cast<float>(
                                       std::sqrt(q.u*q.u + q.v*q.v + q.w*q.w));
            case StreamVar::MACH: {
                const double c = std::sqrt(GAMMA * q.p / q.rho);
                return static_cast<float>(std::sqrt(q.u*q.u+q.v*q.v+q.w*q.w) / c);
            }
            case StreamVar::VORT: {
                const double ih2 = 1.0 / (2.0 * blk.h);
                const double wx = (vel_at(blk,2,i,j+1,k)-vel_at(blk,2,i,j-1,k))*ih2
                                - (vel_at(blk,1,i,j,k+1)-vel_at(blk,1,i,j,k-1))*ih2;
                const double wy = (vel_at(blk,0,i,j,k+1)-vel_at(blk,0,i,j,k-1))*ih2
                                - (vel_at(blk,2,i+1,j,k)-vel_at(blk,2,i-1,j,k))*ih2;
                const double wz = (vel_at(blk,1,i+1,j,k)-vel_at(blk,1,i-1,j,k))*ih2
                                - (vel_at(blk,0,i,j+1,k)-vel_at(blk,0,i,j-1,k))*ih2;
                return static_cast<float>(std::sqrt(wx*wx+wy*wy+wz*wz));
            }
            case StreamVar::QCRIT: {
                const double ih2 = 1.0 / (2.0 * blk.h);
                const double A[3][3] = {
                    {(vel_at(blk,0,i+1,j,k)-vel_at(blk,0,i-1,j,k))*ih2,
                     (vel_at(blk,0,i,j+1,k)-vel_at(blk,0,i,j-1,k))*ih2,
                     (vel_at(blk,0,i,j,k+1)-vel_at(blk,0,i,j,k-1))*ih2},
                    {(vel_at(blk,1,i+1,j,k)-vel_at(blk,1,i-1,j,k))*ih2,
                     (vel_at(blk,1,i,j+1,k)-vel_at(blk,1,i,j-1,k))*ih2,
                     (vel_at(blk,1,i,j,k+1)-vel_at(blk,1,i,j,k-1))*ih2},
                    {(vel_at(blk,2,i+1,j,k)-vel_at(blk,2,i-1,j,k))*ih2,
                     (vel_at(blk,2,i,j+1,k)-vel_at(blk,2,i,j-1,k))*ih2,
                     (vel_at(blk,2,i,j,k+1)-vel_at(blk,2,i,j,k-1))*ih2}
                };
                double Q = 0.0;
                for (int c = 0; c < 3; ++c)
                    for (int d = 0; d < 3; ++d)
                        Q -= 0.5 * A[c][d] * A[d][c];
                return static_cast<float>(Q);
            }
            case StreamVar::SCHLIEREN: {
                const double ih2 = 1.0 / (2.0 * blk.h);
                const double drx = (blk.rho(i+1,j,k)-blk.rho(i-1,j,k))*ih2;
                const double dry = (blk.rho(i,j+1,k)-blk.rho(i,j-1,k))*ih2;
                const double drz = (blk.rho(i,j,k+1)-blk.rho(i,j,k-1))*ih2;
                return static_cast<float>(std::sqrt(drx*drx+dry*dry+drz*drz));
            }
            default: return 0.f;
        }
    };

    // Sort leaves coarse→fine so fine blocks overwrite coarse in shared voxels.
    auto leaves = tree.leaf_indices();
    std::sort(leaves.begin(), leaves.end(), [&](int a, int b) {
        return tree.nodes[a].level < tree.nodes[b].level;
    });

    for (int li : leaves) {
        const BlockNode& node = tree.nodes[li];
        const CellBlock& blk  = *node.block;
        const double h = blk.h;

        for (int kk = 0; kk < NB; ++kk)
        for (int jj = 0; jj < NB; ++jj)
        for (int ii = 0; ii < NB; ++ii) {
            const double cx = node.ox + (ii + 0.5) * h;
            const double cy = node.oy + (jj + 0.5) * h;
            const double cz = node.oz + (kk + 0.5) * h;

            // Voxel range covered by this cell (cell spans [cx−h/2, cx+h/2])
            const int vi0 = static_cast<int>((cx - 0.5*h) / L * N);
            const int vi1 = static_cast<int>((cx + 0.5*h) / L * N);
            const int vj0 = static_cast<int>((cy - 0.5*h) / L * N);
            const int vj1 = static_cast<int>((cy + 0.5*h) / L * N);
            const int vk0 = static_cast<int>((cz - 0.5*h) / L * N);
            const int vk1 = static_cast<int>((cz + 0.5*h) / L * N);

            const float val = cell_val(blk, NG+ii, NG+jj, NG+kk);

            const int vi_lo = std::max(0, vi0);
            const int vi_hi = std::min(N-1, vi1);
            const int vj_lo = std::max(0, vj0);
            const int vj_hi = std::min(N-1, vj1);
            const int vk_lo = std::max(0, vk0);
            const int vk_hi = std::min(N-1, vk1);

            for (int vk = vk_lo; vk <= vk_hi; ++vk)
            for (int vj = vj_lo; vj <= vj_hi; ++vj)
            for (int vi = vi_lo; vi <= vi_hi; ++vi) {
                fb.data[static_cast<size_t>(vk)*N*N + vj*N + vi] = val;
                g_vmin = std::min(g_vmin, val);
                g_vmax = std::max(g_vmax, val);
            }
        }
    }

    if (g_vmin >= g_vmax) { g_vmin = 0.f; g_vmax = 1.f; }
    fb.g_vmin = g_vmin;
    fb.g_vmax = g_vmax;
}

// Serialise FrameBuffer3D → length-prefixed binary blob.
// Implementation delegated to frame_packer.cpp (R9-E1).
void LiveStreamer::serialize_volume(const FrameBuffer3D& fb,
                                    std::vector<uint8_t>& out)
{
    ::serialize_volume(fb, out);
}

// =============================================================================
// LiveStreamer — construction / destruction
// =============================================================================

LiveStreamer::LiveStreamer(const StreamConfig& cfg)
    : cfg_(cfg), t_start_(std::chrono::steady_clock::now()) {
    accept_thread_   = std::thread(&LiveStreamer::run_accept,   this);
    stream_thread_   = std::thread(&LiveStreamer::run_stream,   this);
    stream3d_thread_ = std::thread(&LiveStreamer::run_stream3d, this);
}

LiveStreamer::~LiveStreamer() {
    {
        std::lock_guard<std::mutex> lk(swap_mtx_);
        shutdown_ = true;
    }
    swap_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lk(swap3d_mtx_);
        // shutdown_ already set — no separate flag needed
    }
    swap3d_cv_.notify_all();

    if (accept_thread_.joinable())   accept_thread_.join();
    if (stream_thread_.joinable())   stream_thread_.join();
    if (stream3d_thread_.joinable()) stream3d_thread_.join();

    int fd = stream_fd_.exchange(-1);
    if (fd >= 0) ::close(fd);
    int vfd = vol_stream_fd_.exchange(-1);
    if (vfd >= 0) ::close(vfd);
}

// =============================================================================
// LiveStreamer::snapshot — called by solver after apply_flux_correction()
// =============================================================================

void LiveStreamer::snapshot(const BlockTree& tree, int step, double t) {
    StreamConfig cfg_snap;
    {
        std::lock_guard<std::mutex> lk(cfg_mtx_);
        cfg_snap = cfg_;
    }
    if (cfg_snap.stride > 1 && (step % cfg_snap.stride) != 0) return;

    // 2-D slice
    build_frame(tree, step, t, back_);
    {
        std::lock_guard<std::mutex> lk(swap_mtx_);
        std::swap(back_, front_);
        front_fresh_ = true;
    }
    swap_cv_.notify_one();

    // 3-D volume (P6.6) — only when a volume client is connected
    if (vol_stream_fd_.load(std::memory_order_acquire) >= 0) {
        build_volume(tree, step, t, back3d_);
        {
            std::lock_guard<std::mutex> lk(swap3d_mtx_);
            std::swap(back3d_, front3d_);
            front3d_fresh_ = true;
        }
        swap3d_cv_.notify_one();
    }
}

// =============================================================================
// LiveStreamer::build_frame — extract 2-D slice into FrameBuffer
// =============================================================================

void LiveStreamer::build_frame(const BlockTree& tree, int step, double t,
                               FrameBuffer& fb)
{
    StreamConfig cfg_snap;
    {
        std::lock_guard<std::mutex> lk(cfg_mtx_);
        cfg_snap = cfg_;
    }

    const uint8_t   axis   = cfg_snap.axis;
    const StreamVar svar   = cfg_snap.var;
    const double    L      = tree.domain_L();
    const double    z_phys = cfg_snap.pos * L;   // physical position along slice axis

    fb.step     = step;
    fb.sim_time = t;
    fb.axis     = axis;
    fb.var_id   = static_cast<uint8_t>(svar);
    fb.domain_L = static_cast<float>(L);
    fb.descs.clear();
    fb.data.clear();
    fb.probe.clear();

    float g_vmin = std::numeric_limits<float>::max();
    float g_vmax = std::numeric_limits<float>::lowest();

    // Helper: velocity component at (i,j,k) — comp 0=u,1=v,2=w.
    auto vel_at = [](const CellBlock& blk, int comp, int i, int j, int k) -> double {
        switch (comp) {
            case 0: return blk.rhou(i,j,k) / blk.rho(i,j,k);
            case 1: return blk.rhov(i,j,k) / blk.rho(i,j,k);
            default: return blk.rhow(i,j,k) / blk.rho(i,j,k);
        }
    };

    // Lambda: extract scalar value at interior cell (i,j,k).
    // Ghost cells (index NG-1, NG+NB) are valid for gradient stencils.
    auto cell_val = [&](const CellBlock& blk, int i, int j, int k) -> float {
        switch (svar) {
            case StreamVar::RHO:   return static_cast<float>(blk.rho (i,j,k));
            case StreamVar::RHOU:  return static_cast<float>(blk.rhou(i,j,k));
            case StreamVar::RHOV:  return static_cast<float>(blk.rhov(i,j,k));
            case StreamVar::RHOW:  return static_cast<float>(blk.rhow(i,j,k));
            case StreamVar::ETOT:  return static_cast<float>(blk.E   (i,j,k));
            default: break;
        }
        Prim q = blk.prim(i, j, k);
        switch (svar) {
            case StreamVar::PRESS: return static_cast<float>(q.p);
            case StreamVar::TEMP:  return static_cast<float>(q.T);
            case StreamVar::UMAG:  return static_cast<float>(
                                       std::sqrt(q.u*q.u + q.v*q.v + q.w*q.w));
            case StreamVar::MACH: {
                const double c = std::sqrt(GAMMA * q.p / q.rho);
                return static_cast<float>(std::sqrt(q.u*q.u+q.v*q.v+q.w*q.w) / c);
            }
            case StreamVar::VORT: {
                // ω = ∇×u, |ω|² = ωx²+ωy²+ωz²; central differences, h = blk.h
                const double ih2 = 1.0 / (2.0 * blk.h);
                const double wx = (vel_at(blk,2,i,j+1,k)-vel_at(blk,2,i,j-1,k))*ih2
                                - (vel_at(blk,1,i,j,k+1)-vel_at(blk,1,i,j,k-1))*ih2;
                const double wy = (vel_at(blk,0,i,j,k+1)-vel_at(blk,0,i,j,k-1))*ih2
                                - (vel_at(blk,2,i+1,j,k)-vel_at(blk,2,i-1,j,k))*ih2;
                const double wz = (vel_at(blk,1,i+1,j,k)-vel_at(blk,1,i-1,j,k))*ih2
                                - (vel_at(blk,0,i,j+1,k)-vel_at(blk,0,i,j-1,k))*ih2;
                return static_cast<float>(std::sqrt(wx*wx+wy*wy+wz*wz));
            }
            case StreamVar::QCRIT: {
                // Q = -½ Σ_{c,d} A_cd * A_dc,  A_cd = ∂u_c/∂x_d
                const double ih2 = 1.0 / (2.0 * blk.h);
                // A[row][col]: row=velocity comp (0=u,1=v,2=w), col=spatial axis (0=x,1=y,2=z)
                const double A[3][3] = {
                    {(vel_at(blk,0,i+1,j,k)-vel_at(blk,0,i-1,j,k))*ih2,
                     (vel_at(blk,0,i,j+1,k)-vel_at(blk,0,i,j-1,k))*ih2,
                     (vel_at(blk,0,i,j,k+1)-vel_at(blk,0,i,j,k-1))*ih2},
                    {(vel_at(blk,1,i+1,j,k)-vel_at(blk,1,i-1,j,k))*ih2,
                     (vel_at(blk,1,i,j+1,k)-vel_at(blk,1,i,j-1,k))*ih2,
                     (vel_at(blk,1,i,j,k+1)-vel_at(blk,1,i,j,k-1))*ih2},
                    {(vel_at(blk,2,i+1,j,k)-vel_at(blk,2,i-1,j,k))*ih2,
                     (vel_at(blk,2,i,j+1,k)-vel_at(blk,2,i,j-1,k))*ih2,
                     (vel_at(blk,2,i,j,k+1)-vel_at(blk,2,i,j,k-1))*ih2}
                };
                double Q = 0.0;
                for (int c = 0; c < 3; ++c)
                    for (int d = 0; d < 3; ++d)
                        Q -= 0.5 * A[c][d] * A[d][c];
                return static_cast<float>(Q);
            }
            case StreamVar::SCHLIEREN: {
                const double ih2 = 1.0 / (2.0 * blk.h);
                const double drx = (blk.rho(i+1,j,k)-blk.rho(i-1,j,k))*ih2;
                const double dry = (blk.rho(i,j+1,k)-blk.rho(i,j-1,k))*ih2;
                const double drz = (blk.rho(i,j,k+1)-blk.rho(i,j,k-1))*ih2;
                return static_cast<float>(std::sqrt(drx*drx+dry*dry+drz*drz));
            }
            default: return 0.f;
        }
    };

    for (int li : tree.leaf_indices()) {
        const BlockNode& node = tree.nodes[li];
        const CellBlock& blk  = *node.block;
        const double     h    = blk.h;

        // Find the axis-aligned block range and check intersection
        double lo, hi;
        if      (axis == 0) { lo = node.ox; hi = node.ox + NB * h; }
        else if (axis == 1) { lo = node.oy; hi = node.oy + NB * h; }
        else                { lo = node.oz; hi = node.oz + NB * h; }
        if (z_phys < lo || z_phys >= hi) continue;

        // Slice index along the axis (clamped to interior)
        int s = NG + static_cast<int>((z_phys - lo) / h);
        s = std::clamp(s, NG, NG + NB - 1);

        // 2-D projected origin (the two axes not sliced through)
        BlockDesc2D desc{};
        if      (axis == 0) { desc.ox2d = static_cast<float>(node.oy);
                              desc.oy2d = static_cast<float>(node.oz); }
        else if (axis == 1) { desc.ox2d = static_cast<float>(node.ox);
                              desc.oy2d = static_cast<float>(node.oz); }
        else                { desc.ox2d = static_cast<float>(node.ox);
                              desc.oy2d = static_cast<float>(node.oy); }
        desc.h     = static_cast<float>(h);
        desc.level = static_cast<uint8_t>(node.level);
        fb.descs.push_back(desc);

        // Extract NB × NB values (row = second free axis, col = first free axis)
        for (int b = 0; b < NB; ++b)
        for (int a = 0; a < NB; ++a) {
            const int ia = NG + a, ib = NG + b;
            int ci, cj, ck;
            if      (axis == 0) { ci = s;  cj = ia; ck = ib; }
            else if (axis == 1) { ci = ia; cj = s;  ck = ib; }
            else                { ci = ia; cj = ib; ck = s;  }
            float v = cell_val(blk, ci, cj, ck);
            fb.data.push_back(v);
            g_vmin = std::min(g_vmin, v);
            g_vmax = std::max(g_vmax, v);
            // P12.2: all 8 vars for probe queries
            Prim q = blk.prim(ci, cj, ck);
            fb.probe.push_back({
                static_cast<float>(blk.rho (ci,cj,ck)),
                static_cast<float>(q.p),
                static_cast<float>(q.T),
                static_cast<float>(std::sqrt(q.u*q.u+q.v*q.v+q.w*q.w)),
                static_cast<float>(blk.rhou(ci,cj,ck)),
                static_cast<float>(blk.rhov(ci,cj,ck)),
                static_cast<float>(blk.rhow(ci,cj,ck)),
                static_cast<float>(blk.E   (ci,cj,ck))
            });
        }
    }

    // Guard against empty slice (no blocks intersect)
    if (fb.descs.empty()) { g_vmin = 0.f; g_vmax = 1.f; }
    if (g_vmin == g_vmax)   g_vmax = g_vmin + 1.f;

    fb.g_vmin   = g_vmin;
    fb.g_vmax   = g_vmax;
    fb.n_blocks = static_cast<uint8_t>(std::min((int)fb.descs.size(), 255));
}

// =============================================================================
// LiveStreamer::serialize_frame — pack FrameBuffer into a length-prefixed blob
// Implementation delegated to frame_packer.cpp (R9-E1).
// =============================================================================

void LiveStreamer::serialize_frame(const FrameBuffer& fb, std::vector<uint8_t>& out) {
    ::serialize_frame(fb, out);
}

// =============================================================================
// LiveStreamer::run_stream — stream thread
// =============================================================================

void LiveStreamer::run_stream() {
    std::vector<uint8_t> frame_bytes;

    while (true) {
        // Wait for next frame
        {
            std::unique_lock<std::mutex> lk(swap_mtx_);
            swap_cv_.wait(lk, [&]{ return front_fresh_ || shutdown_; });
            if (shutdown_) break;
            std::swap(work_, front_);
            front_fresh_ = false;
        }

        serialize_frame(work_, frame_bytes);

        int fd = stream_fd_.load(std::memory_order_acquire);
        if (fd < 0) continue;

        // Send as one HTTP chunk: <hex-size>\r\n<data>\r\n
        char hdr[24];
        int  hlen = std::snprintf(hdr, sizeof(hdr), "%zx\r\n", frame_bytes.size());

        if (!http_safe_send(fd, hdr, static_cast<size_t>(hlen)) ||
            !http_safe_send(fd, frame_bytes.data(), frame_bytes.size()) ||
            !http_safe_send(fd, "\r\n", 2))
        {
            // Client disconnected — clear socket; accept thread will set a new one
            int expected = fd;
            stream_fd_.compare_exchange_strong(expected, -1,
                                               std::memory_order_acq_rel);
            ::close(fd);
        }
    }
}

// =============================================================================
// LiveStreamer::run_stream3d — 3-D volume stream thread (P6.6)
// =============================================================================

void LiveStreamer::run_stream3d() {
    std::vector<uint8_t> frame_bytes;

    while (true) {
        {
            std::unique_lock<std::mutex> lk(swap3d_mtx_);
            swap3d_cv_.wait(lk, [&]{ return front3d_fresh_ || shutdown_; });
            if (shutdown_) break;
            std::swap(work3d_, front3d_);
            front3d_fresh_ = false;
        }

        serialize_volume(work3d_, frame_bytes);

        int fd = vol_stream_fd_.load(std::memory_order_acquire);
        if (fd < 0) continue;

        char hdr[24];
        int  hlen = std::snprintf(hdr, sizeof(hdr), "%zx\r\n", frame_bytes.size());
        if (!http_safe_send(fd, hdr, static_cast<size_t>(hlen)) ||
            !http_safe_send(fd, frame_bytes.data(), frame_bytes.size()) ||
            !http_safe_send(fd, "\r\n", 2))
        {
            int expected = fd;
            vol_stream_fd_.compare_exchange_strong(expected, -1,
                                                   std::memory_order_acq_rel);
            ::close(fd);
        }
    }
}

// =============================================================================
// LiveStreamer::run_accept — accept thread
// =============================================================================

void LiveStreamer::run_accept() {
    int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        std::perror("LiveStreamer: socket");
        return;
    }
    int one = 1;
    ::setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(cfg_.port));

    if (::bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("LiveStreamer: bind");
        ::close(sfd);
        return;
    }
    ::listen(sfd, 8);

    std::fprintf(stderr, "[LiveStreamer] listening on http://localhost:%d\n", cfg_.port);
    std::fflush(stderr);

    while (!shutdown_) {
        // Non-blocking accept with 100 ms timeout via select
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sfd, &rfds);
        struct timeval tv{0, 100000};
        if (::select(sfd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;

        int cfd = ::accept(sfd, nullptr, nullptr);
        if (cfd < 0) continue;

        // Dispatch in a detached thread (short-lived except for /stream)
        std::thread([this, cfd]() { this->handle_connection(cfd); }).detach();
    }
    ::close(sfd);
}

// =============================================================================
// LiveStreamer::handle_connection — per-connection dispatcher
// =============================================================================

void LiveStreamer::handle_connection(int cfd) {
    std::string req = http_read_request(cfd);
    if (req.empty()) { ::close(cfd); return; }

    const bool is_get_root      = (req.rfind("GET / ",    0) == 0) ||
                                  (req.rfind("GET /\r",   0) == 0);
    const bool is_get_stream    = (req.rfind("GET /stream",        0) == 0);
    const bool is_get_volume    = (req.rfind("GET /volume ",       0) == 0) ||
                                  (req.rfind("GET /volume\r",      0) == 0);
    const bool is_get_vol_strm  = (req.rfind("GET /volume-stream", 0) == 0);
    const bool is_get_metrics   = (req.rfind("GET /metrics",        0) == 0);
    const bool is_post_probe    = (req.rfind("POST /probe",          0) == 0);
    const bool is_post_cfg      = (req.rfind("POST /config", 0) == 0);

    if (is_get_root) {
        handle_get_root(cfd);
        ::close(cfd);
    } else if (is_get_stream) {
        handle_get_stream(cfd);
    } else if (is_get_volume) {
        handle_get_volume(cfd);
        ::close(cfd);
    } else if (is_get_vol_strm) {
        handle_get_vol_stream(cfd);
    } else if (is_get_metrics) {
        handle_get_metrics(cfd);   // closes cfd inside
    } else if (is_post_probe) {
        handle_post_probe(cfd, req);  // closes cfd inside
    } else if (is_post_cfg) {
        handle_post_config(cfd, req);
        ::close(cfd);
    } else {
        const char* r404 =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Length: 0\r\n\r\n";
        http_safe_send(cfd, r404, std::strlen(r404));
        ::close(cfd);
    }
}

// =============================================================================
// HTTP response helpers
// =============================================================================

void LiveStreamer::handle_get_root(int cfd) {
    const char* html = viewer_html();
    const size_t len = std::strlen(html);

    char hdr[256];
    int hlen = std::snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n", len);

    http_safe_send(cfd, hdr, static_cast<size_t>(hlen));
    http_safe_send(cfd, html, len);
}

void LiveStreamer::handle_get_stream(int cfd) {
    const char* hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n\r\n";

    if (!http_safe_send(cfd, hdr, std::strlen(hdr))) {
        ::close(cfd);
        return;
    }

    // Replace any previous stream socket
    int old = stream_fd_.exchange(cfd, std::memory_order_acq_rel);
    if (old >= 0 && old != cfd) ::close(old);
}

void LiveStreamer::handle_get_volume(int cfd) {
    std::string html = viewer_html_3d();
    char hdr[256];
    int hlen = std::snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n", html.size());
    http_safe_send(cfd, hdr, static_cast<size_t>(hlen));
    http_safe_send(cfd, html.c_str(), html.size());
}

void LiveStreamer::handle_get_vol_stream(int cfd) {
    const char* hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n\r\n";
    if (!http_safe_send(cfd, hdr, std::strlen(hdr))) { ::close(cfd); return; }

    int old = vol_stream_fd_.exchange(cfd, std::memory_order_acq_rel);
    if (old >= 0 && old != cfd) ::close(old);
    // Ownership transferred to stream3d_thread_; do not close here.
}

// P12.1 — push latest metrics from the solver thread (non-blocking).
void LiveStreamer::push_metrics(const MetricsSnapshot& m) {
    std::lock_guard<std::mutex> lk(metrics_mtx_);
    metrics_ = m;
}

// P12.1 — GET /metrics → JSON object.
void LiveStreamer::handle_get_metrics(int cfd) {
    MetricsSnapshot m;
    {
        std::lock_guard<std::mutex> lk(metrics_mtx_);
        m = metrics_;
    }
    m.wall_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start_).count();

    char buf[1024];
    int hlen = std::snprintf(buf, sizeof(buf),
        "{\"step\":%d,\"t\":%.6e,\"dt\":%.6e,\"cfl\":%.6f,"
        "\"ke\":%.6e,\"mass\":%.6e,\"n_leaves\":%d,"
        "\"rho_min\":%.6e,\"rho_max\":%.6e,"
        "\"wall_time_ms\":%.3f,\"gpu_active\":%s,"
        "\"mass_error\":%.3e,\"momentum_error\":%.3e,\"energy_error\":%.3e,"
        "\"leaves_per_level\":[%d,%d,%d,%d,%d,%d,%d,%d]}",
        m.step, m.t, m.dt, m.cfl, m.ke, m.mass, m.n_leaves,
        m.rho_min, m.rho_max, m.wall_time_ms,
        m.gpu_active ? "true" : "false",
        m.mass_error, m.momentum_error, m.energy_error,
        m.leaves_per_level[0], m.leaves_per_level[1],
        m.leaves_per_level[2], m.leaves_per_level[3],
        m.leaves_per_level[4], m.leaves_per_level[5],
        m.leaves_per_level[6], m.leaves_per_level[7]);

    char hdr[256];
    int hlen2 = std::snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n", hlen);
    ::send(cfd, hdr, static_cast<size_t>(hlen2), MSG_NOSIGNAL);
    ::send(cfd, buf, static_cast<size_t>(hlen),  MSG_NOSIGNAL);
    ::close(cfd);
}

// P12.2 — POST /probe {"x":norm_x,"y":norm_y} → JSON with all 8 vars at nearest cell.
// Coordinates are normalised canvas coords (0-1): x left→right, y top→bottom.
void LiveStreamer::handle_post_probe(int cfd, const std::string& req_with_body) {
    int cl = http_content_length(req_with_body);
    std::string body;
    auto pos0 = req_with_body.find("\r\n\r\n");
    if (pos0 != std::string::npos) body = req_with_body.substr(pos0 + 4);
    while ((int)body.size() < cl) {
        char tmp[256]; int want = std::min(cl-(int)body.size(),(int)sizeof(tmp));
        ssize_t n = ::recv(cfd, tmp, static_cast<size_t>(want), 0);
        if (n <= 0) break;
        body.append(tmp, static_cast<size_t>(n));
    }

    double norm_x = 0.5, norm_y = 0.5;
    json_double(body, "x", norm_x);
    json_double(body, "y", norm_y);

    char resp[512];
    int rlen;
    {
        std::lock_guard<std::mutex> lk(swap_mtx_);
        const float L = front_.domain_L;
        // y-axis flipped: physical oy2d increases upward, canvas y increases downward
        const float pa = static_cast<float>(norm_x) * L;
        const float pb = static_cast<float>(1.0 - norm_y) * L;
        int block_idx = -1;
        int cell_flat = -1;
        for (int b = 0; b < (int)front_.descs.size(); ++b) {
            const BlockDesc2D& d = front_.descs[b];
            const float bsize = d.h * NB;
            if (pa < d.ox2d || pa >= d.ox2d + bsize) continue;
            if (pb < d.oy2d || pb >= d.oy2d + bsize) continue;
            const int ia = static_cast<int>((pa - d.ox2d) / d.h);
            const int ib = static_cast<int>((pb - d.oy2d) / d.h);
            block_idx = b;
            cell_flat = b * NB * NB + ib * NB + ia;
            break;
        }
        if (block_idx < 0 || cell_flat >= (int)front_.probe.size()) {
            rlen = std::snprintf(resp, sizeof(resp),
                "{\"ok\":false,\"msg\":\"no block at queried position\"}");
        } else {
            const auto& v = front_.probe[cell_flat];
            rlen = std::snprintf(resp, sizeof(resp),
                "{\"ok\":true,\"block\":%d,\"level\":%d,"
                "\"rho\":%.5g,\"press\":%.5g,\"temp\":%.5g,\"umag\":%.5g,"
                "\"rhou\":%.5g,\"rhov\":%.5g,\"rhow\":%.5g,\"E\":%.5g}",
                block_idx, (int)front_.descs[block_idx].level,
                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        }
    }
    char hdr[256];
    int hlen = std::snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        "Content-Length: %d\r\nAccess-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n", rlen);
    ::send(cfd, hdr,  static_cast<size_t>(hlen), MSG_NOSIGNAL);
    ::send(cfd, resp, static_cast<size_t>(rlen),  MSG_NOSIGNAL);
    ::close(cfd);
}

void LiveStreamer::handle_post_config(int cfd, const std::string& req_with_body) {
    // Read remaining body bytes if Content-Length > what we already buffered
    int cl = http_content_length(req_with_body);
    std::string body;
    auto body_start = req_with_body.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        body = req_with_body.substr(body_start + 4);
    }
    while ((int)body.size() < cl) {
        char tmp[256];
        int want = std::min(cl - (int)body.size(), (int)sizeof(tmp));
        ssize_t n = ::recv(cfd, tmp, static_cast<size_t>(want), 0);
        if (n <= 0) break;
        body.append(tmp, static_cast<size_t>(n));
    }

    // Parse and apply config
    {
        std::lock_guard<std::mutex> lk(cfg_mtx_);
        int ival; double dval;
        if (json_int(body, "var", ival) && ival >= 0 && ival <= 11)
            cfg_.var = static_cast<StreamVar>(ival);
        if (json_int(body, "axis", ival) && ival >= 0 && ival <= 2)
            cfg_.axis = static_cast<uint8_t>(ival);
        if (json_double(body, "pos", dval) && dval >= 0.0 && dval <= 1.0)
            cfg_.pos = dval;
    }

    const char* r200 =
        "HTTP/1.1 200 OK\r\n"
        "Content-Length: 0\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n";
    http_safe_send(cfd, r200, std::strlen(r200));
}

// =============================================================================
// P6.6 — WebGPU 3-D volume viewer HTML (embedded raw string)
// =============================================================================

std::string LiveStreamer::viewer_html_3d() {
    // Port is needed inside the JS, so we format it in.
    // Return value is the complete single-page HTML application.
    int port = cfg_.port;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%d", port);
    std::string html = std::string(R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CFD Volume Viewer (WebGPU)</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#bbb;font-family:monospace;font-size:12px;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
#bar{padding:5px 10px;background:#181818;border-bottom:1px solid #2a2a2a;
     display:flex;gap:12px;align-items:center;flex-shrink:0;flex-wrap:wrap}
#cw{flex:1;position:relative;overflow:hidden}
canvas{display:block;width:100%;height:100%}
label{display:flex;align-items:center;gap:5px}
select,input[type=range]{background:#222;color:#ccc;border:1px solid #3a3a3a;
     padding:2px 5px;cursor:pointer}
#tf-canvas{border:1px solid #3a3a3a;cursor:crosshair;flex-shrink:0}
#info{margin-left:auto;color:#555;font-size:11px}
#err{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
     color:#f66;font-size:16px;text-align:center;pointer-events:none}
#spkw{height:140px;flex-shrink:0;background:#111;border-top:1px solid #222}
#spk{display:block;width:100%;height:100%}
</style>
</head>
<body>
<div id="bar">
  <label>var
    <select id="sv">
      <option value="0">&#961; density</option>
      <option value="1">p pressure</option>
      <option value="2">T temperature</option>
      <option value="3">|u| speed</option>
      <option value="4">&#961;u</option><option value="5">&#961;v</option>
      <option value="6">&#961;w</option><option value="7">E</option>
      <option value="8">Mach</option>
      <option value="9">|&#969;| vorticity</option>
      <option value="10">Q-criterion</option>
      <option value="11">schlieren |&#8711;&#961;|</option>
    </select>
  </label>
  <label>steps<input type="range" id="nsteps" min="32" max="256" step="16" value="96">
    <span id="lns">96</span></label>
  <label>opacity<input type="range" id="opac" min="1" max="40" step="1" value="12">
    <span id="lop">12</span></label>
  <label>colormap
    <select id="cmap">
      <option value="0">Viridis</option>
      <option value="1">Hot</option>
      <option value="2">Cool</option>
      <option value="3">Gray</option>
    </select>
  </label>
  <canvas id="tf-canvas" width="128" height="32" title="Drag to edit opacity"></canvas>
  <span id="info">connecting&hellip;</span>
</div>
<div id="cw">
  <canvas id="c"></canvas>
  <div id="err" style="display:none"></div>
</div>
<div id="spkw"><canvas id="spk"></canvas></div>

<script>
// ── Constants ──────────────────────────────────────────────────────────────
const PORT = )HTML") + port_str + R"HTML(;

// ── State ──────────────────────────────────────────────────────────────────
let device, pipeline, uniformBuf, tfBuf, volTex, volSampler, bindGroup;
let canvasCtx;    // WebGPU canvas context
let N = 32, vmin = 0, vmax = 1, domainL = 1;
let nsteps = 96, opacScale = 12, cmapId = 0;

// Arcball camera (spherical coords, volume centre = 0.5³)
let theta = 0.6, phi = 0.8, radius = 2.2;
let dragStart = null;

// ── Sparklines ─────────────────────────────────────────────────────────────
const spkCanvas = document.getElementById('spk');
const sctx = spkCanvas.getContext('2d');
const SPK_MAX = 2000;
const spkHist = {cfl:[],ke:[],mass:[],leaves:[],mass_err:[],mom_err:[],energy_err:[]};
let spkFetching = false;

function resizeSpk() {
  const el = document.getElementById('spkw');
  spkCanvas.width  = el.clientWidth;
  spkCanvas.height = el.clientHeight;
}
window.addEventListener('resize', resizeSpk);

function drawSparkRow(series, x0row, y0row, rowH, W, logScale) {
  const nS = series.length, sw = W / nS;
  series.forEach((s, idx) => {
    const x0 = x0row + idx * sw;
    if (idx > 0) {
      sctx.strokeStyle = '#2a2a2a'; sctx.lineWidth = 1;
      sctx.beginPath(); sctx.moveTo(x0, y0row); sctx.lineTo(x0, y0row + rowH); sctx.stroke();
    }
    if (s.data.length < 2) return;
    const vals = logScale
      ? s.data.map(v => Math.log10(Math.max(v, 1e-20)))
      : s.data;
    let mn = Infinity, mx = -Infinity;
    for (const v of vals) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const rng = mx > mn ? mx - mn : 1;
    const n = vals.length;
    sctx.strokeStyle = s.color; sctx.lineWidth = 1;
    sctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = x0 + 1 + (i / (n - 1)) * (sw - 2);
      const py = y0row + rowH - 14 - ((vals[i] - mn) / rng) * (rowH - 18);
      i === 0 ? sctx.moveTo(px, py) : sctx.lineTo(px, py);
    }
    sctx.stroke();
    sctx.fillStyle = s.color; sctx.font = '10px monospace';
    const cur = s.data[s.data.length - 1];
    const label = logScale ? `${s.label}: ${cur.toExponential(1)}` : `${s.label}: ${cur.toPrecision(3)}`;
    sctx.fillText(label, x0 + 3, y0row + rowH - 3);
  });
}

function drawSparklines() {
  const W = spkCanvas.width, H = spkCanvas.height;
  sctx.fillStyle = '#111'; sctx.fillRect(0, 0, W, H);
  const rowH = Math.floor(H / 2);

  drawSparkRow([
    {data: spkHist.cfl,    label: 'CFL',    color: '#fa0'},
    {data: spkHist.ke,     label: 'KE',     color: '#4af'},
    {data: spkHist.mass,   label: 'mass',   color: '#4fa'},
    {data: spkHist.leaves, label: 'leaves', color: '#f4a'}
  ], 0, 0, rowH, W, false);

  sctx.strokeStyle = '#333'; sctx.lineWidth = 1;
  sctx.beginPath(); sctx.moveTo(0, rowH); sctx.lineTo(W, rowH); sctx.stroke();

  drawSparkRow([
    {data: spkHist.mass_err,   label: 'Δm/m₀',   color: '#f77'},
    {data: spkHist.mom_err,    label: 'Δp/p₀',   color: '#fa7'},
    {data: spkHist.energy_err, label: 'ΔE/E₀',   color: '#ff7'}
  ], 0, rowH, H - rowH, W, true);
}

function fetchMetrics() {
  if (spkFetching) return; spkFetching = true;
  fetch('/metrics').then(r => r.json()).then(m => {
    const trim = arr => { if (arr.length >= SPK_MAX) arr.shift(); };
    trim(spkHist.cfl);        spkHist.cfl.push(m.cfl);
    trim(spkHist.ke);         spkHist.ke.push(m.ke);
    trim(spkHist.mass);       spkHist.mass.push(m.mass);
    trim(spkHist.leaves);     spkHist.leaves.push(m.n_leaves);
    trim(spkHist.mass_err);   spkHist.mass_err.push(m.mass_error);
    trim(spkHist.mom_err);    spkHist.mom_err.push(m.momentum_error);
    trim(spkHist.energy_err); spkHist.energy_err.push(m.energy_error);
    drawSparklines();
    spkFetching = false;
  }).catch(() => { spkFetching = false; });
}

// Transfer function: 256 control points, each [r,g,b,a] in [0,1]
const TF_SIZE = 256;
let tfData = new Float32Array(TF_SIZE * 4);  // rgba

// ── TF presets ─────────────────────────────────────────────────────────────
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const f=(c0,c1,c2,c3,c4,c5,c6)=>c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
  return [
    f(0.2777273,0.1050930,-0.3308618,-4.6342305, 6.2282699, 4.7763850,-5.4354559),
    f(0.0054073,1.4046135, 0.2148476,-5.7991010,14.1799334,-13.7451454, 4.6458526),
    f(0.3340998,1.3845902, 0.0950952,-19.332441,56.6905526,-65.3530342,26.3124352)
  ].map(v=>Math.max(0,Math.min(1,v)));
}
function hot(t){ return [Math.min(1,t*3), Math.min(1,Math.max(0,t*3-1)), Math.min(1,Math.max(0,t*3-2))]; }
function cool(t){ return [t, 1-t, 1]; }
function gray(t){ return [t, t, t]; }
const CMAPS = [viridis, hot, cool, gray];

function rebuildTF() {
  const cmap = CMAPS[cmapId];
  // Read control-point canvas for opacity modulation
  const tfCtx = document.getElementById('tf-canvas').getContext('2d');
  const imgd  = tfCtx.getImageData(0, 0, 128, 32);
  for (let i = 0; i < TF_SIZE; ++i) {
    const [r, g, b] = cmap(i / (TF_SIZE - 1));
    // Opacity: canvas row 0=max, row 31=zero; read the brightest pixel in column i/2
    const cx = Math.min(127, Math.round(i / TF_SIZE * 128));
    // User-drawn opacity: read 'red' channel as height indicator
    let maxA = 0;
    for (let row = 0; row < 32; row++) {
      const px = (row * 128 + cx) * 4;
      if (imgd.data[px] > maxA) maxA = imgd.data[px];
    }
    const base_a = 4.0 * (i / TF_SIZE) * (1.0 - i / TF_SIZE); // bell
    const user_a = maxA > 0 ? maxA / 255 : base_a;
    const a = user_a * opacScale / 12.0;
    tfData[i*4]   = r; tfData[i*4+1] = g; tfData[i*4+2] = b; tfData[i*4+3] = a;
  }
  if (device && tfBuf) {
    device.queue.writeBuffer(tfBuf, 0, tfData);
  }
}

// ── TF canvas editor (P6.7) ────────────────────────────────────────────────
const tfCv  = document.getElementById('tf-canvas');
const tfCtx = tfCv.getContext('2d');
// Default: paint a viridis gradient
(function initTF() {
  for (let i = 0; i < 128; i++) {
    const [r, g, b] = viridis(i / 127);
    tfCtx.fillStyle = `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
    tfCtx.fillRect(i, 0, 1, 32);
  }
})();

let tfDrag = false;
tfCv.addEventListener('mousedown', e=>{ tfDrag=true; drawTFStroke(e); });
tfCv.addEventListener('mousemove', e=>{ if(tfDrag) drawTFStroke(e); });
tfCv.addEventListener('mouseup',   ()=>{ tfDrag=false; rebuildTF(); });
function drawTFStroke(e) {
  const r = tfCv.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) / r.width  * 128);
  const y = Math.round((e.clientY - r.top)  / r.height * 32);
  tfCtx.fillStyle = '#fff';
  tfCtx.fillRect(x-1, 0, 3, y);
  tfCtx.fillStyle = '#000';
  tfCtx.fillRect(x-1, y, 3, 32-y);
}

// ── LZ4 decompressor (same as 2-D viewer) ──────────────────────────────────
function lz4_decomp(src, src_off, src_len, dst_size) {
  const dst=new Uint8Array(dst_size);
  let si=src_off,se=src_off+src_len,di=0;
  while(si<se){
    const tok=src[si++]; let ll=tok>>4;
    if(ll===15){let x;do{x=src[si++];ll+=x;}while(x===255);}
    for(let i=0;i<ll;i++) dst[di++]=src[si++];
    if(si>=se) break;
    const off=src[si++]|(src[si++]<<8);
    let ml=(tok&0xf)+4;
    if((tok&0xf)===15){let x;do{x=src[si++];ml+=x;}while(x===255);}
    const ms=di-off;
    for(let i=0;i<ml;i++) dst[di++]=dst[ms+i];
  }
  return dst;
}

// ── WebGPU setup ───────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const errEl  = document.getElementById('err');

async function initWebGPU() {
  if (!navigator.gpu) {
    errEl.style.display = 'block';
    errEl.textContent = 'WebGPU not available.\nUse Chrome 113+ or Edge 113+.';
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { errEl.style.display='block'; errEl.textContent='No WebGPU adapter.'; return false; }
  device = await adapter.requestDevice();

  canvasCtx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  canvasCtx.configure({ device, format: fmt, alphaMode: 'opaque' });

  // Uniform buffer: 96 bytes
  //   [0..63]  mat4×4 inv_view_proj  (16 × f32)
  //   [64..75] eye xyz               (3 × f32)
  //   [76]     n_steps (u32)
  //   [80]     vmin (f32)
  //   [84]     vmax (f32)
  //   [88]     canvas_w (u32)
  //   [92]     canvas_h (u32)
  uniformBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // TF buffer: 256 × vec4f = 4096 bytes
  tfBuf = device.createBuffer({ size: TF_SIZE * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  rebuildTF();

  volSampler = device.createSampler({ magFilter:'linear', minFilter:'linear', mipmapFilter:'linear' });

  // Placeholder 2×2×2 volume texture until first frame arrives
  createVolumeTexture(2);

  const shaderCode = `
struct Uniforms {
  inv_vp : mat4x4<f32>,
  eye    : vec3<f32>,
  nsteps : u32,
  vmin   : f32,
  vmax   : f32,
  cw     : u32,
  ch     : u32,
}
@group(0) @binding(0) var<uniform>         u      : Uniforms;
@group(0) @binding(1) var                  volTex : texture_3d<f32>;
@group(0) @binding(2) var                  volSmp : sampler;
@group(0) @binding(3) var<storage,read>    tf     : array<vec4<f32>>;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var xy = array<vec2<f32>,4>(
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(1.0,1.0));
  var o: VOut;
  o.pos = vec4<f32>(xy[vi], 0.0, 1.0);
  o.uv  = xy[vi] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return o;
}

fn ray_aabb(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let inv = 1.0 / rd;
  let t1 = (-ro) * inv;
  let t2 = (vec3<f32>(1.0) - ro) * inv;
  return vec2<f32>(
    max(max(min(t1.x,t2.x), min(t1.y,t2.y)), min(t1.z,t2.z)),
    min(min(max(t1.x,t2.x), max(t1.y,t2.y)), max(t1.z,t2.z)));
}

@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  // Reconstruct ray direction from NDC
  let ndc  = vec4<f32>(uv * vec2<f32>(2.0,-2.0) + vec2<f32>(-1.0,1.0), 1.0, 1.0);
  let wld  = u.inv_vp * ndc;
  let rdir = normalize(wld.xyz / wld.w - u.eye);

  let t = ray_aabb(u.eye, rdir);
  if (t.x >= t.y) { return vec4<f32>(0.05,0.05,0.05,1.0); }

  let t0 = max(t.x, 0.0);
  let t1 = t.y;
  let dt = (t1 - t0) / f32(u.nsteps);

  var col = vec4<f32>(0.0);
  var tc  = t0 + 0.5 * dt;

  for (var i = 0u; i < u.nsteps; i++) {
    let pos = u.eye + tc * rdir;
    let raw = textureSample(volTex, volSmp, pos).r;
    let nm  = clamp((raw - u.vmin) / (u.vmax - u.vmin + 0.0001), 0.0, 1.0);
    let idx = u32(nm * 255.0);
    let rgba = tf[idx];
    let a = rgba.a * dt * f32(u.nsteps) * 0.08;
    col = vec4<f32>(col.rgb + (1.0 - col.a) * a * rgba.rgb,
                    col.a   + (1.0 - col.a) * a);
    if (col.a > 0.99) { break; }
    tc += dt;
  }

  let bg  = vec3<f32>(0.05);
  let rgb = mix(bg, col.rgb / max(col.a, 0.001), col.a);
  return vec4<f32>(pow(clamp(rgb, vec3(0.0), vec3(1.0)), vec3<f32>(0.4545)), 1.0);
}
`;

  const sm = device.createShaderModule({ code: shaderCode });
  pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex:   { module: sm, entryPoint: 'vs' },
    fragment: { module: sm, entryPoint: 'fs',
                targets: [{ format: fmt }] },
    primitive: { topology: 'triangle-strip' }
  });

  rebindGroup();
  return true;
}

function createVolumeTexture(sz) {
  if (volTex) volTex.destroy();
  volTex = device.createTexture({
    size: [sz, sz, sz],
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });
}

function rebindGroup() {
  if (!device || !pipeline || !volTex) return;
  bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: volTex.createView() },
      { binding: 2, resource: volSampler },
      { binding: 3, resource: { buffer: tfBuf } },
    ]
  });
}

// ── Camera math ────────────────────────────────────────────────────────────
function mat4_perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f/aspect,0,0,0,
    0,f,0,0,
    0,0,(far+near)*nf,-1,
    0,0,(2*far*near)*nf,0
  ]);
}
function mat4_lookat(eye, center, up) {
  const f=norm3(sub3(center,eye)), s=norm3(cross3(f,up)), u=cross3(s,f);
  return new Float32Array([
    s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0,
    -dot3(s,eye),-dot3(u,eye),dot3(f,eye),1
  ]);
}
function mat4_mul(a,b){const c=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[i+k*4]*b[k+j*4];c[i+j*4]=s;}return c;}
function mat4_inv(m){
  const s=new Float32Array(6),c=new Float32Array(6),o=new Float32Array(16);
  s[0]=m[0]*m[5]-m[4]*m[1]; s[1]=m[0]*m[9]-m[8]*m[1]; s[2]=m[0]*m[13]-m[12]*m[1];
  s[3]=m[4]*m[9]-m[8]*m[5]; s[4]=m[4]*m[13]-m[12]*m[5]; s[5]=m[8]*m[13]-m[12]*m[9];
  c[0]=m[2]*m[7]-m[6]*m[3]; c[1]=m[2]*m[11]-m[10]*m[3]; c[2]=m[2]*m[15]-m[14]*m[3];
  c[3]=m[6]*m[11]-m[10]*m[7]; c[4]=m[6]*m[15]-m[14]*m[7]; c[5]=m[10]*m[15]-m[14]*m[11];
  const det=1/(s[0]*c[5]-s[1]*c[4]+s[2]*c[3]+s[3]*c[2]-s[4]*c[1]+s[5]*c[0]);
  o[0]=(m[5]*c[5]-m[9]*c[4]+m[13]*c[3])*det; o[4]=(-m[4]*c[5]+m[8]*c[4]-m[12]*c[3])*det;
  o[8]=(m[7]*s[5]-m[11]*s[4]+m[15]*s[3])*det; o[12]=(-m[6]*s[5]+m[10]*s[4]-m[14]*s[3])*det;
  o[1]=(-m[1]*c[5]+m[9]*c[2]-m[13]*c[1])*det; o[5]=(m[0]*c[5]-m[8]*c[2]+m[12]*c[1])*det;
  o[9]=(-m[3]*s[5]+m[11]*s[2]-m[15]*s[1])*det; o[13]=(m[2]*s[5]-m[10]*s[2]+m[14]*s[1])*det;
  o[2]=(m[1]*c[4]-m[5]*c[2]+m[13]*c[0])*det; o[6]=(-m[0]*c[4]+m[4]*c[2]-m[12]*c[0])*det;
  o[10]=(m[3]*s[4]-m[7]*s[2]+m[15]*s[0])*det; o[14]=(-m[2]*s[4]+m[6]*s[2]-m[14]*s[0])*det;
  o[3]=(-m[1]*c[3]+m[5]*c[1]-m[9]*c[0])*det; o[7]=(m[0]*c[3]-m[4]*c[1]+m[8]*c[0])*det;
  o[11]=(-m[3]*s[3]+m[7]*s[1]-m[11]*s[0])*det; o[15]=(m[2]*s[3]-m[6]*s[1]+m[10]*s[0])*det;
  return o;
}
function sub3(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function dot3(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function cross3(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function norm3(a){const d=Math.sqrt(dot3(a,a));return[a[0]/d,a[1]/d,a[2]/d];}

function computeEye() {
  return [
    0.5 + radius * Math.sin(theta) * Math.cos(phi),
    0.5 + radius * Math.cos(theta),
    0.5 + radius * Math.sin(theta) * Math.sin(phi)
  ];
}

function updateUniforms() {
  if (!device || !uniformBuf) return;
  const W = canvas.width, H = canvas.height;
  const eye = computeEye();
  const proj = mat4_perspective(0.9, W / H, 0.01, 10.0);
  const view = mat4_lookat(eye, [0.5,0.5,0.5], [0,1,0]);
  const vp   = mat4_mul(proj, view);
  const inv  = mat4_inv(vp);

  const data = new Float32Array(24);
  data.set(inv, 0);                           // [0..15]  inv_vp
  data[16] = eye[0]; data[17] = eye[1]; data[18] = eye[2]; // [16..18] eye
  new Uint32Array(data.buffer)[19] = nsteps;  // [19]     nsteps
  data[20] = vmin;   data[21] = vmax;         // [20..21] vmin/vmax
  new Uint32Array(data.buffer)[22] = W;       // [22]     canvas_w
  new Uint32Array(data.buffer)[23] = H;       // [23]     canvas_h
  device.queue.writeBuffer(uniformBuf, 0, data);
}

// ── Render loop ────────────────────────────────────────────────────────────
function render() {
  if (!device || !pipeline || !bindGroup) { requestAnimationFrame(render); return; }

  // Resize canvas to match CSS size
  const cw = canvas.parentElement.clientWidth;
  const ch = canvas.parentElement.clientHeight;
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw; canvas.height = ch;
  }

  updateUniforms();
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: canvasCtx.getCurrentTexture().createView(),
      loadOp: 'clear', clearValue: { r:0.05,g:0.05,b:0.05,a:1 },
      storeOp: 'store'
    }]
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(4, 1, 0, 0);
  pass.end();
  device.queue.submit([enc.finish()]);
  requestAnimationFrame(render);
}

// ── Volume stream ingestion ────────────────────────────────────────────────
function ingestVolume(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset);
  let o = 0;
  if (dv.getUint32(o, true) !== 0xCFD00003) return; o += 4;
  const step = dv.getInt32(o, true); o += 4;
  const t = dv.getFloat64(o, true); o += 8;
  const nx = dv.getUint16(o, true); o += 2;
  /* ny = */ dv.getUint16(o, true); o += 2;
  /* nz = */ dv.getUint16(o, true); o += 2;
  o += 2; // pad
  vmin = dv.getFloat32(o, true); o += 4;
  vmax = dv.getFloat32(o, true); o += 4;
  domainL = dv.getFloat32(o, true); o += 4;
  /* var_id = */ dv.getUint8(o++);
  const compressed = dv.getUint8(o++);
  o += 2; // pad

  if (!device) return;

  let vol32;
  if (compressed) {
    const unc_size = dv.getUint32(o, true); o += 4;
    const u8 = lz4_decomp(bytes, o, bytes.length - o, unc_size);
    const udv = new DataView(u8.buffer);
    vol32 = new Float32Array(nx * nx * nx);
    for (let i = 0; i < vol32.length; i++) {
      const q = udv.getUint16(i * 2, true);
      vol32[i] = vmin + (q / 65535) * (vmax - vmin);
    }
  } else {
    vol32 = new Float32Array(bytes.buffer, bytes.byteOffset + o, nx*nx*nx);
  }

  if (N !== nx) { N = nx; createVolumeTexture(N); rebindGroup(); }

  device.queue.writeTexture(
    { texture: volTex },
    vol32,
    { bytesPerRow: N * 4, rowsPerImage: N },
    { width: N, height: N, depthOrArrayLayers: N }
  );

  document.getElementById('info').textContent =
    `step=${step} t=${t.toExponential(3)} N=${N} [${vmin.toPrecision(3)}, ${vmax.toPrecision(3)}]`;
  fetchMetrics();
}

// ── Stream connection ──────────────────────────────────────────────────────
async function connectStream() {
  document.getElementById('info').textContent = 'connecting…';
  try {
    const resp = await fetch(`/volume-stream`);
    document.getElementById('info').textContent = 'streaming';
    const reader = resp.body.getReader();
    let buf = new Uint8Array(0);
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const nb = new Uint8Array(buf.length + value.length);
      nb.set(buf); nb.set(value, buf.length); buf = nb;
      while (buf.length >= 4) {
        const flen = new DataView(buf.buffer, buf.byteOffset, 4).getUint32(0, true);
        if (buf.length < 4 + flen) break;
        ingestVolume(buf.subarray(4, 4 + flen));
        buf = buf.subarray(4 + flen);
      }
    }
  } catch(e) {
    document.getElementById('info').textContent = 'disconnected — retry in 3s';
    setTimeout(connectStream, 3000);
  }
}

// ── UI controls ───────────────────────────────────────────────────────────
function sendCfg() {
  const v = +document.getElementById('sv').value;
  fetch('/config', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ var: v }) }).catch(()=>{});
}
document.getElementById('sv').addEventListener('change', sendCfg);

document.getElementById('nsteps').addEventListener('input', e => {
  nsteps = +e.target.value;
  document.getElementById('lns').textContent = nsteps;
});
document.getElementById('opac').addEventListener('input', e => {
  opacScale = +e.target.value;
  document.getElementById('lop').textContent = opacScale;
  rebuildTF();
});
document.getElementById('cmap').addEventListener('change', e => {
  cmapId = +e.target.value; rebuildTF();
});

// ── Mouse arcball ──────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => { dragStart = [e.clientX, e.clientY]; });
canvas.addEventListener('mousemove', e => {
  if (!dragStart) return;
  const dx = (e.clientX - dragStart[0]) * 0.005;
  const dy = (e.clientY - dragStart[1]) * 0.005;
  phi   += dx;
  theta  = Math.max(0.05, Math.min(Math.PI - 0.05, theta + dy));
  dragStart = [e.clientX, e.clientY];
});
canvas.addEventListener('mouseup',   () => { dragStart = null; });
canvas.addEventListener('mouseleave',() => { dragStart = null; });
canvas.addEventListener('wheel', e => {
  radius = Math.max(0.6, Math.min(5.0, radius + e.deltaY * 0.002));
  e.preventDefault();
}, { passive: false });

// ── Touch support ──────────────────────────────────────────────────────────
let lastTouch = null;
canvas.addEventListener('touchstart', e => { if(e.touches.length===1) lastTouch=[e.touches[0].clientX,e.touches[0].clientY]; });
canvas.addEventListener('touchmove', e => {
  if(!lastTouch||e.touches.length!==1) return;
  phi   += (e.touches[0].clientX - lastTouch[0]) * 0.005;
  theta  = Math.max(0.05, Math.min(Math.PI-0.05, theta + (e.touches[0].clientY - lastTouch[1]) * 0.005));
  lastTouch=[e.touches[0].clientX,e.touches[0].clientY];
  e.preventDefault();
}, { passive:false });

// ── Boot ──────────────────────────────────────────────────────────────────
resizeSpk();
initWebGPU().then(ok => { if(ok) { render(); connectStream(); } });
</script>
</body>
</html>
)HTML";
    return html;
}

// =============================================================================
// Config setters (public, any thread)
// =============================================================================

void LiveStreamer::set_var(StreamVar v) noexcept {
    std::lock_guard<std::mutex> lk(cfg_mtx_);
    cfg_.var = v;
}
void LiveStreamer::set_axis(uint8_t a) noexcept {
    std::lock_guard<std::mutex> lk(cfg_mtx_);
    cfg_.axis = a;
}
void LiveStreamer::set_pos(double p) noexcept {
    std::lock_guard<std::mutex> lk(cfg_mtx_);
    cfg_.pos = p;
}
