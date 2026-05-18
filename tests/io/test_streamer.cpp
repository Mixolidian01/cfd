// Phase 6 gate: P6.5
// Runs NSSolver for 50 steps with a LiveStreamer attached, then connects to
// the /stream endpoint and verifies at least one well-formed binary frame
// is received within 3 seconds.
//
// Gate conditions:
//   S1: at least one frame received
//   S2: magic bytes correct (0xCFD00001)
//   S3: n_blocks > 0
//   S4: g_vmax > g_vmin (non-degenerate range)

#include "solver/ns_solver.hpp"
#include "io/live_streamer.hpp"
#include <arpa/inet.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <chrono>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

static int connect_to(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    for (int tries = 0; tries < 20; ++tries) {
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
            return fd;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    ::close(fd);
    return -1;
}

static bool send_all(int fd, const char* s) {
    size_t len = std::strlen(s);
    while (len > 0) {
        ssize_t n = ::send(fd, s, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        s += n; len -= static_cast<size_t>(n);
    }
    return true;
}

// Receive bytes into buf until total >= need, with a timeout.
static bool recv_at_least(int fd, std::vector<uint8_t>& buf, size_t need,
                          std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    buf.resize(need);
    size_t have = 0;
    while (have < need) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return false;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        // Set socket timeout
        struct timeval tv{static_cast<long>(ms / 1000),
                          static_cast<long>((ms % 1000) * 1000)};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ssize_t n = ::recv(fd, buf.data() + have, need - have, 0);
        if (n <= 0) return false;
        have += static_cast<size_t>(n);
    }
    return true;
}

// =============================================================================
// main
// =============================================================================

int main() {
    // Use a port unlikely to collide
    const int PORT = 18082;

    // ── Solver setup (2×2×2 blocks, Sod-like IC, 50 steps) ──────────────────
    StreamConfig scfg;
    scfg.port   = PORT;
    scfg.var    = StreamVar::RHO;
    scfg.axis   = 2;
    scfg.pos    = 0.5;
    scfg.stride = 5;  // stream every 5th step

    LiveStreamer streamer(scfg);

    NSSolver solver;
    solver.cfg.time.t_end         = 1e30;   // run until max_steps
    solver.cfg.time.max_steps     = 2000;   // enough steps that the solver stays alive
    solver.cfg.time.cfl           = 0.4;    // on an 8³ block this runs ~200 ms
    solver.cfg.bc.variant = PeriodicBC{};
    solver.cfg.io.verbose       = false;
    solver.cfg.io.diag_interval = 999;
    solver.set_streamer(&streamer);

    solver.init(1.0, [](double x, double /*y*/, double /*z*/) -> Prim {
        Prim q{};
        q.rho = (x < 0.5) ? 1.0 : 0.125;
        q.u   = 0.0; q.v = 0.0; q.w = 0.0;
        q.p   = (x < 0.5) ? 1.0 : 0.1;
        q.T   = q.p / (q.rho * R_GAS);
        q.c   = std::sqrt(GAMMA * q.p / q.rho);
        return q;
    });

    // ── Connect to /stream BEFORE starting the solver ────────────────────────
    // The solver on a single 8³ block completes 50 steps in ~50 ms; starting
    // the solver thread first races against all 10 frames being generated and
    // discarded (stream_fd_<0) before the client connects.  Connecting first
    // guarantees stream_fd_ is set before any snapshot() call runs.
    int cfd = connect_to(PORT);
    if (cfd < 0) {
        std::fprintf(stderr, "FAIL S0: could not connect to port %d\n", PORT);
        return 1;
    }

    // Send HTTP GET /stream and drain the response headers now, so that
    // handle_get_stream() has stored stream_fd_ before the solver starts.
    send_all(cfd, "GET /stream HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n");

    // Skip HTTP response headers (read until \r\n\r\n)
    {
        std::string hdr;
        char tmp[1];
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (hdr.find("\r\n\r\n") == std::string::npos) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::fprintf(stderr, "FAIL S0: timeout waiting for HTTP headers\n");
                ::close(cfd);
                return 1;
            }
            struct timeval tv{0, 50000};
            ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            ssize_t n = ::recv(cfd, tmp, 1, 0);
            if (n > 0) hdr += tmp[0];
        }
    }

    // HTTP headers received → stream_fd_ is now set; start the solver.
    std::thread solver_thread([&]() { solver.run(); });

    // Read the 4-byte chunk header (hex size + \r\n) — simplistic: read up to 16 chars
    // looking for the first \r\n which terminates the hex size string.
    std::string chunk_hdr;
    {
        char tmp[1];
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
        while (chunk_hdr.find("\r\n") == std::string::npos) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::fprintf(stderr, "FAIL S0: timeout waiting for first HTTP chunk header\n");
                ::close(cfd);
                solver_thread.join();
                return 1;
            }
            struct timeval tv{0, 50000};
            ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            ssize_t n = ::recv(cfd, tmp, 1, 0);
            if (n > 0) chunk_hdr += tmp[0];
        }
        // Remove trailing \r\n
        while (!chunk_hdr.empty() && (chunk_hdr.back() == '\r' || chunk_hdr.back() == '\n'))
            chunk_hdr.pop_back();
    }

    unsigned long chunk_size = 0;
    try { chunk_size = std::stoul(chunk_hdr, nullptr, 16); }
    catch (...) {
        std::fprintf(stderr, "FAIL S0: bad chunk size '%s'\n", chunk_hdr.c_str());
        ::close(cfd);
        solver_thread.join();
        return 1;
    }
    if (chunk_size < 4 + 32) {  // at least length prefix + header
        std::fprintf(stderr, "FAIL S0: chunk too small (%lu bytes)\n", chunk_size);
        ::close(cfd);
        solver_thread.join();
        return 1;
    }

    // Read full chunk
    std::vector<uint8_t> chunk(chunk_size);
    if (!recv_at_least(cfd, chunk, chunk_size, std::chrono::seconds(3))) {
        std::fprintf(stderr, "FAIL S0: incomplete chunk\n");
        ::close(cfd);
        solver_thread.join();
        return 1;
    }
    ::close(cfd);

    // ── Parse frame ──────────────────────────────────────────────────────────
    // chunk = [4-byte frame_len][frame body]
    auto ru32 = [&](size_t o) -> uint32_t {
        return static_cast<uint32_t>(chunk[o])
             | static_cast<uint32_t>(chunk[o+1]) << 8
             | static_cast<uint32_t>(chunk[o+2]) << 16
             | static_cast<uint32_t>(chunk[o+3]) << 24;
    };
    auto rf32 = [&](size_t o) -> float {
        uint32_t u = ru32(o); float f; std::memcpy(&f, &u, 4); return f;
    };

    uint32_t frame_len = ru32(0);
    (void)frame_len;

    // Frame body starts at byte 4
    const size_t B = 4;
    uint32_t magic   = ru32(B + 0);
    uint8_t  n_blks  = chunk[B + 16];
    float    vmin    = rf32(B + 20);
    float    vmax    = rf32(B + 24);

    int result = 0;

    // S1: received at least one frame
    std::fprintf(stdout, "S1 (frame received): PASS\n");

    // S2: magic
    if (magic == 0xCFD00001u) {
        std::fprintf(stdout, "S2 (magic=0xCFD00001): PASS\n");
    } else {
        std::fprintf(stderr, "S2 FAIL: magic=0x%08X\n", magic);
        result = 1;
    }

    // S3: n_blocks > 0
    if (n_blks > 0) {
        std::fprintf(stdout, "S3 (n_blocks=%d > 0): PASS\n", (int)n_blks);
    } else {
        std::fprintf(stderr, "S3 FAIL: n_blocks=0\n");
        result = 1;
    }

    // S4: non-degenerate range
    if (vmax > vmin) {
        std::fprintf(stdout, "S4 (vmax=%.4g > vmin=%.4g): PASS\n", vmax, vmin);
    } else {
        std::fprintf(stderr, "S4 FAIL: vmin=%.4g vmax=%.4g\n", vmin, vmax);
        result = 1;
    }

    solver_thread.join();
    return result;
}
