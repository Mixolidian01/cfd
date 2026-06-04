// Gate: launcher — tests LiveStreamer launch state machine (LA1–LA4)
// No solver required — tests HTTP endpoints only.

#include "io/live_streamer.hpp"
#include <arpa/inet.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

static int nfail = 0;
static void check(bool ok, const char* tag, const char* msg) {
    if (ok) printf("  PASS  %s  %s\n", tag, msg);
    else   { printf("  FAIL  %s  %s\n", tag, msg); ++nfail; }
}

static std::string http_get(const char* path, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); return ""; }
    std::string req = std::string("GET ") + path + " HTTP/1.0\r\nHost: localhost\r\n\r\n";
    ::send(fd, req.c_str(), req.size(), 0);
    char buf[4096]; std::string resp;
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) resp.append(buf, (size_t)n);
    ::close(fd); return resp;
}

static int http_post(const char* path, const char* body, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(fd); return -1; }
    std::string req = std::string("POST ") + path + " HTTP/1.0\r\n"
        "Host: localhost\r\nContent-Type: application/json\r\n"
        "Content-Length: " + std::to_string(std::strlen(body)) + "\r\n\r\n" + body;
    ::send(fd, req.c_str(), req.size(), 0);
    char buf[256]; std::string resp;
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) resp.append(buf, (size_t)n);
    ::close(fd);
    int code = 0; std::sscanf(resp.c_str(), "HTTP/1.%*d %d", &code); return code;
}

int main() {
    StreamConfig cfg; cfg.port = 0;  // OS-assigned
    LiveStreamer streamer(cfg);
    usleep(100'000);  // give accept-loop time to start
    int port = streamer.port();

    // LA1: GET /status returns waiting before any POST /launch
    std::string r = http_get("/status", port);
    check(r.find("\"state\":\"waiting\"") != std::string::npos,
          "LA1", "GET /status returns waiting initially");

    // LA2: POST /launch returns 200
    int code = http_post("/launch", "{\"cfl\":0.5,\"t_end\":2.0,\"stream_port\":0}", port);
    check(code == 200, "LA2", "POST /launch returns 200");

    // LA3: pop_launch returns the posted JSON
    std::string got;
    bool popped = streamer.pop_launch(got);
    check(popped && got.find("\"cfl\"") != std::string::npos,
          "LA3", "pop_launch returns posted config JSON");

    // LA4: GET /status returns running after set_running()
    streamer.set_running();
    r = http_get("/status", port);
    check(r.find("\"state\":\"running\"") != std::string::npos,
          "LA4", "GET /status returns running after set_running");

    printf("\n%s  4 tests (LA1-LA4)\n", nfail == 0 ? "ALL PASS" : "SOME FAIL");
    return nfail;
}
