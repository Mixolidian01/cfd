// R9-E1: Pure POSIX-socket / HTTP utility functions (no LiveStreamer state).
//
// See include/http_server.hpp for the declaration and design rationale.

#include "../include/http_server.hpp"

#include <cstring>
#include <string>

// POSIX
#include <sys/socket.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// http_safe_send
// ---------------------------------------------------------------------------

bool http_safe_send(int fd, const void* data, std::size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p   += n;
        len -= static_cast<std::size_t>(n);
    }
    return true;
}

// ---------------------------------------------------------------------------
// http_read_request — read until \r\n\r\n or 8 KB
// ---------------------------------------------------------------------------

std::string http_read_request(int fd) {
    std::string req;
    req.reserve(512);
    char buf[256];
    while (req.find("\r\n\r\n") == std::string::npos && req.size() < 8192) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        req.append(buf, static_cast<std::size_t>(n));
    }
    return req;
}

// ---------------------------------------------------------------------------
// http_content_length
// ---------------------------------------------------------------------------

int http_content_length(const std::string& req) {
    auto p = req.find("Content-Length:");
    if (p == std::string::npos) p = req.find("content-length:");
    if (p == std::string::npos) return 0;
    try { return std::stoi(req.substr(p + 15)); }
    catch (...) { return 0; }
}

// ---------------------------------------------------------------------------
// json_int / json_double — minimal single-key JSON parsers
// ---------------------------------------------------------------------------

bool json_int(const std::string& s, const std::string& key, int& out) {
    auto p = s.find('"' + key + '"');
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    try { out = std::stoi(s.substr(p + 1)); return true; }
    catch (...) { return false; }
}

bool json_double(const std::string& s, const std::string& key, double& out) {
    auto p = s.find('"' + key + '"');
    if (p == std::string::npos) return false;
    p = s.find(':', p);
    if (p == std::string::npos) return false;
    try { out = std::stod(s.substr(p + 1)); return true; }
    catch (...) { return false; }
}
