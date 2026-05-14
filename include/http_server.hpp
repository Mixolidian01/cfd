#pragma once
// R9-E1: Pure POSIX-socket / HTTP utility functions extracted from live_streamer.cpp.
//
// These helpers carry no LiveStreamer state.  They are declared here so that
// http_server.cpp is the single definition point (ODR), and so that any
// future TU that needs them can include this header instead of forward-
// declaring them by hand.
//
// Note: the accept-loop, connection dispatcher, and all handle_* methods in
// LiveStreamer are NOT extracted here because they directly access LiveStreamer
// member variables (cfg_, shutdown_, stream_fd_, swap_mtx_, front_, …).
// Forcing a callback-based separation would require a large API surface for
// little gain.  The pure socket utilities below are the extractable boundary.

#include <cstdint>
#include <string>

// Send all bytes in [data, data+len) on fd.  Returns false on error.
bool http_safe_send(int fd, const void* data, std::size_t len);

// Receive HTTP headers from fd until \r\n\r\n or 8 KB limit.
std::string http_read_request(int fd);

// Return value of "Content-Length:" header, or 0 if absent.
int http_content_length(const std::string& req);

// Minimal JSON key-value extractors (integer and double).
// Locate the first occurrence of "key": in s and parse the value.
bool json_int   (const std::string& s, const std::string& key, int&    out);
bool json_double(const std::string& s, const std::string& key, double& out);
