// R9-E1: Wire-format serialisers for 2-D slice and 3-D volume frames.
//
// See include/frame_packer.hpp for the public API.
//
// These functions were extracted from LiveStreamer::serialize_frame and
// LiveStreamer::serialize_volume.  They carry zero LiveStreamer state.

#include "../include/frame_packer.hpp"
#include "../include/cell_block.hpp"    // NB

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// LZ4 block compression — optional, guarded by HAVE_LZ4
#ifdef HAVE_LZ4
#include <lz4.h>
#endif

// =============================================================================
// Internal byte-packing helpers
// =============================================================================

static void push_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x));
    v.push_back(static_cast<uint8_t>(x >>  8));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 24));
}
static void push_i32(std::vector<uint8_t>& v, int32_t x) {
    push_u32(v, static_cast<uint32_t>(x));
}
static void push_f32(std::vector<uint8_t>& v, float x) {
    uint32_t u; std::memcpy(&u, &x, 4); push_u32(v, u);
}
static void push_f64(std::vector<uint8_t>& v, double x) {
    uint64_t u; std::memcpy(&u, &x, 8);
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(u >> (i * 8)));
}

// =============================================================================
// serialize_frame — pack FrameBuffer into a length-prefixed blob
//
// Wire format:
//   [4]    uint32  body length (bytes after these 4)
//   [32]   header  magic/step/time/n_blocks/axis/var_id/compressed/vmin/vmax/domainL
//   [n×16] block descriptors (always uncompressed)
//   data section — one of:
//     compressed=0 : n×NB×NB × float32
//     compressed=1 : [4] uint32 uncompressed_byte_count
//                    [?] LZ4 raw-block of (n×NB×NB × uint16 LE)
//                        uint16 q = round((val-vmin)/(vmax-vmin) * 65535)
// =============================================================================

void serialize_frame(const FrameBuffer& fb, std::vector<uint8_t>& out) {
    out.clear();
    const uint8_t n        = fb.n_blocks;
    const int     n_pixels = static_cast<int>(n) * NB * NB;

    std::vector<uint8_t> body;
    body.reserve(32u + static_cast<uint32_t>(n) * 16u + n_pixels * 4u);

    // ── Header (32 bytes) ────────────────────────────────────────────────────
    push_u32(body, 0xCFD00001u);    //  0-3  magic
    push_i32(body, fb.step);        //  4-7  step
    push_f64(body, fb.sim_time);    //  8-15 time
    body.push_back(n);              // 16    n_blocks
    body.push_back(fb.axis);        // 17    slice_axis
    body.push_back(fb.var_id);      // 18    var_id
    body.push_back(0);              // 19    compressed flag (patched below)
    const size_t compressed_off = body.size() - 1;
    push_f32(body, fb.g_vmin);      // 20-23 vmin
    push_f32(body, fb.g_vmax);      // 24-27 vmax
    push_f32(body, fb.domain_L);    // 28-31 domain_L
    // total header = 32 bytes ✓

    // ── Block descriptors (n × 16 bytes, always uncompressed) ────────────────
    for (uint8_t b = 0; b < n; ++b) {
        const BlockDesc2D& d = fb.descs[b];
        push_f32(body, d.ox2d);
        push_f32(body, d.oy2d);
        push_f32(body, d.h);
        body.push_back(d.level);
        body.push_back(0); body.push_back(0); body.push_back(0);
    }

    // ── Data section ─────────────────────────────────────────────────────────
#ifdef HAVE_LZ4
    const float rng_inv = (fb.g_vmax > fb.g_vmin)
                          ? 1.0f / (fb.g_vmax - fb.g_vmin) : 1.0f;

    std::vector<uint8_t> u16(static_cast<size_t>(n_pixels) * 2u);
    for (int i = 0; i < n_pixels; ++i) {
        float t = std::max(0.0f, std::min(1.0f,
                      (fb.data[i] - fb.g_vmin) * rng_inv));
        uint16_t q = static_cast<uint16_t>(std::lround(t * 65535.0f));
        u16[i*2]   = static_cast<uint8_t>(q);
        u16[i*2+1] = static_cast<uint8_t>(q >> 8);
    }

    const int src_bytes = static_cast<int>(u16.size());
    const int max_dst   = LZ4_compressBound(src_bytes);
    std::vector<char> lz4_out(static_cast<size_t>(max_dst));
    const int cmp_bytes = LZ4_compress_default(
        reinterpret_cast<const char*>(u16.data()),
        lz4_out.data(), src_bytes, max_dst);

    if (cmp_bytes > 0) {
        body[compressed_off] = 1;
        push_u32(body, static_cast<uint32_t>(src_bytes));
        body.insert(body.end(),
                    reinterpret_cast<const uint8_t*>(lz4_out.data()),
                    reinterpret_cast<const uint8_t*>(lz4_out.data()) + cmp_bytes);
    } else {
        for (int i = 0; i < n_pixels; ++i) push_f32(body, fb.data[i]);
    }
#else
    for (int i = 0; i < n_pixels; ++i) push_f32(body, fb.data[i]);
#endif

    const uint32_t body_len = static_cast<uint32_t>(body.size());
    out.resize(4u + body.size());
    out[0] = body_len & 0xffu; out[1] = (body_len >>  8) & 0xffu;
    out[2] = (body_len >> 16) & 0xffu; out[3] = (body_len >> 24) & 0xffu;
    std::memcpy(out.data() + 4, body.data(), body.size());
}

// =============================================================================
// serialize_volume — pack FrameBuffer3D into a length-prefixed blob
//
// Wire format:
//   [4]  uint32  body length
//   [40] header: magic(4) step(4) time(8) nx(2) ny(2) nz(2) pad(2)
//                g_vmin(4) g_vmax(4) domain_L(4) var_id(1) compressed(1) pad(2)
//   data: compressed=0 → nx*ny*nz float32 (z-major)
//         compressed=1 → [4] uint32 unc_bytes + LZ4(nx*ny*nz uint16 LE)
// =============================================================================

void serialize_volume(const FrameBuffer3D& fb, std::vector<uint8_t>& out) {
    out.clear();
    const int N     = static_cast<int>(fb.nx);   // nx == ny == nz
    const int n_vox = N * N * N;

    std::vector<uint8_t> body;
    body.reserve(40u + static_cast<size_t>(n_vox) * 4u);

    // ── Header (40 bytes) ────────────────────────────────────────────────────
    push_u32(body, 0xCFD00003u);             //  0-3  magic (3D frame)
    push_i32(body, fb.step);                //  4-7  step
    push_f64(body, fb.sim_time);            //  8-15 sim_time
    // 16-17 nx, 18-19 ny, 20-21 nz, 22-23 pad
    body.push_back(static_cast<uint8_t>(N));     body.push_back(static_cast<uint8_t>(N >> 8));
    body.push_back(static_cast<uint8_t>(N));     body.push_back(static_cast<uint8_t>(N >> 8));
    body.push_back(static_cast<uint8_t>(N));     body.push_back(static_cast<uint8_t>(N >> 8));
    body.push_back(0); body.push_back(0);    // pad
    push_f32(body, fb.g_vmin);              // 24-27 g_vmin
    push_f32(body, fb.g_vmax);              // 28-31 g_vmax
    push_f32(body, fb.domain_L);            // 32-35 domain_L
    body.push_back(fb.var_id);              // 36    var_id
    body.push_back(0);                      // 37    compressed (patched below)
    const size_t compressed_off = body.size() - 1;
    body.push_back(0); body.push_back(0);   // 38-39 pad
    // total header = 40 bytes ✓

    // ── Data section ─────────────────────────────────────────────────────────
#ifdef HAVE_LZ4
    const float rng_inv = (fb.g_vmax > fb.g_vmin)
                          ? 1.0f / (fb.g_vmax - fb.g_vmin) : 1.0f;

    std::vector<uint8_t> u16(static_cast<size_t>(n_vox) * 2u);
    for (int i = 0; i < n_vox; ++i) {
        float t = std::max(0.f, std::min(1.f,
                    (fb.data[i] - fb.g_vmin) * rng_inv));
        uint16_t q = static_cast<uint16_t>(std::lround(t * 65535.f));
        u16[i*2]   = static_cast<uint8_t>(q);
        u16[i*2+1] = static_cast<uint8_t>(q >> 8);
    }
    const int src_bytes = static_cast<int>(u16.size());
    const int max_dst   = LZ4_compressBound(src_bytes);
    std::vector<char> lz4_out(static_cast<size_t>(max_dst));
    const int cmp_bytes = LZ4_compress_default(
        reinterpret_cast<const char*>(u16.data()),
        lz4_out.data(), src_bytes, max_dst);

    if (cmp_bytes > 0) {
        body[compressed_off] = 1;
        push_u32(body, static_cast<uint32_t>(src_bytes));
        body.insert(body.end(),
                    reinterpret_cast<const uint8_t*>(lz4_out.data()),
                    reinterpret_cast<const uint8_t*>(lz4_out.data()) + cmp_bytes);
    } else {
        for (int i = 0; i < n_vox; ++i) push_f32(body, fb.data[i]);
    }
#else
    for (int i = 0; i < n_vox; ++i) push_f32(body, fb.data[i]);
#endif

    const uint32_t body_len = static_cast<uint32_t>(body.size());
    out.resize(4u + body.size());
    out[0] = body_len & 0xffu; out[1] = (body_len >>  8) & 0xffu;
    out[2] = (body_len >> 16) & 0xffu; out[3] = (body_len >> 24) & 0xffu;
    std::memcpy(out.data() + 4, body.data(), body.size());
}
