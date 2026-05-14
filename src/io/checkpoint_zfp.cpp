// P4.5 — Compressed checkpoint implementation.
//
// Compression paths:
//   F32   : double → float32 downcast (exact at single-precision).
//           Block layout: NVAR × NCELL × sizeof(float) per leaf.
//           Always compiled in; no external dependency.
//
//   ZFP   : Compiled in only when USE_ZFP=ON (libzfp-dev installed).
//           Uses zfp_compress() / zfp_decompress() from <zfp.h>.
//           Fixed-rate mode: zfp_rate bits/value (typ. 8–24).
//           Per-leaf stream stored with an 8-byte size prefix.
//
// Format version = 3 (extends v2; see checkpoint_zfp.hpp for layout).

#include "io/checkpoint_zfp.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unistd.h>

#ifdef HAVE_ZFP
#  include <zfp.h>
#endif

static constexpr uint64_t MAGIC_C  = 0xCFD7CF07ULL;
static constexpr uint32_t VERSION3 = 3;

// ── ZFP availability ──────────────────────────────────────────────────────────
bool ckpt_zfp_available() noexcept {
#ifdef HAVE_ZFP
    return true;
#else
    return false;
#endif
}

// ── F32 helpers ───────────────────────────────────────────────────────────────
static void compress_f32(const double* src, int n, std::vector<uint8_t>& dst) {
    const size_t nbytes = (size_t)n * sizeof(float);
    dst.resize(nbytes);
    float* fp = reinterpret_cast<float*>(dst.data());
    for (int i = 0; i < n; ++i) fp[i] = (float)src[i];
}

static void decompress_f32(const uint8_t* src, int n, double* dst) {
    const float* fp = reinterpret_cast<const float*>(src);
    for (int i = 0; i < n; ++i) dst[i] = (double)fp[i];
}

// ── ZFP helpers ───────────────────────────────────────────────────────────────
#ifdef HAVE_ZFP
static void compress_zfp(const double* src, int n, double rate,
                          std::vector<uint8_t>& dst) {
    zfp_type type   = zfp_type_double;
    // Use 3D mode for full CellBlock variables (NB2³): exploits spatial
    // correlation in all 3 directions, giving ~4× better accuracy than 1D
    // at the same rate.  Fall back to 1D for other sizes.
    zfp_field* field = (n == NCELL)
        ? zfp_field_3d(const_cast<double*>(src), type, NB2, NB2, NB2)
        : zfp_field_1d(const_cast<double*>(src), type, (uint)n);
    zfp_stream* zfp = zfp_stream_open(nullptr);
    zfp_stream_set_rate(zfp, rate, type, (n == NCELL) ? 3 : 1, 0);

    const size_t bufsize = zfp_stream_maximum_size(zfp, field);
    dst.resize(bufsize);
    bitstream* bs = stream_open(dst.data(), bufsize);
    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);

    const size_t compressed = zfp_compress(zfp, field);
    dst.resize(compressed);

    stream_close(bs);
    zfp_field_free(field);
    zfp_stream_close(zfp);
}

static void decompress_zfp(const uint8_t* src, size_t src_size,
                            int n, double rate, double* dst) {
    zfp_type type    = zfp_type_double;
    zfp_field* field = (n == NCELL)
        ? zfp_field_3d(dst, type, NB2, NB2, NB2)
        : zfp_field_1d(dst, type, (uint)n);
    zfp_stream* zfp  = zfp_stream_open(nullptr);
    zfp_stream_set_rate(zfp, rate, type, (n == NCELL) ? 3 : 1, 0);

    bitstream* bs = stream_open(const_cast<uint8_t*>(src), src_size);
    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);

    zfp_decompress(zfp, field);

    stream_close(bs);
    zfp_field_free(field);
    zfp_stream_close(zfp);
}
#endif // HAVE_ZFP

// ── I/O helpers ───────────────────────────────────────────────────────────────
static void write_pod(FILE* f, const void* p, size_t n) {
    if (std::fwrite(p, 1, n, f) != n)
        throw std::runtime_error("checkpoint_save_compressed: write error");
}
static void read_pod(FILE* f, void* p, size_t n) {
    if (std::fread(p, 1, n, f) != n)
        throw std::runtime_error("checkpoint_load_compressed: read error");
}

// =============================================================================
// checkpoint_save_compressed
// =============================================================================
void checkpoint_save_compressed(const NSSolver& s,
                                const std::string& path,
                                CkptCompressMode mode,
                                double zfp_rate)
{
#ifndef HAVE_ZFP
    // Silently downgrade ZFP request to F32 when library not compiled in
    if (mode == CKPT_COMPRESS_ZFP) mode = CKPT_COMPRESS_F32;
#endif

    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("checkpoint_save_compressed: cannot open " + tmp);

    // ── Header ────────────────────────────────────────────────────────────────
    write_pod(f, &MAGIC_C,   sizeof(MAGIC_C));
    write_pod(f, &VERSION3,  sizeof(VERSION3));
    uint32_t mode32 = (uint32_t)mode;
    write_pod(f, &mode32,    sizeof(mode32));
    write_pod(f, &zfp_rate,  sizeof(zfp_rate));

    int32_t step32 = (int32_t)s.step;
    write_pod(f, &step32, sizeof(step32));
    write_pod(f, &s.t,    sizeof(s.t));

    auto leaves = s.tree.leaf_indices();
    int32_t NL  = (int32_t)leaves.size();
    write_pod(f, &NL, sizeof(NL));

    // ── Field data (compressed) ───────────────────────────────────────────────
    const int N = NVAR * NCELL;  // values per leaf
    std::vector<double> raw(N);
    std::vector<uint8_t> cbuf;

    for (int li : leaves) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double   h  = blk.h;
        int32_t  lv = node.level;
        write_pod(f, &h,  sizeof(h));
        write_pod(f, &lv, sizeof(lv));

        // Flatten NVAR × NCELL into raw[] (flat SoA)
        for (int v = 0; v < NVAR; ++v)
            blk.Q[v].copy_to_flat(raw.data() + v * NCELL);

        if (mode == CKPT_COMPRESS_F32) {
            compress_f32(raw.data(), N, cbuf);
            // Write cbuf size then data (size is deterministic = N*sizeof(float),
            // but we store it for uniform load logic)
            uint64_t sz = cbuf.size();
            write_pod(f, &sz, sizeof(sz));
            write_pod(f, cbuf.data(), sz);
        }
#ifdef HAVE_ZFP
        else if (mode == CKPT_COMPRESS_ZFP) {
            // Compress each variable independently (1D ZFP on NCELL values per var)
            // so that per-variable error bounds are meaningful.
            for (int v = 0; v < NVAR; ++v) {
                compress_zfp(raw.data() + v*NCELL, NCELL, zfp_rate, cbuf);
                uint64_t sz = cbuf.size();
                write_pod(f, &sz, sizeof(sz));
                write_pod(f, cbuf.data(), sz);
            }
        }
#endif
        else {
            // CKPT_COMPRESS_NONE: raw doubles
            uint64_t sz = (uint64_t)N * sizeof(double);
            write_pod(f, &sz, sizeof(sz));
            write_pod(f, raw.data(), sz);
        }
    }

    // ── Topology section (identical to v2) ───────────────────────────────────
    for (int li : leaves) {
        uint32_t code = s.tree.nodes[li].morton;
        uint32_t lv   = (uint32_t)s.tree.nodes[li].level;
        write_pod(f, &code, sizeof(code));
        write_pod(f, &lv,   sizeof(lv));
    }

    if (std::fflush(f) != 0)
        throw std::runtime_error("checkpoint_save_compressed: fflush failed");
    if (fsync(fileno(f)) != 0)
        throw std::runtime_error("checkpoint_save_compressed: fsync failed");
    std::fclose(f);

    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("checkpoint_save_compressed: rename failed");
}

// =============================================================================
// checkpoint_load_compressed
// =============================================================================
void checkpoint_load_compressed(NSSolver& s, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("checkpoint_load_compressed: cannot open " + path);

    // ── Header ────────────────────────────────────────────────────────────────
    uint64_t magic = 0;
    read_pod(f, &magic, sizeof(magic));
    if (magic != MAGIC_C)
        throw std::runtime_error("checkpoint_load_compressed: bad magic");

    uint32_t ver = 0;
    read_pod(f, &ver, sizeof(ver));
    if (ver != VERSION3)
        throw std::runtime_error("checkpoint_load_compressed: version "
            + std::to_string(ver) + " (expected " + std::to_string(VERSION3) + ")");

    uint32_t mode32 = 0;
    read_pod(f, &mode32, sizeof(mode32));
    CkptCompressMode mode = (CkptCompressMode)mode32;
    double zfp_rate = 0.0;
    read_pod(f, &zfp_rate, sizeof(zfp_rate));

    int32_t step32 = 0;
    read_pod(f, &step32, sizeof(step32));
    s.step = step32;
    read_pod(f, &s.t, sizeof(s.t));

    int32_t NL = 0;
    read_pod(f, &NL, sizeof(NL));

    // ── Buffer field data ─────────────────────────────────────────────────────
    struct LeafData {
        double  h;
        int32_t level;
        std::vector<double> Q_flat;  // NVAR * NCELL doubles
    };
    std::vector<LeafData> field_data((size_t)NL);
    const int N = NVAR * NCELL;
    std::vector<uint8_t> cbuf;

    for (int ii = 0; ii < NL; ++ii) {
        auto& ld = field_data[ii];
        read_pod(f, &ld.h,     sizeof(ld.h));
        read_pod(f, &ld.level, sizeof(ld.level));
        ld.Q_flat.resize((size_t)N);

        if (mode == CKPT_COMPRESS_F32) {
            uint64_t sz = 0;
            read_pod(f, &sz, sizeof(sz));
            cbuf.resize(sz);
            read_pod(f, cbuf.data(), sz);
            decompress_f32(cbuf.data(), N, ld.Q_flat.data());
        }
#ifdef HAVE_ZFP
        else if (mode == CKPT_COMPRESS_ZFP) {
            for (int v = 0; v < NVAR; ++v) {
                uint64_t sz = 0;
                read_pod(f, &sz, sizeof(sz));
                cbuf.resize(sz);
                read_pod(f, cbuf.data(), sz);
                decompress_zfp(cbuf.data(), sz, NCELL, zfp_rate,
                               ld.Q_flat.data() + v*NCELL);
            }
        }
#endif
        else {
            uint64_t sz = 0;
            read_pod(f, &sz, sizeof(sz));
            cbuf.resize(sz);
            read_pod(f, cbuf.data(), sz);
            std::memcpy(ld.Q_flat.data(), cbuf.data(), sz);
        }
    }

    // ── Read topology ─────────────────────────────────────────────────────────
    std::vector<uint32_t> topo_code((size_t)NL), topo_level((size_t)NL);
    for (int ii = 0; ii < NL; ++ii) {
        read_pod(f, &topo_code[ii],  sizeof(uint32_t));
        read_pod(f, &topo_level[ii], sizeof(uint32_t));
    }
    std::fclose(f);

    // ── Replay topology (same as checkpoint_load) ─────────────────────────────
    s.tree.init(s.tree.domain_L());

    std::vector<int> order((size_t)NL);
    for (int i = 0; i < NL; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (topo_level[a] != topo_level[b]) return topo_level[a] < topo_level[b];
        return topo_code[a] < topo_code[b];
    });

    for (int ii : order) {
        uint32_t target_code  = topo_code[ii];
        uint32_t target_level = topo_level[ii];
        if (target_level == 0) continue;
        int cur = 0;
        for (uint32_t lv = 0; lv < target_level; ++lv) {
            if (s.tree.nodes[cur].is_leaf())
                s.tree.refine(cur);
            int oct = (int)((target_code >> (3u * (target_level - 1u - lv))) & 7u);
            cur = s.tree.nodes[cur].first_child + oct;
        }
    }
    s.tree.rebuild_neighbours();

    // ── Write field data into rebuilt leaves ──────────────────────────────────
    auto final_leaves = s.tree.leaf_indices();
    if ((int32_t)final_leaves.size() != NL)
        throw std::runtime_error(
            "checkpoint_load_compressed: topology replay gave "
            + std::to_string(final_leaves.size())
            + " leaves but checkpoint has " + std::to_string(NL));

    for (int ii = 0; ii < NL; ++ii) {
        auto& blk = *s.tree.nodes[final_leaves[ii]].block;
        const double* flat = field_data[ii].Q_flat.data();
        for (int v = 0; v < NVAR; ++v)
            blk.Q[v].assign_from_flat(flat + v * NCELL);
    }

    s.alloc_scratch();
}
