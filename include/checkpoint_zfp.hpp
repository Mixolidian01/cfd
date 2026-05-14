#pragma once
// P4.5 — Compressed checkpoint: ZFP when available, float32 fallback otherwise.
//
// Two compression modes are supported:
//
//   CKPT_COMPRESS_F32  — lossless within single precision (2× size reduction).
//                        Always available; no external library required.
//                        Relative field error ≤ 1.2e-7 (FLT_EPSILON / 2).
//
//   CKPT_COMPRESS_ZFP  — ZFP fixed-rate encoding (16–32× reduction at rate=8).
//                        Requires libzfp-dev (apt install libzfp-dev) and
//                        cmake re-configure with -DUSE_ZFP=ON.
//                        When ZFP is not compiled in, falls back to F32.
//
// File format (version 3 — extends v2 header):
//   [magic: uint64 = 0xCFD7CF07]
//   [version: uint32 = 3]
//   [compress: uint32 = 0 (none) | 1 (F32) | 2 (ZFP)]
//   [zfp_rate: double]  — bits per value (ZFP fixed-rate; 0 for F32/none)
//   [step: int32]  [t: double]
//   [NL: int32]
//   for each leaf:
//     [h: double]  [level: int32]
//     [NVAR * NCELL values as doubles / float32 / ZFP-stream]
//   topology section (same as v2)
//
// Usage:
//   // Save with float32 compression (2x, always available):
//   checkpoint_save_compressed(s, "restart.bin.f32", CKPT_COMPRESS_F32);
//
//   // Save with ZFP at 8 bits/value (if ZFP is compiled in):
//   checkpoint_save_compressed(s, "restart.bin.zfp", CKPT_COMPRESS_ZFP, 8.0);
//
//   // Load: mode is read from the file header automatically:
//   checkpoint_load_compressed(s, "restart.bin.f32");

#include "ns_solver.hpp"
#include <string>
#include <cstdint>

enum CkptCompressMode : uint32_t {
    CKPT_COMPRESS_NONE = 0,   // raw doubles (same as checkpoint_save/load)
    CKPT_COMPRESS_F32  = 1,   // float32 downcast  (2x, always available)
    CKPT_COMPRESS_ZFP  = 2,   // ZFP fixed-rate    (16-32x, requires libzfp)
};

// Whether ZFP support was compiled in (true if USE_ZFP=ON at cmake time).
bool ckpt_zfp_available() noexcept;

// Save with compression.  zfp_rate is bits-per-value for ZFP mode (ignored for F32).
void checkpoint_save_compressed(const NSSolver& s,
                                const std::string& path,
                                CkptCompressMode mode = CKPT_COMPRESS_F32,
                                double zfp_rate = 8.0);

// Load: decompresses automatically based on the mode stored in the file header.
void checkpoint_load_compressed(NSSolver& s, const std::string& path);
