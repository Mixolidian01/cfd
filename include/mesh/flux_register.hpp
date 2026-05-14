#pragma once
// FluxRegister — stores fine-face fluxes at coarse/fine boundaries.
// For each coarse face shared with a fine block, stores the sum of the
// two (2D) or four (3D) fine-face fluxes that replace the single coarse flux.
//
// Usage per RK stage:
//   register.accumulate(fine_block, dt, axis, side)
// After stage 3:
//   register.correct(coarse_block, dt, h_coarse)

#include "mesh/cell_block.hpp"
#include <cstdint>          // uint64_t
#include <cstring>          // std::memset
#include <unordered_map>
#include <array>

// One register entry per coarse face (identified by coarse cell + axis + side).
// Stores the sum of NVAR*NB*NB fine-face fluxes that replace the coarse flux.
struct FaceRegister {
    // face_flux[v][jf][kf] = sum of fine fluxes for variable v at fine-face (jf,kf)
    static constexpr int NF = NB;  // NB fine faces per coarse face per tangential axis
    double flux[NVAR][NF][NF] = {};
    bool   accumulated = false;

    void zero() { std::memset(flux, 0, sizeof(flux)); accumulated = false; }
};

// Key: (coarse_block_id, axis, side, ci, cj, ck) where (ci,cj,ck) is the
// coarse interior face cell index. We encode as a single 64-bit key.
inline uint64_t face_key(int blk, int axis, int side, int ci, int cj, int ck) {
    return ((uint64_t)blk << 32) | ((uint64_t)axis << 28) | ((uint64_t)side << 24)
         | ((uint64_t)(ci & 0xFF) << 16) | ((uint64_t)(cj & 0xFF) << 8) | (uint64_t)(ck & 0xFF);
}

class FluxRegister {
public:
    void zero_all()  { for (auto& [k,v] : entries_) v.zero(); }
    FaceRegister& get(uint64_t key) { return entries_[key]; }
    bool has(uint64_t key) const { return entries_.count(key) > 0; }
    void clear() { entries_.clear(); }

private:
    std::unordered_map<uint64_t, FaceRegister> entries_;
};
