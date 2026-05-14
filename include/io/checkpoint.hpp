#pragma once
// checkpoint.hpp — binary checkpoint save/load for NSSolver state
//
// File format v2 (little-endian, packed):
//   [magic:   uint64  = 0xCFD7CF07]
//   [version: uint32  = 2         ]
//   [step:    int32 ] [t: double  ]
//   [NL:      int32 ]  — number of leaf blocks
//   for each leaf (NL entries, in leaf_indices() order):
//     [h:     double ]
//     [level: int32  ]
//     [Q:     NVAR * NB2^3 doubles]  — all cells including ghosts
//   topology (NL entries, same order):
//     [morton: uint32] [level: uint32]
//
// Changes from v1:
//   - Topology section appended (FIX B7): load() can now reconstruct any
//     refined AMR tree, not just a flat single-level mesh.
//   - Atomic write via tmp + rename + fsync (FIX P6): crash-safe saves.
//   - Version bumped to 2; v1 files rejected with a clear error.
//
// Usage:
//   checkpoint_save(solver, "restart.bin");
//   checkpoint_load(solver, "restart.bin");

#include "solver/ns_solver.hpp"
#include <string>

void checkpoint_save(const NSSolver& s, const std::string& path);
void checkpoint_load(NSSolver& s,       const std::string& path);
