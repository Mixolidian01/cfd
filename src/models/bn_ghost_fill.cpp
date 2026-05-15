// bn_ghost_fill.cpp — Multi-block BN ghost fill using BlockTree neighbor table.
//
// Design: same-level face-to-face copy from neighbor BNCellBlock.
// Periodic boundary: uses tree.periodic_bc_ flag; wraps via tree.leaf_indices()
// and (level,morton) lookup (same pattern as BlockTree::fill_ghosts_periodic).
// Wall boundary: anti-symmetric momenta (rhou/rhov/rhow), symmetric otherwise.
// Coarse-fine: not yet implemented; BNSolver is flat-tree only for now.

#include "models/bn_solver.hpp"
#include "mesh/block_tree.hpp"
#include <unordered_map>
#include <cstdint>

void bn_fill_ghosts_tree(BlockTree& tree,
                         std::vector<BNCellBlock>& blocks,
                         const BCVariant& bc) noexcept {
    const auto& leaves = tree.leaf_indices();
    const int NL = (int)leaves.size();

    // Build (level,morton) → slot map for periodic wrap lookup.
    std::unordered_map<uint64_t, int> lm_map;
    lm_map.reserve(NL * 2);
    for (int ii = 0; ii < NL; ++ii) {
        const auto& nd = tree.nodes[leaves[ii]];
        lm_map[((uint64_t)nd.level << 32) | nd.morton] = ii;
    }

    // Determine if periodic wrap is needed for boundary faces.
    const bool is_periodic = std::holds_alternative<PeriodicBC>(bc);

    // Face specs: {ghost slab coord, source interior coord, axis, hi-side?}
    struct FaceSpec { int g_slab, s_slab, axis, hi; };
    static const FaceSpec specs[NFACES] = {
        {0,       ihi(), 0, 0},  // XMINUS: ghost i=0..NG-1
        {NB2-1,   ilo(), 0, 1},  // XPLUS:  ghost i=NB+NG..NB2-1
        {0,       ihi(), 1, 0},  // YMINUS
        {NB2-1,   ilo(), 1, 1},  // YPLUS
        {0,       ihi(), 2, 0},  // ZMINUS
        {NB2-1,   ilo(), 2, 1},  // ZPLUS
    };

    // Normal-momentum variable index per axis (rhou=1, rhov=2, rhow=3).
    static const int norm_var[3] = {1, 2, 3};

    for (int ii = 0; ii < NL; ++ii) {
        const auto& nd = tree.nodes[leaves[ii]];
        BNCellBlock& blk = blocks[ii];

        for (int d = 0; d < NFACES; ++d) {
            int ni = nd.neighbours[d];
            const FaceSpec& sp = specs[d];

            // ── Same-level neighbour: direct copy ────────────────────────────
            if (ni >= 0 && tree.nodes[ni].is_leaf()) {
                // Find slot for neighbor ni.
                const auto& nnd = tree.nodes[ni];
                auto it = lm_map.find(((uint64_t)nnd.level << 32) | nnd.morton);
                if (it == lm_map.end()) continue;
                const BNCellBlock& src = blocks[it->second];

                // Copy NG ghost layers on face d from src interior.
                for (int g = 0; g < NG; ++g) {
                    int g_coord = (sp.hi) ? (NB + NG + g) : (NG - 1 - g);
                    int s_coord = (sp.hi) ? (NG + g)       : (NB + NG - 1 - g);
                    if (sp.axis == 0) {
                        for (int k = 0; k < NB2; ++k)
                        for (int j = 0; j < NB2; ++j) {
                            int df = cell_idx(g_coord, j, k);
                            int sf = cell_idx(s_coord, j, k);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = src.Q[v][sf];
                        }
                    } else if (sp.axis == 1) {
                        for (int k = 0; k < NB2; ++k)
                        for (int i = 0; i < NB2; ++i) {
                            int df = cell_idx(i, g_coord, k);
                            int sf = cell_idx(i, s_coord, k);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = src.Q[v][sf];
                        }
                    } else {
                        for (int j = 0; j < NB2; ++j)
                        for (int i = 0; i < NB2; ++i) {
                            int df = cell_idx(i, j, g_coord);
                            int sf = cell_idx(i, j, s_coord);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = src.Q[v][sf];
                        }
                    }
                }
                continue;
            }

            // ── Domain boundary (ni == -1) ───────────────────────────────────
            if (ni != -1) continue;  // ni >= 0 but not a leaf → skip (coarse/fine)

            if (is_periodic) {
                // Periodic wrap: look up the block at the wrapped Morton code.
                int lev = nd.level;
                uint32_t mx, my, mz;
                morton_decode(nd.morton, mx, my, mz);
                uint32_t max_coord = (lev > 0) ? ((1u << lev) - 1u) : 0u;
                static const int face_axis[NFACES]  = {0,0,1,1,2,2};
                static const int face_hi  [NFACES]  = {0,1,0,1,0,1};
                int ax = face_axis[d];
                if (ax == 0) mx = face_hi[d] ? 0 : max_coord;
                else if (ax == 1) my = face_hi[d] ? 0 : max_coord;
                else              mz = face_hi[d] ? 0 : max_coord;

                uint64_t key = ((uint64_t)lev << 32) | morton_encode(mx, my, mz);
                auto it = lm_map.find(key);
                const BNCellBlock* src_blk = nullptr;
                if (it != lm_map.end()) src_blk = &blocks[it->second];

                if (!src_blk) {
                    // Single-block or no periodic neighbor found: self-wrap.
                    src_blk = &blk;
                }

                for (int g = 0; g < NG; ++g) {
                    int g_coord = (sp.hi) ? (NB + NG + g) : (NG - 1 - g);
                    int s_coord = (sp.hi) ? (NG + g)       : (NB + NG - 1 - g);
                    if (src_blk == &blk) {
                        // Self-wrap: remap to opposite interior.
                        s_coord = (sp.hi) ? (NG + g) : (NB + NG - 1 - g);
                    }
                    if (sp.axis == 0) {
                        for (int k = 0; k < NB2; ++k)
                        for (int j = 0; j < NB2; ++j) {
                            int df = cell_idx(g_coord, j, k);
                            int sf = cell_idx(s_coord, j, k);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = src_blk->Q[v][sf];
                        }
                    } else if (sp.axis == 1) {
                        for (int k = 0; k < NB2; ++k)
                        for (int i = 0; i < NB2; ++i) {
                            int df = cell_idx(i, g_coord, k);
                            int sf = cell_idx(i, s_coord, k);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = src_blk->Q[v][sf];
                        }
                    } else {
                        for (int j = 0; j < NB2; ++j)
                        for (int i = 0; i < NB2; ++i) {
                            int df = cell_idx(i, j, g_coord);
                            int sf = cell_idx(i, j, s_coord);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = src_blk->Q[v][sf];
                        }
                    }
                }
            } else {
                // Wall (reflecting): anti-symmetric normal momentum, symmetric rest.
                // ghost = interior mirror; normal momentum flipped.
                int nv = norm_var[sp.axis];
                for (int g = 0; g < NG; ++g) {
                    int g_coord = (sp.hi) ? (NB + NG + g) : (NG - 1 - g);
                    int s_coord = (sp.hi) ? (NB + NG - 1 - g) : (NG + g);
                    if (sp.axis == 0) {
                        for (int k = 0; k < NB2; ++k)
                        for (int j = 0; j < NB2; ++j) {
                            int df = cell_idx(g_coord, j, k);
                            int sf = cell_idx(s_coord, j, k);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = blk.Q[v][sf];
                            blk.Q[nv][df] = -blk.Q[nv][sf];
                        }
                    } else if (sp.axis == 1) {
                        for (int k = 0; k < NB2; ++k)
                        for (int i = 0; i < NB2; ++i) {
                            int df = cell_idx(i, g_coord, k);
                            int sf = cell_idx(i, s_coord, k);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = blk.Q[v][sf];
                            blk.Q[nv][df] = -blk.Q[nv][sf];
                        }
                    } else {
                        for (int j = 0; j < NB2; ++j)
                        for (int i = 0; i < NB2; ++i) {
                            int df = cell_idx(i, j, g_coord);
                            int sf = cell_idx(i, j, s_coord);
                            for (int v = 0; v < NVAR_BN; ++v)
                                blk.Q[v][df] = blk.Q[v][sf];
                            blk.Q[nv][df] = -blk.Q[nv][sf];
                        }
                    }
                }
            }
        }
    }
}
