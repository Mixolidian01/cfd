// bn_ghost_fill.cpp — Multi-block BN ghost fill using BlockTree neighbor table.
//
// Design: same-level face-to-face copy from neighbor BNCellBlock.
// Periodic boundary: uses tree.periodic_bc_ flag; wraps via tree.leaf_indices()
// and (level,morton) lookup (same pattern as BlockTree::fill_ghosts_periodic).
// Wall boundary: anti-symmetric momenta (rhou/rhov/rhow), symmetric otherwise.
// Coarse-fine:
//   Fine ghost from coarse — piecewise-constant (0th-order): both NG ghost
//     layers read from the single coarse cell nearest the C/F interface.
//     Transverse: fine index mapped to parent-octant half of coarse interior.
//   Coarse ghost from fine — 2×2 cell average over fine interior cells covering
//     each coarse ghost slot (conservative; mirrors fill_coarse_ghost_from_fine).
//     Includes A05-fix5 stale-pointer guard via BNCellBlock::h comparison.

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

            if (ni >= 0 && tree.nodes[ni].is_leaf()) {
                const auto& nnd = tree.nodes[ni];

                // ── Same-level neighbour: direct copy ────────────────────────
                if (nnd.level == nd.level) {
                    auto it = lm_map.find(((uint64_t)nnd.level << 32) | nnd.morton);
                    if (it == lm_map.end()) continue;
                    const BNCellBlock& src = blocks[it->second];

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

                // ── C/F: fine block (nd) adjacent to coarser neighbor (nnd) ─
                if (nnd.level < nd.level) {
                    auto it = lm_map.find(((uint64_t)nnd.level << 32) | nnd.morton);
                    if (it == lm_map.end()) continue;
                    const BNCellBlock& csrc = blocks[it->second];

                    // Octant of fine block relative to its parent.
                    int oct = (nd.parent >= 0)
                              ? (leaves[ii] - tree.nodes[nd.parent].first_child)
                              : 0;
                    int ix = oct_ix(oct), iy = oct_iy(oct), iz = oct_iz(oct);

                    // Map fine transverse index → coarse interior half.
                    auto local = [](int a) noexcept -> int {
                        return (a >= NG && a < NG+NB) ? (a-NG) : (a < NG ? 0 : NB-1);
                    };

                    // Piecewise-constant: both NG layers read the same coarse cell.
                    const int s_norm = sp.hi ? ilo() : ihi();

                    for (int g = 0; g < NG; ++g) {
                        int g_coord = sp.hi ? (NB+NG+g) : (NG-1-g);
                        if (sp.axis == 0) {
                            for (int k = 0; k < NB2; ++k)
                            for (int j = 0; j < NB2; ++j) {
                                int cj = NG + iy*(NB/2) + local(j)/2;
                                int ck = NG + iz*(NB/2) + local(k)/2;
                                int df = cell_idx(g_coord, j, k);
                                int sf = cell_idx(s_norm, cj, ck);
                                for (int v = 0; v < NVAR_BN; ++v)
                                    blk.Q[v][df] = csrc.Q[v][sf];
                            }
                        } else if (sp.axis == 1) {
                            for (int k = 0; k < NB2; ++k)
                            for (int i = 0; i < NB2; ++i) {
                                int ci = NG + ix*(NB/2) + local(i)/2;
                                int ck = NG + iz*(NB/2) + local(k)/2;
                                int df = cell_idx(i, g_coord, k);
                                int sf = cell_idx(ci, s_norm, ck);
                                for (int v = 0; v < NVAR_BN; ++v)
                                    blk.Q[v][df] = csrc.Q[v][sf];
                            }
                        } else {
                            for (int j = 0; j < NB2; ++j)
                            for (int i = 0; i < NB2; ++i) {
                                int ci = NG + ix*(NB/2) + local(i)/2;
                                int cj = NG + iy*(NB/2) + local(j)/2;
                                int df = cell_idx(i, j, g_coord);
                                int sf = cell_idx(ci, cj, s_norm);
                                for (int v = 0; v < NVAR_BN; ++v)
                                    blk.Q[v][df] = csrc.Q[v][sf];
                            }
                        }
                    }
                    continue;
                }

                // ── C/F: coarse block (nd) adjacent to finer neighbor (nnd) ─
                // nnd.level > nd.level: nnd is ONE of the 4 fine blocks on this face.
                {
                    int fine_parent = tree.nodes[ni].parent;
                    if (fine_parent < 0) continue;
                    int first_child = tree.nodes[fine_parent].first_child;
                    if (first_child < 0) continue;

                    // A05-fix5 analog: guard stale ni via h comparison.
                    {
                        const auto& fc_nd = tree.nodes[first_child];
                        if (!fc_nd.is_leaf()) continue;
                        auto fc_it = lm_map.find(
                            ((uint64_t)fc_nd.level << 32) | fc_nd.morton);
                        if (fc_it == lm_map.end()) continue;
                        const double h_exp = blk.h * 0.5;
                        if (std::fabs(blocks[fc_it->second].h - h_exp) >
                                1e-12 * blk.h)
                            continue;
                    }

                    const int half = NB / 2;
                    const int axis = sp.axis;
                    const int side = sp.hi;

                    for (int gl = 0; gl < NG; ++gl) {
                        const int g_coord = side ? (NB+NG+gl) : (NG-1-gl);
                        const int face_i  = side ? (ilo()+gl)  : (ihi()-gl);

                        for (int a = ilo(); a <= ihi(); ++a)
                        for (int b = ilo(); b <= ihi(); ++b) {
                            int a_local = a - ilo();
                            int b_local = b - ilo();
                            int ia_blk  = a_local / half;
                            int ib_blk  = b_local / half;

                            int oix, oiy, oiz;
                            if (axis == 0)      { oix=side; oiy=ia_blk; oiz=ib_blk; }
                            else if (axis == 1) { oix=ia_blk; oiy=side; oiz=ib_blk; }
                            else                { oix=ia_blk; oiy=ib_blk; oiz=side; }
                            int fi_node = first_child + oct_from_xyz(oix, oiy, oiz);

                            if (fi_node < 0 || fi_node >= (int)tree.nodes.size())
                                continue;
                            const auto& fnd = tree.nodes[fi_node];
                            if (!fnd.is_leaf()) continue;
                            auto fit = lm_map.find(
                                ((uint64_t)fnd.level << 32) | fnd.morton);
                            if (fit == lm_map.end()) continue;
                            const BNCellBlock& fsrc = blocks[fit->second];

                            int fa_start = NG + 2*(a_local % half);
                            int fb_start = NG + 2*(b_local % half);

                            int gi, gj, gk;
                            if (axis == 0)      { gi=g_coord; gj=a; gk=b; }
                            else if (axis == 1) { gi=a; gj=g_coord; gk=b; }
                            else                { gi=a; gj=b; gk=g_coord; }

                            for (int v = 0; v < NVAR_BN; ++v) {
                                double avg = 0.0;
                                for (int da = 0; da < 2; ++da)
                                for (int db = 0; db < 2; ++db) {
                                    int fa = fa_start + da;
                                    int fb = fb_start + db;
                                    int ci, cj, ck;
                                    if (axis == 0) {
                                        ci=face_i; cj=fa; ck=fb;
                                    } else if (axis == 1) {
                                        ci=fa; cj=face_i; ck=fb;
                                    } else {
                                        ci=fa; cj=fb; ck=face_i;
                                    }
                                    avg += fsrc.Q[v][cell_idx(ci,cj,ck)];
                                }
                                blk.Q[v][cell_idx(gi,gj,gk)] = avg * 0.25;
                            }
                        }
                    }
                    continue;
                }
            }

            // ── Domain boundary (ni == -1) ───────────────────────────────────
            if (ni >= 0) continue;  // is_leaf() returned false — skip (shouldn't happen in 2:1 tree)

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
