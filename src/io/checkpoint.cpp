// checkpoint.cpp — binary checkpoint save/load
//
// FIX B7: checkpoint_load() previously called tree.init() (1 leaf) and then
//         immediately checked for a leaf-count match — impossible for any
//         refined mesh (NL > 1). Tree topology was never saved or replayed.
//
//         New save: appends a topology section — (morton, level) per leaf.
//         New load: replays tree.refine() calls to reconstruct exact topology,
//                   then copies Q field data into the rebuilt leaves.
//
// FIX P6: save writes to "<path>.tmp", fflush + fsync, then renames atomically.
//         A crash during write never leaves a corrupt checkpoint visible.

#include "io/checkpoint.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>   // fsync, fileno

static constexpr uint64_t MAGIC   = 0xCFD7CF07ULL;
static constexpr uint32_t VERSION = 2;   // bumped: topology section added

// =============================================================================
// checkpoint_save
// =============================================================================
void checkpoint_save(const NSSolver& s, const std::string& path) {
    // FIX P6: write to temp file, rename atomically on success
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("checkpoint_save: cannot open " + tmp);

    auto write_pod = [&](const void* p, size_t n) {
        if (std::fwrite(p, 1, n, f) != n)
            throw std::runtime_error("checkpoint_save: write error");
    };

    // ── Header ────────────────────────────────────────────────────────────────
    write_pod(&MAGIC,   sizeof(MAGIC));
    write_pod(&VERSION, sizeof(VERSION));

    int32_t step32 = (int32_t)s.step;
    write_pod(&step32, sizeof(step32));
    write_pod(&s.t,    sizeof(s.t));

    auto leaves = s.tree.leaf_indices();
    int32_t NL  = (int32_t)leaves.size();
    write_pod(&NL, sizeof(NL));

    // ── Field data ────────────────────────────────────────────────────────────
    for (int li : leaves) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double  h  = blk.h;
        int32_t lv = node.level;
        write_pod(&h,  sizeof(h));
        write_pod(&lv, sizeof(lv));
        // P4.2: Q[v] is AoSoA internally; serialise to flat SoA for portability.
        double tmp[NCELL];
        for (int v = 0; v < NVAR; ++v) {
            blk.Q[v].copy_to_flat(tmp);
            write_pod(tmp, NCELL * sizeof(double));
        }
    }

    // ── Topology section (FIX B7) ─────────────────────────────────────────────
    for (int li : leaves) {
        uint32_t code = s.tree.nodes[li].morton;
        uint32_t lv   = (uint32_t)s.tree.nodes[li].level;
        write_pod(&code, sizeof(code));
        write_pod(&lv,   sizeof(lv));
    }

    // FIX P6: flush → fsync → atomic rename
    if (std::fflush(f) != 0)
        throw std::runtime_error("checkpoint_save: fflush failed");
    if (fsync(fileno(f)) != 0)
        throw std::runtime_error("checkpoint_save: fsync failed");
    std::fclose(f);

    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("checkpoint_save: rename failed");
}

// =============================================================================
// checkpoint_load
// =============================================================================
void checkpoint_load(NSSolver& s, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("checkpoint_load: cannot open " + path);

    auto read_pod = [&](void* p, size_t n) {
        if (std::fread(p, 1, n, f) != n)
            throw std::runtime_error("checkpoint_load: read error or truncated file");
    };

    // ── Header ────────────────────────────────────────────────────────────────
    uint64_t magic = 0;
    read_pod(&magic, sizeof(magic));
    if (magic != MAGIC)
        throw std::runtime_error("checkpoint_load: bad magic number");

    uint32_t ver = 0;
    read_pod(&ver, sizeof(ver));
    if (ver != VERSION)
        throw std::runtime_error("checkpoint_load: unsupported version "
                                 + std::to_string(ver)
                                 + " (expected " + std::to_string(VERSION) + ")");

    int32_t step32 = 0;
    read_pod(&step32, sizeof(step32));
    s.step = step32;
    read_pod(&s.t, sizeof(s.t));

    int32_t NL = 0;
    read_pod(&NL, sizeof(NL));

    // ── Buffer field data (tree not rebuilt yet) ──────────────────────────────
    struct LeafData {
        double  h;
        int32_t level;
        std::vector<std::vector<double>> Q;
    };
    std::vector<LeafData> field_data((size_t)NL);
    for (int ii = 0; ii < NL; ++ii) {
        auto& ld = field_data[ii];
        read_pod(&ld.h,     sizeof(ld.h));
        read_pod(&ld.level, sizeof(ld.level));
        ld.Q.resize(NVAR);
        for (int v = 0; v < NVAR; ++v) {
            ld.Q[v].resize((size_t)NB2 * NB2 * NB2);
            read_pod(ld.Q[v].data(), NB2 * NB2 * NB2 * sizeof(double));
        }
    }

    // ── Read topology (FIX B7) ────────────────────────────────────────────────
    std::vector<uint32_t> topo_code((size_t)NL), topo_level((size_t)NL);
    for (int ii = 0; ii < NL; ++ii) {
        read_pod(&topo_code[ii],  sizeof(uint32_t));
        read_pod(&topo_level[ii], sizeof(uint32_t));
    }
    std::fclose(f);

    // ── Replay refine() to reconstruct tree topology (FIX B7) ────────────────
    // Reset to a single root leaf, then replay refinements level by level
    // in Morton order so parents are always created before their children.
    s.tree.init(s.tree.domain_L());

    // Sort by (level ASC, morton ASC) — guarantees parents before children
    std::vector<int> order((size_t)NL);
    for (int i = 0; i < NL; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (topo_level[a] != topo_level[b]) return topo_level[a] < topo_level[b];
        return topo_code[a] < topo_code[b];
    });

    for (int ii : order) {
        uint32_t target_code  = topo_code[ii];
        uint32_t target_level = topo_level[ii];
        if (target_level == 0) continue;   // root always exists

        // Walk from root, refining any leaf on the path to the target
        int cur = 0;
        for (uint32_t lv = 0; lv < target_level; ++lv) {
            if (s.tree.nodes[cur].is_leaf())
                s.tree.refine(cur);
            // Morton code encodes octant at depth lv in bits [3*(L-1-lv) .. +2]
            int oct = (int)((target_code >> (3u * (target_level - 1u - lv))) & 7u);
            cur = s.tree.nodes[cur].first_child + oct;
        }
        // cur is now the target leaf — exists, nothing more to do
    }

    s.tree.rebuild_neighbours();

    // ── Write field data into rebuilt leaf blocks ─────────────────────────────
    auto final_leaves = s.tree.leaf_indices();
    if ((int32_t)final_leaves.size() != NL)
        throw std::runtime_error(
            "checkpoint_load: topology replay gave "
            + std::to_string(final_leaves.size())
            + " leaves but checkpoint has " + std::to_string(NL));

    for (int ii = 0; ii < NL; ++ii) {
        auto& blk = *s.tree.nodes[final_leaves[ii]].block;
        for (int v = 0; v < NVAR; ++v)
            blk.Q[v].assign_from_flat(field_data[ii].Q[v].data());
    }

    // Rebuild RK scratch arrays for the restored mesh
    s.alloc_scratch();
}
