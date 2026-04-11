// vtk_writer.cpp — VTK legacy format output
// FIX B1: node.h → node.block->h (BlockNode::h was removed)
// FIX B1: ORIGIN now uses actual block->ox/oy/oz instead of 0,0,0
#include "../include/vtk_writer.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cmath>

void vtk_write(const NSSolver& s, const std::string& prefix) {
    auto leaves = s.tree.leaf_indices();
    int blk_id  = 0;
    for (int li : leaves) {
        auto& node = s.tree.nodes[li];
        auto& blk  = *node.block;
        double h   = node.block->h;   // FIX B1: was node.h (field removed)

        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s_step%06d_blk%03d.vtk",
                      prefix.c_str(), s.step, blk_id++);

        FILE* f = std::fopen(fname, "w");
        if (!f) throw std::runtime_error(std::string("vtk_write: cannot open ") + fname);

        std::fprintf(f, "# vtk DataFile Version 3.0\n");
        std::fprintf(f, "CFD step %d block %d\n", s.step, blk_id - 1);
        std::fprintf(f, "ASCII\n");
        std::fprintf(f, "DATASET STRUCTURED_POINTS\n");
        std::fprintf(f, "DIMENSIONS %d %d %d\n", NB, NB, NB);
        // FIX B1: use actual block origin; interior starts at ox + NG*h
        std::fprintf(f, "ORIGIN %.10e %.10e %.10e\n",
                     blk.ox + NG * h,
                     blk.oy + NG * h,
                     blk.oz + NG * h);
        std::fprintf(f, "SPACING %.10e %.10e %.10e\n", h, h, h);
        std::fprintf(f, "POINT_DATA %d\n", NB * NB * NB);

        auto write_scalar = [&](const char* name, auto getter) {
            std::fprintf(f, "SCALARS %s double 1\n", name);
            std::fprintf(f, "LOOKUP_TABLE default\n");
            for (int k = NG; k < NG + NB; ++k)
            for (int j = NG; j < NG + NB; ++j)
            for (int i = NG; i < NG + NB; ++i)
                std::fprintf(f, "%.10e\n", getter(i, j, k));
        };

        write_scalar("rho", [&](int i,int j,int k){ return blk.prim(i,j,k).rho; });
        write_scalar("u",   [&](int i,int j,int k){ return blk.prim(i,j,k).u;   });
        write_scalar("v",   [&](int i,int j,int k){ return blk.prim(i,j,k).v;   });
        write_scalar("w",   [&](int i,int j,int k){ return blk.prim(i,j,k).w;   });
        write_scalar("p",   [&](int i,int j,int k){ return blk.prim(i,j,k).p;   });
        write_scalar("T",   [&](int i,int j,int k){ return blk.prim(i,j,k).T;   });

        std::fclose(f);
    }
}
