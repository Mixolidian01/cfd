// vtk_writer.cu — VTK XML binary writer (GPU device pointer path).
// Compiled with nvcc. The legacy ASCII vtk_write() lives in vtk_writer.cpp
// (CPU-only, part of ns_solver). This TU only provides vtk_write_vts and
// vtk_write_pvts which take a raw device pointer and require CUDA memcpy.
#include "io/vtk_writer.hpp"
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "cuda/gpu_constants.cuh"

// ─── VTK XML binary (StructuredGrid, appended raw) ──────────────────────────
// Interior cells per block (no ghosts)
static constexpr int VTK_N3 = GPU_NB * GPU_NB * GPU_NB;   // 512

void vtk_write_vts(const std::string& path_prefix,
                   int step, int blk_id,
                   double ox, double oy, double oz, double h,
                   const double* d_Q)
{
    // Build output filename
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s_step%06d_blk%03d.vts",
                  path_prefix.c_str(), step, blk_id);

    // Download conserved variables from device
    std::vector<double> Q((size_t)GPU_NVAR * GPU_NCELL);
    cudaMemcpy(Q.data(), d_Q, (size_t)GPU_NVAR * GPU_NCELL * sizeof(double),
               cudaMemcpyDeviceToHost);

    // ── Build appended data arrays ────────────────────────────────────────
    // Block 0: Points (3 × VTK_N3 doubles)
    // Blocks 1-5: rho, u, v, w, p (VTK_N3 doubles each)

    const uint64_t bytes_pts  = (uint64_t)3 * VTK_N3 * sizeof(double);
    const uint64_t bytes_scal = (uint64_t)    VTK_N3 * sizeof(double);

    std::vector<double> pts(3 * VTK_N3);
    std::vector<double> a_rho(VTK_N3), a_u(VTK_N3), a_v(VTK_N3),
                        a_w(VTK_N3),   a_p(VTK_N3);

    // VTK convention: i (x) varies fastest, k (z) slowest
    int idx = 0;
    for (int k = 0; k < GPU_NB; ++k)
    for (int j = 0; j < GPU_NB; ++j)
    for (int i = 0; i < GPU_NB; ++i, ++idx) {
        pts[idx*3+0] = ox + (GPU_NG + i + 0.5) * h;
        pts[idx*3+1] = oy + (GPU_NG + j + 0.5) * h;
        pts[idx*3+2] = oz + (GPU_NG + k + 0.5) * h;

        const int flat = (GPU_NG+i) + GPU_NB2 * ((GPU_NG+j) + GPU_NB2*(GPU_NG+k));
        const double rho  = Q[0*GPU_NCELL + flat];
        const double rhou = Q[1*GPU_NCELL + flat];
        const double rhov = Q[2*GPU_NCELL + flat];
        const double rhow = Q[3*GPU_NCELL + flat];
        const double E    = Q[4*GPU_NCELL + flat];
        const double inv  = (rho > 0.0) ? 1.0/rho : 0.0;
        const double u    = rhou * inv;
        const double v    = rhov * inv;
        const double w    = rhow * inv;
        const double p    = (GPU_GAMMA - 1.0) * (E - 0.5*rho*(u*u + v*v + w*w));

        a_rho[idx] = rho;
        a_u[idx]   = u;
        a_v[idx]   = v;
        a_w[idx]   = w;
        a_p[idx]   = p;
    }

    // ── Compute byte offsets for appended section ─────────────────────────
    const uint64_t stride_pts  = sizeof(uint64_t) + bytes_pts;
    const uint64_t stride_scal = sizeof(uint64_t) + bytes_scal;
    const uint64_t off0 = 0;
    const uint64_t off1 = off0 + stride_pts;
    const uint64_t off2 = off1 + stride_scal;
    const uint64_t off3 = off2 + stride_scal;
    const uint64_t off4 = off3 + stride_scal;
    const uint64_t off5 = off4 + stride_scal;

    // ── Write file ────────────────────────────────────────────────────────
    FILE* f = std::fopen(fname, "wb");
    if (!f) throw std::runtime_error(std::string("vtk_write_vts: cannot open ") + fname);

    // XML header (text written as raw bytes into binary file)
    char hdr[4096];
    int hlen = std::snprintf(hdr, sizeof(hdr),
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"StructuredGrid\" version=\"0.1\""
        " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
        "  <StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n"
        "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
        "      <Points>\n"
        "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
        " format=\"appended\" offset=\"%llu\"/>\n"
        "      </Points>\n"
        "      <PointData>\n"
        "        <DataArray type=\"Float64\" Name=\"rho\""
        " format=\"appended\" offset=\"%llu\"/>\n"
        "        <DataArray type=\"Float64\" Name=\"u\""
        " format=\"appended\" offset=\"%llu\"/>\n"
        "        <DataArray type=\"Float64\" Name=\"v\""
        " format=\"appended\" offset=\"%llu\"/>\n"
        "        <DataArray type=\"Float64\" Name=\"w\""
        " format=\"appended\" offset=\"%llu\"/>\n"
        "        <DataArray type=\"Float64\" Name=\"p\""
        " format=\"appended\" offset=\"%llu\"/>\n"
        "      </PointData>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "  <AppendedData encoding=\"raw\">\n"
        "    _",
        GPU_NB-1, GPU_NB-1, GPU_NB-1,
        GPU_NB-1, GPU_NB-1, GPU_NB-1,
        (unsigned long long)off0,
        (unsigned long long)off1,
        (unsigned long long)off2,
        (unsigned long long)off3,
        (unsigned long long)off4,
        (unsigned long long)off5
    );
    if (hlen < 0 || hlen >= (int)sizeof(hdr))
        throw std::runtime_error("vtk_write_vts: header buffer overflow");

    std::fwrite(hdr, 1, (size_t)hlen, f);

    // Appended raw data blocks
    auto write_block = [&](uint64_t nbytes, const void* data) {
        std::fwrite(&nbytes, sizeof(uint64_t), 1, f);
        std::fwrite(data, 1, (size_t)nbytes, f);
    };

    write_block(bytes_pts,  pts.data());
    write_block(bytes_scal, a_rho.data());
    write_block(bytes_scal, a_u.data());
    write_block(bytes_scal, a_v.data());
    write_block(bytes_scal, a_w.data());
    write_block(bytes_scal, a_p.data());

    std::fprintf(f, "\n  </AppendedData>\n</VTKFile>\n");
    std::fclose(f);
}

void vtk_write_pvts(const std::string& path_prefix, int step, int n_blocks)
{
    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s_step%06d.pvts",
                  path_prefix.c_str(), step);

    FILE* f = std::fopen(fname, "w");
    if (!f) throw std::runtime_error(std::string("vtk_write_pvts: cannot open ") + fname);

    std::fprintf(f,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"PStructuredGrid\" version=\"0.1\""
        " byte_order=\"LittleEndian\">\n"
        "  <PStructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\" GhostLevel=\"0\">\n"
        "    <PPoints>\n"
        "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n"
        "    </PPoints>\n"
        "    <PPointData>\n"
        "      <PDataArray type=\"Float64\" Name=\"rho\"/>\n"
        "      <PDataArray type=\"Float64\" Name=\"u\"/>\n"
        "      <PDataArray type=\"Float64\" Name=\"v\"/>\n"
        "      <PDataArray type=\"Float64\" Name=\"w\"/>\n"
        "      <PDataArray type=\"Float64\" Name=\"p\"/>\n"
        "    </PPointData>\n",
        GPU_NB-1, GPU_NB-1, GPU_NB-1);

    // Use base name only for relative Source references
    const size_t slash = path_prefix.rfind('/');
    const std::string base = (slash == std::string::npos)
                           ? path_prefix
                           : path_prefix.substr(slash + 1);

    for (int b = 0; b < n_blocks; ++b) {
        char src[256];
        std::snprintf(src, sizeof(src), "%s_step%06d_blk%03d.vts",
                      base.c_str(), step, b);
        std::fprintf(f,
            "    <Piece Extent=\"0 %d 0 %d 0 %d\" Source=\"%s\"/>\n",
            GPU_NB-1, GPU_NB-1, GPU_NB-1, src);
    }

    std::fprintf(f,
        "  </PStructuredGrid>\n"
        "</VTKFile>\n");
    std::fclose(f);
}
