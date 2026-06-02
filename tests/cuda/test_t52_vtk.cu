// Gate t52 — VTK XML binary writer
//
// Verifies vtk_write_vts and vtk_write_pvts:
//   1. Allocate a TGV-like conserved-variable field on device.
//   2. Call vtk_write_vts with step=1, blk_id=0.
//   3. Check the .vts file exists and starts with "<?xml".
//   4. Call vtk_write_pvts and check the .pvts file exists.
//
// Gate: PASS  V1  VTK file written and valid XML

#include "io/vtk_writer.hpp"
#include "cuda/gpu_constants.cuh"
#include "cuda/gpu_check.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <sys/stat.h>

static constexpr double PI = 3.14159265358979323846;
static constexpr const char* PREFIX = "/tmp/test_t52";

// Fill a TGV-like conserved state on host, return as flat SoA:
//   rho = 1, u = sin(x)cos(y)cos(z), v = -cos(x)sin(y)cos(z), w = 0
//   p   = 1/(GPU_GAMMA) + rho/16 * (cos(2x)+cos(2y))*(cos(2z)+2)
// Cell centres: x = (NG+i+0.5)*h, etc.
static std::vector<double> make_tgv_Q(double ox, double oy, double oz, double h)
{
    std::vector<double> Q((size_t)GPU_NVAR * GPU_NCELL, 0.0);
    for (int k = 0; k < GPU_NB2; ++k)
    for (int j = 0; j < GPU_NB2; ++j)
    for (int i = 0; i < GPU_NB2; ++i) {
        const double x = ox + (i + 0.5) * h;
        const double y = oy + (j + 0.5) * h;
        const double z = oz + (k + 0.5) * h;
        const int flat = i + GPU_NB2 * (j + GPU_NB2 * k);

        const double rho = 1.0;
        const double u   =  sin(x) * cos(y) * cos(z);
        const double v   = -cos(x) * sin(y) * cos(z);
        const double w   =  0.0;
        const double p   = 1.0 / GPU_GAMMA
                         + rho / 16.0 * (cos(2*x) + cos(2*y)) * (cos(2*z) + 2.0);
        const double E   = p / (GPU_GAMMA - 1.0)
                         + 0.5 * rho * (u*u + v*v + w*w);

        Q[0 * GPU_NCELL + flat] = rho;
        Q[1 * GPU_NCELL + flat] = rho * u;
        Q[2 * GPU_NCELL + flat] = rho * v;
        Q[3 * GPU_NCELL + flat] = rho * w;
        Q[4 * GPU_NCELL + flat] = E;
    }
    return Q;
}

int main()
{
    int n_fail = 0;

    // ── V1: single-block .vts + .pvts ──────────────────────────────────────
    {
        const double h  = (2.0 * PI) / GPU_NB;   // domain covers 2π per block
        const double ox = 0.0, oy = 0.0, oz = 0.0;
        const int step = 1, blk_id = 0;

        // Build TGV IC on host, upload to device
        auto h_Q = make_tgv_Q(ox, oy, oz, h);
        double* d_Q = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Q, (size_t)GPU_NVAR * GPU_NCELL * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(),
                              (size_t)GPU_NVAR * GPU_NCELL * sizeof(double),
                              cudaMemcpyHostToDevice));

        // Write .vts
        vtk_write_vts(PREFIX, step, blk_id, ox, oy, oz, h, d_Q);

        // Write .pvts collection (1 block)
        vtk_write_pvts(PREFIX, step, 1);

        CUDA_CHECK(cudaFree(d_Q));

        // Check .vts file exists
        char vts_path[256];
        std::snprintf(vts_path, sizeof(vts_path),
                      "%s_step%06d_blk%03d.vts", PREFIX, step, blk_id);
        struct stat st;
        if (stat(vts_path, &st) != 0) {
            std::fprintf(stderr, "FAIL V1: .vts file not found: %s\n", vts_path);
            ++n_fail;
        } else {
            // Check first 5 bytes = "<?xml"
            FILE* f = std::fopen(vts_path, "r");
            char buf[6] = {};
            if (f) { std::fread(buf, 1, 5, f); std::fclose(f); }
            if (std::strncmp(buf, "<?xml", 5) != 0) {
                std::fprintf(stderr, "FAIL V1: .vts does not start with '<?xml', got: %.5s\n", buf);
                ++n_fail;
            } else {
                std::printf("PASS  V1  VTK file written and valid XML\n");
            }
        }

        // Check .pvts file exists
        char pvts_path[256];
        std::snprintf(pvts_path, sizeof(pvts_path),
                      "%s_step%06d.pvts", PREFIX, step);
        if (stat(pvts_path, &st) != 0) {
            std::fprintf(stderr, "FAIL V1: .pvts file not found: %s\n", pvts_path);
            ++n_fail;
        }
    }

    std::printf("=== Result: %d failure(s) ===\n", n_fail);
    return (n_fail == 0) ? 0 : 1;
}
