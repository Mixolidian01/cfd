// tools/make_stl.cpp — generate binary STL files for IBM primitive objects
//
// Build:  cmake --build build -t make_stl
// Usage:  make_stl <shape> <output.stl> [shape-specific args...]
//
//   make_stl sphere  out.stl  cx cy cz  R  [stacks=16] [slices=32]
//   make_stl box     out.stl  x0 y0 z0  x1 y1 z1
//   make_stl cylinder out.stl cx cy z0 z1 R  [slices=32]
//   make_stl cone    out.stl  cx cy z0 z1 R_base [slices=32]
//   make_stl naca    out.stl  cx_le cy z0 z1 chord [panels=64] [t_over_c=0.12]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <string>

// =============================================================================
// Binary STL writer
// =============================================================================

struct Tri { float n[3], v[3][3]; };

static void write_stl(const char* path, const char* label,
                      const std::vector<Tri>& tris)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "make_stl: cannot open '%s'\n", path); exit(1); }

    char hdr[80] = {};
    snprintf(hdr, sizeof(hdr), "%s", label);
    fwrite(hdr, 1, 80, f);

    uint32_t n = (uint32_t)tris.size();
    fwrite(&n, 4, 1, f);

    for (const auto& t : tris) {
        fwrite(t.n,    4, 3, f);
        fwrite(t.v[0], 4, 3, f);
        fwrite(t.v[1], 4, 3, f);
        fwrite(t.v[2], 4, 3, f);
        uint16_t attr = 0;
        fwrite(&attr, 2, 1, f);
    }
    fclose(f);
}

// Compute outward normal from vertex winding (CCW = outward) and push tri.
static void push_tri(std::vector<Tri>& out,
                     float ax, float ay, float az,
                     float bx, float by, float bz,
                     float cx, float cy, float cz)
{
    float e1x = bx-ax, e1y = by-ay, e1z = bz-az;
    float e2x = cx-ax, e2y = cy-ay, e2z = cz-az;
    float nx = e1y*e2z - e1z*e2y;
    float ny = e1z*e2x - e1x*e2z;
    float nz = e1x*e2y - e1y*e2x;
    float ln = sqrtf(nx*nx + ny*ny + nz*nz);
    if (ln > 1e-20f) { nx /= ln; ny /= ln; nz /= ln; }
    Tri t;
    t.n[0]=nx; t.n[1]=ny; t.n[2]=nz;
    t.v[0][0]=ax; t.v[0][1]=ay; t.v[0][2]=az;
    t.v[1][0]=bx; t.v[1][1]=by; t.v[1][2]=bz;
    t.v[2][0]=cx; t.v[2][1]=cy; t.v[2][2]=cz;
    out.push_back(t);
}

// =============================================================================
// Primitives
// =============================================================================

static std::vector<Tri> make_sphere(float cx, float cy, float cz, float R,
                                    int stacks, int slices)
{
    std::vector<Tri> out;
    // UV-sphere: latitude bands × longitude slices.
    // Each quad becomes 2 triangles; outward normal = vertex - centre / R.
    for (int la = 0; la < stacks; ++la) {
        float t0 = (float)M_PI * la       / stacks - (float)M_PI / 2.0f;
        float t1 = (float)M_PI * (la + 1) / stacks - (float)M_PI / 2.0f;
        for (int lo = 0; lo < slices; ++lo) {
            float p0 = 2.0f * (float)M_PI * lo       / slices;
            float p1 = 2.0f * (float)M_PI * (lo + 1) / slices;
            auto pt = [&](float t, float p) -> std::array<float,3> {
                return { cx + R*cosf(t)*cosf(p),
                         cy + R*cosf(t)*sinf(p),
                         cz + R*sinf(t) };
            };
            auto a = pt(t0,p0), b = pt(t0,p1), c = pt(t1,p0), d = pt(t1,p1);
            // Outward CCW when viewed from outside.
            push_tri(out, a[0],a[1],a[2], b[0],b[1],b[2], c[0],c[1],c[2]);
            push_tri(out, b[0],b[1],b[2], d[0],d[1],d[2], c[0],c[1],c[2]);
        }
    }
    return out;
}

static std::vector<Tri> make_box(float x0, float y0, float z0,
                                 float x1, float y1, float z1)
{
    std::vector<Tri> out;
    // 6 faces × 2 triangles = 12 tris.  Winding gives outward normals.
    // -x face
    push_tri(out, x0,y0,z0, x0,y0,z1, x0,y1,z1);
    push_tri(out, x0,y0,z0, x0,y1,z1, x0,y1,z0);
    // +x face
    push_tri(out, x1,y0,z0, x1,y1,z0, x1,y1,z1);
    push_tri(out, x1,y0,z0, x1,y1,z1, x1,y0,z1);
    // -y face
    push_tri(out, x0,y0,z0, x1,y0,z0, x1,y0,z1);
    push_tri(out, x0,y0,z0, x1,y0,z1, x0,y0,z1);
    // +y face
    push_tri(out, x0,y1,z0, x0,y1,z1, x1,y1,z1);
    push_tri(out, x0,y1,z0, x1,y1,z1, x1,y1,z0);
    // -z face
    push_tri(out, x0,y0,z0, x0,y1,z0, x1,y1,z0);
    push_tri(out, x0,y0,z0, x1,y1,z0, x1,y0,z0);
    // +z face
    push_tri(out, x0,y0,z1, x1,y0,z1, x1,y1,z1);
    push_tri(out, x0,y0,z1, x1,y1,z1, x0,y1,z1);
    return out;
}

// Generalised cylinder/cone: radius varies linearly from r0 at z0 to r1 at z1.
// Set r1=0 for a cone; r0==r1 for a cylinder.
static std::vector<Tri> make_frustum(float cx, float cy,
                                     float z0, float z1,
                                     float r0, float r1,
                                     int slices)
{
    std::vector<Tri> out;
    // Side wall triangles.
    for (int i = 0; i < slices; ++i) {
        float a0 = 2.0f * (float)M_PI * i       / slices;
        float a1 = 2.0f * (float)M_PI * (i + 1) / slices;
        float c0 = cosf(a0), s0 = sinf(a0);
        float c1 = cosf(a1), s1 = sinf(a1);
        float x00 = cx + r0*c0, y00 = cy + r0*s0;
        float x01 = cx + r0*c1, y01 = cy + r0*s1;
        float x10 = cx + r1*c0, y10 = cy + r1*s0;
        float x11 = cx + r1*c1, y11 = cy + r1*s1;
        // CCW outward (outward = away from axis)
        push_tri(out, x00,y00,z0, x11,y11,z1, x01,y01,z0);
        push_tri(out, x00,y00,z0, x10,y10,z1, x11,y11,z1);
    }
    // Bottom cap at z0 (outward = -z).
    if (r0 > 0.0f) {
        for (int i = 0; i < slices; ++i) {
            float a0 = 2.0f * (float)M_PI * i       / slices;
            float a1 = 2.0f * (float)M_PI * (i + 1) / slices;
            // CCW when viewed from -z  → reverse winding
            push_tri(out, cx,cy,z0,
                     cx + r0*cosf(a1), cy + r0*sinf(a1), z0,
                     cx + r0*cosf(a0), cy + r0*sinf(a0), z0);
        }
    }
    // Top cap at z1 (outward = +z).
    if (r1 > 0.0f) {
        for (int i = 0; i < slices; ++i) {
            float a0 = 2.0f * (float)M_PI * i       / slices;
            float a1 = 2.0f * (float)M_PI * (i + 1) / slices;
            push_tri(out, cx,cy,z1,
                     cx + r1*cosf(a0), cy + r1*sinf(a0), z1,
                     cx + r1*cosf(a1), cy + r1*sinf(a1), z1);
        }
    }
    return out;
}

static std::vector<Tri> make_naca_sym(float cx_le, float cy,
                                      float z0, float z1,
                                      float chord, int n_panels,
                                      float t_over_c)
{
    auto thk = [t_over_c](float xn) -> float {
        float s = sqrtf(std::max(xn, 0.0f));
        return 5.0f * t_over_c * (
             0.2969f * s
           - 0.1260f * xn
           - 0.3516f * xn * xn
           + 0.2843f * xn * xn * xn
           - 0.1015f * xn * xn * xn * xn);
    };

    std::vector<std::array<float,2>> upper(n_panels+1), lower(n_panels+1);
    for (int i = 0; i <= n_panels; ++i) {
        float beta = (float)M_PI * (float)i / n_panels;
        float xn   = 0.5f * (1.0f - cosf(beta));
        float yt   = thk(xn) * chord;
        upper[i] = { cx_le + xn * chord, cy + yt };
        lower[i] = { cx_le + xn * chord, cy - yt };
    }

    std::vector<Tri> out;
    for (int i = 0; i < n_panels; ++i) {
        float ux0=upper[i][0], uy0=upper[i][1];
        float ux1=upper[i+1][0], uy1=upper[i+1][1];
        float lx0=lower[i][0], ly0=lower[i][1];
        float lx1=lower[i+1][0], ly1=lower[i+1][1];

        // Upper skin (outward = +y)
        push_tri(out, ux0,uy0,z0, ux1,uy1,z1, ux1,uy1,z0);
        push_tri(out, ux0,uy0,z0, ux0,uy0,z1, ux1,uy1,z1);
        // Lower skin (outward = -y)
        push_tri(out, lx0,ly0,z0, lx1,ly1,z0, lx1,ly1,z1);
        push_tri(out, lx0,ly0,z0, lx1,ly1,z1, lx0,ly0,z1);
        // Endcap -z (outward = -z)
        push_tri(out, ux0,uy0,z0, ux1,uy1,z0, lx0,ly0,z0);
        push_tri(out, ux1,uy1,z0, lx1,ly1,z0, lx0,ly0,z0);
        // Endcap +z (outward = +z)
        push_tri(out, ux0,uy0,z1, lx0,ly0,z1, ux1,uy1,z1);
        push_tri(out, ux1,uy1,z1, lx0,ly0,z1, lx1,ly1,z1);
    }
    return out;
}

// =============================================================================
// main
// =============================================================================

static void usage()
{
    fprintf(stderr,
        "Usage:\n"
        "  make_stl sphere   out.stl cx cy cz R [stacks=16] [slices=32]\n"
        "  make_stl box      out.stl x0 y0 z0 x1 y1 z1\n"
        "  make_stl cylinder out.stl cx cy z0 z1 R [slices=32]\n"
        "  make_stl cone     out.stl cx cy z0 z1 R_base [slices=32]\n"
        "  make_stl naca     out.stl cx_le cy z0 z1 chord [panels=64] [t_over_c=0.12]\n");
    exit(1);
}

static float arg_f(char** argv, int idx, const char* name)
{
    (void)name;
    return (float)atof(argv[idx]);
}

int main(int argc, char* argv[])
{
    if (argc < 3) usage();
    std::string shape  = argv[1];
    const char* output = argv[2];

    std::vector<Tri> tris;
    char label[80] = {};

    if (shape == "sphere") {
        if (argc < 7) usage();
        float cx = arg_f(argv,3,"cx"), cy = arg_f(argv,4,"cy"),
              cz = arg_f(argv,5,"cz"), R  = arg_f(argv,6,"R");
        int stacks = (argc > 7) ? atoi(argv[7]) : 16;
        int slices = (argc > 8) ? atoi(argv[8]) : 32;
        snprintf(label, sizeof(label), "sphere cx=%.4g cy=%.4g cz=%.4g R=%.4g",
                 (double)cx,(double)cy,(double)cz,(double)R);
        tris = make_sphere(cx, cy, cz, R, stacks, slices);

    } else if (shape == "box") {
        if (argc < 9) usage();
        float x0=arg_f(argv,3,"x0"), y0=arg_f(argv,4,"y0"), z0=arg_f(argv,5,"z0");
        float x1=arg_f(argv,6,"x1"), y1=arg_f(argv,7,"y1"), z1=arg_f(argv,8,"z1");
        snprintf(label, sizeof(label),
                 "box [%.4g,%.4g]x[%.4g,%.4g]x[%.4g,%.4g]",
                 (double)x0,(double)x1,(double)y0,(double)y1,(double)z0,(double)z1);
        tris = make_box(x0, y0, z0, x1, y1, z1);

    } else if (shape == "cylinder") {
        if (argc < 8) usage();
        float cx=arg_f(argv,3,"cx"), cy=arg_f(argv,4,"cy");
        float z0=arg_f(argv,5,"z0"), z1=arg_f(argv,6,"z1");
        float R =arg_f(argv,7,"R");
        int slices = (argc > 8) ? atoi(argv[8]) : 32;
        snprintf(label, sizeof(label),
                 "cylinder cx=%.4g cy=%.4g z=[%.4g,%.4g] R=%.4g",
                 (double)cx,(double)cy,(double)z0,(double)z1,(double)R);
        tris = make_frustum(cx, cy, z0, z1, R, R, slices);

    } else if (shape == "cone") {
        if (argc < 8) usage();
        float cx=arg_f(argv,3,"cx"), cy=arg_f(argv,4,"cy");
        float z0=arg_f(argv,5,"z0"), z1=arg_f(argv,6,"z1");
        float R =arg_f(argv,7,"R");
        int slices = (argc > 8) ? atoi(argv[8]) : 32;
        snprintf(label, sizeof(label),
                 "cone cx=%.4g cy=%.4g z=[%.4g,%.4g] R_base=%.4g",
                 (double)cx,(double)cy,(double)z0,(double)z1,(double)R);
        tris = make_frustum(cx, cy, z0, z1, R, 0.0f, slices);

    } else if (shape == "naca") {
        if (argc < 8) usage();
        float cx_le=arg_f(argv,3,"cx_le"), cy=arg_f(argv,4,"cy");
        float z0   =arg_f(argv,5,"z0"),    z1=arg_f(argv,6,"z1");
        float chord=arg_f(argv,7,"chord");
        int   panels  = (argc > 8) ? atoi(argv[8])       : 64;
        float t_over_c= (argc > 9) ? arg_f(argv,9,"toc") : 0.12f;
        snprintf(label, sizeof(label),
                 "NACA_sym chord=%.4g toc=%.4g panels=%d",
                 (double)chord,(double)t_over_c,panels);
        tris = make_naca_sym(cx_le, cy, z0, z1, chord, panels, t_over_c);

    } else {
        fprintf(stderr, "make_stl: unknown shape '%s'\n", shape.c_str());
        usage();
    }

    write_stl(output, label, tris);
    printf("make_stl: wrote %zu triangles to '%s'  [%s]\n",
           tris.size(), output, label);
    return 0;
}
