#include "models/primitive_gen.hpp"
#include <array>
#include <cmath>
#include <vector>

static constexpr float PI = 3.14159265358979323846f;

static Triangle make_tri(std::array<float,3> a, std::array<float,3> b,
                         std::array<float,3> c, std::array<float,3> n) {
    Triangle t; t.v0=a; t.v1=b; t.v2=c; t.normal=n; return t;
}

static std::array<float,3> cross3(std::array<float,3> u, std::array<float,3> v) {
    return {u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]};
}
static std::array<float,3> norm3(std::array<float,3> v) {
    float l = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if (l < 1e-12f) return {0,0,1};
    return {v[0]/l, v[1]/l, v[2]/l};
}
static std::array<float,3> sub3(std::array<float,3> a, std::array<float,3> b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

TriangleMesh make_sphere(float cx, float cy, float cz, float r, int nlon, int nlat) {
    std::vector<std::array<float,3>> V;
    V.push_back({cx, cy, cz+r}); // north pole = 0
    for (int j = 1; j < nlat; ++j) {
        float phi = PI * j / nlat;
        for (int i = 0; i < nlon; ++i) {
            float theta = 2*PI * i / nlon;
            V.push_back({cx + r*std::sin(phi)*std::cos(theta),
                         cy + r*std::sin(phi)*std::sin(theta),
                         cz + r*std::cos(phi)});
        }
    }
    V.push_back({cx, cy, cz-r}); // south pole = last

    auto idx = [&](int j, int i) { return 1 + (j-1)*nlon + (i % nlon); };

    TriangleMesh m;
    auto push = [&](int a, int b, int c) {
        auto n = norm3(cross3(sub3(V[b],V[a]), sub3(V[c],V[a])));
        m.triangles.push_back(make_tri(V[a], V[b], V[c], n));
    };

    for (int i = 0; i < nlon; ++i)
        push(0, idx(1,i), idx(1,i+1));

    for (int j = 1; j < nlat-1; ++j)
        for (int i = 0; i < nlon; ++i) {
            push(idx(j,i), idx(j+1,i+1), idx(j,i+1));
            push(idx(j,i), idx(j+1,i), idx(j+1,i+1));
        }

    int sp = (int)V.size()-1;
    for (int i = 0; i < nlon; ++i)
        push(sp, idx(nlat-1,i+1), idx(nlat-1,i));

    return m;
}

TriangleMesh make_box(float x0, float y0, float z0, float x1, float y1, float z1) {
    TriangleMesh m;
    using A3 = std::array<float,3>;
    struct Face { A3 a,b,c,d,n; };
    Face faces[6] = {
        {{x0,y0,z0},{x1,y0,z0},{x1,y1,z0},{x0,y1,z0},{0,0,-1}},
        {{x0,y0,z1},{x0,y1,z1},{x1,y1,z1},{x1,y0,z1},{0,0, 1}},
        {{x0,y0,z0},{x0,y1,z0},{x0,y1,z1},{x0,y0,z1},{-1,0,0}},
        {{x1,y0,z0},{x1,y0,z1},{x1,y1,z1},{x1,y1,z0},{ 1,0,0}},
        {{x0,y0,z0},{x0,y0,z1},{x1,y0,z1},{x1,y0,z0},{0,-1,0}},
        {{x0,y1,z0},{x1,y1,z0},{x1,y1,z1},{x0,y1,z1},{0, 1,0}},
    };
    for (auto& f : faces) {
        m.triangles.push_back(make_tri(f.a, f.b, f.c, f.n));
        m.triangles.push_back(make_tri(f.a, f.c, f.d, f.n));
    }
    return m;
}

TriangleMesh make_cylinder(float cx, float cy, float cz_lo, float cz_hi, float r, int nseg) {
    TriangleMesh m;
    using A3 = std::array<float,3>;

    auto ring = [&](float z, int i) -> A3 {
        float theta = 2*PI * i / nseg;
        return {cx + r*std::cos(theta), cy + r*std::sin(theta), z};
    };

    for (int i = 0; i < nseg; ++i) {
        A3 b0 = ring(cz_lo, i), b1 = ring(cz_lo, i+1);
        A3 t0 = ring(cz_hi, i), t1 = ring(cz_hi, i+1);
        auto sn = norm3(cross3(sub3(b1,b0), sub3(t0,b0)));
        m.triangles.push_back(make_tri(b0,t1,b1,sn));
        m.triangles.push_back(make_tri(b0,t0,t1,sn));
        A3 tc = {cx,cy,cz_hi};
        auto tn = std::array<float,3>{0.f, 0.f, 1.f};
        m.triangles.push_back(make_tri(t0,tc,t1,tn));
        A3 bc = {cx,cy,cz_lo};
        auto bn = std::array<float,3>{0.f, 0.f, -1.f};
        m.triangles.push_back(make_tri(b0,b1,bc,bn));
    }
    return m;
}
