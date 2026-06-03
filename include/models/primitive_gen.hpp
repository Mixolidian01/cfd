#pragma once
#include "models/mesh_loader.hpp"

// cx,cy,cz = centre; r = radius; nlon = longitude divisions; nlat = latitude divisions
TriangleMesh make_sphere(float cx, float cy, float cz, float r, int nlon=16, int nlat=8);

// Axis-aligned box from corner (x0,y0,z0) to (x1,y1,z1)
TriangleMesh make_box(float x0, float y0, float z0, float x1, float y1, float z1);

// Cylinder centred at (cx,cy), from z=cz_lo to z=cz_hi, radius r, nseg segments
TriangleMesh make_cylinder(float cx, float cy, float cz_lo, float cz_hi, float r, int nseg=16);
