#pragma once
// rigid_body.hpp — FSI-1: host-side 6-DOF rigid body ODE integrator.
//
// Uses explicit Euler for both translational and rotational DOF.
// Quaternion convention: q = (w, x, y, z) with w the scalar part.
// Angular velocity is stored in world frame.

struct RigidBody6DOF {
    double mass   = 1.0;
    double I[3]   = {1.0, 1.0, 1.0};  // principal moments of inertia (body frame, diagonal)
    double x[3]   = {};                // position (world)
    double q[4]   = {1.0, 0.0, 0.0, 0.0}; // quaternion (w, x, y, z)
    double v[3]   = {};                // translational velocity (world)
    double w[3]   = {};                // angular velocity (world frame)

    // Advance state by dt given wrench F[6] = {Fx, Fy, Fz, Tx, Ty, Tz} in world frame.
    // Translational: v += (F[0..2]/mass)*dt,  x += v*dt.
    // Rotational: convert world torque → body frame, compute angular acceleration,
    //             convert back → world, update w and quaternion.
    void step(const double F[6], double dt);

    // Return velocity at world point p[3]: v_out = v_cm + w × (p - x).
    void wall_velocity(const double p[3], double v_out[3]) const;

    // Return current rigid transform: R[9] (row-major rotation matrix) and t[3] (translation).
    // R is derived from the current quaternion.
    void transform(double R[9], double t[3]) const;
};
