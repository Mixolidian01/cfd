// rigid_body.cpp — FSI-1: 6-DOF rigid body ODE integrator (explicit Euler).
#include "fsi/rigid_body.hpp"
#include <cmath>

// Quaternion multiply: out = a ⊗ b  (Hamilton product).
// Convention: q = (w, x, y, z).
static void qmul(const double a[4], const double b[4], double out[4]) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

// Build rotation matrix from quaternion (row-major, 3×3).
static void quat_to_R(const double q[4], double R[9]) {
    double w=q[0], x=q[1], y=q[2], z=q[3];
    R[0] = 1 - 2*(y*y + z*z);  R[1] = 2*(x*y - w*z);    R[2] = 2*(x*z + w*y);
    R[3] = 2*(x*y + w*z);      R[4] = 1 - 2*(x*x + z*z); R[5] = 2*(y*z - w*x);
    R[6] = 2*(x*z - w*y);      R[7] = 2*(y*z + w*x);     R[8] = 1 - 2*(x*x + y*y);
}

void RigidBody6DOF::step(const double F[6], double dt) {
    // ── Translational ────────────────────────────────────────────────────────
    for (int i = 0; i < 3; ++i) v[i] += (F[i] / mass) * dt;
    for (int i = 0; i < 3; ++i) x[i] += v[i] * dt;

    // ── Rotational ───────────────────────────────────────────────────────────
    // 1. Build current rotation matrix R from quaternion.
    double R[9];
    quat_to_R(q, R);

    // 2. World torque → body torque:  T_body = R^T * T_world.
    const double* T = F + 3;
    double T_body[3];
    for (int i = 0; i < 3; ++i) {
        T_body[i] = R[0*3+i]*T[0] + R[1*3+i]*T[1] + R[2*3+i]*T[2];
    }

    // 3. Angular acceleration in body frame: alpha_body = I^{-1} T_body (diagonal I).
    double alpha_body[3];
    for (int i = 0; i < 3; ++i) alpha_body[i] = T_body[i] / I[i];

    // 4. alpha back to world frame: alpha_world = R * alpha_body.
    double alpha_world[3];
    for (int i = 0; i < 3; ++i) {
        alpha_world[i] = R[i*3+0]*alpha_body[0] + R[i*3+1]*alpha_body[1] + R[i*3+2]*alpha_body[2];
    }

    // 5. Update world angular velocity.
    for (int i = 0; i < 3; ++i) w[i] += alpha_world[i] * dt;

    // 6. Quaternion integration: dq/dt = 0.5 * q ⊗ [0, w_body].
    //    Use current w (world) converted to body for the update, or equivalently
    //    use the pure-quaternion path: omega_quat = (0, w[0], w[1], w[2]) in world,
    //    and update q via dq = 0.5 * dt * omega_quat ⊗ q  (left-multiply convention
    //    for world-frame omega).
    //    Standard body-frame formulation: dq/dt = 0.5 * q ⊗ omega_b_quat.
    //    Equivalent world-frame: dq/dt = 0.5 * omega_w_quat ⊗ q.
    //    We use world frame here (w is world angular velocity).
    double omega_quat[4] = {0.0, w[0], w[1], w[2]};
    double dq[4];
    qmul(omega_quat, q, dq);  // dq/dt = 0.5 * omega_w ⊗ q
    for (int i = 0; i < 4; ++i) q[i] += 0.5 * dt * dq[i];

    // 7. Normalize quaternion to prevent drift.
    double norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > 1e-15) { for (int i = 0; i < 4; ++i) q[i] /= norm; }
}

void RigidBody6DOF::wall_velocity(const double p[3], double v_out[3]) const {
    // v_out = v_cm + w × (p - x_cm)
    double r[3] = {p[0]-x[0], p[1]-x[1], p[2]-x[2]};
    v_out[0] = v[0] + w[1]*r[2] - w[2]*r[1];
    v_out[1] = v[1] + w[2]*r[0] - w[0]*r[2];
    v_out[2] = v[2] + w[0]*r[1] - w[1]*r[0];
}

void RigidBody6DOF::transform(double R[9], double t[3]) const {
    quat_to_R(q, R);
    t[0] = x[0]; t[1] = x[1]; t[2] = x[2];
}
