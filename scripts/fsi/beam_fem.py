"""
Euler-Bernoulli corotational beam FEM for FSI structural solver.

1D beam along the span; DOF per node: [w, theta] (transverse displacement,
rotation).  Clamped at x=0 (root), free at x=L (tip).  Newmark-beta time
integration (beta=1/4, gamma=1/2 => unconditionally stable, 2nd order).

Reference: Bathe & Bolourchi (1979); standard 4-DOF Euler-Bernoulli element.
"""
import numpy as np


class EulerBernoulliBeam:
    """1D Euler-Bernoulli clamped-free beam."""

    def __init__(self, L, N, E, I, rho_s, A):
        """
        Parameters
        ----------
        L     : span [m]
        N     : number of nodes (>=2)
        E     : Young's modulus [Pa]
        I     : second moment of area [m^4]
        rho_s : structural density [kg/m^3]
        A     : cross-section area [m^2]
        """
        assert N >= 2
        self.L = float(L)
        self.N = int(N)
        self.E = float(E)
        self.I = float(I)
        self.rho_s = float(rho_s)
        self.A = float(A)

        ndof = 2 * N
        self.ndof = ndof
        self.K = np.zeros((ndof, ndof))
        self.M = np.zeros((ndof, ndof))

        ell = L / (N - 1)  # uniform element length
        self.ell = ell

        # Element stiffness (4x4) for w1,theta1,w2,theta2
        k_e = (E * I / ell**3) * np.array([
            [12,     6 * ell,   -12,     6 * ell],
            [6 * ell, 4 * ell**2, -6 * ell, 2 * ell**2],
            [-12,   -6 * ell,    12,    -6 * ell],
            [6 * ell, 2 * ell**2, -6 * ell, 4 * ell**2],
        ], dtype=float)

        # Consistent mass matrix
        m_e = (rho_s * A * ell / 420.0) * np.array([
            [156,     22 * ell,    54,     -13 * ell],
            [22 * ell,  4 * ell**2,  13 * ell, -3 * ell**2],
            [54,      13 * ell,   156,    -22 * ell],
            [-13 * ell, -3 * ell**2, -22 * ell,  4 * ell**2],
        ], dtype=float)

        # Assemble global K, M
        for e in range(N - 1):
            idx = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
            for i in range(4):
                for j in range(4):
                    self.K[idx[i], idx[j]] += k_e[i, j]
                    self.M[idx[i], idx[j]] += m_e[i, j]

        # State vectors (full ndof; clamped DOFs are held at zero in step())
        self.u = np.zeros(ndof)
        self.u_dot = np.zeros(ndof)
        self.u_ddot = np.zeros(ndof)

        # Newmark coefficients
        self.beta = 0.25
        self.gamma = 0.5

        # Free DOF indices (skip clamped node 0)
        self.free = np.arange(2, ndof)

    def _apply_clamped_bc(self):
        self.u[0] = 0.0
        self.u[1] = 0.0
        self.u_dot[0] = 0.0
        self.u_dot[1] = 0.0
        self.u_ddot[0] = 0.0
        self.u_ddot[1] = 0.0

    def step(self, F_ext, dt):
        """
        Newmark-beta time step.

        F_ext : (2N,) nodal load vector [F_0, M_0, F_1, M_1, ...]
        dt    : time step
        """
        beta, gamma = self.beta, self.gamma
        u_old = self.u.copy()
        v_old = self.u_dot.copy()
        a_old = self.u_ddot.copy()

        # Predictor (no acceleration update yet)
        u_pred = u_old + dt * v_old + (0.5 - beta) * dt * dt * a_old
        v_pred = v_old + (1.0 - gamma) * dt * a_old

        # Effective stiffness: K_eff = K + (1/(beta*dt^2)) M
        K_eff = self.K + (1.0 / (beta * dt * dt)) * self.M

        # Effective load: F_eff = F_ext + M * ( (1/(beta dt^2)) u_old
        #                                     + (1/(beta dt))  v_old
        #                                     + (1/(2 beta) - 1) a_old )
        rhs_inertia = (
            (1.0 / (beta * dt * dt)) * u_old
            + (1.0 / (beta * dt)) * v_old
            + (1.0 / (2.0 * beta) - 1.0) * a_old
        )
        F_eff = F_ext + self.M @ rhs_inertia

        # Apply clamped BC: solve only on free DOFs
        f = self.free
        K_ff = K_eff[np.ix_(f, f)]
        F_f = F_eff[f]
        u_new_f = np.linalg.solve(K_ff, F_f)

        u_new = np.zeros_like(u_old)
        u_new[f] = u_new_f
        # Clamped DOFs remain zero

        # Recover acceleration and velocity from Newmark relations
        a_new = (u_new - u_pred) / (beta * dt * dt)
        v_new = v_pred + gamma * dt * a_new

        self.u = u_new
        self.u_dot = v_new
        self.u_ddot = a_new
        self._apply_clamped_bc()
        return self.u

    def tip_displacement(self):
        """Transverse displacement w at the tip (last) node."""
        return float(self.u[-2])

    def reset(self):
        self.u[:] = 0.0
        self.u_dot[:] = 0.0
        self.u_ddot[:] = 0.0
