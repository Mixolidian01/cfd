"""
Partitioned FSI coupling with Aitken Delta^2 relaxation.

Couples a Python-side EulerBernoulliBeam to a CFD solver exposed via the
pybind11 bindings (cfd module).  The coupling assumes the solver provides:
    solver.advance(dt)
    solver.read_wrench() -> [Fx, Fy, Fz, Tx, Ty, Tz]
If read_wrench is not exposed by the bindings, the load integration helper
remains usable for unit testing the beam in isolation.

Reference: Kuettler & Wall (2008), "Fixed-point fluid-structure interaction
solvers with dynamic relaxation."
"""
import numpy as np

from scripts.fsi.beam_fem import EulerBernoulliBeam  # noqa: F401  (re-export)


def integrate_loads_1d(wrench_y, beam):
    """
    Map a single resultant lift force (scalar) to nodal forces on the beam.

    Assumes uniform pressure distribution: equal share at each interior node,
    no nodal moments.  Clamped root (node 0) gets zero.
    """
    N = beam.N
    F_ext = np.zeros(2 * N)
    n_interior = N - 1
    if n_interior <= 0:
        return F_ext
    f_per_node = float(wrench_y) / n_interior
    for i in range(1, N):
        F_ext[2 * i] = f_per_node
    return F_ext


def fsi_advance(solver, beam, dt, tol=1e-4, max_iter=10):
    """
    One coupled FSI time step with Aitken Delta^2 relaxation.

    Parameters
    ----------
    solver   : object with .advance(dt) and .read_wrench()
    beam     : EulerBernoulliBeam
    dt       : time step
    tol      : displacement residual tolerance
    max_iter : maximum Aitken sub-iterations

    Returns
    -------
    int : number of Aitken sub-iterations performed
    """
    x_pred = beam.u.copy()
    omega = 0.1
    r_prev = None
    it = 0

    for it in range(max_iter):
        solver.advance(dt)
        wrench = solver.read_wrench()  # [Fx, Fy, Fz, Tx, Ty, Tz]
        F_ext = integrate_loads_1d(wrench[1], beam)
        beam.step(F_ext, dt)
        x_new = beam.u.copy()
        r = x_new - x_pred

        if np.linalg.norm(r) < tol:
            break

        if r_prev is not None:
            dr = r - r_prev
            denom = float(np.dot(dr, dr)) + 1e-300
            omega = -omega * float(np.dot(r_prev, dr)) / denom
            omega = float(np.clip(omega, 0.1, 1.0))

        x_pred = x_pred + omega * r
        r_prev = r.copy()

    beam.u = x_pred
    return it + 1
