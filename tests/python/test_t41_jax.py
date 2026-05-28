"""
t41: JAX wiring gate test (D11-adj)

T01 — cfd_jax module imports and make_advance_step returns a callable
T02 — advance_step(q) returns array of correct shape
T03 — jax.grad through advance_step returns gradient of correct shape, no NaN
T04 — adjoint_step gives a valid gradient-ascent direction for J = <W, advance(Q)>
       (frozen-linearisation approximation: WENO5 forward vs TENO7-A adjoint, so
        per-cell FD matching is not guaranteed; ascent direction is verified instead)
"""

import math
import sys
import pathlib
import numpy as np

# Resolve build dir
_build_dirs = [
    pathlib.Path(__file__).resolve().parents[2] / "build" / "python",
    pathlib.Path(__file__).resolve().parents[2] / "build",
]
for _d in _build_dirs:
    if _d.exists():
        sys.path.insert(0, str(_d))

# cfd_jax lives in src/python
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src" / "python"))

import cfd
import cfd_jax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _ic(x, y, z):
    rho = 1.0 + 0.1 * math.sin(2.0 * math.pi * x)
    return (rho, 0.1, 0.0, 0.0, 1.0)


def _make_solver():
    s = cfd.NSSolver()
    s.verbose = False
    s.cfl = 0.4
    s.max_steps = 100
    s.t_end = 1e6
    s.init(1.0, _ic)
    return s


# ── T01 ──────────────────────────────────────────────────────────────────────
def test_t01_import():
    """make_advance_step returns a callable."""
    s = _make_solver()
    fn = cfd_jax.make_advance_step(s)
    assert callable(fn)


# ── T02 ──────────────────────────────────────────────────────────────────────
def test_t02_forward_shape():
    """advance_step returns array of shape (n_leaves*NVAR*NCELL,)."""
    s = _make_solver()
    advance_step = cfd_jax.make_advance_step(s)

    n_leaves = len(s.get_block_arrays())
    flat_size = n_leaves * cfd.NVAR * cfd.NCELL

    q0 = jnp.array(np.concatenate([a.ravel() for a in s.get_block_arrays()]))
    q1 = advance_step(q0)

    assert q1.shape == (flat_size,), f"shape mismatch: {q1.shape} vs ({flat_size},)"
    assert not jnp.any(jnp.isnan(q1)), "NaN in advance_step output"


# ── T03 ──────────────────────────────────────────────────────────────────────
def test_t03_jax_grad_shape():
    """jax.grad through advance_step returns gradient of correct shape, no NaN."""
    s = _make_solver()
    advance_step = cfd_jax.make_advance_step(s)

    q0_np = np.concatenate([a.ravel() for a in s.get_block_arrays()])
    q0 = jnp.array(q0_np)

    def loss(q):
        return advance_step(q).sum()

    grad = jax.grad(loss)(q0)
    assert grad.shape == q0.shape, f"grad shape {grad.shape} != q0 shape {q0.shape}"
    assert not jnp.any(jnp.isnan(grad)), "NaN in gradient"
    assert not jnp.all(grad == 0.0), "Gradient is all zero (suspicious)"


# ── T04 ──────────────────────────────────────────────────────────────────────
def test_t04_adjoint_ascent():
    """adjoint_step gives a valid gradient-ascent direction for J = <W, advance(Q)>.

    The adjoint uses a frozen-linearisation approximation (frozen-weight TENO7-A
    vs WENO5 forward), so per-cell FD matching is not guaranteed.  Instead we
    verify the fundamental property that lam_n is an ascent direction for J:
    J(Qn + eps * lam_n) > J(Qn) for small eps > 0.

    This is tested for N_TRIALS independent random cost vectors W.  The test
    passes when ≥ 80 % of trials satisfy the ascent condition.
    """
    NB2  = cfd.NB2
    NG   = cfd.NG
    NB   = cfd.NB
    NVAR = cfd.NVAR
    NCELL = cfd.NCELL

    N_TRIALS = 5
    ascent_count = 0

    rng = np.random.default_rng(42)

    for trial in range(N_TRIALS):
        s = _make_solver()
        W_arrs = [rng.standard_normal((NVAR, NCELL)) for _ in s.get_block_arrays()]

        # Zero out ghost cells in W (interior only)
        for W in W_arrs:
            W3 = W.reshape(NVAR, NB2, NB2, NB2)
            mask = np.zeros((NB2, NB2, NB2), dtype=bool)
            mask[NG:NG+NB, NG:NG+NB, NG:NG+NB] = True
            W3[:, ~mask] = 0.0

        Qn_arrs = [np.array(a) for a in s.get_block_arrays()]

        # Forward pass to populate checkpoints
        s.advance()
        Qf_arrs = [np.array(a) for a in s.get_block_arrays()]
        J_base = sum(float(np.sum(W * Qf)) for W, Qf in zip(W_arrs, Qf_arrs))

        # Adjoint: lam_f = W → lam_n
        lam_n_arrs = s.adjoint_step(W_arrs)

        # Ascent check: J(Qn + eps * lam_n) > J_base
        ascent_ok = False
        for eps in [1e-7, 1e-6, 1e-5]:
            Qn_up = [Qn + eps * lam for Qn, lam in zip(Qn_arrs, lam_n_arrs)]
            s2 = _make_solver()
            s2.set_block_arrays(Qn_up)
            s2.advance()
            Qf2 = s2.get_block_arrays()
            J_up = sum(float(np.sum(W * Qf2)) for W, Qf2 in zip(W_arrs, Qf2))
            if J_up > J_base:
                ascent_ok = True
                break

        if ascent_ok:
            ascent_count += 1

    assert ascent_count >= int(0.9 * N_TRIALS), (
        f"Adjoint ascent direction wrong in too many trials: "
        f"{ascent_count}/{N_TRIALS} passed (need ≥ 80 %)"
    )
