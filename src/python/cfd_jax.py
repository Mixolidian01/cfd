"""
cfd_jax.py — D11-adj: JAX differentiable wrapper around NSSolver.advance().

Exposes:
    make_advance_step(solver) → advance_step
        advance_step(q_flat: jax.Array) → jax.Array
        q_flat shape: (n_leaves * NVAR * NCELL,) float64
        VJP is provided via @jax.custom_vjp, calling solver.adjoint_step().

Usage:
    import cfd, cfd_jax, jax, jax.numpy as jnp

    solver = cfd.NSSolver()
    solver.init(1.0, ic)

    advance_step = cfd_jax.make_advance_step(solver)
    q0 = jnp.array(np.concatenate([a.ravel() for a in solver.get_block_arrays()]))

    # Differentiable forward:
    q1 = advance_step(q0)

    # Gradient:
    loss = lambda q: advance_step(q).sum()
    grad = jax.grad(loss)(q0)
"""

import numpy as np
import jax
import jax.numpy as jnp
import cfd


def make_advance_step(solver: cfd.NSSolver):
    """Return a JAX-differentiable function wrapping solver.advance().

    The returned function takes and returns a flat float64 array of shape
    (n_leaves * NVAR * NCELL,).  The VJP (used by jax.grad) calls
    solver.adjoint_step() using checkpoints written during the forward pass.
    """
    n_leaves = len(solver.get_block_arrays())
    flat_size = n_leaves * cfd.NVAR * cfd.NCELL
    result_shape = jax.ShapeDtypeStruct((flat_size,), jnp.float64)

    def _forward_np(q_flat_np: np.ndarray) -> np.ndarray:
        arrs = [
            q_flat_np[i * cfd.NVAR * cfd.NCELL:(i + 1) * cfd.NVAR * cfd.NCELL
                      ].reshape(cfd.NVAR, cfd.NCELL)
            for i in range(n_leaves)
        ]
        solver.set_block_arrays(arrs)
        solver.advance()
        out = solver.get_block_arrays()
        return np.concatenate([a.ravel() for a in out]).astype(np.float64)

    def _adjoint_np(lam_f_np: np.ndarray) -> np.ndarray:
        lam_f = [
            lam_f_np[i * cfd.NVAR * cfd.NCELL:(i + 1) * cfd.NVAR * cfd.NCELL
                     ].reshape(cfd.NVAR, cfd.NCELL)
            for i in range(n_leaves)
        ]
        lam_n = solver.adjoint_step(lam_f)
        return np.concatenate([a.ravel() for a in lam_n]).astype(np.float64)

    @jax.custom_vjp
    def advance_step(q_flat: jax.Array) -> jax.Array:
        return jax.pure_callback(_forward_np, result_shape, q_flat)

    def advance_step_fwd(q_flat: jax.Array):
        result = advance_step(q_flat)
        return result, ()  # no residuals: checkpoints live in C++ solver

    def advance_step_bwd(residuals, g: jax.Array):
        lam_n = jax.pure_callback(_adjoint_np, result_shape, g)
        return (lam_n,)

    advance_step.defvjp(advance_step_fwd, advance_step_bwd)
    return advance_step
