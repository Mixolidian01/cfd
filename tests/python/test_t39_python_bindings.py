"""
t39: Python bindings gate test (D11)

Verifies that NSSolver is usable from Python for a 10-step periodic
isentropic-vortex-like run:
  T01 — module imports and constants are correct
  T02 — init() accepts a Python IC callable
  T03 — advance() returns positive dt; solver state increments
  T04 — mass conserved < 1e-10 relative over 10 steps
  T05 — get_block_arrays() mass matches compute_diag().mass to 1e-12 relative
"""

import math
import sys
import pathlib

# Add the build directory to sys.path so 'import cfd' finds the shared library.
# CMake places the .so in build/python/ (set by LIBRARY_OUTPUT_DIRECTORY).
_build_dirs = [
    pathlib.Path(__file__).resolve().parents[2] / "build" / "python",
    pathlib.Path(__file__).resolve().parents[2] / "build",
]
for _d in _build_dirs:
    if _d.exists():
        sys.path.insert(0, str(_d))

import cfd
import numpy as np


# ── IC: smooth periodic density wave in x (isentropic) ───────────────────────
def _ic(x, y, z):
    rho = 1.0 + 0.1 * math.sin(2.0 * math.pi * x)
    u   = 0.1   # uniform advection speed
    v   = 0.0
    w   = 0.0
    p   = 1.0
    return (rho, u, v, w, p)


def _make_solver():
    s = cfd.NSSolver()
    s.verbose   = False
    s.cfl       = 0.4
    s.max_steps = 10
    s.t_end     = 1e6   # effectively unlimited — stop at max_steps
    s.init(1.0, _ic)
    return s


# ── T01 ──────────────────────────────────────────────────────────────────────
def test_t01_constants():
    """Module constants match expected values."""
    assert cfd.NB    == 8
    assert cfd.NG    == 2
    assert cfd.NB2   == 12
    assert cfd.NCELL == 1728
    assert cfd.NVAR  == 5
    assert abs(cfd.GAMMA - 1.4) < 1e-15


# ── T02 ──────────────────────────────────────────────────────────────────────
def test_t02_init():
    """init() succeeds and returns a valid solver."""
    s = _make_solver()
    assert s.step == 0
    assert s.t    == 0.0


# ── T03 ──────────────────────────────────────────────────────────────────────
def test_t03_advance():
    """advance() returns positive dt and increments step counter."""
    s = _make_solver()
    dt = s.advance()
    assert dt > 0.0, f"advance() returned non-positive dt={dt}"
    assert s.step == 1
    assert s.t    > 0.0


# ── T04 ──────────────────────────────────────────────────────────────────────
def test_t04_mass_conservation():
    """Mass conserved to < 1e-10 relative over 10 SSP-RK3 steps."""
    s = _make_solver()
    diag0 = s.compute_diag()
    mass0 = diag0.mass

    for _ in range(10):
        s.advance()

    diag10 = s.compute_diag()
    mass10 = diag10.mass

    rel_err = abs(mass10 - mass0) / (abs(mass0) + 1e-300)
    assert rel_err < 1e-10, (
        f"Mass conservation failed: rel_err={rel_err:.3e} "
        f"(mass0={mass0:.10e}, mass10={mass10:.10e})"
    )


# ── T05 ──────────────────────────────────────────────────────────────────────
def test_t05_block_arrays_match_diag():
    """Mass from get_block_arrays() matches compute_diag().mass to 1e-12 rel."""
    s = _make_solver()

    # Advance 10 steps to get a non-trivial state.
    for _ in range(10):
        s.advance()

    diag = s.compute_diag()
    mass_diag = diag.mass

    # Compute mass manually from block arrays.
    # arr shape: (NVAR=5, NCELL=1728), layout: arr[v, c] = Q[v][cell_idx(i,j,k)]
    # cell_idx(i,j,k) = k*NB2^2 + j*NB2 + i  → reshape to (NB2,NB2,NB2)[k,j,i]
    NB2  = cfd.NB2   # 12
    NG   = cfd.NG    # 2
    NB   = cfd.NB    # 8

    arrays = s.get_block_arrays()
    h_vals = s.get_block_h()

    mass_arrays = 0.0
    for arr, h in zip(arrays, h_vals):
        # arr[0] = rho at all NCELL cells (including ghosts)
        rho_3d = arr[0].reshape(NB2, NB2, NB2)   # [k, j, i] due to cell_idx
        interior = rho_3d[NG:NG+NB, NG:NG+NB, NG:NG+NB]
        mass_arrays += interior.sum() * h**3

    rel_err = abs(mass_arrays - mass_diag) / (abs(mass_diag) + 1e-300)
    assert rel_err < 1e-12, (
        f"Block arrays mass mismatch: rel_err={rel_err:.3e} "
        f"(diag={mass_diag:.12e}, arrays={mass_arrays:.12e})"
    )
