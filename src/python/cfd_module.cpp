// =============================================================================
// cfd_module.cpp — D11: pybind11 bindings for NSSolver
// =============================================================================
// Exposes a thin Python wrapper around NSSolver that drives the C++ solver
// and reads/writes block arrays as NumPy views.  No physics logic in Python.
//
// Public API:
//   cfd.NSSolver()
//     .init(domain_L, ic_fn)       ic_fn(x,y,z) → (rho,u,v,w,p)
//     .advance() → float           one SSP-RK3 step, returns dt
//     .run()                       advance until t_end or max_steps
//     .compute_diag() → StepDiag
//     .get_block_arrays() → list[np.ndarray shape=(NVAR,NCELL)]
//     .set_block_arrays(list)
//     .get_block_h()    → list[float]  cell size per leaf block
//     .adjoint_step(lam_f) → list[np.ndarray]
//     .cfl, .t_end, .max_steps, .verbose  (read-write properties)
//     .t, .step                    (read-only)
//   cfd.StepDiag  (read-only fields: step, t, dt, mass, momentum_x,
//                                    kinetic_energy, total_energy)
//   cfd.NVAR, NCELL, NB, NG, NB2, GAMMA
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "solver/ns_solver.hpp"
#include "mesh/cell_block.hpp"
#include "mesh/block_tree.hpp"

#include <cmath>
#include <stdexcept>
#include <cstdio>

namespace py = pybind11;
using namespace pybind11::literals;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Convert a Python (rho, u, v, w, p) tuple to a Prim for the IC.
static Prim prim_from_tuple(const py::tuple& t) {
    if ((int)t.size() != 5)
        throw std::runtime_error("IC function must return a 5-tuple (rho, u, v, w, p)");
    Prim p{};
    p.rho     = t[0].cast<double>();
    p.u       = t[1].cast<double>();
    p.v       = t[2].cast<double>();
    p.w       = t[3].cast<double>();
    p.p       = t[4].cast<double>();
    p.gamma_m = GAMMA;
    p.p_inf_m = 0.0;
    p.T       = p.p / (p.rho * R_GAS);
    p.c       = std::sqrt(p.gamma_m * p.p / p.rho);
    return p;
}

// ── get_block_arrays ──────────────────────────────────────────────────────────
// Returns one (NVAR, NCELL) C-contiguous double array per leaf block.
// Layout: arr[v, c] = Q[v][c] (AoSoA → flat SoA).
static std::vector<py::array_t<double>>
get_block_arrays(const NSSolver& solver) {
    std::vector<py::array_t<double>> result;
    for (int li : solver.tree.leaf_indices()) {
        const auto& node = solver.tree.nodes[li];
        if (!node.has_block()) continue;
        const CellBlock& blk = *node.block;

        py::array_t<double> arr({NVAR, NCELL});
        auto buf = arr.mutable_unchecked<2>();
        for (int v = 0; v < NVAR; ++v) {
            double flat[NCELL];
            blk.Q[v].copy_to_flat(flat);
            for (int c = 0; c < NCELL; ++c)
                buf(v, c) = flat[c];
        }
        result.push_back(std::move(arr));
    }
    return result;
}

// ── set_block_arrays ──────────────────────────────────────────────────────────
// Writes a list of (NVAR, NCELL) arrays back to the leaf blocks.
static void set_block_arrays(
        NSSolver& solver,
        const std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>>& arrays) {
    const auto& leaves = solver.tree.leaf_indices();
    if ((int)arrays.size() != (int)leaves.size())
        throw std::runtime_error("set_block_arrays: length must equal n_leaves()");

    for (int k = 0; k < (int)leaves.size(); ++k) {
        auto& node = solver.tree.nodes[leaves[k]];
        if (!node.has_block()) continue;
        CellBlock& blk = *node.block;

        auto buf = arrays[k].unchecked<2>();
        for (int v = 0; v < NVAR; ++v) {
            double flat[NCELL];
            for (int c = 0; c < NCELL; ++c)
                flat[c] = buf(v, c);
            blk.Q[v].assign_from_flat(flat);
        }
    }
}

// ── get_block_h ──────────────────────────────────────────────────────────────
// Returns h (cell size) for each leaf block, in the same order as
// get_block_arrays().
static std::vector<double> get_block_h(const NSSolver& solver) {
    std::vector<double> result;
    for (int li : solver.tree.leaf_indices()) {
        const auto& node = solver.tree.nodes[li];
        if (!node.has_block()) continue;
        result.push_back(node.block->h);
    }
    return result;
}

// ── adjoint_step ─────────────────────────────────────────────────────────────
// Takes list[(NVAR,NCELL)] lam_f (one per leaf block), returns list[(NVAR,NCELL)] lam_n.
// Mirrors set_block_arrays input convention (index-based iteration over leaves).
static std::vector<py::array_t<double>>
py_adjoint_step(
    NSSolver& solver,
    const std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>>& lam_f_arrs)
{
    const auto& leaves = solver.tree.leaf_indices();
    if ((int)lam_f_arrs.size() != (int)leaves.size())
        throw std::runtime_error("adjoint_step: lam_f length must equal n_leaves()");

    std::vector<CellBlock> lam_f_blks;
    lam_f_blks.reserve(leaves.size());
    for (int k = 0; k < (int)leaves.size(); ++k) {
        const auto& node = solver.tree.nodes[leaves[k]];
        if (!node.has_block())
            throw std::runtime_error("adjoint_step: leaf has no block (regrid during advance?)");
        CellBlock blk(node.block->ox, node.block->oy, node.block->oz, node.block->h);
        auto buf = lam_f_arrs[k].unchecked<2>();
        for (int v = 0; v < NVAR; ++v) {
            double flat[NCELL];
            for (int c = 0; c < NCELL; ++c) flat[c] = buf(v, c);
            blk.Q[v].assign_from_flat(flat);
        }
        lam_f_blks.push_back(std::move(blk));
    }

    auto lam_n_blks = solver.adjoint_step(lam_f_blks);

    std::vector<py::array_t<double>> result;
    result.reserve(lam_n_blks.size());
    for (int k = 0; k < (int)lam_n_blks.size(); ++k) {
        py::array_t<double> arr({NVAR, NCELL});
        auto out = arr.mutable_unchecked<2>();
        for (int v = 0; v < NVAR; ++v) {
            double flat[NCELL];
            lam_n_blks[k].Q[v].copy_to_flat(flat);
            for (int c = 0; c < NCELL; ++c) out(v, c) = flat[c];
        }
        result.push_back(std::move(arr));
    }
    return result;
}

// =============================================================================
// Module definition
// =============================================================================
PYBIND11_MODULE(cfd, m) {
    m.doc() = "cfd — Python bindings for the compressible CFD solver (D11)";

    // ── StepDiag ─────────────────────────────────────────────────────────────
    py::class_<StepDiag>(m, "StepDiag")
        .def_readonly("step",           &StepDiag::step)
        .def_readonly("t",              &StepDiag::t)
        .def_readonly("dt",             &StepDiag::dt)
        .def_readonly("mass",           &StepDiag::mass)
        .def_readonly("momentum_x",     &StepDiag::momentum_x)
        .def_readonly("kinetic_energy", &StepDiag::kinetic_energy)
        .def_readonly("total_energy",   &StepDiag::total_energy)
        .def("__repr__", [](const StepDiag& d) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "StepDiag(step=%d, t=%.6e, mass=%.10e, ke=%.6e)",
                d.step, d.t, d.mass, d.kinetic_energy);
            return std::string(buf);
        });

    // ── NSSolver ─────────────────────────────────────────────────────────────
    py::class_<NSSolver>(m, "NSSolver")
        .def(py::init<>())

        .def("init",
            [](NSSolver& self, double domain_L, const py::function& ic_fn) {
                self.init(domain_L, [ic_fn](double x, double y, double z) -> Prim {
                    return prim_from_tuple(ic_fn(x, y, z).cast<py::tuple>());
                });
            },
            "domain_L"_a, "ic_fn"_a,
            "Initialise solver on [0,domain_L]^3; ic_fn(x,y,z)→(rho,u,v,w,p)")

        .def("init_rect",
            [](NSSolver& s, double Lx, double Ly, double Lz,
               std::function<py::tuple(double,double,double)> ic_fn) {
                s.init(Lx, Ly, Lz, [ic_fn](double x, double y, double z) -> Prim {
                    return prim_from_tuple(ic_fn(x, y, z));
                });
            },
            py::arg("Lx"), py::arg("Ly"), py::arg("Lz"), py::arg("ic_fn"),
            "Initialise solver on [0,Lx] x [0,Ly] x [0,Lz].\n"
            "ic_fn(x,y,z) → (rho,u,v,w,p).")

        .def("advance", &NSSolver::advance,
             "Advance one SSP-RK3 step; return dt")
        .def("run",     &NSSolver::run,
             "Advance until cfg.time.t_end or cfg.time.max_steps")
        .def("compute_diag", &NSSolver::compute_diag,
             "Return StepDiag with mass, KE, momentum, etc.")

        .def("get_block_arrays", &get_block_arrays,
             "list[(NVAR,NCELL) float64] — one array per leaf block")
        .def("set_block_arrays", &set_block_arrays,
             "Write a list of (NVAR,NCELL) arrays back to the leaf blocks")
        .def("get_block_h", &get_block_h,
             "list[float] — cell size h for each leaf block (same order as get_block_arrays)")
        .def("adjoint_step", &py_adjoint_step,
             "Adjoint of last advance() step. lam_f: list[(NVAR,NCELL)] → list[(NVAR,NCELL)] (∂J/∂Qn)")

        // Time config properties
        .def_property("cfl",
            [](const NSSolver& s){ return s.cfg.time.cfl; },
            [](NSSolver& s, double v){ s.cfg.time.cfl = v; })
        .def_property("t_end",
            [](const NSSolver& s){ return s.cfg.time.t_end; },
            [](NSSolver& s, double v){ s.cfg.time.t_end = v; })
        .def_property("max_steps",
            [](const NSSolver& s){ return s.cfg.time.max_steps; },
            [](NSSolver& s, int v){ s.cfg.time.max_steps = v; })
        .def_property("verbose",
            [](const NSSolver& s){ return s.cfg.io.verbose; },
            [](NSSolver& s, bool v){ s.cfg.io.verbose = v; })

        .def_readonly("t",    &NSSolver::t,    "Current simulation time")
        .def_readonly("step", &NSSolver::step, "Current step counter");

    // ── Module-level constants ────────────────────────────────────────────────
    m.attr("NVAR")  = NVAR;
    m.attr("NCELL") = NCELL;
    m.attr("NB")    = NB;
    m.attr("NG")    = NG;
    m.attr("NB2")   = NB2;
    m.attr("GAMMA") = GAMMA;
}
