# /phase

Reports the status of each refactor phase and what is safe to start next.

## Usage
/phase

## Instructions

1. Check R0 (C++20): run `cmake -LA build | grep CXX_STANDARD` — must show 20.
2. Check R1 (concepts): search for `include/concepts.hpp`.
   If present, check it defines at least `RiemannFlux` and `EquationOfState`.
3. Check R2 (functors): search for `include/physics/`. If present, list structs found.
4. Check R3 (variant BC): grep `src/` for `std::variant` and `std::visit` in ghost fill.
5. Check R4 (backend tags): search for `include/execution.hpp` and `make_solver` factory.
6. Check R5 (instantiation matrix): search for `src/instantiation_matrix.cpp`.
7. Check R6 (mdspan): grep `cell_block.hpp` for `std::mdspan`.

Report each as: ✅ Done / 🔄 In progress / ❌ Not started.
State which phase is safe to start next and its entry condition.
