# /migrate

Migrates one free function or module to the target three-layer architecture.

## Usage
/migrate [function_name or file] [layer: functor|concept|backend]

## Instructions

Given a target (e.g. `/migrate hllc_es_flux_t functor`):

### If layer = functor (Layer P)
1. Read the current implementation in `src/operators.cpp`.
2. Create `include/physics/[name].hpp` as a struct with:
   - `operator()` annotated `__host__ __device__`
   - `template <Axis DIR>` with `constexpr int IX=...` index rotation
   - Physics state as member variables only
   - The concept from `include/concepts.hpp` satisfied
3. Replace the original free function call sites with the functor `operator()`.
4. Run `/validate R2` to confirm no regression.

### If layer = concept (Layer C)
1. Read `include/concepts.hpp` (create it if absent).
2. Define the concept with: `requires` clause on `operator()` signature,
   and `static constexpr bool is_entropy_stable` compile-time property.
3. Add `static_assert(ConceptName<TargetType>)` at the first call site.
4. Run `/validate R1`.

### If layer = backend (Layer E)
1. Read `include/execution.hpp` (create if absent).
2. Add a tag struct (e.g. `struct CPUSerial{};`).
3. Write a `dispatch(CPUSerial, ...)` overload wrapping the existing CPU call.
4. Run `/validate R4`.

Do not touch the existing implementation until the new layer passes tests.
