# to_avoid_bugs.md — Rules derived from misbehaviour analysis

This file is appended whenever the AI assistant produces incorrect output.
Each entry contains a derived rule that must be followed going forward.

---

## Rule 001 — Always use `../include/` prefix for headers in `src/*.cpp`

**Derived from:** Answer #14 (Phase 1 build fix)

**Misbehaviour:** `src/block_tree.cpp` was committed with
`#include "block_tree.hpp"` and `#include "amr_operators.hpp"`,
which are paths relative to `src/`. All other translation units in `src/`
use `#include "../include/<name>.hpp"`. The compiler could not resolve
the headers when invoked from the build root, causing `test_block` and
`test_operators` to fail to build.

**Rule:** Before committing any file in `src/`, verify that every
`#include` for project headers uses the `../include/` prefix.
Never write bare `#include "<header>.hpp"` in a `src/*.cpp` file
without checking the existing include convention in that file
and in its siblings.

---

## Rule 002 — Place a `.cpp` in the library that first calls its symbols

**Derived from:** Answer #15 (fill_cf_ghosts undefined reference)

**Misbehaviour:** `amr_operators.cpp` was assigned only to `ns_solver`
(Layer 3), but `block_tree.cpp` (Layer 1) calls `fill_cf_ghosts()` which
is defined in `amr_operators.cpp`. `test_block` and `test_operators`
link only against `block`/`operators`, so the symbol was absent at link
time.

**Rule:** Before assigning a `.cpp` to a CMake library target, check
which other `.cpp` files call functions defined in it. The source file
must be compiled into the lowest-layer library that contains a caller.
If a higher-layer library also needs it, it inherits the symbol
transitively via `target_link_libraries(... PUBLIC ...)` — do NOT list
the same `.cpp` in multiple `add_library()` calls (ODR violation).

---

## Rule 003 — Free-list allocators must guarantee contiguous group allocation; test files must not use removed API fields

**Derived from:** Answer #16 (A05 mass leak + test_step7 compile failure)

**Misbehaviour A — A05 (mass leak through regrid):**
`refine()` called `alloc_node()` eight times individually. After a
refine → coarsen → refine cycle the free-list returns non-contiguous
slots. All downstream code (`restrict_to_parent`, `prolongate_to_children`,
`coarsen`, `balance`) uses `first_child + oct` which assumes the 8
children are stored at consecutive indices. Non-contiguous allocation
caused wrong nodes to be read/written, corrupting the restriction and
leaking ∼2.7×10⁻⁸ relative mass per regrid step.

**Rule A:** Whenever a data structure requires N objects to be addressed
as a contiguous run (index 0..N-1 relative to a base), they MUST be
allocated as a group. Provide a dedicated `alloc_node_group(n)` that
either finds n consecutive free-list slots or appends n new nodes.
Never substitute N individual `alloc_node()` calls when contiguity is
required by callers.

**Misbehaviour B — test_step7 (stale API):**
`test_step7.cpp` used `node.h` which was removed from `BlockNode` in
an earlier fix ("Cell size h is always read from block->h"). The
test was written against the old API and never updated.

**Rule B:** Before writing or committing any test file, cross-check
every struct/class member access against the current header. When a
header comment says a field was removed or replaced (e.g. "BlockNode::h
— use block->h"), grep test sources for the old field name and update
them before committing.

---

## Rule 004 — Ghost cells for derived arrays (mu_t, etc.) must receive periodic/wall treatment identical to Q

**Derived from:** Answer #21 (S07/S08 SGS momentum conservation failure)

**Misbehaviour:** In `sgs.cpp`, the `mu_t` precomputation loop skipped
ghost indices (`i<1 || i>NB2-2`), leaving `mu_t` at the `+x/+y/+z`
ghost layer (index `NB2-1`) as 0. Face-centred stresses at the periodic
boundary used `mu_t=0` on one side, halving the boundary stress. This
broke the telescoping sum of `div(τ)` on the periodic domain, producing
a non-zero net momentum source and failing S07/S08.

**Rule:** Whenever a derived per-cell array (mu_t, nu_t, indicator
functions, etc.) is computed from Q and then used with stencils that
access neighbouring cells (including ghost layers), the ghost cells of
that derived array MUST be filled with the same BC as Q (periodic wrap,
wall reflection, CF interpolation). Always apply the ghost fill
immediately after the interior computation loop, before any stencil
operation that reads ghost neighbours.

---

## Rule 005 — Every defined inter-layer communication function must be wired into the call graph; verify with grep before committing

**Derived from:** Answer #21 (A05 mass leak — accumulate_fine_flux never called)

**Misbehaviour:** `BlockTree::accumulate_fine_flux()` was defined in
`block_tree.cpp` and declared in `block_tree.hpp`, but was never called
from `operators.cpp`, `tree_rhs()`, or any other site. Consequently the
flux register was always empty, `apply_flux_correction()` was a no-op,
and coarse cells at coarse-fine interfaces received zero net flux — a
pure mass leak.

**Rule:** After defining any new function that is part of a protocol
(Berger-Colella reflux, ghost fill, restriction, etc.), immediately
grep the entire codebase for the function name to confirm it is called
from the correct site. A function that is defined but has zero call
sites is a bug. Document the intended call site in a comment on the
function declaration (`// called from tree_rhs() for each fine leaf at
a CF face`). Do not commit until at least one call site exists and is
verified to pass the associated test.

---

## Rule 006 — regrid() must run BEFORE the RK3+Berger-Colella cycle, never after apply_flux_correction()

**Derived from:** Answer #35 (A05-fix5 — mass leak ~2.69e-8 per coarsening event)

**Misbehaviour:** `regrid()` was called at the END of `advance()`, after
`apply_flux_correction(dt)`. When `regrid()` triggered `coarsen()` →
`restrict_to_parent()`, the coarse parent cell was re-averaged from its
8 fine children. This overwrote the flux-corrected value that
`apply_flux_correction` had just written into the coarse CF boundary
cell, discarding the Berger-Colella correction entirely. Each coarsening
event leaked the exact magnitude of the correction (~2.69e-8 relative
mass), accumulating monotonically over regrid steps and failing A05.

**Rule:** `regrid()` must always be called at the **top** of `advance()`,
operating on `Q^n` BEFORE `zero_flux_registers()`, the RK3 stages, and
`apply_flux_correction()`. The correct invariant is:

```
regrid(Q^n)  →  zero_regs  →  RK3 stages  →  apply_flux_correction(dt)
```

This guarantees the tree topology is immutable during the entire
zero→accumulate→correct sequence, and that `apply_flux_correction` is
the **last write** to every leaf cell in each `advance()` call. Never
place `regrid()` after `apply_flux_correction()`. Add a `step > 0`
guard so that regrid is not triggered at step 0 (initial conditions,
no dynamics yet).
