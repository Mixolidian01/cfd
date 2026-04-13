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
an earlier fix (“Cell size h is always read from block->h”). The
test was written against the old API and never updated.

**Rule B:** Before writing or committing any test file, cross-check
every struct/class member access against the current header. When a
header comment says a field was removed or replaced (e.g. “BlockNode::h
— use block->h”), grep test sources for the old field name and update
them before committing.
