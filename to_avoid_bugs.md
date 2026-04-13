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
