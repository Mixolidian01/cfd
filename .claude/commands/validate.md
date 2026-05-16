# /validate

Runs the full validation gate suite and reports convergence rates.

## Usage
/validate [phase_number or test_target]

## Instructions

If no argument: run `cmake --build build -t ba` (all t1–t26 gates).

If a phase number is given (e.g. `/validate R2`):
- R0: verify `cmake -LA build | grep CMAKE_CXX_STANDARD` shows 20
- R1: `cmake --build build -t t3` — check T08 rate ≥ 1.8 (concept checks are compile-time)
- R2: `cmake --build build -t t3 && cmake --build build -t t4` — 15/15 and 28/28
- R3: `cmake --build build -t t4` — T16a/T16b (contact angle BC) must pass
- R4: `cmake --build build -t t25 && cmake --build build -t t26` — GPU parity
- R5: `cmake --build build -t t4` and new scheme-selection test if written
- R6: `cmake --build build -t t3` — T08 rate must remain ≥ 1.8

If a test target is given (e.g. `/validate t4`):
- Run `cmake --build build -t [target]` and show the full output.

Always report: pass/fail counts, T08 convergence rate, and any regression vs. baseline.
