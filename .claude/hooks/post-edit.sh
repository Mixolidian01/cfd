#!/usr/bin/env bash
# PostToolUse hook: runs the fast operator gate (t3) after any edit to operators.cpp
# or include/physics/, to catch T08 convergence regressions immediately.

input=$(cat)
file=$(python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('file_path') or d.get('path') or '')
" <<< "$input" 2>/dev/null)

if echo "$file" | grep -qE "operators\.cpp|include/physics/|include/concepts\.hpp"; then
    echo "--- post-edit: running t3 (operator convergence gate) ---"
    cmake --build build -t t3 --quiet 2>&1 | tail -6
fi

exit 0
