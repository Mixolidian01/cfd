#!/usr/bin/env bash
# PreToolUse hook: blocks edits to to_debug and warns on kernel edits without tests.
# Claude Code passes tool input as JSON on stdin.

input=$(cat)
file=$(python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('file_path') or d.get('path') or '')
" <<< "$input" 2>/dev/null)

# Block any path that could be on to_debug via worktree
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$branch" = "to_debug" ] || [ "$branch" = "main" ]; then
    echo "BLOCKED: edits are not permitted on branch '$branch'. Switch to to_refactor."
    exit 2
fi

# Warn if a GPU kernel is edited without a corresponding CPU unit test
if echo "$file" | grep -q "src/cuda/\|kernels/"; then
    scheme=$(basename "$file" .cu | sed 's/_kernel//')
    if [ ! -f "tests/unit/test_${scheme}.cpp" ] && \
       [ ! -f "tests/cuda/test_${scheme}.cu" ]; then
        echo "WARNING: no unit test found for '${scheme}'."
        echo "Write a CPUSerial unit test in tests/unit/test_${scheme}.cpp first (Rule 14)."
        # Non-fatal: prints warning but does not block (exit 0)
    fi
fi

exit 0
