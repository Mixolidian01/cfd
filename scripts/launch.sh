#!/usr/bin/env bash
# launch.sh — unified CFD simulation launcher
#
# Usage:
#   ./scripts/launch.sh [options] [config.json]
#
# Options:
#   --backend cpu|gpu|mpi   execution path (default: cpu)
#   --ranks N               number of MPI ranks (mpi backend only, default: 2)
#   --build-dir DIR         build directory (default: <repo_root>/build)
#   -h, --help              print this message and exit
#
# Backends:
#   cpu   Single-rank CPU solver using NSSolver + WENO5-Z SSP-RK3.
#         Binary: build/simulate
#
#   mpi   Multi-rank CPU solver with MPI domain decomposition.
#         Requires MPI-enabled build (cmake detects OpenMPI/MPICH automatically).
#         Binary: mpirun -n N build/simulate
#
#   gpu   Single-GPU solver using GpuGraphSolver (TENO5-A, CUDA Graphs).
#         Requires CUDA build (cmake detects nvcc automatically).
#         Binary: build/simulate_gpu
#
# Examples:
#   ./scripts/launch.sh apps/sod.json
#   ./scripts/launch.sh --backend gpu apps/taylor_green.json
#   ./scripts/launch.sh --backend mpi --ranks 4 apps/taylor_green.json
#
# Build targets:
#   cmake --build build            # builds simulate (cpu/mpi path)
#   cmake --build build -t simulate_gpu  # builds simulate_gpu (gpu path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND="cpu"
RANKS=2
CONFIG=""
BUILD_DIR="$REPO_ROOT/build"

usage() {
    sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)   BACKEND="$2";   shift 2 ;;
        --ranks)     RANKS="$2";     shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; break ;;
        -*)          echo "Error: unknown option '$1'" >&2; usage; exit 1 ;;
        *)           CONFIG="$1"; shift ;;
    esac
done

# Default config
if [[ -z "$CONFIG" ]]; then
    CONFIG="$REPO_ROOT/apps/sod.json"
    echo "launch.sh: no config specified — using default: $CONFIG"
fi

# Resolve to absolute path
CONFIG="$(realpath "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG" >&2
    exit 1
fi

case "$BACKEND" in
    cpu)
        BIN="$BUILD_DIR/simulate"
        if [[ ! -x "$BIN" ]]; then
            echo "Error: $BIN not found." >&2
            echo "  Run: cmake --build build" >&2
            exit 1
        fi
        echo "launch.sh: backend=cpu  config=$CONFIG"
        exec "$BIN" "$CONFIG"
        ;;

    mpi)
        BIN="$BUILD_DIR/simulate"
        if [[ ! -x "$BIN" ]]; then
            echo "Error: $BIN not found." >&2
            echo "  Run: cmake --build build" >&2
            exit 1
        fi
        if ! command -v mpirun &>/dev/null; then
            echo "Error: mpirun not found — install OpenMPI or MPICH." >&2
            exit 1
        fi
        # Warn if binary was built without MPI (stub path gives n_ranks=1)
        if ! nm "$BIN" 2>/dev/null | grep -q "MPI_Init"; then
            echo "Warning: $BIN does not appear to be linked against MPI." >&2
            echo "  Rebuild with an MPI-enabled cmake (cmake should detect it automatically)." >&2
        fi
        echo "launch.sh: backend=mpi  ranks=$RANKS  config=$CONFIG"
        exec mpirun -n "$RANKS" "$BIN" "$CONFIG"
        ;;

    gpu)
        BIN="$BUILD_DIR/simulate_gpu"
        if [[ ! -x "$BIN" ]]; then
            echo "Error: $BIN not found." >&2
            echo "  Run: cmake --build build -t simulate_gpu" >&2
            exit 1
        fi
        # Quick GPU availability check (non-fatal)
        if command -v nvidia-smi &>/dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
            echo "launch.sh: GPU detected: ${GPU_NAME:-unknown}"
        else
            echo "Warning: nvidia-smi not found — GPU availability unknown." >&2
        fi
        echo "launch.sh: backend=gpu  config=$CONFIG"
        exec "$BIN" "$CONFIG"
        ;;

    *)
        echo "Error: unknown backend '$BACKEND' — must be cpu, mpi, or gpu." >&2
        echo "  ./scripts/launch.sh --help" >&2
        exit 1
        ;;
esac
