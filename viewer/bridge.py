#!/usr/bin/env python3
# bridge.py — CFD live viewer WebSocket bridge
# Streams solver JSON metrics + VTK 2D slices to the browser.
#
# Usage:
#   python viewer/bridge.py [--vtk-dir PATH] [--port 8765] [--solver CMD]
#
# The bridge does two things:
#   1. If --solver is given: spawns the solver, tails its JSON stdout,
#      broadcasts each {"step":...} line to all connected clients.
#   2. On a "slice" request from the browser, reads the latest .vtk file
#      from --vtk-dir, extracts a 2D slice, and sends it as binary.
#
# Requirements:  pip install websockets numpy

import asyncio
import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import websockets

# ── globals ──────────────────────────────────────────────────────────────────
CLIENTS: set = set()
LATEST_VTK_DIR: Path = Path(".")
HISTORY: list = []           # last 2000 metric dicts kept in memory

# ── VTK parser ───────────────────────────────────────────────────────────────

def parse_vtk_structured_points(path: Path) -> dict:
    """
    Parse VTK legacy ASCII structured-points file.
    Returns dict with keys: nx, ny, nz, spacing, fields (name -> np.array shape (nx,ny,nz))
    """
    text = path.read_text()
    lines = text.splitlines()

    nx = ny = nz = 1
    sx = sy = sz = 1.0
    n_points = 0
    fields = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        m = re.match(r'^DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if m:
            nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
            n_points = nx * ny * nz

        m = re.match(r'^SPACING\s+([\deE+\-.]+)\s+([\deE+\-.]+)\s+([\deE+\-.]+)', line)
        if m:
            sx, sy, sz = float(m.group(1)), float(m.group(2)), float(m.group(3))

        m = re.match(r'^SCALARS\s+(\w+)\s+double', line)
        if m:
            fname = m.group(1)
            i += 1  # skip LOOKUP_TABLE line
            i += 1
            vals = []
            while len(vals) < n_points and i < len(lines):
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            arr = np.array(vals, dtype=np.float32).reshape((nz, ny, nx))
            fields[fname] = arr
            continue

        i += 1

    return {"nx": nx, "ny": ny, "nz": nz,
            "sx": sx, "sy": sy, "sz": sz,
            "fields": fields}


def extract_slice(vtk_data: dict, field: str, axis: str, index: int | None) -> bytes:
    """
    Extract a 2D slice from parsed VTK data.
    Returns a binary message:
      4 bytes: width (int32)
      4 bytes: height (int32)
      4 bytes: vmin (float32)
      4 bytes: vmax (float32)
      width*height*4 bytes: float32 values (row-major)
    """
    fields = vtk_data["fields"]
    if field not in fields:
        field = next(iter(fields))  # fallback to first available

    arr = fields[field]  # shape: (nz, ny, nx)
    nz, ny, nx = arr.shape

    if axis == "z":
        idx = index if index is not None else nz // 2
        idx = max(0, min(idx, nz - 1))
        slc = arr[idx, :, :]       # (ny, nx)
        h, w = ny, nx
    elif axis == "y":
        idx = index if index is not None else ny // 2
        idx = max(0, min(idx, ny - 1))
        slc = arr[:, idx, :]       # (nz, nx)
        h, w = nz, nx
    else:  # x
        idx = index if index is not None else nx // 2
        idx = max(0, min(idx, nx - 1))
        slc = arr[:, :, idx]       # (nz, ny)
        h, w = nz, ny

    slc = np.ascontiguousarray(slc, dtype=np.float32)
    vmin = float(np.nanmin(slc))
    vmax = float(np.nanmax(slc))
    if math.isnan(vmin) or vmin == vmax:
        vmax = vmin + 1.0

    header = struct.pack("<iiiff", w, h, idx, vmin, vmax)
    return header + slc.tobytes()


def latest_vtk_files(vtk_dir: Path, step: int | None) -> list[Path]:
    """Return .vtk files for a given step, or the latest step found."""
    files = sorted(vtk_dir.glob("*.vtk"))
    if not files:
        return []
    if step is None:
        # Find highest step number
        def step_num(p):
            m = re.search(r'step(\d+)', p.name)
            return int(m.group(1)) if m else -1
        max_step = max(step_num(f) for f in files)
        files = [f for f in files if step_num(f) == max_step]
    else:
        files = [f for f in files if f"step{step:06d}" in f.name]
    return files


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handler(websocket):
    CLIENTS.add(websocket)
    # Send full history on connect
    for entry in HISTORY[-500:]:
        try:
            await websocket.send(json.dumps(entry))
        except Exception:
            break

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
                
            if msg.get("type") == "shutdown":
                await websocket.send(json.dumps({"info": "shutting down"}))
                await asyncio.sleep(0.1)
                os._exit(0)

            if msg.get("type") == "slice":
                field  = msg.get("field", "rho")
                axis   = msg.get("axis", "z")
                step   = msg.get("step", None)
                index  = msg.get("index", None)

                files = latest_vtk_files(LATEST_VTK_DIR, step)
                if not files:
                    await websocket.send(json.dumps({"error": "no vtk files found"}))
                    continue

                # Use first block file (blk000)
                path = files[0]
                try:
                    vtk_data = parse_vtk_structured_points(path)
                    payload  = extract_slice(vtk_data, field, axis, index)
                    await websocket.send(payload)
                except Exception as e:
                    await websocket.send(json.dumps({"error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)


async def broadcast(msg: str):
    dead = set()
    for ws in list(CLIENTS):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    CLIENTS.difference_update(dead)


# ── Solver stdout reader ──────────────────────────────────────────────────────

async def tail_solver(cmd: list[str]):
    """Spawn solver, read JSON lines from stdout, broadcast."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        HISTORY.append(data)
        if len(HISTORY) > 2000:
            HISTORY.pop(0)
        await broadcast(line)
    await proc.wait()


async def tail_file(path: Path):
    """Tail a log file that the solver writes JSON lines to."""
    with path.open() as f:
        f.seek(0, 2)  # seek to end
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.05)
                continue
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            HISTORY.append(data)
            if len(HISTORY) > 2000:
                HISTORY.pop(0)
            await broadcast(line)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global LATEST_VTK_DIR

    parser = argparse.ArgumentParser(description="CFD live viewer bridge")
    parser.add_argument("--vtk-dir",  default=".",      help="Directory containing .vtk output files")
    parser.add_argument("--port",     type=int, default=8765, help="WebSocket port (default 8765)")
    parser.add_argument("--solver",   default=None,     help="Solver command to spawn (optional)")
    parser.add_argument("--log-file", default=None,     help="Tail a JSON log file instead of spawning solver")
    args = parser.parse_args()

    LATEST_VTK_DIR = Path(args.vtk_dir).expanduser().resolve()
    print(f"[bridge] VTK dir : {LATEST_VTK_DIR}")
    print(f"[bridge] WS port : {args.port}")

    async def run():
        server = await websockets.serve(handler, "localhost", args.port)
        print(f"[bridge] Listening on ws://localhost:{args.port}")
        tasks = [asyncio.ensure_future(server.wait_closed())]
        if args.solver:
            cmd = args.solver.split()
            print(f"[bridge] Spawning solver: {cmd}")
            tasks.append(asyncio.ensure_future(tail_solver(cmd)))
        elif args.log_file:
            lp = Path(args.log_file).expanduser().resolve()
            print(f"[bridge] Tailing log: {lp}")
            tasks.append(asyncio.ensure_future(tail_file(lp)))
        await asyncio.gather(*tasks)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\\n[bridge] Stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n[bridge] Stopped.")
