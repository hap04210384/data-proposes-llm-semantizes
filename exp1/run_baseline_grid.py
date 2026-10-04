# -*- coding: utf-8 -*-
"""Experiment 1 baseline: threshold-grid sweep with conventional mining
(TensorFIM bitmap baseline engine).

Re-runs the miner from scratch at every threshold of the grid on a given dataset,
recording per-threshold wall time / engine computation time / MFI count. This is
exactly the supply pattern the paper attacks: "unknown threshold -> grid search ->
full re-run per threshold".

Usage:
    python exp1/run_baseline_grid.py <dataset path> <output CSV> [comma-separated thresholds]

Notes:
- engine: ../engines/TensorFIM/code/engine/run/CoParaCG_baseline.exe (precompiled,
  conventional bitmap MFI mining)
- the engine's "exit code 1 = success" is a historical convention; check that the
  output file exists instead;
- the first threshold is run twice (warmup + timed); the warmup result is discarded
  because the first engine start pays a GPU self-check cost.
"""
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(ROOT)
BASELINE_EXE = os.path.join(WORKSPACE, "engines", "TensorFIM", "code", "engine", "run",
                           "CoParaCG_baseline.exe")
COMP = re.compile(r"computation_time\(s\):\s*([\d.]+)")


def run_one(dataset, thr, timeout=1800):
    """Single-threshold run; returns dict(wall_s, computation_s, n_mfi, ok)."""
    out_file = f"{dataset}-{thr:.6f}=Results.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    t0 = time.perf_counter()
    try:
        r = subprocess.run([BASELINE_EXE, dataset, str(thr)], capture_output=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"wall_s": "", "computation_s": "", "n_mfi": "", "ok": "timeout"}
    wall = time.perf_counter() - t0
    log = r.stdout.decode("gbk", "replace")
    if not os.path.exists(out_file):
        return {"wall_s": round(wall, 4), "computation_s": "", "n_mfi": "",
                "ok": f"exit={r.returncode}"}
    comp = COMP.findall(log)
    n = sum(1 for ln in open(out_file, encoding="utf-8", errors="replace")
            if re.match(r"^\d+th:", ln.strip()))
    return {"wall_s": round(wall, 4),
            "computation_s": float(comp[-1]) if comp else "",
            "n_mfi": n, "ok": "ok"}


def main():
    ds = sys.argv[1]
    out = sys.argv[2]
    if len(sys.argv) > 3:
        grid = [float(x) for x in sys.argv[3].split(",")]
    else:
        name = os.path.basename(ds)
        if "chess" in name:                       # dense: sweep from very high support down
            grid = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.85, 0.80]
        elif "retail" in name:                    # sparse: low-support region
            grid = [0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002]
        else:
            grid = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    assert os.path.exists(BASELINE_EXE), f"baseline engine not found: {BASELINE_EXE}"
    # resume support: skip completed thresholds
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8-sig") as f:
            for rec in csv.DictReader(f):
                if rec.get("ok") == "ok":
                    done.add(float(rec["threshold"]))
    # warmup (discarded) to absorb the first-start GPU self-check cost
    if grid[0] not in done:
        print(f"[warmup] {ds} @ {grid[0]}")
        run_one(ds, grid[0])

    rows = []
    for thr in grid:
        if thr in done:
            print(f"[skip] {os.path.basename(ds)} @ {thr} (done)")
            continue
        print(f"[run] {os.path.basename(ds)} @ {thr}")
        rec = run_one(ds, thr)
        rec.update({"dataset": os.path.basename(ds), "threshold": thr})
        rows.append(rec)
        print("   ", rec)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    exists = os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "threshold", "wall_s",
                                          "computation_s", "n_mfi", "ok"])
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print("CSV:", out)


if __name__ == "__main__":
    main()
