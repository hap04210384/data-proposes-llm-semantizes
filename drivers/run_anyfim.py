# -*- coding: utf-8 -*-
"""Run the AnyFIM anytime engine's progressive activation stream on a dataset and
emit a per-stage CSV.

Usage:
    python drivers/run_anyfim.py <dataset path> <stage count> [output CSV]

Engine sources are not modified: build_anyfim.ensure_anyfim_built produces the patched
binary in build/; at run time the dataset is staged as ..\\TransactionSets\\data.txt
(engine-relative path convention).
"""
import csv
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_anyfim import ensure_anyfim_built, TXN_DIR, ROOT  # noqa: E402,F401
from parse_mfi import parse_anyfim_anytime  # noqa: E402


def run_anyfim(dataset_path, upto_stage, dense=True):
    """Return (rows, results_path); rows is a list of per-stage dicts. dense=False is the ablation control."""
    exe = ensure_anyfim_built(dataset_path, upto_stage, dense=dense)
    exe_dir = os.path.dirname(exe)
    # the engine writes results next to the data file; the data path is cwd-relative ..\TransactionSets\data.txt
    stage_dir = os.path.abspath(os.path.join(exe_dir, "..", "TransactionSets"))
    os.makedirs(stage_dir, exist_ok=True)
    shutil.copyfile(dataset_path, os.path.join(stage_dir, "data.txt"))
    shutil.copyfile(dataset_path, os.path.join(TXN_DIR, "data.txt"))

    # CPU/GPU calibration cache: remember the calibrated value per (dataset md5, stage count)
    # and inject it via the ANYFIM_GPU_PCT environment variable, skipping the ~148 s probe at
    # every start (calibration is a one-time hardware probe, not part of the supply cost).
    import hashlib, json
    cache_p = os.path.join(ROOT, "results", "gpu_pct_cache.json")
    try:
        cache = json.load(open(cache_p, encoding="utf-8"))
    except Exception:
        cache = {}
    key = hashlib.md5(open(dataset_path, "rb").read()).hexdigest()[:12] + f":{int(upto_stage)}"
    env = dict(os.environ)
    if key in cache:
        env["ANYFIM_GPU_PCT"] = str(cache[key])

    t0 = time.perf_counter()
    r = subprocess.run([exe], cwd=exe_dir, capture_output=True, timeout=3600, env=env)
    wall = time.perf_counter() - t0

    # the engine prints "ANYFIM_GPU_PCT=<v>" (fresh calibration) or "GPUtaskPercentage (cached): <v>"
    out = r.stdout.decode("gbk", "replace")
    m = re.search(r"ANYFIM_GPU_PCT=([\d.eE+-]+)", out)
    if m and key not in cache:
        cache[key] = float(m.group(1))
        os.makedirs(os.path.dirname(cache_p), exist_ok=True)
        json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1)

    results = os.path.join(stage_dir, f"data.txt-{int(upto_stage)}-stages=Results.txt")
    if not os.path.exists(results):
        raise RuntimeError(f"engine produced no result file: {results}\n" + out[-1500:])
    header, stages = parse_anyfim_anytime(results)

    runtimes = header.get("runtimePerStage(s)", [])
    sup = header.get("supportThrePerStage", [])
    freq = header.get("frequencyThrePerStage", [])
    rows, cum = [], 0.0
    for k in range(1, int(upto_stage) + 1):
        cum += runtimes[k - 1] if k - 1 < len(runtimes) else 0.0
        rows.append({
            "stage": k,
            "runtime_stage_s": runtimes[k - 1] if k - 1 < len(runtimes) else "",
            "cum_engine_s": cum,
            "wall_total_s": round(wall, 4),
            "support_threshold": sup[k - 1] if k - 1 < len(sup) else "",
            "freq_threshold": int(freq[k - 1]) if k - 1 < len(freq) else "",
            "n_mfi_cum": stages[k]["n"] if k in stages else "",
        })
    return rows, results


def main():
    ds = sys.argv[1]
    st = int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    rows, results = run_anyfim(ds, st)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("CSV:", out)
    for r in rows:
        print(r)
    print("results:", results)


if __name__ == "__main__":
    main()
