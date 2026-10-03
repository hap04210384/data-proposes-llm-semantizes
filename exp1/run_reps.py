# -*- coding: utf-8 -*-
"""实验 1 定稿编排：anytime 流 + 基线网格，N 次重复取中位数 + 同阈值集合验证。

用法:
    python exp1/run_reps.py <chess|retail> <轮数K> [重复次数N]

产出（results/exp1/）:
    anyfim_<name>_K<K>_median<N>.csv     anytime 逐轮中位数
    baseline_<name>_median<N>.csv        基线网格中位数
    pairs_<name>_K<K>.csv                同阈值集合验证（单次）
    各原始重复数据保留在同目录 *_rep<i>.csv
"""
import csv
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "exp1")
sys.path.insert(0, os.path.join(ROOT, "drivers"))
from run_anyfim import run_anyfim  # noqa: E402
from parse_mfi import parse_tensorfim_results, parse_anyfim_anytime  # noqa: E402
from build_anyfim import ROOT as _ROOT  # noqa: E402,F401

BASELINE_EXE = os.path.join(os.path.dirname(ROOT), "engines", "TensorFIM", "code",
                            "engine", "run", "CoParaCG_baseline.exe")
EPS = 1e-9

GRIDS = {
    "chess": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.85, 0.80],
    "retail": [0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.007, 0.005],
}


def median_rows(list_of_rows):
    """同 stage 多组 rows -> 按数值列取中位数的 rows。"""
    by_stage = {}
    for rows in list_of_rows:
        for r in rows:
            by_stage.setdefault(int(r["stage"]), []).append(r)
    out = []
    for k in sorted(by_stage):
        grp = by_stage[k]
        row = {"stage": k}
        for col in grp[0]:
            if col == "stage":
                continue
            vals = []
            for r in grp:
                try:
                    vals.append(float(r[col]))
                except (ValueError, TypeError, KeyError):
                    pass
            row[col] = round(statistics.median(vals), 6) if vals else grp[0][col]
        out.append(row)
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_grid_once(ds, thr):
    out_file = f"{ds}-{thr:.6f}=Results.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    t0 = time.perf_counter()
    try:
        subprocess.run([BASELINE_EXE, ds, str(thr)], capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    wall = time.perf_counter() - t0
    if not os.path.exists(out_file):
        return None
    n = sum(1 for ln in open(out_file, encoding="utf-8", errors="replace")
            if ln.strip()[:1].isdigit() and "th:" in ln[:8])
    return {"threshold": thr, "wall_s": round(wall, 4), "n_mfi": n}


def main():
    name = sys.argv[1]
    K = int(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    ds = os.path.abspath(os.path.join(RES, f"{name}.txt"))
    assert os.path.exists(ds), f"数据集不在 {ds}"

    # ---- 1. anytime N 次重复 ----
    rep_rows = []
    for i in range(N):
        rows, _ = run_anyfim(ds, K, dense=True)
        rep_rows.append(rows)
        print(f"[anytime rep {i+1}/{N}] stage{K} cum={rows[-1]['cum_engine_s']:.4f}s")
    med = median_rows(rep_rows)
    write_csv(os.path.join(RES, f"anyfim_{name}_K{K}_median{N}.csv"), med)
    for i, rows in enumerate(rep_rows):
        write_csv(os.path.join(RES, f"anyfim_{name}_K{K}_rep{i+1}.csv"), rows)

    # ---- 2. 基线网格 N 次重复 ----
    grid = GRIDS[name]
    run_grid_once(ds, grid[0])  # 预热
    grid_rep = []
    for i in range(N):
        recs = []
        for thr in grid:
            r = run_grid_once(ds, thr)
            recs.append(r or {"threshold": thr, "wall_s": "", "n_mfi": ""})
        grid_rep.append(recs)
        print(f"[grid rep {i+1}/{N}] cum={sum(r['wall_s'] for r in recs if r['wall_s']):.3f}s")
    by_thr = {}
    for recs in grid_rep:
        for r in recs:
            by_thr.setdefault(r["threshold"], []).append(r)
    med_g, cum = [], 0.0
    for thr in grid:
        grp = [r for r in by_thr[thr] if r["wall_s"] != ""]
        if not grp:
            continue
        w = round(statistics.median([r["wall_s"] for r in grp]), 4)
        cum += w
        med_g.append({"threshold": thr, "wall_s_median": w,
                      "n_mfi": grp[0]["n_mfi"], "grid_cum_wall_s": round(cum, 4)})
    write_csv(os.path.join(RES, f"baseline_{name}_median{N}.csv"), med_g)

    # ---- 3. 同阈值集合验证（单次）----
    import glob as g
    any_res = sorted(g.glob(os.path.join(ROOT, "build", "anyfim", "src", "x64",
                                         "TransactionSets", "data.txt-*-stages=Results.txt")))[-1]
    header, stages = parse_anyfim_anytime(any_res)
    freq = header.get("frequencyThrePerStage", [])
    trans_num = int(next(ln.split(":")[1] for ln in open(any_res, encoding="utf-8", errors="replace")
                         if ln.startswith("transNum:")))
    pairs = []
    n_bad = 0
    for k in range(1, K + 1):
        th = freq[k - 1] / trans_num - EPS
        r = run_grid_once(ds, th)
        if r is None:
            continue
        a = set(stages[k]["mfis"].keys())
        b = set(parse_tensorfim_results(f"{ds}-{th:.6f}=Results.txt").keys())
        jac = len(a & b) / len(a | b) if (a or b) else 1.0
        if jac < 0.9999:
            n_bad += 1
        pairs.append({"stage": k, "theta": round(th, 6), "n_anytime": len(a),
                      "n_baseline": len(b), "jaccard": round(jac, 4)})
    write_csv(os.path.join(RES, f"pairs_{name}_K{K}.csv"), pairs)
    print(f"[verify] {K - n_bad}/{K} 轮 Jaccard=1.0（差异轮: {n_bad}）")

    # ---- 4. 汇总打印 ----
    any_cum = med[-1]["cum_engine_s"]
    grid_cum = med_g[-1]["grid_cum_wall_s"]
    print(f"\n== {name} 定稿（各 {N} 次中位）==")
    print(f"anytime 到第{K}轮: 引擎累计 {any_cum}s, n_mfi={med[-1]['n_mfi_cum']}")
    print(f"基线网格 {len(grid)} 档累计墙钟: {grid_cum}s")
    print(f"墙钟口径加速比: {grid_cum/any_cum:.1f}x")


if __name__ == "__main__":
    main()
