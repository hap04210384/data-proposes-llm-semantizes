# -*- coding: utf-8 -*-
"""实验 1 双口径对照：引擎时间口径 vs 墙钟口径，各 5 次重复取中位数。

anytime 侧直接用 run_reps 已保存的 rep CSV（含 cum_engine_s 与 wall_total_s）；
网格侧重跑 5 次，除墙钟外解析每次结果文件头的 computation_time(s)。

产出 results/exp1/dual_caliber_<ds>.csv:
    any_cum_engine_s, any_wall_s, grid_cum_engine_s, grid_cum_wall_s,
    speedup_engine (网格引擎累计/anytime引擎累计),
    speedup_wall   (网格墙钟累计/anytime墙钟)
"""
import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "exp1")
sys.path.insert(0, os.path.join(ROOT, "drivers"))
from run_reps import GRIDS, run_grid_once, BASELINE_EXE  # noqa: E402

COMP = re.compile(r"^computation_time\(s\):\s*(\S+)", re.M)


def grid_once_with_engine(ds_path, thr):
    """跑一次网格档，返回 (wall_s, engine_s) 或 None。"""
    out_file = f"{ds_path}-{thr:.6f}=Results.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    t0 = time.perf_counter()
    try:
        subprocess.run([BASELINE_EXE, ds_path, str(thr)], capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    wall = time.perf_counter() - t0
    if not os.path.exists(out_file):
        return None
    m = COMP.search(open(out_file, encoding="utf-8", errors="replace").read())
    return wall, float(m.group(1)) if m else None


def anytime_medians(name, K, N):
    cums, walls = [], []
    for f in glob.glob(os.path.join(RES, f"anyfim_{name}_K{K}_rep*.csv")):
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        cums.append(float(rows[-1]["cum_engine_s"]))
        walls.append(float(rows[-1]["wall_total_s"]))
    assert len(cums) == N, f"rep 数 {len(cums)} != {N}"
    return statistics.median(cums), statistics.median(walls)


def main():
    name, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 5
    ds = os.path.abspath(os.path.join(RES, f"{name}.txt"))

    any_eng, any_wall = anytime_medians(name, K, N)

    grid = GRIDS[name]
    grid_once_with_engine(ds, grid[0])  # 预热
    walls, engis = [], []
    for i in range(N):
        w_rec, e_rec = [], []
        for thr in grid:
            r = grid_once_with_engine(ds, thr)
            w_rec.append(r[0] if r else None)
            e_rec.append(r[1] if r else None)
        walls.append(w_rec)
        engis.append(e_rec)
        print(f"[grid rep {i+1}/{N}] wall_cum={sum(w for w in w_rec if w):.3f}s "
              f"engine_cum={sum(e for e in e_rec if e):.5f}s")

    def med_sum(recs):
        per_slot = list(zip(*recs))
        return sum(statistics.median([x for x in slot if x is not None])
                   for slot in per_slot if any(x is not None for x in slot))

    grid_wall = med_sum(walls)
    grid_eng = med_sum(engis)

    row = {
        "dataset": name, "K": K, "reps": N,
        "any_cum_engine_s": round(any_eng, 6),
        "any_wall_s": round(any_wall, 4),
        "grid_cum_engine_s": round(grid_eng, 6),
        "grid_cum_wall_s": round(grid_wall, 4),
        "speedup_engine": round(grid_eng / any_eng, 2),
        "speedup_wall": round(grid_wall / any_wall, 2),
    }
    out = os.path.join(RES, f"dual_caliber_{name}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print("\n== 双口径 ==")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print("CSV:", out)


if __name__ == "__main__":
    main()
