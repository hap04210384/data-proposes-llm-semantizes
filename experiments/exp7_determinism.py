# -*- coding: utf-8 -*-
"""Exp7 确定性与消融实验。

(a) 确定性：同输入连续 2 次运行 anytime 流（dense），结果集逐字节/逐集合比对；
(b) 消融：dense ID 映射 on/off 各跑一次，比对逐轮规则集合（Jaccard）与引擎耗时。
结果落盘 results/exp7/（不入 git）。
"""
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.run_anyfim import run_anyfim, parse_anyfim_anytime  # noqa: E402
DATA = os.path.join(ROOT, "results", "exp1", "retail.txt")
OUT = os.path.join(ROOT, "results", "exp7")
K = 50


def snapshot(rows, results_path, tag):
    _, stages = parse_anyfim_anytime(results_path)
    sets = {str(k): sorted(tuple(sorted(m)) for m in v["mfis"]) for k, v in stages.items()}
    raw = open(results_path, "rb").read()
    return {"sets": sets,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "engine_s": [float(r["runtime_stage_s"]) for r in rows]}


def jaccard(a, b):
    a, b = set(map(tuple, a)), set(map(tuple, b))
    return len(a & b) / len(a | b) if a | b else 1.0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("[Exp7-a] 确定性：连续 2 次运行 retail K=50 (dense) ...")
    snaps = []
    for i in (1, 2):
        rows, res_path = run_anyfim(DATA, K, dense=True)
        s = snapshot(rows, res_path, f"run{i}")
        snaps.append(s)
        print(f"  run{i}: {len(rows)} stages, sha={s['sha256'][:16]}..., engine_total={sum(s['engine_s']):.4f}s")

    same_hash = snaps[0]["sha256"] == snaps[1]["sha256"]
    stages = sorted(set(snaps[0]["sets"]) | set(snaps[1]["sets"]))
    min_j = min(jaccard(snaps[0]["sets"][str(k)], snaps[1]["sets"][str(k)]) for k in stages)
    print(f"  字节一致: {same_hash}, 逐轮 Jaccard 最小值: {min_j:.6f}")

    print("[Exp7-b] 消融：retail K=50 dense=False ...")
    rows_nd, res_nd = run_anyfim(DATA, K, dense=False)
    snap_nd = snapshot(rows_nd, res_nd, "nodense")
    dense_sets = snaps[0]["sets"]
    stages = sorted(set(dense_sets) | set(snap_nd["sets"]))
    jac = {k: jaccard(dense_sets[str(k)], snap_nd["sets"][str(k)]) for k in stages}
    min_j_nd = min(jac.values())
    eng_dense = sum(snaps[0]["engine_s"])
    eng_nd = sum(snap_nd["engine_s"])
    print(f"  dense 引擎总耗时 {eng_dense:.4f}s vs nodense {eng_nd:.4f}s "
          f"(dense 加速 {eng_nd / eng_dense:.2f}x); 逐轮 Jaccard 最小值 {min_j_nd:.6f}")

    with open(os.path.join(OUT, "exp7_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"determinism": {"byte_identical": same_hash, "min_stage_jaccard": min_j,
                                   "run_sha256": [s["sha256"] for s in snaps],
                                   "engine_total_s": [sum(s["engine_s"]) for s in snaps]},
                   "dense_remap_ablation": {"engine_total_dense_s": eng_dense,
                                            "engine_total_nodense_s": eng_nd,
                                            "speedup_x": eng_nd / eng_dense,
                                            "min_stage_jaccard": min_j_nd,
                                            "stage_jaccards": {str(k): round(v, 6) for k, v in jac.items()}}},
                  f, ensure_ascii=False, indent=1)
    print("saved results/exp7/exp7_summary.json")
