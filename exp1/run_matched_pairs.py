# -*- coding: utf-8 -*-
"""实验 1 同阈值对照：anytime 第 k 轮 vs 基线在 θ_k 的单次挖掘。

对 anytime 流的每一轮 k：
  1. 用其轮阈值 θ_k = freq_k / transNum（微偏 ε 规避引擎浮点边界）跑基线引擎一次；
  2. 校验集合一致性（Jaccard；差异应仅限"支持度恰等于阈值"的并列项）；
  3. 记录两侧耗时 —— 这是"同等质量"前提下的公平成本对照。

用法:
    python exp1/run_matched_pairs.py <数据集名如chess> [最大轮数K]
产出:
    results/exp1/pairs_<name>_K<K>.csv
"""
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "exp1")
sys.path.insert(0, os.path.join(ROOT, "drivers"))
from parse_mfi import parse_tensorfim_results, parse_anyfim_anytime  # noqa: E402
from run_anyfim import run_anyfim  # noqa: E402

BASELINE_EXE = os.path.join(os.path.dirname(ROOT), "engines", "TensorFIM", "code",
                            "engine", "run", "CoParaCG_baseline.exe")
EPS = 1e-9  # 避开 3196*θ 浮点回绕；不影响整数频数截断


def main():
    name = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    ds = os.path.abspath(os.path.join(RES, f"{name}.txt"))
    any_csv = os.path.join(RES, f"anyfim_{name}_stages{K}.csv")

    if not os.path.exists(any_csv):
        run_anyfim(ds, K)  # 引擎产物在 build/ 下，CSV 随后重读

    # anytime 结果（重新跑以确保与 build 目录一致，或直接用已有）
    import glob as g
    any_res = sorted(g.glob(os.path.join(ROOT, "build", "anyfim", "src", "x64",
                                         "TransactionSets", "data.txt-*-stages=Results.txt")))[-1]
    header, stages = parse_anyfim_anytime(any_res)
    freq = header.get("frequencyThrePerStage", [])
    runtimes = header.get("runtimePerStage(s)", [])
    trans_num = None
    for ln in open(any_res, encoding="utf-8", errors="replace"):
        if ln.startswith("transNum:"):
            trans_num = int(ln.split(":")[1])
            break
    assert trans_num, "结果文件中未找到 transNum"

    # 预热基线引擎（丢弃）
    subprocess.run([BASELINE_EXE, ds, "0.99"], capture_output=True, timeout=1800)

    rows, cum_any = [], 0.0
    for k in range(1, K + 1):
        cum_any += runtimes[k - 1]
        th = freq[k - 1] / trans_num - EPS
        out = f"{ds}-{th:.6f}=Results.txt"
        if os.path.exists(out):
            os.remove(out)
        t0 = time.perf_counter()
        subprocess.run([BASELINE_EXE, ds, repr(th)], capture_output=True, timeout=1800)
        wall = time.perf_counter() - t0
        if not os.path.exists(out):
            rows.append({"stage": k, "theta": round(th, 6), "n_anytime": stages[k]["n"],
                         "n_baseline": "", "jaccard": "", "anytime_cum_s": round(cum_any, 6),
                         "baseline_wall_s": round(wall, 4)})
            continue
        a = set(stages[k]["mfis"].keys())
        b = set(parse_tensorfim_results(out).keys())
        jac = len(a & b) / len(a | b) if (a or b) else 1.0
        rows.append({"stage": k, "theta": round(th, 6), "n_anytime": stages[k]["n"],
                     "n_baseline": len(b), "jaccard": round(jac, 4),
                     "anytime_cum_s": round(cum_any, 6), "baseline_wall_s": round(wall, 4)})
        print(rows[-1])

    op = os.path.join(RES, f"pairs_{name}_K{K}.csv")
    with open(op, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("CSV:", op)


if __name__ == "__main__":
    main()
