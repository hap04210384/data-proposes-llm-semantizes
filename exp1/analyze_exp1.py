# -*- coding: utf-8 -*-
"""实验 1 分析： anytime 零调参供给 vs 阈值网格扫荡的成本-质量对照。

输入:
    results/exp1/baseline_<name>_grid.csv   基线网格扫荡（run_baseline_grid.py 产出）
    results/exp1/anyfim_<name>_stages<K>.csv  anytime 逐轮（run_anyfim.py 产出）
    以及两者对应的结果文件（含 MFI 集合，用于 Jaccard 质量核验）

用法:
    python exp1/analyze_exp1.py <数据集名如chess> [anytime轮数CSV路径]

产出（打印 + 写 results/exp1/analysis_<name>.csv）:
    对每个基线阈值 t（从高到低扫描，模拟从业者不知道阈值的真实过程）：
      - 网格侧累计 wall-clock（扫到 t 为止的全部重跑成本）
      - anytime 侧：最早达到 n_mfi(t) 的轮次 k* 及其累计引擎耗时
      - 加速比 = 网格累计 / anytime 累计
      - 质量核验：Jaccard(anytime@k*, baseline@t)
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "exp1")
sys.path.insert(0, os.path.join(ROOT, "drivers"))
from parse_mfi import parse_tensorfim_results, parse_anyfim_anytime  # noqa: E402


def load_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main():
    name = sys.argv[1]                       # 如 chess
    ds = os.path.join(RES, f"{name}.txt")
    base_csv = os.path.join(RES, f"baseline_{name}_grid.csv")
    any_csv = sys.argv[2] if len(sys.argv) > 2 else sorted(
        glob.glob(os.path.join(RES, f"anyfim_{name}_stages*.csv")))[-1]

    base = [r for r in load_csv(base_csv) if r["ok"] == "ok"]
    base.sort(key=lambda r: float(r["threshold"]), reverse=True)   # 从高到低扫描
    any_rows = load_csv(any_csv)

    # anytime 结果文件 -> 每轮累计 MFI 集合
    any_res = sorted(glob.glob(os.path.join(
        ROOT, "build", "anyfim", "src", "x64", "TransactionSets",
        "data.txt-*-stages=Results.txt")))
    assert any_res, "未找到 anytime 结果文件"
    header, stages = parse_anyfim_anytime(any_res[-1])

    # 基线结果文件 -> 阈值到 MFI 集合
    base_sets = {}
    for r in base:
        p = f"{ds}-{float(r['threshold']):.6f}=Results.txt"
        if os.path.exists(p):
            base_sets[float(r["threshold"])] = set(parse_tensorfim_results(p).keys())

    out, cum_grid = [], 0.0
    for r in base:
        t = float(r["threshold"])
        cum_grid += float(r["wall_s"] or 0)
        n_target = int(r["n_mfi"])
        bset = base_sets.get(t, set())
        # 质量口径：召回率 = |anytime@k ∩ baseline@t| / |baseline@t|
        # （anytime 第 k 轮只激活前 k 个高频项；理论上当轮阈值≈t 时两边 MFI 族应一致，
        #   因此按"覆盖基线规则族 ≥90%"对齐轮次，而不是按条数对齐）
        k_star, t_any, recall_best = None, None, None
        if bset:
            for k in sorted(stages):
                a = set(stages[k]["mfis"].keys())
                rec = len(a & bset) / len(bset)
                if rec >= 0.9:
                    k_star, recall_best = k, rec
                    row = next(x for x in any_rows if int(x["stage"]) == k)
                    t_any = float(row["cum_engine_s"])
                    break
        out.append({
            "threshold": t, "n_mfi_baseline": n_target,
            "grid_cum_wall_s": round(cum_grid, 4),
            "anytime_stage_k*": k_star or "",
            "anytime_cum_engine_s": t_any if t_any is not None else "",
            "speedup_grid_over_anytime": round(cum_grid / t_any, 1) if t_any else "",
            "recall@k*": round(recall_best, 4) if recall_best else "",
        })
        print(out[-1])

    op = os.path.join(RES, f"analysis_{name}.csv")
    with open(op, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("CSV:", op)


if __name__ == "__main__":
    main()
