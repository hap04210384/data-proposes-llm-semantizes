# -*- coding: utf-8 -*-
"""Exp4b 供给深度 vs 规则有趣度：各阶段 top-20 size>=2 MFI 的 lift 中位数。

假设：浅阶段（高阈值）的多项项集是高频项的偶然同现（lift≈1）；
越深（阈值越低），幸存的多项项集 lift 越高——zero-threshold 供给把
"罕见但强关联"的规则也交出来，这正是固定阈值方法容易错过的部分。
数据集：TCGA-BRCA (K=3000) 与 retail (K=50) 对照。
"""
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.run_anyfim import run_anyfim, parse_anyfim_anytime  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "exp4")
GPU_PCT = 0.0788544


def seed_cache(path, k):
    cache_p = os.path.join(RES, "gpu_pct_cache.json")
    cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
    key = hashlib.md5(open(path, "rb").read()).hexdigest()[:12] + f":{k}"
    cache[key] = GPU_PCT
    json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1)


def exact_counts(path):
    tx, f1 = [], {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        items = [int(x) for x in ln.split()]
        if items:
            tx.append(set(items))
            for i in items:
                f1[i] = f1.get(i, 0) + 1
    return tx, f1


def lift(m, tx, f1):
    n = len(tx)
    cnt = sum(1 for t in tx if m <= t)
    indep = 1.0
    for it in m:
        indep *= f1[it] / n
    return (cnt / n) / indep if indep > 0 else None


def depth_curve(data_path, k, sample_every=100):
    seed_cache(data_path, k)
    _, res = run_anyfim(data_path, k, dense=True)
    _, stages = parse_anyfim_anytime(res)
    tx, f1 = exact_counts(data_path)
    curve = []
    for stage_no in sorted(stages):
        if stage_no % sample_every and stage_no != max(stages):
            continue
        mfis = stages[stage_no]["mfis"]
        multi = sorted((m for m in mfis if len(m) >= 2), key=lambda m: -mfis[m])[:20]
        if not multi:
            continue
        lifts = [lift(m, tx, f1) for m in multi]
        lifts = [L for L in lifts if L is not None]
        curve.append({"stage": stage_no,
                      "median_lift": round(statistics.median(lifts), 3) if lifts else None,
                      "max_lift": round(max(lifts), 2) if lifts else None,
                      "n_multi": len(multi)})
    return curve


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("[Exp4b] TCGA-BRCA K=3000 深度曲线 ...")
    tcga_curve = depth_curve(os.path.join(RES, "exp2", "tcga.txt"), 3000, sample_every=200)
    print("[Exp4b] retail K=50 深度曲线 ...")
    retail_curve = depth_curve(os.path.join(RES, "exp1", "retail.txt"), 50, sample_every=5)
    out = {"tcga_K3000": tcga_curve, "retail_K50": retail_curve}
    json.dump(out, open(os.path.join(OUT, "exp4_depth_curve.json"), "w", encoding="utf-8"), indent=1)
    for name, c in out.items():
        print(f"--- {name} ---")
        for r in c:
            print(r)
