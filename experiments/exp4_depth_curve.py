# -*- coding: utf-8 -*-
"""Exp 4b supply depth vs rule interestingness: median lift of the top-20 size>=2
MFIs at sampled supply stages.

Hypothesis: shallow stages (high threshold) yield multi-item sets that are chance
co-occurrences of frequent items (lift ~ 1); the deeper the stage (lower the
threshold), the higher the lift of surviving multi-item sets -- zero-threshold
supply also delivers the "rare but strongly associated" rules that a fixed-threshold
method easily misses.

Datasets: TCGA-BRCA (K=3000) vs retail (K=50).
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
    print("[Exp4b] TCGA-BRCA K=3000 depth curve ...")
    tcga_curve = depth_curve(os.path.join(RES, "exp2", "tcga.txt"), 3000, sample_every=200)
    print("[Exp4b] retail K=50 depth curve ...")
    retail_curve = depth_curve(os.path.join(RES, "exp1", "retail.txt"), 50, sample_every=5)
    out = {"tcga_K3000": tcga_curve, "retail_K50": retail_curve}
    json.dump(out, open(os.path.join(OUT, "exp4_depth_curve.json"), "w", encoding="utf-8"), indent=1)
    for name, c in out.items():
        print(f"--- {name} ---")
        for r in c:
            print(r)
