# -*- coding: utf-8 -*-
"""Exp4 TCGA-BRCA 规则质量实验（无金标准场景）。

指标 1 稳定性：80% bootstrap 子抽样 x 5，top-50 MFI 集合两两 Jaccard。
指标 2 统计强度：全量 top-50 MFI 的 lift = P(XY)/(P(X)P(Y)) 分布，
              lift >> 1 表示远超独立同现基线（数据驱动的质量证据）。
指标 3 规模构成：top-50 中 size>=2 的项集占比（可作为规则派生的原料）。
LLM 不入本实验：项为匿名基因 ID，无语义映射表，避免编造基因名（复现底线）。
"""
import hashlib
import itertools
import json
import os
import random
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.run_anyfim import run_anyfim, parse_anyfim_anytime  # noqa: E402

RES2 = os.path.join(ROOT, "results", "exp2")
DATA = os.path.join(RES2, "tcga.txt")
OUT = os.path.join(ROOT, "results", "exp4")
GPU_PCT = 0.0788544
K = 3000
TOP = 50
SEED = 42


def seed_cache(path, k):
    cache_p = os.path.join(ROOT, "results", "gpu_pct_cache.json")
    cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
    key = hashlib.md5(open(path, "rb").read()).hexdigest()[:12] + f":{k}"
    cache[key] = GPU_PCT
    json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1)


def top_mfis(results_path, top=TOP):
    """最终阶段按频数排序取 top 个 size>=2 的 MFI（规则原料；单独特集另计）。"""
    _, stages = parse_anyfim_anytime(results_path)
    final = stages[max(stages)]
    ranked = sorted(final["mfis"].items(), key=lambda kv: -kv[1])
    multi = [set(m) for m, _ in ranked if len(m) >= 2]
    single = sum(1 for m, _ in ranked if len(m) == 1)
    return multi[:top], {"n_mfis_final": len(ranked), "n_singletons_top": single,
                         "n_size_ge2_final": len(multi)}


def jaccard(a, b):
    a, b = set(map(frozenset, a)), set(map(frozenset, b))
    return len(a & b) / len(a | b) if a or b else 1.0


def exact_freqs(path):
    tx = []
    freq1 = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            items = [int(x) for x in ln.split()]
            if items:
                tx.append(set(items))
                for i in items:
                    freq1[i] = freq1.get(i, 0) + 1
    return tx, freq1


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    lines = open(DATA, encoding="utf-8", errors="replace").read().splitlines()

    sets = {}
    stats = {}
    print("[Exp4] 全量 + 5 个 80% bootstrap 子抽样, K=3000 ...")
    paths = {"full": DATA}
    for i in range(5):
        idx = sorted(rng.sample(range(len(lines)), int(0.8 * len(lines))))
        p = os.path.join(OUT, f"tcga_boot{i}.txt")
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines[j] for j in idx) + "\n")
        paths[f"boot{i}"] = p

    for name, p in paths.items():
        seed_cache(p, K)
        _, res = run_anyfim(p, K, dense=True)
        sets[name], stats[name] = top_mfis(res)
        print(f"  {name}: top-{TOP} size>=2 MFI collected {stats[name]}")

    pairs = [(a, b) for a, b in itertools.combinations(sets, 2) if a != "full" and b != "full"]
    jacs = [jaccard(sets[a], sets[b]) for a, b in pairs]
    jacs_full = [jaccard(sets["full"], sets[b]) for b in sets if b != "full"]

    # lift：全量 top-50 中 size>=2 项集
    tx, freq1 = exact_freqs(DATA)
    n = len(tx)
    lifts, sizes = [], []
    for m in sets["full"]:
        sizes.append(len(m))
        if len(m) >= 2:
            cnt = sum(1 for t in tx if m <= t)
            pxy = cnt / n
            indep = 1.0
            for it in m:
                indep *= freq1[it] / n
            lifts.append(pxy / indep if indep > 0 else float("inf"))

    summary = {
        "stability": {
            "bootstrap_pairwise_jaccard_mean": round(statistics.mean(jacs), 4),
            "bootstrap_pairwise_jaccard_min": round(min(jacs), 4),
            "full_vs_bootstrap_jaccard_mean": round(statistics.mean(jacs_full), 4),
            "full_vs_bootstrap_jaccard_min": round(min(jacs_full), 4),
        },
        "lift_top50": {
            "n_size_ge2": len(lifts),
            "median_lift": round(statistics.median(lifts), 2) if lifts else None,
            "min_lift": round(min(lifts), 2) if lifts else None,
            "max_lift": round(max(lifts), 2) if lifts else None,
            "frac_lift_gt_10": round(sum(1 for L in lifts if L > 10) / len(lifts), 3) if lifts else None,
        },
        "size_distribution_top50": {str(s): sizes.count(s) for s in sorted(set(sizes))},
        "final_stage_stats_full": stats["full"],
        "k": K, "top": TOP, "bootstrap": 5, "subsample": 0.8, "seed": SEED,
    }
    json.dump(summary, open(os.path.join(OUT, "exp4_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=2))
