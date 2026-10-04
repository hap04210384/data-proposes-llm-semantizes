# -*- coding: utf-8 -*-
"""Exp2 归因诊断：kosarak 预处理 135 s 到底花在哪——位图分配还是解析？

方法：构造 3 个变体，与全量对比（均 K=1，使引擎计算量可忽略，墙钟≈启动+预处理）：
  full      990,002 事务, 41,270 项（位图 ≈ n_items*n_tx/8 = 5.16 GB）
  quarter   前 250,000 事务（位图 1.29 GB，解析量 1/4）
  tenth     前 100,000 事务（位图 0.52 GB，解析量 1/10）
  vocab100k 全量事务但只保留前 10 万行出现过的项（位图 ∝ 截断词表 × 全量事务）
GPU 标定常数（同机硬件常数 0.0788544）预先注入缓存，避免 148 s 标定污染计时。
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.run_anyfim import run_anyfim  # noqa: E402

RES2 = os.path.join(ROOT, "results", "exp2")
FULL = os.path.join(RES2, "kosarak.txt")
OUT = os.path.join(RES2, "preprocess_diagnosis.json")
GPU_PCT = 0.0788544
K = 1


def make_variants():
    variants = {"full": FULL}
    lines = open(FULL, encoding="utf-8", errors="replace").read().splitlines()
    for name, n in (("quarter", 250_000), ("tenth", 100_000)):
        p = os.path.join(RES2, f"kosarak_{name}.txt")
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines[:n]) + "\n")
        variants[name] = p
    # vocab100k：只保留前 10 万行出现过的项
    p = os.path.join(RES2, "kosarak_vocab100k.txt")
    if not os.path.exists(p):
        vocab = set()
        for ln in lines[:100_000]:
            vocab.update(ln.split())
        kept = 0
        with open(p, "w", encoding="utf-8") as f:
            for ln in lines:
                items = [t for t in ln.split() if t in vocab]
                if items:
                    f.write(" ".join(items) + "\n")
                    kept += 1
        print(f"vocab100k: vocab={len(vocab)}, kept_tx={kept}")
    variants["vocab100k"] = p
    return variants


def seed_cache(path, k):
    cache_p = os.path.join(ROOT, "results", "gpu_pct_cache.json")
    cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
    key = hashlib.md5(open(path, "rb").read()).hexdigest()[:12] + f":{k}"
    cache[key] = GPU_PCT
    json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    if os.path.exists(OUT):
        print("already done:", OUT)
        print(open(OUT, encoding="utf-8").read())
        raise SystemExit
    variants = make_variants()
    rows_out = {}
    for name, path in variants.items():
        # 变体基础统计
        n_tx = 0
        vocab = set()
        t_parse0 = time.perf_counter()
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                toks = ln.split()
                n_tx += 1
                vocab.update(toks)
        py_parse_s = time.perf_counter() - t_parse0
        seed_cache(path, K)
        rows, _ = run_anyfim(path, K, dense=True)
        wall = rows[-1]["wall_total_s"]
        bitmap_gb = len(vocab) * n_tx / 8 / 1e9
        rows_out[name] = {"n_tx": n_tx, "n_items": len(vocab), "bitmap_gb": round(bitmap_gb, 3),
                          "python_parse_s": round(py_parse_s, 2), "wall_total_s": round(wall, 2)}
        print(f"{name:10s} tx={n_tx:7d} items={len(vocab):6d} bitmap={bitmap_gb:5.2f}GB "
              f"py_parse={py_parse_s:5.1f}s wall={wall:7.1f}s")
    json.dump(rows_out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("saved", OUT)
