# -*- coding: utf-8 -*-
"""驱动 AnyFIM anytime 引擎跑指定数据集的渐进激活流，输出逐轮 CSV。

用法:
    python drivers/run_anyfim.py <数据集路径> <轮数> [输出CSV]

不改引擎源码：build_anyfim.ensure_anyfim_built 在 build/ 目录出补丁版二进制；
运行时数据文件 staged 为 ..\\TransactionSets\\data.txt（引擎相对路径约定）。
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
    """返回 (rows, results_path)；rows 为逐轮 dict 列表。dense=False 为消融对照。"""
    exe = ensure_anyfim_built(dataset_path, upto_stage, dense=dense)
    exe_dir = os.path.dirname(exe)
    # 引擎把结果写到数据文件同目录；数据路径是相对 cwd 的 ..\\TransactionSets\\data.txt
    stage_dir = os.path.abspath(os.path.join(exe_dir, "..", "TransactionSets"))
    os.makedirs(stage_dir, exist_ok=True)
    shutil.copyfile(dataset_path, os.path.join(stage_dir, "data.txt"))
    shutil.copyfile(dataset_path, os.path.join(TXN_DIR, "data.txt"))

    # CPU/GPU 标定缓存：按 (数据集md5, 轮数) 记住标定值，经环境变量 ANYFIM_GPU_PCT 注入，
    # 跳过每次启动约 148s 的标定探测（标定是一次性硬件探测，不属于供给耗时）。
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

    # 引擎打印 "ANYFIM_GPU_PCT=<v>"（新标定）或 "GPUtaskPercentage (cached): <v>"（用缓存）
    out = r.stdout.decode("gbk", "replace")
    m = re.search(r"ANYFIM_GPU_PCT=([\d.eE+-]+)", out)
    if m and key not in cache:
        cache[key] = float(m.group(1))
        os.makedirs(os.path.dirname(cache_p), exist_ok=True)
        json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1)

    results = os.path.join(stage_dir, f"data.txt-{int(upto_stage)}-stages=Results.txt")
    if not os.path.exists(results):
        raise RuntimeError(f"引擎未产出结果文件：{results}\n" + out[-1500:])
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
