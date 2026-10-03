# -*- coding: utf-8 -*-
"""实验 1 基线：传统阈值挖掘的网格扫荡（TensorFIM 位图基线引擎）。

对给定数据集在阈值网格上逐档完整重跑，记录每档墙钟时间/引擎计算时间/MFI 数。
这正是论文要攻击的供给方式："不知道阈值 → 网格搜索 → 每档从头重跑"。

用法:
    python exp1/run_baseline_grid.py <数据集路径> <输出CSV> [阈值逗号列表]

说明:
- 引擎: ../engines/TensorFIM/code/engine/run/CoParaCG_baseline.exe（预编译，传统位图 MFI 挖掘）
- 引擎"退出码 1 = 成功"是历史约定，以输出文件存在为准；
- 首个阈值自动做两次（预热 + 计时），预热结果丢弃——引擎首次启动有 GPU 自检开销。
"""
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(ROOT)
BASELINE_EXE = os.path.join(WORKSPACE, "engines", "TensorFIM", "code", "engine", "run",
                           "CoParaCG_baseline.exe")
COMP = re.compile(r"computation_time\(s\):\s*([\d.]+)")


def run_one(dataset, thr, timeout=1800):
    """单档运行；返回 dict(wall_s, computation_s, n_mfi, ok)。"""
    out_file = f"{dataset}-{thr:.6f}=Results.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    t0 = time.perf_counter()
    try:
        r = subprocess.run([BASELINE_EXE, dataset, str(thr)], capture_output=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"wall_s": "", "computation_s": "", "n_mfi": "", "ok": "timeout"}
    wall = time.perf_counter() - t0
    log = r.stdout.decode("gbk", "replace")
    if not os.path.exists(out_file):
        return {"wall_s": round(wall, 4), "computation_s": "", "n_mfi": "",
                "ok": f"exit={r.returncode}"}
    comp = COMP.findall(log)
    n = sum(1 for ln in open(out_file, encoding="utf-8", errors="replace")
            if re.match(r"^\d+th:", ln.strip()))
    return {"wall_s": round(wall, 4),
            "computation_s": float(comp[-1]) if comp else "",
            "n_mfi": n, "ok": "ok"}


def main():
    ds = sys.argv[1]
    out = sys.argv[2]
    if len(sys.argv) > 3:
        grid = [float(x) for x in sys.argv[3].split(",")]
    else:
        name = os.path.basename(ds)
        if "chess" in name:                       # 稠密：从极高支持度往下扫
            grid = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.85, 0.80]
        elif "retail" in name:                    # 稀疏：低支持度区
            grid = [0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002]
        else:
            grid = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    assert os.path.exists(BASELINE_EXE), f"基线引擎不存在：{BASELINE_EXE}"
    # 断点续跑：已完成的阈值跳过
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8-sig") as f:
            for rec in csv.DictReader(f):
                if rec.get("ok") == "ok":
                    done.add(float(rec["threshold"]))
    # 预热（丢弃），消除首次 GPU 自检开销
    if grid[0] not in done:
        print(f"[warmup] {ds} @ {grid[0]}")
        run_one(ds, grid[0])

    rows = []
    for thr in grid:
        if thr in done:
            print(f"[skip] {os.path.basename(ds)} @ {thr}（已完成）")
            continue
        print(f"[run] {os.path.basename(ds)} @ {thr}")
        rec = run_one(ds, thr)
        rec.update({"dataset": os.path.basename(ds), "threshold": thr})
        rows.append(rec)
        print("   ", rec)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    exists = os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "threshold", "wall_s",
                                          "computation_s", "n_mfi", "ok"])
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print("CSV:", out)


if __name__ == "__main__":
    main()
