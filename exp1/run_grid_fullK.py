# -*- coding: utf-8 -*-
"""实验 1 补测：网格基线在全部 K 个供给阈值上逐个全量重跑（与 anytime 等量交付对照）。

背景：dual_caliber 只跑了去重后的 12/9 个阈值，手稿 Fig.2 与 III-B 的
10.8x / 6.6x 用的是"单档墙钟中位 x K"线性模型。本脚本按 anytime 实际交付的
全部 K 个阈值（含重复档）逐一实测网格成本，5 次重复取中位，彻底锚定手稿数字。

产出 results/exp1/fullK_grid_<name>.csv:
    rep, run_idx, threshold, wall_s, engine_s, cum_wall_s, cum_engine_s
"""
import csv
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
from run_reps import BASELINE_EXE  # noqa: E402

COMP = re.compile(r"^computation_time\(s\):\s*(\S+)", re.M)

STAGE_THRESHOLDS = {
    "chess": [0.999687, 0.996558, 0.995307, 0.991865, 0.985294, 0.969650,
              0.962453, 0.957447, 0.951189, 0.945244, 0.945244, 0.942741,
              0.929599, 0.899249, 0.894869, 0.888298, 0.849186, 0.823217,
              0.817272, 0.799750],
    "retail": [0.574794, 0.477927, 0.176902, 0.172036, 0.169517, 0.050725,
               0.043522, 0.036943, 0.035151, 0.034391, 0.033302, 0.031692,
               0.029423, 0.025374, 0.024580, 0.023752, 0.021324, 0.021132,
               0.020258, 0.020179, 0.019668, 0.019453, 0.018148, 0.016889,
               0.016867, 0.016719, 0.016640, 0.016175, 0.015608, 0.015562,
               0.014927, 0.014768, 0.014644, 0.014451, 0.013918, 0.013725,
               0.013713, 0.013657, 0.013418, 0.013396, 0.013226, 0.013169,
               0.013158, 0.012965, 0.012897, 0.012851, 0.012704, 0.012693,
               0.012590, 0.012500],
}


def one_run(ds_path, thr):
    out_file = f"{ds_path}-{thr:.6f}=Results.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    t0 = time.perf_counter()
    subprocess.run([BASELINE_EXE, ds_path, f"{thr:.6f}"],
                   capture_output=True, timeout=1800)
    wall = time.perf_counter() - t0
    m = COMP.search(open(out_file, encoding="utf-8", errors="replace").read())
    return wall, float(m.group(1))


def main():
    name, n_rep = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ths = STAGE_THRESHOLDS[name]
    K = len(ths)
    ds = os.path.abspath(os.path.join(RES, f"{name}.txt"))

    one_run(ds, ths[0])  # 预热（不计入）

    out = os.path.join(RES, f"fullK_grid_{name}.csv")
    recs = []
    for rep in range(1, n_rep + 1):
        cw = ce = 0.0
        for i, thr in enumerate(ths, 1):
            w, e = one_run(ds, thr)
            cw += w
            ce += e
            recs.append((rep, i, f"{thr:.6f}", round(w, 4), e,
                         round(cw, 4), round(ce, 6)))
        print(f"[{name} rep {rep}/{n_rep}] K={K} cum_wall={cw:.4f}s "
              f"cum_engine={ce:.6f}s", flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["rep", "run_idx", "threshold", "wall_s", "engine_s",
                     "cum_wall_s", "cum_engine_s"])
        wr.writerows(recs)

    # 汇总：各 run_idx 的墙钟/引擎中位，累计后与 anytime 中位对比
    med_wall, med_engine = [], []
    for i in range(1, K + 1):
        ws = [r[3] for r in recs if r[1] == i]
        es = [r[4] for r in recs if r[1] == i]
        med_wall.append(statistics.median(ws))
        med_engine.append(statistics.median(es))
    cum_w = sum(med_wall)
    cum_e = sum(med_engine)
    print(f"[{name}] median grid over K={K}: cum_wall={cum_w:.4f}s "
          f"cum_engine={cum_e:.6f}s")
    for f in (f"anyfim_{name}_K{K}_median5.csv",):
        rows = list(csv.DictReader(open(os.path.join(RES, f),
                                        encoding="utf-8-sig")))
        aw = float(rows[-1]["wall_total_s"])
        ae = float(rows[-1]["cum_engine_s"])
        print(f"[{name}] anytime median: wall={aw:.4f}s engine={ae:.6f}s "
              f"-> wall speedup={cum_w / aw:.2f}x, "
              f"engine ratio grid/anytime={cum_e / ae:.2f}x")


if __name__ == "__main__":
    main()
