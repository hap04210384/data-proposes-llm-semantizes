# -*- coding: utf-8 -*-
"""Exp5 幻觉对照实验（论文 Exp.5）。

对照设计（retail 数据集，N=50 条规则/项集，temperature=0，3 次重复取中位）：
  A 组（幻觉基线）：DeepSeek 凭空 author 规则并自报 support/confidence。
     指标：项存在率、support 绝对偏差、confidence 绝对偏差、完全编造率。
  B 组（本系统）：引擎供给 top-50 MFI -> 派生规则（引擎 support + 我们从
     原始事务独立复算的 confidence）-> DeepSeek 仅语义化。
     指标：引擎 support 与我们复算值的 parity、语义化输出是否出现任何
     数字改写（必须为 0 才符合构造性保证）。

复现性：prompt 全文、模型返回的 model 字段、时间戳全部落盘 results/exp5/。
"""
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drivers.llm_semantizer import llm_author_rules, semantize_rules  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "results", "exp1", "retail.txt")
MFI_FILE = os.path.join(ROOT, "results", "exp1", "retail.txt-0.007000=Results.txt")
OUT = os.path.join(ROOT, "results", "exp5")
N_RULES = 50
REPS = 3


def load_transactions(path):
    tx = []
    with open(path) as f:
        for line in f:
            items = line.split()
            if items:
                tx.append(set(map(int, items)))
    return tx


def load_mfis(path):
    """解析引擎结果文件中的 MFI 行: 'kth: n { i j k }  support: s  frequency: f'"""
    mfis = []
    pat = re.compile(r"^\d+th:\s+\d+\s+\{\s*([\d\s]+?)\s*\}\s+support:\s+([\d.]+)\s+frequency:\s+(\d+)")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                items = tuple(sorted(map(int, m.group(1).split())))
                mfis.append({"items": items,
                             "support_engine": float(m.group(2)),
                             "freq": int(m.group(3))})
    return mfis


def exact_stats(tx, items):
    """从事务集独立复算 support 与 confidence（规则 = items[:-1] -> items[-1]）。"""
    n = len(tx)
    ante = set(items[:-1])
    full = set(items)
    cnt_ante = sum(1 for t in tx if ante <= t)
    cnt_full = sum(1 for t in tx if full <= t)
    sup = cnt_full / n
    conf = (cnt_full / cnt_ante) if cnt_ante else 0.0
    return sup, conf, cnt_ante


def parse_json_array(text):
    """从 LLM 输出中提取第一个 JSON 数组（容忍 markdown 代码围栏）。"""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_group_a(vocab, n_rules, reps):
    """A 组：LLM 凭空 author 规则。domain 不给数据访问，只描述任务背景。"""
    raw_path = os.path.join(OUT, "group_a_raw.json")
    raws = []
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            raws = json.load(f)
    while len(raws) < reps:
        r = llm_author_rules("a grocery retail chain (shopping basket analysis)", n=n_rules)
        raws.append(r)
        with open(raw_path, "w") as f:
            json.dump(raws, f, ensure_ascii=False, indent=1)
    per_rep = []
    for r in raws[:reps]:
        rules = parse_json_array(r["content"])
        rep = {"model": r["model"], "ts": r["ts"], "n_parsed": 0, "n_returned": None,
               "item_exist_rate": [], "support_abs_err": [], "conf_abs_err": [],
               "n_total_rules": 0, "n_all_items_exist": 0}
        if rules is None:
            per_rep.append(rep)
            continue
        rep["n_returned"] = len(rules)
        for rl in rules:
            try:
                ant = [str(x) for x in rl.get("antecedent", [])]
                con = [str(x) for x in rl.get("consequent", [])]
                sup_llm = float(rl.get("support", 0))
                conf_llm = float(rl.get("confidence", 0))
            except (TypeError, ValueError):
                continue
            rep["n_total_rules"] += 1
            items = ant + con
            exist = [1 if (i.isdigit() and int(i) in vocab) else 0 for i in items]
            rep["item_exist_rate"].append(sum(exist) / max(1, len(exist)))
            if len(exist) == sum(exist):
                rep["n_all_items_exist"] += 1
            # 只要存在不存在的项，真实 support/confidence 即为 0，偏差=LLM 自报值
            if sum(exist) == len(exist) and len(items) >= 2:
                sup_true, conf_true, _ = exact_stats(tx_global, tuple(sorted(set(int(i) for i in items))))
            else:
                sup_true, conf_true = 0.0, 0.0
            rep["support_abs_err"].append(abs(sup_llm - sup_true))
            rep["conf_abs_err"].append(abs(conf_llm - conf_true))
        per_rep.append(rep)
    return per_rep


def run_group_b(tx, mfis, n_rules, reps):
    """B 组：引擎供给 + 独立复算 + LLM 仅语义化。"""
    vocab_rules = [m for m in mfis if len(m["items"]) >= 2][:n_rules]
    n = len(tx)
    rules_text_rows = []
    exact_rows = []
    for i, m in enumerate(vocab_rules):
        items = m["items"]
        sup_exact, conf_exact, cnt_ante = exact_stats(tx, items)
        exact_rows.append({"id": f"R{i+1}", "items": list(items),
                           "support_engine": m["support_engine"],
                           "support_exact": sup_exact, "conf_exact": conf_exact,
                           "freq_engine": m["freq"], "n_tx": n})
        ant = ", ".join(map(str, items[:-1]))
        rules_text_rows.append(
            f"R{i+1}: {{{ant}}} -> {{{items[-1]}}}; support {sup_exact:.6f}; confidence {conf_exact:.4f}")
    raw_path = os.path.join(OUT, "group_b_raw.json")
    raws = []
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            raws = json.load(f)
    while len(raws) < reps:
        r = semantize_rules("\n".join(rules_text_rows))
        raws.append(r)
        with open(raw_path, "w") as f:
            json.dump(raws, f, ensure_ascii=False, indent=1)
    per_rep = []
    for r in raws[:reps]:
        parsed = parse_json_array(r["content"])
        n_named = len(parsed) if isinstance(parsed, list) else 0
        # 数字保真检查：语义化输出中不得出现 support/confidence 字段或改写数字
        num_violations = 0
        if isinstance(parsed, list):
            for p in parsed:
                if not isinstance(p, dict):
                    num_violations += 1
                    continue
                for banned in ("support", "confidence", "sup", "conf"):
                    if banned in {k.lower() for k in p.keys()}:
                        num_violations += 1
        per_rep.append({"model": r["model"], "ts": r["ts"], "n_rules_in": len(vocab_rules),
                        "n_named": n_named, "num_violations": num_violations})
    parity = [abs(e["support_engine"] - e["support_exact"]) for e in exact_rows]
    return {"per_rep": per_rep, "exact": exact_rows,
            "parity_max": max(parity) if parity else None,
            "parity_median": statistics.median(parity) if parity else None}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("[1/3] 加载 retail 事务集与引擎 MFI ...")
    tx_global = load_transactions(DATA)
    vocab = set().union(*tx_global)
    mfis = load_mfis(MFI_FILE)
    print(f"      {len(tx_global)} 事务, {len(vocab)} 项, {len(mfis)} MFI (>=2 项: {sum(1 for m in mfis if len(m['items'])>=2)})")

    print("[2/3] A 组: LLM 凭空 author (3 次) ...")
    group_a = run_group_a(vocab, N_RULES, REPS)
    print("[3/3] B 组: 引擎供给 + 语义化 (3 次) ...")
    group_b = run_group_b(tx_global, mfis, N_RULES, REPS)

    def med(xs):
        return round(statistics.median(xs), 4) if xs else None

    summary = {
        "dataset": "retail", "n_tx": len(tx_global), "n_vocab": len(vocab),
        "n_rules": N_RULES, "reps": REPS,
        "group_a": {
            "model": [r["model"] for r in group_a],
            "n_returned_median": med([r["n_returned"] or 0 for r in group_a]),
            "item_exist_rate_median": med([statistics.mean(r["item_exist_rate"]) for r in group_a if r["item_exist_rate"]]),
            "rules_all_items_exist": med([r["n_all_items_exist"] for r in group_a]),
            "support_abs_err_median": med([statistics.mean(r["support_abs_err"]) for r in group_a if r["support_abs_err"]]),
            "support_abs_err_max": med([max(r["support_abs_err"]) for r in group_a if r["support_abs_err"]]),
            "conf_abs_err_median": med([statistics.mean(r["conf_abs_err"]) for r in group_a if r["conf_abs_err"]]),
        },
        "group_b": {
            "model": [r["model"] for r in group_b["per_rep"]],
            "n_rules_supplied": group_b["per_rep"][0]["n_rules_in"],
            "n_named_median": med([r["n_named"] for r in group_b["per_rep"]]),
            "num_violations_total": sum(r["num_violations"] for r in group_b["per_rep"]),
            "support_parity_max_abs_diff": group_b["parity_max"],
            "support_parity_median_abs_diff": group_b["parity_median"],
        },
    }
    with open(os.path.join(OUT, "exp5_summary.json"), "w") as f:
        json.dump({"summary": summary, "group_a_detail": group_a,
                   "group_b_detail": group_b["per_rep"],
                   "group_b_exact": group_b["exact"]}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
