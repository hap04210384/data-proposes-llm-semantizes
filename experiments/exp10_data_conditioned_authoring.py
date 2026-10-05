# -*- coding: utf-8 -*-
"""Exp 10 data-conditioned authoring baseline (paper Exp. 5, Group A2).

Motivation (reviewer Major 3.2 of 2026-10-05): Group A authors rules with NO data
access -- a strawman. Group A2 conditions the LLM on the data the way a serious
practitioner would: it sees (i) the frequency table of the top-100 items and
(ii) 20 sample baskets drawn deterministically (seed 42). It must still author
50 association rules WITH estimated support and confidence.

Hypothesis under test: with data exposure the LLM stops fabricating item IDs,
but its self-reported numbers still cannot be certified, because exact
support/confidence require counting all 88,162 baskets -- the failure mode
shifts from fabrication to unverifiable counting.

Reproducibility: every prompt (including the shown statistics), the returned
model field, and timestamps persist under results/exp10/.
"""
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.llm_semantizer import _chat  # noqa: E402
from exp5_hallucination import load_transactions, exact_stats, parse_json_array  # noqa: E402

DATA = os.path.join(ROOT, "results", "exp1", "retail.txt")
OUT = os.path.join(ROOT, "results", "exp10")
N_RULES = 50
REPS = 3
TOP_ITEMS = 100
N_SAMPLE_BASKETS = 20

PROMPT_A2 = """You are a domain expert writing association rules for a grocery retail chain (shopping basket analysis).
Items are identified by numeric IDs. Here is what you know about the data:
(1) the {k} most frequent items with their occurrence counts:
{freq_table}
(2) {m} sample baskets (each line = one basket):
{baskets}

Based on this information, author {n} plausible association rules in the form
"antecedent -> consequent", each with an estimated support (fraction of all baskets)
and confidence. Return strict JSON:
[{{"antecedent": [...], "consequent": [...], "support": 0.0, "confidence": 0.0}}, ...]"""


def build_context(tx):
    from collections import Counter
    f1 = Counter()
    for t in tx:
        f1.update(t)
    top = f1.most_common(TOP_ITEMS)
    freq_table = "\n".join(f"  item {i}: {c} baskets" for i, c in top)
    rng = random.Random(42)
    baskets = "\n".join("  " + " ".join(map(str, sorted(rng.choice(tx)))) for _ in range(N_SAMPLE_BASKETS))
    return freq_table, baskets, f1


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tx = load_transactions(DATA)
    vocab = set().union(*tx)
    freq_table, baskets, f1 = build_context(tx)
    prompt = PROMPT_A2.format(k=TOP_ITEMS, freq_table=freq_table, m=N_SAMPLE_BASKETS,
                              baskets=baskets, n=N_RULES)
    print(f"prompt chars: {len(prompt)} (top-{TOP_ITEMS} freq table + {N_SAMPLE_BASKETS} baskets)")

    raw_path = os.path.join(OUT, "group_a2_raw.json")
    raws = json.load(open(raw_path, encoding="utf-8")) if os.path.exists(raw_path) else []
    while len(raws) < REPS:
        r = _chat(prompt)
        raws.append(r)
        json.dump(raws, open(raw_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  rep {len(raws)} done, model={r['model']}")

    per_rep = []
    for r in raws[:REPS]:
        rules = parse_json_array(r["content"])
        rep = {"model": r["model"], "ts": r["ts"], "n_returned": len(rules) if rules else None,
               "item_exist_rate": [], "support_abs_err": [], "conf_abs_err": [],
               "n_all_items_exist": 0, "n_total": 0}
        if rules:
            for rl in rules:
                try:
                    ant = [str(x) for x in rl.get("antecedent", [])]
                    con = [str(x) for x in rl.get("consequent", [])]
                    sup_llm = float(rl.get("support", 0))
                    conf_llm = float(rl.get("confidence", 0))
                except (TypeError, ValueError):
                    continue
                items = ant + con
                rep["n_total"] += 1
                exist = [1 if (i.isdigit() and int(i) in vocab) else 0 for i in items]
                rep["item_exist_rate"].append(sum(exist) / max(1, len(exist)))
                if len(exist) == sum(exist):
                    rep["n_all_items_exist"] += 1
                    # all items exist: recompute exact support/confidence from all baskets
                    sup_true, conf_true, cnt_ante = exact_stats(tx, tuple(sorted(set(int(i) for i in items))))
                else:
                    # any nonexistent item => the pattern never occurs => exact support = confidence = 0
                    sup_true, conf_true = 0.0, 0.0
                rep["support_abs_err"].append(abs(sup_llm - sup_true))
                rep["conf_abs_err"].append(abs(conf_llm - conf_true))
        per_rep.append(rep)

    def med(xs):
        return round(statistics.median(xs), 4) if xs else None

    summary = {
        "dataset": "retail", "n_tx": len(tx), "n_vocab": len(vocab),
        "conditioning": f"top-{TOP_ITEMS} item frequency table + {N_SAMPLE_BASKETS} sample baskets (seed 42)",
        "n_rules": N_RULES, "reps": REPS,
        "group_a2": {
            "model": [r["model"] for r in per_rep],
            "n_returned_median": med([r["n_returned"] or 0 for r in per_rep]),
            "item_exist_rate_median": med([statistics.mean(r["item_exist_rate"]) for r in per_rep if r["item_exist_rate"]]),
            "rules_all_items_exist_median": med([r["n_all_items_exist"] for r in per_rep]),
            "support_abs_err_median": med([statistics.mean(r["support_abs_err"]) for r in per_rep if r["support_abs_err"]]),
            "support_abs_err_max": med([max(r["support_abs_err"]) for r in per_rep if r["support_abs_err"]]),
            "conf_abs_err_median": med([statistics.mean(r["conf_abs_err"]) for r in per_rep if r["conf_abs_err"]]),
            "conf_abs_err_max": med([max(r["conf_abs_err"]) for r in per_rep if r["conf_abs_err"]]),
        },
    }
    json.dump({"summary": summary, "group_a2_detail": per_rep},
              open(os.path.join(OUT, "exp10_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
