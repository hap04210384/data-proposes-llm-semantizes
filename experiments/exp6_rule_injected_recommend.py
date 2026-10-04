# -*- coding: utf-8 -*-
"""Exp 6 downstream-consumption experiment: rule-injected next-item recommendation.

Task: given a basket prefix {A}, ask the LLM for the most likely next item
(retail, items are numeric IDs).
  Group A (baseline): no rules supplied, the LLM answers from "general knowledge";
  Group B (supply): the engine's top-10 exact rules (with support/confidence) are
     injected before answering.
Ground truth: argmax_B conf(A->B) computed exactly from the data (recomputable).
Metrics: hit@1 (answered item ID exactly equals ground truth), valid-answer rate
(ID exists in the vocabulary).
prompt/temperature/model/timestamps are persisted automatically by the adapter
layer (results/exp6/raw/).
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.llm_semantizer import _chat  # noqa: E402

DATA = os.path.join(ROOT, "results", "exp1", "retail.txt")
OUT = os.path.join(ROOT, "results", "exp6")
N_PROBES = 50
TOP_RULES = 10

PROMPT_BASE = """You are assisting a grocery retail analyst. Items are identified by numeric IDs.
A customer has bought item {a}. Which single item are they most likely to buy next?
Answer with one integer item ID only, no explanation."""

PROMPT_RULES = """You are assisting a grocery retail analyst. Items are identified by numeric IDs.
A customer has bought item {a}. Which single item are they most likely to buy next?
Here are exact association rules mined from the full transaction data (support; confidence):
{rules}
Answer with one integer item ID only, no explanation."""


def build_probes():
    tx = [set(map(int, ln.split())) for ln in open(DATA) if ln.split()]
    n = len(tx)
    f1 = Counter()
    pair = Counter()
    ante = Counter()
    for t in tx:
        for i in t:
            f1[i] += 1
        for a in t:
            ante[a] += 1
            for b in t:
                if a != b:
                    pair[(a, b)] += 1
    # one pass over the pair table for each antecedent's best consequent (O(|pairs|))
    best = {}
    for (a, b), c in pair.items():
        conf = c / ante[a]
        if a not in best or conf > best[a][1]:
            best[a] = (b, conf)
    # probes: antecedents in the top 200 by frequency whose best confidence >= 0.03; take 50
    probes = []
    for a, _ in f1.most_common(200):
        if a in best and best[a][1] >= 0.03:
            probes.append({"ante": a, "truth": best[a][0], "conf": round(best[a][1], 4)})
        if len(probes) >= N_PROBES:
            break
    # inject the per-antecedent top-10 rules (by confidence) for each probe
    per_ante = {}
    for (a, b), c in pair.most_common():
        conf = c / ante[a]
        if conf < 0.02:
            break
        per_ante.setdefault(a, []).append(f"  {a} -> {b} (support {c/n:.4f}; confidence {conf:.3f})")
    return probes, per_ante


def top_rules_for(per_ante, a, top=TOP_RULES):
    return "\n".join(per_ante.get(a, [])[:top])


def parse_id(content):
    m = re.search(r"\d+", content)
    return int(m.group(0)) if m else None


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    probes, per_ante = build_probes()
    vocab = set()
    for ln in open(DATA):
        vocab.update(map(int, ln.split()))
    print(f"{len(probes)} probes; per-antecedent top-{TOP_RULES} rules for injection")

    results = []
    for i, p in enumerate(probes):
        row = {"probe": i, "ante": p["ante"], "truth": p["truth"], "conf": p["conf"]}
        rules_text = top_rules_for(per_ante, p["ante"]) or "(no mined rules for this item)"
        for grp, prompt in (("A", PROMPT_BASE.format(a=p["ante"])),
                            ("B", PROMPT_RULES.format(a=p["ante"], rules=rules_text))):
            cache = os.path.join(OUT, "raw", f"p{i}_{grp}.json")
            if os.path.exists(cache):
                r = json.load(open(cache, encoding="utf-8"))
            else:
                r = _chat(prompt)
                json.dump(r, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
            ans = parse_id(r["content"])
            row[f"{grp}_model"] = r["model"]
            row[f"{grp}_answer"] = ans
            row[f"{grp}_valid"] = ans in vocab
            row[f"{grp}_hit"] = ans == p["truth"]
        results.append(row)
        print(f"  probe{i} ante={p['ante']} truth={p['truth']} | A:{row['A_answer']}({row['A_valid']}) B:{row['B_answer']} hitA={row['A_hit']} hitB={row['B_hit']}")

    def rate(k):
        vals = [r[k] for r in results]
        return round(sum(vals) / len(vals), 4)

    summary = {"n_probes": len(results),
               "group_A": {"hit_at_1": rate("A_hit"), "valid_id_rate": rate("A_valid"),
                           "model": results[0]["A_model"]},
               "group_B": {"hit_at_1": rate("B_hit"), "valid_id_rate": rate("B_valid"),
                           "model": results[0]["B_model"]}}
    json.dump({"summary": summary, "detail": results},
              open(os.path.join(OUT, "exp6_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=2))
