# -*- coding: utf-8 -*-
"""Exp 8 expanded downstream-consumption study (response to Reviewer Major 3).

Extends Exp 6 in two directions:
  scale: every qualifying anchor in the top-200 frequent items (no 50-probe cap)
         plus 50 cold anchors with no qualifying rules;
  metrics: ranked top-10 lists -> hit@1 / hit@5 / hit@10 and NDCG@5 / NDCG@10,
           with the global-popularity baseline computed locally (no API).
Groups:
  A (cold): the LLM answers from general knowledge, no rules supplied;
  B (supply): the engine's top-10 exact rules are injected with support/confidence.
Ground truth (rule-backed probes only): argmax-confidence consequent, exactly
recomputed from the data. Cold anchors have no ground truth by design; we report
abstention/valid-ID behavior for them instead.
Raw prompts/answers are cached under results/exp8/raw/ (one JSON per call),
so the script is resumable and idempotent. Run with --max-calls to stay within
time budgets; re-run until "calls made: 0".
"""
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.llm_semantizer import _chat  # noqa: E402

DATA = os.path.join(ROOT, "results", "exp1", "retail.txt")
OUT = os.path.join(ROOT, "results", "exp8")
TOP_RULES = 10
LIST_LEN = 10
N_COLD = 50
MIN_BEST_CONF = 0.03   # same criterion as Exp 6
RULE_MIN_CONF = 0.02   # same injection pool as Exp 6

PROMPT_COLD = """You are assisting a grocery retail analyst. Items are identified by numeric IDs.
A customer has bought item {a}. Which {k} items are they most likely to buy next, in order of likelihood?
Answer with {k} integer item IDs, most likely first, separated by commas or one per line. No explanation."""

PROMPT_RULES = """You are assisting a grocery retail analyst. Items are identified by numeric IDs.
A customer has bought item {a}. Which {k} items are they most likely to buy next, in order of likelihood?
Here are exact association rules mined from the full transaction data (support; confidence):
{rules}
Base your ranking on these rules. Answer with {k} integer item IDs, most likely first, separated by commas or one per line. No explanation."""

PROMPT_COLD_NORULES = """You are assisting a grocery retail analyst. Items are identified by numeric IDs.
A customer has bought item {a}. No association rules could be mined for this item in the data.
Given that, which {k} items are they most likely to buy next, in order of likelihood, based on your general knowledge?
Answer with {k} integer item IDs, most likely first, separated by commas or one per line. No explanation."""


def build_probes():
    tx = [set(map(int, ln.split())) for ln in open(DATA) if ln.split()]
    n = len(tx)
    f1, pair, ante = Counter(), Counter(), Counter()
    for t in tx:
        for a in t:
            f1[a] += 1
            ante[a] += 1
            for b in t:
                if a != b:
                    pair[(a, b)] += 1
    best = {}
    for (a, b), c in pair.items():
        conf = c / ante[a]
        if a not in best or conf > best[a][1]:
            best[a] = (b, conf)
    backed, cold = [], []
    for a, fa in f1.most_common():
        if a in best and best[a][1] >= MIN_BEST_CONF and len(backed) < 200:
            backed.append({"ante": a, "truth": best[a][0], "conf": round(best[a][1], 4)})
        elif a in best and best[a][1] < RULE_MIN_CONF and fa >= 20 and len(cold) < N_COLD:
            cold.append({"ante": a})  # frequent enough to probe, but the rule pool is empty here
    per_ante = {}
    for (a, b), c in pair.items():
        conf = c / ante[a]
        if conf < RULE_MIN_CONF:
            continue
        per_ante.setdefault(a, []).append((b, c, conf))
    for a in per_ante:  # keep the strongest rules first, as in Exp 6
        per_ante[a].sort(key=lambda x: -x[2])
    pop = [b for b, _ in f1.most_common(LIST_LEN)]
    return backed, cold, per_ante, pop, n


def rules_text(per_ante, a):
    lines = [f"  {a} -> {b} (support {c/n_tx:.4f}; confidence {conf:.3f})"
             for b, c, conf in per_ante.get(a, [])[:TOP_RULES]]
    return "\n".join(lines) or "(no mined rules for this item)"


def parse_ranked(content, k=LIST_LEN):
    ids = []
    for tok in re.findall(r"\d+", content):
        v = int(tok)
        if v not in ids:
            ids.append(v)
        if len(ids) >= k:
            break
    return ids if ids else None


def metrics(ranked_lists, truths):
    """ranked_lists: list of ordered ID lists (or None=refusal); truths: parallel list."""
    n = len(truths)
    res = {"n": n, "refusals": sum(1 for r in ranked_lists if not r)}
    scored = [(r, t) for r, t in zip(ranked_lists, truths) if r]
    for k in (1, 5, 10):
        # strict: a refusal counts as a miss (denominator = all probes)
        res[f"hit@{k}"] = round(sum(1 for r, t in scored if t in r[:k]) / n, 4) if n else None
    for k in (5, 10):
        dcgs = []
        for r, t in zip(ranked_lists, truths):
            if r and t in r[:k]:
                rank = r.index(t) + 1
                dcgs.append(1.0 / math.log2(rank + 1))
            else:
                dcgs.append(0.0)
        res[f"ndcg@{k}"] = round(sum(dcgs) / n, 4) if n else None
    return res


if __name__ == "__main__":
    max_calls = int(sys.argv[sys.argv.index("--max-calls") + 1]) if "--max-calls" in sys.argv else 10**9
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    backed, cold, per_ante, pop, n_tx = build_probes()
    vocab = set()
    for ln in open(DATA):
        vocab.update(map(int, ln.split()))
    print(f"rule-backed probes: {len(backed)}; cold probes: {len(cold)}; model deepseek-chat, T=0")

    jobs = []
    for i, p in enumerate(backed):
        jobs.append((f"b{i}_A", PROMPT_COLD.format(a=p["ante"], k=LIST_LEN)))
        jobs.append((f"b{i}_B", PROMPT_RULES.format(a=p["ante"], k=LIST_LEN, rules=rules_text(per_ante, p["ante"]))))
    for i, p in enumerate(cold):
        jobs.append((f"c{i}_A", PROMPT_COLD.format(a=p["ante"], k=LIST_LEN)))
        jobs.append((f"c{i}_B", PROMPT_COLD_NORULES.format(a=p["ante"], k=LIST_LEN)))

    calls = 0
    for name, prompt in jobs:
        cache = os.path.join(OUT, "raw", name + ".json")
        if os.path.exists(cache) or calls >= max_calls:
            continue
        r = _chat(prompt)
        json.dump(r, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
        calls += 1
        print(f"  call {calls}: {name} -> {r['content'][:60]!r}")
    print(f"calls made: {calls}")

    if calls:
        sys.exit(0)  # more to do; re-run

    # ---------------- everything cached: score ----------------
    def load(name):
        return json.load(open(os.path.join(OUT, "raw", name + ".json"), encoding="utf-8"))

    A_lists, B_lists, truths = [], [], []
    for i, p in enumerate(backed):
        A_lists.append(parse_ranked(load(f"b{i}_A")["content"]))
        B_lists.append(parse_ranked(load(f"b{i}_B")["content"]))
        truths.append(p["truth"])
    mA = metrics(A_lists, truths)
    mB = metrics(B_lists, truths)
    mP = metrics([pop] * len(truths), truths)

    cold_stats = {"A": {"refusals": 0, "valid_lists": 0}, "B": {"refusals": 0, "valid_lists": 0}}
    for i, p in enumerate(cold):
        for g in ("A", "B"):
            r = parse_ranked(load(f"c{i}_{g}")["content"])
            if not r:
                cold_stats[g]["refusals"] += 1
            elif all(x in vocab for x in r):
                cold_stats[g]["valid_lists"] += 1

    summary = {
        "n_rule_backed": len(backed), "n_cold": len(cold),
        "popularity_baseline": mP, "group_A_cold_llm": mA, "group_B_rule_injected_llm": mB,
        "cold_anchor_behavior": cold_stats,
        "model": "deepseek-chat (DeepSeek-V3 family), temperature 0",
    }
    json.dump({"summary": summary, "backed": backed, "cold": cold},
              open(os.path.join(OUT, "exp8_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=2))
