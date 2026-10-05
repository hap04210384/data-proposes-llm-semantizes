# -*- coding: utf-8 -*-
"""Exp 9 holdout-split downstream evaluation (response to Reviewer Major 3.1 of
2026-10-05: the Exp 6/8 ground truth is derived from the supplied rules, making
the comparison circular).

Design:
  * retail transactions are split deterministically (seed 42) into halves A and B.
  * Consumers see ONLY half A: the injected rules (top-10 per anchor by confidence
    on A, support/confidence computed exactly on A), the popularity ranking, and
    the item-item co-occurrence ranking are all computed on A.
  * Ground truth is derived from half B ALONE: the argmax-confidence consequent
    of the anchor under B's counts. No consumer ever sees B, so a correct answer
    now evidences generalization, not construction.
  * Consumers: (1) cold LLM; (2) LLM + A-rules; (3) symbolic argmax-confidence
    selector over the injected A-rules (no LLM -- isolates the LLM interface
    contribution); (4) popularity top-10 (A counts); (5) item-item co-occurrence
    top-10 (A pair counts).
  * Metrics: hit@1/5/10, NDCG@5/10, refusals; McNemar exact test between the
    LLM+rules consumer and popularity / symbolic consumers on hit@1.

Reproducibility: raw prompts/answers cached under results/exp9/raw/.
Run with --max-calls N in batches; re-run until "calls made: 0".
"""
import json
import math
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drivers.llm_semantizer import _chat  # noqa: E402

DATA = os.path.join(ROOT, "results", "exp1", "retail.txt")
OUT = os.path.join(ROOT, "results", "exp9")
SEED = 42
TOP_RULES = 10
LIST_LEN = 10
N_PROBES = 200
RULE_MIN_CONF = 0.02

PROMPT_COLD = """You are assisting a grocery retail analyst. Items are identified by numeric IDs.
A customer has bought item {a}. Which {k} items are they most likely to buy next, in order of likelihood?
Answer with {k} integer item IDs, most likely first, separated by commas or one per line. No explanation."""

PROMPT_RULES = """You are assisting a grocery retail analyst. Items are identified by numeric IDs.
A customer has bought item {a}. Which {k} items are they most likely to buy next, in order of likelihood?
Here are exact association rules mined from past transaction data (support; confidence):
{rules}
Base your ranking on these rules. Answer with {k} integer item IDs, most likely first, separated by commas or one per line. No explanation."""


def load_tx(path):
    return [set(map(int, ln.split())) for ln in open(path) if ln.split()]


def counters(tx):
    f1, pair, ante = Counter(), Counter(), Counter()
    for t in tx:
        for a in t:
            f1[a] += 1
            ante[a] += 1
            for b in t:
                if a != b:
                    pair[(a, b)] += 1
    return f1, pair, ante


def best_map(pair, ante):
    best = {}
    for (a, b), c in pair.items():
        conf = c / ante[a]
        if a not in best or conf > best[a][1]:
            best[a] = (b, conf)
    return best


def per_ante_rules(pair, ante, n_tx):
    per = {}
    for (a, b), c in pair.items():
        conf = c / ante[a]
        if conf < RULE_MIN_CONF:
            continue
        per.setdefault(a, []).append((b, c, conf, c / n_tx))
    for a in per:
        per[a].sort(key=lambda x: -x[2])
    return per


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
    n = len(truths)
    res = {"n": n, "refusals": sum(1 for r in ranked_lists if not r)}
    scored = [(r, t) for r, t in zip(ranked_lists, truths) if r]
    for k in (1, 5, 10):
        res[f"hit@{k}"] = round(sum(1 for r, t in scored if t in r[:k]) / n, 4)
    for k in (5, 10):
        dcgs = []
        for r, t in zip(ranked_lists, truths):
            if r and t in r[:k]:
                dcgs.append(1.0 / math.log2(r.index(t) + 2))
            else:
                dcgs.append(0.0)
        res[f"ndcg@{k}"] = round(sum(dcgs) / n, 4)
    return res


def mcnemar_exact(va, vb):
    """va, vb: boolean vectors. Exact two-sided binomial test on discordant pairs."""
    n10 = sum(1 for a, b in zip(va, vb) if a and not b)
    n01 = sum(1 for a, b in zip(va, vb) if b and not a)
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "p": 1.0}
    from math import comb
    p = sum(comb(n, k) for k in range(0, min(n10, n01) + 1)) / 2 ** n
    return {"n10": n10, "n01": n01, "p": round(p, 6)}


if __name__ == "__main__":
    max_calls = int(sys.argv[sys.argv.index("--max-calls") + 1]) if "--max-calls" in sys.argv else 10 ** 9
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)

    tx = load_tx(DATA)
    n = len(tx)
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    half = n // 2
    txA = [tx[i] for i in idx[:half]]
    txB = [tx[i] for i in idx[half:]]
    json.dump({"seed": SEED, "n_tx": n, "half_A": idx[:half], "half_B": idx[half:]},
              open(os.path.join(OUT, "split.json"), "w"))

    f1A, pairA, anteA = counters(txA)
    f1B, pairB, anteB = counters(txB)
    bestA, bestB = best_map(pairA, anteA), best_map(pairB, anteB)
    perA = per_ante_rules(pairA, anteA, len(txA))

    probes = []
    for a, _ in f1A.most_common(400):
        if a in bestB and len(probes) < N_PROBES:
            probes.append({"ante": a,
                           "truth_B": bestB[a][0], "conf_B": round(bestB[a][1], 4),
                           "truth_A": bestA[a][0] if a in bestA else None,
                           "conf_A": round(bestA[a][1], 4) if a in bestA else None})
    agree = sum(1 for p in probes if p["truth_A"] == p["truth_B"])
    print(f"split seed {SEED}: |A|={len(txA)} |B|={len(txB)}; probes={len(probes)}; "
          f"A-argmax == B-argmax on {agree}/{len(probes)} anchors (split instability ceiling)")

    def rules_text(a):
        lines = [f"  {a} -> {b} (support {s:.4f}; confidence {c:.3f})"
                 for b, cnt, c, s in perA.get(a, [])[:TOP_RULES]]
        return "\n".join(lines) or "(no mined rules for this item)"

    # ---------- deterministic consumers ----------
    pop_list = [b for b, _ in f1A.most_common(LIST_LEN)]
    cooc_map = {}
    for (x, b), c in pairA.items():
        cooc_map.setdefault(x, []).append((b, c))
    for x in cooc_map:
        cooc_map[x].sort(key=lambda t: -t[1])
    symbolic_lists, cooc_lists = [], []
    for p in probes:
        a = p["ante"]
        symbolic_lists.append([b for b, _, _, _ in perA.get(a, [])[:LIST_LEN]] or None)
        cooc_lists.append([b for b, _ in cooc_map.get(a, [])[:LIST_LEN]] or None)

    # ---------- LLM consumers (cached) ----------
    jobs = []
    for i, p in enumerate(probes):
        jobs.append((f"p{i}_cold", PROMPT_COLD.format(a=p["ante"], k=LIST_LEN)))
        jobs.append((f"p{i}_rules", PROMPT_RULES.format(a=p["ante"], k=LIST_LEN, rules=rules_text(p["ante"]))))
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
        sys.exit(0)

    # ---------- scoring ----------
    def load(name):
        return json.load(open(os.path.join(OUT, "raw", name + ".json"), encoding="utf-8"))

    cold_lists, rules_lists, truths = [], [], []
    for i, p in enumerate(probes):
        cold_lists.append(parse_ranked(load(f"p{i}_cold")["content"]))
        rules_lists.append(parse_ranked(load(f"p{i}_rules")["content"]))
        truths.append(p["truth_B"])

    res = {
        "n_probes": len(probes), "seed": SEED,
        "split_agreement": {"A_argmax_eq_B_argmax": agree, "n": len(probes)},
        "consumers": {
            "cold_llm": metrics(cold_lists, truths),
            "llm_plus_rules": metrics(rules_lists, truths),
            "symbolic_argmax": metrics(symbolic_lists, truths),
            "popularity": metrics([pop_list] * len(truths), truths),
            "item_item_cooccurrence": metrics(cooc_lists, truths),
        },
        "significance": {
            "llm_plus_rules_vs_popularity_hit1": mcnemar_exact(
                [t in (r[:1] if r else []) for r, t in zip(rules_lists, truths)],
                [t in (r[:1] if r else []) for r, t in zip([pop_list] * len(truths), truths)]),
            "llm_plus_rules_vs_symbolic_hit1": mcnemar_exact(
                [t in (r[:1] if r else []) for r, t in zip(rules_lists, truths)],
                [t in (r[:1] if r else []) for r, t in zip(symbolic_lists, truths)]),
        },
        "model": load("p0_rules")["model"],
    }
    json.dump({"summary": res, "probes": probes},
              open(os.path.join(OUT, "exp9_summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=2))
