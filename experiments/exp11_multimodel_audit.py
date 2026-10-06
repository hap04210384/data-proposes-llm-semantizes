# -*- coding: utf-8 -*-
"""Exp 11 multi-model audit of the numeric firewall (paper Exp. 11).

Same protocol as Exp. 5, Group B: the identical 50 engine-supplied retail rules
(exact support/confidence) are semantized behind the numeric firewall
(no numeric field in the output schema; the prompt forbids writing numbers),
temperature 0, 3 repetitions per model. The DeepSeek family results are reused
from Exp. 5 (group_b_raw.json); Qwen and GLM are called over their public APIs.

Audited per repetition (identical criteria for every family):
  n_named          - parsed semantizations in the returned JSON array
  num_violations   - entries that carry a numeric field (support/confidence/sup/conf)
  citation_viol    - number references in names/tags/conflict notes that point
                     to items outside the annotated rule's exact item set

Reproducibility: every call persists prompt, server-resolved model, and timestamp.
"""
import json
import os
import re
import statistics
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "exp11")
os.makedirs(OUT, exist_ok=True)

# --- identical rule text as Exp. 5 Group B, rebuilt from the archived exact rows ---


PROMPT_SEMANTIZE = """You are semantizing association rules mined from data.
For each rule below, provide: (1) a short human-readable name; (2) domain tags;
(3) a conflict note if it contradicts another rule in the list.
The support and confidence values are EXACT computations from the data.
You must NOT invent, modify, or restate any number.

Rules (id: antecedent -> consequent; support; confidence):
{rules}

Return strict JSON: [{{"id": ..., "name": ..., "tags": [...], "conflict": ...}}, ...]"""

REPS = 3


def load_rule_rows():
    """Rebuild the exact Exp. 5 Group B rule rows (archived exact recomputation)."""
    src = os.path.join(ROOT, "results", "exp5", "exp5_summary.json")
    s = json.load(open(src, encoding="utf-8"))
    rows = s["group_b_exact"]
    lines = []
    for e in rows:
        items = e["items"]
        ant = ", ".join(map(str, items[:-1]))
        lines.append(f"{e['id']}: {{{ant}}} -> {{{items[-1]}}}; "
                     f"support {e['support_exact']:.6f}; confidence {e['conf_exact']:.4f}")
    return rows, "\n".join(lines)


def parse_json_array(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def audit_rep(parsed, rows_by_id):
    """Identical criteria for every model family."""
    n_named = 0
    num_violations = 0
    citation_viol = 0
    if not isinstance(parsed, list):
        return {"n_named": 0, "num_violations": 1, "citation_viol": 0}
    for p in parsed:
        if not isinstance(p, dict):
            num_violations += 1
            continue
        keys = {k.lower() for k in p.keys()}
        if keys & {"support", "confidence", "sup", "conf"}:
            num_violations += 1
        n_named += 1
        rid = str(p.get("id", ""))
        row = rows_by_id.get(rid)
        if row is None:
            citation_viol += 1
            continue
        allowed = {str(x) for x in row["items"]}
        text_bits = [str(p.get("name", "")), str(p.get("conflict", "") or "")]
        for t in p.get("tags", []) or []:
            text_bits.append(str(t))
        refs = re.findall(r"\d+", " ".join(text_bits))
        bad = [x for x in refs if x not in allowed and x != rid.lstrip("R")]
        if bad:
            citation_viol += 1
    return {"n_named": n_named, "num_violations": num_violations,
            "citation_viol": citation_viol}


# ---------------- model endpoints (OpenAI-compatible chat completions) ----------------
def chat(base, key_env, model, prompt, max_retries=4):
    key = os.environ.get(key_env)
    assert key, f"{key_env} environment variable missing"
    body = json.dumps({"model": model, "temperature": 0.0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(base, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            msg = out["choices"][0]["message"]["content"]
            return {"model": out.get("model", model), "content": msg,
                    "prompt": prompt, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def run_family(name, base, key_env, model, prompt, rows_by_id):
    raw_path = os.path.join(OUT, f"{name}_raw.json")
    raws = json.load(open(raw_path, encoding="utf-8")) if os.path.exists(raw_path) else []
    while len(raws) < REPS:
        print(f"  [{name}] rep {len(raws) + 1}/{REPS} ...", flush=True)
        raws.append(chat(base, key_env, model, prompt))
        json.dump(raws, open(raw_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    per_rep = []
    for r in raws[:REPS]:
        a = audit_rep(parse_json_array(r["content"]), rows_by_id)
        a.update({"model": r["model"], "ts": r["ts"]})
        per_rep.append(a)
    return {"endpoint_model": model, "per_rep": per_rep}


def main():
    rows, rules_text = load_rule_rows()
    rows_by_id = {e["id"]: e for e in rows}
    prompt = PROMPT_SEMANTIZE.format(rules=rules_text)
    print(f"{len(rows)} rules, prompt {len(prompt)} chars")

    # DeepSeek family: reuse Exp. 5 Group B raw calls (identical prompt/protocol)
    ds_raw_path = os.path.join(ROOT, "results", "exp5", "group_b_raw.json")
    ds_raws = json.load(open(ds_raw_path, encoding="utf-8"))[:REPS]
    ds_per_rep = []
    for r in ds_raws:
        assert r["prompt"] == prompt, "DeepSeek archived prompt differs from rebuilt prompt"
        a = audit_rep(parse_json_array(r["content"]), rows_by_id)
        a.update({"model": r["model"], "ts": r["ts"]})
        ds_per_rep.append(a)
    families = {"deepseek": {"endpoint_model": "deepseek-chat (Exp. 5 archive)",
                             "per_rep": ds_per_rep}}
    print("deepseek (reused):", ds_per_rep)

    families["qwen"] = run_family(
        "qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "QWEN_API_KEY", "qwen-plus", prompt, rows_by_id)
    print("qwen:", families["qwen"]["per_rep"])

    families["glm"] = run_family(
        "glm", "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "GLM_API_KEY", "glm-4-air", prompt, rows_by_id)
    print("glm:", families["glm"]["per_rep"])

    def agg(fam):
        pr = fam["per_rep"]
        return {
            "server_model": sorted({r["model"] for r in pr}),
            "n_named_median": statistics.median(r["n_named"] for r in pr),
            "num_violations_total": sum(r["num_violations"] for r in pr),
            "citation_viol_total": sum(r["citation_viol"] for r in pr),
            "audited_semantizations": sum(r["n_named"] for r in pr),
        }

    summary = {"experiment": "Exp11: multi-model numeric-firewall audit (same 50 rules, "
                             "same prompt/protocol as Exp. 5 Group B; temperature 0; 3 reps per family)",
               "families": {k: {**v, "audit": agg(v)} for k, v in families.items()}}
    with open(os.path.join(OUT, "exp11_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v["audit"] for k, v in summary["families"].items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
