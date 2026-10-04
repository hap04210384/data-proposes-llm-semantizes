# -*- coding: utf-8 -*-
"""DeepSeek-V3 semantization adapter (used by Experiments 5 and 6).

Design principles (reproduction floor of the paper; to be disclosed in an appendix):
- the model name and version are pinned: deepseek-chat (DeepSeek-V3 family); the model
  field actually returned by the API is recorded with every call;
- temperature = 0 (reproducible); the exact prompt text is persisted with every result;
- the API key enters only via the DEEPSEEK_API_KEY environment variable, never the repo.

Usage:
    from drivers.llm_semantizer import semantize_rules, llm_author_rules
"""
import json
import os
import time
import urllib.request

BASE = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.0

# semantization: engine-supplied rules -> the LLM only names/tags/conflict-resolves (never writes numbers)
PROMPT_SEMANTIZE = """You are semantizing association rules mined from data.
For each rule below, provide: (1) a short human-readable name; (2) domain tags;
(3) a conflict note if it contradicts another rule in the list.
The support and confidence values are EXACT computations from the data.
You must NOT invent, modify, or restate any number.

Rules (id: antecedent -> consequent; support; confidence):
{rules}

Return strict JSON: [{{"id": ..., "name": ..., "tags": [...], "conflict": ...}}, ...]"""

# hallucination-control baseline: let the LLM act as the rule author outright (the "opponent" in Exp. 5)
PROMPT_AUTHOR = """You are a domain expert writing association rules for {domain}.
Based on your knowledge, author {n} plausible association rules in the form
"antecedent -> consequent", each with an estimated support (fraction of cases)
and confidence. Return strict JSON:
[{{"antecedent": [...], "consequent": [...], "support": 0.0, "confidence": 0.0}}, ...]"""


def _chat(prompt, max_retries=4):
    key = os.environ.get("DEEPSEEK_API_KEY")
    assert key, "DEEPSEEK_API_KEY environment variable missing"
    body = json.dumps({"model": MODEL, "temperature": TEMPERATURE,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(BASE, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            msg = out["choices"][0]["message"]["content"]
            return {"model": out.get("model", MODEL), "content": msg,
                    "prompt": prompt, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def semantize_rules(rules_text):
    """Semantize supplied rules; rules_text fills the {rules} slot of PROMPT_SEMANTIZE."""
    return _chat(PROMPT_SEMANTIZE.format(rules=rules_text))


def llm_author_rules(domain, n=20):
    """Hallucination-control baseline: the LLM authors rules from scratch (with its own estimated support/confidence)."""
    return _chat(PROMPT_AUTHOR.format(domain=domain, n=n))


if __name__ == "__main__":
    demo = semantize_rules("R1: {{aspirin, warfarin}} -> {{bleeding}}; support 0.0034; confidence 0.61")
    print(demo["model"], demo["content"][:300])
