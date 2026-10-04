# -*- coding: utf-8 -*-
"""DeepSeek-V3 语义化适配层（实验 5/6 用）。

设计原则（论文复现底线，附录需披露）：
- 模型全名与版本固定：deepseek-chat（对应 DeepSeek-V3 系列），调用时记录实际返回的 model 字段；
- temperature=0（可复现），所有 prompt 原文随结果落盘；
- API key 只经环境变量 DEEPSEEK_API_KEY 注入，不入库。

用法:
    from drivers.llm_semantizer import semantize_rules, llm_author_rules
"""
import json
import os
import time
import urllib.request

BASE = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.0

# 语义化：引擎供给的规则 -> LLM 只起名/打标签/消解冲突（永远不写数字）
PROMPT_SEMANTIZE = """You are semantizing association rules mined from data.
For each rule below, provide: (1) a short human-readable name; (2) domain tags;
(3) a conflict note if it contradicts another rule in the list.
The support and confidence values are EXACT computations from the data.
You must NOT invent, modify, or restate any number.

Rules (id: antecedent -> consequent; support; confidence):
{rules}

Return strict JSON: [{{"id": ..., "name": ..., "tags": [...], "conflict": ...}}, ...]"""

# 幻觉对照基线：让 LLM 直接当规则作者（Exp 5 的"对手"）
PROMPT_AUTHOR = """You are a domain expert writing association rules for {domain}.
Based on your knowledge, author {n} plausible association rules in the form
"antecedent -> consequent", each with an estimated support (fraction of cases)
and confidence. Return strict JSON:
[{{"antecedent": [...], "consequent": [...], "support": 0.0, "confidence": 0.0}}, ...]"""


def _chat(prompt, max_retries=4):
    key = os.environ.get("DEEPSEEK_API_KEY")
    assert key, "缺少环境变量 DEEPSEEK_API_KEY"
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
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def semantize_rules(rules_text):
    """供给规则的语义化。rules_text 为 PROMPT_SEMANTIZE 的 {rules} 填充内容。"""
    return _chat(PROMPT_SEMANTIZE.format(rules=rules_text))


def llm_author_rules(domain, n=20):
    """幻觉对照基线：LLM 凭空 authored 规则（含其自己估计的支持度/置信度）。"""
    return _chat(PROMPT_AUTHOR.format(domain=domain, n=n))


if __name__ == "__main__":
    demo = semantize_rules("R1: {{aspirin, warfarin}} -> {{bleeding}}; support 0.0034; confidence 0.61")
    print(demo["model"], demo["content"][:300])
