# -*- coding: utf-8 -*-
"""统一解析两个引擎的 MFI 输出。

- TensorFIM / AnyFIM-GivenThreshold 格式:
    `N-th: <len> { i1 i2 ... }   support: <rel>  frequency: <abs>`
- AnyFIM anytime 结果文件（-stages=Results.txt）:
    头部有 runtimePerStage / supportThrePerStage 等汇总行；
    之后每轮一个块 `====== Nth stage MFIs Number: K ======`，块内为累计 MFI 族
    （第 k 轮块 = 前 k 轮的 MFI 族，与 anytime 语义一致）。

所有 support 为相对支持度，frequency 为绝对计数。
"""
import re

LINE = re.compile(r"^\d+th:\s+(\d+)\s+\{\s*([^}]*)\}\s+support:\s*(\S+)\s+frequency:\s*(\d+)")
STAGE_HDR = re.compile(r"^=+\s*(\d+)th stage MFIs Number:\s*(\d+)\s*=+$")
STAGE_LIST = re.compile(r"^(\S*PerStage\S*):\s*(.*)$")


def parse_mfi_lines(text):
    """解析 `N-th: ...` 行序列 -> {frozenset(items): abs_freq}"""
    out = {}
    for ln in text.splitlines():
        m = LINE.match(ln.strip())
        if m:
            items = frozenset(int(x) for x in m.group(2).split())
            out[items] = int(m.group(4))
    return out


def parse_tensorfim_results(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_mfi_lines(f.read())


def parse_anyfim_anytime(path):
    """返回 (header, stages)；header 为标量汇总 dict，stages 为 {k: {mfis, n}}。"""
    text = open(path, encoding="utf-8", errors="replace").read()
    header, stages = {}, {}
    cur = None
    for ln in text.splitlines():
        s = ln.strip()
        m = STAGE_HDR.match(s)
        if m:
            cur = int(m.group(1))
            stages[cur] = {"n": int(m.group(2)), "mfis": {}}
            continue
        m = STAGE_LIST.match(s)
        if m and cur is None:
            key, vals = m.group(1), m.group(2).strip().rstrip(",")
            header[key] = [float(v) for v in vals.split(",")] if vals else []
            continue
        m = LINE.match(s)
        if m and cur is not None:
            items = frozenset(int(x) for x in m.group(2).split())
            stages[cur]["mfis"][items] = int(m.group(4))
    return header, stages
