# -*- coding: utf-8 -*-
"""Unified parser for both engines' MFI output.

- TensorFIM / AnyFIM-GivenThreshold format:
    `N-th: <len> { i1 i2 ... }   support: <rel>  frequency: <abs>`
- AnyFIM anytime result file (-stages=Results.txt):
    the header carries summary lines such as runtimePerStage / supportThrePerStage;
    each stage is then one block `====== Nth stage MFIs Number: K ======` containing the
    cumulative MFI family (the k-th block = the MFI family of the first k stages,
    consistent with anytime semantics).

All support values are relative support; frequency is the absolute count.
"""
import re

LINE = re.compile(r"^\d+th:\s+(\d+)\s+\{\s*([^}]*)\}\s+support:\s*(\S+)\s+frequency:\s*(\d+)")
STAGE_HDR = re.compile(r"^=+\s*(\d+)th stage MFIs Number:\s*(\d+)\s*=+$")
STAGE_LIST = re.compile(r"^(\S*PerStage\S*):\s*(.*)$")


def parse_mfi_lines(text):
    """Parse a sequence of `N-th: ...` lines -> {frozenset(items): abs_freq}"""
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
    """Return (header, stages); header is a dict of scalar summaries, stages is {k: {mfis, n}}."""
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
