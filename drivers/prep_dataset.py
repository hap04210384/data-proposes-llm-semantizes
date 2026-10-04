# -*- coding: utf-8 -*-
"""Standard dataset preprocessing: strip kernel items (items that appear in every transaction).

Rationale: a kernel item contributes the same support to every itemset, so itemsets
containing it have exactly the same support as the same itemsets without it, and the
MFI structure is fully preserved (same idea as strip_tcga.py in the TensorFIM repo).
Usage:
    python drivers/prep_dataset.py <input> <output>
If nothing needs stripping, the input is copied as-is and 0 is reported.
"""
import sys
from collections import Counter


def strip_kernel(src, dst):
    freq = Counter()
    rows = []
    with open(src, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ids = ln.split()
            rows.append(ids)
            freq.update(set(ids))
    n = len(rows)
    kernel = {x for x, c in freq.items() if c == n}
    with open(dst, "w", encoding="utf-8") as f:
        for ids in rows:
            f.write(" ".join(x for x in ids if x not in kernel) + "\n")
    return n, len(freq), len(kernel)


if __name__ == "__main__":
    n, m, k = strip_kernel(sys.argv[1], sys.argv[2])
    print(f"transactions={n} items={m} kernel_removed={k} -> {sys.argv[2]}")
