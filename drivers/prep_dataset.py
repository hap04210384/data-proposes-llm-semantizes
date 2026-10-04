# -*- coding: utf-8 -*-
"""数据集标准预处理：剔除核项（kernel items，出现在全部事务中的项）。

原理：核项对所有项集支持度贡献相同，含核项的项集与去掉核项后的项集
支持度一致，MFI 结构完全保留（TensorFIM 仓库 strip_tcga.py 同理）。
用法:
    python drivers/prep_dataset.py <输入> <输出>
已剔除则原样复制并报告 0。
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
