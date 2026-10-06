# -*- coding: utf-8 -*-
"""Exp. 10 (external consistency): per-stage MFI families delivered by the anytime
supply stream vs. standard public FIM implementations (mlxtend fpmax, and
fpgrowth followed by a maximality filter) at identical full-precision thresholds.

Usage:
    python experiments/exp10_external_consistency.py <chess|retail> [anytime_results.txt]

Output:
    results/exp10/fimi_check_<name>.json   (incremental, resumable)

Notes:
    - The implicit stage threshold is theta_k = (frequency of the k-th most
      frequent item) / n_transactions at FULL float precision. The header field
      supportThrePerStage in the engine output is rounded to six decimals and
      must NOT be used for threshold replay.
    - Requires: pip install mlxtend pandas scipy (reviewer-side only; no engine
      dependency is changed).
"""
import os, sys, json, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'drivers'))
RES = os.path.join(ROOT, 'results')
OUTDIR = os.path.join(RES, 'exp10')
os.makedirs(OUTDIR, exist_ok=True)

from run_anyfim import run_anyfim          # noqa: E402
from parse_mfi import parse_anyfim_anytime  # noqa: E402

KMAP = {'chess': 20, 'retail': 50}
_te_cache = {}

def load_tx(path):
    tx = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            items = ln.split()
            if items:
                tx.append(frozenset(int(x) for x in items))
    return tx

def encode(tx):
    if 'df' not in _te_cache:
        import pandas as pd
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        arr = te.fit_transform([list(t) for t in tx], sparse=True)
        _te_cache['df'] = pd.DataFrame.sparse.from_spmatrix(arr, columns=[str(c) for c in te.columns_])
    return _te_cache['df']

def maximal_filter(sets):
    sets = sorted(sets, key=len, reverse=True)
    out = []
    for s in sets:
        if not any(s < o for o in out):
            out.append(s)
    return set(out)

def mlxtend_mfis(tx, theta, algo):
    from mlxtend.frequent_patterns import fpmax, fpgrowth
    df = encode(tx)
    fn = fpmax if algo == 'fpmax' else fpgrowth
    res = fn(df, min_support=theta, use_colnames=True)
    sets = {frozenset(int(x) for x in iset) for iset in res['itemsets']}
    if algo == 'fpgrowth':
        sets = maximal_filter(sets)
    return sets

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0

def main():
    name = sys.argv[1]
    K = KMAP[name]
    ds = os.path.join(RES, 'exp1', f'{name}.txt')
    tx = load_tx(ds)

    # anytime stream results: prefer an existing engine output, else run the engine
    res_path = sys.argv[2] if len(sys.argv) > 2 else None
    if res_path is None:
        cand = os.path.join(ROOT, 'build', 'anyfim', 'src', 'x64',
                            'TransactionSets', f'data.txt-{K}-stages=Results.txt')
        if os.path.exists(cand):
            res_path = cand
    if res_path is None:
        rows, res_path = run_anyfim(ds, K)
    header, stages = parse_anyfim_anytime(res_path)
    assert len(stages) == K, (len(stages), K)
    # exact thresholds: theta_k = k-th largest item frequency / n
    from collections import Counter
    freq = Counter()
    for t in tx:
        for i in t:
            freq[i] += 1
    fs = sorted(freq.values(), reverse=True)
    n = len(tx)
    sup = [fs[k - 1] / n for k in range(1, K + 1)]

    out_json = os.path.join(OUTDIR, f'fimi_check_{name}.json')
    done = {}
    if os.path.exists(out_json):
        done = {r['stage']: r for r in json.load(open(out_json, encoding='utf-8'))['stages']}

    results = []
    for k in range(1, K + 1):
        if k in done:
            results.append(done[k]); continue
        th = sup[k - 1]
        a = set(stages[k]['mfis'].keys())
        t0 = time.perf_counter()
        fpm = mlxtend_mfis(tx, th, 'fpmax')
        t_fpmax = time.perf_counter() - t0
        fgr = mlxtend_mfis(tx, th, 'fpgrowth')
        rec = {
            'stage': k, 'theta': th,
            'n_anytime': len(a), 'n_fpmax': len(fpm), 'n_fpgrowth': len(fgr),
            'anytime==fpmax': a == fpm, 'anytime==fpgrowth': a == fgr,
            'jaccard_fpmax': round(jaccard(a, fpm), 4),
            'jaccard_fpgrowth': round(jaccard(a, fgr), 4),
            'fpmax_s': round(t_fpmax, 3),
        }
        results.append(rec)
        print(rec, flush=True)
        json.dump({'dataset': name, 'K': K, 'stages': results},
                  open(out_json, 'w', encoding='utf-8'), indent=1)

    n_exact = sum(1 for r in results if r['anytime==fpmax'])
    print(f'\n{name}: anytime==fpmax exact at {n_exact}/{len(results)} stages')
    print('min jaccard(fpmax):', min(r['jaccard_fpmax'] for r in results))
    print('total fpmax time (s):', round(sum(r['fpmax_s'] for r in results), 1))

if __name__ == '__main__':
    main()
