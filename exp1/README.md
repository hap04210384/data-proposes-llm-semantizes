# Exp.1 — Cost of the supply stream vs the threshold-grid baseline (chess / retail)

Scripts reproducing manuscript Exp. 1 (Table I, Fig. 2). All result files are
written to `../results/exp1/` (that directory is not tracked; the scripts
regenerate it).

## Run order

1. **Anytime stream, 5 repetitions** (cumulative engine time + total wall-clock,
   per stage):
   ```
   python exp1/run_reps.py chess 20 5
   python exp1/run_reps.py retail 50 5
   ```
   Outputs `anyfim_{chess_K20,retail_K50}_rep{1..5}.csv` and `_median5.csv`.

2. **Full-K measured grid** (the grid side of manuscript Fig. 2 / Table I; the
   baseline miner is re-run in full at every one of the K supply thresholds,
   medians of 5):
   ```
   python exp1/run_grid_fullK.py chess 5
   python exp1/run_grid_fullK.py retail 5
   ```
   Outputs `fullK_grid_{chess,retail}.csv` (per-threshold wall/engine + cumulative).
   Wall speedup = sum of per-threshold wall medians / anytime wall median;
   engine ratio = sum of per-threshold engine medians / anytime cumulative engine median.

3. **Dual-caliber comparison (deduplicated-threshold subset)**, optional:
   ```
   python exp1/run_dual_caliber.py chess 20 5
   python exp1/run_dual_caliber.py retail 50 5
   ```
   Outputs `dual_caliber_{chess,retail}.csv` (12/9 deduplicated thresholds only;
   a caliber cross-check).

4. **Per-stage correctness (Jaccard alignment)**:
   ```
   python exp1/run_matched_pairs.py
   ```
   Outputs `pairs_{chess_K20,retail_K50}.csv` (Jaccard of each stage against the
   same-threshold baseline).

## Manuscript number provenance (since v0.3)

| Manuscript number | Source file |
|---|---|
| chess wall 0.186 s / 11.5x | last row `wall_total_s` of `anyfim_chess_K20_median5.csv`; `fullK_grid_chess.csv` per-threshold median sum 2.140 s |
| retail wall 0.928 s / 6.9x | `anyfim_retail_K50_median5.csv`; `fullK_grid_retail.csv` per-threshold median sum 6.363 s |
| engine ratio 1.01x / 1.89x | `fullK_grid_*.csv` per-threshold `engine_s` median sum / anytime `cum_engine_s` |
| Jaccard 18/20, 49/50 | `pairs_*.csv` |
