# Exp.1 — Cost of the supply stream vs threshold-grid baseline (chess / retail)

复现手稿 Exp.1（Table I、Fig.2）的全部脚本。所有结果文件写入 `../results/exp1/`
（该目录不入库，脚本可直接再生成）。

## 运行顺序

1. **anytime 流 5 次重复**（引擎累计时间 + 全程墙钟，逐阶段输出）：
   ```
   python exp1/run_reps.py chess 20 5
   python exp1/run_reps.py retail 50 5
   ```
   产出 `anyfim_{chess_K20,retail_K50}_rep{1..5}.csv` 与 `_median5.csv`。

2. **全 K 网格实测**（手稿 Fig.2 / Table I 的网格侧；在全部 K 个供给阈值上
   逐个全量重跑基线矿工，5 次重复取中位）：
   ```
   python exp1/run_grid_fullK.py chess 5
   python exp1/run_grid_fullK.py retail 5
   ```
   产出 `fullK_grid_{chess,retail}.csv`（逐档 wall/engine + 累计）。
   加速比 = 网格各档墙钟中位之和 ÷ anytime 墙钟中位；
   引擎比 = 网格各档引擎中位之和 ÷ anytime 引擎累计中位。

3. **双口径对照（去重阈值子集）**，可选：
   ```
   python exp1/run_dual_caliber.py chess 20 5
   python exp1/run_dual_caliber.py retail 50 5
   ```
   产出 `dual_caliber_{chess,retail}.csv`（仅 12/9 个去重阈值，用于口径对照）。

4. **逐阶段正确性（Jaccard 对齐）**：
   ```
   python exp1/run_matched_pairs.py
   ```
   产出 `pairs_{chess_K20,retail_K50}.csv`（各阶段与同学阈值基线的 Jaccard）。

## 手稿数字溯源（v0.3 起）

| 手稿数字 | 来源文件 |
|---|---|
| chess 墙钟 0.186 s / 11.5× | `anyfim_chess_K20_median5.csv` 末行 `wall_total_s`；`fullK_grid_chess.csv` 各档中位和 2.140 s |
| retail 墙钟 0.928 s / 6.9× | `anyfim_retail_K50_median5.csv`；`fullK_grid_retail.csv` 各档中位和 6.363 s |
| 引擎比 1.01× / 1.89× | `fullK_grid_*.csv` 各档 `engine_s` 中位和 ÷ anytime `cum_engine_s` |
| Jaccard 18/20、49/50 | `pairs_*.csv` |
