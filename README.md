# Zero-Threshold Rule Supply

Data-proposed, LLM-semantized rule supply for neuro-symbolic systems: an anytime, threshold-free mining pipeline (AnyFIM) with bit-exact support/confidence assembly (TensorFIM), semantized by an LLM that never authors numeric content.

本仓库是论文 *Zero-Threshold Rule Supply for Neuro-Symbolic Systems*（在研）的实验代码与复现包，将来公开供审稿人与读者复现。

## 依赖的两个计算引擎（不在本仓库内，各自独立开源）

| 引擎 | 作用 | 仓库 |
|---|---|---|
| AnyFIM | 免阈值 anytime MFI 流挖掘（P1） | https://github.com/hap04210384/Anytime-Frequent-Itemset-Mining |
| TensorFIM | 张量核精确位图计数（P2b 规则装配的支持度查询后端） | https://github.com/hap04210384/TensorFIM |

引擎下载后置于本仓库同级目录 `../engines/AnyFIM`、`../engines/TensorFIM`（见 `docs/engines.md`）。

## 目录结构

```
drivers/     引擎驱动封装（统一命令行调用、计时、结果解析）
exp1/        实验1：阈值网格扫荡 vs anytime 零调参供给的成本对比
exp2/        实验2：规则质量三方对拍（AnyFIM+LLM / 人工 / LLM直生）
exp3/        实验3：下游消费增益 + 消融
exp4/        实验4：供给管线吞吐与逐位精确性
mimic_txn/   MIMIC-IV 事务化脚本（hadm_id=事务，CCS 聚合）
llm_client.py  统一 LLM 调用封装（记录 model/prompt版本/温度/时间戳）
docs/        引擎约定、实验设计、数据说明
```

## 数据

- FIMI 基准（chess / retail / mushroom / pumsb / accidents）：FIMI 语料 / AnyFIM 仓库自带；
- MIMIC-IV（医疗案例）：PhysioNet 认证后获取，**不入库**（合规）。

## 状态

开发中。实验脚本逐步提交；实验结果（CSV/日志）不入库，图件由脚本从数据重新生成。
