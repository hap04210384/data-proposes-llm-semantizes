# data-proposes-llm-semantizes

**Data proposes, LLM semantizes.** Zero-threshold, anytime rule supply for
neuro-symbolic systems: an anytime, threshold-free mining engine streams maximal
co-occurrence patterns with exact support; a rule-assembly layer derives implication
rules with bit-exact confidence on GPU tensor cores; an LLM is restricted to naming,
tagging, and conflict resolution — never numeric content. The supplied rule base is
**hallucination-free by construction**.

This repository is the experiment and reproduction package for the paper
*Zero-Threshold Rule Supply for Neuro-Symbolic Systems* (under review).
It implements the controlled studies behind the paper's claims:

| Exp. | Question | Script |
|---|---|---|
| 1 | Cost of manual threshold tuning at equal delivered thresholds | `exp1/` (drivers) |
| 2 | Where the preprocessing wall is (bitmap allocation vs. parsing) | `experiments/exp2_preprocess_diagnosis.py` |
| 3 | Anytime supply curves | `exp1/` + `results` (Fig. 3) |
| 4 | Rule quality at supply depth (lift, bootstrap stability) | `experiments/exp4_depth_curve.py`, `experiments/exp4_tcga_quality.py` |
| 5 | Hallucination control: LLM-authored vs. engine-supplied rules | `experiments/exp5_hallucination.py` |
| 6 | Downstream consumption: rule-injected next-item recommendation | `experiments/exp6_rule_injected_recommend.py` |
| 7 | Determinism and dense-ID-remap ablation | `experiments/exp7_determinism.py` |

## Engines (not vendored here)

| Engine | Role | Repository |
|---|---|---|
| AnyFIM | anytime, threshold-free MFI streaming (paper [1]) | https://github.com/hap04210384/Anytime-Frequent-Itemset-Mining |
| TensorFIM | tensor-core exact bitmap counting (paper [2]) | https://github.com/hap04210384/TensorFIM |

Clone the engines next to this repository as `../engines/AnyFIM` and
`../engines/TensorFIM` (see `docs/engines.md`). Engine sources are **not modified**;
`drivers/build_anyfim.py` only applies a build-side patch into `build/`.

## Datasets

All four datasets ship with this repository under `data/` (TCGA-BRCA is
gzip-compressed because of GitHub's 100 MB file limit; decompress it before use):
`data/chess.txt`, `data/retail.txt`, `data/kosarak.txt`,
`data/TCGA_BRCA_stripped.txt.gz`.

| Dataset | Shape | Source |
|---|---|---|
| chess | 3,196 tx × 75 items | FIMI repository |
| retail | 88,162 tx × 16,470 items | FIMI repository |
| kosarak | 990,002 tx × 41,270 items | FIMI repository |
| TCGA-BRCA (stripped) | 1,226 tx × 31,683 items | open TCGA cohort; 28,977 ubiquitous core genes stripped by `drivers/prep_dataset.py` |

Place the decompressed files under `results/exp1/` and `results/exp2/` as
referenced by the scripts, or adjust paths at the top of each experiment script.
TCGA items are anonymous gene IDs.

## Environment

- Windows 11, NVIDIA GPU (experiments used an RTX 3060 Ti), CUDA 12.5, VS2022 toolchain
- Python 3.12 (standard scientific stack: `csv json hashlib matplotlib` only)
- DeepSeek API key via the environment variable `DEEPSEEK_API_KEY`
  (only Exp. 5 and Exp. 6 call an LLM; temperature is fixed to 0 and every prompt,
  resolved model version, and timestamp is written to `results/exp5/` / `results/exp6/`)

## Reproducing

Each experiment script is standalone and writes its raw evidence under `results/`
(not committed, regenerable):

```bash
python experiments/exp7_determinism.py            # determinism + dense-remap ablation
python experiments/exp5_hallucination.py          # Exp. 5 (needs DEEPSEEK_API_KEY)
python experiments/exp6_rule_injected_recommend.py# Exp. 6 (needs DEEPSEEK_API_KEY)
python experiments/exp2_preprocess_diagnosis.py   # Exp. 2 (kosarak, ~4 min)
python experiments/exp4_depth_curve.py            # Exp. 4 lift-vs-depth curves
python experiments/exp4_tcga_quality.py           # Exp. 4 TCGA bootstrap stability
```

The first AnyFIM run on a new dataset performs a one-time GPU calibration (~148 s);
the calibration constant is cached in `results/gpu_pct_cache.json`.

## Paper references

[1] Y. Zhang, J. Yang, X. Zhang, W. Yu, M. Huang, "Anytime frequent itemset mining:
A threshold-free progressive item-activation framework with CPU–GPU heterogeneous
collaboration," under review, 2026.
[2] Y. Zhang, T. Chen, J. Yang, X. Zhang, M. Huang, "TensorFIM: Exact maximal frequent
itemset mining on tensor cores," under review, 2026.
