# Engine conventions (directory layout and invocation)

> The two engines are independent open-source repositories; **this repository does
> not contain engine code**. Engines are downloaded as-is and left untouched; all
> adaptations are implemented in this repository's `drivers/` wrappers and patch
> scripts. A pristine copy of the engines is kept at
> `../bak/engines-pristine-20261003.zip` (local backup, not tracked).

## Directory layout (relative to the root of this repository)

```
../engines/AnyFIM/      Anytime-Frequent-Itemset-Mining (main branch)
../engines/TensorFIM/   TensorFIM (main branch)
```

## Environment (measured on this machine, 2026-10-03)

- Windows 11 x64, Intel i9-12900, NVIDIA RTX 3060 Ti (sm_86, same as the TensorFIM paper's reference platform)
- CUDA Toolkit v12.5 (v12.8 also installed), Visual Studio 2022 Community (MSBuild 17.14)
- Note: the `nvidia-smi` NVML error on this machine does not affect CUDA programs;
  building CUDA projects with MSBuild requires explicitly passing
  `//p:CudaToolkitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\'`

## Engine notes (verified by smoke tests; see the lab experiment-launch checklist)

### TensorFIM (exact-counting backend)

- Precompiled binaries: `../engines/TensorFIM/code/engine/run/CoParaCG_{baseline,bmma}.exe`
- Invocation: `<exe> <dataset path> <relative support threshold>`; output `<dataset>-<threshold>=Results.txt`
- Success criterion: **exit code 1 = success** (historical convention); check that the output file exists
- Micro-benchmark: `code/microbench/bmma_bench.exe` (includes a correctness gate)
- Gold verification: `python scripts/verify_all.py` (set equality = pass)
- Ships with 7 small/medium FIMI datasets in `data/small/`

### AnyFIM (threshold-free anytime mining frontend)

- Source project: `../engines/AnyFIM/code/AnyFIM-Anytime/AnytimeMining.sln`
- Build (Release x64):
  ```
  MSBuild AnytimeMining.sln //p:Configuration=Release //p:Platform=x64 //p:CudaToolkitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\'
  ```
- Artifact: `x64/Release/AnytimeMining.exe`
- Warning: the dataset path and stage count are **hard-coded in kernel.cu** (currently
  TCGA_BRCA, UptoStage=10). To run arbitrary datasets without touching the sources,
  `drivers/` uses a "temporary build directory + patch" approach: see `drivers/README.md`.
- Ships with FIMI datasets in `../engines/AnyFIM/datasets/` (chess/mushroom/pumsb/accidents)

## Local engine changes (relative to the pristine download; diff in anyfim_kernel_dense.diff)

**The AnyFIM anytime kernel.cu now natively includes "dense remapping after pruning"**
(2026-10-03, made a permanent part of the engine): surviving items are remapped by
frequency rank to [0, S-1] and the bitmap scales with the number of surviving items;
output is mapped back to original IDs.
- Correctness: chess 20 stages, same-threshold Jaccard = 1.0; dense/non-dense modes
  agree stage-by-stage on retail (30 stages);
- Performance (RTX 3060 Ti, retail 50 stages, median of 5): dense 0.0220 s vs
  nodense 0.0226 s, about 1.03x — **no significant time difference at the current
  workload** (the earlier single-run 2.2x was millisecond-level noise); the benefit
  is mainly **memory**: the retail bitmap shrinks from 16,470 rows to the number of
  surviving rows (about 181 MB to about 1 MB); at MIMIC scale (thousands of items x
  hundreds of thousands of transactions) a sparse-ID bitmap would exceed 1 GB, so
  densification is a necessity;
- Ablation: the build driver applies a reverse patch (commenting out the
  denseRemapItems call) when `dense=False`.

## Driver-layer patches (applied to the build/ copy; engine sources untouched)

| Patch | Content | Switch |
|---|---|---|
| 1 Data path | hard-coded dataset path in kernel.cu → `..\\TransactionSets\\data.txt` (UptoStage compiled in) | always |
| 2 Calibration cache | CPU/GPU throughput calibration (about 148 s per cold start, PTX JIT) is cached and skipped via the `ANYFIM_GPU_PCT` environment variable | always |

(Dense remapping was formerly patch 3; it has been merged into the engine source, see above.)

## Notes on measurement conventions

- The two engines emit MFIs in different text formats (TensorFIM:
  `N-th: {items} support: x frequency: y`; AnyFIM anytime: per-stage output);
  the unified parser is `drivers/parse_mfi.py`.
