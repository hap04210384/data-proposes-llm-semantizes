# 引擎约定（engines 目录布局与调用方式）

> 引擎是两个独立开源仓库，**本仓库不包含引擎代码**；引擎原样下载、保持不动，
> 任何改造以本仓库 `drivers/` 的封装与补丁脚本实现。引擎原样备份见
> `../备份/engines-pristine-20261003.zip`。

## 目录约定（相对本仓库根）

```
../engines/AnyFIM/      Anytime-Frequent-Itemset-Mining（main 分支）
../engines/TensorFIM/   TensorFIM（main 分支）
```

## 环境（本机实测 2026-10-03）

- Windows 11 x64，Intel i9-12900，NVIDIA RTX 3060 Ti（sm_86，与 TensorFIM 论文参考平台一致）
- CUDA Toolkit v12.5（另装 v12.8），Visual Studio 2022 Community（MSBuild 17.14）
- 注意：本机 `nvidia-smi` 报 NVML 错误不影响 CUDA 程序运行；MSBuild 编译 CUDA 工程需显式传
  `//p:CudaToolkitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\'`

## 引擎要点（冒烟测试已验证，详见 ../设计文档/实验启动清单.md）

### TensorFIM（精确计数后端）

- 预编译二进制：`../engines/TensorFIM/code/engine/run/CoParaCG_{baseline,bmma}.exe`
- 调用：`<exe> <数据集路径> <相对支持度阈值>`；输出 `<数据集>-<阈值>=Results.txt`
- 成功判定：**退出码 1 = 成功**（历史约定），以输出文件存在为准
- 微基准：`code/microbench/bmma_bench.exe`（含正确性门）
- gold 校验：`python scripts/verify_all.py`（集合一致为通过）
- 自带 7 个中小 FIMI 数据集在 `data/small/`

### AnyFIM（免阈值 anytime 挖掘前端）

- 源码工程：`../engines/AnyFIM/code/AnyFIM-Anytime/AnytimeMining.sln`
- 构建（Release x64）：
  ```
  MSBuild AnytimeMining.sln //p:Configuration=Release //p:Platform=x64 //p:CudaToolkitDir='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\'
  ```
- 产物：`x64/Release/AnytimeMining.exe`
- ⚠️ 数据集路径与轮数**硬编码在 kernel.cu**（当前激活行为 TCGA_BRCA, UptoStage=10）。
  不改源码的前提下，`drivers/` 用"临时构建目录 + 补丁"方式跑任意数据集：
  见 `drivers/README.md`（改造脚本待写）。
- 自带 FIMI 数据集在 `../engines/AnyFIM/datasets/`（chess/mushroom/pumsb/accidents）

## 引擎本地改动（相对下载原样，diff 见 anyfim_kernel_dense.diff）

**AnyFIM anytime kernel.cu 已原生加入"精简后稠密重映射"**（2026-10-03，张老师指示固化进引擎）：
幸存项按频率名次映射到 [0, S-1]，位图随幸存项数缩放；输出回映原始 ID。
- 正确性：chess 20 轮同阈值 Jaccard=1.0；稠密/无稠密两模式 retail 30 轮逐轮集合一致；
- 性能（RTX 3060 Ti，retail 50 轮，各 5 次中位）：dense 0.0220 s vs nodense 0.0226 s ≈ 1.03×——
  **当前负载下时间基本无差**（之前单次跑的 2.2× 是毫秒级噪声）；收益主要在**内存**：
  retail 位图 16470 行→幸存项数行（约 181 MB→约 1 MB 量级），MIMIC 规模（数千项×数十万事务）下
  稀疏 ID 位图将超 1 GB，稠密化是必要条件；
- 消融：构建驱动 `dense=False` 时自动反向补丁（注释掉 denseRemapItems 调用）。

## 驱动层补丁清单（打在 build/ 副本上，引擎原码不动）

| 补丁 | 内容 | 开关 |
|---|---|---|
| 1 数据路径 | kernel.cu 硬编码数据路径 → `..\\TransactionSets\\data.txt`（UptoStage 编译入） | 始终 |
| 2 标定缓存 | CPU/GPU 算力标定（每次启动约 148s，冷机为 PTX JIT）经环境变量 `ANYFIM_GPU_PCT` 缓存跳过 | 始终 |

（稠密重映射原为补丁3，已固化进引擎源码，见上一节。）

## 口径备忘

- 两边引擎输出 MFI 的文本格式不同（TensorFIM: `N-th: {items} support: x frequency: y`；
  AnyFIM anytime: 逐轮 stage 输出），统一解析器见 `drivers/parse_mfi.py`（待写）。
