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

## 驱动层补丁清单（打在 build/ 副本上，引擎原码不动）

| 补丁 | 内容 | 开关 |
|---|---|---|
| 1 数据路径 | kernel.cu 硬编码数据路径 → `..\\TransactionSets\\data.txt`（UptoStage 编译入） | 始终 |
| 2 标定缓存 | CPU/GPU 算力标定（每次启动约 148s，冷机为 PTX JIT）经环境变量 `ANYFIM_GPU_PCT` 缓存跳过 | 始终 |
| 3 稠密重映射 | 事务精简后幸存项 ID 按频率名次重映射到 [0, S-1]；位图随 S 缩放；输出回映原始 ID，结果文件编号不变 | `dense=True/False`（消融） |

补丁3实测（RTX 3060 Ti）：chess 20 轮 0.0383→0.0346 s（-10%）；retail 30 轮 0.0128→0.0057 s（**2.2×**）；
两种模式 30/30 轮输出集合逐一轮完全一致；与 TensorFIM 基线同阈值 Jaccard=1.0 不受影响。

## 口径备忘

- 两边引擎输出 MFI 的文本格式不同（TensorFIM: `N-th: {items} support: x frequency: y`；
  AnyFIM anytime: 逐轮 stage 输出），统一解析器见 `drivers/parse_mfi.py`（待写）。
