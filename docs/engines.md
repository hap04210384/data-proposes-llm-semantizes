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

## 口径备忘

- 两边引擎输出 MFI 的文本格式不同（TensorFIM: `N-th: {items} support: x frequency: y`；
  AnyFIM anytime: 逐轮 stage 输出），统一解析器见 `drivers/parse_mfi.py`（待写）。
