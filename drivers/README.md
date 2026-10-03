# drivers/ — 引擎驱动封装

目标：不动引擎源码，用命令行 + 工作目录约定统一驱动两个引擎。

- `build_anyfim.py` — 在临时构建目录应用 kernel.cu 补丁（数据路径/UptoStage 参数化），
  调 MSBuild 出二进制（待写）
- `run_anyfim.py` — 跑 anytime 流，落盘逐轮 MFI（待写）
- `run_tensorfim.py` — 跑给定阈值挖掘 / 批量支持度查询（待写）
- `parse_mfi.py` — 统一解析两引擎输出（待写）
