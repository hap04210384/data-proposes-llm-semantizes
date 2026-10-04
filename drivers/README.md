# drivers/ — engine driver wrappers

Goal: drive both engines through a unified command-line + working-directory
convention without modifying engine sources.

- `build_anyfim.py` — applies the kernel.cu patches (dataset path / UptoStage
  parameterization) in a temporary build directory and invokes MSBuild
- `run_anyfim.py` — runs the anytime stream and writes per-stage MFIs to CSV
- `llm_semantizer.py` — DeepSeek adapter for rule semantization (Exp. 5/6)
- `parse_mfi.py` — unified parser for both engines' output formats
- `prep_dataset.py` — strips ubiquitous kernel items from a dataset
