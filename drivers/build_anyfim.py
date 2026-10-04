# -*- coding: utf-8 -*-
"""Build driver for the AnyFIM anytime engine ("temporary build directory + patch" approach).

Principle: the engines/AnyFIM sources are never modified. The AnytimeMining project is
copied into this repository's build/ directory, only the hard-coded dataset path line in
the kernel.cu copy is changed, and the binary is produced with MSBuild.

Usage:
    from drivers.build_anyfim import ensure_anyfim_built
    exe = ensure_anyfim_built(dataset_path, upto_stage)
    # exe lives at build/anyfim/src/x64/Release/AnytimeMining.exe
    # at run time the cwd must be the exe's directory (the dataset path is
    # relative: ..\\TransactionSets\\data.txt)
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                    # repository root
WORKSPACE = os.path.dirname(ROOT)                               # kimi_work/
ENGINE_SRC = os.path.join(WORKSPACE, "engines", "AnyFIM", "code", "AnyFIM-Anytime")
BUILD_DIR = os.path.join(ROOT, "build", "anyfim")
SRC_DIR = os.path.join(BUILD_DIR, "src")
TXN_DIR = os.path.join(SRC_DIR, "TransactionSets")
MSBUILD = r"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
CUDA_DIR = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"

KERNEL_REL = os.path.join("AnytimeMining", "kernel.cu")
PATCH_TAG = "//patched-by-drivers"


def _copy_sources():
    """Copy engine sources -> build directory (full sync; extra files removed)."""
    if os.path.exists(SRC_DIR):
        shutil.rmtree(SRC_DIR)
    ignore = shutil.ignore_patterns("x64", ".vs", "TransactionSets")
    shutil.copytree(ENGINE_SRC, SRC_DIR, ignore=ignore)
    os.makedirs(TXN_DIR, exist_ok=True)


def _patch_kernel(upto_stage):
    """Patch 1: parameterize the dataset path; patch 2: cache the CPU/GPU calibration
    in the ANYFIM_GPU_PCT environment variable.

    Calibration is a one-time hardware probe (about 148 s per process start) and is
    not part of the supply cost; engine sources stay untouched.
    """
    kp = os.path.join(SRC_DIR, KERNEL_REL)
    s = open(kp, encoding="utf-8").read()
    lines = s.split("\n")
    n_patch = n_cal = 0
    for i, ln in enumerate(lines):
        if re.match(r"^CString transSetFile = _T\(", ln):
            # the line carries a Windows relative path with 4 literal backslashes (as in the source repo)
            lines[i] = (
                'CString transSetFile = _T("..\\\\\\\\TransactionSets\\\\\\\\data.txt"); '
                f"int UptoStage = {int(upto_stage)};{PATCH_TAG}"
            )
            n_patch += 1
        if re.match(r"^\s*GPUtaskPercentage = getGPUtaskPercentage\(CPU_parallel_num\);", ln):
            lines[i] = (
                "    { const char* e = getenv(\"ANYFIM_GPU_PCT\");"
                " if (e && *e) { GPUtaskPercentage = atof(e);"
                " std::cout << \"GPUtaskPercentage (cached): \" << GPUtaskPercentage << std::endl; }"
                " else { GPUtaskPercentage = getGPUtaskPercentage(CPU_parallel_num);"
                " std::cout << \"ANYFIM_GPU_PCT=\" << GPUtaskPercentage << std::endl; } }"
                + PATCH_TAG
            )
            n_cal += 1
    assert n_patch == 1, f"expected exactly 1 active dataset-path line in kernel.cu, got {n_patch}"
    assert n_cal == 1, f"expected exactly 1 GPUtaskPercentage calibration call in kernel.cu, got {n_cal}"
    open(kp, "w", encoding="utf-8", newline="\n").write("\n".join(lines))


def _patch_disable_dense():
    """Reverse patch for the ablation: comment out the engine-native denseRemapItems() call (nodense builds only)."""
    kp = os.path.join(SRC_DIR, KERNEL_REL)
    s = open(kp, encoding="utf-8").read()
    old = "    reduceTransSet(freqPerItem_inSort[UptoStage]);\n    denseRemapItems();"
    new = "    reduceTransSet(freqPerItem_inSort[UptoStage]);\n    //denseRemapItems();//patched-by-drivers ablation: dense remapping disabled"
    assert s.count(old) == 1, f"reverse-patch anchor occurs {s.count(old)} times"
    open(kp, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))


def _replace_once(s, old, new, tag):
    assert s.count(old) == 1, f"patch anchor [{tag}] occurs {s.count(old)} times (expected 1): {old[:60]!r}"
    return s.replace(old, new, 1)


def ensure_anyfim_built(dataset_path, upto_stage, rebuild=False, dense=True):
    """Make sure an AnytimeMining.exe for the given dataset/stage count is built (building if
    necessary) and return its path.

    dense=True keeps the engine-native dense remapping (default); dense=False builds the
    ablation control.
    """
    tag = "dense" if dense else "nodense"
    exe = os.path.join(SRC_DIR, "x64", "Release", "AnytimeMining.exe")
    # a single exe path is shared and overwritten by builds with different (stage, tag),
    # so rebuild decisions must key on the *last actual build* parameters, not on a
    # parameter-named stamp (stamps do not invalidate on overwrite; a retail-50 binary
    # once ran a chess-20 experiment because of this).
    import json
    marker_p = os.path.join(BUILD_DIR, "last_build.json")
    try:
        last = json.load(open(marker_p, encoding="utf-8"))
    except Exception:
        last = {}
    need = (rebuild or not os.path.exists(exe)
            or last.get("stage") != int(upto_stage) or last.get("tag") != tag)
    if need:
        if not os.path.isdir(ENGINE_SRC):
            raise FileNotFoundError(f"engine sources not found: {ENGINE_SRC} (clone AnyFIM into engines/ first)")
        _copy_sources()
        _patch_kernel(upto_stage)
        if dense:
            # dense remapping is native in the engine sources; no patch needed, just verify presence
            src = open(os.path.join(SRC_DIR, KERNEL_REL), encoding="utf-8").read()
            assert "void denseRemapItems()" in src, "engine sources lack the native denseRemapItems"
        else:
            _patch_disable_dense()  # ablation: disable dense remapping
        env = dict(os.environ, CudaToolkitDir=CUDA_DIR)
        cmd = [
            MSBUILD, os.path.join(SRC_DIR, "AnytimeMining.sln"),
            "/p:Configuration=Release", "/p:Platform=x64",
            "/m", "/v:m", "/nologo",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env)
        if not os.path.exists(exe):
            print(r.stdout[-3000:])
            print(r.stderr[-2000:], file=sys.stderr)
            raise RuntimeError("AnytimeMining build failed")
        json.dump({"stage": int(upto_stage), "tag": tag},
                  open(marker_p, "w", encoding="utf-8"))
    # refresh the dataset file on every call (its content may change)
    shutil.copyfile(dataset_path, os.path.join(TXN_DIR, "data.txt"))
    return exe


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORKSPACE, "engines", "AnyFIM", "datasets", "chess.txt")
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print("exe:", ensure_anyfim_built(ds, st, rebuild=True))
