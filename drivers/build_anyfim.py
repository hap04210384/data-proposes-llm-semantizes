# -*- coding: utf-8 -*-
"""AnyFIM anytime 引擎的"临时构建目录 + 补丁"驱动。

原则：不动 engines/AnyFIM 原码。把 AnytimeMining 工程拷到本库 build/ 目录，
只改副本里 kernel.cu 的数据路径行（该路径在源仓中硬编码），用 MSBuild 出二进制。

用法：
    from drivers.build_anyfim import ensure_anyfim_built
    exe = ensure_anyfim_built(dataset_path, upto_stage)
    # exe 位于 build/anyfim/src/x64/Release/AnytimeMining.exe
    # 运行时 cwd 必须为 exe 所在目录（数据路径是相对 ..\\TransactionSets\\data.txt）
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                    # 仓库根
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
    """引擎源码 -> 构建目录（已存在则增量同步，删除多余文件）。"""
    if os.path.exists(SRC_DIR):
        shutil.rmtree(SRC_DIR)
    ignore = shutil.ignore_patterns("x64", ".vs", "TransactionSets")
    shutil.copytree(ENGINE_SRC, SRC_DIR, ignore=ignore)
    os.makedirs(TXN_DIR, exist_ok=True)


def _patch_kernel(upto_stage):
    """把未注释的 transSetFile 行替换为参数化版本（数据集固定为 TransactionSets\\data.txt）。

    同时打第二个补丁：CPU/GPU 算力标定（getGPUtaskPercentage，每次进程启动约 148s）
    改为"缓存到 exe 目录 gpu_pct_cache.txt，存在即跳过"——标定是一次性硬件标定，
    不属于供给耗时；此为驱动层补丁，引擎原码不动（详见 docs/engines.md）。
    """
    kp = os.path.join(SRC_DIR, KERNEL_REL)
    s = open(kp, encoding="utf-8").read()
    lines = s.split("\n")
    n_patch = n_cal = 0
    for i, ln in enumerate(lines):
        if re.match(r"^CString transSetFile = _T\(", ln):
            # 行内是 4 个字面反斜杠的 Windows 相对路径（与源仓保持一致）
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
    assert n_patch == 1, f"kernel.cu 中应恰有 1 行激活的数据路径行，实际 {n_patch}"
    assert n_cal == 1, f"kernel.cu 中应恰有 1 处 GPUtaskPercentage 标定调用，实际 {n_cal}"
    open(kp, "w", encoding="utf-8", newline="\n").write("\n".join(lines))


def ensure_anyfim_built(dataset_path, upto_stage, rebuild=False):
    """确保（必要时构建）指定数据集/轮数的 AnytimeMining.exe，返回 exe 路径。"""
    stamp = os.path.join(BUILD_DIR, f"built_for_stage{int(upto_stage)}.stamp")
    exe = os.path.join(SRC_DIR, "x64", "Release", "AnytimeMining.exe")
    if rebuild or not os.path.exists(stamp) or not os.path.exists(exe):
        if not os.path.isdir(ENGINE_SRC):
            raise FileNotFoundError(f"引擎源码不存在：{ENGINE_SRC}（请先下载 AnyFIM 到 engines/）")
        _copy_sources()
        _patch_kernel(upto_stage)
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
            raise RuntimeError("AnytimeMining 构建失败")
        open(stamp, "w").write(str(upto_stage))
    # 每次调用都刷新数据集文件（内容可能变化）
    shutil.copyfile(dataset_path, os.path.join(TXN_DIR, "data.txt"))
    return exe


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORKSPACE, "engines", "AnyFIM", "datasets", "chess.txt")
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print("exe:", ensure_anyfim_built(ds, st, rebuild=True))
