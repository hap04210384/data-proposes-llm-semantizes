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
    """补丁1：数据路径参数化；补丁2：CPU/GPU 标定经 ANYFIM_GPU_PCT 环境变量缓存。

    标定是每次进程启动约 148s 的一次性硬件探测，不属于供给耗时；引擎原码不动。
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


# 补丁3：事务集精简后把幸存项 ID 重映射到稠密区间 [0, S-1]（按频率降序名次分配），
# 位图内存与缓存局部性随幸存项数 S 缩放而非原始 itemIDmax；输出时映射回原始 ID，
# 因此结果文件的项编号与未打补丁时完全一致（集合一致性可复验）。
_DENSE_REMAP_FN = r'''
static void denseRemapItems() {//patched-by-drivers 精简后项ID稠密重映射
    const int finalThreshold = freqPerItem_inSort[UptoStage];
    std::vector<int> old2new(freqPerItem.size(), -1);
    dense2orig.clear();
    for (int rank = 0; rank < (int)items_inFreqSort.size(); rank++) {
        int orig = items_inFreqSort[rank];
        if (orig >= 0 && orig < (int)freqPerItem.size() && freqPerItem[orig] >= finalThreshold) {
            old2new[orig] = (int)dense2orig.size();
            dense2orig.push_back(orig);
        }
    }
    const int S = (int)dense2orig.size();
    for (auto& row : h_transSet)
        for (auto& x : row) x = old2new[x];
    std::vector<int> freqDense(S, 0);
    for (int n = 0; n < S; n++) freqDense[n] = freqPerItem[dense2orig[n]];
    freqPerItem = freqDense;
    items_inFreqSort.resize(S);
    freqPerItem_inSort.resize(S);
    for (int n = 0; n < S; n++) { items_inFreqSort[n] = n; freqPerItem_inSort[n] = freqDense[n]; }
    itemIDmax = S - 1;
    itemIDmin = 0;
    std::cout << "denseRemapItems: " << S << " items remapped to [0," << S - 1 << "]" << std::endl;
}
'''


def _replace_once(s, old, new, tag):
    assert s.count(old) == 1, f"补丁锚点 [{tag}] 出现 {s.count(old)} 次（期望 1）: {old[:60]!r}"
    return s.replace(old, new, 1)


def _patch_dense_remap():
    """补丁3：稠密重映射（函数定义 + 调用点 + 位图循环 + 两处输出回映）。"""
    kp = os.path.join(SRC_DIR, KERNEL_REL)
    s = open(kp, encoding="utf-8").read()
    # 3.1 全局变量声明
    s = _replace_once(
        s,
        "std::vector<int> freqPerItem_inSort;",
        "std::vector<int> freqPerItem_inSort;\nstatic std::vector<int> dense2orig;   //patched-by-drivers",
        "globals",
    )
    # 3.2 函数定义（插在 create_d_transSet 之前）
    s = _replace_once(
        s,
        "static void create_d_transSet(int max_transLengthReduced, int transNumReduced)",
        _DENSE_REMAP_FN + "\nstatic void create_d_transSet(int max_transLengthReduced, int transNumReduced)",
        "fn-def",
    )
    # 3.3 调用点：reduceTransSet 之后立即重映射（同处 preprocessing 计时区内）
    s = _replace_once(
        s,
        "    reduceTransSet(freqPerItem_inSort[UptoStage]);",
        "    reduceTransSet(freqPerItem_inSort[UptoStage]);\n    denseRemapItems();//patched-by-drivers",
        "call-site",
    )
    # 3.4 位图幸存项循环：稠密ID从0开始，必须包含0（原循环 1.. 会漏掉稠密0号=最高频项）
    s = _replace_once(
        s,
        "    for (int i = 1; i <= itemIDmax; i++) {\n        if (freqPerItem[i] >= finalThreshold) freqItems.push_back(i);",
        "    for (int i = 0; i <= itemIDmax; i++) {//patched-by-drivers 稠密ID从0起\n        if (freqPerItem[i] >= finalThreshold) freqItems.push_back(i);",
        "bitmap-loop",
    )
    # 3.5/3.6 两处输出回映原始ID
    s = _replace_once(
        s,
        "            std::cout << MFIsPool[UptoDimensionStep][i].itemsSet[j] << \" \";",
        "            std::cout << (dense2orig.empty() ? MFIsPool[UptoDimensionStep][i].itemsSet[j] : dense2orig[MFIsPool[UptoDimensionStep][i].itemsSet[j]]) << \" \";//patched-by-drivers",
        "console-out",
    )
    s = _replace_once(
        s,
        "                outfile << MFIsPool[Step][i].itemsSet[j] << \" \";",
        "                outfile << (dense2orig.empty() ? MFIsPool[Step][i].itemsSet[j] : dense2orig[MFIsPool[Step][i].itemsSet[j]]) << \" \";//patched-by-drivers",
        "file-out",
    )
    open(kp, "w", encoding="utf-8", newline="\n").write(s)


def ensure_anyfim_built(dataset_path, upto_stage, rebuild=False, dense=True):
    """确保（必要时构建）指定数据集/轮数的 AnytimeMining.exe，返回 exe 路径。

    dense=True 打稠密重映射补丁（默认）；dense=False 用于消融对照。
    """
    tag = "dense" if dense else "nodense"
    stamp = os.path.join(BUILD_DIR, f"built_for_stage{int(upto_stage)}_{tag}.stamp")
    exe = os.path.join(SRC_DIR, "x64", "Release", "AnytimeMining.exe")
    if rebuild or not os.path.exists(stamp) or not os.path.exists(exe):
        if not os.path.isdir(ENGINE_SRC):
            raise FileNotFoundError(f"引擎源码不存在：{ENGINE_SRC}（请先下载 AnyFIM 到 engines/）")
        _copy_sources()
        _patch_kernel(upto_stage)
        if dense:
            _patch_dense_remap()
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
