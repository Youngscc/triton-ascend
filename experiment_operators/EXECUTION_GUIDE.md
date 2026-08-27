# A3/A5 三参数实验操作手册

本手册只描述当前正式流程。实验源码使用 `codex/experiment-main-dev`，基于
Triton-Ascend `upstream/main-dev`。所有命令都在环境机器上的项目根目录执行；
除创建和进入容器外，构建与实验命令都在容器内执行。

## 1. 创建 Docker 环境

A5 实验使用以下 `linux/amd64` devel 镜像作为环境基线：

```text
quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12-devel
```

若使用镜像归档文件，其内容应当就是这个 CANN 9.1.0、Ascend 950、Ubuntu 22.04、Python 3.12 的 devel 镜像。用它创建自己的实验容器；容器名、NPU 卡号和挂载路径按所在机器设置，不属于项目配置。

A3 使用同样为 CANN 9.1.0 的 A3 devel 镜像。容器必须能够访问所选 NPU，并把项目挂载到宿主机与容器内相同的绝对路径。

进入自己创建的容器后：

```bash
cd <项目在容器内的绝对路径>
```

实验脚本会根据 `torch.npu.get_device_name(0)` 自动选择 A3 或 A5 流程。

## 2. 准备源码

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

确认两个版本：

```bash
git rev-parse --short HEAD
git rev-parse --short HEAD:third_party/ascend/AscendNPU-IR
```

该分支固定的 AscendNPU-IR gitlink 是 `f67dc61f0`。它以兼容基线
`aea934a66` 为基础，包含实验所需的参数暴露、原生 CV depth 语义，以及在
调用旧版 `hivmc-a5` 前执行的 SSBuffer 兼容转换层。

## 3. 维护环境配置

首次使用或项目绝对路径发生变化时，编辑本机配置：

```bash
vi tools/remote_experiment/config.local.sh
```

确保 `REMOTE_PROJECT` 是当前项目的绝对路径：

```bash
REMOTE_PROJECT="<当前项目绝对路径>"
```

`config.sh` 会根据它重新生成 `REMOTE_VENV`、`REMOTE_COMPILER_BUILD`、缓存和
临时目录等路径。若不创建 `config.local.sh`，则自动使用当前 checkout。

## 4. 首次构建

仍在容器内执行：

```bash
./tools/remote_experiment/setup-dev-environment.sh
./tools/remote_experiment/rebuild-compiler.sh
source tools/remote_experiment/activate-dev-environment.sh
```

激活成功时会输出 `DEV_ENVIRONMENT_OK`。A5 还会输出 `A5_HIVMC_OK`。构建结果位于 `.codex-remote/`，不会写入源码目录。

### 可选：使用离线 Triton LLVM

`setup-dev-environment.sh` 首次构建 Triton 时可能需要下载当前源码指定的预编译 LLVM。环境机器不能下载时，先在相同 CPU 架构、相同容器镜像和相同 Triton commit 的联网环境完成一次构建，然后找到下载出的目录：

```bash
find ~/.triton/llvm -mindepth 1 -maxdepth 1 -type d -name 'llvm-*' -print
```

在联网环境把匹配目录整体打包：

```bash
LLVM_ROOT=$HOME/.triton/llvm/llvm-<matching-version>
tar -C "$(dirname "$LLVM_ROOT")" -czf triton-llvm.tar.gz \
  "$(basename "$LLVM_ROOT")"
```

将 `triton-llvm.tar.gz` 传到环境机器并解压。解压后的 `LLVM_ROOT` 必须直接包含 `bin`、`include` 和 `lib`：

```bash
mkdir -p "$HOME/triton-llvm"
tar -xzf /path/to/triton-llvm.tar.gz -C "$HOME/triton-llvm"
LLVM_ROOT=$HOME/triton-llvm/llvm-<matching-version>
test -x "$LLVM_ROOT/bin/FileCheck"
test -f "$LLVM_ROOT/lib/cmake/mlir/MLIRConfig.cmake"
test -f "$LLVM_ROOT/lib/cmake/lld/LLDConfig.cmake"
```

指定该目录构建 Triton：

```bash
LLVM_SYSPATH="$LLVM_ROOT" \
  ./tools/remote_experiment/setup-dev-environment.sh
```

这个 LLVM 只供顶层 Triton 构建使用；不能拿 AscendNPU-IR submodule 内的 LLVM 代替，也不要复用其他 Triton commit 下载出的目录。

以后每次重新进入容器，只需：

```bash
cd <宿主机项目的同一绝对路径>
source tools/remote_experiment/activate-dev-environment.sh
```

### 清除编译产物与缓存

需要排除旧构建或旧缓存影响时，先停止正在运行的构建和实验，然后在容器内执行：

```bash
./tools/remote_experiment/clean-build-cache.sh
```

脚本只删除当前项目内白名单中的 Triton CMake/扩展产物、AscendNPU-IR
build、项目专用 `ccache`、Triton kernel cache、Python cache 和 `./tmp`。它不会删除
`.codex-remote/results`、日志、venv、Git 元数据或已经下载的 Triton LLVM。
清理完成会输出 `CLEAN_BUILD_CACHE_OK`，之后重新构建：

```bash
./tools/remote_experiment/setup-dev-environment.sh
./tools/remote_experiment/rebuild-compiler.sh
source tools/remote_experiment/activate-dev-environment.sh
```

## 5. 设置实验参数

所有可调整的实验取值只在一个文件中：

```text
experiment_operators/experiment_config.py
```

主要配置为：

```python
A3_DEPTH_VALUES = (1, 2, 3, 4)
A5_BUF_SLOT_NUM_OF_VECCORE_VALUES = ("off", 1, 2, 3, 4)
MULTIBUFFER_NUM_VALUES = ("off", 1, 2, 3, 4)
VF_MERGE_LEVEL_VALUES = (0, 1)

WARMUP = 5
ACTIVE = 30
CASE_TIMEOUT_SECONDS = 120
TIMEOUT_RETRIES = 1
```

三条实验轴的含义：

| 配置项 | A3 | A5 |
| --- | --- | --- |
| `A3_DEPTH_VALUES` | 静态 CV `depth`，DynamicCV 固定关闭 | 不使用 |
| `A5_BUF_SLOT_NUM_OF_VECCORE_VALUES` | 不使用 | 通过 `--set-cv-pipeline-mode=off` 始终关闭静态 CVPipeline；`"off"` 同时关闭 DynamicCV，数字表示仅开启 DynamicCV 并设置 `buf_slot_num_of_veccore` |
| `MULTIBUFFER_NUM_VALUES` | `"off"` 关闭普通 MultiBuffer；数字表示开启并设置 local buffer 数量 | 同 A3 |
| `VF_MERGE_LEVEL_VALUES` | `0` 关闭 VF merge，`1` 开启 level 1 | 同 A3 |

`"off"` 是真实关闭状态：A5 第一轴的 `"off"` 同时关闭静态和动态 CVPipeline；MultiBuffer 的 `"off"` 会传入 `multibuffer=False`，不传 `--set-local-multibuffer`。数值 `1` 仍然开启对应 pass，只是使用一个 buffer。配置顺序就是运行顺序，因此默认先跑关闭状态。默认配置下，A3 有 40 行，A5 有 50 行，其中 A5 前 10 行关闭全部 CVPipeline。`vf_merge_level=2` 当前不在默认配置中；编译器问题修复后，直接把 `2` 加回配置文件即可。

UnitFlag synchronization 不是实验轴，算子不会显式传入 `unit_flag`。当前固定的兼容编译器在 A3 和 A5 上都采用关闭默认值。

## 6. 运行完整实验

命令只有一个算子文件参数：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

其他候选算子：

```bash
./run_all_sweeps.sh experiment_operators/candidates/flash_attention_npu_v8.py
./run_all_sweeps.sh experiment_operators/candidates/unified_attention.py
./run_all_sweeps.sh experiment_operators/candidates/hstu_attention.py
```

前台终端固定使用两行显示：第一行是总进度及成功、失败、不支持数量，第二行是当前参数；两行会原地刷新，不会为每个 case 新建进度条。无 TTY 的后台日志使用 `CASE_START` 和 `CASE_RESULT` 追加事件，便于 `tail -F` 查看。超时 case 会先排队，等其他初始 case 全部结束后再补测。

一次完整运行默认生成所有报告所需内容：

```text
.codex-remote/results/<run-id>-<operator>/
├── manifest.json
├── measurements.jsonl
├── results.csv
└── logs/
    └── <case>.log
```

- `results.csv`：直接查看每个组合的成功、失败、不支持、Bisheng
  编译耗时、运行延迟、UB 和原因。`Bisheng编译耗时_ms` 从 TTAdapter
  输入进入 `bishengir-compile` 开始计时，到 NPU 二进制生成后返回为止；
  它不包含正确性检查和 benchmark。
- `measurements.jsonl`：唯一的完整机器记录，包含编译审计字段和尝试历史。
- `manifest.json`：实验取值、源码版本、编译器依赖版本和测量策略。
- `logs/<case>.log`：该组合的完整编译、正确性和 benchmark 输出。

延迟来自 upstream 的 CANN NPU profiler 流程；实验层不修改 profiler schedule、
不调用额外的 `prof.step()`，也不提供 NPU Event fallback。

运行完成后会自动刷新：

```text
.codex-remote/results/latest-summary/experiment-report.html
.codex-remote/results/latest-summary/combined-results.csv
```

`combined-results.csv` 与 HTML 使用完全相同的各算子最新完整实验目录，保留每份
`results.csv` 的全部行，并增加算子、run ID 和结果目录三列用于追溯来源。

## 7. 补测一个 case

单 case 命令的后三项依次为：第一轴、`multibuffer_num`、`vf_merge_level`。

A3 示例：

```bash
./run_all_sweeps.sh --case \
  experiment_operators/candidates/fused_attention.py 3 2 1
```

A5 DynamicCV 开启示例：

```bash
./run_all_sweeps.sh --case \
  experiment_operators/candidates/fused_attention.py 4 2 1
```

A5 DynamicCV 关闭基线示例：

```bash
./run_all_sweeps.sh --case \
  experiment_operators/candidates/fused_attention.py off 2 1
```

关闭普通 MultiBuffer 时，第二个值写 `off`；下面的例子同时关闭 DynamicCV、普通 MultiBuffer 和 VF merge：

```bash
./run_all_sweeps.sh --case \
  experiment_operators/candidates/fused_attention.py off off 0
```

补测只会修改当前架构下该算子最新完整记录中已经存在的那一行：

- 原行最终状态是超时：直接运行，完成后替换该行。
- 原行不是超时：终端询问是否覆盖；输入 `y` 才运行，其他输入直接跳过。
- 每次补测都追加到原 case 日志，并记录尝试次数和手动补测次数。
- 更新完成后自动刷新 HTML。

## 8. 单独刷新 HTML

已有结果不需要重新实验：

```bash
./experiment_operators/generate_latest_report.sh
```

生成器会为每个算子选择最新完整记录。失败和不支持行仍保留在覆盖率与配置表中，不会被静默删除，也不会选择所谓最佳配置。

## 9. 何时重新构建

| 修改内容 | 操作 |
| --- | --- |
| `experiment_config.py`、候选算子、实验控制器或报告代码 | 不重新编译 |
| Triton Python 代码 | 通常不重新编译 |
| Triton C++/MLIR 核心 | `./tools/remote_experiment/setup-dev-environment.sh` |
| `third_party/ascend/AscendNPU-IR` 编译器代码 | `./tools/remote_experiment/rebuild-compiler.sh` |
| 仅文档或 shell 调用方式 | 不重新编译 |
