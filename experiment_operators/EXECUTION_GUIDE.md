# A3/A5 三参数实验操作手册

本手册只描述当前正式流程。所有命令都在环境机器上的项目根目录执行；除创建和进入容器外，构建与实验命令都在容器内执行。

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

## 3. 首次构建

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

## 4. 设置实验参数

所有可调整的实验取值只在一个文件中：

```text
experiment_operators/experiment_config.py
```

主要配置为：

```python
A3_DEPTH_VALUES = (1, 2, 3, 4)
A5_INTRA_CACHE_NUM_VALUES = ("off", 1, 2, 3, 4)
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
| `A5_INTRA_CACHE_NUM_VALUES` | 不使用 | `"off"` 关闭 DynamicCV；数字表示开启并设置 `intra_cache_num` |
| `MULTIBUFFER_NUM_VALUES` | `"off"` 关闭普通 MultiBuffer；数字表示开启并设置 local buffer 数量 | 同 A3 |
| `VF_MERGE_LEVEL_VALUES` | `0` 关闭 VF merge，`1` 开启 level 1 | 同 A3 |

`"off"` 是真实关闭状态：MultiBuffer 会传入 `multibuffer=False`，不传 `--set-local-multibuffer`；数值 `1` 仍然开启该 pass，只是使用一个 buffer。配置顺序就是运行顺序，因此默认先跑关闭状态。默认配置下，A3 有 40 行，A5 有 50 行，其中 A5 前 10 行关闭 DynamicCV。`vf_merge_level=2` 当前不在默认配置中；编译器问题修复后，直接把 `2` 加回配置文件即可。

## 5. 运行完整实验

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

每个 case 的进度行会显示三项参数、成功数、失败数和不支持数。超时 case 会先排队，等其他初始 case 全部结束后再补测。

一次完整运行默认生成所有报告所需内容：

```text
.codex-remote/results/<run-id>-<operator>/
├── manifest.json
├── measurements.jsonl
├── results.csv
└── logs/
    └── <case>.log
```

- `results.csv`：直接查看每个组合的成功、失败、不支持、延迟、UB 和原因。
- `measurements.jsonl`：唯一的完整机器记录，包含编译审计字段和尝试历史。
- `manifest.json`：实验取值、源码版本、编译器依赖版本和测量策略。
- `logs/<case>.log`：该组合的完整编译、正确性和 benchmark 输出。

运行完成后会自动刷新：

```text
.codex-remote/results/latest-summary/experiment-report.html
```

## 6. 补测一个 case

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

## 7. 单独刷新 HTML

已有结果不需要重新实验：

```bash
./experiment_operators/generate_latest_report.sh
```

生成器会为每个算子选择最新完整记录。失败和不支持行仍保留在覆盖率与配置表中，不会被静默删除，也不会选择所谓最佳配置。

## 8. 何时重新构建

| 修改内容 | 操作 |
| --- | --- |
| `experiment_config.py`、候选算子、实验控制器或报告代码 | 不重新编译 |
| Triton Python 代码 | 通常不重新编译 |
| Triton C++/MLIR 核心 | `./tools/remote_experiment/setup-dev-environment.sh` |
| `third_party/ascend/AscendNPU-IR` 编译器代码 | `./tools/remote_experiment/rebuild-compiler.sh` |
| 仅文档或 shell 调用方式 | 不重新编译 |
