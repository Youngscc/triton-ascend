# Triton-Ascend A3/A5 单算子实验手册

服务器仓库是主工作区。构建和前台实验在已有容器内执行，服务器项目必须以相同
绝对路径挂载进容器。

当前默认实验遍历以下 32 组配置：

```text
depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..1)
```

`vf_merge_level=2` 因 A5 RegBase 编译器的 dominance 错误暂时排除。仅在验证
该问题时设置 `SWEEP_INCLUDE_VF_MERGE_LEVEL_2=1`，恢复全部 48 组。

正式 sweep 对每组配置固定 `enable_dynamic_cv_pipeline=false`，避免 A5 前端
动态 CV 路径把请求的 `depth` 改写为 0。BishengIR 仍可根据 Triton kernel 的
`mix_mode` 自动识别 MixedCV。显式的 `multibuffer_num` 同时固定
`limit_auto_multi_buffer_buffer=no-limit`：四个 count 值在同一个策略下比较，
并允许普通 multibuffer 作用到 MIX 函数 Vector 侧的 UB Load/Store。未显式传入
`multibuffer_num` 时仍使用上游默认的 `only-cube`，这种默认运行不属于正式
count 轴的对照数据。

| 环境 | Python/CANN | bitcode | A5 RegBase |
| --- | --- | --- | --- |
| A3 | Python 3.11 / CANN 9.0 | `c220` | 关闭 |
| A5/Ascend 950 | Python 3.12 / CANN 9.1 | `c310` | 开启 |

## 1. 准备仓库和容器配置

在服务器宿主机执行：

```bash
git clone --recurse-submodules git@github.com:Youngscc/triton-ascend.git
cd triton-ascend

cp tools/remote_experiment/config.local.sh.example \
  tools/remote_experiment/config.local.sh
vi tools/remote_experiment/config.local.sh
```

填写：

```bash
REMOTE_PROJECT="/服务器绝对路径/triton-ascend"
REMOTE_CONTAINER="已有实验容器名"
```

项目已存在时更新：

```bash
git fetch origin
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

检查容器与挂载：

```bash
source tools/remote_experiment/config.sh
docker inspect "$REMOTE_CONTAINER" --format '{{.State.Status}}'
docker exec "$REMOTE_CONTAINER" test -d "$REMOTE_PROJECT"
```

预期状态为 `running`，项目目录检查退出码为 0。

## 2. 首次构建

进入容器：

```bash
source tools/remote_experiment/config.sh
docker exec -u root -it "$REMOTE_CONTAINER" bash
cd "$REMOTE_PROJECT"
source tools/remote_experiment/config.sh
```

检查基础环境：

```bash
python3 --version
ninja --version
clang-15 --version | head -1
ld.lld-15 --version | head -1
command -v ccec
command -v hivmc
npu-smi info
```

缺少常用 Python 包或 Debian 包时：

```bash
python3 -m pip install <package>
apt-get update && apt-get install -y <package>
```

A5 使用：

```text
torch==2.7.1
torch-npu==2.7.1.post8
```

构建当前仓库的 Triton-Ascend 和 BishengIR：

```bash
JOBS=32 ./tools/remote_experiment/setup-dev-environment.sh
JOBS=32 ./tools/remote_experiment/rebuild-compiler.sh
```

无法访问 Triton LLVM 下载地址的环境（包括当前 A5 离线环境）必须先定位已解压
的 LLVM，并为 Triton 构建显式启用离线模式：

```bash
FILECHECK=$(find "$PWD/.codex-remote/llvm" -type f -name FileCheck | head -1)
LLVM_ROOT=$(dirname "$(dirname "$FILECHECK")")
test -x "$LLVM_ROOT/bin/FileCheck" && echo LLVM_OK

TRITON_OFFLINE_BUILD=1 \
LLVM_SYSPATH="$LLVM_ROOT" \
TRITON_PARALLEL_LINK_JOBS=2 \
JOBS=16 \
  ./tools/remote_experiment/setup-dev-environment.sh

JOBS=16 ./tools/remote_experiment/rebuild-compiler.sh
```

`LLVM_ROOT` 必须是直接包含 `bin/`、`include/` 和 `lib/` 的目录。离线构建
检查应先输出 `LLVM_OK`，并且构建期间不应再尝试下载 LLVM。

成功输出包含：

```text
MLIR_BYTECODE_ROUNDTRIP_OK
TRITON_DEV_IMPORT_OK
BISHENGIR_PACKAGE_OK soc=<设备型号> bitcode_arch=<c220或c310>
```

`setup-dev-environment.sh` 创建 `.codex-remote/venv`，复用容器的 Torch、
Torch-NPU 和 CANN，并构建当前 checkout 的 Triton。`rebuild-compiler.sh` 构建
项目固定版本的 BishengIR 和对应设备的 bitcode。

## 3. 每次进入容器后激活环境

```bash
cd "$REMOTE_PROJECT"
source tools/remote_experiment/activate-dev-environment.sh
```

预期输出：

```text
DEV_ENVIRONMENT_OK soc=<设备型号> bitcode_arch=<c220或c310> native_a5_regbase=<0或1>
```

快速检查：

```bash
which python
python - <<'PY'
import os
import pandas
import triton
from triton._C import libtriton

print("triton:", triton.__file__)
print("libtriton:", libtriton.__file__)
print("compiler:", os.environ["TRITON_NPU_COMPILER_PATH"])
PY
```

Python 应位于 `.codex-remote/venv/bin`，Triton、`libtriton` 和编译器均应来自
当前项目。不要只激活 venv；完整环境使用上述激活脚本。

## 4. 冒烟验证

查看空闲卡：

```bash
npu-smi info
```

运行 Vector Add：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> \
  python -u third_party/ascend/tutorials/01-vector-add.py
```

单卡容器可省略 `ASCEND_RT_VISIBLE_DEVICES`。预期输出：

```text
The maximum difference between torch and triton is 0.0
======Vector Add Test Passed!======
```

运行一组实验配置：

```bash
SWEEP_LIMIT=1 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

预期终端显示 `实验完成：成功=1 失败=0 不支持=0`，生成的 `results.csv`
包含一行结果。

### A5 mismatch 快速诊断

当静态 CV depth 出现大规模 correctness mismatch 时，先固定
`multibuffer_num=1` 和默认的 `vf_merge_level=1`，只比较 dynamic CV 与少量
静态 depth。容器内执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> \
  python -u experiment_operators/diagnose_a5_mismatch.py --operator fused
```

单卡容器可省略设备变量。该命令只跑 `F-DYN`、`F-D4`、`F-D3`、`F-D1`
四例，使用各自独立的临时缓存，结束后自动删除。终端不会输出 IR，只显示：

```text
CASE F-DYN result=PASS
CASE F-D4 result=PASS
CASE F-D3 result=MISMATCH count=... total=... max_abs=... lhs_zero=... rhs_zero=... chunks=...,...,...,...
CASE F-D1 result=...
CONCLUSION ...
```

手工反馈时只需提供这四个 `CASE` 和最后的 `CONCLUSION`。其中 `chunks` 是将
输出连续等分为四段后的 mismatch 数，可以快速判断错误是否集中在某个流水
区间。继续诊断 HSTU 或 Unified 时分别执行：

```bash
python -u experiment_operators/diagnose_a5_mismatch.py --operator hstu
python -u experiment_operators/diagnose_a5_mismatch.py --operator unified
```

`--operator all` 会串行运行全部十例，仅在单算子结论不足时使用。

## 5. 运行完整实验

容器内前台运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

命令只接收一个算子 Python 文件。单卡容器可省略设备变量。正式实验默认每组
5 次 warmup、30 次 active 测量和 120 秒超时；失败配置仍会记录并继续运行。

每个算子显示一条总进度条，下一行显示当前
`depth`、`multibuffer_num`、`vf_merge_level` 以及累计的 `success`、`failed`、
`unsupported`。其中 `success` 对应持久化状态 `measured`，`unsupported` 对应
同名状态，其余编译失败和正确性失败计入 `failed`。前台终端原地刷新两行；
后台运行输出可由 `logs.sh latest` 读取的纯文本进度快照。可用
`SWEEP_PROGRESS_MODE=plain` 强制纯文本，或用 `SWEEP_PROGRESS_MODE=off` 关闭。

默认结果目录只包含一个 `results.csv`。完整实验应有 32 行，每一行对应唯一的
`(depth, multibuffer_num, vf_merge_level)`，`结果` 列直接显示 `成功`、
`失败` 或 `不支持`。其余列只保留简短原因、延迟、UB 和该轮总耗时；
不显示缓存键、哈希、二进制路径或完整编译命令。

每轮缓存 metadata 还必须满足
`set_workspace_multibuffer == depth`、
`enable_dynamic_cv_pipeline == false` 和
`limit_auto_multi_buffer_buffer == no-limit`；不满足时该产物不会被计为有效测量。

在服务器宿主机后台运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> REMOTE_MODE=dev \
  ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py

./tools/remote_experiment/logs.sh latest
```

`Ctrl-C` 只停止日志跟随，不终止后台实验。

只有排查编译器问题时才启用详细审计模式：

```bash
SWEEP_DETAILED_OUTPUT=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

该模式才会额外记录完整参数、编译命令、每轮日志、哈希和聚合报告；正式查看
32 种默认配置结果时不要启用。

## 6. 结果和报告

结果位于：

```text
.codex-remote/results/<UTC+8时间>-<operator>/
```

刷新所有算子的最新完整结果和 HTML：

```bash
source tools/remote_experiment/activate-dev-environment.sh
python experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

报告位于：

```text
.codex-remote/results/latest-summary/experiment-report.html
```

## 7. 常用维护

```bash
# 预览后清理 venv 和构建产物
./tools/remote_experiment/clean-environment.sh rebuild
./tools/remote_experiment/clean-environment.sh rebuild --execute

# 清理日志和 Triton cache
./tools/remote_experiment/clean-environment.sh runtime --execute

# 每个算子只保留最新完整结果
./tools/remote_experiment/clean-environment.sh latest-results --execute

# 删除全部实验结果
./tools/remote_experiment/clean-environment.sh results --execute
```

服务器无法访问 GitHub 时，可在个人电脑的 `config.local.sh` 额外填写：

```bash
LOCAL_PROJECT="/本地绝对路径/triton-ascend"
REMOTE_HOST="服务器SSH别名"
REMOTE_SOURCE_MODE="rsync"
```

然后从个人电脑执行：

```bash
./tools/remote_experiment/sync.sh
./tools/remote_experiment/pull-results.sh
```

`sync.sh` 只传源码和必要的 Git 元数据，统一排除 `__pycache__`、Python
字节码、测试/覆盖率缓存、虚拟环境、构建目录、本机动态库以及
生成的 Triton 工具和 Python 包。排除项在服务器上已有的内容会保留，因而
不会删除服务器自身构建的环境、缓存和实验结果。若旧版同步曾带入电脑端的
`python/triton/_C/libtriton.so`，需在容器内执行一次：

```bash
./tools/remote_experiment/clean-environment.sh rebuild --execute
```

随后按本章的离线 LLVM 参数重新执行环境和编译器构建。
