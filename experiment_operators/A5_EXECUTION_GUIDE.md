# A5/Ascend 950 单算子实验执行手册

本手册适用于已经由用户创建并能正常启动的 A5 实验容器。服务器仓库是主工作
区；源码更新在服务器通过 Git 完成，构建和实验在已有容器内完成。本流程不创建、
删除或重建容器。

## 1. 在 A5 服务器准备源码

服务器能够访问 GitHub 时，在服务器宿主机直接 clone：

```bash
cd /服务器代码目录
git clone --recurse-submodules git@github.com:Youngscc/triton-ascend.git
cd triton-ascend
```

更新已有仓库：

```bash
git fetch origin
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

确认当前分支和依赖版本：

```bash
git branch --show-current
git status --short
git submodule status --recursive
```

## 2. 配置已有 A5 容器

在服务器仓库执行：

```bash
cp tools/remote_experiment/config.local.sh.example \
  tools/remote_experiment/config.local.sh
vi tools/remote_experiment/config.local.sh
```

填写服务器项目路径和实际容器名：

```bash
REMOTE_PROJECT="/服务器绝对路径/triton-ascend"
REMOTE_CONTAINER="已有A5容器名"
```

检查容器状态、项目挂载和设备节点：

```bash
source tools/remote_experiment/config.sh
docker inspect "$REMOTE_CONTAINER" --format '{{.State.Status}}'
docker exec "$REMOTE_CONTAINER" test -d "$REMOTE_PROJECT"
docker inspect "$REMOTE_CONTAINER" \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
docker exec "$REMOTE_CONTAINER" bash -c \
  'ls -l /dev/davinci* /dev/davinci_manager /dev/hisi_hdc 2>&1'
```

服务器宿主机和容器内的项目绝对路径必须相同。项目目录必须是持久化挂载，确保
`.codex-remote/venv`、编译产物、缓存和实验结果在容器重启后仍然存在。

## 3. 进入容器并检查 A5 环境

在服务器宿主机执行：

```bash
source tools/remote_experiment/config.sh
docker exec -u root -it "$REMOTE_CONTAINER" bash
cd "$REMOTE_PROJECT"
source tools/remote_experiment/config.sh
source tools/remote_experiment/load-cann-environment.sh
```

检查 CANN、Python、构建工具和 NPU：

```bash
python3 --version
cmake --version | head -1
ninja --version
clang-15 --version | head -1
clang++-15 --version | head -1
ld.lld-15 --version | head -1
command -v ccec || true
command -v hivmc
npu-smi info

python3 - <<'PY'
import acl
import torch
import torch_npu

print("CANN SoC name:", acl.get_soc_name())
print("visible NPU count:", torch.npu.device_count())
PY
```

环境检查必须满足：Python 为项目要求的 3.11，CMake 不低于 3.28，Ninja 不低于
1.12，Clang/Clang++ 与 LLD 为 15.x，`ccec` 和 `hivmc` 可执行，`npu-smi info`
能列出设备，并且 `visible NPU count` 大于 0。任一命令缺失、版本过低或设备数为
0，都应先修复环境，不能进入项目构建。

容器只挂一张物理卡时，程序通常看到逻辑 `npu:0`。不要把宿主机物理卡编号
直接当成容器逻辑编号。

## 4. 创建或修复 A5 项目 venv

当前仓库的 Triton core 还需要仓库固定版本的预编译 LLVM。A5 无法访问制品站时，
先把与容器架构匹配的 LLVM 解压到项目持久化目录。x86_64 A5 的目录应为：

```text
.codex-remote/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/
├── bin/llvm-config
├── include/
├── lib/
└── share/
```

先验证离线 LLVM 和宿主工具链：

```bash
test -x \
  "$PWD/.codex-remote/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/llvm-config" \
  && echo TRITON_LLVM_OK

source tools/remote_experiment/config.sh
"$REMOTE_HOST_CC" --version | head -1
"$REMOTE_HOST_CXX" --version | head -1
"$REMOTE_HOST_LD_LLD" --version | head -1
```

四条命令必须成功；LLVM 检查必须打印 `TRITON_LLVM_OK`，编译器必须是 Clang，
不能是 GCC。然后在容器内项目根目录执行：

```bash
TRITON_OFFLINE_BUILD=1 \
LLVM_SYSPATH="$PWD/.codex-remote/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64" \
TRITON_PARALLEL_LINK_JOBS=2 \
JOBS=4 \
./tools/remote_experiment/setup-dev-environment.sh
```

该命令负责创建和维护：

```text
$REMOTE_PROJECT/.codex-remote/venv
```

它会复用容器已有的 Torch、Torch-NPU 和 CANN，构建当前服务器 checkout 的
Triton-Ascend 与 `libtriton.so`，并执行 editable 安装。服务器直接 clone 的
仓库使用自身 `.git`，不需要 `.codex-remote/top-git`。

宿主侧构建必须使用 Clang。脚本会优先自动选择 Ubuntu 的
`clang-15`、`clang++-15`、`lld-15` 和 `ld.lld-15`；找不到 Clang 时直接失败，
不会回退到 GCC。`ccec`、BishengIR 与 `hivmc` 的设备侧编译链与宿主 Clang
是不同工具。

检查 venv 和导入位置：

```bash
test -x "$REMOTE_VENV/bin/python"
REMOTE_PROJECT="$REMOTE_PROJECT" REMOTE_VENV="$REMOTE_VENV" \
  PYTHONPATH="$REMOTE_PROJECT/python" "$REMOTE_VENV/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

import torch
import torch_npu
import triton
from triton._C import libtriton

project_python = (Path(os.environ["REMOTE_PROJECT"]) / "python").resolve()
venv = Path(os.environ["REMOTE_VENV"]).resolve()
python_executable = Path(sys.executable).resolve()
triton_file = Path(triton.__file__).resolve()
libtriton_file = Path(libtriton.__file__).resolve()

assert python_executable.is_relative_to(venv), python_executable
assert triton_file.is_relative_to(project_python), triton_file
assert libtriton_file.is_relative_to(project_python), libtriton_file

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("python:", python_executable)
print("triton:", triton_file)
print("libtriton:", libtriton_file)
print("TRITON_DEV_IMPORT_OK")
PY
```

没有问题的判据是命令退出码为 0，并打印
`MLIR_BYTECODE_ROUNDTRIP_OK` 和 `TRITON_DEV_IMPORT_OK`。前一个标记证明当前
LLVM 22 `triton-mlir-opt` 能生成 bytecode version 4，并且 LLVM 19.1.7
`bishengir-opt` 能读取其中的 `llvm.inttoptr`；后一个标记证明：

```text
python     位于 $REMOTE_VENV
triton     位于 $REMOTE_PROJECT/python/triton
libtriton  位于 $REMOTE_PROJECT/python/triton/_C
```

任何路径落入 `/usr/local/.../site-packages/triton` 都表示混用了容器预装版本，
不能继续接受编译和实验结果。`--system-site-packages` 只用于复用 Torch、
Torch-NPU 和 CANN；手动运行当前 checkout 时必须像上面一样把
`$REMOTE_PROJECT/python` 放到 `PYTHONPATH` 首位。正式实验入口会自动完成该设置。

运行时保持 `use_bytecode=true`。项目的 LLVM 22 `triton-mlir-opt` 固定写出
bytecode version 4，CANN 的 LLVM 19.1.7 `bishengir-opt` 将其解码为文本
TTAdapter IR，再交给仓库固定的 LLVM 19.1.7 `bishengir-compile`。bytecode
version 5 或 6 会携带 MLIR 19 无法读取的 native properties；日志中出现
`missing BytecodeOpInterface` 表示没有使用兼容版本 4。

出现 `development venv not found` 时，直接在当前容器和当前项目路径重新执行
`setup-dev-environment.sh`。不要从其他机器、容器或项目路径复制 venv。

## 5. 构建 A5 BishengIR

继续在容器内执行：

```bash
JOBS=32 ./tools/remote_experiment/rebuild-compiler.sh
```

当前构建包含 A5/NPUIR 开关：

```text
-DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON
-DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON
```

检查完整编译器包：

```bash
"$REMOTE_COMPILER_BUILD/bin/bishengir-compile" --version
for file in \
  meta_op.aic.c220.bc \
  meta_op.aiv.c220.bc \
  meta_op.mix.aic.c220.bc \
  meta_op.mix.aiv.c220.bc \
  host.bc; do
  test -s "$REMOTE_COMPILER_BUILD/lib/$file" || exit 1
done
echo BISHENGIR_PACKAGE_OK
```

`bishengir-compile --version` 必须成功，五个 bitcode 必须非空，最后必须打印
`BISHENGIR_PACKAGE_OK`。`bishengir-compile` 和相邻 bitcode 必须来自同一次构建。

## 6. 冒烟验证

容器只暴露一张卡时直接运行：

```bash
PYTHONPATH="$REMOTE_PROJECT/python" \
PATH="$REMOTE_COMPILER_BUILD/bin:$REMOTE_VENV/bin:$PATH" \
TRITON_NPU_COMPILER_PATH="$REMOTE_COMPILER_BUILD/bin" \
  "$REMOTE_VENV/bin/python" -u \
  third_party/ascend/tutorials/01-vector-add.py
```

Vector Add 命令必须以退出码 0 结束，打印通过信息，并且最大误差为 0 或处于脚本
给定容差内；编译错误、设备异常、超时或路径落入 `/usr/local` 都不算通过。

容器暴露多张卡时，先用 `npu-smi info` 选择健康空闲卡，再为当前命令指定：

```bash
ASCEND_RT_VISIBLE_DEVICES=<物理卡编号> \
PYTHONPATH="$REMOTE_PROJECT/python" \
PATH="$REMOTE_COMPILER_BUILD/bin:$REMOTE_VENV/bin:$PATH" \
TRITON_NPU_COMPILER_PATH="$REMOTE_COMPILER_BUILD/bin" \
  "$REMOTE_VENV/bin/python" -u \
  third_party/ascend/tutorials/01-vector-add.py
```

单配置实验：

```bash
SWEEP_LIMIT=1 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

该配置必须通过正确性检查，记录非零 UB 和 NPU latency。失败时终端会直接打印
完整错误，独立日志仍保存在结果目录。冒烟结果必须包含一行 `measured`，且该行的
`latency_ms > 0`、`required_ub_bits > 0`；`compile_failed`、`incorrect`、
`unsupported` 或缺失 UB 都不算通过。

## 7. 完整实验与日志

在容器内前台运行一个算子：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

在服务器宿主机后台运行：

```bash
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py

./tools/remote_experiment/logs.sh latest
```

多卡容器可在 `run.sh` 命令前添加：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> REMOTE_MODE=dev \
  ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

完整实验产生 48 行，覆盖：

```text
depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..2)
```

## 8. 结果和报告

结果保存在服务器项目：

```text
.codex-remote/results/<UTC+8时间>-<operator>/
```

在容器内刷新所有算子的最新完整结果：

```bash
source .codex-remote/venv/bin/activate
python experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

报告位于：

```text
.codex-remote/results/latest-summary/experiment-report.html
```

## 9. GitHub 不可达时的备用同步

仅当 A5 服务器不能连接 GitHub 时，在个人电脑配置 `LOCAL_PROJECT`、
`REMOTE_PROJECT`、`REMOTE_HOST`、`REMOTE_CONTAINER`，并设置：

```bash
REMOTE_SOURCE_MODE="rsync"
```

然后从个人电脑执行：

```bash
./tools/remote_experiment/sync.sh
```

源码同步排除 `.codex-remote`，不会覆盖服务器 venv、编译产物、缓存、日志和
结果。同步完成后仍然登录服务器、进入已有 A5 容器，并在容器内执行所有构建和
实验命令。
