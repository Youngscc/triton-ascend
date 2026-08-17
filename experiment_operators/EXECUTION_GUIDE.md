# Triton-Ascend A3/A5 单算子实验手册

本手册默认所有命令都在实验环境机器上执行；只有“准备离线材料”一节需要一台能
联网、且 CPU 架构与实验机器相同的 Linux 机器。实验源码、容器和结果均保存在
实验环境机器本地。

当前默认实验会根据设备选择第一轴，共遍历 32 组配置：

```text
A3: depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..1)
A5: intra_cache_num(1..4) × multibuffer_num(1..4) × vf_merge_level(0..1)
```

`vf_merge_level=2` 因 A5 RegBase 编译器的 dominance 错误暂时排除。仅在诊断该
问题时设置 `SWEEP_INCLUDE_VF_MERGE_LEVEL_2=1`，恢复全部 48 组。

| 环境 | 推荐基础镜像 | Python/Torch | bitcode |
| --- | --- | --- | --- |
| A3 | 与设备匹配的 CANN 9.0 devel 镜像 | Python 3.11 / Torch 2.7.1 | `c220` |
| A5/Ascend 950 | CANN 9.1.0 950 devel 镜像 | Python 3.12 / Torch 2.7.1 | `c310` |

下面给出的完整离线文件名和 Python 命令以 A5 x86 为准。A3 使用 CANN 9.0 时，
`torch-npu` 应改为与其匹配的 `2.7.1.post4`，并在 Python 3.11 的 A3 镜像中准备
wheelhouse；不要把 A5 的 Python 3.12 wheelhouse 混用到 A3。

## 快速入口

- 容器、Python 环境、Triton 和 BishengIR 都已经构建好：从[第 8 节](#8-每次进入容器后激活环境)开始。
- 容器和 Python 依赖已经就绪，但项目尚未构建：从[第 7 节](#7-首次构建)开始。
- 环境机器不能联网：先在联网机器完成[第 1 节](#1-在联网机器准备离线材料)，再从第 2 节开始。
- 环境机器可以联网：镜像可直接在第 2 节拉取，Python 包按第 6 节的在线命令安装。

## 1. 在联网机器准备离线材料

完全离线安装需要四类材料：Docker 镜像、Python wheelhouse、包含全部 submodule
的源码，以及与当前源码匹配的 Triton LLVM 预编译包。四者应在与实验机器相同
CPU 架构的 Linux 环境中准备；本项目的 A5 环境使用 `linux/amd64`。

### 1.1 下载并导出 CANN Docker 镜像

A5 使用包含编译工具的 `devel` 镜像：

```bash
mkdir -p triton-ascend-offline
cd triton-ascend-offline

IMAGE=quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12-devel
docker pull --platform linux/amd64 "$IMAGE"
docker image inspect "$IMAGE" --format '{{.RepoTags}} {{.Architecture}}'
docker save "$IMAGE" | gzip -1 > cann-9.1.0-950-py3.12-devel-amd64.tar.gz
sha256sum cann-9.1.0-950-py3.12-devel-amd64.tar.gz \
  > cann-9.1.0-950-py3.12-devel-amd64.tar.gz.sha256
```

预期架构为 `amd64`。不要手工解压这个文件；其中的 `layer.tar`、`json` 和
`VERSION` 是 Docker 镜像层，必须由 `docker load` 导入。

A3 应在 Quay Tags 页面选择与实际 SoC、操作系统、Python 和 CPU 架构匹配的
`devel` 标签，再用相同的 `docker pull`、`docker save` 流程导出，不要把 A5 的
`950` 镜像用于 A3。

若确认官方 devel 镜像缺少宿主编译工具，可在联网机器先构建完整派生镜像，再把
派生镜像而非基础镜像传到离线环境：

```bash
cat > Dockerfile.triton-ascend-env <<'EOF'
ARG CANN_BASE_IMAGE
FROM ${CANN_BASE_IMAGE}
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ca-certificates build-essential zlib1g-dev clang-15 lld-15 ccache && \
    rm -rf /var/lib/apt/lists/*
EOF

BUILD_IMAGE=triton-ascend-env:cann9.1.0-950-py3.12-amd64
docker build --platform linux/amd64 \
  --build-arg CANN_BASE_IMAGE="$IMAGE" \
  -t "$BUILD_IMAGE" -f Dockerfile.triton-ascend-env .
docker save "$BUILD_IMAGE" | gzip -1 \
  > triton-ascend-env-cann9.1.0-950-amd64.tar.gz
sha256sum triton-ascend-env-cann9.1.0-950-amd64.tar.gz \
  > triton-ascend-env-cann9.1.0-950-amd64.tar.gz.sha256

# 后续 wheelhouse 也用派生镜像准备。
IMAGE=$BUILD_IMAGE
```

使用派生镜像时，后续所有 `IMAGE=...` 均改成
`triton-ascend-env:cann9.1.0-950-py3.12-amd64`。

### 1.2 下载离线 Python wheelhouse

使用刚拉取的同一镜像解析依赖和下载 wheel，可避免 Python 版本、CPU 架构以及
`torch`/`torch-npu` 组合不匹配。先固定项目的直接依赖；随后在干净 venv 中让
pip 解析传递依赖、生成精确 lock，并用该 lock 下载全部 wheel：

```bash
mkdir -p python-offline/wheelhouse
cat > python-offline/requirements-a5-direct.txt <<'EOF'
pip==24.3.1
setuptools==75.8.0
wheel==0.45.1
cmake==3.31.10
ninja==1.11.1.1
pybind11==2.13.6
attrs==24.2.0
numpy==1.26.4
scipy==1.13.1
decorator==5.1.1
psutil==6.0.0
PyYAML==6.0.2
pandas==2.2.3
pytest==8.3.2
pytest-xdist==3.6.1
torch==2.7.1+cpu
torch-npu==2.7.1.post8
EOF

docker run --rm --platform linux/amd64 \
  -v "$PWD/python-offline:/out" "$IMAGE" bash -c '
set -euo pipefail
python3 -m venv /tmp/resolve
source /tmp/resolve/bin/activate

python -m pip install "pip==24.3.1" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
python -m pip install -r /out/requirements-a5-direct.txt \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu

python -m pip check
python -m pip freeze --all | LC_ALL=C sort \
  > /out/requirements-a5-py312-amd64.lock.txt
if grep -qiE "^triton(==| @ )" /out/requirements-a5-py312-amd64.lock.txt; then
  echo "unexpected PyPI triton dependency in lock" >&2
  exit 1
fi

python -m pip download --only-binary=:all: --dest /out/wheelhouse \
  -r /out/requirements-a5-py312-amd64.lock.txt \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu

python3 -m venv /tmp/offline-check
/tmp/offline-check/bin/python -m pip install --no-index \
  --find-links /out/wheelhouse \
  -r /out/requirements-a5-py312-amd64.lock.txt
/tmp/offline-check/bin/python -m pip check
'

tar -czf python-wheelhouse-py312-amd64.tar.gz python-offline
sha256sum python-wheelhouse-py312-amd64.tar.gz \
  > python-wheelhouse-py312-amd64.tar.gz.sha256
```

第一次 `pip check` 验证在线解析结果，第二次验证 wheelhouse 在完全不访问索引时
仍能独立安装。`--only-binary=:all:` 保证离线目录中只有 wheel；任何包没有适配
Python 3.12/x86-64 的 wheel 时，下载会直接失败而不是混入源码包。lock 中禁止出现
PyPI 的 `triton`；实验使用当前仓库源码构建的 Triton，另一个 `triton` wheel 会
造成 Python 文件与 `libtriton.so` 不匹配。

A3 必须在 A3 的 Python 3.11/CANN 9.0 镜像中重新执行整个解析和验证流程，将
`torch-npu` 改为 `2.7.1.post4`，并将 lock 和压缩包文件名中的 `a5-py312` 改为
`a3-py311`。不要手工复用 A5 生成的 lock 或 wheelhouse。

### 1.3 准备包含 submodule 的源码

```bash
git clone --branch experiment --recurse-submodules \
  git@github.com:Youngscc/triton-ascend.git
cd triton-ascend
git submodule sync --recursive
git submodule update --init --recursive
git status --short
cd ..

tar --exclude='triton-ascend/.codex-remote' \
  -czf triton-ascend-experiment-source.tar.gz triton-ascend
sha256sum triton-ascend-experiment-source.tar.gz \
  > triton-ascend-experiment-source.tar.gz.sha256
```

压缩包必须保留顶层 `.git`、`.git/modules` 和嵌套 submodule 内容。只复制普通
源码文件会导致构建脚本无法确定当前 commit 和依赖版本。

### 1.4 准备 Triton LLVM

Triton 顶层构建使用项目指定的预编译 LLVM；AscendNPU-IR 则使用它自己
submodule 中的 LLVM 源码，两者不能互相替代。最稳妥的离线准备方式是在同一
镜像和同一源码 commit 上完成一次联网的 `setup-dev-environment.sh`，然后打包
下载到 `~/.triton/llvm/` 下的实际 LLVM 目录：

```bash
find ~/.triton/llvm -mindepth 1 -maxdepth 1 -type d -name 'llvm-*' -print

# LLVM_ROOT 必须直接包含 bin、include 和 lib。
LLVM_ROOT=/root/.triton/llvm/<当前源码下载出的实际目录>
test -x "$LLVM_ROOT/bin/FileCheck"
tar -C "$(dirname "$LLVM_ROOT")" -czf triton-llvm-amd64.tar.gz \
  "$(basename "$LLVM_ROOT")"
sha256sum triton-llvm-amd64.tar.gz > triton-llvm-amd64.tar.gz.sha256
```

不要复用其他 Triton commit 的 LLVM 目录。联网环境若不需要离线备份，可跳过
本小节，让第 7 节首次构建自动下载当前 commit 对应的 LLVM。

## 2. 将离线材料放到环境机器

建议把大文件放到容量充足的持久化磁盘，而不是 `/run` 这类 tmpfs。示例目录：

```bash
mkdir -p /opt/triton-ascend-offline
cd /opt/triton-ascend-offline
```

将第 1 节生成的文件放入该目录后校验。基础镜像和派生镜像只校验实际传输的
那个；其余三类离线材料都要通过校验：

```bash
# 二选一。
sha256sum -c cann-9.1.0-950-py3.12-devel-amd64.tar.gz.sha256
# sha256sum -c triton-ascend-env-cann9.1.0-950-amd64.tar.gz.sha256

sha256sum -c python-wheelhouse-py312-amd64.tar.gz.sha256
sha256sum -c triton-ascend-experiment-source.tar.gz.sha256
sha256sum -c triton-llvm-amd64.tar.gz.sha256
```

环境机器能联网时，可只准备源码，并直接执行：

```bash
IMAGE=quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12-devel
docker pull "$IMAGE"
```

完全离线时导入镜像：

```bash
gzip -dc cann-9.1.0-950-py3.12-devel-amd64.tar.gz | docker load
IMAGE=quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12-devel
docker image inspect "$IMAGE" --format '{{.RepoTags}} {{.Architecture}}'
```

若准备的是上一节的派生镜像，则改为：

```bash
gzip -dc triton-ascend-env-cann9.1.0-950-amd64.tar.gz | docker load
IMAGE=triton-ascend-env:cann9.1.0-950-py3.12-amd64
docker image inspect "$IMAGE" --format '{{.RepoTags}} {{.Architecture}}'
```

导入镜像通常会额外占用接近压缩包解压后大小的 Docker 存储空间。确认
`docker info` 中的 `Docker Root Dir` 所在分区空间充足后再导入：

```bash
docker info --format 'root={{.DockerRootDir}} driver={{.Driver}}'
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

## 3. 检查宿主机和 NPU

```bash
uname -m
docker version --format '{{.Server.Version}}'
docker info --format '{{json .Runtimes}}'
npu-smi info
```

A5 x86 环境预期 `uname -m` 为 `x86_64`，Docker runtime 列表中包含 `ascend`，
并且 `npu-smi info` 能看到 Ascend 950。先确认要使用的物理卡空闲；以下示例使用
物理卡 7。

## 4. 准备项目源码

环境机器能访问 GitHub 时：

```bash
cd /home
git clone --branch experiment --recurse-submodules \
  git@github.com:Youngscc/triton-ascend.git
cd triton-ascend
git submodule sync --recursive
git submodule update --init --recursive
```

已有仓库更新时：

```bash
cd /home/triton-ascend
git fetch origin
git checkout experiment
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

完全离线时解压第 1 节准备的完整 checkout：

```bash
cd /home
tar -xzf /opt/triton-ascend-offline/triton-ascend-experiment-source.tar.gz
cd /home/triton-ascend
git status --short
git submodule status --recursive
```

## 5. 创建并进入实验容器

只选择一种设备注入方式。宿主机已配置 Ascend Docker Runtime 时，使用下面的
推荐命令；不要再同时添加 `--device=/dev/davinci7` 或
`ASCEND_RT_VISIBLE_DEVICES=7`，否则可能发生二次卡号映射。

```bash
cd /home/triton-ascend
PROJECT=$(pwd -P)
CONTAINER=triton-ascend-exp
IMAGE=quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12-devel

docker run -u root -dit \
  --name "$CONTAINER" \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=7 \
  --net=host \
  --security-opt seccomp=unconfined \
  --shm-size=64g \
  -v /opt/triton-ascend-offline:/opt/triton-ascend-offline:ro \
  -v "$PROJECT:$PROJECT" \
  -w "$PROJECT" \
  "$IMAGE" bash
```

容器中只暴露一张物理卡时，它是逻辑卡 `npu:0`。后续命令不要再把宿主机物理
编号 7 传给 `torch.npu.set_device` 或 `ASCEND_RT_VISIBLE_DEVICES`。

如果宿主机没有 Ascend Docker Runtime，改用显式设备挂载，且同样不要混用两套
方式：

```bash
docker run -u root -dit \
  --name "$CONTAINER" \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=7 \
  --net=host \
  --security-opt seccomp=unconfined \
  --shm-size=64g \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /opt/triton-ascend-offline:/opt/triton-ascend-offline:ro \
  -v "$PROJECT:$PROJECT" \
  -w "$PROJECT" \
  "$IMAGE" bash
```

进入容器：

```bash
docker start "$CONTAINER"
docker exec -u root -it "$CONTAINER" bash
cd /home/triton-ascend
```

## 6. 安装系统和 Python 依赖

先加载 CANN 并检查 devel 镜像中的编译工具：

```bash
source tools/remote_experiment/load-cann-environment.sh
python3 --version
command -v ccec
command -v hivmc
command -v bishengir-opt
command -v clang-15 || command -v clang
command -v ld.lld-15 || command -v ld.lld
command -v ninja
```

联网且缺少 Ubuntu 构建工具时：

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git ca-certificates build-essential zlib1g-dev clang-15 lld-15 ccache
```

离线环境不能用 pip wheel 补系统动态库、编译器或 `zlib1g-dev`。这些工具若不在
镜像中，应使用第 1.1 节的派生镜像命令；不要在离线机器上反复执行
`apt-get update`。

创建项目虚拟环境：

```bash
python3 -m venv --system-site-packages .codex-remote/venv
source .codex-remote/venv/bin/activate
```

联网安装 A5 Python 包：

```bash
python -m pip install "pip==24.3.1" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
python -m pip install \
  "pip==24.3.1" "setuptools==75.8.0" "wheel==0.45.1" \
  "cmake==3.31.10" "ninja==1.11.1.1" "pybind11==2.13.6" \
  "attrs==24.2.0" "numpy==1.26.4" "scipy==1.13.1" \
  "decorator==5.1.1" "psutil==6.0.0" "PyYAML==6.0.2" \
  "pandas==2.2.3" \
  "pytest==8.3.2" "pytest-xdist==3.6.1" \
  "torch==2.7.1+cpu" "torch-npu==2.7.1.post8" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip check
```

离线安装时，把只读离线包解压到项目的构建目录，再禁止 pip 访问网络：

```bash
mkdir -p .codex-remote/offline/python
tar -xzf /opt/triton-ascend-offline/python-wheelhouse-py312-amd64.tar.gz \
  -C .codex-remote/offline/python

python -m pip install --no-index \
  --find-links .codex-remote/offline/python/python-offline/wheelhouse \
  -r .codex-remote/offline/python/python-offline/requirements-a5-py312-amd64.lock.txt
```

检查环境：

```bash
python -m pip check
python - <<'PY'
import numpy
import pandas
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("visible NPU count:", torch.npu.device_count())
torch.npu.set_device(0)
print("logical npu:0:", torch.npu.get_device_name(0))
PY
```

单卡容器预期 `visible NPU count: 1`，设备名包含 Ascend 950。若数量为 0，不要
继续构建或实验，先检查容器设备注入方式。

## 7. 首次构建

### 7.1 联网构建

```bash
cd /home/triton-ascend
JOBS=16 TRITON_PARALLEL_LINK_JOBS=2 \
  ./tools/remote_experiment/setup-dev-environment.sh
JOBS=16 ./tools/remote_experiment/rebuild-compiler.sh
```

### 7.2 离线构建

解压第 1 节准备的 LLVM：

```bash
mkdir -p .codex-remote/llvm
tar -xzf /opt/triton-ascend-offline/triton-llvm-amd64.tar.gz \
  -C .codex-remote/llvm

FILECHECK=$(find "$PWD/.codex-remote/llvm" -type f -name FileCheck | head -1)
LLVM_ROOT=$(dirname "$(dirname "$FILECHECK")")
test -x "$LLVM_ROOT/bin/FileCheck" && echo LLVM_OK
test -f "$LLVM_ROOT/lib/cmake/mlir/MLIRConfig.cmake" && echo MLIR_OK
test -f "$LLVM_ROOT/lib/cmake/lld/LLDConfig.cmake" && echo LLD_OK
```

`LLVM_ROOT` 必须直接包含 `bin/`、`include/` 和 `lib/`，上述三项检查都必须
输出 `OK`。setup 脚本会将该目录下的 MLIR 和 LLD 配置路径显式传给 CMake，
避免复用 `LLD_DIR-NOTFOUND`。随后执行：

```bash
TRITON_OFFLINE_BUILD=1 \
LLVM_SYSPATH="$LLVM_ROOT" \
TRITON_PARALLEL_LINK_JOBS=2 \
JOBS=16 \
  ./tools/remote_experiment/setup-dev-environment.sh

JOBS=16 ./tools/remote_experiment/rebuild-compiler.sh
```

成功输出应包含：

```text
MLIR_BYTECODE_ROUNDTRIP_OK
TRITON_DEV_IMPORT_OK
BISHENGIR_PACKAGE_OK soc=<设备型号> bitcode_arch=<c220或c310>
```

`setup-dev-environment.sh` 构建当前 checkout 的 Triton 和 `libtriton.so`；
`rebuild-compiler.sh` 构建仓库 gitlink 指定的 AscendNPU-IR/BishengIR，并生成当前
SoC 对应的 `c220` 或 `c310` bitcode。

## 8. 每次进入容器后激活环境

```bash
cd /home/triton-ascend
source tools/remote_experiment/activate-dev-environment.sh
```

预期输出：

```text
DEV_ENVIRONMENT_OK soc=<设备型号> bitcode_arch=<c220或c310> native_a5_regbase=<0或1>
```

确认所有组件来自当前项目：

```bash
which python
python - <<'PY'
import os
import sys
import torch
import triton
from triton._C import libtriton

print("python prefix:", sys.prefix)
print("triton:", triton.__file__)
print("libtriton:", libtriton.__file__)
print("compiler:", os.environ["TRITON_NPU_COMPILER_PATH"])
print("device:", torch.npu.get_device_name(0))
PY
```

Python prefix 应位于 `.codex-remote/venv`；Triton、`libtriton.so` 和
`bishengir-compile` 都应来自当前项目目录。不要只执行 venv 的 `activate`，完整
实验环境必须使用本节的激活脚本。

## 9. 冒烟验证

运行 Vector Add：

```bash
python -u third_party/ascend/tutorials/01-vector-add.py
```

单卡容器直接使用逻辑卡 `npu:0`，不再设置物理卡号。预期输出包含：

```text
The maximum difference between torch and triton is 0.0
======Vector Add Test Passed!======
```

只运行一组实验配置：

```bash
SWEEP_LIMIT=1 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

预期显示 `实验完成：成功=1 失败=0 不支持=0`，并生成只有一行数据的
`results.csv`。冒烟通过后再运行完整实验。

## 10. 运行完整实验

每条命令只运行一个算子，默认每组 5 次 warmup、30 次 active 测量和 120 秒
超时：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
./run_all_sweeps.sh experiment_operators/candidates/unified_attention.py
./run_all_sweeps.sh experiment_operators/candidates/hstu_attention.py
./run_all_sweeps.sh experiment_operators/candidates/flash_attention_npu_v8.py
```

每个算子的正式实验包含 32 行。A3 每行对应唯一的
`(depth, multibuffer_num, vf_merge_level)`；A5 每行对应唯一的
`(intra_cache_num, multibuffer_num, vf_merge_level)`。失败或不支持的配置不会被
丢弃。终端会持续显示当前参数以及成功、失败、不支持数量；每轮
结束后还会保留一行不被进度条覆盖的核心日志，例如：

```text
CASE 1/32 key=i1-b1-m0 intra_cache_num=1 multibuffer_num=1 vf_merge_level=0 结果=成功 status=measured latency_ms=... ub_kib=... wall_time_s=... log=logs/i1-b1-m0.log
```

失败或不支持行会额外显示简短原因，默认不会把完整 IR 打印到终端。每个
case 的完整 stdout/stderr 都会单独保存到同一结果目录下的 `logs/<case>.log`；
编译错误、pipeline 诊断、正确性 mismatch 和 benchmark 输出都在该文件中。

A3 元数据必须满足：

```text
set_workspace_multibuffer == depth
enable_dynamic_cv_pipeline == false
limit_auto_multi_buffer_buffer == no-limit
```

A5 元数据必须满足：

```text
intra_cache_num == 请求值
inter_cache_num == 1
load_cache_num == 1
set_workspace_multibuffer == 0
enable_dynamic_cv_pipeline == true
limit_auto_multi_buffer_buffer == no-limit
```

DynamicCV 若回退为 false，该配置会记录为“不支持”，不会混入有效测量。

可调整运行策略：

```bash
# 纯文本进度，适合重定向到一个日志文件
SWEEP_PROGRESS_MODE=plain \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py \
  2>&1 | tee fused_attention.log

# 仅诊断时恢复 vf_merge_level=2，共 48 组
SWEEP_INCLUDE_VF_MERGE_LEVEL_2=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py

# 仅排查编译器时额外保留哈希、manifest 和完整审计信息
SWEEP_DETAILED_OUTPUT=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

## 11. 查看结果和生成报告

默认结果位于：

```text
.codex-remote/results/<UTC+8时间>-<operator>/results.csv
.codex-remote/results/<UTC+8时间>-<operator>/logs/<case>.log
```

`results.csv` 每组一行，直接显示 `成功`、`失败` 或 `不支持`，并保留简短原因、
延迟、UB 使用量、该轮总耗时和对应日志文件。`logs/` 中每组一个完整日志，
例如 A5 的 `i3-b2-m1.log` 或 A3 的 `d3-b2-m1.log`。

详细模式下刷新所有算子的最新完整结果和 HTML：

```bash
source tools/remote_experiment/activate-dev-environment.sh
python experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

报告位于：

```text
.codex-remote/results/latest-summary/experiment-report.html
```

## 12. 何时需要重新构建

| 改动 | 操作 |
| --- | --- |
| 只修改实验控制器、候选算子参数、报告或文档 | `git pull` 后直接运行，不重新构建 |
| 修改顶层 `python/`、Triton C++/MLIR 或顶层 LLVM 版本 | 重新执行 `setup-dev-environment.sh` |
| 修改 AscendNPU-IR 源码、其 nested submodule 或顶层 gitlink | 重新执行 `rebuild-compiler.sh` |
| 新机器、删除 `.codex-remote/venv` 或首次使用该 checkout | 两个构建脚本都执行 |

## 13. 清理

```bash
# 先预览，再清理 venv 和构建产物
./tools/remote_experiment/clean-environment.sh rebuild
./tools/remote_experiment/clean-environment.sh rebuild --execute

# 清理日志和 Triton cache
./tools/remote_experiment/clean-environment.sh runtime --execute

# 每个算子只保留最新完整结果
./tools/remote_experiment/clean-environment.sh latest-results --execute

# 删除全部实验结果
./tools/remote_experiment/clean-environment.sh results --execute
```
