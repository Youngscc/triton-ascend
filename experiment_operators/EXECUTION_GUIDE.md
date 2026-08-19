# Triton-Ascend A3/A5 单算子实验手册

本手册默认所有命令都在实验环境机器上执行；只有“准备离线材料”一节需要一台能
联网、且 CPU 架构与实验机器相同的 Linux 机器。实验源码、容器和结果均保存在
实验环境机器本地。

当前默认实验会根据设备选择配置空间：

```text
A3: depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..1)
A5 首先运行 DynamicCV 关闭组:
    dynamic_cv=false × multibuffer_num(1..4) × vf_merge_level(0..1)
A5 随后运行 DynamicCV 开启组:
    intra_cache_num(1..4) × multibuffer_num(1..4) × vf_merge_level(0..1)
```

`vf_merge_level=2` 因 A5 RegBase 编译器的 dominance 错误暂时排除。仅在诊断该
问题时设置 `SWEEP_INCLUDE_VF_MERGE_LEVEL_2=1`。默认 A3 为 32 组、A5 为 40
组；启用 level 2 后分别为 48 组和 60 组。

关闭语义必须按参数本身判断：`enable_dynamic_cv_pipeline=false` 才是关闭
DynamicCV，`intra_cache_num=1` 仍属于 DynamicCV 开启组；
`multibuffer_num=1` 是普通 local buffer 的单缓冲基线；
`vf_merge_level=0` 表示不运行 VF merge pass。

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
mkdir -p ./tmp
python3 -m venv ./tmp/resolve
source ./tmp/resolve/bin/activate

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

mkdir -p ./tmp
python3 -m venv ./tmp/offline-check
./tmp/offline-check/bin/python -m pip install --no-index \
  --find-links /out/wheelhouse \
  -r /out/requirements-a5-py312-amd64.lock.txt
./tmp/offline-check/bin/python -m pip check
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

构建目录使用 `.source-revisions` 记录 AscendNPU-IR 和其 vendored LLVM 身份。
普通 Git checkout 使用 commit；离线 rsync checkout 没有子模块 `.git`，此时根据
关键源码、HIVM schema 和 CMake 输入计算内容哈希，并输出
`BISHENGIR_SOURCE_ID compiler_mode=content llvm_mode=content`。旧构建目录没有
记录，或者任一身份发生变化时，脚本会先输出
`BISHENGIR_BUILD_CACHE_RESET` 并清理该构建树的旧 TableGen/目标文件，再进行
完整重建。这可以防止 Git checkout 后旧生成文件时间戳较新，导致源码已包含新
枚举而二进制仍链接旧 HIVM schema。

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
BISHENGIR_SOURCE_ID compiler_mode=<git或content> llvm_mode=<git或content>
HIVM_TABLEGEN_SSBUF_OK file=...
HIVM_SSBUF_BYTECODE_ROUNDTRIP_OK
BISHENGIR_COMPILE_SSBUF_PARSE_OK returncode=<0或后续pipeline返回码>
BISHENGIR_PACKAGE_OK soc=<设备型号> bitcode_arch=<c220或c310> tools=compile,opt
```

`setup-dev-environment.sh` 构建当前 checkout 的 Triton 和 `libtriton.so`；
当前 upstream 存在 `setup_ascend.py` 时，脚本会使用该入口执行开发构建；不能
手动替换为通用的 `pip install -e .`，否则会得到不含 Ascend binding 的
`libtriton.so`，且不会安装 `triton-mlir-opt`。
`rebuild-compiler.sh` 从仓库 gitlink 指定的 AscendNPU-IR 同时构建
`bishengir-compile` 和 `bishengir-opt`，并生成当前 SoC 对应的 `c220` 或 `c310`
bitcode。`HIVM_TABLEGEN_SSBUF_OK` 先确认本次构建生成的 HIVM enum 实现已经
包含 `ssbuf`；`HIVM_SSBUF_BYTECODE_ROUNDTRIP_OK` 表明项目 MLIR 22 写出的
DynamicCV SSBUF 属性能被同源 MLIR 19 reader 读取。
`BISHENGIR_COMPILE_SSBUF_PARSE_OK` 表明项目 `bishengir-compile` 自身也接受该属性；
探针允许在解析后的无效测试 module 上由后续 pipeline 返回非零，但任何
`HIVM_AddressSpaceAttr` 解析错误都会让重建立即失败。A5 环境会清除
`BISHENGIR_LEGACY_A5_REGBASE`，禁止委托给缺少 SSBUF 的旧 CANN A5 编译器。

## 8. 每次进入容器后激活环境

```bash
cd /home/triton-ascend
source tools/remote_experiment/activate-dev-environment.sh
```

预期输出：

```text
DEV_ENVIRONMENT_OK soc=<设备型号> bitcode_arch=<c220或c310> native_a5_regbase=<0或1> tmpdir=<项目路径>/tmp
```

激活脚本会创建项目根目录下的 `tmp/`，并将 `TMPDIR`、`TMP`、`TEMP` 全部指向
该目录。Python `tempfile`、`mktemp` 和编译器子进程的临时文件因此不会写入系统
`/tmp`；`tmp/` 已被 Git 忽略。

项目 `triton-mlir-opt` 使用 MLIR 22，同源项目 `bishengir-opt` 使用 MLIR 19，
二者通过 bytecode version 4 连接。不能使用 CANN 中缺少 `ssbuf` 枚举的旧
`bishengir-opt`，也不能全局关闭 bytecode；后者会改变 TTAdapter 输入路径并使
Vector Add 回归。`hivmc` 仍使用 CANN 版本。

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

Python prefix 应位于 `.codex-remote/venv`；Triton、`libtriton.so`、
`bishengir-compile` 和 `bishengir-opt` 都应来自当前项目目录。可另外执行：

```bash
which bishengir-compile
which bishengir-opt
```

两者都应位于 `.codex-remote/ascendnpu-ir-build-explicit/bin`。不要只执行 venv 的
`activate`，完整实验环境必须使用本节的激活脚本。

### 对比两套编译环境

在能运行和报错的环境中分别激活完整开发环境，然后执行同一个只读诊断脚本：

```bash
source tools/remote_experiment/activate-dev-environment.sh
./tools/remote_experiment/inspect-compiler-environment.sh \
  | tee tmp/compiler-environment-report.txt
```

脚本不会运行 NPU kernel，也不会安装或修改任何软件。它会输出系统、NPU、
Python 包、Triton/CANN/BishengIR/HIVM 工具的实际路径与版本，仓库及 gitlink
版本、CMake cache、动态库解析，以及 LLVM/MLIR 22 writer 和 LLVM/MLIR 19.1.7
reader/compiler 的关系。末尾还会执行 bytecode version 4 的普通 IR 和 DynamicCV
`ssbuf` 两种小型 roundtrip，并打印 `PASS`、`FAIL` 或 `SKIP`。

正常 A5 开发环境的关键结果应为：

```text
CHECK TOP_LEVEL_MLIR_WRITER_22             PASS
CHECK BISHENGIR_MLIR_READER_19             PASS
CHECK HIVM_SOURCE_HAS_SSBUF                 PASS
CHECK HIVM_TABLEGEN_HAS_SSBUF               PASS
CHECK BYTECODE_V4_GENERIC_ROUNDTRIP         PASS
CHECK BYTECODE_V4_SSBUF_ROUNDTRIP           PASS
CHECK BISHENGIR_COMPILE_SSBUF_PARSE         PASS
```

若脚本无法自动取得 SoC 名称，只有最后一项会显示 `SKIP`，可明确传入后重试：

```bash
ENV_REPORT_SOC_NAME=Ascend950 \
  ./tools/remote_experiment/inspect-compiler-environment.sh
```

若 `HIVM_SOURCE_HAS_SSBUF=PASS` 但 `HIVM_TABLEGEN_HAS_SSBUF=FAIL`，说明源码已包含
新枚举，但当前 BishengIR 构建仍是旧产物。若普通 roundtrip 通过而 SSBUF
roundtrip 失败，问题集中在 DynamicCV 的 HIVM schema/reader，而不是 bytecode
version 4 的通用兼容性。

### 使用已安装 wheel 和环境 Bisheng 的独立探针

若环境中有 Triton-Ascend wheel，可在独立 venv 中验证该 wheel 与环境 BishengIR
是否能够处理 DynamicCV。不要把发布 wheel 安装到正式实验使用的
`.codex-remote/venv`。复用环境中已经安装的 Torch、Torch-NPU 和基础依赖，只把
待测 wheel 安装到隔离目录：

```bash
cd /home/y00969467/triton-ascend

BASE_PYTHON="$(readlink -f .codex-remote/venv/bin/python)"
WHEEL_PROBE_VENV="$PWD/.codex-remote/wheel-probe-venv"
TRITON_ASCEND_WHEEL=/离线包路径/triton_ascend-版本-python架构.whl

"$BASE_PYTHON" -m venv --system-site-packages "$WHEEL_PROBE_VENV"
"$WHEEL_PROBE_VENV/bin/python" -m pip install \
  --no-index --no-deps --force-reinstall "$TRITON_ASCEND_WHEEL"
```

`--no-deps` 防止 pip 替换正式环境已有的 Torch、Torch-NPU、NumPy 等包，
`--no-index` 保证离线执行时不会访问网络。不要激活这个 venv，直接把其 Python
绝对路径传给探针；也不要 `source` 探针脚本：

```bash
SYSTEM_PROBE_PYTHON="$WHEEL_PROBE_VENV/bin/python" \
  ./tools/remote_experiment/probe-installed-wheel-toolchain.sh
echo "exit=$?"
```

脚本只在子 shell 中加载 CANN，并临时清除项目开发环境的 `PYTHONPATH`、
`TRITON_BUILD_DIR` 和 `TRITON_NPU_COMPILER_PATH`。它不会修改当前 shell，也不会
使用 `.codex-remote/venv` 或项目自编译 BishengIR。它允许发布 wheel 位于独立的
`.codex-remote/wheel-probe-venv`，并要求 Triton、`libtriton` 和 Ascend backend
都来自该 Python 前缀。脚本首先打印 Python、wheel、`libtriton`、
`triton-mlir-opt`、`triton-opt`、`bishengir-opt`、`bishengir-compile`、`hivmc`
和 `bisheng` 的实际路径与版本，然后依次执行：

1. MLIR bytecode version 4 普通 roundtrip；
2. `#hivm.address_space<ssbuf>` roundtrip；
3. `bishengir-compile` 的 SSBUF 解析检查；
4. fused attention 的 DynamicCV 编译、正确性检查和一次短 NPU benchmark。

该探针只设置 `intra_cache_num` 和 `vf_merge_level`，不会设置
`EXPERIMENT_MULTIBUFFER_NUM` 或传入 `--set-local-multibuffer`。默认参数为
`intra_cache_num=1`、`vf_merge_level=0`、一次 warmup 和一次 active 测量。需要时
可以只对本次命令覆盖：

```bash
SYSTEM_PROBE_PYTHON="$WHEEL_PROBE_VENV/bin/python" \
SYSTEM_PROBE_INTRA_CACHE_NUM=2 \
SYSTEM_PROBE_VF_MERGE_LEVEL=1 \
SYSTEM_PROBE_FULL_TIMEOUT=600 \
  ./tools/remote_experiment/probe-installed-wheel-toolchain.sh
```

正常结果以以下输出结束，并返回 `0`：

```text
CHECK BYTECODE_V4_SSBUF_ROUNDTRIP       PASS
CHECK BISHENGIR_COMPILE_SSBUF_PARSE    PASS
CHECK MULTIBUFFER_OMITTED               PASS
CHECK FULL_PIPELINE                     PASS
RESULT=PASS
```

完整终端摘要只保存为项目 `tmp/installed-wheel-toolchain-<时间>.log` 一个文件。
`UB_OBSERVATION=WARN` 不影响本次兼容性结论，但表示该环境还不能直接用于正式 UB
测量。若设备名无法自动识别，可在单次命令上设置
`SYSTEM_PROBE_SOC_NAME=Ascend950`；若当前环境的 wheel Python 不是 `python`，可设置
`SYSTEM_PROBE_PYTHON=/绝对路径/python`。

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

如果 Vector Add 无输出或超时，先运行不经过 Triton 编译器的 runtime 探针：

```bash
timeout 30 python -u experiment_operators/diagnose_npu_runtime.py
echo "exit=$?"
```

正常结果最后是 `NPU_RUNTIME_OK` 和 `exit=0`。`exit=124` 表示超时，最后一个
`RUNTIME_STAGE` 精确标明卡在 import、`set_device`、allocate、torch add 或
`synchronize`。单卡 Ascend Runtime 容器应显示 `device_count 1`；使用
`ASCEND_VISIBLE_DEVICES=<物理卡>` 创建的容器中，不要再设置
`ASCEND_RT_VISIBLE_DEVICES` 造成二次编号映射。

只运行一组实验配置：

```bash
SWEEP_LIMIT=1 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

预期显示 `实验完成：成功=1 失败=0 不支持=0`，并生成只有一行数据的
`results.csv`。冒烟通过后再运行完整实验。

## 10. 运行完整实验

每条命令只运行一个算子，默认每组 5 次 warmup、30 次 active 测量、120 秒
超时和 1 次超时补测：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
./run_all_sweeps.sh experiment_operators/candidates/unified_attention.py
./run_all_sweeps.sh experiment_operators/candidates/hstu_attention.py
./run_all_sweeps.sh experiment_operators/candidates/flash_attention_npu_v8.py
```

每个算子的 A3 正式实验包含 32 行。A5 正式实验包含 40 行：先运行 8 个
`DynamicCV=false` 基线，再运行 32 个唯一的
`(intra_cache_num, multibuffer_num, vf_merge_level)` 开启组。失败或不支持的配置
不会被丢弃。终端会持续显示当前参数以及成功、失败、不支持数量；每轮
结束后还会保留一行不被进度条覆盖的核心日志，例如：

```text
CASE 1/40 key=dynoff-b1-m0 dynamic_cv=false intra_cache_num=N/A multibuffer_num=1 vf_merge_level=0 结果=成功 status=measured latency_ms=... ub_kib=... wall_time_s=... log=logs/dynoff-b1-m0.log
```

失败或不支持行会额外显示简短原因，默认不会把完整 IR 打印到终端。每个
case 的完整 stdout/stderr 都会单独保存到同一结果目录下的 `logs/<case>.log`；
编译错误、pipeline 诊断、正确性 mismatch 和 benchmark 输出都在该文件中。
日志从 case 启动时开始流式写入，终端的 `requested_parameters` 行会显示其绝对
路径，因此运行疑似卡住时可以在另一个终端直接 `tail -f`。case 超时后 runner
会终止该候选及其启动的编译器/runtime 子进程，再继续下一组配置。所有配置的
首轮都结束后，runner 会按原顺序补测超时项。每个配置在 `results.csv` 中仍只有
一行，但会显示尝试次数、首轮是否超时和最终是否超时；同一 case 的各次输出
按 attempt 分段追加在同一个 `logs/<case>.log` 中。

若日志出现 `ERR9999` 且显示 `bishengir-opt` 返回 1，真正原因位于同一 case
日志中随后输出的 `bishengir-opt failed` 段。该段会记录实际 executable、返回码
以及完整 stdout/stderr。这个阶段负责把项目 `triton-mlir-opt` 产生的 MLIR
bytecode 解码成文本，发生在 `bishengir-compile` 和三个实验 pass 参数之前。

若错误来自 `bishengir-compile`，同一 case 日志会出现
`bishengir-compile failed`，随后记录返回码、实际命令以及捕获到的 stdout/stderr。
实验固定关闭 `--mlir-print-ir-after-failure`，因此这里保留 pass 诊断和栈信息，
但不会被整份失败 IR 淹没。不要只看最外层的 `MLIRCompilationError`。

A3 元数据必须满足：

```text
set_workspace_multibuffer == depth
enable_dynamic_cv_pipeline == false
limit_auto_multi_buffer_buffer == no-limit
```

A5 DynamicCV 关闭组元数据必须满足：

```text
intra_cache_num == N/A
set_workspace_multibuffer == 1
enable_dynamic_cv_pipeline == false
limit_auto_multi_buffer_buffer == no-limit
```

A5 DynamicCV 开启组元数据必须满足：

```text
intra_cache_num == 请求值
inter_cache_num == 1
load_cache_num == 1
set_workspace_multibuffer == 0
enable_dynamic_cv_pipeline == true
limit_auto_multi_buffer_buffer == no-limit
```

DynamicCV 若回退为 false，该配置会记录为“不支持”，不会混入有效测量。
case 日志会输出 `dynamic_cv_pipeline_fallback` 及其 return code，CSV 的原因列还会
列出最近 compiler metadata 中每个不匹配字段的实际值和期望值。若 HSTU 或
unified 的全部配置都显示 `enable_dynamic_cv_pipeline=False expected=True`，说明
这些 kernel 全部触发了 DynamicCV fallback，不是实验参数没有传入。
其中 return code `2` 是 DynamicCV 的 `ERRCODE_IGNORED`，表示当前 IR 不适用该
pass（例如没有 `linalg.matmul`、命中黑名单或已有 `scope.scope`），不是 pass
崩溃。此类 kernel 不进入 A5 `intra_cache_num` 正式实验样本。

可调整运行策略：

```bash
# 纯文本进度，适合重定向到一个日志文件
SWEEP_PROGRESS_MODE=plain \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py \
  2>&1 | tee fused_attention.log

# 调整超时补测次数；设为 0 可禁用补测
SWEEP_TIMEOUT_RETRIES=2 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py

# 仅诊断时恢复 vf_merge_level=2：A3 共 48 组，A5 共 60 组
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

页面底部会显示当前所选算子实验所记录的 Triton Ascend commit 和
AscendNPU-IR gitlink。离线 rsync 环境通过 `.codex-remote/top-git` 解析版本，
项目根目录不需要 `.git`，也不需要传输 submodule 的 Git 仓库。
报告生成器可以直接读取完整的简洁版 `results.csv`。旧简洁结果未记录源码版本时
会显示“未记录”，无需为生成图表重新运行实验；新结果会保留一个内部 manifest。

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
