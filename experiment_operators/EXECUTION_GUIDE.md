# Triton-Ascend 单算子三参数实验执行手册

服务器仓库是主工作区。源码更新、环境准备、构建、编译、运行、日志查看和结果
汇总均在服务器完成；构建和前台实验命令在已有实验容器内执行。服务器宿主机和
容器必须以相同绝对路径挂载项目。

每次实验接收一个 Python 算子并遍历：

```text
depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..2) = 48 组
```

## 1. 在服务器准备仓库

服务器能够访问 GitHub 时，直接在服务器宿主机 clone：

```bash
cd /服务器代码目录
git clone --recurse-submodules git@github.com:Youngscc/triton-ascend.git
cd triton-ascend
```

更新已有服务器仓库：

```bash
git fetch origin
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

进入需要实验的分支后，确认顶层仓库和子模块状态：

```bash
git branch --show-current
git status --short
git submodule status --recursive
```

## 2. 配置已有容器

在服务器仓库创建 Git 忽略的配置文件：

```bash
cp tools/remote_experiment/config.local.sh.example \
  tools/remote_experiment/config.local.sh
vi tools/remote_experiment/config.local.sh
```

设置服务器项目绝对路径和已有容器名：

```bash
REMOTE_PROJECT="/服务器绝对路径/triton-ascend"
REMOTE_CONTAINER="已有实验容器名"
```

检查容器和项目挂载：

```bash
source tools/remote_experiment/config.sh
test -d "$REMOTE_PROJECT"
docker inspect "$REMOTE_CONTAINER" --format '{{.State.Status}}'
docker exec "$REMOTE_CONTAINER" test -d "$REMOTE_PROJECT"
docker inspect "$REMOTE_CONTAINER" \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

项目目录必须在宿主机和容器内保持同一个绝对路径。`.codex-remote/venv`、编译
产物、缓存、日志和结果都位于该挂载目录，因此容器重启不会丢失。

## 3. 创建或修复项目 venv

从服务器宿主机进入已有容器：

```bash
source tools/remote_experiment/config.sh
docker exec -u root -it "$REMOTE_CONTAINER" bash
cd "$REMOTE_PROJECT"
source tools/remote_experiment/config.sh
```

在容器内执行：

```bash
JOBS=32 ./tools/remote_experiment/setup-dev-environment.sh
```

该命令可重复执行，并完成：

1. 缺失时创建 `$REMOTE_PROJECT/.codex-remote/venv`；
2. 通过 `--system-site-packages` 复用容器已有的 Torch、Torch-NPU 和 CANN；
3. CMake 低于 3.28 时，只在项目 venv 内安装新版 CMake；
4. 构建当前仓库的 Triton C++/MLIR 核心和 `libtriton.so`；
5. editable 安装当前 Triton-Ascend，并验证实际导入路径。

脚本优先使用服务器 clone 自身的 `.git`。只有离线 rsync 得到的无 `.git`
工作树才使用 `.codex-remote/top-git`。

检查 venv：

```bash
test -x .codex-remote/venv/bin/python
.codex-remote/venv/bin/python - <<'PY'
import triton
from triton._C import libtriton

print("triton:", triton.__file__)
print("libtriton:", libtriton.__file__)
PY
```

不要从宿主机、其他容器或其他项目路径复制 venv。项目路径改变后，应在最终
容器和最终挂载路径重新执行 `setup-dev-environment.sh`。

## 4. 构建当前仓库的 BishengIR

继续在容器内执行：

```bash
JOBS=32 ./tools/remote_experiment/rebuild-compiler.sh
```

构建源和产物：

```text
third_party/ascend/AscendNPU-IR
.codex-remote/ascendnpu-ir-build-explicit/
├── bin/bishengir-compile
└── lib/
    ├── meta_op.aic.c220.bc
    ├── meta_op.aiv.c220.bc
    ├── meta_op.mix.aic.c220.bc
    ├── meta_op.mix.aiv.c220.bc
    └── host.bc
```

确认工具链：

```bash
.codex-remote/ascendnpu-ir-build-explicit/bin/bishengir-compile --version
command -v hivmc
```

`bishengir-compile` 及相邻 bitcode 必须来自同一次构建。最终 NPU 二进制仍由
容器内 CANN 的 `hivmc` 生成。

## 5. 选择设备并做基线验证

在服务器宿主机用以下命令查看设备，不运行会主动初始化所有卡的探测程序：

```bash
npu-smi info
```

如果容器挂入多张卡，进入容器后为当前命令选择一张健康空闲的物理卡。例如
物理卡 2：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 \
  .codex-remote/venv/bin/python -u \
  third_party/ascend/tutorials/01-vector-add.py
```

如果容器创建时只暴露一张物理卡，该卡通常映射为容器内逻辑 `npu:0`，不需要
再次指定物理编号。Vector Add 必须输出最大误差 0 或容差内结果并通过测试。

## 6. 运行一个算子的实验

### 前台运行

在容器内项目根目录执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

单卡容器直接执行：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

冒烟测试：

```bash
SWEEP_LIMIT=1 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

入口检查：

```bash
DRY_RUN=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

正式运行默认使用 5 次 warmup、30 次 active 测量和每组 120 秒超时。每个失败
配置会直接在终端打印状态、返回码、完整子进程输出和独立日志路径；随后继续
执行剩余配置并保留失败行。

### 服务器后台运行

退出容器，在服务器宿主机的仓库根目录执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> REMOTE_MODE=dev \
  ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

单卡容器可省略 `ASCEND_RT_VISIBLE_DEVICES`。查看日志：

```bash
./tools/remote_experiment/logs.sh latest
./tools/remote_experiment/logs.sh <run-id>
```

`Ctrl-C` 只停止日志跟随，不终止后台实验。

## 7. 结果与报告

每次运行写入服务器项目：

```text
.codex-remote/results/<UTC+8时间>-<operator>/
├── manifest.json
├── measurements.jsonl
├── measurements.csv
├── summary.json
└── logs/
```

完整实验的 `summary.json` 应包含：

```json
{
  "complete": true,
  "expected_row_count": 48,
  "row_count": 48
}
```

失败、超时、精度错误和 UB 缺失仍是实验观察，必须保留。手工刷新所有算子的
最新完整结果时，在容器内执行：

```bash
source .codex-remote/venv/bin/activate
python experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

HTML 位于：

```text
.codex-remote/results/latest-summary/experiment-report.html
```

## 8. GitHub 不可达时的源码同步备用方案

只有服务器不能连接 GitHub 时，才从个人电脑使用 rsync。个人电脑的本地仓库
配置以下附加字段：

```bash
LOCAL_PROJECT="/本地绝对路径/triton-ascend"
REMOTE_PROJECT="/服务器绝对路径/triton-ascend"
REMOTE_HOST="服务器SSH别名"
REMOTE_CONTAINER="已有实验容器名"
REMOTE_SOURCE_MODE="rsync"
```

然后在个人电脑执行：

```bash
./tools/remote_experiment/sync.sh
```

该同步排除 `.codex-remote`，不会传输或删除服务器 venv、编译产物、缓存、日志
和结果。需要把结果复制到个人电脑时，可单独执行：

```bash
./tools/remote_experiment/pull-results.sh
```
