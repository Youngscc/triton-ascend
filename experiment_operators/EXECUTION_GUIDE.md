# Triton-Ascend A3/A5 单算子实验手册

服务器仓库是主工作区。构建和前台实验在已有容器内执行，服务器项目必须以相同
绝对路径挂载进容器。

实验遍历以下 48 组配置：

```text
depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..2)
```

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

预期汇总包含 `"row_count": 1` 和 `"measured": 1`。

## 5. 运行完整实验

容器内前台运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

命令只接收一个算子 Python 文件。单卡容器可省略设备变量。正式实验默认每组
5 次 warmup、30 次 active 测量和 120 秒超时；失败配置仍会记录并继续运行。

完整结果应满足：

```json
{
  "complete": true,
  "expected_row_count": 48,
  "row_count": 48
}
```

在服务器宿主机后台运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡> REMOTE_MODE=dev \
  ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py

./tools/remote_experiment/logs.sh latest
```

`Ctrl-C` 只停止日志跟随，不终止后台实验。

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
