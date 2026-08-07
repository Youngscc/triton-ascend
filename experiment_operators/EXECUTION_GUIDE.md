# 单算子三参数实验执行手册

这套流程每次只执行一个 Python 算子，但会完整遍历该算子的 48 组配置：

```text
depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..2)
```

`run_all_sweeps.sh` 中的 “all” 指一个算子的全部配置，不再表示一次执行所有算子。每次完整运行结束后，脚本会扫描结果目录中已经存在的所有算子，为每个算子选择最新的一次完整结果，并重新生成汇总表和交互式 HTML。

## 0. 首次配置 SSH 和两个项目路径

以下命令都在**本地终端**执行。先创建 SSH 目录并生成密钥；如果
`~/.ssh/id_ed25519` 已经存在，可以跳过 `ssh-keygen`，不要覆盖已有密钥：

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
ssh-keygen -t ed25519 -C "huawei-server" -f "$HOME/.ssh/id_ed25519"
```

编辑本地的 `~/.ssh/config`：

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
${EDITOR:-vi} "$HOME/.ssh/config"
```

加入以下固定配置：

```sshconfig
Host huawei-server
    HostName 192.168.25.217
    User root
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

保存后检查连接：

```bash
chmod 600 "$HOME/.ssh/config"
ssh huawei-server
```

随后进入本地仓库，一次性编辑配置文件：

```bash
vi tools/remote_experiment/config.sh
```

只修改文件开头的两个路径：

```bash
LOCAL_PROJECT="/你的本地绝对路径/triton-ascend"
REMOTE_PROJECT="/服务器上的绝对路径/triton-ascend"
```

`REMOTE_HOST="huawei-server"` 和 `REMOTE_CONTAINER="sgl-sky"` 已在文件中固定，
不需要用户修改或 `export`。所有远程实验脚本都会自动读取这个配置文件，
两个路径只需修改一次。检查配置：

```bash
source tools/remote_experiment/config.sh
test -d "$LOCAL_PROJECT" || echo "本地路径不存在: $LOCAL_PROJECT"
ssh huawei-server "test -d '$REMOTE_PROJECT'" || \
  echo "服务器路径尚不存在，首次 sync.sh 会创建它"
```

## A5 环境下需要特别执行的命令

本项目把 A5/Ascend 950 系列识别为 `Ascend910_95*` 或 `Ascend950*`，并走
A5 NPUIR 编译路径。代码同步、结果回拉和汇总命令与前文相同；A5 上不同的
地方主要是先确认芯片、激活该机器的 CANN，以及使用带 A5 开关的编译器。

### 1. 创建只允许使用物理 7 卡的 A5 容器

安装指南中的通用 `docker run` 示例不能原样用于本实验：它挂载了
`/dev/davinci0` 到 `/dev/davinci7` 全部设备。部分示例还带有
`--privileged`；特权容器会削弱设备隔离，因此“只允许使用 7 卡”时必须删掉
该选项。

先在本地同步源码，然后登录 A5 服务器并进入服务器仓库：

```bash
./tools/remote_experiment/sync.sh
source tools/remote_experiment/config.sh
ssh -t huawei-server "cd '$REMOTE_PROJECT' && exec bash"
```

以下构建和创建命令均在 **A5 服务器宿主机**执行。镜像必须选择 950，而不是
安装指南示例中的 A3 标签：

```bash
docker build \
  --build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:9.0.0-950-ubuntu22.04-py3.11 \
  -t triton-ascend-a5-dev:latest \
  -f docker/Dockerfile .
```

创建容器前先确认物理 7 卡和三个管理设备存在：

```bash
for dev in /dev/davinci7 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  test -e "$dev" || { echo "缺少设备: $dev" >&2; exit 1; }
done
```

如果 `docker ps -a --format '{{.Names}}'` 已经列出 `sgl-sky`，不要重复执行
下面的创建命令；应先确认旧容器是否可以继续使用。新建容器的最终命令为：

```bash
docker run -u 0 -dit \
  --name=sgl-sky \
  --net=host \
  --workdir="$PWD" \
  --shm-size=512g \
  --security-opt seccomp=unconfined \
  --device=/dev/davinci7:/dev/davinci7:rwm \
  --device=/dev/davinci_manager:/dev/davinci_manager:rwm \
  --device=/dev/devmm_svm:/dev/devmm_svm:rwm \
  --device=/dev/hisi_hdc:/dev/hisi_hdc:rwm \
  -e ASCEND_RT_VISIBLE_DEVICES=7 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /home:/home \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  triton-ascend-a5-dev:latest \
  /bin/bash
```

容器创建后，为当前仓库建立隔离的开发 Python 环境。以下命令仍在 A5 宿主机
执行；`docker exec` 后面的命令在新容器中运行：

```bash
docker exec -u root -it sgl-sky /bin/bash
# docker run 已把服务器仓库设置为容器工作目录；先确认当前位置。
pwd
if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  source /usr/local/Ascend/cann/set_env.sh
else
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
python3 -m venv --system-site-packages .codex-remote/venv
source .codex-remote/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

虚拟环境放在 `$REMOTE_PROJECT/.codex-remote/venv`，不会修改镜像的全局
Python，也不会被普通 `sync.sh` 覆盖。远程执行脚本和
`run_all_sweeps.sh` 默认都会使用这个环境。

这里使用两层限制：

- Docker 只挂载 `/dev/davinci7`，其他计算卡设备节点不会进入容器；
- `ASCEND_RT_VISIBLE_DEVICES=7` 只向容器进程暴露物理 7 卡，并把它重编号为
  逻辑设备 `0`。

因此算子代码应继续使用 `npu:0` 或当前默认设备，**不要在容器里使用
`npu:7`**。`npu-smi info` 可能通过管理设备显示宿主机的其他卡，但这不代表
其他 `/dev/davinci*` 计算设备已经授权给容器。

如果 A5 主机安装并配置了 Ascend Docker Runtime，也可以用
`ASCEND_VISIBLE_DEVICES=7` 让该 runtime 自动挂载设备；上述命令采用显式
`--device`，不依赖主机额外配置 Ascend Docker Runtime，两种挂载方式不要
混在同一个命令中。

### 2. 进入 A5 容器并确认芯片和设备隔离

在本地仓库根目录执行：

```bash
source tools/remote_experiment/config.sh
ssh -t huawei-server \
  "docker exec -it sgl-sky bash -c 'cd \"$REMOTE_PROJECT\" && exec bash'"
```

以下命令改为在 **A5 容器内**执行：

```bash
if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  source /usr/local/Ascend/cann/set_env.sh
else
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
npu-smi info
python3 - <<'PY'
import acl
import torch
import torch_npu

soc = acl.get_soc_name()
print("CANN SoC name:", soc)
assert soc.startswith(("Ascend910_95", "Ascend950")), \
    f"当前设备没有被识别为 A5/Ascend 950: {soc}"
print("visible NPU count:", torch.npu.device_count())
assert torch.npu.device_count() == 1, "容器没有被限制为单卡"
torch.npu.set_device(0)
print("logical npu:0:", torch.npu.get_device_name(0))
PY
ls -l /dev/davinci* /dev/davinci_manager
```

预期只有一个计算设备节点 `/dev/davinci7`，而 PyTorch 报告一个逻辑设备
`npu:0`。如果还能看到 `/dev/davinci0` 到 `/dev/davinci6`，说明容器仍是
特权容器或挂载参数不正确，不要开始正式实验。

判断 A5 编译路径时以 `acl.get_soc_name()` 的输出为准，不要仅凭服务器名称
判断。正常的板上执行不需要手工设置 `TRITON_ASCEND_ARCH`；项目会从 CANN
运行时读取实际 SoC。只有离线编译时才考虑使用准确的
`TRITON_ASCEND_ARCH=Ascend910_95xx`，不能照抄其他 A5 型号。

### 3. 在 A5 上重建自编译 BishengIR

修改过 AscendNPU-IR C++，或者第一次在 A5 环境部署自编译工具链时，退出
容器并在**本地仓库根目录**执行：

```bash
./tools/remote_experiment/sync.sh
./tools/remote_experiment/rebuild-compiler.sh
```

`rebuild-compiler.sh` 已固定包含 A5 所需的 CMake 开关：

```text
-DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON
-DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON
```

不要在 A5 上改用只为 A2/A3 构建的 `bishengir-compile`，也不要只复制一个
编译器可执行文件；它旁边对应版本的 `lib/meta_op.*.bc` 和 `host.bc` 也必须
存在。上述重建脚本会准备这套配套文件。

### 4. A5 冒烟验证

先从本地启动项目自带的 Vector Add，确认 Python、当前源码、A5 编译器和
NPU 运行时能够连通：

```bash
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  python -u third_party/ascend/tutorials/01-vector-add.py
./tools/remote_experiment/logs.sh latest
```

Vector Add 通过后，在 A5 容器内用一个配置做实验控制器冒烟测试：

```bash
if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  source /usr/local/Ascend/cann/set_env.sh
else
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
SWEEP_LIMIT=1 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

确认该行完成正确性、NPU latency 和非零 UB 记录之后，再执行正式 48 组：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

不要把 A2/A3 上生成的 Triton cache 或 NPU 二进制复制到 A5 复用。
`run_all_sweeps.sh` 会为正式运行创建新的 cache；A5 的性能和 UB 结果也应当
作为独立设备数据保存，不能与 A2/A3 数据直接合并比较。

## 1. 本地代码同步到服务器

由于实验在服务器展开，并用root登录，为安全考虑，建议在本地拉取或修改代码，并向服务器的对应路径文件夹同步，服务器上的运行结果也可同步回本地。

在本地仓库根目录执行：

```bash
source tools/remote_experiment/config.sh
cd "$LOCAL_PROJECT"
./tools/remote_experiment/sync.sh
```

普通同步是增量、非删除同步，并排除 `.git`、`.codex-remote`、缓存和实验结果。

只有明确需要让服务器源码精确镜像本地删除时才使用：

```bash
RSYNC_DELETE=1 ./tools/remote_experiment/sync.sh
```

仅修改 Python 算子、实验控制器或报告脚本不需要重编译。修改 AscendNPU-IR C++ 后还要执行：

```bash
./tools/remote_experiment/rebuild-compiler.sh
```

## 2. 在服务器容器中执行一个算子

从本地直接进入服务器容器，并把初始目录设为 `$REMOTE_PROJECT`：

```bash
source tools/remote_experiment/config.sh
ssh -t huawei-server \
  "docker exec -it sgl-sky bash -c 'cd \"$REMOTE_PROJECT\" && exec bash'"
```

执行一个算子的完整 48 组实验：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

入口脚本会自动完成：

1. 激活远程实验配置指定的开发虚拟环境；
2. 选择 `.codex-remote/ascendnpu-ir-build-explicit/bin/bishengir-compile`；
3. 为本次运行创建独立的 Triton cache；
4. 依次执行该算子的 48 组配置，记录正确性、耗时、UB、hash 和诊断；
5. 扫描 `.codex-remote/results` 中所有算子的最新完整结果；
6. 更新 `latest-summary` 表格、效果图和交互式 HTML。

不要在同一张 NPU 上并行运行多个 sweep。正式数据默认使用 5 次 warmup、30 次 active 测量和每个候选 120 秒超时。临时覆盖参数不需要修改源码：

```bash
SWEEP_WARMUP=2 SWEEP_ACTIVE=5 SWEEP_TIMEOUT=60 \
  ./run_all_sweeps.sh experiment_operators/candidates/hstu_attention.py
```

只验证入口和命令，不启动 NPU 实验：

```bash
DRY_RUN=1 ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

只跑前 N 组用于控制器 smoke test：

```bash
SWEEP_LIMIT=2 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

`SWEEP_LIMIT` 产生的是不完整结果，不会替换 HTML 中该算子此前的完整正式结果。

### 算子 Python 文件需要满足的接口

算子从以下环境变量读取配置：

```text
EXPERIMENT_DEPTH
EXPERIMENT_MULTIBUFFER_NUM
EXPERIMENT_VF_MERGE_LEVEL
EXPERIMENT_WARMUP
EXPERIMENT_ACTIVE
```

正确性失败时必须以非零状态退出；成功 benchmark 后输出一行：

```text
BENCHMARK operator=<operator_name> latency_ms=<number> warmup=<N> active=<N>
```

已有的三个候选算子还使用各自的 `Test Passed!` 标记做强正确性检查。对于其他路径，控制器从源码中的 `BENCHMARK operator=...` 推导算子名；推导不到时使用 Python 文件名。
如果算子最初位于本地仓库之外，应先复制和适配到
`experiment_operators/candidates/`，否则 `sync.sh` 不会把它同步到服务器。

## 3. 结果目录和完成标志

每次运行写入：

```text
.codex-remote/results/<UTC+8时间>-<operator>/
├── manifest.json
├── measurements.jsonl
├── measurements.csv
├── summary.json
└── logs/
```

完整正式实验应看到类似：

```json
{
  "complete": true,
  "expected_row_count": 48,
  "row_count": 48
}
```

状态不要求全部为 `measured`；失败、超时、精度错误和 UB 缺失也必须保留为一行观察结果。

## 4. 汇总所有算子的最新结果

完整 sweep 结束后会自动汇总。需要手工重新生成时，在结果所在机器的仓库根目录执行：

```bash
python3 experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

汇总器自动遍历 `.codex-remote/results/*/manifest.json`，按 `manifest.operator` 区分算子，只考虑满足以下条件的结果：

- `executed_configuration_count == requested_configuration_count`；
- `measurements.jsonl` 行数等于请求配置数；
- 不是 `--limit` smoke run。

如果同一个算子有多次完整运行，则按带时区的 `run_id` 选择最新一次。不同算子的最新结果会共同进入：

```text
.codex-remote/results/latest-summary/
├── supported.csv
├── supported.md
├── summary.json
├── effects.csv
├── effects.json
├── effects.md
├── effects.svg
└── experiment-report.html
```

在本地 macOS 打开 HTML：

```bash
source tools/remote_experiment/config.sh
cd "$LOCAL_PROJECT"
open .codex-remote/results/latest-summary/experiment-report.html
```

## 5. 服务器结果同步到本地

退出服务器，在本地仓库根目录执行：

```bash
source tools/remote_experiment/config.sh
cd "$LOCAL_PROJECT"
./tools/remote_experiment/pull-results.sh
```

该命令使用兼容旧版 rsync 的 `--progress`，默认做增量、非删除拉取。连同顶层 session log 一起拉取：

```bash
PULL_SESSION_LOGS=1 ./tools/remote_experiment/pull-results.sh
```

如果服务器结果目录应当成为本地的精确镜像，可以显式启用删除；这会删除本地存在、服务器不存在的旧结果：

```bash
RSYNC_DELETE=1 PULL_SESSION_LOGS=1 \
  ./tools/remote_experiment/pull-results.sh
```

拉取后建议在本地重新汇总一次，确保 HTML 使用刚拉取的全部最新结果：

```bash
python3 experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

## 6. 不进入容器，使用后台任务运行

也可以从本地直接同步并在 `sgl-sky` 中启动后台任务：

```bash
./tools/remote_experiment/sync.sh
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
./tools/remote_experiment/logs.sh latest
```

`Ctrl-C` 只停止本地日志跟随，不会停止服务器中的实验。实验完成后仍使用 `pull-results.sh` 拉回结果。
