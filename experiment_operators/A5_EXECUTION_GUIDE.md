# A5 单卡实验环境执行手册

本手册用于 `huawei-server-A5` 上的 Ascend 950/A5 环境。实验容器固定为
`yy-npu`，只授权物理 7 卡；容器内使用逻辑设备 `npu:0`。基础镜像固定为：

```text
quay.io/ascend/cann:9.0.0-950-ubuntu22.04-py3.11
```

## 1. 配置 SSH

在本地生成 SSH 密钥。已有密钥时跳过 `ssh-keygen`：

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
ssh-keygen -t ed25519 -C "huawei-server-A5" -f "$HOME/.ssh/id_ed25519"
```

在 `~/.ssh/config` 中配置：

```sshconfig
Host huawei-server-A5
    HostName 192.168.25.217
    User root
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

验证连接：

```bash
chmod 600 "$HOME/.ssh/config"
ssh huawei-server-A5
```

## 2. 配置项目路径

在本地仓库编辑：

```bash
vi tools/remote_experiment/config.sh
```

设置本地与服务器项目绝对路径，并保持 A5 主机和容器名称如下：

```bash
LOCAL_PROJECT="/你的本地绝对路径/triton-ascend"
REMOTE_PROJECT="/服务器上的绝对路径/triton-ascend"
REMOTE_HOST="huawei-server-A5"
REMOTE_CONTAINER="yy-npu"
```

同步源码：

```bash
./tools/remote_experiment/sync.sh
```

## 3. 创建并配置容器

登录 A5 服务器宿主机并进入服务器仓库：

```bash
source tools/remote_experiment/config.sh
ssh -t huawei-server-A5 "cd '$REMOTE_PROJECT' && exec bash"
```

在 A5 服务器宿主机执行：

```bash
./tools/remote_experiment/setup-a5-container.sh
```

脚本完成以下操作：

1. 使用 `npu-smi info` 检查宿主机 NPU；
2. 拉取缺失的官方 CANN 950 镜像；
3. 根据宿主机能力选择 Ascend Docker Runtime 或显式设备挂载；
4. 创建不带 `--privileged` 的 `yy-npu`；
5. 只授权物理 7 卡；
6. 在 `.codex-remote/venv` 创建项目隔离环境并安装当前仓库；
7. 验证容器内只有一个逻辑设备 `npu:0`。

已经存在的 `yy-npu` 不会被脚本删除或覆盖。脚本会输出启动、进入或人工重建
容器所需的命令。

## 4. 进入容器并验证 A5

在本地执行：

```bash
source tools/remote_experiment/config.sh
ssh -t huawei-server-A5 \
  "docker exec -it yy-npu bash -c 'cd \"$REMOTE_PROJECT\" && exec bash'"
```

在容器内执行：

```bash
if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  source /usr/local/Ascend/cann/set_env.sh
else
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
source .codex-remote/venv/bin/activate

npu-smi info

python3 - <<'PY'
import acl
import torch
import torch_npu

soc = acl.get_soc_name()
print("CANN SoC name:", soc)
assert soc.startswith(("Ascend910_95", "Ascend950")), soc

count = torch.npu.device_count()
print("visible NPU count:", count)
assert count == 1, count

torch.npu.set_device(0)
print("logical npu:0:", torch.npu.get_device_name(0))
PY
```

## 5. 构建 A5 编译器

在本地仓库执行：

```bash
./tools/remote_experiment/sync.sh
./tools/remote_experiment/rebuild-compiler.sh
```

编译配置包含：

```text
-DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON
-DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON
```

编译器目录中的 `bishengir-compile`、`lib/meta_op.*.bc` 和 `host.bc` 必须来自
同一次构建。

## 6. 冒烟验证

从本地启动 Vector Add：

```bash
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  python -u third_party/ascend/tutorials/01-vector-add.py
./tools/remote_experiment/logs.sh latest
```

在容器内执行单配置实验验证：

```bash
SWEEP_LIMIT=1 SWEEP_WARMUP=1 SWEEP_ACTIVE=1 \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

该配置需要完成正确性检查，记录 NPU latency，并得到非零 UB 数据。

## 7. 执行完整实验

在 `yy-npu` 内传入一个算子文件：

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

完整实验遍历：

```text
depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..2)
```

总计 48 组配置。不要在同一张 NPU 上并行启动多个 sweep。

## 8. 日志、结果与报告

从本地启动后台实验并跟踪日志：

```bash
./tools/remote_experiment/sync.sh
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
./tools/remote_experiment/logs.sh latest
```

同步结果到本地并生成报告：

```bash
./tools/remote_experiment/pull-results.sh
python3 experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

A5 的 Triton cache、NPU 二进制、性能数据和 UB 数据作为独立设备数据保存。
