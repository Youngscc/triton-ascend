# 单算子三参数实验执行手册

这套流程每次只执行一个 Python 算子，但会完整遍历该算子的 48 组配置：

```text
depth(1..4) × multibuffer_num(1..4) × vf_merge_level(0..2)
```

`run_all_sweeps.sh` 中的 “all” 指一个算子的全部配置。每次完整运行结束后，脚本会扫描结果目录中已经存在的所有算子，为每个算子选择最新的一次完整结果，并重新生成汇总表和交互式 HTML。

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

`SWEEP_LIMIT` 产生的是不完整结果，不会替换 HTML 中该算子已经存在的完整正式结果。

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
