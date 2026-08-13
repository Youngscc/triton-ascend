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

预期：显示 `Cloning into 'triton-ascend'...` 和各子模块检出信息，命令退出码为 0。

更新已有服务器仓库：

```bash
git fetch origin
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

预期：`git pull` 显示 `Already up to date.` 或正常的 fast-forward 更新；子模块命令
退出码为 0。

进入需要实验的分支后，确认顶层仓库和子模块状态：

```bash
git branch --show-current
git status --short
git submodule status --recursive
```

预期：第一行是实验分支名，`git status --short` 没有输出；每个子模块行以空格
开头，不以 `-`、`+` 或 `U` 开头。

## 2. 配置已有容器

在服务器仓库创建 Git 忽略的配置文件：

```bash
cp tools/remote_experiment/config.local.sh.example \
  tools/remote_experiment/config.local.sh
vi tools/remote_experiment/config.local.sh
```

预期：`cp` 无报错；保存后文件中存在实际的 `REMOTE_PROJECT` 和
`REMOTE_CONTAINER`。

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

预期：状态输出为 `running`，两个 `test` 命令无输出且退出码为 0，挂载列表包含：

```text
/服务器绝对路径/triton-ascend -> /服务器绝对路径/triton-ascend
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

预期：进入容器 shell，`pwd` 应为 `$REMOTE_PROJECT`，三条命令均无报错。

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

宿主侧编译必须使用 Clang。脚本优先自动选择版本后缀为 15 的
`clang`、`clang++`、`lld` 和 `ld.lld`，找不到 Clang 时直接失败，不会回退
到 GCC。宿主 Clang 不替代设备侧的 `ccec`、BishengIR 或 `hivmc`。

脚本优先使用服务器 clone 自身的 `.git`。只有离线 rsync 得到的无 `.git`
工作树才使用 `.codex-remote/top-git`。

setup 成功时必须以退出码 0 结束，并打印：

```text
MLIR_BYTECODE_ROUNDTRIP_OK
TRITON_DEV_IMPORT_OK
```

- `MLIR_BYTECODE_ROUNDTRIP_OK`：MLIR 22 生成的 bytecode version 4 可由
  MLIR 19.1.7 `bishengir-opt` 读取。
- `TRITON_DEV_IMPORT_OK`：项目 venv、当前仓库的 Triton 和
  `libtriton.so` 均已正确加载。

日志中的 `python prefix` 应为 `$REMOTE_VENV`，`triton` 和 `libtriton` 应位于
`$REMOTE_PROJECT/python/triton`。`python` 的基础解释器位于 `/usr/local` 是正常
的；如果 `triton` 位于 `/usr/local/.../site-packages`，则环境错误。

正常编译保持 `use_bytecode=true`，脚本会使用兼容的 bytecode version 4。

不要从宿主机、其他容器或其他项目路径复制 venv。项目路径改变后，应在最终
容器和最终挂载路径重新执行 `setup-dev-environment.sh`。

## 4. 构建当前仓库的 BishengIR

继续在容器内执行：

```bash
JOBS=32 ./tools/remote_experiment/rebuild-compiler.sh
```

预期：CMake 和 Ninja 构建成功，最后打印：

```text
BISHENGIR_PACKAGE_OK soc=<A3型号> bitcode_arch=c220
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
"$REMOTE_COMPILER_BUILD/bin/bishengir-compile" --version
for file in \
  meta_op.aic.c220.bc \
  meta_op.aiv.c220.bc \
  meta_op.mix.aic.c220.bc \
  meta_op.mix.aiv.c220.bc \
  host.bc; do
  test -s "$REMOTE_COMPILER_BUILD/lib/$file" || exit 1
done
command -v hivmc
echo BISHENGIR_PACKAGE_OK
```

预期输出包含：

```text
bishengir-compile 1.2.0
llvm 19.1.7
/usr/local/.../hivmc
BISHENGIR_PACKAGE_OK
```

`bishengir-compile` 及相邻 bitcode 必须来自同一次构建。最终 NPU 二进制仍由
容器内 CANN 的 `hivmc` 生成。以上命令必须以退出码 0 结束并打印
`BISHENGIR_PACKAGE_OK`；任一 bitcode 缺失或为空都不算通过。

每次打开新的容器 shell，在构建完成后激活一次项目开发环境：

```bash
source tools/remote_experiment/activate-dev-environment.sh
```

A3 预期输出：

```text
DEV_ENVIRONMENT_OK soc=<A3型号> bitcode_arch=c220 native_a5_regbase=0
```

后续手动 Python 命令可直接使用项目 venv 的 `python`，无需重复设置
`PYTHONPATH`、`PATH` 和 `TRITON_NPU_COMPILER_PATH`。

## 5. 选择设备并做基线验证

在服务器宿主机用以下命令查看设备，不运行会主动初始化所有卡的探测程序：

```bash
npu-smi info
```

预期：列出 NPU、`Health`、`AICore(%)`、内存和进程；选择 `Health=OK` 且没有占用
进程的设备。

如果容器挂入多张卡，进入容器后为当前命令选择一张健康空闲的物理卡。例如
物理卡 2：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 \
  python -u third_party/ascend/tutorials/01-vector-add.py
```

预期末尾包含：

```text
The maximum difference between torch and triton is 0.0
======Vector Add Test Passed!======
```

如果容器创建时只暴露一张物理卡，该卡通常映射为容器内逻辑 `npu:0`，不需要
再次指定物理编号。Vector Add 必须以退出码 0 结束，输出最大误差 0 或容差内
结果；编译错误、设备异常、超时或导入 `/usr/local` 的 Triton 都不算通过。

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

入口检查预期输出包含：

```text
project=...
operator_file=.../fused_attention.py
bishengir_compile=.../bishengir-compile
dry_run command=...
dry run complete; no experiment was launched
```

冒烟测试预期：输出一个配置的 `results=...` 和汇总 JSON；汇总中
`row_count` 为 `1`，`status_counts` 中为 `"measured": 1`。

实验启动时检查以下路径：

```text
ascend_backend=.../triton-ascend/python/triton/backends/ascend/utils.py (project checkout; required)
bishengir_compile=.../.codex-remote/ascendnpu-ir-build-explicit/bin/bishengir-compile (project build; required)
bitcode_package=soc:<A3型号> arch:c220 (project build; required)
bishengir_opt=/usr/local/Ascend/.../bishengir-opt (CANN bytecode reader; expected)
hivmc=/usr/local/Ascend/.../hivmc (CANN binary backend; expected)
```

backend、编译器和 `c220` bitcode 必须来自项目，后两条应来自 CANN；不符合时
脚本立即退出。

完整前台实验预期：逐项执行 48 个配置，末尾输出 `completed operator_file=...`、
`sweep and aggregate report complete` 和 HTML 路径；`summary.json` 中
`complete=true`、`expected_row_count=48`、`row_count=48`。

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

预期立即返回：

```text
run_id=<运行编号>
log=<服务器日志绝对路径>
pid=<宿主机进程号>
```

单卡容器可省略 `ASCEND_RT_VISIBLE_DEVICES`。查看日志：

```bash
./tools/remote_experiment/logs.sh latest
./tools/remote_experiment/logs.sh <run-id>
```

预期第一行是 `following <日志路径> (Ctrl-C to stop following)`，随后持续显示与
前台实验相同的运行输出。

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

预期汇总命令退出码为 0，报告命令打印：

```text
Open the report at: .../.codex-remote/results/latest-summary/experiment-report.html
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

预期：rsync 显示传输文件和统计信息，末尾打印本地路径到服务器路径的同步完成
信息，退出码为 0。

该同步排除 `.codex-remote`，不会传输或删除服务器 venv、编译产物、缓存、日志
和结果。需要把结果复制到个人电脑时，可单独执行：

```bash
./tools/remote_experiment/pull-results.sh
```

预期：rsync 显示新增或更新的结果文件和统计信息，退出码为 0；本地
`.codex-remote/results` 中出现对应运行目录。
