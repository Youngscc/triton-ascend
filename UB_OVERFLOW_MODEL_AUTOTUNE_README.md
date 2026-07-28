# UB Overflow 轻量模型 Autotune Compile-Only 性能测试

## 1. 测试目标

本文档说明 UB overflow 轻量模型在 Autotune compile-only 流程中的三种测试模式、运行方法、
参数空间和已有测量结果。实验回答两个问题：

1. 对高频 overflow 的 `attn_fwd`，模型提前剪枝能让完整候选搜索加速多少？
2. 对没有发生 overflow 的典型算子，运行模型但继续真实编译，平均增加多少时间成本？

实验在没有 Ascend NPU 的 Mac 环境执行。每个候选从固定 `.ttadapter` 输入开始，执行到真实
local PlanMemory 完成后停止，不生成可运行二进制，也不执行真机 benchmark：

```text
.ttadapter
  -> BiShengIR 前缀 pipeline
  -> CVPipelining 前执行轻量模型（shadow/prune）
  -> BiSheng fallback（如果发生 UB overflow）
  -> CVPipelining
  -> local PlanMemory
  -> BISHENGIR_STOP_AFTER_LOCAL_PLAN_MEMORY
```

因此，这里测量的是模型对 **Autotune 编译阶段总时间** 的影响，不是 kernel 的设备运行性能，
也不会从候选中选出真机运行最快的配置。

## 2. 三种测试模式

相同的 adapter 和 Autotune Config 分别运行以下三种模式：

| 模式 | `enable-ub-overflow-prediction` | `prune-predicted-ub-overflow` | 行为 |
| --- | --- | --- | --- |
| `baseline` | `false` | `false` | 不执行轻量模型，走真实 pipeline 到 PlanMemory |
| `shadow` | `true` | `false` | 执行轻量模型，但无论预测结果如何都继续真实 pipeline |
| `prune` | `true` | `true` | 模型精确预测 overflow 时提前结束当前 attempt，并进入现有 BiSheng fallback |

`baseline` 用于给出没有模型时的编译时间。`shadow` 隔离模型本身的时间成本；`prune` 测量
“模型成本 + overflow 提前结束收益”合并后的净效果。

BiSheng fallback 语义保持不变。例如原配置 overflow 后，编译器可能依次关闭 code motion 和
auto multi-buffer 后重试。一次外层 Autotune Config 因而可能包含多个内部编译 attempt，模型也会
在每个 attempt 的 CVPipelining 前执行一次。

## 3. 指标定义

### 3.1 未发生 overflow 时的模型成本

只统计 `baseline` 和 `shadow` 都成功到达 PlanMemory，并且没有 overflow、fallback 或 timeout
的配对候选：

```text
总额外时间 = sum(shadow_wall_ns) - sum(baseline_wall_ns)

平均每候选额外时间 = 总额外时间 / 配对候选数

额外时间比例 = 总额外时间 / sum(baseline_wall_ns) * 100%
```

这里使用 `shadow`，因为它执行模型但不剪枝，最接近“没有发生 overflow 时模型增加的纯成本”。

### 3.2 完整 Autotune compile-only 加速

在相同 Config 集合上配对 `baseline` 和 `prune`：

```text
节省时间 = sum(baseline_wall_ns) - sum(prune_wall_ns)

节省比例 = 节省时间 / sum(baseline_wall_ns) * 100%

加速比 = sum(baseline_wall_ns) / sum(prune_wall_ns)
```

所有 Config 都会执行。这里的收益来自 overflow attempt 在 PlanMemory 前被模型提前终止，并非
减少了外层 Autotune Config 数量。

## 4. 搜索参数

compile-only 使用与现有 Ascend Autotune 对齐的参数集合。`num_stages` 和 `multibuffer` 最终控制
同一个 BiSheng 参数，因此归一化后共有 9 个独立变量：

| 参数 | 搜索值 | BiSheng 参数 |
| --- | --- | --- |
| `num_stages` | `1, 2` | `--enable-auto-multi-buffer` |
| `unit_flag` | `false, true` | `--enable-hivm-unit-flag-sync` |
| `limit_auto_multi_buffer_only_for_local_buffer` | `false, true` | 同名参数 |
| `limit_auto_multi_buffer_of_local_buffer` | `no-l0c, no-limit` | 同名参数 |
| `set_workspace_multibuffer` | `2, 4` | 同名参数 |
| `enable_hivm_auto_cv_balance` | `false, true` | 同名参数 |
| `tile_mix_vector_loop` | `2, 4, 8` | 同名参数 |
| `tile_mix_cube_loop` | `2, 4, 8` | 同名参数 |
| `enable_ubuf_saving` | `false, true` | `--enable-ubuf-saving` |

不同 kernel type 使用不同搜索 profile：

| Kernel type | 候选数 | 参与搜索的变量 |
| --- | ---: | --- |
| `vector` | 4 | `num_stages`、`enable_ubuf_saving` |
| `cube` | 8 | `num_stages`、`unit_flag`、local multi-buffer 限制 |
| `mixcv` | 1152 | 全部 9 个独立变量 |

manifest 还固定传入以下公共参数，三种模式保持一致：

```text
--enable-hfusion-compile=true
--enable-hivm-compile=true
--enable-triton-kernel-compile=true
--disable-auto-cv-work-space-manage=false
--enable-preload=false
--enable-code-motion=true
--enable-auto-bind-sub-block=true
--enable-hivm-auto-storage-align=true
--limit-auto-multi-buffer-buffer=only-cube
--enable-hivm-cross-core-gss=true
--enable-hivm-inject-block-all-sync=false
--disable-auto-inject-block-sync=false
```

## 5. 运行准备

从 Triton-Ascend 仓库根目录执行。首先准备带轻量模型的 `bishengir-compile`：

```bash
cmake --build /path/to/AscendNPU-IR/build \
  --target bishengir-compile \
  -j8

export BISHENGIR_COMPILE_PATH=/path/to/AscendNPU-IR/build/bin/bishengir-compile
```

建议先确认编译器和输入文件：

```bash
test -x "$BISHENGIR_COMPILE_PATH"
test -f third_party/ascend/AscendNPU-IR/ub_overflow_model_cpp/data/adapter/attn_fwd.ttadapter
```

正式计时使用 `jobs=1`，避免多个 BiSheng 进程争用 CPU 和内存。三种模式以固定顺序种子交错
运行，减轻系统负载随时间变化造成的偏差。计时期间不要打开 validation、IR dump 或详细 pass
timing。

## 6. 实验一：attn_fwd × 1152

### 6.1 运行命令

`attn_fwd` 使用 `mixcv` profile，完整展开 1152 个 Config。每个 Config 运行三种模式，共执行
`1152 × 3 = 3456` 个候选：

```bash
python3 third_party/ascend/backend/runtime/adapter_compile_only.py \
  --manifest third_party/ascend/backend/runtime/configs/attn_fwd_compile_only_manifest.json \
  --compiler "$BISHENGIR_COMPILE_PATH" \
  --modes baseline,shadow,prune \
  --repeat 1 \
  --jobs 1 \
  --timeout 300 \
  --order-seed 0 \
  --progress-interval 100 \
  --report-dir output/attn_fwd_compile_only
```

运行前可用 dry-run 确认候选数：

```bash
python3 third_party/ascend/backend/runtime/adapter_compile_only.py \
  --manifest third_party/ascend/backend/runtime/configs/attn_fwd_compile_only_manifest.json \
  --compiler "$BISHENGIR_COMPILE_PATH" \
  --modes baseline,shadow,prune \
  --repeat 1 \
  --jobs 1 \
  --timeout 300 \
  --order-seed 0 \
  --report-dir output/attn_fwd_compile_only \
  --dry-run
```

中断后可在参数和编译器二进制均未变化时增加 `--resume`，从 `results.jsonl` 检查点继续。

### 6.2 测量结果

2026-07-28 的完整结果如下：

| 模式 | Config 数 | 总时间 | 结果 |
| --- | ---: | ---: | --- |
| `baseline` | 1152 | 431.773 s | 1152 个真实 UB overflow |
| `shadow` | 1152 | 671.984 s | 1152 个真实 UB overflow |
| `prune` | 1152 | 326.615 s | 1152 个预测 overflow 并提前剪枝 |

最终净收益：

```text
节省时间 = 431.773 s - 326.615 s = 105.157 s
节省比例 = 105.157 s / 431.773 s = 24.355%
加速比   = 431.773 s / 326.615 s = 1.3220x
```

即在 `attn_fwd × 1152` 的完整 Autotune compile-only 搜索中，引入轻量模型并启用剪枝后，
**总编译时间缩短 24.35%，加速约 1.32 倍**。

该输入的 1152 个 Config 全部 overflow，因此它适合测量剪枝收益，但不能用于测量新的
non-overflow 快速返回路径。由于 fallback，`shadow` 和 `prune` 各执行了 2880 次模型调用。

结果文件：

```text
output/attn_fwd_direct_mlir_all_2118af59_20260728/summary.json
output/attn_fwd_direct_mlir_all_2118af59_20260728/results.jsonl
output/attn_fwd_direct_mlir_all_2118af59_20260728/comparison_with_20260727.md
```

这次结果使用固定编译器快照
`bishengir-compile-2118af59fd0a-dirty-01b14fe0f4c9`，SHA-256 为
`3a7bc7decc0e95bdee11e95d69a07eab924fea4a2062616409e97398dcd2ee95`。快照包含当时尚未提交的
direct-MLIR API 改动；若使用其他编译器版本复跑，应保留新的版本和二进制哈希，不应把两种
二进制的 checkpoint 合并。

## 7. 实验二：典型无 overflow 算子的成本

### 7.1 输入和运行命令

代表性 manifest 包含以下输入：

| Adapter | Profile | 本次状态 |
| --- | --- | --- |
| `python_tutorial_02-fused-softmax.ttadapter` | `vector` | 无 overflow |
| `ascend_tutorial_03-matrix-multiplication.ttadapter` | `cube` | 无 overflow |
| `python_tutorial_06-fused-attention.ttadapter` | `mixcv` | 无 overflow |
| `python_tutorial_09-persistent-matmul.ttadapter` | `cube` | 无 overflow |
| `attn_fwd.ttadapter` | `mixcv` | overflow，仅作对照，不计入本节成本 |

为快速得到代表性结果，每个 adapter 取搜索空间中的前 4 个 Config：

```bash
python3 third_party/ascend/backend/runtime/adapter_compile_only.py \
  --manifest third_party/ascend/backend/runtime/configs/adapter_compile_only_manifest.json \
  --compiler "$BISHENGIR_COMPILE_PATH" \
  --modes baseline,shadow,prune \
  --repeat 1 \
  --jobs 1 \
  --timeout 300 \
  --order-seed 0 \
  --limit-configs 4 \
  --progress-interval 10 \
  --report-dir output/adapter_compile_only_quick
```

总候选数为 `5 adapters × 4 configs × 3 modes = 60`。汇总器只将 4 个无 overflow 算子的
16 对 `baseline/shadow` 候选纳入 `no_overflow_model_overhead`，自动排除 `attn_fwd`。

### 7.2 分算子结果

2026-07-27 的代表性测试结果如下：

| 算子 | 配对 Config 数 | Baseline 总时间 | Shadow 总时间 | 平均每 Config 增加 | 增加比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matrix multiplication | 4 | 194.512 ms | 237.513 ms | 10.750 ms | 22.11% |
| Fused softmax | 4 | 131.512 ms | 144.921 ms | 3.352 ms | 10.20% |
| Fused attention | 4 | 533.926 ms | 646.463 ms | 28.134 ms | 21.08% |
| Persistent matmul | 4 | 274.841 ms | 348.381 ms | 18.385 ms | 26.76% |

聚合结果：

```text
Baseline 总时间 = 1134.790 ms
Shadow 总时间   = 1377.278 ms
总额外时间      = 242.488 ms
平均每 Config   = 242.488 ms / 16 = 15.155 ms
加权额外比例    = 242.488 ms / 1134.790 ms = 21.369%
```

四个算子各自百分比的简单算术平均为：

```text
(22.11% + 10.20% + 21.08% + 26.76%) / 4 = 20.03%
```

因此可以表述为：在这组典型无 overflow 输入上，轻量模型让单个候选平均增加约
**15.16 ms**；按总基线时间加权，编译成本增加约 **21.37%**，按四个算子等权平均约
**20.03%**。

该数据是 2026-07-27 使用当时模型版本得到的历史实测值。新版模型已经加入部分 non-overflow
结果的提前返回，理论上可能降低本节成本；在新版编译器上重新执行相同命令前，应将 21.37%
视为已有基线，而不是新版路径的重新测量结果。

结果文件：

```text
output/adapter_compile_only_quick_20260727/summary.json
output/adapter_compile_only_quick_20260727/results.jsonl
```

## 8. 如何读取输出

每次运行生成：

```text
<report-dir>/results.jsonl
<report-dir>/summary.json
```

`results.jsonl` 每行对应一个 adapter、Config、mode 和 repeat，包含：

- `candidate_wall_ns`：候选的完整墙钟时间；
- `status`：最终到达 PlanMemory、真实 overflow 或模型预测 overflow；
- `attempt_results`：每次 BiSheng fallback attempt 的模型结果；
- `fallback_count`：内部 fallback 次数；
- `model_ns`：候选中所有模型 attempt 的内部耗时总和；
- `decision_path`：完整 plan 或 non-overflow 上界快速返回路径。

`summary.json` 中重点查看：

```text
metrics.no_overflow_model_overhead
metrics.overall_prune_speedup
mode_summaries.baseline
mode_summaries.shadow
mode_summaries.prune
```

## 9. 结果解释与限制

1. `attn_fwd` 的 24.35% 是 overflow 密集输入上的 compile-only 加速，不代表所有算子的平均收益。
2. 无 overflow 成本来自 4 个典型算子、每个 4 个 Config 的小样本。它适合说明成本量级，若要给出
   统计置信区间，应增加 repeat，并扩大每个算子的 Config 数。
3. 当前实验止于 local PlanMemory，不包括 PlanMemory 后端、链接、设备加载和真机 benchmark。
4. 不同日期的系统负载和编译器版本会影响绝对秒数。跨版本比较优先使用配对后的百分比，并记录
   adapter 哈希、编译器提交和二进制 SHA-256。
5. 新模型支持部分 non-overflow 上界快速返回，但 `attn_fwd` 没有命中该路径；应另选无 overflow
   数据集评估这项优化，而不能由 overflow 剪枝结果外推。
