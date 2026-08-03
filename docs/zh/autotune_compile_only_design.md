# Adapter 到 PlanMemory 的 Autotune Compile-Only 改造方案

## 1. 背景与目标

本方案用于在没有 Ascend NPU、driver、`torch_npu` 和真机 benchmark 的 Mac 环境中，验证轻量
UB overflow 模型对 autotune 编译总耗时的影响。

每个 autotune 候选都从固定的 `.ttadapter` 文件开始，启动一个独立的
`bishengir-compile` 进程，执行到真实本地 PlanMemory 完成后停止：

```text
adapter
  -> BiSheng 前缀 pipeline
  -> CVPipelining 前运行轻量模型
  -> BiSheng 内部 UB overflow fallback（如有）
  -> CVPipelining 及后续 pass
  -> 真实本地 PlanMemory
  -> BISHENGIR_STOP_AFTER_LOCAL_PLAN_MEMORY
```

本方案只测量 compile-only autotune，不运行设备 benchmark，也不选择运行时间最短的配置。
所有候选仍会执行，模型的收益来自缩短发生 UB overflow 的内部编译 attempt，而不是减少外层
autotune 候选数量。

## 2. 设计原则

1. 默认 autotune 行为不变。仅当显式设置 `compile_only=True` 时进入新路径。
2. 尽量复用现有 Ascend autotune 的 `Config`、参数校验、kernel type 参数集合和候选展开逻辑。
3. compile-only 路径不访问 Tensor、NPU driver、device、stream、launcher 或 profiler。
4. 每个候选启动独立的 `bishengir-compile` 进程，保证从 adapter 重新执行完整前缀。
5. 保留 BiSheng 当前 UB overflow fallback，不增加 prediction fail-fast 语义。
6. timed run 不启用 validation dump、PlanMemory dump 或详细 pass timing，避免测量被诊断开销污染。
7. baseline、shadow、prune 必须使用相同 adapter、候选配置、固定参数、顺序策略和重复次数。

## 3. 总体架构

```text
adapter manifest
  -> AutoTilingTuner(compile_only=True)
       -> 按 kernel type 展开全部合法 Config
       -> AdapterCompileExecutor
            -> config 映射为 BiSheng CLI 参数
            -> subprocess: bishengir-compile adapter ...
            -> 解析模型结果、fallback、PlanMemory 和错误
       -> 每个候选立即写 JSONL checkpoint
       -> 汇总每轮和每种模式的时间报告
```

普通 autotune 保持原路径：

```text
AutoTilingTuner.run()
  -> generate_key_and_configs()
  -> prune_configs()
  -> _batch_bench()
  -> benchmark
  -> min(timings)
  -> 最终 kernel launch
```

compile-only 路径不复用 benchmark 和最终 launch，仅复用适合 adapter 输入的配置基础设施。

## 4. Autotuner 改造

修改 `third_party/ascend/backend/runtime/autotuner.py`，为 `AutoTilingTuner` 和 Ascend
`autotune()` 入口增加可选参数：

```python
compile_only: bool = False
compile_only_options: Optional[CompileOnlyOptions] = None
```

在 `AutoTilingTuner.run()` 最前面增加分支：

```python
def run(self, *args, **kwargs):
    if self.compile_only:
        return self._run_adapter_compile_only()

    # 现有 autotune 逻辑保持不变。
    key = self.generate_key_and_configs(*args, **kwargs)
    ...
```

构造函数也只在 `compile_only=True` 时初始化 compile-only 所需的最小状态，跳过 kernel AST、
axis、Tensor dtype 和设备相关初始化。默认分支维持当前初始化顺序和字段语义。

compile-only 分支不调用以下现有能力：

- `generate_key_and_configs()` 中的 Tensor/dtype 分析；
- 自动生成 Triton 前端 tiling 参数；
- `_make_kernel_call()`；
- `do_bench()` 和 NPU profiler；
- `_prune_by_time_limit()`；
- `min(timings)` 和 `best_config`；
- 最终 `self.fn.run()` 和 kernel launch；
- 原 autotune 的设备相关磁盘缓存。

compile-only 返回 `CompileOnlySummary`，`best_config` 明确为 `None`，不能把编译最快的配置解释
为设备运行最快的配置。

## 5. 全量候选搜索空间

继续复用现有：

- `triton.Config`；
- `_ALL_PARAMS`；
- `_VALIDATION_RULES`；
- `_VECTOR_PARAMS`；
- `_CUBE_PARAMS`；
- `_MIXCV_PARAMS`；
- `get_max_configs()`（已安装 Triton 的模块入口）。

standalone 入口不能导入 Triton 扩展，因此用标准库 `itertools.product` 展开同一组
`KERNEL_TYPE_PARAMS` 和 `ALL_SEARCH_VALUES`。测试必须校验两种入口的候选数量、归一化配置和
唯一性完全相同。

新增独立的全量搜索值，不修改原 `_DEFAULTS`：

```python
ALL_SEARCH_VALUES = {
    "num_stages": [1, 2],
    "unit_flag": [False, True],
    "limit_auto_multi_buffer_only_for_local_buffer": [False, True],
    "limit_auto_multi_buffer_of_local_buffer": ["no-l0c", "no-limit"],
    "set_workspace_multibuffer": [2, 4],
    "enable_hivm_auto_cv_balance": [False, True],
    "tile_mix_vector_loop": [2, 4, 8],
    "tile_mix_cube_loop": [2, 4, 8],
    "enable_ubuf_saving": [False, True],
}
```

autotune 对外有 10 个参数名，但 `num_stages` 与 `multibuffer` 最终都控制
`--enable-auto-multi-buffer`。compile-only 将二者归一化为一个独立维度：

```text
num_stages=1 -> multibuffer=false -> enable-auto-multi-buffer=false
num_stages=2 -> multibuffer=true  -> enable-auto-multi-buffer=true
```

禁止生成二者冲突或仅名称不同的重复候选。因此实际是 10 个参数名、9 个独立变量。

按现有 kernel type 参数集合全量穷举：

| Kernel type | 参与变量 | 候选数 |
| --- | ---: | ---: |
| Vector | `num_stages`、`enable_ubuf_saving` | 4 |
| Cube | `num_stages`、`unit_flag`、local multi-buffer strategy | 8 |
| MIXCV | 全部 9 个独立变量 | 1152 |

manifest 中的 `kernel_type` 表示现有 autotune 使用的搜索参数 profile，不等同于 adapter 在后续
BiSheng pipeline 中形成的 `mix_mode` IR attribute。例如 matmul adapter 可以在 BiSheng 中形成
MIX IR，但实验仍可按现有 `kernel_type="cube"` profile 只搜索 Cube autotune 参数。

`search_space=all` 时禁用 `early_config_prune`、perf model 和 top-k，非法或完全重复的配置除外。

## 6. 参数到 BiSheng 的映射

每个候选必须显式传递以下参数：

| Autotune 参数 | BiSheng 参数 |
| --- | --- |
| `num_stages` / `multibuffer` | `--enable-auto-multi-buffer` |
| `unit_flag` | `--enable-hivm-unit-flag-sync` |
| `limit_auto_multi_buffer_only_for_local_buffer` | `--limit-auto-multi-buffer-only-for-local-buffer` |
| `limit_auto_multi_buffer_of_local_buffer` | `--limit-auto-multi-buffer-of-local-buffer` |
| `set_workspace_multibuffer` | `--set-workspace-multibuffer` |
| `enable_hivm_auto_cv_balance` | `--enable-hivm-auto-cv-balance` |
| `tile_mix_vector_loop` | `--tile-mix-vector-loop` |
| `tile_mix_cube_loop` | `--tile-mix-cube-loop` |
| `enable_ubuf_saving` | `--enable-ubuf-saving` |

其余会影响 adapter 到 PlanMemory 的固定编译参数必须显式保存在 manifest 或
`CompileOnlyOptions` 中，并在三种模式中保持一致。配置身份必须同时包含 adapter 内容哈希、
BiSheng 二进制身份、全部固定参数和全部 tunable 参数。

## 7. Adapter 编译执行器

新增文件：

```text
third_party/ascend/backend/runtime/adapter_compile_only.py
```

主要类型：

```python
CompileOnlyOptions
CompileOnlyCandidateResult
CompileOnlySummary
AdapterCompileExecutor
```

每个 Config 运行一个新进程：

```bash
BISHENGIR_STOP_AFTER_LOCAL_PLAN_MEMORY=1 \
BISHENGIR_UB_MODEL_EMIT_RESULT=1 \
bishengir-compile input.ttadapter \
  -o /dev/null \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --enable-ub-overflow-prediction=<true|false> \
  --prune-predicted-ub-overflow=<true|false> \
  <fixed options> \
  <current Config options>
```

执行器直接调用 BiSheng，不经过 `compiler.py` 中要求最终 `.o` 存在的普通 npubin 路径。
`BISHENGIR_STOP_AFTER_LOCAL_PLAN_MEMORY=1` 下没有二进制是预期行为，不能判定为编译失败。

## 8. BiSheng fallback 语义

保持 AscendNPU-IR 当前行为，不增加 fail-fast 参数：

```text
attempt 1: 使用 autotune 原始配置
  -> 模型预测 overflow
attempt 2: BiSheng fallback 设置 enable-code-motion=false
  -> 仍预测或真实发生 overflow
attempt 3: BiSheng fallback 设置 enable-auto-multi-buffer=false
  -> 成功到达 PlanMemory，或最终失败
```

一次 `bishengir-compile` 可能产生多条 `BISHENGIR_UB_MODEL_RESULT`。执行器必须保留并解析全部
attempt，不能只读取第一条或最后一条。

候选经过 fallback 后成功，记录为 `success_after_fallback`；只有全部 fallback 耗尽仍失败，才记
为最终 overflow。模型结果为 blocker/incomplete 时不提前结束，继续真实 pipeline。

## 9. 三种对照模式

每个 adapter 和 Config 都运行三种模式：

| 模式 | Prediction | Prune | 行为 |
| --- | --- | --- | --- |
| `baseline` | false | false | 不运行模型，真实执行到 PlanMemory |
| `shadow` | true | false | 运行模型但不提前结束，继续真实 PlanMemory |
| `prune` | true | true | 精确预测 overflow 时触发现有 BiSheng fallback |

两个模型布尔参数必须始终显式传入，不能依赖 BiSheng 默认值。

timed run 不设置以下环境变量：

```text
BISHENGIR_UB_MODEL_VALIDATION
BISHENGIR_DUMP_PLAN_MEMORY_ATTEMPTS
BISHENGIR_PLAN_MEMORY_FORCE_SEED
```

成功返回且启用了 `BISHENGIR_STOP_AFTER_LOCAL_PLAN_MEMORY`，即表示最终 attempt 已完成真实本地
PlanMemory。详细模型与真实 PlanMemory 对齐属于独立 correctness run，不计入性能时间。

## 10. 代表性输入

新增 manifest，例如：

```text
third_party/ascend/backend/runtime/configs/adapter_compile_only_manifest.json
```

首批输入：

```json
{
  "adapter_root": "third_party/ascend/AscendNPU-IR/ub_overflow_model_cpp/data/adapter",
  "adapters": [
    {
      "path": "python_tutorial_02-fused-softmax.ttadapter",
      "kernel_type": "vector"
    },
    {
      "path": "ascend_tutorial_03-matrix-multiplication.ttadapter",
      "kernel_type": "cube"
    },
    {
      "path": "python_tutorial_06-fused-attention.ttadapter",
      "kernel_type": "mixcv"
    },
    {
      "path": "python_tutorial_09-persistent-matmul.ttadapter",
      "kernel_type": "cube"
    },
    {
      "path": "attn_fwd.ttadapter",
      "kernel_type": "mixcv"
    }
  ]
}
```

`adapter_root` 相对于 Triton 仓库根目录解析；CLI 启动后立即将所有输入转换为绝对路径，后续
checkpoint 和子进程执行不依赖当前工作目录。

`attn_fwd.ttadapter` 是已知 UB 压力输入：production-default 下 required 为 1716224 bits、
capacity 为 1572864 bits，并会触发 BiSheng UB overflow fallback。正式运行前仍需对其余输入做
一次类型校准，但不再依赖运行过程中临时选择 overflow 样本。

按当前分类，每个 mode/repeat 共有 `4 + 8 + 1152 + 8 + 1152 = 2324` 个候选。三种模式、重复
5 次共 34860 次 BiSheng 进程调用，内部 fallback 还可能让单个进程执行多个 pipeline attempt。

## 11. CLI 与执行口径

Mac 源码树执行不要求先构建或导入 Triton Python 扩展，使用同一实现文件的 standalone 入口：

```bash
python3 third_party/ascend/backend/runtime/adapter_compile_only.py \
  --manifest third_party/ascend/backend/runtime/configs/adapter_compile_only_manifest.json \
  --search-space all \
  --modes baseline,shadow,prune \
  --repeat 5 \
  --jobs 1 \
  --timeout 300 \
  --progress-interval 100 \
  --report-dir output/adapter_compile_only
```

已安装并可导入 Triton 时，也可以使用模块入口：

```bash
python -m triton.backends.ascend.runtime.adapter_compile_only <同上参数>
```

模块入口通过 `AutoTilingTuner(compile_only=True)` 分支执行；standalone 入口仅用标准库承载相同
的全量配置组合，避免 Mac 因缺少 Triton 扩展或 PyTorch 而无法启动。两者共享同一 BiSheng
执行器、参数映射、checkpoint 和报告代码。

开发 smoke 可增加 `--limit-configs 1 --repeat 1`；正式实验不得使用 `--limit-configs`。

正式性能口径：

```text
jobs   = 1
repeat = 5
汇总   = median，同时保留 min/max/p95
顺序   = 使用固定随机种子轮换 mode 和候选顺序
```

允许 `--jobs > 1` 做补充并发实验，但不能与 `jobs=1` 的主结果混合。

## 12. Checkpoint 与恢复

长任务必须支持断点恢复：

1. 每个候选结束后立即追加一条 JSONL 记录并 flush。
2. 记录唯一 run identity 和 candidate identity。
3. `--resume` 只跳过 identity 完全匹配且已有终态的候选。
4. adapter、BiSheng、参数、模式或执行器版本变化时不得复用旧结果。
5. timeout、非 UB 编译错误和最终 overflow 都是已完成候选，可以恢复时跳过。
6. 进程被中断时只允许丢失正在运行的一个候选。

## 13. 结果 Schema

每个 `CompileOnlyCandidateResult` 至少包含：

```text
run_id
adapter_path / adapter_digest / kernel_type
mode / repeat / order_index
config_id / normalized_config / bisheng_arguments
compiler_returncode / timed_out
status
reached_plan_memory
candidate_wall_ns
model_serialize_ns / model_ns
model_status / precision / overflow
ub_peak_bits / required_bits / capacity_bits / selected_seed
attempt_count
attempt_results
fallback_count / fallback_actions
diagnostic_category
stderr_digest
```

候选终态至少区分：

```text
success_plan_memory
success_after_fallback
predicted_ub_overflow_final
native_ub_overflow_final
model_blocker_then_success
non_ub_compile_error
timeout
```

汇总同时记录候选耗时之和与完整 sweep 墙钟，核心指标为：

```text
shadow_overhead = shadow - baseline
prune_net_change = prune - baseline
overflow_attempt_saving = baseline 对应候选耗时 - prune 对应候选耗时
```

还需统计模型预测 overflow 数、fallback attempt 数、最终 overflow 数和到达 PlanMemory 的候选数。

## 14. 文件级改动

计划修改或新增：

```text
M third_party/ascend/backend/runtime/autotuner.py
A third_party/ascend/backend/runtime/adapter_compile_only.py
A third_party/ascend/backend/runtime/configs/adapter_compile_only_manifest.json
A third_party/ascend/unittest/autotune_ut/test_adapter_compile_only.py
A docs/zh/autotune_compile_only_design.md
```

AscendNPU-IR 已具备模型嵌入、机器结果和 PlanMemory 后停止能力，本阶段不修改其 fallback 语义。

## 15. 测试方案

单元测试必须覆盖：

1. Vector、Cube、MIXCV 候选数分别为 4、8、1152。
2. 1152 个 MIXCV 配置全部唯一。
3. `num_stages` 与 `multibuffer` 归一化正确且不存在冲突配置。
4. 全部 autotune 参数正确映射到 BiSheng CLI。
5. baseline、shadow、prune 显式传入正确模型参数。
6. 多条模型结果和 BiSheng fallback 可以完整解析。
7. PlanMemory 后没有 `.o` 文件仍判定成功。
8. blocker 不被误判为 non-overflow 或直接剪枝。
9. timeout、非 UB 错误、checkpoint 和 resume 行为正确。
10. fake compiler 可完成不依赖 NPU 的端到端测试。
11. `compile_only=False` 时新执行器不会被构造或调用。

集成验证：

1. Mac 上使用真实 `bishengir-compile` 和一个小 adapter 跑两个 Config。
2. 确认每个成功候选都执行到真实 PlanMemory 后停止。
3. 使用一个会触发 fallback 的候选确认多 attempt 解析。
4. 运行现有 Ascend autotune 单元测试，保证原有 kernel autotune 行为无回归。

## 16. 验收标准

实现完成需同时满足：

1. Mac 上不安装 NPU driver 也能从 adapter 启动 compile-only autotune。
2. 每个候选都从 adapter 开始，并在真实本地 PlanMemory 后结束。
3. 全部合法候选按 kernel type 完整执行，无 top-k 或性能模型剪枝。
4. BiSheng 内部 fallback 保持当前生产语义。
5. 三种模式使用完全一致的候选集合，能恢复中断任务。
6. 报告能够计算 shadow 模型开销和 prune 相对 baseline 的净时间变化。
7. 默认 autotune 不进入 compile-only 分支，现有测试全部通过。

`summary.json` 额外直接给出实验所需的两个配对指标：

1. `metrics.no_overflow_model_overhead`：仅统计 baseline 和 shadow 都成功且未发生 overflow/fallback
   的同一候选，报告平均每次候选执行的模型开销。
2. `metrics.overall_prune_speedup`：统计 baseline 和 prune 的全部配对候选，报告总节省时间、节省
   百分比和加速比。正的 `time_saved_ns` 表示启用模型剪枝后更快。

## 17. 当前实现验证记录

2026-07-27 在 Mac 上使用独立构建的真实 `bishengir-compile` 完成以下验证：

1. standalone dry-run 得到 Vector 4、Cube 8、MIXCV 1152 个唯一配置；当前 5 个 adapter、3 种
   mode、5 次 repeat 的正式任务总数为 34860。
2. 5 个 adapter 各取 1 个配置执行 baseline、shadow、prune，共 15 个真实 BiSheng 进程调用；
   普通候选均在 PlanMemory 后成功停止，且不要求生成 `.o`。
3. `attn_fwd.ttadapter` 在 baseline/shadow 下由真实 PlanMemory 报告 UB overflow，在 prune 下由
   模型于 CVPipelining 前报告相同 overflow；结果分类分别为 native 和 predicted。
4. `attn_fwd.ttadapter` 的 auto-multi-buffer 压力配置产生 3 个内部 attempt，依次 fallback
   `enable-code-motion=false`、`enable-auto-multi-buffer=false`，证明执行器保留了完整 BiSheng
   fallback 语义并能解析全部 action。
5. 对 15 个候选再次使用 `--resume` 时 `executed_candidates=0`、`resumed_candidates=15`。

## 18. Overflow 语料扩增

`adapter_overflow_screen.py` 用两阶段流程从现有 adapter corpus 中发现更多真实 UB overflow：

1. 每个 adapter 使用 8 个显式 UB 压力配置运行 prune。配置覆盖 auto-multi-buffer、local
   no-limit、CV balance、高 Vector/Cube tiling 和 unit sync，不依赖配置排序抽取。
2. 只对模型最终预测 overflow 的候选关闭模型重新运行 baseline，以真实 PlanMemory 的
   `native_ub_overflow_final` 作为确认标准。
3. 输出 `overflow_enriched_manifest.json` 和包含精确参数的 `confirmed_overflow_cases.json`。

```bash
python3 third_party/ascend/backend/runtime/adapter_overflow_screen.py \
  --adapter-root third_party/ascend/AscendNPU-IR/ub_overflow_model_cpp/data/adapter \
  --compiler /path/to/AscendNPU-IR/build/bin/bishengir-compile \
  --jobs 1 \
  --progress-interval 100 \
  --report-dir output/adapter_overflow_screen
```

此流程扩增的是现有 corpus 中可由合法 BiSheng 参数触发的 overflow 样本；它不降低 UB 容量，
也不向 IR 注入无意义 allocation。扩增语料应与自然语料分开报告，避免把筛选后的 overflow
比例解释为生产分布。
