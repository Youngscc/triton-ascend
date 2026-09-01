# PlanComputeBlock UB Cost Model

输入 A5 DynamicCV 的 `PlanComputeBlock` 输出 IR，在固定 `vf_merge_level=0` 和编译器 profile 下，
预测 `d=intra_cache_num` 取 `1..3`、`m=multibuffer_num` 取 `1..4` 时相对 `(1,1)` 的 UB 变化：

```text
Delta_total(d,m) = Delta_d(d) + Delta_m(m) + Delta_dm(d,m)
```

如果提供绑定 `d=1,m=1,v=0`、IR、profile 和已验证编译器版本来源的真实 `PlanMemory` 证书，
还会得到：

```text
U(d,m) = U(1,1) + Delta_total(d,m)
```

## 代码结构

只需要先读 `cost_model.py`。它是唯一的模型主文件，依次调用 `stages/` 中的五个阶段：

```text
cost_model_demo/
├── cost_model.py                    模型主流程和编译器/autotune API
├── stages/
│   ├── prepare_input.py             Stage 1：准备 IRGraph 和编译上下文
│   ├── build_ir_graph.py            Stage 1 的 IR 索引实现
│   ├── validate_context.py          Stage 1 的上下文验证实现
│   ├── analyze_dynamic_cv.py        Stage 2：识别 DynamicCV 候选缓冲
│   ├── analyze_buffer_families.py   Stage 3：普通缓冲、大小和参数归属
│   ├── buffer_geometry.py           两类候选缓冲共用的 AIV/对齐投影
│   ├── build_parametric_model.py    Stage 4：综合 D[d]、B 和交互项
│   └── evaluate.py                  Stage 5：查询配置并校准 UB
│
├── model_types.py                   各阶段共享的数据结构
├── memory_planner.py                可选 PlanLite，不属于 pure-Delta 主流程
├── cli.py                           命令行入口
├── benchmark.py                     Prepare/Evaluate 微基准
├── test_cost_model.py               回归测试
├── a5_profile.json                  固定编译器 profile
├── baseline_certificate.schema.json baseline 证书格式
└── fused_d1_baseline.example.json   Fused 示例证书
```

## 主流程

`cost_model.py::prepare_cost_model()` 中的调用顺序就是模型准备流程：

```python
graph, context = prepare_input(
    text,
    vf_merge_level=vf_merge_level,
    profile=profile,
    observed_provenance=provenance,
)
dynamic = analyze_dynamic_cv(graph, context.profile)
analysis = analyze_buffer_families(graph, context.profile, dynamic)
prepared = build_parametric_model(graph, context, analysis)
```

随后 Stage 5 对一个配置求值：

```python
estimate = evaluate_configuration(
    prepared,
    intra_cache_num=d,
    multibuffer_num=m,
    baseline=baseline,
)
```

`run_cost_model()` 串起全部五个阶段；`UbCostModel` 则缓存前四个阶段的 `PreparedCostModel`，供
autotune 循环反复执行 Stage 5。

## 五个阶段

### Stage 1：Prepare Input

输入 PlanComputeBlock IR 和固定编译 profile，输出 `IRGraph + ValidatedContext`。该阶段建立 SSA、循环、
compute block、GM boundary 等只读索引，同时检查 A5、`vf_merge_level=0`、DynamicCV count、AIV 投影和
编译器版本来源。它只回答“这份输入能否按当前模型解释”，不筛选 UB buffer。

### Stage 2：识别 DynamicCV 候选缓冲

输入 `IRGraph + CompilerProfile`，输出 `DynamicCVAnalysis`。该阶段按 loop 建立带权跨 compute-block
依赖图，比较调整计算位置前后的跨块数据量；只有新边界更小时才调整 operation 的计算块归属，随后
从调整后的 Vector 跨块依赖中提取 DynamicCV 候选缓冲。边界字节数无法闭合、同一起点产生多条
独立边界、消费端分叉过多等情况会记录为不支持原因。

### Stage 3：确定缓冲大小与参数归属

输入 `IRGraph + DynamicCVAnalysis`，输出统一的 `BufferAnalysis`：

1. 从 GM load/store boundary 识别 ordinary MultiBuffer 候选；
2. 沿 view/cast 链解析真实 allocation、shape 和 dtype；
3. 投影单个 AIV function 的物理大小并按 UB 规则对齐；
4. 按 DynamicCV、fixed GM-load、Fixpipe、preload、ordinary 的优先级确定唯一参数归属；
5. 检查 DynamicCV 与普通 MultiBuffer 候选缓冲是否重合或存在尚未建模的耦合。

Stage 2 负责“谁受 d 影响”，Stage 3 负责“谁受 m 影响、每份多大、由哪个机制控制”。二者发现的
不支持原因在这里汇总；只要存在不支持原因，Stage 5 就返回 `unknown` 并继续真实编译。

### Stage 4：Synthesize Parameter Model

输入 `IRGraph + ValidatedContext + BufferAnalysis`，输出可复用的 `PreparedCostModel`：

```text
D[d] = DynamicCV 相对 d=1 的 profile-rule Delta
B    = m 每增加 1 的 ordinary entry Delta
X    = 0，仅适用于当前已验证的独立结构
```

该阶段把每个候选缓冲的单份字节数与参数控制的份数结合起来，生成 `D[d]`、
`ordinary_step_table[d]`、`coupled_adjustment[d,m]`、候选缓冲明细和不支持原因。当前支持域内
DynamicCV 的有效份数为 `[1,2,3]`，ordinary 单步对所有 `d` 取同一个 `B`，交互项为 0。

### Stage 5：Evaluate Configurations

输入 `PreparedCostModel + (d,m) + optional baseline`，输出 `CostEstimate`：

```text
Delta_d       = D[d]
Delta_m       = (m-1) * B[1]
Delta_total   = D[d] + (m-1) * B[d] + X[d,m]
Delta_dm      = Delta_total - Delta_d - Delta_m
```

没有 baseline 时只输出 pure Delta。提供严格绑定 `(d,m,v)=(1,1,0)`、IR identity、profile、model
和 compiler revision identity 的 PlanMemory baseline 后，可输出 `U_calibrated` 和保守 verdict。当前
PlanLite 没有接入 Evaluate，模型不返回 `overflow`，也不会提前剪枝。

## API

单次执行完整模型：

```python
from experiment_operators.cost_model_demo import run_cost_model

estimate = run_cost_model(
    module_op,
    intra_cache_num=3,
    multibuffer_num=4,
    vf_merge_level=0,
    baseline=certificate,
)
```

autotune 中准备一次、求值多次：

```python
from experiment_operators.cost_model_demo import UbCostModel

model = UbCostModel()
prepared = model.prepare(module_op, vf_merge_level=0)

estimate = model.evaluate(
    prepared,
    intra_cache_num=3,
    multibuffer_num=4,
    baseline=certificate,
)
```

## 命令行

```bash
python3 -m experiment_operators.cost_model_demo \
  --ir outputs/mac_ub_latest_dev_prev_npuir_20260825/mlir/fused_attention/dynamic_1/after-plan-compute-block.mlir \
  --intra-cache-num 3 \
  --multibuffer-num 4 \
  --vf-merge-level 0 \
  --profile-json experiment_operators/cost_model_demo/a5_profile.json \
  --baseline-certificate experiment_operators/cost_model_demo/fused_d1_baseline.example.json \
  --benchmark-runs 0
```

Fused `(3,4)` 的预期结果为：

```text
Delta_total = 13184 B
U(3,4)      = 76288 B = 610304 bit
```

上述命令没有提供可验证的编译器版本来源，因此 `76288 B` 是 calibrated 数值，verdict 为 `unknown`。
只有实际 revision manifest 与 profile 匹配时，完整 no-reuse baseline 才能得到 `safe`；当前主流程
无论哪种 verdict 都继续真实编译。

人类可读输出还会列出 DynamicCV 跨块边界调整明细：

```text
DynamicCV 分析: 跨块依赖=11 边界调整=2 候选缓冲=2
DynamicCV 跨块边界调整:
  %m_ij_69 b11->b12: 512B->256B; 候选缓冲=%alpha
  %p b11->b12: 33024B->256B; 候选缓冲=%l_ij_81
```

Python 调用方可读取 `prepared.dynamic_cuts` 和 `prepared.dynamic_buffers`；JSON 输出会完整保留
起点数据、调整前后跨块字节数、被移动的 value、调整后的 block、AIV shape 和来源证据。

## 大规模 shape 与源码结构编译器验证

`large_validation_corpus.json` 定义 Fused Attention 的全局尺寸、局部 tile、shape 交互、拓扑边界
和受控源码 DAG 变体。验证器不修改原始 candidate，而是在结果目录生成每个 case 的完整源码；
随后从 Python kernel 重新编译 `d=1..3`，固定 `m=1,v=0`，比较 PlanComputeBlock identity、block
rewrite、InnerScope raw alloc 和最终 PlanMemory Delta：

```bash
PYTHONPATH=/path/to/built-triton-ascend/python:$PWD \
/path/to/built-python \
  experiment_operators/cost_model_demo/run_shape_validation.py \
  --worktree "$PWD" \
  --compiler /path/to/bishengir-compile \
  --triton-mlir-opt /path/to/triton-mlir-opt \
  --corpus experiment_operators/cost_model_demo/large_validation_corpus.json \
  --output-dir /path/to/shape-validation-output \
  --compact \
  --strict
```

可用多个 `--case CASE_ID` 只执行部分 corpus。输出包括：

```text
shape_validation.csv       每个 shape/d 的完整比较
shape_validation.json      case 汇总、编译器身份和结论
cases/                     实际源码、PlanComputeBlock、TTAdapter 和关键 pass IR
logs/                      真实 PlanMemory 编译命令和诊断
```

`exact_prediction` 表示模型和编译器逐层一致；`correct_fail_open` 表示真实编译器仍可执行，但模型
发现图不完整并返回 `unknown`。两者都通过安全性验证，只有模型输出了错误数值或该拒绝的结构没有
拒绝时才是 `mismatch`。当前域包含 29 个 case/87 个真实编译配置：22 个精确预测，7 个结构边界
正确 fail-open，0 个 mismatch。完整方案和结果见
[`DYNAMIC_CV_MODEL_VALIDATION_REPORT.md`](../DYNAMIC_CV_MODEL_VALIDATION_REPORT.md)。

## 验证

```bash
PYTHONPYCACHEPREFIX=/tmp/cost-model-pycache \
python3 -m unittest experiment_operators.cost_model_demo.test_cost_model
```

测试覆盖四个真实算子的 48 个 `(d,m)` Delta、由它们派生的 92 个相邻/混合差分、跨块边界调整证据、
shape/stride 参数化、真实 raw-alloc oracle、名称无关性、baseline identity、Prepare cache、fail-open
和 PlanLite 安全性质。完整算法与编译器对应关系见
[`UB_COST_MODEL_DESIGN.md`](../UB_COST_MODEL_DESIGN.md)。
