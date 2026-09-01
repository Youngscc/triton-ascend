# DynamicCV UB 模型验证报告

## 结论

有必要扩大测试。只改变矩阵尺寸只能验证同一类 dependency DAG 上的 shape 投影，不能覆盖
block rewrite、分支汇合、多 consumer 和多 main-loop 等结构变化。

当前最终验证结果为：

| 验证层 | 规模 | 结果 |
| --- | ---: | --- |
| Python 单元与回归测试 | 32 项 | 32 通过 |
| 四个真实算子的二维 UB 表 | 48 个 `(d,m)` | 48 个总 Delta 逐字节一致 |
| 四个真实算子的局部差分 | 92 项 | 92 项逐字节一致 |
| 扩大后的 Fused 编译器 oracle | 29 case × 3 个 `d` = 87 个配置 | 87 个均完成真实 PlanMemory |
| oracle 数值覆盖 | 22 case × 3 = 66 个配置 | block rewrite、raw alloc、最终 Delta 全部一致 |
| oracle 安全边界 | 7 case × 3 = 21 个配置 | 全部正确返回 `unknown`，保留真实 UB |
| 错误数值预测 | 29 case | 0 |

因此，模型在当前明确支持域内是逐字节准确的；对尚未模拟的后续 compute-block 合并和跨阶段
结构会 fail-open，不会用部分 dependency 给出一个看似合理但错误的 Delta。

## 固定环境与判定方法

本报告使用模型 profile 固定的编译器组合：

```text
target                 Ascend950PR_9579
AscendNPU-IR           4b9f1a56092d66a991b857ca4ca2b40f2cf06e53
BishengIR              1.2.0
vendored LLVM          19.1.7 / d3fea2c7ae5436f63fa35b4d01e0aa76d1071396
DynamicCV d            1, 2, 3
ordinary multibuffer   1（固定）
VF merge               0（固定）
```

这是 host-only 编译验证，不需要 NPU，也不测 latency。每个 case 都从 Python Triton kernel 开始，
依次经过真实前端、PlanComputeBlock、DynamicCV 和 PlanMemory。三个 `d` 分别检查：

1. normalized PlanComputeBlock IR 是否只含允许的 count 差异；
2. 模型预测的 block rewrite 是否出现在真实中间 IR；
3. `AddMultiBufferInnerScope` 新增的 raw alloc 类型与份数是否一致；
4. 模型相对 `d=1` 的 UB Delta 是否与真实 PlanMemory 逐字节一致；
5. 证据不完整时是否返回 `unknown`。

`exact_prediction` 要求上述四层全部一致。`correct_fail_open` 要求模型不输出数值，而真实编译结果
仍完整保留。只有输出了错误数值或漏掉应拒绝的结构才算 `mismatch`。

## 测试方案

### 1. 模型单元与真实表回归

32 项测试覆盖：IR 图、循环携带依赖、跨块数据边界调整、候选缓冲来源证据、AIV shape 投影、
普通 MultiBuffer 候选缓冲、baseline identity、Prepare cache、PlanLite、fail-open 和名称无关性。

四个真实算子各有完整 `3×4` 的 `(d,m)` UB 表：

| 算子 | `Delta_d(1..3)` | 每增加一份 ordinary buffer | 二阶交互 |
| --- | --- | ---: | ---: |
| Fused Attention | `[0,256,512] B` | 4224 B | 0 B |
| Flash Attention | `[0,256,512] B` | 4224 B | 0 B |
| HSTU Attention | `[0,0,0] B` | 0 B | 0 B |
| Unified Attention | `[0,64,128] B` | 2048 B | 0 B |

由 48 个总 Delta 还能派生 92 个局部一致性检查：32 个相邻 `d` 差分、36 个相邻 `m` 差分和
24 个二维二阶差分，全部与真实表一致。这些差分共享同一组原始观测，不是 92 份相互独立的实验
样本。

### 2. 扩大 shape 网格

对 Fused Attention 改变 Python 编译期常量，而不是编辑生成后的 MLIR：

```text
N_CTX      256, 512, 1024, 2048, 4096
BLOCK_M    16, 32, 64, 128, 256
BLOCK_N    16, 32, 64, 128
HEAD_DIM   16, 32, 64, 128, 256
Z/H        1/1, 4/32, 8/64
STAGE      1, 3
```

网格采用单因素、交互组合和边界 case，不做没有信息增益的全笛卡尔积。它验证三件事：全局尺寸与
launch geometry 不应错误进入单 AIV UB；行缓冲应随 `BLOCK_M` 缩放；拓扑边界必须被识别。

### 3. 受控 kernel 结构变体

原始 candidate 文件保持不变。验证器把每个变体生成到结果目录，并记录完整源码和 SHA-256。
变体包括：

| 变体 | 结构目的 |
| --- | --- |
| `alpha_identity` | 插入代数恒等式，验证 canonicalization 后保持不变 |
| `denominator_diamond` | denominator 分裂后汇合，形成 diamond DAG |
| `parallel_reductions` | `p` 同时进入 sum/max 两条 reduction |
| `alpha_parallel_consumer` | 给 alpha 增加第三个下游 consumer |
| `cross_branch_accumulator` | denominator 同时进入 `l_i` 与 accumulator 分支 |
| `probability_branch` | dot 与 denominator 使用 `p` 的不同后继 |
| `detached_denominator` | 移除 `p -> denominator` dependency |

除恒等变体外，这些是编译结构压力样例，不是等价算子的运行时正确性测试；它们只用于检验模型
是否真正按 dependency graph 工作，以及面对未知结构时能否安全拒答。

### 4. 性能微基准

四个算子各执行 50 次预热和 500 次测量。`Prepare+Evaluate12` 包含 IR 解析和 12 个配置求值；
`Evaluate12` 复用 Prepared 模型：

| 算子 | Prepare+Evaluate12 median / p95 | Evaluate12 median / p95 |
| --- | ---: | ---: |
| Fused | 6.096 / 7.720 ms | 41.584 / 87.125 µs |
| Flash | 12.269 / 14.453 ms | 41.312 / 85.667 µs |
| HSTU | 9.819 / 11.836 ms | 40.938 / 98.875 µs |
| Unified | 11.428 / 13.418 ms | 41.292 / 101.875 µs |

全部满足 `Prepare p95 < 30 ms`、`Evaluate12 p95 < 2 ms` 的目标。

## 扩大 oracle 的详细结果

当前 `d=1..3` 域的 87/87 个配置状态都是 `measured`。

### 精确预测：22 case

| case 组 | case | 真实值与模型 `Delta_d(1..3)` |
| --- | --- | --- |
| reference / K-V tile | reference、`BN=16/32/64` | `[0,256,512] B` |
| global context | `N=256/512/2048/4096` | `[0,256,512] B` |
| row tile | `BM=16` | `[0,64,128] B` |
| row tile | `BM=32` | `[0,128,256] B` |
| row tile | `BM=128` | `[0,512,1024] B` |
| row tile | `BM=256` | `[0,1024,2048] B` |
| head width | `HD=16/32/128` | `[0,256,512] B` |
| shape interaction | `BM32-HD128` | `[0,128,256] B` |
| shape interaction | `BM128-HD32/HD128` | `[0,512,1024] B` |
| launch geometry | `Z/H=1/1`、`8/64` | `[0,256,512] B` |
| source control | `alpha_identity` | `[0,256,512] B` |
| source DAG | `probability_branch` | `[0,256,512] B` |

这些结果表明：两个 DynamicCV 行缓冲的单份总大小是 `BLOCK_M × f32` 在 AIV 二分后的物理
大小之和，因此随 `BLOCK_M` 线性缩放；它不随 `N_CTX`、`BLOCK_N`、`HEAD_DIM<=128`、batch/head
数量变化。当前参数域的有效份数是 `[1,2,3]`。

### 正确 fail-open：7 case

| case | 真实 `Delta_d(1..3)` | 拒答依据 |
| --- | --- | --- |
| `HEAD_DIM=256` | `[0,0,0] B` | sliced accumulator 的 GM view/alias 图不完整 |
| causal `STAGE=3` | `[0,4608,9216] B` | 同一父循环含两个产出候选缓冲的 mixed-core 子循环 |
| `denominator_diamond` | `[0,384,768] B` | 模型选择的跨块边界为 512 B，但可证明的候选缓冲只有 256 B |
| `parallel_reductions` | `[0,512,1024] B` | 同一个 `%p` 起点产生两个独立跨块边界 |
| `alpha_parallel_consumer` | `[0,384,768] B` | 跨块首个 consumer 的 dependency fanout 为 3 |
| `cross_branch_accumulator` | `[0,512,1024] B` | 模型选择的边界与最终候选缓冲的字节数不闭合 |
| `detached_denominator` | `[0,128,256] B` | 同一个 `%m_ij` 起点产生两个独立跨块边界 |

causal case 的真实单步 `4608 B` 可由中间产物解释为：

```text
一个 64x32xf32 dependency 经 AIV 投影后 = 4096 B
四个 64xf32 dependency 经 AIV 投影后   = 4 * 128 B
合计                                    = 4608 B
```

它说明“只统计带权依赖图选出的候选缓冲”对多阶段 loop 不充分。当前模型选择明确拒答，而不是把
四个行缓冲的 512 B 误报成完整 Delta。要扩展此支持域，需要继续模拟 `BroadcastUBOpt`、
`MergeSameSourceAxis`、`MergeSmallBlock` 与跨 main-loop state，而不是拟合 4608。

## 产物与复现

正式结果位于：

```text
outputs/dynamic_cv_large_validation_20260828/
├── shape_validation.json     汇总、环境和编译器身份
├── shape_validation.csv      87 行逐配置结果
├── cases/<case>/
│   ├── kernel_variant.py     实际编译源码
│   ├── constants.json
│   └── dynamic_<d>/
│       ├── after-plan-compute-block.mlir
│       ├── final.ttadapter.mlir
│       └── oracle-stages/    三个关键 pass 快照
└── logs/                     真实 PlanMemory 命令和诊断
```

通用执行命令：

```bash
PYTHONPATH=/path/to/built-triton-ascend/python:$PWD \
/path/to/built-python \
  experiment_operators/cost_model_demo/run_shape_validation.py \
  --worktree "$PWD" \
  --compiler /path/to/bishengir-compile \
  --triton-mlir-opt /path/to/triton-mlir-opt \
  --corpus experiment_operators/cost_model_demo/large_validation_corpus.json \
  --output-dir /path/to/output \
  --compact \
  --strict
```

成功判据是退出码 0，且最终输出：

```json
{
  "all_dynamic_values_measured": 29,
  "correct_fail_open": 7,
  "exact_predictions": 22,
  "validation_pass": 29
}
```

## 尚未覆盖

本轮扩大测试只隔离 DynamicCV 的 `d`，固定 `m=1,v=0`，并且没有把编译器版本来源清单
传入模型；它验证给定 IR 上的结构和 Delta，不证明顶层 Triton revision 或完整 driver option 已经
匹配，也不替代三轴 NPU 性能实验。后续若要
扩大模型支持域，优先补充：多阶段 loop 的后续 block-pass 模拟、`HEAD_DIM>=256` 的完整 alias/
lifetime 图、非零 `Delta_dm` 人工样例、跨 no-reuse 容量边界和多 device function 峰值切换。
