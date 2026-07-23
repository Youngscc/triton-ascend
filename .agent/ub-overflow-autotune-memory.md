# UB Overflow Model Autotune 实验记忆

更新时间：2026-07-23

## 总体目标

在 Triton-Ascend autotune 的候选编译流程中，于 BiShengIR 的
`CVPipelining` 之前接入轻量 UB Overflow Model。模型快速预测候选配置是否会在
后续 `PlanMemory` 阶段发生 UB overflow：

- `overflow`：立即把该候选标记为编译不可行并反馈给 autotune，跳过后续编译；
- `success`：继续执行真实的 `CVPipelining -> PlanMemory`；
- `blocker` 或模型异常：不信任模型结果，回退到真实编译流程。

本实验不需要 Ascend 真机，不做 kernel benchmark，也不宣称选出了运行性能最优配置。

## 第一阶段目标

验证引入轻量模型后，从候选生成到 `PlanMemory` 判定完成这一段流程的时间收益，并验证
模型判定与真实 `PlanMemory` 结果的一致性。

第一阶段统一以真实 `PlanMemory` 为终点，不包含 PlanMemory 后 lowering、`hivmc`、
NPU 二进制生成和真机运行。

## 对照流程

基线组：

```text
autotune candidate
  -> compile to pre-CVPipelining IR
  -> CVPipelining
  -> PlanMemory
  -> success / overflow
```

模型组：

```text
autotune candidate
  -> compile to pre-CVPipelining IR
  -> UB Overflow Model
       overflow -> reject candidate and report to autotune
       success  -> CVPipelining -> PlanMemory
       blocker  -> CVPipelining -> PlanMemory (fallback)
```

## 两种实验模式

1. Shadow 模式：模型只记录预测，所有候选仍运行真实 PlanMemory。用于验证准确率，不能
   用来宣称加速。
2. Prune 模式：模型判定为 `overflow` 时真正跳过后续编译。用于测量实际时间收益。

必须先完成 Shadow 模式的一致性验证，再测 Prune 模式收益。

## 模型结果协议

生产判断消费模型 JSON 顶层字段：

```text
precision
status
overflow
ub_peak_bits
required_bits
capacity_bits
selected_seed（函数级结果）
```

判定规则：

```text
precision != exact 或 status == blocker -> fallback
precision == exact 且 status == overflow -> reject
precision == exact 且 status == success  -> continue
```

不能在调用方重新实现 `ub_peak_bits > capacity_bits`，也不能仅根据进程返回码或
`overflow` 布尔值处理 blocker。

## 第一阶段指标

- 候选总数、模型预测 overflow/success/blocker 数量；
- 真实 PlanMemory overflow/success 数量；
- false positive：模型报 overflow、PlanMemory 实际成功；
- false negative：模型报 success、PlanMemory 实际 overflow；
- 模型单次和累计耗时；
- 基线组从 pre-CVPipelining 边界到 PlanMemory 的累计 wall time；
- Prune 模式相同范围的累计 wall time；
- 绝对节省时间、相对加速比；
- success 候选因额外运行模型产生的开销；
- overflow 候选因提前终止节省的时间。

## 测量约束

- 基线组和模型组使用完全相同的 kernel、shape、候选配置、候选顺序和编译选项；
- 冷缓存与热缓存分开报告，不能混合；
- 先串行编译测量模型本身收益，再按现有 autotune 默认并行策略测量端到端 wall time；
- 固定 PlanMemory seed 策略。默认应复现真实的 seed 0..19 retry，而不是只测 seed 0；
- 记录模型与真实编译使用的所有 UB 相关选项，确保 CVPipelining、multi-buffer、对齐、
  code motion 等设置一致；
- 模型进程启动、IR 序列化、JSON 解析时间均计入模型组；
- 构建时间不计入实验运行时间。

## 当前可复用实现

轻量模型仓库：

```text
/Users/YokeLove/huawei/AscendNPU-IR/ub_overflow_model_cpp
```

模型入口：

```text
ub_overflow_model_cpp/output/bin/bishengir-ub-overflow-model
```

真实后缀对照工具：

```text
build/bin/bishengir-cvpipeline-suffix-compile
```

模型输入是 `CVPipelining` 前的 Generic MLIR；模型内部模拟影响 UB 的
`CVPipelining -> PlanMemory` 流程及默认 seed retry。

## 预计接入点

AscendNPU-IR 的真实 CVPipelining 插入位置：

```text
bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp
```

Triton-Ascend autotune 候选编译和失败过滤位置：

```text
third_party/ascend/backend/runtime/autotuner.py
```

Triton-Ascend 调用 BiShengIR 编译器的位置：

```text
third_party/ascend/backend/compiler.py
```

第一版优先使用独立模型进程和结构化 JSON 接口，避免立即把模型实现链接进
`bishengir-compile`。完成收益和正确性验证后，再决定是否改为进程内调用以减少启动开销。

## 第一阶段完成标准

- autotune 能对每个候选取得准确的 pre-CVPipelining IR；
- Shadow 模式生成逐候选模型结果和真实 PlanMemory 对照记录；
- Prune 模式能让预测 overflow 的候选不进入真实 CVPipelining/PlanMemory；
- overflow、success、blocker 能以结构化状态反馈给 autotune；
- 模型错误或不完整覆盖不会导致候选被错误地当作成功；
- 输出可复现的时间、准确率和候选统计报告；
- 整个实验可在当前 macOS arm64 环境运行，无需 CANN、`hivmc` 或 Ascend 设备。

## 后续阶段（暂不实施）

第一阶段完成后，再评估把成功候选继续编译到 `hivmc` 前的
`module.hivm.opt.mlir`，用于测量更接近完整 host compile-only autotune 的时间。
