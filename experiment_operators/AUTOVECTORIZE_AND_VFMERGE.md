# AutoVectorizeV2、VF 与 VFMerge Level 1

[交互可视化](AUTOVECTORIZE_AND_VFMERGE.html)

## 1. 阅读主线与算子背景

本文沿一条 IR 变化主线展开：

```text
linalg Tensor 计算
        ↓ AutoVectorizeV2：分块、向量化、tile 内融合
scf.for + vector 计算
        ↓ OutlineVectorFunction：抽出执行边界
Vector Function（VF）
        ↓ VFMerge Level 1：合并无依赖 VF
更少的 VF 和 func.call
```

涉及的算子和作用：

| 算子或计算 | 主要步骤 | 观察重点 |
| --- | --- | --- |
| Add + Exp | 逐元素加法、指数 | AutoVectorize 如何把生产者和消费者放进同一个 tile |
| Softmax | Max、Sub、Exp、Sum、Div | VF 依赖链为什么阻止 Level 1 合并 |
| Flash Attention | `Q × K`、Online Softmax、`P × V` | Cube 与 Vector 计算共存，四个独立初始化 VF 合并 |
| HSTU Attention backward | 梯度、地址计算、内存访问、同步 | Region、同步和内存依赖怎样限制合并 |

Flash Attention 的矩阵乘主要由 Cube Core 完成；scale、mask、Reduction、
Exp、归一化和初始化主要是 Vector 计算。VFMerge 不合并整个 Attention，
只处理已经形成的 VF。

## 2. 阶段一：AutoVectorizeV2 形成 Vector 计算

### 2.1 输入与输出

`linalg.*` 描述 Tensor 计算语义，不表示已经按 Vector 执行：

```mlir
%out = linalg.elemwise_binary
    {fun = #linalg.binary_fn<add>}
    ins(%a, %b : tensor<256xf32>, tensor<256xf32>)
    outs(%init : tensor<256xf32>) -> tensor<256xf32>
```

AutoVectorizeV2 的直接结果主要有三类：

1. 用 `scf.for` 将大 Tensor 分成适合 Vector Core 的 tile；
2. 用 `vector.transfer_read/write` 搬入、写回一个 tile；
3. 让 `arith.*`、`math.*` 和 Reduction 使用 `vector<...>` 类型，并融合
   适合共享同一个 tile 的 producer/consumer。

真正显式的 Vector 加法形如：

```mlir
%va = vector.transfer_read %a_tile[%c0]
    : tensor<64xf32>, vector<64xf32>
%vb = vector.transfer_read %b_tile[%c0]
    : tensor<64xf32>, vector<64xf32>
%vc = arith.addf %va, %vb : vector<64xf32>
%written = vector.transfer_write %vc, %out_tile[%c0]
    : vector<64xf32>, tensor<64xf32>
```

AutoVectorizeV2 主要规划可向量化的 `linalg.*`，也处理少量 HFusion 特殊
操作。`linalg.generic` 内部的 `arith.*`、`math.*` 会随父操作一起变成
Vector 类型。

### 2.2 Add + Exp：完整分块循环

原始计算：

```text
tmp = a + b
out = exp(tmp)
```

假设 Tensor 有 256 个 `f32`，一个 tile 有 64 个元素：

```mlir
%result = scf.for %offset = %c0 to %c256 step %c64
    iter_args(%current_out = %out_init) -> tensor<256xf32> {

  %a_tile = tensor.extract_slice %a[%offset] [64] [1]
      : tensor<256xf32> to tensor<64xf32>
  %b_tile = tensor.extract_slice %b[%offset] [64] [1]
      : tensor<256xf32> to tensor<64xf32>

  %va = vector.transfer_read %a_tile[%c0]
      : tensor<64xf32>, vector<64xf32>
  %vb = vector.transfer_read %b_tile[%c0]
      : tensor<64xf32>, vector<64xf32>
  %vadd = arith.addf %va, %vb : vector<64xf32>
  %vexp = math.exp %vadd : vector<64xf32>

  %out_tile = tensor.extract_slice %current_out[%offset] [64] [1]
      : tensor<256xf32> to tensor<64xf32>
  %written = vector.transfer_write %vexp, %out_tile[%c0]
      : vector<64xf32>, tensor<64xf32>
  %next_out = tensor.insert_slice %written into %current_out[%offset]
      [64] [1] : tensor<64xf32> into tensor<256xf32>

  scf.yield %next_out : tensor<256xf32>
}
```

循环的四次迭代分别处理：

```text
offset=0    → [0, 64)
offset=64   → [64, 128)
offset=128  → [128, 192)
offset=192  → [192, 256)
```

Add 的 `%vadd` 直接交给 Exp。完整的 `%tmp` 不需要先写回 Tensor，再被
另一个循环读出。这是 **tile 内的 producer/consumer 融合**。

### 2.3 独立计算不会被强制放进同一个循环

两个无数据关系的初始化：

```text
A[:] = 0
B[:] = -inf
```

可以分别形成两个 Vector 分块循环：

```mlir
%new_A = scf.for %offset = %c0 to %c256 step %c64 {
  %a_tile = tensor.extract_slice %current_A[%offset] [64] [1]
  %written = vector.transfer_write %zero, %a_tile[%c0]
  %next_A = tensor.insert_slice %written into %current_A[%offset] [64] [1]
  scf.yield %next_A
}

%new_B = scf.for %offset = %c0 to %c256 step %c64 {
  %b_tile = tensor.extract_slice %current_B[%offset] [64] [1]
  %written = vector.transfer_write %neg_inf, %b_tile[%c0]
  %next_B = tensor.insert_slice %written into %current_B[%offset] [64] [1]
  scf.yield %next_B
}
```

`%a_tile`、`%b_tile` 是当前输出 Tensor 的真实切片，不是占位符。
`vector.transfer_write` 返回更新后的切片，`tensor.insert_slice` 再把它放回
完整 Tensor。

## 3. 阶段二：Outline 形成 VF

Outline 将 Vector 循环抽成函数：

```mlir
func.func @kernel_outlined_vf_0(...)
    attributes {hivm.vector_function, no_inline} {
  scf.for ... {
    ... vector.transfer_read ...
    ... arith.addf ... : vector<64xf32>
    ... vector.transfer_write ...
  }
  return
}
```

根函数只保留调用：

```mlir
%out = func.call @kernel_outlined_vf_0(...)
    {hivm.vector_function, no_inline}
```

编译器判断 VF 的直接条件是函数带 `hivm.vector_function`。VF 通常来自
Vector 循环，但函数体仍可包含 `scf.for`、`tensor.*`、`arith.*` 和
`math.*`，不要求全部是 `vector.*`。

Add + Exp 已被 AutoVectorize 放进同一个循环，Outline 后只有一个 VF，
VFMerge 没有第二个 VF 可合并。两个独立初始化可能形成两个 VF，随后才由
VFMerge 处理。

## 4. 阶段三：VFMerge Level 1 选择候选

### 4.1 Level 语义

| Level | 逻辑 |
| ---: | --- |
| 0 | 不运行 VFMerge |
| 1 | Bufferization 前，只合并彼此无依赖的 Tensor VF |
| 2 | Bufferization 后进行更宽松的 VF 合并 |

本文只讨论 Level 1。默认 `merge-vf-num-limit=4`，一个 merged VF 最多
包含四个原始 VF。

### 4.2 按调用顺序建立 `vfs`

Pass 收集 `hivm.vector_function`，按它们第一次出现在根函数中的调用顺序
排列：

```mlir
call @vf_3(...)
call @vf_1(...)
call @vf_0(...)
```

得到：

```text
vfs = [vf_3, vf_1, vf_0]
```

“相邻 VF”指 `vfs` 中相邻，不要求两条 `func.call` 在 IR 中紧挨着。

### 4.3 建立依赖闭包

Pass 合并两类依赖：

- **SSA 依赖**：使用者依赖值的定义者，也统计 Region 对外部值的捕获；
- **内存依赖**：对可能 alias 的 memref 检查 RAW、WAR、WAW，只要至少
  一侧写内存，就不能随意换序。

直接依赖：

```mlir
%x = call @vf_a(%input)
%y = call @vf_b(%x)
```

```text
vf_a → vf_b
```

传递依赖也会被识别：

```text
vf_a → op_x → op_y → vf_b
```

### 4.4 `score` 与两个列表

若后面的 `VF[j]` 依赖前面的 `VF[i]`：

```text
useScoreMat[i][j] = 1
```

一行求和得到 `score(i)`，表示有多少后续 VF 依赖它：

```text
A → B

score(A) = 1
score(B) = 0
```

`score(B)=0` 只表示没有后续 VF 使用 B，不表示 B 没有上游依赖。Level 1
只用 score 选择初始起点：

```text
vfs      ：当前仍存在的全部 VF
worklist ：哪些 VF 可以作为本轮 vf1
```

`score>0` 的 VF 没有从 `vfs` 删除，仍可成为右侧 `vf2`。

## 5. VFMerge Level 1 的贪心循环

### 5.1 核心伪代码

```text
vfs = 按调用顺序排列的全部 VF
worklist = 初始 score=0 的 VF

依次处理 worklist 中的 vf1：
  在当前 vfs 中找到 vf1
  vf2 = vf1 在 vfs 中的右邻居

  vf1/vf2 已删除，或 vf1 没有右邻居：跳过
  vf1 与 vf2 存在依赖：跳过
  两组合计超过 4 个原始 VF：跳过
  Region、同步、ExtractKind 等检查失败：跳过

  否则立即合并 vf1 + vf2
  用 merged 替换 vfs 中的 [vf1, vf2]
  更新依赖图
  把 merged 追加到 worklist，之后继续向右尝试
```

它不根据性能或 UB 搜索全局最优组合；每次只看当前右邻居，成功后不
回溯，因此是局部、顺序相关的贪心算法。

### 5.2 五个独立 VF

```text
初始：vfs = [A, B, C, D, E]

A + B → AB       vfs = [AB, C, D, E]
C + D → CD       vfs = [AB, CD, E]
AB + CD → ABCD   vfs = [ABCD, E]
ABCD + E         4 + 1 > 4，拒绝
```

处理 E 时，即使 worklist 后面是 `AB、CD`，配对仍从 `vfs` 查找。E 在
`vfs` 最右侧，没有右邻居，因此不会出现 `E + AB`。最终为：

```text
[ABCD, E]
```

### 5.3 为什么 score 过滤后仍需 `areRelated()`

```text
score       决定“谁先主动尝试”
areRelated  决定“当前这一对能不能合并”
mergeNodes  在合并后维护最新依赖图
```

每次仍要检查依赖，因为：

1. `score>0` 的 VF 仍在完整 `vfs` 中，可以作为右侧 `vf2`；
2. 合并产生的新 VF 会重新进入 worklist，其依赖已经变化，没有对应的
   初始 score。

### 5.4 合并改写

合并前：

```mlir
%a = call @vf_a(%x)
%b = call @vf_b(%y)
return %a, %b
```

新函数依次克隆两个旧函数体：

```mlir
func.func @vf_a_merged_vf_0(%x, %y) -> (tensor<...>, tensor<...>)
    attributes {hivm.vector_function} {
  ...vf_a body...
  ...vf_b body...
  return %a_result, %b_result
}
```

两个调用替换成一个：

```mlir
%r:2 = call @vf_a_merged_vf_0(%x, %y)
    {hivm.vector_function, no_inline, ptc_simdvf}
return %r#0, %r#1
```

新函数参数是所需输入的并集，返回值按“VF1 结果 + VF2 结果”拼接。旧调用
和旧 VF 定义随后删除。

## 6. Level 1 的安全边界

### 6.1 合并检查

| 条件 | 处理 |
| --- | --- |
| 两个 VF 有 SSA 或内存依赖 | 不合并 |
| 两个调用位于不同 Region | 不合并 |
| 中间存在 `hivm.sync_block_set/wait` | 不合并 |
| ExtractKind 不一致 | 不合并 |
| 多结果 VF 同时包含 extracted 与 non-extracted | 不合并 |
| anchor 操作会因移动破坏支配关系 | 不合并 |
| 中间形成 `vf1 → op → vf2` | 不合并 |
| 合并组超过四个原始 VF | 不合并 |

若中间普通操作无冲突，Pass 将它们移动到合并调用之前或之后；必要时拆成
前移和后移两组。Level 1 成功时，两个调用之间不再遗留可移动普通操作。

### 6.2 `tensor.extract` 是什么

`tensor.extract` 从 Tensor 中取一个标量：

```mlir
%t = ... : tensor<4xf32>
%x = tensor.extract %t[%i] : tensor<4xf32>
// %x : f32
```

`tensor.extract_slice` 仍返回 Tensor：

```mlir
%tile = tensor.extract_slice %t[%offset] [64] [1]
    : tensor<256xf32> to tensor<64xf32>
```

当前 ExtractKind 检查针对前者。

### 6.3 为什么不混合 Extracted 与 NotExtracted

VF 在 Vector 侧产生 Tensor，`tensor.extract` 将一个元素交给 Scalar 侧。
Scalar 读取前需要等待 Vector 结果完成，形成 V→S 同步边界。

合并前：

```mlir
%small = call @small_vf(...)       // tensor<1xf32>
%scalar = tensor.extract %small[0]
%large = call @large_vf(...)       // 独立的大 Tensor 计算
```

Scalar 只需等待 `small_vf`。如果强行合并：

```mlir
%r:2 = call @merged_vf(...)
%scalar = tensor.extract %r#0[0]
```

标量可能要等整个 merged VF，使同步范围从“小 VF”扩大到“小 VF + 大
VF”。Double buffering 依靠 ping/pong 两份 buffer 重叠相邻 tile；等待
扩大后，buffer 释放和下一 tile 启动可能被推迟。

因此只允许相同类别继续合并：

| VF1 | VF2 | 是否允许 |
| --- | --- | ---: |
| Extracted | Extracted | 是 |
| NotExtracted | NotExtracted | 是 |
| Extracted | NotExtracted | 否 |
| NotExtracted | Extracted | 否 |
| Mixed | 任意类型 | 否 |

## 7. 算子中的实际表现

### 7.1 Softmax：依赖链保留 VF 边界

```text
max_value = reduce_max(x)
shifted   = x - max_value
exp_value = exp(shifted)
sum_value = reduce_sum(exp_value)
out       = exp_value / sum_value
```

AutoVectorize 可以把 Sub + Exp 放进同一个 tile，并向量化 Reduction 和
Div。Outline 后可能形成：

```text
vf_max → vf_sub_exp → vf_sum → vf_div
```

这是严格依赖链。Level 1 不合并上下游 VF；若旁边还有独立初始化 VF，
它仍可能和其中一个无关节点合并。

### 7.2 Flash Attention：四个初始化 VF 合并

VFMerge 前：

```mlir
%7  = tensor.empty() : tensor<1x64xf32>
%8  = call @vf_3(%7)       // 写 0.5
%9  = tensor.empty() : tensor<64xf32>
%10 = call @vf_2(%9)       // 写 0
%11 = tensor.empty() : tensor<64x64xf32>
%12 = call @vf_1(%11)      // 矩阵清零
%13 = tensor.empty() : tensor<64xf32>
%14 = call @vf_0(%13)      // 写 -inf
```

四个 VF 初始化不同数据，彼此无依赖，位于同一 Region，ExtractKind 一致，
总数正好为 4。VFMerge 后：

```mlir
%7  = tensor.empty() : tensor<1x64xf32>
%8  = tensor.empty() : tensor<64xf32>
%9  = tensor.empty() : tensor<64x64xf32>
%10 = tensor.empty() : tensor<64xf32>

%11:4 = call @_attn_fwd_outlined_merged_merged_vf_0(
    %7, %8, %9, %10)
    {hivm.vector_function, no_inline, ptc_simdvf}
```

四项初始化仍然执行，只是从四个 VF、四次调用变成一个 VF、一次调用和
四个结果。`Q × K`、`P × V` 等 Cube 矩阵乘不属于这次合并。

- [VFMerge 前 IR](vfmerge_pr_cases/level1_ir/flash_attention_hstu.before.mlir)
- [VFMerge 后 IR](vfmerge_pr_cases/level1_ir/flash_attention_hstu.after.mlir)
- [差异](vfmerge_pr_cases/level1_ir/flash_attention_hstu.diff)

### 7.3 HSTU backward：正反例

| 场景 | 结果 | 原因 |
| --- | --- | --- |
| 地址计算、reinterpret、subview 位于调用之间 | 合并 | 中间操作可安全前移 |
| 中间有 `sync_block_set/wait` | 不合并 | 不跨显式同步边界 |
| 中间 `scf.if` 使用第一个 VF 结果 | 合并 | Region 可后移并改接新调用结果 |
| `affine.store → to_tensor → vf2` | 不合并 | 同一底层 memref 存在跨表示依赖 |
| 依赖通过 `scf.if` 捕获 | 不合并 | Region-aware SSA 分析识别依赖 |

### 7.4 依赖图更新：防止二次错误合并

合成用例：

```text
producer 独立

zero ──> reduce_a
     └─> reduce_b
```

调用顺序和分数：

```text
vfs = [producer, zero, reduce_a, reduce_b]

score(producer) = 0
score(zero)     = 2
score(reduce_a) = 0
score(reduce_b) = 0
```

`zero` 不在初始 worklist，但仍在 `vfs`，可以作为 `producer` 的右邻居：

```text
producer + zero → producer_zero
reduce_a + reduce_b → reductions
```

依赖图随后更新为：

```text
producer_zero → reductions
```

两个 merged VF 重新进入 worklist 后，`areRelated()` 使用最新依赖图阻止
它们继续合并。最终 IR：

```mlir
%0:2 = call @dependency_update_producer_merged_vf_0(...)
%1:2 = call @dependency_update_reduce_a_merged_vf_0(%0#1, %0#1)
```

- [用例](vfmerge_pr_cases/dependency_update_level1.mlir)
- [合并前 IR](vfmerge_pr_cases/level1_ir/dependency_update.before.mlir)
- [合并后 IR](vfmerge_pr_cases/level1_ir/dependency_update.after.mlir)

## 8. 如何从 IR 判断 pass 生效

AutoVectorizeV2：

1. 出现 `scf.for` 分块循环；
2. 出现 `vector.transfer_read/write`；
3. `arith.*`、`math.*` 使用 `vector<...>`；
4. Reduction 变成 Vector Reduction；
5. producer/consumer 进入同一个 tile 循环。

VFMerge Level 1：

1. VF 定义和调用数减少；
2. 生成 `*_merged_vf_*`；
3. 多次调用变为一次带 `ptc_simdvf` 的调用；
4. 参数和结果按原接口正确合并；
5. 中间普通操作被安全前移或后移；
6. 有依赖、同步分隔或 ExtractKind 不匹配的 VF 保持未合并。

UB 和性能是后续流水线结果，不是判断两个 pass 是否生效的条件。
