# PlanComputeBlock UB Cost Model 讲解稿

## 1. 建模背景、目标任务与整体路线

### 1.1 建模背景与目标任务

在 A5 编译参数搜索中，同一个算子需要尝试多组 DynamicCV 和 ordinary MultiBuffer 参数。参数会改变
本地缓冲的份数和 UB 占用，但如果每个组合都执行完整后端编译和 PlanMemory，搜索成本较高，也不容易
解释 UB 变化具体来自哪些 value。

本模型的任务是：从一份 `PlanComputeBlock` 输出 IR 中提取后续内存规划所需的结构信息，找出分别受
`d` 和 `m` 控制的实际本地缓冲，并快速预测所有参数组合相对基准配置的 UB 增量。

当前支持范围为：

| 参数 | 取值或固定条件 |
| --- | --- |
| DynamicCV `intra_cache_num=d` | `1,2,3` |
| ordinary MultiBuffer `multibuffer_num=m` | `1,2,3,4` |
| `vf_merge_level` | 固定为 `0` |
| DynamicCV `inter/load count` | 固定为 `1/1` |

模型负责为参数搜索提供快速、可解释的 UB Delta。

### 1.2 整体建模路线

```text
PlanComputeBlock 输出 IR + 固定 CompilerProfile
                        |
                        v
构建一次 IRGraph：记录 SSA、循环、计算块、shape 和存储关系
                        |
                        v
还原跨块与循环携带关系，并识别 GM ↔ local 边界
                        |
                        v
筛选受 d / m 控制的具体 value，确定唯一参数归属
                        |
                        v
投影单个 AIV 的物理大小，并执行 UB 对齐
                        |
                        v
生成可复用的参数化 UB 模型
                        |
                        v
查询 (d,m) ──> Delta_total(d,m)
```

`IRGraph` 只构建一次；之后的 12 个 `(d,m)` 组合都在同一模型上查询，不再重新读取 IR 或执行编译器
pass。模型以 `(d,m)=(1,1)` 为参照：

```text
Delta_total(d,m) = U(d,m) - U(1,1)
```

## 2. 静态模型是什么

静态模型采用 SSA value graph，核心节点只有 value。

一个 value 节点保存：

```text
value 标识
定义该 value 的 operation
输入 value 和使用者
shape / dtype
block_id / core_type
loop_id / loop_depth
GM、view、copy、materialize 和 allocation 来源
```

operation、计算块、循环和 function 都不是另一类核心节点，而是 value 的属性或查询索引。

例如 Fused Attention 中的一段关系为：

对应的原始 MLIR 代码如下。三条定义在原文件中并不连续，这里按照数据依赖关系摘出：

```mlir
%p = math.exp %qk_71 {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x128xf32> loc(#loc110)

%l_ij_81 = linalg.reduce ins(%p : tensor<64x128xf32>) outs(%l_ij : tensor<64xf32>) dimensions = [1]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
  (%p_87: f32 loc(callsite(#loc86 at #loc2)), %l_ij_88: f32 loc(callsite(#loc27 at #loc104))) {
    %l_ij_89 = arith.addf %p_87, %l_ij_88 : f32 loc(#loc123)
    linalg.yield %l_ij_89 : f32 loc(#loc120)
  } loc(#loc120)

%l_i_83 = arith.addf %l_i, %l_ij_81 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32> loc(#loc117)
```

将这些 MLIR operation 按照 SSA 定义和使用关系转换为 value 节点后，得到：

```text
%qk_71
  shape：64x128xf32
        |
        | 作为 math.exp 输入
        v
%p
  定义：math.exp
  block：VECTOR 11
  shape：64x128xf32
        |
        | 作为 linalg.reduce 的数据输入
        | 每次取出一个元素，对应 reduce 区域参数 %p_87
        v
%p_87
  类型：f32
        |
        |                            %l_ij
        |                              shape：64xf32
        |                                    |
        |                                    | 作为 linalg.reduce 的初始累加值
        |                                    | 对应 reduce 区域参数 %l_ij_88
        |                                    v
        |                              %l_ij_88
        |                                类型：f32
        |                                    |
        +---------------+--------------------+
                        |
                        | 作为 arith.addf 的两个输入
                        v
                  %l_ij_89
                    定义：arith.addf
                    类型：f32
                        |
                        | 由 linalg.yield 返回给 reduce
                        v
                  %l_ij_81
                    定义：linalg.reduce
                    block：VECTOR 12
                    shape：64xf32
                        |
                        |                            %l_i
                        |                              shape：64xf32
                        |                                    |
                        +---------------+--------------------+
                                        |
                                        | 作为 arith.addf 的两个输入
                                        v
                                  %l_i_83
                                    定义：arith.addf
                                    block：VECTOR 12
                                    shape：64xf32
```

Fused Attention 的静态模型规模为：

| 内容 | 数量 |
| --- | ---: |
| 核心 value | 136 |
| 定义 operation | 117 |
| 函数参数 | 15 |
| `scf.for` 循环 | 2 |
| 计算块 | 11，包含 4 个 Cube 和 7 个 Vector |
| GM 路径 | 5，包含 3 条 load 和 2 条 store |

计算块依赖图只是这张 value graph 的聚合视图：按 `block_id` 对 value 分组，再把跨组 def-use
汇总成计算块之间的边。参数筛选最终仍然要回到具体 value。

静态模型只构建一次。后续改变 `d`、`m` 时，不重新读取 MLIR，也不重新构建图。

---

## 3. DynamicCV 到底筛选什么

DynamicCV 不会复制循环内的所有 tensor。模型需要找到的是：

> 在 mixed-core 主循环中，经过计算块归属调整以后，仍然需要跨 Vector 计算块传递，并且会形成
> 本地存储的 shaped value。

### 3.1 第一层：筛选合格循环

一个循环必须同时满足：

1. 同时包含 Cube 和 Vector 计算；
2. 当前支持 `scf.for`；
3. 存在循环携带状态；
4. `iter_arg` 和 `yield` 能一一对应。

Fused Attention 有两个循环：

| 循环 | 包含的计算块 | 循环携带状态 | 结果 |
| --- | --- | ---: | --- |
| 外层循环 | CUBE 2、VECTOR 8、VECTOR 9 | 0 | 排除 |
| 内层循环 `loop 101` | CUBE 4/6、VECTOR 10/11/12/13 | 5 | 保留 |

外层循环虽然同时出现 Cube 和 Vector，但没有循环携带状态，因此不是 DynamicCV 的候选循环。

### 3.2 第二层：收集跨计算块的 shaped 依赖

模型先在整张 IRGraph 上收集跨计算块的 shaped def-use，作为完整依赖证据；真正的计算归属调整只在
上一层选出的合格循环内进行。收集一条依赖时要求：

1. producer 和 consumer 的 `block_id` 不同；
2. producer 产生静态 shape 的 tensor、vector 或 memref；
3. 排除 scalar 和 index；
4. 排除只表示分配动作的 `alloc_tensor`、`memref.alloc`、`tensor.empty`。

第 2、3 条在代码中由同一个类型检查完成：`_shape_and_dtype(producer)` 只识别维度全部已知的
tensor、vector 和 memref。动态 shape、scalar 和 index 都会返回空，因此不会进入跨计算块依赖表。

下面用真实 IR 说明每条规则具体排除了什么。

#### 例 1：定义端和使用端位于同一个计算块

Fused Attention 的 softmax 指数计算包含下面的连续关系：

```mlir
%qk_70 = linalg.broadcast ...
  {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
  -> tensor<64x128xf32>

%qk_71 = arith.subf %qk_67, %qk_70
  {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
  : tensor<64x128xf32>

%p = math.exp %qk_71
  {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
  : tensor<64x128xf32>
```

`%qk_70 -> %qk_71` 和 `%qk_71 -> %p` 都是真实的 def-use，但两端的 `block_id` 都是 11。
这些 value 没有跨越计算块边界，不需要为 DynamicCV 建立块间缓存，因此由
`producer.block_id == consumer.block_id` 排除。

#### 例 2：结果 shape 中仍含动态维度

当前四个测试算子中，没有出现“已经跨计算块、并且结果为动态 shape”的依赖边。HSTU Attention
中可以看到一种动态结果的典型 value：

```mlir
%extracted_slice = tensor.extract_slice %acc_97[0, 0] [%5, 32] [1, 1]
  {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"}
  : tensor<64x32xf32> to tensor<?x32xf32>

bufferization.materialize_in_destination %extracted_slice in writable %subview
  {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"}
  : (tensor<?x32xf32>, memref<?x32xf32, ...>) -> ()
```

`tensor<?x32xf32>` 的第一维在该 IR 中仍是 `?`，无法仅凭静态类型算出确定字节数。这个例子本身
没有跨块，因为定义端和使用端都在 block 16，所以不会影响当前结果。

如果同类 value 真正跨越计算块，模型无法给出可复核的确定字节数，因此不把它当作普通的可计算
依赖，也不将未知大小视为 0；该输入不在当前精确预测范围内。

与它相对，`tensor<64x128xf32>`、`tensor<64xf32>` 和 `memref<64x64xbf16>` 的维度都已确定，
能够直接计算大小，才可能继续成为依赖候选。

#### 例 3：跨块传递的是 scalar 或 index

Fused Attention 中，下面两组 value 确实跨越了 block 7 和 block 8：

```mlir
%NUM_BLOCKS_M = arith.constant
  {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} 16 : i32

%task_hz_idx = arith.divsi %arg15, %NUM_BLOCKS_M
  {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : i32

%V_block_ptr = arith.constant
  {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} 64 : index

%Q_block_ptr_29 = arith.muli %Q_block_ptr_28, %V_block_ptr
  {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : index
```

`%NUM_BLOCKS_M` 是单个 `i32` 标量，`%V_block_ptr` 是地址计算使用的 `index`。它们会影响控制或
地址计算，却不代表一个需要按 `d` 复制的 tensor/vector 缓冲区，所以不会计入 DynamicCV 的 UB
增量。代码上它们同样由 `_shape_and_dtype(producer) is None` 排除，而不是另有一次独立判断。

#### 例 4：value 只表示分配占位，尚未产生有效数据

Fused Attention 中的 `%qk` 是最直接的例子：

```mlir
%qk = tensor.empty()
  {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
  : tensor<64x128xf32>

%qk_70 = linalg.broadcast ins(%m_ij_69 : tensor<64xf32>)
  outs(%qk : tensor<64x128xf32>) dimensions = [1]
  {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"}
```

这条关系跨越 block 7 和 block 11，`%qk` 也具有静态 shape，但 `tensor.empty` 只创建目标 tensor
的占位对象，并没有写入 softmax 数据。真正产生内容的是 block 11 中的 `linalg.broadcast`，因此
把 `%qk` 当成一条需要缓存的生产结果会重复计算内存。模型通过
`producer.name in _NON_ALLOCATING_OPS` 排除它。同类实例还有作为输出占位的 `%acc_ptr` 和
`%m_ij_4`。

`memref.alloc` 的典型形态如下：

```mlir
%v = memref.alloc()
  {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"}
  : memref<128x64xbf16>
memref.copy %V_block_ptr_76, %v
  {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"}
%v_77 = bufferization.to_tensor %v
  {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"}
  : memref<128x64xbf16> to tensor<128x64xbf16>
```

这里 `%v` 只完成空间分配，数据由 `memref.copy` 写入，随后 `%v_77` 才作为 matmul 的输入。在当前
Fused Attention 中，这组操作始终位于同一个 Cube block，因此先被“同块”规则排除；
`memref.alloc` 名称检查则避免其他 IR 把纯分配动作误记为数据生产结果。需要注意，真实编译器遇到
不受支持的跨块 memref 依赖时可能放弃 DynamicCV 优化，因此模型不能把任意跨块 memref 都静默
视为零成本。

Fused Attention 共得到 11 条跨块 shaped 依赖：

| # | 跨块 value（producer） | consumer | 计算块路径 | 类型 | 逻辑大小 | 依赖类型 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | `%qk_18` (`linalg.fill`) | `%qk_65` (`linalg.matmul`) | CUBE 1 → CUBE 4 | `tensor<64x128xf32>` | 32768 B | 同核 |
| 2 | `%q_52` (`bufferization.to_tensor`) | `%qk_65` (`linalg.matmul`) | CUBE 2 → CUBE 4 | `tensor<64x64xbf16>` | 8192 B | 同核 |
| 3 | `%qk_65` (`linalg.matmul`) | `%qk_67` (`arith.mulf`) | CUBE 4 → VECTOR 11 | `tensor<64x128xf32>` | 32768 B | 跨核 |
| 4 | `%acc_ptr_80` (`linalg.matmul`) | `%acc_ptr_86` (`arith.addf`) | CUBE 6 → VECTOR 13 | `tensor<64x64xf32>` | 16384 B | 跨核 |
| 5 | `%m_ij_5` (`linalg.fill`) | `%m_ij_68` (`linalg.reduce`) | VECTOR 7 → VECTOR 11 | `tensor<64xf32>` | 256 B | 同核 |
| 6 | `%qk_7` (`linalg.fill`) | `%qk_67` (`arith.mulf`) | VECTOR 7 → VECTOR 11 | `tensor<64x128xf32>` | 32768 B | 同核 |
| 7 | `%l_ij` (`linalg.fill`) | `%l_ij_81` (`linalg.reduce`) | VECTOR 10 → VECTOR 12 | `tensor<64xf32>` | 256 B | 同核 |
| 8 | `%p_cast` (`arith.truncf`) | `%acc_ptr_80` (`linalg.matmul`) | VECTOR 11 → CUBE 6 | `tensor<64x128xbf16>` | 16384 B | 跨核 |
| 9 | `%m_ij_69` (`arith.maxnumf`) | `%alpha` (`arith.subf`) | VECTOR 11 → VECTOR 12 | `tensor<64xf32>` | 256 B | 同核 |
| 10 | `%p` (`math.exp`) | `%l_ij_81` (`linalg.reduce`) | VECTOR 11 → VECTOR 12 | `tensor<64x128xf32>` | 32768 B | 同核 |
| 11 | `%acc_ptr_84` (`linalg.broadcast`) | `%acc_ptr_85` (`arith.mulf`) | VECTOR 12 → VECTOR 13 | `tensor<64x64xf32>` | 16384 B | 同核 |

其中 8 条是同核跨块依赖，3 条是跨核依赖。表中“逻辑大小”由静态 shape 和
element type 直接计算。

### 3.3 第三层：还原循环携带关系

循环中的 `iter_arg` 不是一个独立的新状态，需要映射回同一循环的 yield value。

例如：

```text
上一轮状态：%arg19
本轮 yield：%m_ij_69
```

二者表示同一个跨轮行最大值状态。循环交界处旧状态和新状态可能同时存活，因此该边的权重乘 2：

```text
64xf32 = 64 × 4 B = 256 B
循环携带边权 = 256 × 2 = 512 B
```

其他边的权重为：

```text
静态 tensor/vector = product(shape) × element_bytes
memref               = INF，禁止轻易跨越
scalar/index         = 0
loop-carried value   = 对应 yield value 大小 × 2
```

### 3.4 第四层：寻找并选择更小的计算归属边界

> **对应的编译器实现：** 这一层抽象的是 DynamicCV `ComputeBlockOptPass` 内部的
> `UBUsageOptPass`（`--ub-usage-opt`），具体对应“建立带 UB 边权的依赖图、选择更小 cut、再改写
> compute block id”的 `collectNeedUbOpts()`、`collectRecordChange()` 和 `applyRecordChange()` 流程；

不是每条 Vector 跨块依赖都可以调整。模型只接受同时满足以下条件的移动：

1. 起点和目标都属于 Vector；
2. 目标计算依赖的数据只落在起点块和目标块中；
3. 沿目标块内部只能形成唯一计算链，不能出现当前无法解释的分叉；
4. 移动 operation 后的新跨块数据量必须严格小于原数据量。

Fused Attention 中只有 `%m_ij_69` 和 `%p` 两个起点进入收益比较：

| 起点 | 原始跨块边界 | 调整的计算归属 | 新跨块边界 | 结论 |
| --- | ---: | --- | ---: | --- |
| `%m_ij_69` | 循环携带的 `64xf32` 新旧状态同时存活：`256 × 2 = 512 B` | 将生成 `%alpha` 的 `arith.subf` 从 VECTOR 12 移到 VECTOR 11 | `%alpha: 64xf32 = 256 B` | `256 < 512`，接受 |
| `%p` | `%p: 64x128xf32 = 32768 B` 加 `%l_ij: 64xf32 = 256 B`，共 `33024 B` | 将生成 `%l_ij_81` 的 `linalg.reduce` 从 VECTOR 12 移到 VECTOR 11，并带上必要依赖 `%l_ij` | `%l_ij_81: 64xf32 = 256 B` | `256 < 33024`，接受 |

调整后，第一条路径不再跨块保存循环携带的行最大值，而是保存 `%alpha`；第二条路径不再跨块保存 `%p`，而是保存 reduce 结果 `%l_ij_81`。

其余典型 value 被排除或合并的原因包括：

| value | 没有单独成为最终候选的原因 |
| --- | --- |
| `%qk_65`、`%acc_ptr_80`、`%p_cast` | 属于 Cube/Vector 跨核依赖 |
| `%m_ij_5`、`%qk_7` | 定义在合格内层循环之外 |
| `%l_ij` | 是 reduce 的必要输入，会随 reduce 一起调整归属，不是最终跨块结果 |
| `%acc_ptr_84` | 沿后续计算移动后没有形成严格更小的边界 |

### 3.5 第五层：生成最终 DynamicCV 候选

调整 block 归属后重新构建 def-use，只保留：

1. producer 和 consumer 仍属于不同计算块；
2. 两端都是 Vector；
3. value 具有静态 shape；
4. 该 value 会在后续形成实际本地存储；
5. 没有被其他缓冲机制接管。

最终留下两个 value：

| value | 逻辑 shape | AIV 二分后的物理 shape | 单份物理大小 |
| --- | --- | --- | ---: |
| `%alpha` | `64xf32` | `32xf32` | 128 B |
| `%l_ij_81` | `64xf32` | `32xf32` | 128 B |

因此：

```text
DynamicCV 的单步 UB Delta
  = 128 + 128
  = 256 B
```

`d` 对应的份数规则为：

```text
n(d,m) = d
Delta_d(d) = 256 × (d - 1)
```

---

## 4. ordinary MultiBuffer 到底筛选什么

ordinary MultiBuffer 关注的是 Vector 侧 GM load/store 对应的本地 staging，不是所有 local allocation。

### 4.1 第一层：从 GM 边界操作开始

模型先搜索下面两种原始语句，并且只提取它们的直接 source、destination 和所在循环链：

```mlir
bufferization.materialize_in_destination %source in writable %destination
memref.copy %source, %destination
```

此时得到的是边界候选，还不能确定直接端点背后的 GM function argument。Fused Attention 中找到的
直接产物为：

| 原始语句 | 直接 source | 直接 destination |
| --- | --- | --- |
| `materialize_in_destination` | `%m_i_24` | `%m_ptrs_30` |
| `materialize_in_destination` | `%0` | `%O_block_ptr` |
| `memref.copy` | `%Q_block_ptr_21` | `%q` |
| `memref.copy` | `%V_block_ptr_36` | `%k` |
| `memref.copy` | `%V_block_ptr_41` | `%v` |

### 4.2 第二层：沿 view/cast 找到真实对象

第二层解析直接端点的真实含义。

GM 一侧沿 `memref.reinterpret_cast`、`memref.subview` 和 `memref.cast` 的 base memref 反向追踪
到带 `tt.tensor_kind` 的 function argument。load 的 local 一侧沿 `memref.subview/memref.cast`
追踪到 `memref.alloc`。input load 的大小由 local allocation 确定；output store 此时没有显式
local allocation，因此使用 GM destination view 的逻辑 shape 和 dtype。

十个直接 value 各自解析到：

| 直接 value | 第二层找到的结果 |
| --- | --- |
| `%m_i_24` | store 的 tensor source，留给后续来源检查 |
| `%m_ptrs_30` | 追踪到 GM output argument `%M`；同时提供 store 的逻辑类型 `64xf32` |
| `%0` | store 的 tensor source，留给后续来源检查 |
| `%O_block_ptr` | 追踪到 GM output argument `%Out`；同时提供 store 的逻辑类型 `64x64xbf16` |
| `%Q_block_ptr_21` | 追踪到 GM input argument `%Q` |
| `%q` | 追踪到 load 的 local `memref.alloc`；由其类型确定 `64x64xbf16` |
| `%V_block_ptr_36` | 追踪到 GM input argument `%K` |
| `%k` | 追踪到 load 的 local `memref.alloc`；由其类型确定 `128x64xbf16` |
| `%V_block_ptr_41` | 追踪到 GM input argument `%V` |
| `%v` | 追踪到 load 的 local `memref.alloc`；由其类型确定 `128x64xbf16` |

因此，只有完成第二层后才能把五条候选总结为 2 条输出 store（`%M`、`%Out`）和 3 条输入 load
（`%Q`、`%K`、`%V`）。例如 `%V_block_ptr_36` 的名字虽然带 `V`，其
`memref.reinterpret_cast` 的 base memref 实际是 `%K`。

### 4.3 第三层：检查循环和核类型

要求：

1. 路径处于受支持的 `scf.for` 或 `scf.while` 循环链；
2. output store 必须属于 Vector；
3. input load 必须能确定属于 Vector 或 Cube。

Fused Attention 的 Q/K/V 本地 allocation 都属于 Cube，因此：

```text
Q/K/V load → 排除，不属于 Vector UB ordinary MultiBuffer
```

`%Out`、`%M` 的 store 属于 Vector，因此继续筛选。

### 4.4 第四层：按照优先级确定唯一归属

同一个实际 allocation 只能由一个机制控制，优先级为：

```text
non-UB / Cube
  → Fixpipe / tightly-coupled
  → DynamicCV
  → fixed GM-load
  → preload / explicit mark
  → ordinary MultiBuffer
```

需要排除：

1. Cube 或非 UB 缓冲；
2. source 能沿透明 view/cast 追踪到 `ssbuffer.add_from_matmul` 的 Fixpipe store；
3. 已被 DynamicCV 控制的存储；
4. 已被固定 GM-load count 预标记的 load；
5. 已被 preload 或显式 marker 接管的缓冲。

Fused Attention 的 O、M 不属于以上类型，因此最终归 ordinary MultiBuffer。

### 4.5 第五层：物理 shape 和 UB 对齐

当前 AIV 子块化因子为 2：

```text
O 逻辑 shape = 64x64xbf16
O 物理 shape = 32x64xbf16
O 单份大小   = 32 × 64 × 2 B = 4096 B

M 逻辑 shape = 64xf32
M 物理 shape = 32xf32
M 单份大小   = 32 × 4 B = 128 B
```

两者都已经满足 32 B UB 对齐。

因此：

```text
ordinary MultiBuffer 的单步 UB Delta
  = 4096 + 128
  = 4224 B
```

`m` 对应的份数规则为：

```text
n(d,m) = m
Delta_m(m) = 4224 × (m - 1)
```

---

## 5. 四个 value 如何得到最终结果

Fused Attention 最终只需要四条贡献记录：

| 作用 value | 控制参数 | 单份物理大小 | 份数规则 | 相对 `(1,1)` 的贡献 |
| --- | --- | ---: | --- | ---: |
| `%alpha` | `d` | 128 B | `n=d` | `128×(d-1)` |
| `%l_ij_81` | `d` | 128 B | `n=d` | `128×(d-1)` |
| 输出 O staging | `m` | 4096 B | `n=m` | `4096×(m-1)` |
| 行统计量 M staging | `m` | 128 B | `n=m` | `128×(m-1)` |

当前两个参数作用在不同存储上，而且不会互相改变 shape、生命周期或地址复用关系，因此：

```text
Delta_dm(d,m) = 0
```

最终公式为：

```text
Delta_total(d,m)
  = 128(d-1) + 128(d-1) + 4096(m-1) + 128(m-1)
  = 256(d-1) + 4224(m-1)
```

全部查询结果如下，单位为字节：

| `d \ m` | 1 | 2 | 3 | 4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 4224 | 8448 | 12672 |
| 2 | 256 | 4480 | 8704 | 12928 |
| 3 | 512 | 4736 | 8960 | 13184 |

以 `(d,m)=(3,4)` 为例：

```text
Delta_d     = 256 × (3-1)  =   512 B
Delta_m     = 4224 × (4-1) = 12672 B
Delta_dm    = 0 B
Delta_total = 13184 B
```

真实 `(1,1)` 参照 UB 为 `63104 B`：

```text
U_calibrated(3,4)
  = 63104 + 13184
  = 76288 B
  = 74.000 KiB
```

该结果与真实 PlanMemory 数据逐字节一致。

---

## 6. 当前准确率与耗时

### 6.1 四个真实算子

当前四个真实算子的结果为：

| 算子 | `d=1,2,3` 的 Delta | `m` 每增加 1 的 Delta | 联合 Delta |
| --- | --- | ---: | ---: |
| Fused Attention | `[0,256,512] B` | 4224 B | 0 B |
| Flash Attention | `[0,256,512] B` | 4224 B | 0 B |
| HSTU Attention | `[0,0,0] B` | 0 B | 0 B |
| Unified Attention | `[0,64,128] B` | 2048 B | 0 B |

总计：

```text
48 / 48 个 UB Delta 与真实数据逐字节一致
```

### 6.2 扩展结构样例

在包含 shape、局部 tile 和依赖拓扑变化的 87 个真实编译样例中：

```text
精确预测         = 66
未给出精确数值   = 21
输出错误数值     = 0
总体精确预测率   = 66 / 87 = 75.9%
```

### 6.3 Python 原型耗时

| 算子 | 构建模型并评估全部参数，中位数 | 仅评估全部 12 个参数组合，中位数 |
| --- | ---: | ---: |
| Fused Attention | 6.096 ms | 41.584 μs |
| Flash Attention | 12.269 ms | 41.312 μs |
| HSTU Attention | 9.819 ms | 40.938 μs |
| Unified Attention | 11.428 ms | 41.292 μs |

正式 C++ 版本可以直接访问 `ModuleOp` 和 `Value::getUsers()`，不需要 Python 原型中的 MLIR 文本
索引，预计构建时间还会进一步下降。
