# PlanComputeBlock UB Cost Model 设计报告

## 1. 问题与目标

本项目面向基于 UB 建模的编译路径全局规划：在不执行完整后端编译和 PlanMemory 的情况下，
根据 `PlanComputeBlock` 输出 IR，快速预测编译参数变化引起的 UB 使用量变化。

当前建模两个参数：

| 参数 | 当前取值 | 编译作用 |
| --- | --- | --- |
| DynamicCV `intra_cache_num=d` | `1,2,3` | 控制 DynamicCV 循环内缓冲份数 |
| ordinary MultiBuffer `multibuffer_num=m` | `1,2,3,4` | 控制普通本地缓冲份数 |

`vf_merge_level` 固定为 `0`，DynamicCV 的 `inter_cache_num` 和 `load_cache_num` 固定为 `1`。当前
不建模 DynamicCV off、ordinary MultiBuffer off、`vf_merge_level=1/2` 或 HFusion AutoSchedule。

模型以两个参数都取 1 时的编译状态作为参照，输出：

```text
Delta_d(d)       = U(d,1) - U(1,1)
Delta_m(m)       = U(1,m) - U(1,1)
Delta_dm(d,m)    = U(d,m) - U(d,1) - U(1,m) + U(1,1)
Delta_total(d,m) = U(d,m) - U(1,1)
                   = Delta_d + Delta_m + Delta_dm
```

如果调用方另外提供身份完整的真实 `U(1,1)` PlanMemory 结果，还可以得到：

```text
U_calibrated(d,m) = U(1,1)_real + Delta_total(d,m)
```

这属于“一次真实参照值 + 模型预测增量”，不是独立的绝对 UB 峰值预测。当前模型也不直接决定
overflow，不允许据此跳过真实编译。

## 2. 总体结构

报告只把模型分成两个部分：

```text
第一部分：IRGraph 构建

  PlanComputeBlock IR
  + CompilerProfile
  + 固定编译上下文
          |
          v
  IRGraph（静态 value graph）


第二部分：受参数影响的 value 筛选与 Delta 计算

  IRGraph
  + 查询参数 (d,m)
          |
          v
  DynamicCV 作用 value
  + ordinary MultiBuffer 作用 value
          |
          v
  逐 value 计算 Delta
```

第一部分只执行一次。第二部分可以在同一份 `IRGraph` 上评估全部参数组合，不再读取原始 MLIR，
不再调用编译器 pass，也不使用真机结果反向修正预测值。

### 2.1 除 IR 以外的输入

整个 cost model 的输入分为静态构建输入和查询输入：

| 输入 | 用途 | 使用位置 |
| --- | --- | --- |
| `PlanComputeBlock ModuleOp` | 提供 operation、SSA value、循环、计算块、shape、dtype 和 GM 边界 | 构建 `IRGraph` |
| `CompilerProfile` | 提供架构、AIV 子块化、UB 对齐和固定编译选项 | 构建 `IRGraph` |
| `vf_merge_level=0` | 固定会影响缓冲语义的编译模式 | 构建 `IRGraph` |
| 可选编译器版本信息 | 验证 IR 与编译规则是否来自匹配版本 | 构建 `IRGraph` |
| `(d,m)` | 指定本次要评估的两个参数值 | Delta 计算 |
| 可选真实 `U(1,1)` | 将 Delta 转换为校准后的 UB 数值 | Delta 计算后 |

当前 A5 profile 中最重要的固定信息是：

```text
目标架构                     = Ascend 950/A5
AIV 子块化因子               = 2
UB 对齐                      = 32 B
DynamicCV inter/load count   = 1/1
vf_merge_level               = 0
```

算子名称、Python kernel 源码、benchmark 延迟、autotune 历史、最终二进制和真机结果表都不是
数值预测输入。它们只用于定位、验证和展示。

## 3. 第一部分：IRGraph 构建

本报告中的模型就是 `IRGraph`，即从 PlanComputeBlock IR 建立的一张静态 value 依赖图。这里的
“静态”表示 value 之间的依赖，以及各 value 对应的 operation、计算块、循环、shape 和存储关系
已经固定，不表示 `d`、`m` 已经取某个具体值。

`IRGraph` 不预先决定某个 value 一定受 `d` 或 `m` 控制，而是保存后续筛选所需的全部事实。这样
DynamicCV 和 ordinary MultiBuffer 都在同一张模型上筛选，不需要分别重建 IR 索引。

### 3.1 图节点只有 Value

核心模型采用 SSA value graph，节点只有 value：

```text
ValueNode {
  value 标识
  定义该 value 的 operation
  该 operation 使用的输入 value
  使用该 value 的 operation/result value
  shape / dtype / 逻辑字节数
  block_id / core_type
  loop_id / loop_depth
  function / UB 地址域
  GM、view、copy、materialize、allocation 来源信息
}
```

节点包括：

- function argument；
- operation 产生的一个或多个 SSA result；
- 循环的 iter_arg 和 yield 对应 value。

operation 本身不是独立节点，而是定义 result value 的属性；计算块、循环、function 和 UB 地址域
也不是节点，而是 value 的归属属性和查询索引。没有 SSA result 的 store/copy/materialize 操作，
作为相关 value 的终端使用记录保存在图中。

### 3.2 Value 之间的边

如果 operation `op_b` 使用 value `%a` 并产生 value `%b`，核心图建立：

```text
%a  ->  %b
```

边表示 def-use，并保存 operand 位置、使用 operation、逻辑字节数和关系类型。循环携带、view、copy
等语义仍然表现为 value 之间的特殊边或附加属性：

| 关系 | 图中表示 |
| --- | --- |
| 普通 def-use | 输入 value 指向 operation 的 result value |
| 循环携带 | yield value 与下一轮 iter_arg 建立对应边 |
| view 与别名 | 源 value 指向 cast/reinterpret/subview 的 result value |
| copy 与 materialize | 源 value 指向本地或 GM 目标路径记录 |
| 生命周期与冲突 | 根据 value 的定义位置和最后使用位置形成区间及冲突属性 |

计算块依赖图不是另一份核心模型，而是 value graph 的聚合视图：先按 `block_id` 对 value 分组，再把
跨组的 value 边汇总为计算块之间的边。因此计算块依赖图中以 block 为节点，只是为了观察跨块
数据量；所有筛选最终仍要回到具体 value。

同一 value graph 可以展示成三种视图：

1. **IR 数据流图**：以 value 为节点，节点中显示 defining operation，边表示 def-use；
2. **计算块依赖图**：按 `block_id` 聚合 value，以计算块为展示节点、跨块 value 为边；
3. **UB 缓冲视图**：只显示筛选后的 value，并标注物理大小和参数控制规则。

三种视图共享稳定的 value id 和对应的 block id。例如计算块依赖图中的 `%alpha` 边，
能够回到 IR 数据流图中的 `%alpha` 定义，也能够在筛选后对应到一个 128 B 的 DynamicCV 缓冲。
它们只是同一模型的不同观察方式，不是三套独立数据。

### 3.3 从逻辑数据到物理缓冲所需的信息

IR 中的 tensor shape 是逻辑 shape，而 UB 规划针对单个 device function 中的物理缓冲。因此静态
模型还要保存物理化规则：

```text
逻辑 shape
  -> AIV 子块投影
  -> element bit width
  -> 原始字节数
  -> UB alignment
  -> 对齐后单份字节数
```

例如 `64xf32` 在当前 profile 下由两个 AIV function 分担：

```text
逻辑数据        = 64 x f32 = 256 B
单 AIV 物理数据 = 32 x f32 = 128 B
UB 对齐后       = 128 B
```

`IRGraph` 可以保存已经算出的物理大小，也可以保存可重复计算的 shape 表达式和 profile 规则；无论
采用哪种形式，后续筛选和 Delta 计算都不能重新从原始 IR 文本猜测大小。

### 3.4 模型完整性

模型分别记录以下信息是否完整：

```text
IR 结构和 def-use 是否完整
循环携带关系是否完整
计算块归属是否完整
GM 路径和别名关系是否完整
物理 shape 与对齐规则是否完整
生命周期与地址复用关系是否完整
编译器版本来源是否验证
```

缺少结构、候选来源或物理大小时，不能输出 Delta。生命周期关系不完整时，仍可在经过验证的线性
支持域中输出参数增量，但不能将所有缓冲大小的简单求和解释成一般的 PlanMemory 峰值。

多个 function 的 UB 地址空间分别规划。模型应先在各自地址域内计算，只有编译器明确要求同时
共存时才相加，否则绝对峰值取各地址域结果的最大值。当前原型只支持一个 `func.func`。

### 3.5 IRGraph 的使用约束

为了保证后续所有测算都来自同一个模型，必须满足：

1. 模型构建完成后，参数筛选不得重新解析 MLIR；
2. 每个被选中的 value 必须能回溯到定义、使用、计算块和物理大小；
3. 每项 Delta 必须能分解为“value、单份大小、份数变化”三部分；
4. 任何汇总表都只能是 `IRGraph` 的缓存，删除后必须能重新生成；
5. 不得根据算子名称查询经验值；
6. 两个参数存在别名、shape 或生命周期联动而又无法说明时，返回 `unknown`；
7. 每个输出都保留逐 value 贡献，确保总数可以人工复核。

### 3.6 当前原型与目标模型的差距

当前 Python 原型已经保证求值时不重新读取 IR，并在 `PreparedCostModel` 中保存计算块归属调整、
DynamicCV 缓冲、ordinary 缓冲、排除缓冲、参数表和不支持原因。

不过，当前 `PreparedCostModel` 保存的是可追溯的缓冲摘要，没有持有完整 value graph 及其
计算块/循环索引。后续工程实现需要让它持有完整 `IRGraph` 或对应的只读引用，并确保所有参数表
都能从 value 节点和边重新生成。本报告先确定这一目标结构，不修改现有代码。

另一个已确认差距是动态 shape 依赖的表示。当前原型从整条 operation 文本中搜索静态 shaped type，
可能把输入类型误当成结果类型；找不到静态 shape 时又会直接删除依赖边。编译器的
`UBUsageOptPass` 则读取 SSA value 自身的结果类型：静态 tensor/vector 使用实际字节数，动态
tensor 和 memref 保留依赖并赋予 `MAX_EDGE_SIZE`，scalar/index 的边权为 0。因此目标模型必须
显式保存结果类型和“大小未知但依赖存在”的状态，不能用删边表示动态 shape。

## 4. 第二部分：受参数影响的 `value` 筛选与 Delta 计算

第二部分只接收 `IRGraph` 和参数查询。它先依据编译器语义筛选受影响的 value，再把每个 value 转换
成统一的缓冲贡献记录，最后代入 `d`、`m` 计算 Delta。

筛选结果可以在同一 `IRGraph` 上缓存，因为当前支持域中 `d`、`m` 的具体数值只改变缓冲份数，不
改变哪些 value 被选中。全部参数组合复用同一组筛选结果。

### 4.1 筛选受 DynamicCV 影响的 `value`

DynamicCV 参数并不会复制循环内的所有 tensor。真正受 `d` 控制的是 mixed-core 主循环中，经过
编译器计算块归属调整后，仍然需要跨 Vector 计算块传递并形成本地存储的 shaped value。

筛选过程是：

1. 在每个 mixed-core 循环内收集跨计算块的 shaped def-use；
2. 把 iter_arg 映射回同一循环的 yield value，还原跨轮依赖；
3. 给依赖边计算逻辑字节数；
4. 比较计算移动前后的跨块数据量；
5. 只有新边界更小时，采用新的 operation 计算块归属；
6. 在调整后的图中保留跨 Vector 计算块的 shaped value；
7. 排除 `AddMultiBufferInnerScope` 不会分配本地存储的对象；
8. 投影为单 AIV 物理大小并建立受 `d` 控制的贡献记录。

边权与编译器语义一致：

```text
static tensor/vector = product(shape) * element_bytes
memref               = INF
scalar/index         = 0
loop-carried value   = 对应 yield value 的边权 * 2
```

循环携带 value 乘 2，是因为旧状态与本轮新状态在循环交界处可能同时存活。计算块归属调整记录
必须保存：

```text
起点 value
调整前跨块 value 和总字节数
调整后跨块 value 和总字节数
被移动的 operation
调整前后的 block id
```

一个 DynamicCV value 的单份物理大小为 `S` 时，其贡献为：

```text
Delta_value(d) = (d - 1) * S
```

### 4.2 筛选受 ordinary MultiBuffer 影响的 `value`

ordinary MultiBuffer 关注 Vector 侧 GM load/store 对应的本地 staging。筛选过程是：

1. 从 `materialize_in_destination` 识别 Vector local 到 GM output 的路径；
2. 从 `memref.copy` 识别 GM input 到 local allocation 的路径；
3. 沿 view/cast 链找到实际 allocation 和 GM argument；
4. 判断 allocation 是否处于受支持循环；
5. 依据控制优先级判断它是否真正归普通 MultiBuffer 管理；
6. 投影单 AIV 物理 shape 并计算对齐后大小；
7. 建立受 `m` 控制的贡献记录。

参数控制优先级是：

```text
non-UB / Cube
  -> Fixpipe / tightly-coupled
  -> DynamicCV
  -> fixed GM-load
  -> preload / explicit mark
  -> ordinary MarkMultiBuffer
```

这保证同一个实际 allocation 只有一个参数控制来源。已经被 DynamicCV、fixed GM-load、preload、
Fixpipe 或显式 marker 接管的缓冲不能再次按 ordinary MultiBuffer 计算。

一个普通 MultiBuffer value 的单份物理大小为 `S` 时，其贡献为：

```text
Delta_value(m) = (m - 1) * S
```

### 4.3 统一贡献记录

两类筛选最终生成相同格式的记录：

```text
受参数影响的 value {
  value 标识
  实际 allocation / GM 来源
  生产计算块 / 消费计算块
  function / 循环 / 核类型 / 存储空间
  逻辑 shape / 物理 shape / dtype
  原始字节数 / 对齐后单份字节数
  别名组 / 生命周期
  控制参数：d、m 或固定值
  份数规则 n(d,m)
  选中证据 / 排除原因
}
```

fixed 和 excluded 对象也保留在分析结果中。它们对当前 Delta 的贡献是 0，但用于解释某个 GM
边界或循环 value 为什么没有被遗漏。

### 4.4 动态 Delta 算法

设筛选出的 value `b` 对应单份物理大小 `S_b`，份数规则为 `n_b(d,m)`：

```text
Delta_b(d,m) = S_b * (n_b(d,m) - n_b(1,1))
Delta_total  = sum_b Delta_b(d,m) + X(d,m)
```

`X(d,m)` 表示两个参数共同改变别名、shape、生命周期或地址复用时的联合修正。当前支持域要求：

1. DynamicCV 和 ordinary MultiBuffer 作用 value 不指向同一存储；
2. ordinary value 的集合和单份大小不随 `d` 改变；
3. `d,m` 不改变已选 value 的 shape、生命周期和地址复用顺序；
4. 参数增加的缓冲份数会完整反映到当前 PlanMemory 统计口径。

因此当前：

```text
DynamicCV value: n_b(d,m) = d
ordinary value:  n_b(d,m) = m
fixed value:     n_b(d,m) = constant
X(d,m)                      = 0

Delta_d(d)       = sum(dynamic b)  S_b * (d - 1)
Delta_m(m)       = sum(ordinary b) S_b * (m - 1)
Delta_dm(d,m)    = 0
Delta_total      = Delta_d + Delta_m
```

求值过程只遍历筛选结果。例如：

```text
value            单份大小   参照份数   查询份数   Delta
dynamic.alpha      128 B       1          d       128*(d-1)
ordinary.O        4096 B       1          m      4096*(m-1)
```

如果某个 value 缺少物理大小、参数归属或必要的联合关系，模型返回 `unknown`，不使用默认值补齐。

## 5. Fused Attention 完整示例

本节用 Fused Attention 的 `_attn_fwd` 说明两部分如何衔接：先建立完整 `IRGraph`，再从图中筛选
受 `d`、`m` 影响的 value。

### 5.1 构建 IRGraph

输入 IR 的主要结构为：

```text
function                       = _attn_fwd
带 SSA result 的 operation 记录 = 117
compute block count            = 11
Cube block count               = 4
Vector block count             = 7
target                         = Ascend950PR_9579
inter/load count               = 1/1
vf_merge_level                 = 0
```

IR 中有两个 `scf.for`：

| 循环 | 计算块 | iter_arg | 是否包含 DynamicCV 候选结构 |
| --- | --- | ---: | --- |
| 外层循环 | CUBE 2、VECTOR 8、VECTOR 9 | 0 | 否；没有循环携带状态 |
| 内层主循环 | CUBE 4/6、VECTOR 10/11/12/13 | 5 | 是；包含 Cube/Vector 计算和在线状态 |

`IRGraph` 记录 value 节点、def-use、循环属性以及 11 条 shaped 跨块依赖：

| 跨块 value | producer -> consumer | 核类型 | 逻辑大小 |
| --- | --- | --- | ---: |
| `%qk_18` | CUBE 1 -> CUBE 4 | 同核 | 32768 B |
| `%q_52` | CUBE 2 -> CUBE 4 | 同核 | 8192 B |
| `%qk_65` | CUBE 4 -> VECTOR 11 | 跨核 | 32768 B |
| `%acc_ptr_80` | CUBE 6 -> VECTOR 13 | 跨核 | 16384 B |
| `%m_ij_5` | VECTOR 7 -> VECTOR 11 | 同核 | 256 B |
| `%qk_7` | VECTOR 7 -> VECTOR 11 | 同核 | 32768 B |
| `%l_ij` | VECTOR 10 -> VECTOR 12 | 同核 | 256 B |
| `%p_cast` | VECTOR 11 -> CUBE 6 | 跨核 | 16384 B |
| `%m_ij_69` | VECTOR 11 -> VECTOR 12 | 同核 | 256 B |
| `%p` | VECTOR 11 -> VECTOR 12 | 同核 | 32768 B |
| `%acc_ptr_84` | VECTOR 12 -> VECTOR 13 | 同核 | 16384 B |

这些只是模型中的事实，不表示 11 个 value 都会被复制。

内层循环的五个循环携带关系是：

```text
%arg17 -> %acc_ptr_86
%arg18 -> %l_i_83
%arg19 -> %m_ij_69
%arg20 -> %K_block_ptr_66
%arg21 -> %V_block_ptr_57
```

例如 `%arg19` 是上一轮行最大值，`%m_ij_69` 是本轮更新后的行最大值。`IRGraph` 通过这条关系
知道它们属于同一跨轮状态。

### 5.2 筛选受 d 影响的 `value`

#### 行最大值更新路径

相关计算是：

```text
VECTOR 11: %m_ij_69 = 更新后的行最大值
VECTOR 12: %alpha    = %arg19 - %m_ij_69
VECTOR 12: %alpha_82 = exp(%alpha)
```

调整前，进入 `%alpha` 的循环携带边界为：

```text
64 x f32 x 2 = 512 B
```

如果把 `%alpha` 的减法移动到 VECTOR 11，新的跨块 value 是：

```text
%alpha: VECTOR 11 -> VECTOR 12
64 x f32 = 256 B
```

因为 `256 B < 512 B`，模型采用新的计算块归属：

```text
%alpha: block 12 -> block 11
```

调整后由 `%alpha` 跨块保存，而不是由原始 `%m_ij_69` 继续跨块。

#### 行求和路径

相关计算是：

```text
VECTOR 11: %p       = exp(score)             32768 B
VECTOR 10: %l_ij    = row-sum initial value    256 B
VECTOR 12: %l_ij_81 = reduce_sum(%p, %l_ij)
VECTOR 12: %l_i_83  = old_L * alpha + %l_ij_81
```

调整前进入 reduce 的跨块数据量为：

```text
%p     32768 B
%l_ij    256 B
合计   33024 B
```

把 reduce 移到 VECTOR 11 后，只需跨块保存结果：

```text
%l_ij_81: VECTOR 11 -> VECTOR 12
64 x f32 = 256 B
```

因为 `256 B < 33024 B`，模型记录：

```text
%l_ij_81: block 12 -> block 11
%l_ij:    block 10 -> block 11
```

因此 32 KiB 的 `%p` 不会被选为 DynamicCV 缓冲，最终选中的是两个 `64xf32` value：

| 起点 | 调整前边界 | 最终作用 value | 逻辑 shape | 逻辑大小 |
| --- | ---: | --- | --- | ---: |
| `%m_ij_69` | 512 B | `%alpha` | `64xf32` | 256 B |
| `%p` | 33024 B | `%l_ij_81` | `64xf32` | 256 B |

AIV 子块化因子为 2，两个 value 在单 AIV function 中均为：

```text
32 x f32 = 128 B
```

所以受 `d` 控制的 value 是：

| value | 单份物理大小 | 份数规则 |
| --- | ---: | --- |
| `%alpha` | 128 B | `n(d,m)=d` |
| `%l_ij_81` | 128 B | `n(d,m)=d` |

### 5.3 筛选受 m 影响的 `value`

从 `IRGraph` 中记录的 GM store 路径找到：

```text
%O_block_ptr = reinterpret_cast %Out -> memref<64x64xbf16>
%m_ptrs_35   = reinterpret_cast %M   -> memref<64xf32>
```

经过 AIV 二分和 UB 对齐后：

| value | 逻辑 shape | 单 AIV 物理 shape | 单份大小 | 份数规则 |
| --- | --- | --- | ---: | --- |
| 输出 `O` | `64x64xbf16` | `32x64xbf16` | 4096 B | `n(d,m)=m` |
| 行统计量 `M` | `64xf32` | `32xf32` | 128 B | `n(d,m)=m` |

Q/K/V 的本地搬运位于 Cube 侧，不属于普通 Vector MultiBuffer；`O`、`M` 也没有与 `%alpha`、
`%l_ij_81` 指向同一存储。因此 Fused Attention 的筛选结果是：

```text
受 d 控制的单份大小合计 = 128 + 128 = 256 B
受 m 控制的单份大小合计 = 4096 + 128 = 4224 B
两类 value 重合          = 无
```

### 5.4 从筛选结果计算 Delta

四个 value 的贡献为：

```text
Delta_alpha(d,m)   =  128 * (d - 1)
Delta_row_sum(d,m) =  128 * (d - 1)
Delta_O(d,m)       = 4096 * (m - 1)
Delta_M(d,m)       =  128 * (m - 1)

Delta_total(d,m)
  = 128*(d-1) + 128*(d-1) + 4096*(m-1) + 128*(m-1)
  = 256*(d-1) + 4224*(m-1)
```

全部参数查询结果为：

| d \ m | 1 | 2 | 3 | 4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 B | 4224 B | 8448 B | 12672 B |
| 2 | 256 B | 4480 B | 8704 B | 12928 B |
| 3 | 512 B | 4736 B | 8960 B | 13184 B |

以 `(d,m)=(3,4)` 为例：

```text
Delta_d       = 512 B
Delta_m       = 12672 B
Delta_dm      = 0 B
Delta_total   = 13184 B
```

真实参照 UB 为 `63104 B` 时：

```text
U_calibrated(3,4) = 63104 + 13184
                  = 76288 B
                  = 74.000 KiB
```

该结果与真实 PlanMemory 数据逐字节一致。

## 6. 其他真实算子结果

### 6.1 Flash Attention

Flash Attention 的 `IRGraph` 包含 265 个带 SSA result 的 operation 记录、16 个计算块分组和 18 条
跨块依赖。筛选得到：

| 参数 | 作用 value | 单份物理大小 |
| --- | --- | ---: |
| `d` | `%alpha` | 128 B |
| `d` | `%p_sum_202` | 128 B |
| `m` | 输出 `O` | 4096 B |
| `m` | 行统计量 `M` | 128 B |

因此：

```text
Delta_total(d,m) = 256*(d-1) + 4224*(m-1)
```

真实参照 UB 为 `46592 B` 时，`(d,m)=(3,4)` 得到：

```text
U_calibrated = 46592 + 13184 = 59776 B
```

### 6.2 四个算子的汇总

| 算子 | `d=1,2,3` 对应的 DynamicCV Delta | m 每增加 1 的 Delta | 两参数联合 Delta |
| --- | --- | ---: | ---: |
| Fused Attention | `[0,256,512] B` | 4224 B | 0 B |
| Flash Attention | `[0,256,512] B` | 4224 B | 0 B |
| HSTU Attention | `[0,0,0] B` | 0 B | 0 B |
| Unified Attention | `[0,64,128] B` | 2048 B | 0 B |

HSTU 的零结果来自模型没有筛选出受当前两个参数控制的 shaped value，不是把未知情况当成 0。

## 7. 支持范围

以下情况不能给出精确 Delta：

| 情况 | 原因 |
| --- | --- |
| 动态 shape 无法静态定界 | 无法计算物理字节数 |
| mixed-core `scf.while` 或 iter/yield 不匹配 | 循环依赖不完整 |
| 同一起点产生多个独立跨块边界 | 同源结果合并尚未建模 |
| 消费端分叉超出当前限制 | 后续 pass 可能改变跨块边界 |
| 同一父循环包含多个有效 mixed-core 子循环 | 存在跨循环耦合 |
| DynamicCV 与 ordinary value 指向同一存储 | 不能按独立加法处理 |
| `d` 改变 ordinary value 的 shape 或生命周期 | 当前联合规则不适用 |
| 多 device function | 当前原型只支持一个 `func.func` |
| `vf_merge_level!=0` 或 HFusion AutoSchedule 开启 | 可能改变 value 集合或物理 shape |
| 真实参照值缺少 IR/profile/版本身份 | 不能形成可信的校准结果 |

不满足支持条件时返回 `unknown` 并继续真实编译，不能用 0 或经验值代替。

## 8. 接口与 Autotune 使用方式

```python
model = UbCostModel()

# 构建一次 IRGraph 和可复用的筛选结果。
prepared = model.prepare(
    plan_compute_block_module,
    vf_merge_level=0,
    profile=profile,
    observed_provenance=provenance,
)

# 每次查询只传参数和可选参照值，不再传 IR。
result = model.evaluate(
    prepared,
    intra_cache_num=3,
    multibuffer_num=4,
    baseline=baseline11,
)
```

autotune 中的调用顺序为：

```text
同一份 normalized PlanComputeBlock IR
  -> 构建 IRGraph 一次
  -> 筛选参数作用 value 一次并缓存
  -> 对全部参数组合执行轻量 Delta 求值
  -> safe / unknown 都继续真实编译
  -> 保留原有 fallback、正确性检查和 benchmark
```

缓存身份由以下信息组成：

```text
normalized IR hash
+ vf_merge_level
+ profile fingerprint
+ model fingerprint
+ compiler revision hash
```

模型不改变 autotune 的候选数量、执行顺序或最终性能选择逻辑。

## 9. 性能与准确率

`IRGraph` 只构建一次，之后的参数查询只遍历少量被选中的 value。

当前 Python 原型的测量结果为：

| 算子 | 构建模型并评估全部参数中位数 / p95 | 仅评估全部参数中位数 / p95 |
| --- | ---: | ---: |
| Fused Attention | 6.096 / 7.720 ms | 41.584 / 87.125 us |
| Flash Attention | 12.269 / 14.453 ms | 41.312 / 85.667 us |
| HSTU Attention | 9.819 / 11.836 ms | 40.938 / 98.875 us |
| Unified Attention | 11.428 / 13.418 ms | 41.292 / 101.875 us |

在四个真实算子的全部参数结果中：

```text
48 / 48 个 UB Delta 逐字节一致
预测准确率 = 100%
```

扩展到包含 shape、局部 tile 和依赖拓扑变化的 87 个真实编译样例后：

```text
精确预测             = 66
未能给出精确数值     = 21
输出错误数值         = 0
总体精确预测率       = 66 / 87 = 75.9%
```

这 21 个样例计入不精确部分，主要用于暴露当前模型尚未覆盖的复杂依赖结构。

建议性能验收目标为：

```text
IRGraph 构建 p95 <= 30 ms
全部 12 个参数查询 p95 <= 2 ms
模型构建和全部查询 <= 一次真实后端编译中位时间的 20%
```

正式 C++ 集成可以直接访问 `ModuleOp` 和 `Value::getUsers()`，不需要 Python 原型中的 MLIR 文本
索引，预计 `IRGraph` 构建时间还会进一步下降。

## 10. 结论与后续工作

当前原型已经完成：

1. 从 PlanComputeBlock IR 建立以 value 为节点的依赖图，并记录 operation、循环和计算块属性；
2. 根据编译器语义筛选受 DynamicCV 和 ordinary MultiBuffer 影响的 value；
3. 计算单 AIV function 中的物理大小和参数份数变化；
4. 对 `d=1..3`、`m=1..4` 生成可逐 value 追溯的 UB Delta；
5. 在四个真实算子上达到 100% 准确率，在扩展复杂样例中达到 75.9% 精确预测率；
6. 单次全部参数求值中位耗时约 41 μs。

下一步按以下顺序推进：

1. 修正动态 shape 依赖语义。构建 `IRGraph` 时解析并保存每个 SSA value 的准确结果类型，不再从
   整条 operation 文本推测；静态 tensor/vector 记录实际字节数，动态 tensor 和 memref 使用与
   编译器一致的 `MAX_EDGE_SIZE` 并保留图边，scalar/index 使用 0 权重。增加“静态输入、动态结果”的
   `tensor.extract_slice`、真正跨块的动态 tensor，以及静态/动态 memref 对照用例，要求图连通关系和
   边权与 `UBUsageOptPass` 一致。修复前，动态 shape 出现在跨块路径时必须返回 `unknown`，不能用于
   提前剪枝；
2. 修正循环筛选规则。当前原型要求候选循环具有 `iter_args`，但编译器的
   `AddMultiBufferInnerScope` 可以处理没有 `iter_args` 的主循环，因此这不是 DynamicCV 的必要条件。
   同时不能简单改成“只分析最内层循环”：需要复刻 `MarkMainLoop` 对嵌套和并列循环的归属规则，
   区分 `AddMultiBufferInnerScope` 与 `AddMultiBufferOuterScope` 处理的 value，并避免外层与内层重复
   计数。修复前，不能把现有四个算子未触发该边界当作一般正确性证明；
3. 为循环筛选问题增加编译器对照用例，包括无 `iter_args` 的跨 Vector block 依赖、只有外层存在依赖、
   只有内层存在依赖、内外层同时存在依赖以及多个并列主循环。逐层比较 `SeparateCVScope`、
   `AddMultiBufferInnerScope`、`AddMultiBufferOuterScope` 和最终 PlanMemory 的结果；
4. 让 `PreparedCostModel` 持有完整静态依赖模型，而不只是缓冲摘要；
5. 为未能精确预测的复杂依赖结构补充筛选规则；
6. 构造并验证至少一个两个参数产生非零联合影响的 kernel；
7. 支持多 device function 和不同 UB 地址域的峰值切换；
8. 补齐完整 StorageEntry 生命周期和地址复用建模；
9. 具备严格 overflow 判断后，再允许 autotune 提前剪枝。
