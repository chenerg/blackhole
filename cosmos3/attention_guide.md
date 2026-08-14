# Cosmos Attention 使用与选择指南

本文系统介绍 Cosmos 中 attention 的语义、mask、数据布局、模型级 attention 方法、CUDA backend 选择，以及迁移到 Ascend NPU 时的主要差异。

如果只想深入理解 MoT 的 `two_way_attention` 和 `SequencePack`，参见 [two_way_attention.md](./two_way_attention.md)。本文关注更上层的问题：

> 面对一种 attention 需求，应该怎样表达 mask、选择 API，并判断最终可能运行在哪个 kernel 上？

本文依据当前仓库代码编写。CUDA、NATTEN、FlashAttention、CANN 和 `torch_npu` 的能力与版本相关，部署时仍应以实际安装版本的兼容性检查为准。

## 1. 快速选择

先用下面这张表确定大方向：

| 需求 | 推荐入口 | 典型实现 |
|---|---|---|
| 普通 full/causal attention | `attention()` | Flash3、cuDNN、NATTEN、Flash2 |
| 多个不等长样本 pack 到一起 | `attention()` + cumulative offsets | Flash3 或 NATTEN |
| 1D/2D/3D 局部窗口 attention | `multi_dimensional_attention()` | NATTEN |
| 任意规则或 block-sparse mask | FlexAttention | `mask_mod` + `BlockMask` |
| MoT 中 U causal、G 看全部 U/G | `two_way_attention()` | 两次标准 attention |
| G→G 需要稀疏或多维 mask | `three_way_attention()` | 标准 attention + NATTEN/Flex |
| 多 control transfer inference | `multi_control_two_way_attention()` | 多次独立 maskless attention |
| CUDA varlen full/causal/window | `flash_attn_varlen_func` | CUDA FlashAttention |
| NPU full/causal varlen 兼容调用 | `npu_flash_attn_varlen_func` | `npu_fusion_attention` 包装 |
| NPU 显式二维 mask | `npu_fusion_attention` | CANN FusionAttention |

最重要的选择原则是：

```text
能用 full、causal、window 或 sequence offsets 表达
    -> 使用结构化 fused attention

不能用上述结构表达
    -> 使用 FlexAttention、PyTorch SDPA 或支持显式 mask 的算子
```

## 2. 理解 attention 的四个层次

讨论 attention 时，经常把模型语义、mask 和 CUDA kernel 混为一谈。更清晰的方式是分成四层。

### 2.1 模型语义层

首先回答：谁应该看到谁？

例如：

```text
文本 token 只能看历史文本
生成 token 可以看全部文本和生成 token
不同 packed sample 之间不能互相看到
不同视频帧只看时间窗口内的 token
```

这层决定逻辑 attention mask，但不决定如何执行。

### 2.2 数据布局层

同一种逻辑 mask 可以使用不同布局：

```text
Dense batch:   [B, S, H, D]
Packed varlen: [1, total_tokens, H, D]
Kernel varlen: [total_tokens, H, D]
Multi-dim:     [B, T, H, W, heads, D]
```

布局决定 padding、offsets 和 kernel launch metadata 的组织方式。

### 2.3 Mask 表达层

逻辑 mask 不一定要物化为矩阵。常见表达方式有：

- `is_causal=True`：隐式 causal mask；
- cumulative offsets：隐式 packed block-diagonal mask；
- `window_size`：隐式局部窗口；
- `mask_mod`/`BlockMask`：程序化或 block-sparse mask；
- bool/float Tensor：显式 `[Q,KV]` 或广播 mask。

### 2.4 Kernel 执行层

最后才是 FlashAttention、cuDNN、NATTEN、FlexAttention 或 NPU FusionAttention。

在 Cosmos 中，普通 `attention()` 会根据架构、dtype、shape、varlen、causal 类型、训练/推理和 deterministic 要求动态选择 backend，见 `cosmos_framework/model/attention/frontend.py:207-249` 与 `cosmos_framework/model/attention/backends.py:165-259`。

## 3. Q、K、V 与形状

标准 attention 为：

\[
O = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{D}} + M\right)V
\]

其中 `M` 是 mask 或 bias。

Cosmos 普通前端使用 BSHD：

```text
Q: [B, S_q,  H_q,  D_qk]
K: [B, S_kv, H_kv, D_qk]
V: [B, S_kv, H_kv, D_v]
O: [B, S_q,  H_q,  D_v]
```

接口定义见 `cosmos_framework/model/attention/frontend.py:45-64`。

### 3.1 Q/KV 可以不等长

`S_q != S_kv` 是合法的，常见于：

- cross-attention；
- decoding 时少量 query 读取较长 KV cache；
- `two_way_attention` 的 G→`[U,G]`；
- 不同 Q/KV 边界的 packed-varlen attention。

Q/KV 不等长本身不决定是否 causal。是否 causal 由 `is_causal` 决定；如果启用 causal，还必须确定左上或右下对齐。

### 3.2 MHA、GQA 与 MQA

```text
MHA: H_q == H_kv
GQA: H_q > H_kv 且 H_q % H_kv == 0
MQA: H_kv == 1
```

Cosmos 前端允许 GQA/MQA，相关检查位于 `cosmos_framework/model/attention/checks.py:150-167`。FlashAttention 和 NATTEN 通常原生处理 GQA；cuDNN adapter 会在需要时复制 K/V heads，见 `cosmos_framework/model/attention/cudnn/functions.py:159-177`。

## 4. Mask 类型

下面统一使用：

```text
■ = Query 可以看到该 Key
□ = Query 看不到该 Key
```

### 4.1 Full attention

```text
■ ■ ■ ■
■ ■ ■ ■
■ ■ ■ ■
■ ■ ■ ■
```

适用于双向 self-attention 或普通 cross-attention：

```python
attention(q, k, v, is_causal=False)
```

### 4.2 左上对齐 causal

当 Q/KV 等长时就是普通下三角：

```text
■ □ □ □
■ ■ □ □
■ ■ ■ □
■ ■ ■ ■
```

当 `Sq=2, Skv=5` 时：

```text
■ □ □ □ □
■ ■ □ □ □
```

Cosmos 表示为 `CausalType.TopLeft`。cuDNN 和 NATTEN 的当前 adapter 支持这种对齐，见：

- `cosmos_framework/model/attention/cudnn/checks.py:197-211`
- `cosmos_framework/model/attention/natten/checks.py:323-335`

### 4.3 右下对齐 causal

当 `Sq=2, Skv=5` 时：

```text
■ ■ ■ ■ □
■ ■ ■ ■ ■
```

它适合 query 对齐到 KV 序列尾部的 decoding 语义。CUDA FlashAttention 2.1+ 默认使用右下对齐。

Cosmos 表示为 `CausalType.BottomRight`。Flash2/Flash3 的当前 adapter 支持右下对齐，见：

- `cosmos_framework/model/attention/flash2/checks.py:104-116`
- `cosmos_framework/model/attention/flash3/checks.py:112-124`

### 4.4 `DontCare`

当每条序列都满足 `Sq == Skv` 时，左上和右下 causal 完全相同，因此可以使用：

```python
causal_type=CausalType.DontCare
```

定义见 `cosmos_framework/model/attention/masks.py:47-49`。

注意：packed tensor 的总 Q/KV 长度相等还不够，每个 sample 的 Q/KV 长度也必须对应相等。

### 4.5 Packed-varlen block mask

假设两个样本长度分别为 3 和 5：

```text
packed = [sample 0 tokens | sample 1 tokens]
offsets = [0, 3, 8]
```

逻辑 full mask 是：

```text
■ ■ ■ │ □ □ □ □ □
■ ■ ■ │ □ □ □ □ □
■ ■ ■ │ □ □ □ □ □
──────┼──────────
□ □ □ │ ■ ■ ■ ■ ■
□ □ □ │ ■ ■ ■ ■ ■
□ □ □ │ ■ ■ ■ ■ ■
□ □ □ │ ■ ■ ■ ■ ■
□ □ □ │ ■ ■ ■ ■ ■
```

代码不会创建这个 8×8 mask。`offsets=[0,3,8]` 告诉 varlen kernel：

```text
sequence 0 = tokens[0:3]
sequence 1 = tokens[3:8]
```

如果同时启用 causal，则每个对角 block 内再应用自己的 causal mask。

### 4.6 Sliding-window mask

每个 query 只看附近 key：

```text
■ ■ □ □ □
■ ■ ■ □ □
□ ■ ■ ■ □
□ □ ■ ■ ■
□ □ □ ■ ■
```

CUDA FlashAttention 通过 `window_size=(left,right)` 表达；Cosmos 多维局部 attention 通过 NATTEN 的 `window_size`、`stride`、`dilation` 表达，见 `cosmos_framework/model/attention/frontend.py:252-350`。

### 4.7 任意或不规则 mask

```text
■ □ ■ □
□ ■ ■ □
■ □ □ ■
□ ■ □ ■
```

这种 mask 无法由单个 full、causal、window 或 offsets 描述。可选方案：

- 把它拆成多次规则 attention，再合并结果；
- 使用 PyTorch FlexAttention 的 `mask_mod`/`BlockMask`；
- 使用 PyTorch SDPA 的显式 `attn_mask`，但不保证仍走 Flash kernel；
- NPU 上直接使用支持显式 mask 的 `npu_fusion_attention`。

## 5. Mask 矩阵与 mask metadata 的区别

### 5.1 Cosmos 普通前端没有显式 mask 参数

`cosmos_framework.model.attention.attention()` 接收：

```python
attention(
    query,
    key,
    value,
    is_causal=False,
    causal_type=None,
    seqlens_Q=None,
    seqlens_KV=None,
    cumulative_seqlen_Q=None,
    cumulative_seqlen_KV=None,
    max_seqlen_Q=None,
    max_seqlen_KV=None,
    backend=None,
)
```

它没有 `[Q,KV]` 的 `attention_mask` Tensor 参数，完整签名见 `cosmos_framework/model/attention/frontend.py:45-64`。

因此普通前端直接表达的是：

- full；
- causal；
- packed-varlen；
- full/causal 下的 Q/KV 不等长。

### 5.2 MoT 中的 `attention_mask` 实际是 `SplitInfo`

MoT 代码中：

```python
AttentionMaskType = SplitInfo
```

见 `cosmos_framework/model/generator/mot/attention.py:15-68`。

它保存 `split_lens`、`attn_modes`、`sample_lens`、`is_three_way` 等结构信息，不是 bool mask 矩阵。`dispatch_attention()` 根据这些 metadata 选择 two-way、three-way 或 multi-control，见 `cosmos_framework/model/generator/mot/attention.py:521-559`。

### 5.3 不同库的 bool mask 语义可能相反

不能只看 dtype，必须查对应 API：

```text
NPU FusionAttention: 1/True 表示屏蔽
本文方块图:         ■ 表示可见
```

PyTorch SDPA 与 `nn.MultiheadAttention` 的 bool mask 约定也不完全相同。迁移实现时应以目标函数文档为准，不能直接复用 bool Tensor 而不确认语义。

## 6. Dense 与 packed-varlen

### 6.1 Dense batch

```text
Q: [B, Sq,  Hq,  D]
K: [B, Skv, Hkv, D]
V: [B, Skv, Hkv, Dv]
```

适合 batch 内长度相同或已经 padding 的情况。优点是接口简单；缺点是长度差异大时浪费计算。

### 6.2 Packed-varlen

```text
Q: [1, total_q,  Hq,  D]
K: [1, total_kv, Hkv, D]
V: [1, total_kv, Hkv, Dv]

cumulative_seqlen_Q:  [0, q0, q0+q1, ...]
cumulative_seqlen_KV: [0, k0, k0+k1, ...]
```

Cosmos 前端要求 varlen 使用 sequence-packed 布局，说明见 `cosmos_framework/model/attention/frontend.py:69-90`。

两种传参方法：

1. 传 `seqlens_Q/KV`，前端再计算 cumulative/max；
2. 直接传 `cumulative_seqlen_Q/KV` 和 `max_seqlen_Q/KV`，效率更高。

### 6.3 Q offsets 与 KV offsets 按 sample 配对

例如：

```text
Q lengths  = [1, 2]
KV lengths = [3, 5]

Q offsets  = [0, 1, 3]
KV offsets = [0, 3, 8]
```

kernel 执行两次独立 attention：

```text
sample 0: [1,3]
sample 1: [2,5]
```

第 `i` 个 Q segment 必须和第 `i` 个 KV segment 属于同一个样本，否则会产生静默的跨样本语义错误。

## 7. `SequencePack` 的角色

`SequencePack` 是一个 `TypedDict` 风格的数据结构，定义位于 `cosmos_framework/data/generator/sequence_packing/runtime.py:29-81`。

主要内容：

```text
causal_seq                U/causal token 的物理 buffer
full_only_seq             G/full token 的物理 buffer
_causal_indices           U token 在原始顺序中的位置
_full_indices             G token 在原始顺序中的位置
_causal_seq_offsets       每个样本 U segment 的边界
_full_only_seq_offsets    每个样本 G segment 的边界
sample_offsets            每个完整样本 [U,G] 的边界
max_causal_len            单样本最大 U 长度
max_full_len              单样本最大 G 长度
max_sample_len            单样本最大总长度
is_sharded                是否处于 context-parallel shard 状态
```

`SequencePack` 同时解决三个问题：

1. 按 attention mode 拆出 U/G；
2. 保留恢复原始 `[U,G]` 顺序所需的 indices；
3. 为 fused varlen kernel 提供 offsets 和最大长度。

标准构造逻辑见 `cosmos_framework/data/generator/sequence_packing/runtime.py:300-366`；`get_all_seq()` 通过 indices 恢复原始顺序，见 `runtime.py:505-525`。

## 8. 模型级 attention 方法

这里选择的是模型语义，而不是 CUDA kernel。

### 8.1 `two_way_attention`

```text
第一次：Q=U，KV=U，causal=True
第二次：Q=G，KV=[U,G]，causal=False
```

整体逻辑 mask：

```text
                   Key U          Key G
Query U      causal 下三角          屏蔽
Query G          全可见             全可见
```

适用场景：

- U 是自回归理解/文本 token；
- G 需要读取全部上下文；
- G→G 不需要局部或复杂稀疏 mask。

实现位于 `cosmos_framework/model/generator/mot/attention.py:114-214`。

### 8.2 `three_way_attention`

three-way 把 G 的计算进一步拆成：

```text
U → U    causal self-attention
G → G    generation self-attention
G → U    cross-attention
```

G→G 可以选择：

- FlexAttention；
- NATTEN 多维/稀疏 attention；
- 普通 dense varlen attention。

G→G 与 G→U 分别得到 output/LSE 后，再使用 `merge_attentions()` 合并成与一次 G→`[U,G]` softmax 等价的结果。实现位于 `cosmos_framework/model/generator/mot/attention.py:217-354`。

适用场景：

- G→G 需要多维局部窗口；
- G→G 需要 multiview 或自定义稀疏规则；
- 希望单独优化占主要开销的 generation self-attention。

### 8.3 `multi_control_two_way_attention`

布局为：

```text
full = [control 1 | control 2 | ... | control N | noisy]
```

每个 control 独立计算：

```text
KV = [text | control_i | noisy]
```

noisy output 再按 control weights 加权合并。实现位于 `cosmos_framework/model/generator/mot/attention.py:357-518`。

它用于 multi-control transfer inference，不应当作为普通 packed 多样本 attention 的替代品。

### 8.4 Dispatch 规则

```text
存在 control_stream_token_ranges
    -> multi_control_two_way_attention

否则 is_three_way=True
    -> three_way_attention

否则
    -> two_way_attention
```

源码见 `cosmos_framework/model/generator/mot/attention.py:521-559`。

## 9. Cosmos attention API

### 9.1 `attention()`

用途：普通 1D full/causal/dense/varlen attention。

候选 backend：

```python
BACKEND_MAP = {
    "cudnn": cudnn_attention,
    "natten": natten_attention,
    "flash2": flash2_attention,
    "flash3": flash3_attention,
}
```

定义见 `cosmos_framework/model/attention/frontend.py:32-38`。

### 9.2 `multi_dimensional_attention()`

用途：1D、2D、3D token layout 上的窗口、stride、dilation 和 per-dimension causal mask。

当前 `MULTI_DIM_BACKEND_MAP` 只有 NATTEN，见 `cosmos_framework/model/attention/frontend.py:40-42`。

适用于视频、图像或时空 token 的局部稀疏 attention。

### 9.3 FlexAttention 路径

Cosmos MoT FlexAttention 使用：

```text
FlexMetadata
    -> mask_mod(b,h,q_idx,kv_idx)
    -> create_block_mask(...)
    -> flex_attention(..., block_mask=...)
```

实现见：

- mask predicate：`cosmos_framework/model/generator/mot/flex_attention.py:137-180`
- 构造 `BlockMask`：`flex_attention.py:210-234`
- 执行 kernel：`flex_attention.py:237-315`

适合规则复杂但可以程序化描述、并且具有 block sparsity 的 mask。

## 10. CUDA backend 选择

### 10.1 选择过程

```text
attention(...)
    -> 通用 tensor/参数检查
    -> 判断 dense 或 varlen
    -> choose_backend(...)
    -> 按架构取得候选列表
    -> 逐个运行 backend compatibility check
    -> 选择第一个兼容 backend
```

指定 `backend="flash3"` 并不会绕过兼容性检查；不兼容时会报错，而不是强制运行，见 `cosmos_framework/model/attention/backends.py:216-232`。

### 10.2 默认候选顺序

| CUDA 架构 | 默认顺序 |
|---|---|
| SM90 | Flash3 → cuDNN → NATTEN → Flash2 |
| SM100/103 | cuDNN → NATTEN → Flash2 |
| SM110/120/121 | cuDNN → Flash2 → NATTEN |
| 其他 SM80+ | Flash2 → cuDNN → NATTEN |
| SM75–79 | NATTEN |

定义见 `cosmos_framework/model/attention/backends.py:110-161`。

这是候选顺序，不是最终结果。

### 10.3 当前 backend 能力矩阵

| Backend | Dense | Varlen | Causal 对齐 | GQA/MQA | 显式任意 mask |
|---|---:|---:|---|---:|---:|
| Flash3 | 支持 | 支持 | BottomRight / DontCare | 支持 | 不支持 |
| Flash2 | 支持 | 当前禁用 | BottomRight / DontCare | 支持 | 不支持 |
| cuDNN adapter | 支持 | 当前未集成 | TopLeft / DontCare | adapter 处理 | 不支持 |
| NATTEN FMHA | 支持 | 支持，依版本/内部 backend | TopLeft / DontCare | 支持 | 不支持 |
| NATTEN multi-dim | 支持 | 不使用普通 varlen 语义 | 多维 causal/window | 支持 | 结构化局部 mask |
| FlexAttention | 独立路径 | 自定义 packed metadata | 由 `mask_mod` 决定 | 支持配置 | 支持 `BlockMask` |

关键限制来源：

- Flash2 varlen 禁用：`cosmos_framework/model/attention/flash2/checks.py:78-83`
- Flash3 causal 类型：`cosmos_framework/model/attention/flash3/checks.py:112-124`
- cuDNN varlen/causal：`cosmos_framework/model/attention/cudnn/checks.py:193-211`
- NATTEN causal/varlen：`cosmos_framework/model/attention/natten/checks.py:183-243,323-353`

### 10.4 为什么两次 attention 可能选到不同 backend

backend 选择的 cache key 包含：

```text
Q/K/V shape
dtype 和 device
requires_grad
is_causal / causal_type
is_varlen
deterministic
```

因此同一层中的 U→U 和 G→`[U,G]` 会独立选择；它们可能使用相同 kernel，也可能因为 causal 类型或 shape 不同而走不同 backend。

## 11. CUDA `flash_attn_varlen_func`

CUDA FlashAttention 的 varlen API 直接接收去 padding 后的 token：

```python
flash_attn_varlen_func(
    q,                    # [total_q, Hq, D]
    k,                    # [total_kv, Hkv, D]
    v,                    # [total_kv, Hkv, Dv]
    cu_seqlens_q,         # [B+1]
    cu_seqlens_k,         # [B+1]
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
)
```

它没有任意 `attention_mask` Tensor 参数，支持的 mask 由以下结构化参数产生：

| 参数 | 产生的逻辑 mask |
|---|---|
| `causal=False` | full |
| `causal=True` | 右下对齐 causal |
| `window_size=(l,r)` | sliding-window |
| causal + window | 两种 mask 的交集 |
| `cu_seqlens_q/k` | packed sample block diagonal |

官方接口见 [Dao-AILab FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)，Q/KV 不等长时的右下 causal 语义见 [FlashAttention 2.1 说明](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#21-change-behavior-of-causal-flag)。

Cosmos Flash2 adapter 的调用位于 `cosmos_framework/model/attention/flash2/functions.py:126-170`。当前仓库虽然保留 varlen 调用代码，但 compatibility check 会提前排除它。

## 12. Ascend NPU attention

### 12.1 先区分包装和底层算子

常见的 `npu_flash_attn_varlen_func` 来自 Hugging Face Transformers 兼容层；真正的 `torch_npu` 底层算子是：

```python
torch_npu.npu_fusion_attention(...)
```

包装实现见 [Transformers NPU FlashAttention 集成](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/npu_flash_attention.py)。

### 12.2 NPU varlen 兼容包装

```python
npu_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    causal=False,
)
```

当前包装行为：

- `causal=False`：`atten_mask=None`；
- `causal=True`：内部创建 `[2048,2048]` bool 上三角压缩 mask；
- 默认 `sparse_mode=3`，即右下 causal；
- `NPU_FA2_SPARSE_MODE=2` 可切到左上 causal；
- `cu_seqlens[1:]` 转成 `actual_seq_qlen/kvlen`；
- `max_seqlen_q/k` 仅用于兼容 CUDA 参数顺序，当前包装没有使用；
- `window_size`、任意 mask、ALiBi 没有被包装转发。

NPU mask Tensor 约定：

```text
1 / True  = 屏蔽
0 / False = 参与计算
```

### 12.3 NPU 原生显式 mask

直接调用：

```python
torch_npu.npu_fusion_attention(
    q,
    k,
    v,
    head_num,
    input_layout="TND",
    atten_mask=mask,
    actual_seq_qlen=actual_q_offsets,
    actual_seq_kvlen=actual_kv_offsets,
    sparse_mode=...,
)
```

varlen 常见 mask 形式：

| `sparse_mode` | 含义 | 典型 mask shape |
|---:|---|---|
| 0 | full 或普通自定义 mask | `[maxSq,maxSkv]` |
| 1 | 完整 mask | `[maxSq,maxSkv]` |
| 2 | 左上 causal 压缩模板 | `[2048,2048]` |
| 3 | 右下 causal 压缩模板 | `[2048,2048]` |
| 4 | band/window | 结合 `pre_tockens/next_tockens` |
| 6 | prefix 压缩模板 | `[3072,2048]` |

varlen 普通显式 mask 使用二维 `SS=[maxSq,maxSkv]`；非 varlen 还可以支持 `BNSS`、`B1SS`、`11SS` 等广播形状。具体约束随 CANN/torch_npu 版本变化，参见：

- [Ascend FusionAttention 文档](https://gitee.com/ascend/AscendSpeed/blob/master/docs/ops/fusion_attention.md)
- [FlashAttentionVarLenScoreV2](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/FlashAttentionVarLenScoreV2.md)

`padding_mask` 在当前相关文档中仍是预留参数。varlen 通常应去除 padding，再用 cumulative lengths 表达样本边界。

## 13. CUDA 与 NPU 对比

| 能力 | CUDA `flash_attn_varlen_func` | NPU 兼容包装 | NPU 原生 FusionAttention |
|---|---|---|---|
| Full | 支持 | 支持 | 支持 |
| Packed-varlen | `cu_seqlens` | `cu_seqlens` 转 actual lengths | actual lengths |
| 右下 causal | 支持 | 默认 mode 3 | mode 3 |
| 左上 causal | 不提供切换 | 环境变量切 mode 2 | mode 2 |
| Sliding-window | `window_size` | 不支持 | band/window mode |
| 显式 `[Q,KV]` mask | 不支持 | 不支持 | 支持 |
| ALiBi | `alibi_slopes` | 不支持 | PSE/ALiBi，接口不同 |
| causal mask Tensor | 不传 | 内部 `[2048,2048]` | 调用方提供 |

对 `two_way_attention` 而言：

```text
U→U：
    Q/KV 等长 causal
    CUDA = causal=True + cu_seqlens
    NPU  = causal compressed mask + actual lengths

G→[U,G]：
    Q/KV 可不等长，non-causal
    CUDA = causal=False + cu_seqlens_q/k
    NPU  = atten_mask=None + actual lengths
```

U→U 中 Q/KV 等长，所以左上与右下 causal 等价；G→`[U,G]` 是 non-causal，所以也没有对齐差异。

## 14. Attention 选择决策

### 14.1 第一步：选择模型语义

```text
U causal，G 看全部 U/G？
    -> two_way

G→G 需要单独稀疏/多维 mask？
    -> three_way

多个 control 独立影响 noisy target？
    -> multi-control
```

### 14.2 第二步：选择 mask 表达

```text
Full？
    -> is_causal=False

Causal？
    -> is_causal=True + 正确 causal_type

Packed 多样本？
    -> cumulative offsets

局部窗口？
    -> Flash window 或 NATTEN multi-dimensional

任意复杂规则？
    -> 拆分多次 attention 或 FlexAttention
```

### 14.3 第三步：选择 dense 或 varlen

```text
所有样本长度一致，padding 开销很小
    -> dense

长度差异明显，或本来就是 SequencePack
    -> packed-varlen
```

### 14.4 第四步：让前端选择 backend

普通情况下不要硬编码 backend。统一前端会根据实际 shape 和环境选第一个兼容实现。

只在以下场景指定 backend：

- 对齐数值差异；
- 做 benchmark；
- 排查某个 backend 的兼容性或稳定性；
- 已通过目标硬件测试并需要固定执行路径。

### 14.5 第五步：跨硬件迁移检查

从 CUDA 迁移 NPU 时逐项确认：

1. causal 是左上还是右下；
2. mask 中 True/1 表示可见还是屏蔽；
3. offsets 是否包含开头的 0；
4. API 接收 device Tensor 还是 Host cumulative lengths；
5. `max_seqlen` 是否真正参与计算；
6. window、ALiBi、prefix 是否被兼容包装转发；
7. 显式 mask 是完整矩阵还是压缩模板。

## 15. 常用调用模板

### 15.1 Dense full

```python
out = attention(q, k, v)
```

### 15.2 Dense causal self-attention

```python
out = attention(
    q,
    k,
    v,
    is_causal=True,
    causal_type=CausalType.DontCare,
)
```

这里只能在 `Sq == Skv` 时使用 `DontCare`。

### 15.3 Packed-varlen full

```python
out = attention(
    q,  # [1,total_q,Hq,D]
    k,  # [1,total_kv,Hkv,D]
    v,
    cumulative_seqlen_Q=q_offsets,
    cumulative_seqlen_KV=kv_offsets,
    max_seqlen_Q=max_q,
    max_seqlen_KV=max_kv,
)
```

### 15.4 Packed-varlen causal self-attention

```python
out = attention(
    q,
    k,
    v,
    is_causal=True,
    causal_type=CausalType.DontCare,
    cumulative_seqlen_Q=offsets,
    cumulative_seqlen_KV=offsets,
    max_seqlen_Q=max_len,
    max_seqlen_KV=max_len,
)
```

### 15.5 多维局部 attention

```python
out = multi_dimensional_attention(
    q,
    k,
    v,
    window_size=(3, 5, 5),
    stride=(1, 1, 1),
    dilation=(1, 1, 1),
    is_causal=(True, False, False),
)
```

## 16. 常见误区

### 16.1 “有一个复杂 mask，就应该创建完整 bool 矩阵”

不一定。优先尝试 causal、offsets、window 或拆分多次 attention。完整矩阵的显存和带宽成本通常是 `O(Sq×Skv)`。

### 16.2 “Packed tensor 的 batch 维是 1，所以只有一个样本”

错误。`[1,total_tokens,H,D]` 中可以包含多个逻辑样本，数量由 offsets 长度决定。

### 16.3 “Q/KV 不等长就不能 causal”

错误。需要明确的是 causal 对齐方式，而不是禁止不等长。

### 16.4 “函数叫 `attention_mask`，所以一定是 bool Tensor”

错误。MoT 中的 `attention_mask` 是 `SplitInfo` metadata。

### 16.5 “指定 FlashAttention 后就一定会执行 Flash kernel”

错误。Cosmos 仍会检查架构、dtype、shape、varlen、causal 类型、deterministic 和训练支持。

### 16.6 “CUDA 与 NPU 同名 varlen API 完全等价”

错误。NPU 兼容包装会构造显式压缩 causal mask，并且没有转发 CUDA API 的所有能力。

### 16.7 “ALiBi 是 attention mask”

严格说它是加到 attention score 上的位置 bias；它改变权重但不一定把位置硬屏蔽为不可见。

### 16.8 “Padding mask 和 varlen offsets 是同一件事”

不是。padding mask 保留 padded token 并屏蔽它们；varlen 通常直接移除 padding，用 sequence boundaries 描述真实 token。

## 17. 调试与验证

遇到 backend 选择或 mask 问题时，按以下顺序检查：

1. 打印每个 sample 的 Q/KV lengths；
2. 检查 cumulative offsets 是否单调、是否从 0 开始；
3. 检查最后一个 offset 是否等于真实 token 总数；
4. 检查 Q/KV offsets 的 sample 数是否一致；
5. 检查 `causal_type` 与 Q/KV 对齐语义；
6. 检查 dtype、head dim、GQA 比例和 compute capability；
7. 打开 attention debug logs，查看各 backend 被排除的原因；
8. 用小序列物化 reference mask，与 PyTorch eager/SDPA 输出比较；
9. 跨 CUDA/NPU 比较时统一 mask 的 0/1 语义和 causal alignment。

不要只比较 packed 后的总长度。varlen 正确性的核心是每个 segment 的边界和对应关系。

## 18. 源码阅读顺序

1. `cosmos_framework/model/attention/frontend.py:45-249`

   普通 attention API、varlen 参数和 backend dispatch。

2. `cosmos_framework/model/attention/masks.py:14-49`

   TopLeft、BottomRight、DontCare 定义。

3. `cosmos_framework/model/attention/backends.py:110-259`

   架构优先级和兼容性选择。

4. `cosmos_framework/model/attention/{flash2,flash3,cudnn,natten}/checks.py`

   每个 backend 的真实能力边界。

5. `cosmos_framework/data/generator/sequence_packing/runtime.py:29-81,300-366`

   `SequencePack` 和 packed metadata 构造。

6. `cosmos_framework/model/generator/mot/attention.py:114-559`

   two-way、three-way、multi-control 和 dispatch。

7. `cosmos_framework/model/generator/mot/flex_attention.py:137-315`

   自定义 `mask_mod`、`BlockMask` 和 FlexAttention。

8. [two_way_attention.md](./two_way_attention.md)

   MoT two-way 的完整 mask、双样本 varlen 和源码讲解。

## 19. 总结

一套 attention 是否正确和高效，取决于四件事是否同时匹配：

```text
模型语义：谁应该看到谁
数据布局：dense、packed-varlen 或 multi-dimensional
mask 表达：causal、offsets、window、BlockMask 或显式 Tensor
执行 backend：Flash3、cuDNN、NATTEN、Flash2、Flex 或 NPU FusionAttention
```

推荐的思考顺序是：

```text
先定义逻辑可见性
    -> 再选择最紧凑的 mask 表达
        -> 再选择 dense/varlen 数据布局
            -> 最后让兼容性检查选择 kernel
```

不要从“我要用 FlashAttention”开始设计 mask。应先保证 attention 语义正确，再让规则化的 mask 和数据布局为 fused kernel 创造条件。
