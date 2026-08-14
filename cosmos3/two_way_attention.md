# Cosmos3 `two_way_attention` 设计与实现

## Two-way attention mask 可视化

下面先用一个包含 3 个 understanding token 和 3 个 generation token 的单样本说明整体可见性：

```text
■ = Query 可以看到该 Key
□ = Query 看不到该 Key

                         Key
               U0  U1  U2 │ G0  G1  G2
             ┌─────────────┼─────────────┐
Query     U0 │ ■   □   □  │ □   □   □  │
          U1 │ ■   ■   □  │ □   □   □  │
          U2 │ ■   ■   ■  │ □   □   □  │
             ├─────────────┼─────────────┤
          G0 │ ■   ■   ■  │ ■   ■   ■  │
          G1 │ ■   ■   ■  │ ■   ■   ■  │
          G2 │ ■   ■   ■  │ ■   ■   ■  │
             └─────────────┴─────────────┘
```

这个整体 mask 实际由两次独立的 attention 调用实现，并不会在代码中物化为上面的完整二维矩阵。

第一次调用计算 U→U causal self-attention：

```text
                         Key
                   U0  U1  U2
                 ┌─────────────┐
Query         U0 │ ■   □   □  │
              U1 │ ■   ■   □  │
              U2 │ ■   ■   ■  │
                 └─────────────┘
```

第二次调用计算 G→[U,G] non-causal attention：

```text
                         Key
               U0  U1  U2 │ G0  G1  G2
             ┌─────────────┼─────────────┐
Query     G0 │ ■   ■   ■  │ ■   ■   ■  │
          G1 │ ■   ■   ■  │ ■   ■   ■  │
          G2 │ ■   ■   ■  │ ■   ■   ■  │
             └─────────────┴─────────────┘
```

因此整体结构可以记成一个 block mask：

```text
                   Key U          Key G
Query U      causal 下三角          全屏蔽
Query G          全可见             全可见
```

对于多样本 packed-varlen 输入，每个样本都拥有一份这样的 block；不同样本之间全部屏蔽。实现上由 `_causal_seq_offsets`、`_full_only_seq_offsets` 和 `sample_offsets` 描述边界，而不是建立样本间的显式方块 mask。两次调用对应源码位于 `cosmos_framework/model/generator/mot/attention.py:180-208`。

### `num_samples = 2`：实际 packed sequence 的 attention mask

下面使用两个长度不同的样本，直接展示 pack 后的 token 顺序以及两次 attention 的等效 mask：sample 0 包含 2 个 understanding token 和 1 个 generation token，sample 1 包含 3 个 understanding token 和 2 个 generation token。

```text
sample 0: [0U0, 0U1, 0G0]                 # 2U + 1G，共 3 token
sample 1: [1U0, 1U1, 1U2, 1G0, 1G1]     # 3U + 2G，共 5 token

原始 packed sequence：
[0U0, 0U1, 0G0 | 1U0, 1U1, 1U2, 1G0, 1G1]
 ^ sample 0       ^ sample 1

sample_lens    = [3, 5]
sample_offsets = [0, 3, 8]
num_samples    = len(sample_offsets) - 1 = 2
```

这里的 `0U1` 表示 sample 0 的第 1 个 understanding token；竖线只表示样本边界，不占用 tensor 中的位置。`num_samples` 的计算和单样本 dense 快速路径判断见 `cosmos_framework/model/generator/mot/attention.py:149-168`。

在原始 packed 顺序下，两次 attention 合在一起形成下面这个**等效总 mask**：

```text
■ = 可见，□ = 屏蔽

                                  Key
             0U0 0U1 0G0 │ 1U0 1U1 1U2 1G0 1G1
           ┌──────────────┼───────────────────────┐
Query 0U0 │  ■   □   □  │  □   □   □   □   □  │
      0U1 │  ■   ■   □  │  □   □   □   □   □  │
      0G0 │  ■   ■   ■  │  □   □   □   □   □  │
          ├──────────────┼───────────────────────┤
      1U0 │  □   □   □  │  ■   □   □   □   □  │
      1U1 │  □   □   □  │  ■   ■   □   □   □  │
      1U2 │  □   □   □  │  ■   ■   ■   □   □  │
      1G0 │  □   □   □  │  ■   ■   ■   ■   ■  │
      1G1 │  □   □   □  │  ■   ■   ■   ■   ■  │
           └──────────────┴───────────────────────┘
```

这个 8×8 矩阵只用于解释逻辑，代码**不会真正创建它**。它有两个关键特征：

- 每个样本内部仍是 `U→U` 下三角、`G→[U,G]` 全可见。
- sample 0 与 sample 1 之间全部为 `□`，所以 pack 不会造成跨样本信息泄漏。

#### 第一次 attention：把两个样本的 U token pack 在一起

`SequencePack.causal_seq` 会从每个样本中抽出 U token，并保持样本顺序：

```text
Q = K = V = [0U0, 0U1 | 1U0, 1U1, 1U2]

_causal_indices     = [0, 1, 3, 4, 5]
_causal_seq_lens    = [2, 3]
_causal_seq_offsets = [0, 2, 5]
max_causal_len      = 3
```

传给 varlen attention 的有效 mask 是两个彼此独立的 causal 三角块：

```text
                              Key
             0U0 0U1 │ 1U0 1U1 1U2
           ┌─────────┼──────────────┐
Query 0U0 │  ■   □  │  □   □   □  │
      0U1 │  ■   ■  │  □   □   □  │
          ├─────────┼──────────────┤
      1U0 │  □   □  │  ■   □   □  │
      1U1 │  □   □  │  ■   ■   □  │
      1U2 │  □   □  │  ■   ■   ■  │
           └─────────┴──────────────┘
```

对应的关键调用参数是：

```python
attention(
    causal_q.unsqueeze(0),
    causal_k.unsqueeze(0),
    causal_v.unsqueeze(0),
    is_causal=True,
    cumulative_seqlen_Q=causal_q_offsets,   # [0, 2, 5]
    cumulative_seqlen_KV=causal_k_offsets,  # [0, 2, 5]
    max_seqlen_Q=3,
    max_seqlen_KV=3,
)
```

源码见 `cosmos_framework/model/generator/mot/attention.py:170-188`。虽然物理 tensor 的 batch 维仍为 1，但 `_causal_seq_offsets=[0,2,5]` 告诉 varlen kernel：这里实际包含两条长度分别为 2 和 3 的独立序列，不能让第一段和第二段互相 attention。忽略 head 维差异时，Q/K/V 的形状都是 `[1, 5, H, D]`；`max_seqlen_Q=max_seqlen_KV=3` 取两条 U 序列长度的最大值，而不是总 packed 长度 5。

#### 第二次 attention：G 作为 Q，原始完整 pack 作为 KV

G token 也会被单独抽出形成 query；key 和 value 则通过 `get_all_seq()` 恢复为每个样本的 `[U,G]` 原始顺序：

```text
Q  = [0G0 | 1G0, 1G1]
KV = [0U0, 0U1, 0G0 | 1U0, 1U1, 1U2, 1G0, 1G1]

_full_indices          = [2, 6, 7]
_full_only_seq_lens    = [1, 2]
_full_only_seq_offsets = [0, 1, 3]  # Q 的样本边界
sample_offsets         = [0, 3, 8]  # KV 的样本边界
max_full_len           = 2
max_sample_len         = 5
```

因此第二次调用的 Q/KV 不等长，并且两个样本的 Q/KV 长度也各不相同。等效 mask 是互相隔离的 1×3 和 2×5 full-attention 块：

```text
                                  Key / Value
             0U0 0U1 0G0 │ 1U0 1U1 1U2 1G0 1G1
           ┌──────────────┼───────────────────────┐
Query 0G0 │  ■   ■   ■  │  □   □   □   □   □  │
          ├──────────────┼───────────────────────┤
      1G0 │  □   □   □  │  ■   ■   ■   ■   ■  │
      1G1 │  □   □   □  │  ■   ■   ■   ■   ■  │
           └──────────────┴───────────────────────┘
```

对应的关键调用参数是：

```python
attention(
    full_q.unsqueeze(0),
    get_all_seq(packed_key_normalized).unsqueeze(0),
    get_all_seq(packed_value_states).unsqueeze(0),
    cumulative_seqlen_Q=full_q_offsets,     # [0, 1, 3]
    cumulative_seqlen_KV=sample_offsets,    # [0, 3, 8]
    max_seqlen_Q=2,
    max_seqlen_KV=5,
)
```

源码见 `cosmos_framework/model/generator/mot/attention.py:193-208`。忽略 head 维差异时，Q 的形状是 `[1, 3, Hq, D]`，K/V 的形状是 `[1, 8, Hkv, D]`。`full_q_offsets=[0,1,3]` 与 `sample_offsets=[0,3,8]` 按相同的 sample index 配对：sample 0 计算 1×3 attention，sample 1 计算 2×5 attention。这里 Q/KV 不等长并不会自动启用 causal；第二次调用没有传 `is_causal`，因而采用统一前端定义的默认值 `False`，见 `cosmos_framework/model/attention/frontend.py:49`。

总结这两个 packed 调用：

```text
第一次：5 个 U query × 5 个 U key，offsets 把它切成 2×2 和 3×3 causal attention
第二次：3 个 G query × 8 个 U/G key，offsets 把它切成 1×3 和 2×5 full attention
```

`SequencePack` 拆分 U/G 的实现见 `cosmos_framework/data/generator/sequence_packing/runtime.py:343-365`，重新取得完整 `[U,G]` sequence 的 `get_all_seq()` 见 `cosmos_framework/data/generator/sequence_packing/runtime.py:505-525`。如果为了 context parallel 或 CUDA Graph 在 packed tensor 尾部附加了 padding，真实 token 的可见范围仍由这些 offsets 截止；上图为突出 attention 语义而省略了尾部 padding。

本文介绍 Cosmos3 MoT（Mixture of Transformers）生成模型中的 `two_way_attention`：它解决什么问题、输入 `SequencePack` 如何组织、两次底层 attention 分别计算什么、单样本和多样本怎样执行，以及运行时最终如何选择 FlashAttention、cuDNN 或 NATTEN。文档同时对比 CUDA 与 Ascend NPU 的 varlen 接口和不同 mask 支持，见第 10.4 节。

本文聚焦以下主调用链：

```text
PackedAttentionMoT.forward
    -> 构造 Q/K/V SequencePack
    -> dispatch_attention
    -> two_way_attention
        -> understanding causal self-attention
        -> generation-to-all non-causal attention
    -> attention frontend
    -> FlashAttention 2/3、cuDNN 或 NATTEN
```

相关入口：

- `PackedAttentionMoT.forward`：`cosmos_framework/model/generator/mot/unified_mot.py:583`
- `dispatch_attention`：`cosmos_framework/model/generator/mot/attention.py:521`
- `two_way_attention`：`cosmos_framework/model/generator/mot/attention.py:114`
- 统一 attention 前端：`cosmos_framework/model/attention/frontend.py:45`
- `SequencePack` 运行时工具：`cosmos_framework/data/generator/sequence_packing/runtime.py:29`

## 1. 核心结论

`two_way_attention` 不是某个具体 CUDA kernel，而是一种 attention 可见性组织方式。它将 token 分成两类：

- **understanding（简称 U）**：主要是 text/reasoner token，使用 causal attention。
- **generation（简称 G）**：主要是 vision、action、sound 等生成 token，使用 non-causal full attention。

它在每个 Transformer 层中执行两次 attention：

```text
第一次：Q = U，KV = U，causal=True
第二次：Q = G，KV = [U, G]，causal=False
```

可见性可表示为：

```text
U_i -> U_0 ... U_i
G_i -> 所有 U + 所有 G（仅限当前样本）
```

对应数学形式：

\[
O_U = \operatorname{softmax}\left(\frac{Q_U K_U^T}{\sqrt{D}} + M_{causal}\right)V_U
\]

\[
O_G = \operatorname{softmax}\left(\frac{Q_G [K_U;K_G]^T}{\sqrt{D}}\right)[V_U;V_G]
\]

因此，“two-way”指两种信息流，而不是两个固定的底层实现。两次调用最终都进入同一个统一 attention 前端，但传入的 Q/K/V、mask、序列边界和计算规模不同。

## 2. `two_way_attention` 的上游

### 2.1 U/G 使用独立的投影参数

`PackedAttentionMoT` 为 understanding 和 generation 定义了两套 QKV/O 投影：

```python
# Understanding pathway
self.q_proj = nn.Linear(...)
self.k_proj = nn.Linear(...)
self.v_proj = nn.Linear(...)
self.o_proj = nn.Linear(...)

# Generation pathway
self.q_proj_moe_gen = nn.Linear(...)
self.k_proj_moe_gen = nn.Linear(...)
self.v_proj_moe_gen = nn.Linear(...)
self.o_proj_moe_gen = nn.Linear(...)
```

源码：`cosmos_framework/model/generator/mot/unified_mot.py:501-548`。

这意味着 U/G 不仅拥有不同的 attention 可见性，还可以学习不同的投影空间。

### 2.2 QK Norm 和 RoPE

前向过程中先分别投影 U/G，再 reshape 成 heads-last 布局：

```python
q_und = q_und_in.view(-1, num_attention_heads, head_dim)
k_und = k_und_in.view(-1, num_key_value_heads, head_dim)
v_und = v_und_in.view(-1, num_key_value_heads, head_dim)

q_gen = q_gen_in.view(-1, num_attention_heads, head_dim)
k_gen = k_gen_in.view(-1, num_key_value_heads, head_dim)
v_gen = v_gen_in.view(-1, num_key_value_heads, head_dim)
```

源码：`cosmos_framework/model/generator/mot/unified_mot.py:611-626`。

随后分别执行可配置的 QK RMSNorm，并对 Q/K 应用 RoPE：

```python
q_und = self.q_norm(q_und)
k_und = self.k_norm(k_und)
q_gen = self.q_norm_moe_gen(q_gen)
k_gen = self.k_norm_moe_gen(k_gen)

q_und_, k_und_ = self._apply_rotary_pos_emb(...)
q_gen_, k_gen_ = self._apply_rotary_pos_emb(...)
```

源码：`cosmos_framework/model/generator/mot/unified_mot.py:628-650`。

最终形状为：

```text
Q_U: [N_und, H_q,  D]
K_U: [N_und, H_kv, D]
V_U: [N_und, H_kv, D]

Q_G: [N_gen, H_q,  D]
K_G: [N_gen, H_kv, D]
V_G: [N_gen, H_kv, D]
```

`H_q` 可以大于 `H_kv`，即模型支持 GQA/MQA。

### 2.3 构造 Q/K/V `SequencePack`

投影和 RoPE 完成后，代码把 U/G 两段重新装入三个结构相同的 `SequencePack`：

```python
packed_query_states_ = from_und_gen_splits(q_und_, q_gen_, pack)
packed_key_states_ = from_und_gen_splits(k_und_, k_gen_, pack)
packed_value_states_ = from_und_gen_splits(v_und, v_gen, pack)
```

源码：`cosmos_framework/model/generator/mot/unified_mot.py:652-654`。

三个 pack 共享相同的样本边界、索引和长度 metadata，区别只在于其中存放的是 Q、K 还是 V。

## 3. `SequencePack` 的内容

### 3.1 它不是一个自定义类

当前定义只是：

```python
SequencePack = dict[str, Any]
```

源码：`cosmos_framework/data/generator/sequence_packing/runtime.py:29`。

它可以理解为：

> 两个连续的 token buffer，加上一组描述样本边界、真实长度、padding 和原始顺序的 metadata。

典型结构如下：

```python
{
    # Token buffers
    "causal_seq": Tensor,
    "full_only_seq": Tensor,

    # Varlen boundaries
    "_causal_seq_offsets": Tensor,
    "_full_only_seq_offsets": Tensor,
    "sample_offsets": Tensor,

    # Mapping back to original packed order
    "_causal_indices": Tensor,
    "_full_indices": Tensor,

    # Real token counts, excluding tail padding
    "_num_causal_tokens": int,
    "_num_full_tokens": int,

    # Per-sample maximum lengths
    "max_causal_len": int,
    "max_full_len": int,
    "max_sample_len": int,

    # Runtime state
    "max_num_tokens": int,
    "is_sharded": bool,
}
```

metadata 的定义与导出位于 `cosmos_framework/data/generator/sequence_packing/runtime.py:32-81`，pack 构造位于 `runtime.py:300-366`。

### 3.2 两个物理 token buffer

当前 `SequencePack` 支持的布局有一个关键不变量：

```text
understanding == causal_seq
generation    == full_only_seq
```

因此：

```python
get_und_seq(pack)  # 返回 pack["causal_seq"]
get_gen_seq(pack)  # 返回 pack["full_only_seq"]
```

源码：`cosmos_framework/data/generator/sequence_packing/runtime.py:459-502`。

Q/K/V pack 的物理内容可写成：

```text
packed_query_states:
    causal_seq    = Q_U [N_und_padded, H_q,  D]
    full_only_seq = Q_G [N_gen_padded, H_q,  D]

packed_key_states:
    causal_seq    = K_U [N_und_padded, H_kv, D]
    full_only_seq = K_G [N_gen_padded, H_kv, D]

packed_value_states:
    causal_seq    = V_U [N_und_padded, H_kv, D]
    full_only_seq = V_G [N_gen_padded, H_kv, D]
```

### 3.3 三组 offsets

#### `_causal_seq_offsets`

它描述 `causal_seq` 中每个样本 U 序列的累计边界。

如果两个样本的 U 长度分别为 3、2：

```python
_causal_seq_offsets = [0, 3, 5]
```

则：

```text
sample 0 U = causal_seq[0:3]
sample 1 U = causal_seq[3:5]
```

`get_causal_seq()` 返回 `(causal_seq, _causal_seq_offsets)`，见 `cosmos_framework/data/generator/sequence_packing/runtime.py:541-550`。

#### `_full_only_seq_offsets`

它描述 `full_only_seq` 中每个样本 G 序列的累计边界。

如果两个样本的 G 长度分别为 2、4：

```python
_full_only_seq_offsets = [0, 2, 6]
```

`get_full_only_seq()` 返回 `(full_only_seq, _full_only_seq_offsets)`，见 `cosmos_framework/data/generator/sequence_packing/runtime.py:553-562`。

#### `sample_offsets`

它描述原始完整 `[U, G]` 序列中每个样本的累计边界。

```text
sample 0: 3 U + 2 G = 5
sample 1: 2 U + 4 G = 6
```

对应：

```python
sample_offsets = [0, 5, 11]
```

`sample_offsets` 由 `sample_lens` 的累计和产生，见 `cosmos_framework/data/generator/sequence_packing/runtime.py:197-229`。

三组 offsets 的用途不同：

| 字段 | 描述对象 | 在 `two_way_attention` 中的用途 |
|---|---|---|
| `_causal_seq_offsets` | 每个样本的 U | 第一次 attention 的 Q/KV 边界 |
| `_full_only_seq_offsets` | 每个样本的 G | 第二次 attention 的 Q 边界 |
| `sample_offsets` | 每个样本的 `[U,G]` | 第二次 attention 的 KV 边界 |

### 3.4 `_causal_indices` 与 `_full_indices`

这两个字段记录 U/G 在原始 packed sequence 中的位置。

例如原始顺序是：

```text
[U0 U1 U2 G0 G1 | U3 U4 G2 G3 G4 G5]
```

则：

```python
_causal_indices = [0, 1, 2, 5, 6]
_full_indices   = [3, 4, 7, 8, 9, 10]
```

构造 pack 时：

```python
causal_seq = packed_sequence[_causal_indices]
full_only_seq = packed_sequence[_full_indices]
```

源码：`cosmos_framework/data/generator/sequence_packing/runtime.py:343-345`。

反过来，`get_all_seq()` 使用这些索引恢复原始顺序：

```python
out[_causal_indices] = causal_seq[:num_real_causal]
out[_full_indices] = full_only_seq[:num_real_full]
```

源码：`cosmos_framework/data/generator/sequence_packing/runtime.py:505-525`。

这个恢复动作对第二次 attention 很关键，因为它的 KV 必须按每个样本的原始 `[U,G]` 顺序排列，才能与 `sample_offsets` 对齐。

### 3.5 最大长度与真实 token 数

```text
max_causal_len = max_i len(U_i)
max_full_len   = max_i len(G_i)
max_sample_len = max_i (len(U_i) + len(G_i))
```

这些是单个样本的最大长度，不是物理 buffer 的总长度。它们作为 varlen kernel 的 `max_seqlen_Q/KV`。

另外：

```text
_num_causal_tokens = 所有真实 U token 数
_num_full_tokens   = 所有真实 G token 数
```

它们不含尾部 padding，主要用于 KV cache 和后续 MLP 截断。相关使用见 `cosmos_framework/model/generator/mot/unified_mot.py:691-704`、`unified_mot.py:1160-1161`。

### 3.6 Padding 与 `is_sharded`

U/G buffer 可能为了以下目的在尾部补零：

- 使长度能被 Context Parallel world size 整除；
- 对长度做 bucket 化，减少 `torch.compile` 重编译；
- 减少 CUDA Graph 重捕获。

padding 逻辑见 `cosmos_framework/data/generator/sequence_packing/runtime.py:147-177`、`runtime.py:254-297`。

`is_sharded` 表示当前 U/G buffer 是否已被 Context Parallel 按序列维切片。分片状态下当前 rank 不拥有完整原始序列，因此 `get_all_seq()` 会拒绝执行，见 `runtime.py:513-517`。

## 4. `two_way_attention` 的完整结构

函数主体位于 `cosmos_framework/model/generator/mot/attention.py:114-214`，可概括为：

```python
def two_way_attention(q_pack, k_pack, v_pack, normalized_k_pack=None):
    # 1. 取出 U 路径 Q/K/V
    causal_q, causal_q_offsets = get_causal_seq(q_pack)
    causal_k, causal_k_offsets = get_causal_seq(k_pack)
    causal_v, _ = get_causal_seq(v_pack)

    # 2. U -> U causal self-attention
    causal_out = attention(
        causal_q.unsqueeze(0),
        causal_k.unsqueeze(0),
        causal_v.unsqueeze(0),
        is_causal=True,
        causal_type=...,
        ...,
    )

    # 3. 取出 G queries，恢复完整 [U,G] K/V
    full_q, full_q_offsets = get_full_only_seq(q_pack)
    all_k = get_all_seq(normalized_k_pack or k_pack)
    all_v = get_all_seq(v_pack)

    # 4. G -> [U,G] non-causal attention
    full_out = attention(
        full_q.unsqueeze(0),
        all_k.unsqueeze(0),
        all_v.unsqueeze(0),
        ...,
    )

    # 5. 输出仍按 U/G 两段存入 SequencePack
    return from_mode_splits(causal_out, full_out, q_pack)
```

下面分别说明两次调用。

## 5. 第一次调用：U 到 U 的 causal self-attention

### 5.1 Q/K/V 与形状

源码：

```python
causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
causal_v, _ = get_causal_seq(packed_value_states)

causal_res = attention(
    causal_q.unsqueeze(0),
    causal_k.unsqueeze(0),
    causal_v.unsqueeze(0),
    is_causal=True,
    causal_type=(
        CausalType.DontCare
        if use_dont_care_mask
        else CausalType.TopLeft
    ),
    **causal_varlen_kwargs,
)
```

来源：`cosmos_framework/model/generator/mot/attention.py:135-188`。

输入统一成 BSHD：

```text
Q: [1, N_und_padded, H_q,  D]
K: [1, N_und_padded, H_kv, D]
V: [1, N_und_padded, H_kv, D]
```

它是 self-attention，因为 Q/K/V 都来自 U；它是 causal，因为显式设置了 `is_causal=True`。

### 5.2 可见性

对一个 U 序列：

```text
        K0 K1 K2 K3
Q0      ✓  ✗  ✗  ✗
Q1      ✓  ✓  ✗  ✗
Q2      ✓  ✓  ✓  ✗
Q3      ✓  ✓  ✓  ✓
```

U token 保持语言模型/reasoner 的自回归语义。

### 5.3 为什么使用 `DontCare` 或 `TopLeft`

仓库支持三种 causal 对齐：

- `TopLeft`：Q 从 KV 左端开始对齐；
- `BottomRight`：Q 从 KV 右端对齐，常用于增量 decode；
- `DontCare`：仅当 Q/KV 等长时可用，因为此时两种对齐等价。

定义见 `cosmos_framework/model/attention/masks.py:14-49`。

第一次 attention 中，每个样本的 Q/KV 都是同一段 U，理论上等长。代码通过 offsets 是否为同一个 Tensor 对象做零开销确认：

```python
use_dont_care_mask = causal_q_offsets is causal_k_offsets
```

源码：`cosmos_framework/model/generator/mot/attention.py:140-147`。

如果共享同一个 offsets 对象，就使用 `DontCare`，从而允许更多 backend 参与选择。如果 offsets 只是数值相等、但并非同一个对象，代码保守地使用 `TopLeft`。

这一点会影响底层 kernel：

| Backend | 当前支持的 causal 对齐 |
|---|---|
| FlashAttention 2 | `BottomRight`、`DontCare` |
| FlashAttention 3 | `BottomRight`、`DontCare` |
| cuDNN | `TopLeft`、`DontCare` |
| NATTEN | `TopLeft`、`DontCare` |

对应检查位于：

- `cosmos_framework/model/attention/flash2/checks.py:105-116`
- `cosmos_framework/model/attention/flash3/checks.py:112-124`
- `cosmos_framework/model/attention/cudnn/checks.py:197-209`
- `cosmos_framework/model/attention/natten/checks.py:323-335`

因此共享 offsets 不只是小优化，它还能避免不必要地排除 FlashAttention。

## 6. 第二次调用：G 到 `[U,G]` 的 non-causal attention

### 6.1 Q/K/V 与形状

源码：

```python
full_q, full_q_offsets = get_full_only_seq(packed_query_states)

full_res = attention(
    full_q.unsqueeze(0),
    get_all_seq(packed_key_normalized).unsqueeze(0),
    get_all_seq(packed_value_states).unsqueeze(0),
    **full_varlen_kwargs,
)
```

来源：`cosmos_framework/model/generator/mot/attention.py:138`、`attention.py:193-208`。

这里：

```text
Q  = Q_G
K  = [K_U, K_G]
V  = [V_U, V_G]
```

形状通常为：

```text
Q:  [1, N_gen_padded, H_q,  D]
K:  [1, N_all_real,   H_kv, D]
V:  [1, N_all_real,   H_kv, D]
```

它没有传 `is_causal=True`，因此使用默认的 `is_causal=False`。

### 6.2 可见性

对单个样本：

```text
        U0 U1 U2 G0 G1 G2
G0      ✓  ✓  ✓  ✓  ✓  ✓
G1      ✓  ✓  ✓  ✓  ✓  ✓
G2      ✓  ✓  ✓  ✓  ✓  ✓
```

每个 G token 可以看到：

- 当前样本的全部 U token；
- 当前样本的全部 G token；
- 包括时间或空间顺序上位于自己“未来”的 G token。

因此第二次调用不是 autoregressive attention，而是 diffusion/generation tower 的双向 joint attention。

### 6.3 Q/KV 不等长为什么没有问题

通常：

```text
S_Q  = S_G
S_KV = S_U + S_G
```

Q/KV 不等长本身是合法的。只有同时满足以下条件时，才需要讨论 `TopLeft` 或 `BottomRight`：

```text
Q/KV 不等长 + is_causal=True
```

第二次调用是 non-causal，所以不存在 causal 对齐歧义。它可以直接作为变长 cross/joint attention 执行。

## 7. 单样本与多样本执行方式

### 7.1 单样本、无梯度推理：dense 快路径

代码只在以下条件启用 dense 快路径：

```python
use_dense = num_samples == 1 and not torch.is_grad_enabled()
```

源码：`cosmos_framework/model/generator/mot/attention.py:149-168`。

此时不传 cumulative offsets：

```python
causal_varlen_kwargs = {}
full_varlen_kwargs = {}
```

但两次调用的语义不变：

```text
第一次：dense causal U self-attention
第二次：dense non-causal G -> [U,G] attention
```

这样可以省去单样本下冗余的 varlen metadata，并减少 `torch.compile` 对不同 batch composition 的 specialization。

### 7.2 训练或多样本推理：packed-varlen 路径

训练时 `torch.is_grad_enabled()` 为真；多样本推理时 `num_samples > 1`。两种情况都会进入 packed-varlen 路径。

所有样本被拼进物理 batch size 为 1 的长序列：

```text
[sample 0 tokens | sample 1 tokens | ...]
```

样本边界由 cumulative offsets 传给 fused kernel。统一前端明确规定 varlen 输入使用 sequence-packed 布局，见 `cosmos_framework/model/attention/frontend.py:69-90`。

#### 第一次 attention 的 varlen 参数

```python
cumulative_seqlen_Q=causal_q_offsets
cumulative_seqlen_KV=causal_k_offsets
max_seqlen_Q=packed_query_states["max_causal_len"]
max_seqlen_KV=packed_query_states["max_causal_len"]
```

来源：`cosmos_framework/model/generator/mot/attention.py:170-178`。

每个样本内部独立执行 U causal self-attention。

#### 第二次 attention 的 varlen 参数

```python
cumulative_seqlen_Q=full_q_offsets
cumulative_seqlen_KV=sample_offsets
max_seqlen_Q=packed_query_states["max_full_len"]
max_seqlen_KV=packed_query_states["max_sample_len"]
```

来源：`cosmos_framework/model/generator/mot/attention.py:193-201`。

Q offsets 与 KV offsets 不同，因为每个样本内：

```text
Q 长度  = len(G_i)
KV 长度 = len(U_i) + len(G_i)
```

这仍然是一个 kernel launch；offsets 使其等价于分别对每个样本调用 attention，同时避免 padding batch 和显式二维 mask。

## 8. 双样本完整例子

假设输入为：

```text
sample 0: [U0 U1 U2 | G0 G1]
sample 1: [U3 U4    | G2 G3 G4 G5]
```

### 8.1 `SequencePack` metadata

```python
causal_seq = [U0, U1, U2, U3, U4]
full_only_seq = [G0, G1, G2, G3, G4, G5]

_causal_seq_offsets = [0, 3, 5]
_full_only_seq_offsets = [0, 2, 6]
sample_offsets = [0, 5, 11]

_causal_indices = [0, 1, 2, 5, 6]
_full_indices = [3, 4, 7, 8, 9, 10]

max_causal_len = 3
max_full_len = 4
max_sample_len = 6
```

### 8.2 第一次 attention

物理输入：

```text
Q/K/V = [U0 U1 U2 | U3 U4]
offsets = [0, 3, 5]
causal = True
```

逻辑上等价于：

```text
sample 0: causal_attention(U0..U2)
sample 1: causal_attention(U3..U4)
```

样本之间不可见。

### 8.3 第二次 attention

物理输入：

```text
Q = [G0 G1 | G2 G3 G4 G5]
Q offsets = [0, 2, 6]

KV = [U0 U1 U2 G0 G1 | U3 U4 G2 G3 G4 G5]
KV offsets = [0, 5, 11]

causal = False
```

逻辑上等价于：

```text
sample 0: attention(Q=[G0,G1],       KV=[U0,U1,U2,G0,G1])
sample 1: attention(Q=[G2,G3,G4,G5], KV=[U3,U4,G2,G3,G4,G5])
```

同样不会发生跨样本 attention。

## 9. `packed_key_states_normalized` 的作用

函数的第四个参数是一个可选备用 K pack：

```python
packed_key_normalized = (
    packed_key_states_normalized
    if packed_key_states_normalized is not None
    else packed_key_states
)
```

源码：`cosmos_framework/model/generator/mot/attention.py:129-133`。

它通常包含：

```text
causal_seq    = 额外 RMSNorm 后再应用 RoPE 的 K_U
full_only_seq = 原来的 K_G
metadata      = 与普通 K pack 相同
```

构造逻辑位于 `cosmos_framework/model/generator/mot/unified_mot.py:656-671`。

这个备用 K 只用于第二次 G→all attention：

- 第一次 U→U causal attention 始终使用普通 `packed_key_states`；
- 第二次可以让归一化后的 G query 与同尺度的 U key 做点积；
- G key 不需要变化。

它主要解决某些配置中 `qk_norm_for_diffusion=True`、`qk_norm_for_text=False` 导致的尺度失配。相关模块创建条件见 `cosmos_framework/model/generator/mot/unified_mot.py:523-534`。

## 10. 统一 attention 前端与最终 kernel

两次调用都进入：

```python
attention(
    query, key, value,
    is_causal=...,
    causal_type=...,
    cumulative_seqlen_Q=...,
    cumulative_seqlen_KV=...,
    ...,
)
```

统一前端位于 `cosmos_framework/model/attention/frontend.py:45-249`，输入输出布局均为 BSHD。

候选 backend：

```python
BACKEND_MAP = {
    "cudnn": cudnn_attention,
    "natten": natten_attention,
    "flash2": flash2_attention,
    "flash3": flash3_attention,
}
```

源码：`cosmos_framework/model/attention/frontend.py:32-38`。

运行时会分别根据每次调用的以下属性选择 backend：

- GPU compute capability；
- dtype；
- dense 或 varlen；
- causal 类型；
- Q/KV shape；
- head dimension；
- GQA/MQA；
- 是否需要 backward；
- deterministic 要求。

选择逻辑位于 `cosmos_framework/model/attention/backends.py:164-259`。

### 10.1 默认优先级

| GPU 架构 | 默认优先顺序 |
|---|---|
| SM90（H100/H200） | FlashAttention 3 → cuDNN → NATTEN → FlashAttention 2 |
| SM100/103 | cuDNN → NATTEN → FlashAttention 2 |
| SM110/120/121 | cuDNN → FlashAttention 2 → NATTEN |
| 其他 SM80+ | FlashAttention 2 → cuDNN → NATTEN |
| SM75–79 | NATTEN |

定义：`cosmos_framework/model/attention/backends.py:110-161`。

这是候选顺序，不是静态保证。每次 attention 都会单独做兼容性判断，所以同一层中的两次调用也可能选择不同 backend。

### 10.2 多样本推理的常见结果

多样本推理时两次均为 varlen。当前 cuDNN adapter 尚未集成 varlen，见 `cosmos_framework/model/attention/cudnn/checks.py:193-195`，因此常见结果是：

| GPU | 第一次：causal varlen | 第二次：non-causal varlen |
|---|---|---|
| SM90 | 通常 FlashAttention 3 | 通常 FlashAttention 3 |
| SM80 | 通常 NATTEN | 通常 NATTEN |
| SM100/103 | 通常 NATTEN | 通常 NATTEN |
| SM110/120/121 | 通常 NATTEN | 通常 NATTEN |

实际结果仍取决于安装版本、dtype、head dimension 和 causal alignment。当前仓库因为稳定性问题显式禁用了 FlashAttention 2 varlen，见 `cosmos_framework/model/attention/flash2/checks.py:78-83`。尤其是第一次如果退化为 `TopLeft` causal，FlashAttention 3 也会被当前兼容性检查排除，而第二次 non-causal 调用不受此限制。

### 10.3 GQA 的处理

统一前端允许：

```text
H_q >= H_kv
H_q % H_kv == 0
```

检查位于 `cosmos_framework/model/attention/checks.py:150-167`。

FlashAttention、NATTEN 通常原生处理 GQA。cuDNN 的底层 ATen op 要求 Q/K/V heads 相同，因此 adapter 会显式复制 K/V heads：

```python
k = k.repeat_interleave(group_size, dim=1)
v = v.repeat_interleave(group_size, dim=1)
```

源码：`cosmos_framework/model/attention/cudnn/functions.py:159-177`。

### 10.4 CUDA 与 NPU：varlen attention 和 mask 支持对比

这一节区分三个容易混在一起的概念：

```text
逻辑 mask：从模型角度看，某个 Query 能看到哪些 Key
显式 mask：真正传给算子的 bool/uint8 Tensor
varlen 边界：用 cumulative offsets 隔离 packed samples，不物化二维 mask
```

Cosmos 当前统一前端只注册 CUDA 相关的 cuDNN、NATTEN、FlashAttention 2/3 backend，不包含 `torch_npu`，见 `cosmos_framework/model/attention/frontend.py:32-38`。下面的 NPU 部分用于说明：如果将相同的 `two_way_attention` 语义迁移到 Ascend，应当如何映射 mask 和 varlen metadata。

#### 10.4.1 总体差异

这里的 NPU 兼容包装指 Hugging Face Transformers 中的 `npu_flash_attn_varlen_func`。它不是 `torch_npu` 的底层算子；真正执行计算的是 `torch_npu.npu_fusion_attention`。

| 能力 | CUDA `flash_attn_varlen_func` | NPU `npu_flash_attn_varlen_func` 包装 | NPU `npu_fusion_attention` 原生接口 |
|---|---|---|---|
| Full attention | 支持 | 支持 | 支持 |
| Packed-varlen 样本隔离 | `cu_seqlens_q/k` | `cu_seqlens_q/k`，内部转为 `actual_seq_qlen/kvlen` | `actual_seq_qlen/kvlen` |
| 右下对齐 causal | 支持，默认 causal 语义 | 支持，默认 `sparse_mode=3` | 支持，`sparse_mode=3` |
| 左上对齐 causal | 不提供切换；FA 2.1+ 使用右下对齐 | 支持，设置 `NPU_FA2_SPARSE_MODE=2` | 支持，`sparse_mode=2` |
| Sliding-window | 支持 `window_size=(left,right)` | 不支持，`kwargs` 没有转发到算子 | 支持 band/window 模式，受 CANN 版本约束 |
| 任意二维 mask | 不支持 | 不支持 | 支持 varlen `[maxSq,maxSkv]` mask |
| ALiBi | 支持 `alibi_slopes` | 不支持 | 可通过 PSE/ALiBi 能力表达，接口不同 |
| Prefix 等压缩 mask | 不支持 | 不支持 | 部分 `sparse_mode` 支持 |
| 是否传显式 causal mask | 不传 | 内部传 `[2048,2048]` bool 压缩 mask | 由调用方提供 |

CUDA 官方接口见 [Dao-AILab FlashAttention `flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)，NPU 包装见 [Transformers `npu_flash_attention.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/npu_flash_attention.py)，NPU 原生算子约束见 [Ascend FusionAttention 文档](https://gitee.com/ascend/AscendSpeed/blob/master/docs/ops/fusion_attention.md)。

#### 10.4.2 CUDA `flash_attn_varlen_func`

CUDA varlen API 的核心输入是：

```python
flash_attn_varlen_func(
    q,                    # [total_q, Hq, D]
    k,                    # [total_kv, Hkv, D]
    v,                    # [total_kv, Hkv, Dv]
    cu_seqlens_q,         # [num_samples + 1]
    cu_seqlens_k,         # [num_samples + 1]
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    window_size=(-1, -1),
)
```

这个 API **没有任意 `attention_mask` Tensor 参数**。它只能通过结构化参数表达以下 mask：

1. `causal=False, window_size=(-1,-1)`：每个样本内部 full attention。
2. `causal=True`：每个样本内部右下对齐 causal attention。
3. `window_size=(left,right)`：局部 sliding-window attention。
4. `causal=True` 加有限 window：causal mask 与 window mask 的交集。
5. `cu_seqlens_q/k`：形成样本间不可见的 block-diagonal 边界。

例如 `Sq=2, Skv=5` 时，CUDA 的 `causal=True` 使用右下对齐：

```text
■ = 可见，□ = 屏蔽

Key      K0  K1  K2  K3  K4
Query Q0  ■   ■   ■   ■   □
      Q1  ■   ■   ■   ■   ■
```

这个逻辑 `[2,5]` mask 不会被创建。kernel 根据 `causal`、Q/KV 长度和 offsets 在计算时直接判断可见性。Q/KV 不等长时的右下对齐定义见 [FlashAttention 2.1 causal 说明](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#21-change-behavior-of-causal-flag)。

Cosmos 的 FlashAttention 2 adapter 同样只传 `causal` 和 cumulative offsets，没有传显式 mask，见 `cosmos_framework/model/attention/flash2/functions.py:126-150`。不过，当前兼容性检查会禁止这条 Flash2 varlen 分支，见 `cosmos_framework/model/attention/flash2/checks.py:78-83`。

#### 10.4.3 NPU `npu_flash_attn_varlen_func` 兼容包装

Transformers 的 NPU 包装保持与 CUDA 近似的调用形式：

```python
npu_flash_attn_varlen_func(
    q,                    # [total_q, Hq, D]，TND
    k,                    # [total_kv, Hkv, D]
    v,                    # [total_kv, Hkv, Dv]
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    causal=False,
)
```

但其内部行为和 CUDA 不完全相同：

- `causal=False`：向 `npu_fusion_attention` 传 `atten_mask=None`。
- `causal=True`：内部创建 `[2048,2048]` 的 bool 上三角压缩 mask。
- `cu_seqlens[1:]` 会被转成 Host 侧 cumulative lengths，作为 `actual_seq_qlen/kvlen`。
- `max_seqlen_q/k` 当前只是为了兼容 CUDA 参数顺序，包装内部没有使用。
- `window_size`、任意 mask、ALiBi 等参数没有被该包装转发。

NPU causal 压缩 mask 的构造是：

```python
atten_mask_npu = torch.triu(
    torch.ones([2048, 2048], device=device),
    diagonal=1,
).bool()
```

NPU mask Tensor 的数值语义与本文方块图相反：

```text
NPU atten_mask 中：True/1 = 屏蔽，False/0 = 参与计算
本文可视化中：    ■      = 可见，□       = 屏蔽
```

`sparse_mode` 决定这份压缩三角模板相对实际 `[Sq,Skv]` attention 矩阵如何对齐：

```text
sparse_mode=2：左上对齐，Sq=2、Skv=5

■ □ □ □ □
■ ■ □ □ □

sparse_mode=3：右下对齐，Sq=2、Skv=5

■ ■ ■ ■ □
■ ■ ■ ■ ■
```

Transformers 默认使用 `sparse_mode=3`，因此与 CUDA FlashAttention 2.1+ 的 causal 语义一致；只有明确需要旧式左上对齐时，才设置 `NPU_FA2_SPARSE_MODE=2`。

#### 10.4.4 直接调用 NPU `npu_fusion_attention`

如果绕过兼容包装，直接调用原生算子，则可以传显式 `atten_mask`：

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

varlen 场景的主要 mask 形式为：

| `sparse_mode` | mask 含义 | 典型显式 shape |
|---:|---|---|
| 0 | 无 mask 时为 full；传入时为普通自定义 mask | `[maxSq,maxSkv]` |
| 1 | 完整 mask | `[maxSq,maxSkv]` |
| 2 | 左上 causal 压缩模板 | `[2048,2048]` |
| 3 | 右下 causal 压缩模板 | `[2048,2048]` |
| 4 | band/sliding-window | 结合 `pre_tockens/next_tockens` |
| 6 | prefix 压缩模板 | `[3072,2048]` |

原生 NPU varlen 的普通显式 mask 只支持二维 `SS=[maxSq,maxSkv]`；非 varlen 场景还可以使用 `BNSS`、`B1SS`、`11SS` 等广播形状。mask dtype 支持 bool/uint8，值 `1` 表示屏蔽。具体 sparse modes 和 shape 约束随 CANN/torch_npu 版本变化，应以部署版本的算子文档为准；底层算子定义可参考 [FlashAttentionVarLenScoreV2](https://gitee.com/ascend/cann-ops-adv/blob/master/docs/FlashAttentionVarLenScoreV2.md)。

需要特别注意：`padding_mask` 在当前 NPU FusionAttention 文档中仍是预留参数。变长样本通常应当去掉 padding，再通过 `actual_seq_qlen/kvlen` 表达边界，而不是依赖 padding mask。

#### 10.4.5 不同 mask 应选择什么实现

| 目标 mask | CUDA 推荐 | NPU 推荐 |
|---|---|---|
| Full attention | `causal=False` | `atten_mask=None` |
| 标准右下 causal | `causal=True` | 包装默认模式或原生 `sparse_mode=3` |
| 左上 causal | 需要换实现或重新组织 Q/KV | 原生/包装 `sparse_mode=2` |
| Packed-varlen block diagonal | `cu_seqlens_q/k` | `actual_seq_qlen/kvlen` |
| Sliding window | CUDA `window_size` | 原生 NPU band/window 模式 |
| 任意 `[Q,KV]` bool mask | FlexAttention、PyTorch SDPA 等 | 原生 `npu_fusion_attention` 的 SS mask |
| 多维局部稀疏 mask | NATTEN/FlexAttention | NPU 对应 sparse operator；不能假设兼容包装支持 |

选择时最重要的原则是：**只要 mask 能由 full、causal、window 或 varlen offsets 表达，就不要物化完整二维矩阵；只有不规则可见性无法拆分时，才使用显式 mask 或 FlexAttention。**

#### 10.4.6 映射回 `two_way_attention`

`two_way_attention` 在 CUDA 和 NPU 上都不需要传入完整的整体 mask：

```text
第一次 U→U：
    Q/KV 等长 causal
    CUDA：causal=True + cu_seqlens
    NPU：causal compressed mask + actual_seq_qlen/kvlen

第二次 G→[U,G]：
    Q/KV 可以不等长，但 non-causal
    CUDA：causal=False + cu_seqlens_q/k
    NPU：atten_mask=None + actual_seq_qlen/kvlen
```

第一次调用中 Q/KV 等长，所以左上和右下 causal 的结果相同；第二次虽然 Q/KV 不等长，但它是 full attention，因此也不存在 causal 对齐差异。真正必须保持一致的是：第 `i` 段 Q offsets 必须与第 `i` 段 KV offsets 对应同一个样本。

## 11. Context Parallel 下的行为

Context Parallel 使用 Ulysses 风格的 sequence/head all-to-all：

```text
每卡 [S/CP, H, D]
    -> gather sequence, scatter heads
每卡 [S, H/CP, D]
    -> 执行 two_way_attention
    -> gather heads, scatter sequence
每卡 [S/CP, H, D]
```

实现见 `cosmos_framework/model/generator/mot/context_parallel_utils.py:298-470`。

在进入 inner `two_way_attention` 前，CP wrapper 已恢复完整 sequence，只保留本 rank 的 head shard。因此尽管外部 local `SequencePack` 的 `is_sharded=True`、不能调用 `get_all_seq()`，inner attention 使用的是 `is_sharded=False` 的完整序列 pack，第二次 attention 可以正常恢复 `[U,G]` KV。

相关转换位于：

- 输入 sequence 分片：`cosmos_framework/model/generator/mot/context_parallel_utils.py:123-182`
- attention 前 gather sequence/scatter heads：`context_parallel_utils.py:391-452`
- 输出 gather heads/scatter sequence：`context_parallel_utils.py:454-470`

## 12. 输出如何回到 Transformer 层

两次 attention 输出分别被 flatten：

```text
causal_out: [N_und, H_q, D] -> [N_und, H_q * D]
full_out:   [N_gen, H_q, D] -> [N_gen, H_q * D]
```

源码：`cosmos_framework/model/generator/mot/attention.py:190-213`。

随后通过：

```python
from_mode_splits(causal_out, full_out, packed_query_states)
```

重新生成一个 metadata 不变、token buffer 已替换的 `SequencePack`。上游再分别应用：

```python
und_seq = self.o_proj(get_und_seq(packed_attn_output))
gen_seq = self.o_proj_moe_gen(get_gen_seq(packed_attn_output))
```

见 `cosmos_framework/model/generator/mot/unified_mot.py:736-739`。

最后 U/G 各自与残差相加，见 `cosmos_framework/model/generator/mot/unified_mot.py:1140-1149`。

## 13. 性能与复杂度

设：

- U 长度为 `S_U`；
- G 长度为 `S_G`；
- head dimension 为 `D`。

忽略常数和 causal 三角区域优化，第一次 attention 的复杂度约为：

\[
O(S_U^2 D)
\]

第二次约为：

\[
O(S_G(S_U + S_G)D)
\]

在视频生成中，`S_G` 通常远大于 `S_U`，所以第二次 G→all attention 往往是主要 attention 开销。这也是需要 three-way/NATTEN 等稀疏方案时，重点拆分和优化 G→G 部分的原因。

## 14. 常见误区

### 14.1 “two-way 使用两个不同的底层实现”

不准确。两次调用都进入统一 `attention()` 前端；底层实现是运行时分别选择的，可能相同，也可能不同。

### 14.2 “第二次是 causal cross-attention”

不是。第二次没有设置 `is_causal=True`，G 对本样本全部 U/G 双向可见。

### 14.3 “Q/KV 不等长就不能 causal”

不准确。Q/KV 不等长时仍可 causal，但必须明确 Q 与 KV 是左对齐还是右对齐。第二次虽然 Q/KV 不等长，但它是 non-causal，因此没有对齐问题。

### 14.4 “多样本使用普通 `[B,S,H,D]` padding batch”

不是。MoT 主路径把多个样本拼成 `[1,total_tokens,H,D]`，通过 cumulative offsets 执行 packed-varlen attention。

### 14.5 “`max_causal_len` 是 `causal_seq.shape[0]`”

不是。前者是 batch 内单个样本 U 序列的最大长度；后者是所有样本 U 拼接后再加尾部 padding 的物理长度。

### 14.6 “`SequencePack` 保存一份完整 all-sequence Tensor”

通常不是。主存储只有 `causal_seq` 和 `full_only_seq`；`get_all_seq()` 使用 indices 按需重建完整原始顺序。代码虽然兼容可选的 `all_seq` 字段，但标准构造器不会写入它，见 `cosmos_framework/data/generator/sequence_packing/runtime.py:513-525`。

### 14.7 “attention dropout 会自动传到底层”

当前统一前端没有 dropout 参数。`PackedAttentionMoT` 虽保存 `config.attention_dropout`，但 `two_way_attention` 没有将它传给底层；现有相关模型通常配置为 0。前端签名见 `cosmos_framework/model/attention/frontend.py:45-64`。

## 15. 阅读源码的推荐顺序

1. `cosmos_framework/model/generator/mot/unified_mot.py:501-681`  
   理解 U/G 独立投影、QK Norm、RoPE 和 Q/K/V pack 的构造。

2. `cosmos_framework/data/generator/sequence_packing/runtime.py:29-81`  
   理解 `SequencePackMetadata` 字段。

3. `cosmos_framework/data/generator/sequence_packing/runtime.py:300-366`  
   理解如何从原始 packed sequence 拆出 U/G。

4. `cosmos_framework/data/generator/sequence_packing/runtime.py:459-562`  
   理解 `get_und_seq()`、`get_gen_seq()`、`get_all_seq()` 和 offsets getter。

5. `cosmos_framework/model/generator/mot/attention.py:114-214`  
   阅读 `two_way_attention` 的两次调用。

6. `cosmos_framework/model/attention/frontend.py:45-249`  
   理解统一前端的校验和 backend dispatch。

7. `cosmos_framework/model/attention/backends.py:110-259`  
   理解不同 GPU 上的 backend 优先级和动态兼容性选择。

## 16. 总结

`two_way_attention` 可以用一句话概括：

> 使用同一套 fused attention 后端，分别计算 U 的 causal self-attention，以及 G 对当前样本完整 `[U,G]` 上下文的 non-causal attention。

`SequencePack` 让这个设计能够高效运行：

- `causal_seq` 直接提供第一次 attention 的 U Q/K/V；
- `full_only_seq` 提供第二次 attention 的 G query；
- indices 恢复第二次需要的 `[U,G]` K/V；
- offsets 在一个 packed-varlen kernel 中隔离多个样本；
- 最大长度为 fused kernel 提供 launch metadata；
- 真实 token 数将尾部 padding 与 KV cache、后续计算隔离；
- `is_sharded` 支持 Context Parallel 的 sequence/head 布局转换。

理解 `two_way_attention` 的关键不是记住某个 FlashAttention kernel，而是同时理解三层：

```text
模型语义层：U causal，G full
数据布局层：SequencePack + varlen offsets
执行层：统一 attention frontend + 动态 backend dispatch
```

跨硬件迁移时还要额外确认：底层 API 是通过结构化参数隐式生成 mask，还是要求传入显式 mask Tensor；同名的 varlen 兼容包装并不代表 CUDA 与 NPU 具有完全相同的 mask 能力。
