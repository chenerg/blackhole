# TND 的 Q / KV 怎么拼（UND C1 C2 N1 N2）

配套周报：[weekly_summary.md §3](weekly_summary.md#3-当前-tnd-如何实现利弊)。对照表：[comparison_tests.md §1](comparison_tests.md#1-当前-tnd-如何实现利弊)。

**结论：Query 和 KV 分开组装。** Q 是 GEN 张量上的连续切片（不 gather）；KV 用 `kv_gather_index` 从 packed 双流里挑可见切片再拼起来。可见性靠 gather 编码，不走 Dense Mask。`Ci` / `Ni` 表示整个 causal block，block 内 token 可见性相同。

---

## 两套布局

一条样本、2 个 causal block。`K≥1` 时历史窗口盖住 C1（2 个 block 时 `K=1` 和 `K=2` 的 `lo` 一样）。

```text
packed KV 源序列:   [ UND | C1 C2 | N1 N2 ]
GEN query (full_q): [ C1 C2 | N1 N2 ]
```

Q 按「先全部 CLEAN，再全部 NOISY」排，所以是 `full_q` 的连续 view。KV 还在双流 packed 序里，一个 noisy block 的合法 KV（UND + 历史 clean + 当前 noisy）在原序列上不连续，中间还夹着必须挡住的当前 clean，所以 KV 必须 gather。

---

## 四个 TND segment

`lo(i) = max(0, i - K)`。CLEAN 看到 clean `[lo, i]`（含当前 clean）；NOISY 看到 clean `[lo, i)` + 当前 noisy（**不含当前 clean**）。

| segment | Q（view） | 从 packed 里 gather 的 KV | 明确不看 |
| --- | --- | --- | --- |
| 1 | `C1` | `UND C1` | `C2`、全部 N |
| 2 | `C2` | `UND C1 C2` | 全部 N |
| 3 | `N1` | `UND N1` | 当前答案 `C1`、`C2`、`N2` |
| 4 | `N2` | `UND C1 N2` | 当前答案 `C2`、`N1` |

`N1` 没有更早的 clean，所以只有题目 `UND` + 自己。`N2` 可以看历史 `C1`，但不能看 `C2`。

---

## 组装结果（Q / KV 分开）

```text
packed Q  (连续切片，不复制):

  [ C1 | C2 | N1 | N2 ]
    s1   s2   s3   s4


packed KV (kv_gather_index 拼起来，重叠段会复制):

  [ UND C1 | UND C1 C2 | UND N1 | UND C1 N2 ]
      s1        s2        s3        s4
```

对上之后，每个 segment 内部是 unmasked attention：`C1` 只和自己那一段 `UND C1` 做 softmax，不会扫到后面的 token。隔离靠 `cumulative_seqlen_q` / `cumulative_seqlen_kv` 对齐，不靠 mask。

UND 被复制 4 次；C1 出现 3 次（自己的 query、C2 的历史、N2 的历史）。这就是「同一份 UND/history 按 group 复制」。

---

## 和 Dense Mask 的对应关系

把 gather 展开成 Q×KV 可见性，就是 oracle mask 的非零位置：

```text
         UND  C1  C2  N1  N2
    C1    ✓   ✓   .   .   .
    C2    ✓   ✓   ✓   .   .
    N1    ✓   .   .   ✓   .
    N2    ✓   ✓   .   .   ✓
```

TND 不分配这张表，只把每行的 `✓` 收成那段 KV。`.` 既不算 QK，也不进 softmax。

`per_sample` 对单条样本仍是整张 4×5（再加 token 维），空洞照算。`global` 还要再叠跨样本区域。

---

## 若再加 C3 / N3 且 `K=1`

2 个 block 看不出窗口裁剪。第三个 block、`lo = max(0, i-1)`：

```text
C3 → Q=C3    KV=UND C2 C3          (看不到 C1，出窗)
N3 → Q=N3    KV=UND C2 N3          (看不到 C1；看不到当前答案 C3)
```

| segment | Q | gathered KV | 明确不看 |
| --- | --- | --- | --- |
| CLEAN 3 | `C3` | `UND C2 C3` | `C1`（出窗）、全部 N |
| NOISY 3 | `N3` | `UND C2 N3` | `C1`、当前答案 `C3`、其它 N |

---

## 两条样本 pack 在一起

只是 A 的四段后面再接 B 的四段：

```text
Q:  [ C1A C2A N1A N2A | C1B C2B N1B N2B ]
KV: [ UND_A C1A | … | UND_A C1A N2A | UND_B C1B | … | UND_B C1B N2B ]
```

A 的 Q 不会 gather 到 B 的 token，不用 `same_sample` bit。

---

## 实现落点

- group：[`build_teacher_forcing_block_attention_groups()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/data/generator/sequence_packing/teacher_forcing.py)
- Q view + KV gather index：[`build_teacher_forcing_packed_kv_metadata()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/data/generator/sequence_packing/teacher_forcing.py)
- NPU：`K/V.index_select(0, kv_gather_index)` → packed varlen unmasked attention（Ascend `npu_fusion_attention`，`input_layout="TND"`）
- 预算：gathered KV 262144 token（约 1 GiB bf16 K+V）或 1024 个 TND segment 切 chunk
