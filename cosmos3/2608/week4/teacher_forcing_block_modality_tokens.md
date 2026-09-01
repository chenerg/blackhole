# Teacher-forcing block：原始模态大小 vs tokens

假设 Video / Action **无限长**（没有末 block、没有长度上限）。Wan VAE `cf = 4`：原帧 0 → latent 0，之后每 4 个原帧 → 1 个 latent。

- 每 latent 帧 Video tokens = **K**，且 **K ≡ 0 (mod 64)**（`H×W` 空间 token）。
- 每个 action step = **1 token**。
- Action step `j` → latent `ceil(j / 4)` → `block_id = latent // block_size`。实现：[teacher_forcing.py](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L120)（`assign_action_steps_to_latent_frames`）、[同文件](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L243)（按 block 交错 `[V_b | A_b]`）。

令 `B = block_size`。

## 当前方案

block 0 含 latent 0（1 个原帧 / 1 个 action），所以 Action 和原帧比满 block 少 3；Video 的 latent 帧数和 tokens 两边相同。

| | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| block 0 | `B` latent 帧（`4B − 3` 原帧） | `B·K` | `4B − 3` steps | `4B − 3` |
| block n ≥ 1 | `B` latent 帧（`4B` 原帧） | `B·K` | `4B` steps | `4B` |

### block_size = 4

| block | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| 0 | 4 latent 帧（13 原帧） | 4K | 13 steps | 13 |
| n ≥ 1 | 4 latent 帧（16 原帧） | 4K | 16 steps | 16 |

### block_size = 8

| block | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| 0 | 8 latent 帧（29 原帧） | 8K | 29 steps | 29 |
| n ≥ 1 | 8 latent 帧（32 原帧） | 8K | 32 steps | 32 |

## 新方案：前置 4 个原始数据，编码后丢掉第一个 latent / action

对 **Video 和 Action 都在序列最前面各添加 4 个原始数据**（pad 帧 / pad action），再走 VAE + action→latent 映射。编码结果里：

- 第 1 个 latent frame 只来自第 1 个 pad 帧（VAE 的 1 帧特例）
- 第 1 个 action token 落在该 latent 上

**删掉这第一个 latent frame 和第一个 action token**，用剩下的无限长 token 流按 `block_size` 组 block。剩下的每个 latent 都对应 **4 原帧 + 4 action**，因此 **block 0 与之后每个 block 的原始大小和 tokens 完全一致**。

| | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| block 0 | `B` latent 帧（`4B` 原帧） | `B·K` | `4B` steps | `4B` |
| block n ≥ 1 | `B` latent 帧（`4B` 原帧） | `B·K` | `4B` steps | `4B` |

### block_size = 4

| block | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| 0 | 4 latent 帧（16 原帧） | 4K | 16 steps | 16 |
| n ≥ 1 | 4 latent 帧（16 原帧） | 4K | 16 steps | 16 |

### block_size = 8

| block | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| 0 | 8 latent 帧（32 原帧） | 8K | 32 steps | 32 |
| n ≥ 1 | 8 latent 帧（32 原帧） | 8K | 32 steps | 32 |

与当前方案相比：block 0 的 Action / 原帧从 `4B − 3` 变成 `4B`，和后续 block 对齐；Video tokens 仍是 `B·K`，没有变化。
