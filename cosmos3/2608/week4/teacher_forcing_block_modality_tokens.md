# Teacher-forcing block：原始模态大小 vs tokens

示意 clip：Video 原像素 **33 帧**（Wan VAE `4N+1`，N=8）→ **9 latent 帧**；Action 原序列 **33 steps**。

- 每 latent 帧 Video tokens = **K**，且 **K ≡ 0 (mod 64)**（`H×W` 空间 token）。
- 每个 action step = **1 token**。
- Action step `j` → latent `ceil(j / 4)` → `block_id = latent // block_size`。实现：[teacher_forcing.py](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L120)（`assign_action_steps_to_latent_frames`）、[同文件](../cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L243)（按 block 交错 `[V_b | A_b]`）。

满 block（`B = block_size`）：Video = `B` latent 帧 / `B·K` tokens，Action = `4B` steps / `4B` tokens。block 0 含 latent 0（只有 1 个原帧 / 1 个 action），所以 Action = `4B − 3`；Video 的 latent 帧数和 tokens 与满 block 相同。

## block_size = 4

| block | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| 0 | 4 latent 帧（13 原帧） | 4K | 13 steps | 13 |
| 1（满） | 4 latent 帧（16 原帧） | 4K | 16 steps | 16 |
| 2（末） | 1 latent 帧（4 原帧） | K | 4 steps | 4 |
| 合计 | 9 latent 帧（33 原帧） | 9K | 33 steps | 33 |

## block_size = 8

| block | Video 原始 | Video tokens | Action 原始 | Action tokens |
| --- | --- | --- | --- | --- |
| 0 | 8 latent 帧（29 原帧） | 8K | 29 steps | 29 |
| 1（末） | 1 latent 帧（4 原帧） | K | 4 steps | 4 |
| 合计 | 9 latent 帧（33 原帧） | 9K | 33 steps | 33 |

满 `block_size=8` 应是 Video 8 latent / `8K` tokens、Action 32 steps / 32 tokens。本例只有 9 latent，所以没有下一个满 block；block 0 的 Action 是 `4×8 − 3 = 29`。
