# 周报：Cosmos3 DROID Edge teacher-forcing 加速与上限（2026/09/09 – 2026/09/15）

仓库：`cosmos-framework` 分支 `feat/omni-mot-causal-action-training`（相对 origin **超前 18 个 commit，未 push**）。HEAD `6565bf5`。工作区另有未提交的 recipe 默认回退、VAE latent cache、dataset hook。

本周主线：在 week1 已经跑通的整 episode teacher-forcing 上，把 **8 卡 256p packing 的稳态墙钟从 ~57 s 打到 ~23 s**，并扫清 **4 卡 64 GiB 上 256p / 480p × TND / per_sample 的 token 上限**。先读 [§2 三个因素](#2-影响-iter-时间和显存的三个因素) 和 [§3 TND](#3-当前-tnd-如何实现利弊)；对比测试见 [comparison_tests.md](comparison_tests.md)。

机器：`liteserver-8da9-2`，Ascend **910B3 × 8**，每卡 HBM 64 GiB。conda `cosmos-framework`（Python 3.11 + `torch_npu==2.10.0.post4`，CANN 9.1）。数据：`/data5T/Embodied-AI/datasets/droid_plus_lerobot_640x360_20260412`（`info.json` **500** episode / 146133 帧；`split_val_ratio=0.03` → train 485，whole-episode 丢掉 &lt;33 帧后 **484**）。

---

## 目录

1. [一句话总结](#1-一句话总结)
2. [影响 iter 时间和显存的三个因素](#2-影响-iter-时间和显存的三个因素)
3. [当前 TND 如何实现、利弊](#3-当前-tnd-如何实现利弊)
4. [代码改动](#4-代码改动)
5. [加速战役（09/09–09/13）](#5-加速战役0909-0913)
6. [480p 与预存 VAE（09/14）](#6-480p-与预存-vae0914)
7. [4 卡 max-token 上限（09/14–09/15）](#7-4-卡-max-token-上限0914-0915)
8. [当前默认与落地判定](#8-当前默认与落地判定)
9. [环境与长跑坑](#9-环境与长跑坑)
10. [未提交与待办](#10-未提交与待办)
11. [附录](#11-附录)

---

## 1. 一句话总结

8 卡 256p packing 稳态从 **57 s → 23.3 s**（TND + 关 VAE pack + `num_workers=8` + `prefetch_factor=2`）；同卡 VAE pack / sidestream 在 nw=8 之后是负优化，预存 VAE 还能再砍到 **18.4 s** 但不进默认。4 卡钉死 TF `block_size=4 / history=32` 时，**256p 长 T OOM 是 Wan VAE / CANN 3D workspace（跟 T 走），不是 TND**：256p TND 上限 **T=2233 / 47452 tok**，480p TND 能到 **T=681 / 68041 tok**。把 256p 的 `encode_chunk_frames` 从 68 降到 24，上限提到 **52408 tok**，但同 ~5 万 token 仍比 480p 慢约 30%。

---

## 2. 影响 iter 时间和显存的三个因素

一步训练的墙钟，是 DataLoader、Wan VAE、网络 forward+backward 串行或重叠之后的结果；显存则要分 **host RSS** 和 **NPU HBM**。本周所有对照都是在拆这三块。读数字时不要把 AV1 气泡判成网络慢，也不要把 256p 长 T OOM 判成 TND。

```text
磁盘（packed AV1 或预存 .pt）          ← 加载下限，受盘限制
    → DataLoader worker 解码 / 组 batch  ← 主要打 host RSS 和气泡时间
        → packing（token 预算）
            → Wan VAE encode             ← 时间 + NPU HBM（跟帧数 T 走）
                → MoT teacher-forcing    ← 网络本身：激活显存 + 计算下限
```

现行默认 `encode_vision_latents=false`：VAE 在 `training_step` 里和网络 **串行**。DataLoader 可以和下一步重叠（`num_workers` / `prefetch_factor`），VAE 和网络默认不重叠。

| 因素 | 主要打什么 | 本周怎么动 | 还能再抠吗 |
| --- | --- | --- | --- |
| **1. DataLoader** | host RSS、墙钟气泡（min 不变、max/spread 变） | nw=8 藏 AV1；480p `pf=1` | 不解视频、直接读 latent |
| **2. VAE** | 稳态时间 **和** NPU HBM（跟 T） | 关同卡 pack；预存 cache −24% | **可以提前算掉**，省很大一块 |
| **3. 网络** | 计算下限（`min_s`）、transformer 激活 | TND 换 dense mask | kernel 已较满，再抠有限 |

怎么从一张表读出来：

- `min_s` ≈ 网络 +（若在线）VAE 的下限；
- `max − min` / spread ≈ DataLoader 没藏住的解码 / 读盘；
- `npu-smi` 贴 64 GiB 但 PyTorch `reserved` 只有 ~38 GiB → **VAE workspace**，不是 attention 账本；
- host RSS 几百 GiB → DataLoader 的 uint8 视频，不是网络 leak。

### 2.1 DataLoader：主要影响内存和时间，加载受磁盘限制

整 episode 从 `/data5T` 读 packed AV1，解成 uint8 `[C,T,H,W]`，再 packing。**几乎不占 NPU HBM**（workers 不改 65508 MB），但：

1. **Host RSS** = 像素体积 × 卡数 × `num_workers` × `prefetch_factor`（再加 lookahead）。8 卡仍解 AV1 时 ~470–540 GiB；`COSMOS_VAE_SKIP_VIDEO` 只读 `.pt` 后掉到 ~150 GiB。week1 放开 `max_samples_per_batch`、一条 batch 堆多条整视频，主机打到 **1.46 TiB** 被 OOM killer 杀掉——那是 DataLoader，不是 NPU。
2. **墙钟气泡**：解码跟不上时 GPU 空转。同一套 TND 关 pack：`nw=4` **35.19 s**（spread 26 s，min 仍 ~21 s）；`nw=8` **23.30 s**（spread 7.4 s）。min 几乎不变，说明计算下限没动，变的是 AV1 露不露出来。`nw=16` 只再 **−2.1%**，RSS 翻到 898 GiB → drop。
3. **磁盘是加载下限**。无论解 AV1 还是 `torch.load` `.pt`，都走 `/data5T`。8 卡 × 8 worker 并发读 cache 比 packing 单线程更打盘（480p worker-load 27.81 s vs packing 24.51 s，max 42 s）。加 worker / prefetch 只能把盘延迟藏进队列，藏不住就进 IterSpeed。

256p 要用 `nw=8` **`prefetch_factor=2`** 才能把 AV1 藏住；480p 解码已经不是瓶颈，**`pf=1`**，再加 prefetch 不加速（31.67→32.21）还涨 RSS。`lookahead_limit=1` 避免 packing 额外缓存整段视频。

### 2.2 VAE：同时影响时间和显存；这块时间可以提前计算、省掉很大一部分

Wan VAE 把 uint8 编成 latent。现行在 `get_data_and_condition` / `training_step` 同步 encode。

1. **时间**。4 卡 profile 里 Conv3DV2 **2.79 s**；开 pack 后 8 卡仍有 4.27 s。关 pack 时它和网络串在同一步里：256p 的「~21 s 下限」其实含同卡 VAE。预存 `.pt` 之后 24.21→**18.36 s（−24%）**，min 21.2→15.0——地板掉下来的就是 encode + uint8 H2D。480p 关 pack 31.67 → 指纹 cache **23.71（−25%）**。回本：256p 预计算 8.9 min / 约 91 步；480p 5.5 min / 约 57 步。
2. **NPU HBM**。CANN 3D conv / 时间 cache 的 workspace **跟帧数 T 走**，经常在 PyTorch `memory_allocated` 外面。同样 ~4.8 万 pre-TF token：256p（K=80）T=2257，480p（K=391）T=477，差约 **4.7× 帧**。256p TND 在 T=2257 **step 1 前 OOM**（reserved ~38 GiB，`npu-smi` 64 GiB）；480p 同 token 还能加到 6.8 万。把 `encode_chunk_frames.256` 68→24，上限从 47452 提到 52408，换的是更多次 3D conv、稳态更慢。
3. **可以提前算**。离线 8 卡把 train episode 编成 `.pt`（256p 2.06 GiB / 484 条；480p 6.55 GiB / 404 条），训练跳过 encode。这是后续还能再抠掉的最大一块 **可离线** 时间。注意：指纹 cache 仍解 AV1，RSS 和 spread 还在；episode-id + skip-video 才把 RSS 打下来，但 8 卡抢盘墙钟不一定更快。

同卡 VAE pack + sidestream 本想把 encode 和下一步重叠。`nw=8` 之后是负优化（256p 24.21 vs 23.30；480p 34.11 vs 31.67），工作区已关。真要省 VAE 时间，走预计算，不要在训练卡上边训边编。

### 2.3 网络本身的设计

DataLoader 藏住、VAE 预存之后，剩下的就是 MoT + Scheme B teacher forcing。这是 **`min_s` 的下限**，也是 transformer 激活显存。

- **双流展开**：clean + noisy 同一次 forward，packed ≈ `UND + 2×(vis+action)`。激活跟展开后长度走，不是原始视频帧的线性。
- **GEN attention**：`tnd` / `per_sample` / `global`。TND 不建 Dense Mask，只 gather 每个 block 看得见的 KV 再做 packed 无 mask attention（本周默认，墙钟 −11%～−22%），下一节展开。
- **并行与 dtype**：加卡主要摊参数、省 HBM，不加速（week1：32 档 2/4/8 卡都是 ~10 s）。FSDP2 mixed precision **不省激活**（200-step 57.15 vs 56.16 s，HBM 63.7→64.0 GiB）。
- **几何**：`block_size` S、`history_blocks` K。max-token 扫参钉死 S=4 K=32，是最吃显存的一档。
- **token 预算**：`max_sequence_length=48000` 把每步填满。256p 同 cap 比 480p 快（格子少），但 T 长 4.7×，VAE 先爆。

4 卡无 pack profile：FA（TND）3.06 s、MatMulV3 3.44 s（cube ~97%）、fused RMS 0.47 s。网络 kernel 已经比较满，再抠单算子收益有限。8 卡 sidestream 之后 HCCL 非重叠仍 6.6 s，属于通信/调度，不是 TND 本身。

---

## 3. 当前 TND 如何实现、利弊

**一句话：** TND 不建 Dense Mask，只把每个 causal block 真正看得见的 KV gather 成变长段，用 packed 无 mask 融合 attention 一次算完。

本周对照默认的 GEN attention 是 **`teacher_forcing_dense_mode=tnd`**。可见性仍是 Scheme B teacher forcing，只换 kernel：Dense Mask + masked SDPA → 按 block gather + packed varlen 无 mask attention（Ascend 上是 `npu_fusion_attention` 的 TND 布局）。UND 的 causal attention 三种模式都走 packed varlen，不受这个开关影响。

更完整的 Scheme B 设计说明见仓库 [`docs/omni_mot_causal_training_implementation_zh.md`](../../../../workspace/chenzhi/cosmos-framework/docs/omni_mot_causal_training_implementation_zh.md) §7.9。下面写的是 **当前代码实际在跑的路径**。

### 3.1 它解决什么问题

原 OmniMoT 一次只吃被加噪的视频。causal teacher forcing（Scheme B）同一次 forward 同时放进：

```text
UND（题目/文本） + 完整 clean 视频（答案/历史） + 完整 noisy 视频（要预测的内容）
```

视频按 VAE latent 切成 causal block（`S` 个 latent / block，本周默认 Uniform `S∈[1,4]`；max-token 扫参钉死 `S=4`）。noisy block `i` 可以看题目和有限段更早的 clean 历史（`K` 个 block，默认 Uniform `K∈[1,32]`），以及当前 noisy block 内部；**不能看当前 clean 答案，也不能看未来**。否则训练会抄答案，推理没有这份 clean 就会垮。

Naive 做法是给 GEN query 做一张 `[Q, KV]` bool mask，一次 masked SDPA。packing 到 4.8 万 pre-TF token 之后，mask 和被挡住的 QK 都按二次增长，HBM 和 FA launch 都顶不住。TND 的目标就是：**可见性仍按 block 规则，但不分配 Dense Mask，也不算被挡住的 QK。**

### 3.2 可见性（三种 kernel 必须一致）

设 query 在 block `i`，历史左边界 `lo = max(0, i - history_blocks)`。

| Query | 可以读的视觉 KV | 禁止 |
| --- | --- | --- |
| clean `Ci` | UND + clean `[lo, i]`（含当前 clean） | 所有 noisy、窗口外 / 未来 clean |
| noisy `Ni` | UND + clean `[lo, i)` + 当前 noisy `Ni` | **当前 clean `Ci`**、未来、其他 noisy block |

多样本 packing 时样本之间完全隔离。实现：[`build_teacher_forcing_block_attention_groups()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L545)（TND）与 [`build_dense_teacher_forcing_gen_mask()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/data/generator/sequence_packing/teacher_forcing.py) / [`build_per_sample_teacher_forcing_gen_masks()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/data/generator/sequence_packing/teacher_forcing.py)（oracle）。CPU 测试要求 TND 的 output / QKV 梯度对齐 dense mask。

### 3.3 三种 GEN kernel

开关：`model.config.teacher_forcing_dense_mode`，默认 **`tnd`**（`244068c`）。只换 GEN attention；Norm、QKV、output proj、MLP 仍对完整 packing 跑一次。

```text
tnd:         每个 (sample, block, stream) 是一个 varlen segment
             Q_j 只对 gather 出的同样本可见 KV_j 做 unmasked attention
             多样本拼进同一次（或按 segment / KV 预算切开的几次）TND 调用

per_sample:  GEN_A × ALL_A -> masked SDPA A
             GEN_B × ALL_B -> masked SDPA B
             不跨样本，但仍算样本内部被历史窗口挡住的空洞

global:      Q=[GEN_A|GEN_B] × KV=[ALL_A|ALL_B] -> 一张 Dense Mask，一次 masked SDPA
             跨样本 QK 也算，靠 same_sample bit 遮掉；长 packing 上 mask 本身就是 HBM 风险
```

主计算量（`Ui` = 样本 i 的 UND token，`Vi` = 原始视觉 token，`P` ≈ 一个 block 的 GEN token）：

```text
global:     2 * sum(Vi) * [sum(Ui) + 2*sum(Vi)]
per_sample: sum(2*Vi * [Ui + 2*Vi])
tnd:        近似 sum(2*Vi * [Ui + (K+1)*P])   ← 再去掉样本内被历史挡住的空洞
```

`global` 不要当 DROID 48k 默认。本周对照只用 `tnd` vs `per_sample`。

### 3.4 当前 TND 实现（block-gather + packed varlen）

提交 `2a451d3`。数据流：

```text
TeacherForcingLayout（每条 sample 的 block span / S / K）
        │
        ▼
build_teacher_forcing_block_attention_groups
  每个 CLEAN / NOISY block → 一个 group
  query = 该 block 在拼接 GEN 张量上的 [start, end)
  kv_slices = 2～3 段 packed 序列上的连续区间：
      CLEAN i : (UND) + (clean[lo .. i])
      NOISY i : (UND) + (clean[lo .. i) 若非空) + (noisy i)
        │
        ▼
build_teacher_forcing_packed_kv_metadata
  kv_gather_index     : 把上述 slice 展成一维 gather 下标（重叠的 UND/history 会复制）
  cumulative_seqlen_q : 各 group 的 query 前缀和
  cumulative_seqlen_kv: 各 group 的 gathered KV 前缀和
        │
        ▼
teacher_forcing_block_gather_attention
  NPU/CUDA bf16|fp16|fp32:
      gathered_K/V = K/V.index_select(0, kv_gather_index)
      按预算切 chunk 后调用 packed varlen unmasked attention
      （Ascend 上是 npu_fusion_attention 的 TND 布局：变长、无 mask）
  CPU / fp64 测试:
      按 (query_len, kv_len) 分桶，stack 成 batched unmasked SDPA，对齐 dense oracle
```

关键约束与实现细节：

- Query groups 必须覆盖全部 GEN query、不重叠（[`_validate_block_attention_groups`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/model/generator/mot/teacher_forcing_attention.py#L133)）。先发完一条 sample 的全部 CLEAN group，再发 NOISY，拼接 GEN 是一段连续分区。
- Gather 后的 attention **不再带 mask**：看不见的 token 根本不在 KV 里。当前 clean 对 noisy 不可见，是因为 noisy 的 `kv_slices` 停在 `clean_start + gen_start`，不含当前 clean。
- Gather 会复制：同一段 UND / clean history 会被后面每个 block 再 gather 一次。代码按 **262144 gathered KV token**（约 1 GiB bf16 K+V，8 head × 128 dim）和 **1024 个 TND segment** 切 chunk，避免一次调用撑爆（[`_PACKED_KV_TOKEN_BUDGET`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/model/generator/mot/teacher_forcing_attention.py#L16)）。
- GQA：query head 多于 KV head 时，gather 之后 `repeat_interleave` 扩 KV。
- 分发入口：[`teacher_forcing_attention()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/model/generator/mot/attention.py#L289)。UND 仍是独立的 packed causal；只有 `get_full_only_seq` 的 GEN query 走 TND。
- 不要求与 dense SDPA **bitwise** 一致（tiling / softmax reduction / BF16 累加顺序不同），但 forward 和 QKV 梯度必须对照 oracle。CPU 回归覆盖 `S=1..4`、`K=1/32`、首 block 无 history、尾部 partial、mixed-length packing、noisy 不读当前 clean。

这 **不是** Scheme A（先 clean forward 存逐层可微 KV，再让 noisy 读 cache）。也不是「多次 attention 再用 LSE 合并」。当前生产路径仍是 Scheme B 的一次 joint forward，只是 GEN softmax 作用在 gather 出来的可见 KV 上。

### 3.5 利

| 点 | 说明 |
| --- | --- |
| 不算空洞 QK | 历史窗口外、跨样本、noisy 对当前 clean，都不进 softmax。相对 `per_sample` 再少一层样本内空洞。 |
| 不分配 Dense Mask | `global` 的 `[Q, KV]` bool 在 48k packing 上本身就是 HBM 炸弹；`per_sample` 每条一张较小 mask，480p 扫到 6.4 万时一次申请 **11.30 GiB**，step 1 进不去。TND 没有这张表。 |
| 更少 FA launch | 多样本 / 多 block 拼进少数几次 varlen 调用。4 卡 profile：FA 从旧 dense 的 5.66 s / 27% 降到 TND 3.06 s / 18%。 |
| 墙钟 | 8 卡 nw=4 15-step：**48.67 vs 54.81 s（−11.2%）**；nw=8 关 pack 21-step：**23.30 vs 29.85 s（−21.9%）**。 |
| 480p token 上限更高 | 4 卡钉死 S=4 K=32：TND **68041** vs per_sample **55796**。transformer 反传能多撑约 1.2 万 pre-TF token。 |
| 语义可测 | 与 dense mask 同一套 layout；CPU 路径故意走 SDPA 对齐 oracle，坏规则（`j <= i` 泄漏当前 clean）测得出来。 |

### 3.6 弊

| 点 | 说明 |
| --- | --- |
| Gather 复制 KV | 每个 block 都拷一份 UND + 滑动 clean 窗。历史越长、block 越多，gathered KV 相对「不复制的 packed KV」放大得越厉害，所以才有 262144 token chunk。这是用带宽换掉 mask / 空洞 FLOP。 |
| 临时 buffer | `index_select` 出的 K/V 和 varlen workspace 是额外 HBM，不在 PyTorch `memory_allocated` 的 transformer 账本里时，看起来像「TND 更吃显存」。8 卡短测 TND 峰值经常贴 655xx，per_sample 有时更低（15-step 63557 vs 65531）。 |
| 浮点不等价 | fused TND 与 masked SDPA 的 softmax 归约不同，只保证合理误差，不保证 bitwise。不能拿单 step loss 逐位对。 |
| 救不了 VAE | 256p 同样 ~4.8 万 token 对应 T=2257，是 480p T=477 的 ~4.7× 帧。OOM 在 Wan VAE / CANN 3D workspace（跟 **T** 走）。TND 在 256p 的 token 上限（47452）反而低于 per_sample（49972），因为 TND 还没碰到 attention 墙，VAE 先炸。**不要把 256p 长 T OOM 判成 TND gather 不够用。** |
| 实现复杂 | group 构造、gather index、cu_seqlen、chunk、GQA expand、CPU 对照路径都要保持与 mask 一致。chunk 切错或 group 重叠会静默错 attention。 |
| 仍是 Scheme B | clean/noisy 同一次 forward，展开后 packed ≈ `UND + 2×(vis+action)`。激活仍随双流涨。真正按 block 复用 clean KV 的 Scheme A 还没做。 |
| Host 侧 metadata | `kv_gather_index` 在 packing 时建，长 episode × 多 block 是一段不小的 int64 索引；相对 NPU 不是主因，但不是零。 |

### 3.7 怎么用本周数字读利弊

- **要墙钟、要 480p 更长序列：用 TND。** 这是 Edge 默认。
- **要对齐旧 dense、或怀疑 gather 梯度：切 `per_sample`。** `EXTRA_TAIL_OVERRIDES="model.config.teacher_forcing_dense_mode=per_sample"`。
- **256p 想加 token：先封 T / 降 `encode_chunk_frames`，再谈 TND。** chunk=24 把 256p TND 上限从 47452 提到 52408，同 ~5 万 token 仍比 480p TND 慢约 30%。
- **不要开 `global`。** 本周没测，设计上 mask 二次增长。

---

## 4. 代码改动

### 4.1 已提交（09/09–09/11）

| 日期 | 提交 | 说明 |
|------|------|------|
| 9/9 | `d1053a9` | DROID Edge 默认改 **256p**，压 host RSS knobs（lookahead / prefetch）。 |
| 9/9 | `a7740fc` | teacher forcing 改成 **per-sample block geometry** 的 joint topology（不再全局一份 S/K）。 |
| 9/9 | `0d70077` / `c833542` | VFM **FSDP2 mixed precision** 合入并默认打开（fp32 master + bf16 compute）。 |
| 9/9 | `cbb2bbd` | 整 episode 改 **token packing**（`max_sequence_length=48000`，`max_samples_per_batch=null`）；NPU profiling 入口。 |
| 9/9 | `99ff341` | masked-SDPA 跳过重复的 device mask sync。 |
| 9/10 | `2a451d3` | GEN dense-mask SDPA 换成 **block-span TND gather**（packed varlen）。 |
| 9/10 | `2989b9b` | packing 路径去掉 per-token `.item()`。 |
| 9/10 | `f2b526e` | MoT RMSNorm 走 Ascend `npu_rms_norm`。 |
| 9/10 | `4bd909f` | `PackingDataLoader` 可在 packing 线程预编 Wan VAE（`encode_vision_latents`）。 |
| 9/11 | `244068c` | `teacher_forcing_dense_mode` 三选一：`tnd` / `per_sample` / `global`。 |
| 9/11 | `4ea988a` | wandb 默认 `online`，失败回落 `offline`。 |
| 9/11 | `6565bf5` | VAE encode 叠到本卡 **NPU side stream**（避开 HCCL default stream）。 |

实验目录：`experiments/` 一 run 一目录（`README.md` + `per_iter.tsv` + `meta.json`）。战役汇总：[`experiments/20260911_droid_edge_tf_opt_campaign.md`](../../../../workspace/chenzhi/cosmos-framework/experiments/20260911_droid_edge_tf_opt_campaign.md)。

### 4.2 未提交（工作区，09/13 之后的结论已写进 recipe 注释）

- **Edge 默认回退 VAE pack**：`encode_vision_latents=false`，`encoded_prefetch_depth=0`。同卡 pack+sidestream 在 nw=8 之后比串行 encode 慢（256p 24.21 vs 23.30；480p 34.11 vs 31.67），还可能在 FSDP backward OOM。
- **`num_workers=8`**，256p `prefetch_factor=2`；480p 应改 `pf=1`。
- `joint_dataloader.py`：`COSMOS_VAE_LATENT_CACHE` episode-id 读 `.pt`、cache fill、skip-video、bf16 encode dtype、side-stream 开关。
- dataset：whole-episode hook / transform 补齐；`cosmos3-action-dataset` skill；packing / SFT 单测。
- `AGENTS.md`：`LD_LIBRARY_PATH` 要 **先 prepend `$ENV/lib`**（torchcodec `GLIBCXX_3.4.32`）；256p 长 T OOM 归因。

---

## 5. 加速战役（09/09–09/13）

公共配方（除非某档另注）：8 卡 FSDP、整 episode `chunk_length=-1`、`max_sequence_length=48000`、256p（K=80）、selective AC、FSDP2 MP、TF S∈[1,4] K∈[1,32]、`grad_accum_iter=1`。主指标 **`avg_skip1`** = Rank0 IterSpeed 丢掉 iter 1。8 卡「21-step」= 20 个稳态点。**带 profiler dump 的 iter 不当墙钟。**

### 5.1 墙钟怎么掉下来的

```text
persample 200-step（09-09 MP）              57.15 s
  → TND gather 21-step（09-10）             51.46 s    −10%
  → packing .item() 向量化                  51.13 s    可忽略
  → fused npu_rms_norm                      49.86 s    −2.5%
  → VAE pack depth=1（nw=4 pf=1）           40.37 s    −19%
  → pf=2 + sidestream（09-11 终测 nw=4）    28.17 s    −30% vs pack
  → nw=8 + pack+stream（09-13）             24.21 s
  → TND 关 pack nw=8（现行方向）            23.30 s
  → 预存 VAE cache（可选）                  18.36 s
```

09-11 当时把 VAE pack + sidestream 写成默认，是因为 nw=4 时 Preparing 19.88 s、HCCL 非重叠被同卡 encode 顶到 7.5 s。09-13 把 worker 加到 8 之后，AV1 已经藏住，在线 pack 不再比「TND 同步 encode」快。

### 5.2 关键对照（详见 comparison_tests）

| 对照 | 配置差异 | 结果 | 判定 |
| --- | --- | --- | --- |
| FSDP2 MP vs 纯 bf16 | 256p packing 200 vs 189 step | 57.15 vs 56.16 s；HBM 63.7→64.0 GiB；loss 对齐 | MP **不省显存**，作数值路径保留 |
| TND vs per_sample（15-step，nw=4） | 只改 `teacher_forcing_dense_mode` | 48.67 vs 54.81 s（**−11.2%**） | **keep TND** |
| TND vs per_sample（21-step，nw=8 关 pack） | 同上 | 23.30 vs 29.85 s（**−21.9%**） | 再确认 |
| VAE pack 无 sidestream vs 关 pack | nw=8 | 28.63 vs 23.30 s（**+5.33 s**） | pack 必须配 stream，否则负优化 |
| pack+sidestream vs 关 pack | nw=8 | 24.21 vs 23.30 s | **关 pack 更快** |
| nw=8 vs nw=4（关 pack） | 只改 workers | 23.30 vs 35.19 s；spread 7.4 vs 26.3 s | **keep nw=8**（藏 AV1） |
| nw=16 vs nw=8 | 只改 workers | 23.71 vs 24.21 s（−2.1%），RSS 898 vs 499 GiB | **drop** |
| 预存 VAE vs 在线 pack+stream | `COSMOS_VAE_LATENT_CACHE` | 18.36 vs 24.21 s（−24%） | **可选 pipeline**，不进默认 |

4 卡 profile（09-11，TND、**未**开 VAE pack，iter 6）：Preparing 19.88 s、Computing 16.91 s、FA 3.06 s、HCCL 非重叠 0.07 s。8 卡现行 pack+stream profile（09-13 r2）：Preparing **0.002 s**，HCCL 非重叠仍 **6.62 s**——sidestream 没有把 8 卡 HCCL 压回去。

### 5.3 10k 正式长跑（未跑满）

同一套 8 卡 256p packing + TND + AI_CPU，`max_iter=10000`，wandb offline：

| run | 结果 |
| --- | --- |
| `20260910_155517` r1 | rank0 **SIGKILL @ iter 93**，无 dmesg OOM，NPU 正常；last loss 1.48 |
| `20260910_171645` r2 | setsid 重开，**HCCL 默认 16666 bind 冲突** 失败 |
| `20260910_235029` r3 | `HCCL_NPU_SOCKET_PORT_RANGE=35100-35200`，跑到 iter 120 后 SIGINT；last loss 1.45 |

多卡启动必须带端口区间。HCCL 保持默认 **AI_CPU**，不要 `HCCL_OP_EXPANSION_MODE=AIV`（week1 910C 上 AIV 也救不了大 all_gather）。

---

## 6. 480p 与预存 VAE（09/14）

基线：8 卡、TND、token cap 48000、`nw=8`、开 pack+sidestream、`pf=1`，11-step avg **34.11 s**。不落盘 cache。

| 档 | avg_skip1 | 判定 |
| --- | ---: | --- |
| 关 VAE pack（`pf=1`） | **31.67 s（−7.2%）** | **keep**：480p 重叠也是负优化；HBM 63959、RSS 379 都更松 |
| `encode_chunk_frames` 24→8 | OOM step 0 | **drop** |
| packing 像素 bf16 | 33.52 s | **drop**（噪声） |
| `prefetch_factor=2`（仍开 pack） | 32.06 s | 略快，RSS 418→482 |
| 关 pack + pf=2 | 32.21 s | **不叠加**：比关 pack pf=1 还慢 0.5 s |

**256p 用 `pf=2`，480p 用 `pf=1`。** 480p 关 pack 后再加 prefetch 不加速，解码已经不是瓶颈，VAE 长包才是。480 / 256 各自最好 pf = 31.67 / 23.30 ≈ **1.36×**。

预存 VAE（packing 读 `.pt`，训练仍解 AV1）：

| res | 路径 | avg | RSS GiB |
| --- | --- | ---: | ---: |
| 256 | 指纹 cache pf=2 | **19.22**（11-step）/ 18.36（21-step） | ~478 |
| 480 | 指纹 cache pf=2 | **23.71** | 471 |
| 480 | episode-id 不解视频 | 24.51 | **153** |
| 256 | episode-id packing 读 `.pt` | 20.93 | 152 |
| 256 / 480 | worker 并行读 + bf16 | 20.07 / 27.81 | 89 / 151 |

结论：指纹 cache 墙钟最好，但 host 仍解 AV1（RSS ~470 GiB）。episode-id 把 RSS 打到 ~150 GiB，稳态几乎不更快（8 卡 × 8 worker 抢 `/data5T` 更差）。40-step 复核：256p 指纹 19.63 vs epid 19.81；480p 指纹 **HBM OOM**（仍解视频 RSS 567），epid 跑满 27.95 s。packing 线程 load vs worker load 在 4 卡 40-step 上几乎打平（256p 19.20 vs 18.93；480p 26.16 vs 25.59）。

预计算：256p 8 卡 532.9 s / 2.06 GiB / 484 条；480p 330 s / 6.55 GiB / 404 条。相对在线关 pack，480p 回本约 57 步。

---

## 7. 4 卡 max-token 上限（09/14–09/15）

phy **0–3**，FSDP 4 卡，**一条样本填满预算**（`max_samples_per_batch=1`），VAE 在 `training_step`，TF **钉死 block_size=4、history_blocks=32**，`COSMOS_FORCE_VIDEO_FRAMES=T=1+4N`。pre-TF = `500 + K + N*(K+4)`；K=80（256p）、K=391（480p）。每档 5 step。token 间隔 ≤ 1000 停搜。

### 7.1 最大不爆

| 分辨率 | 模式 | N | T | pre-TF | iter1 / s | 2–5 均 / s | 下一档 OOM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256p | TND | 558 | 2233 | **47452** | 99.86 | 43.61 | N=564 T=2257 tok=47956，**step1 前**，VAE/CANN 3D |
| 256p | per_sample | 588 | 2353 | **49972** | 133.51 | 82.89 | N=600 T=2401 tok=50980，step1 过、再申请 7.79 GiB |
| 480p | TND | 170 | 681 | **68041** | 116.21 | 72.82 | N=172 T=689 tok=68831，step1 过、backward 再申请 4.89 GiB |
| 480p | per_sample | 139 | 557 | **55796** | 150.82 | 100.79 | N=141 T=565 tok=56586 |

```
480p TND (68041)  >  480p per_sample (55796)  >  256p per_sample (49972)  >  256p TND (47452)
```

同样 ~4.8 万 token：256p T=2257，480p T=477（约 **4.7×** 帧）。256p TND 在 47956 **step 1 前 OOM**：PyTorch reserved ~38 GiB，`npu-smi` 已经 64 GiB。差的显存在 CANN / VAE 3D conv / 时间 cache，不在 transformer token 账本里。**不要把这次 OOM 判成 TND gather 不够用，也不要在 256p 上只加 `max_sequence_length` 却不封 T。**

480p TND 从 49000 一路加到 64000 都过，64000–70000 二分停在 N=170。per_sample 稠密 mask 在 64000 档一次申请 **11.30 GiB**，step1 都进不去。

### 7.2 减小 `encode_chunk_frames.256`

只改 VAE 时间窗（必须是 4 的倍数）。T=2233 上 chunk 68 / 32 / 24 都能跑；chunk=24 峰值 HBM 从 65512 降到 **63015**，稳态时间 43.6→45.4 s。

chunk=24 再搜上限：最大 ok **N=617 T=2469 tok=52408**（相对 chunk=68 的 47452 **+4956**）。同 ~5 万 tokens：256p 2–5 步 **49.2 s**，480p **37.7 s**（256p 慢约 **30%**）。480p 上限仍是 6.8 万（T 只有 681）。

---

## 8. 当前默认与落地判定

工作区 Edge recipe（未 commit）相对 HEAD `6565bf5` 的变化：

| Knob | HEAD（09-11 终测） | 工作区（09-15） | 原因 |
| --- | --- | --- | --- |
| `encode_vision_latents` | true | **false** | nw=8 后同卡 pack 负优化 |
| `encoded_prefetch_depth` | 1 | **0** | 关 pack 强制 0 |
| `num_workers` | 4 | **8** | 藏 AV1；nw=16 drop |
| `prefetch_factor` | 2 | **2**（256p）；480p 改 1 | 480p pf=2 不加速还涨 RSS |
| `teacher_forcing_dense_mode` | tnd | tnd | 8 卡 −11%～−22% |
| `max_sequence_length` | 48000 | 48000 | 256p 对应 T≈2257，贴 VAE 上限 |
| `resolution` | 256 | 256 | 同 token 比 480p 快、HBM 松（短测）；长 T 会被 VAE 卡住 |
| HCCL | 默认 AI_CPU | 同左 + `HCCL_NPU_SOCKET_PORT_RANGE` | 16666 bind |
| 预存 VAE | 无 | env `COSMOS_VAE_LATENT_CACHE` | 可选，不进默认 |

关掉 pack：`EXTRA_TAIL_OVERRIDES="dataloader_train.encode_vision_latents=false"`。切 per_sample：`model.config.teacher_forcing_dense_mode=per_sample`。切 480p 时再加 `prefetch_factor=1`。

明确没采用：给 VAE 单独占一张卡；`num_workers=16`；`HCCL=AIV`；packing 像素 bf16；`encode_chunk_frames=8`；把 encode 串到 optimizer 之后（Adam ~19 ms 装不下 ~3 s VAE）。

---

## 9. 环境与长跑坑

- **torchcodec / GLIBCXX_3.4.32**：CANN `set_env.sh` 把 `/lib` 放最前，worker 绑到系统 `libstdc++`（最高 GLIBCXX_3.4.30）。必须 `export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH`，**不要**清掉 CANN。09-13 8 卡 profile r1 因此挂在 `_prewarm_dataloaders`。
- **HCCL**：默认 AI_CPU。多卡设 `HCCL_NPU_SOCKET_PORT_RANGE=35100-35200`。10k r1 SIGKILL / r2 端口冲突。
- **host RSS**：uint8 整 episode × ranks × workers × prefetch。指纹 cache 仍解 AV1 → 8 卡 ~470–540 GiB；episode-id skip-video → ~150 GiB。week1 的 packing 主机 OOM（1.46 TiB）是 `mspb` 放开后堆多条视频，不是 NPU leak。
- 实验记录不要只丢在 `bench_max_episode_blocks/`：一律进 `experiments/<run_id>/`。

---

## 10. 未提交与待办

工作区脏文件（相对 `6565bf5`）：

- recipe：DROID / LIBERO / RoboTwin Edge 关 VAE pack、nw=8
- `joint_dataloader.py` VAE cache / skip-video / dtype / sidestream 开关（+516 行）
- dataset whole-episode hook、transforms、packing / SFT 单测
- skills：`cosmos3-action-dataset`、post-training / env-troubleshoot 补丁
- `AGENTS.md` GLIBCXX + 256p VAE OOM
- `experiments/` 本周全部 run 目录与战役 md（未进 git）

待办：

1. 18 个训练相关 commit **还没 push**；09-13 之后的默认回退还没 commit。
2. 8 卡 10k 正式长跑没跑满（SIGKILL / 端口 / 手动停）。要用 AI_CPU + 端口区间 + prepend `$ENV/lib` 再开。
3. DataLoader 直接读 latent、不解 AV1（指纹 cache 的 spread 还在 8 s；epid 墙钟没赢）。
4. 查清 8 卡 sidestream 后 HCCL 非重叠仍 6.6 s。
5. 256p 若维持 packing 48000，等于顶在 VAE T=2233 边上；要么封 T / 降 `encode_chunk_frames`，要么接受 480p 同 token 更安全。
6. 预存 VAE 作可选 pipeline 还没写成 launcher 开关（现在靠 env）。

---

## 11. 附录

| 内容 | 路径 |
| --- | --- |
| 对比测试全文（配置 + 结果表） | [comparison_tests.md](comparison_tests.md) |
| 实验目录 / 索引 | `cosmos-framework/experiments/INDEX.md` |
| 09-11 战役 | `experiments/20260911_droid_edge_tf_opt_campaign.md` |
| 09-13 时间测试 / ablation | `experiments/20260913_droid_edge_tf_time_tests.md`、`…_nw8_ablation.md` |
| 09-14 480p opt | `experiments/20260914_480p_opt_1235.md` |
| 4 卡 max-token | `experiments/20260915_4npu_max_tokens_compare.md` |
| 256p 长 T OOM 归因 | `experiments/20260914_4npu_max_tokens_oom.md` |
| week1（09/02–09/08） | [`../week1/weekly_summary.md`](../week1/weekly_summary.md) |

启动脚本：`cosmos/cookbooks/cosmos3/generator/action/finetune/launch_sft_action_policy_droid_edge_local.sh`（`model=mot_causal_fsdp`）。
