# 本周对比测试：配置与结果（2026/09/09 – 2026/09/15）

配套周报：[weekly_summary.md](weekly_summary.md)（§2 三个因素，§3 TND 实现）。数字以各 run 的 `README.md` / `per_iter.tsv` 为准；本文件是对照汇总。

**`avg_skip1`** = Rank0 IterSpeed 丢掉 iter 1 后的算术平均。8 卡 21-step → 20 个稳态点；11-step → iter 2..11；4 卡 5-step → iter 2..5。**带 profiler dump 的 iter 不当墙钟。** 峰值 HBM 来自 `npu-smi` 采样（MB）；RSS 是所有 `scripts.train` 进程 VmRSS 之和（含 worker）。

机器：Ascend **910B3**，64 GiB/卡。git 主干：09-11 终测 `4ea988a`，09-13 之后 `6565bf5`。数据：DROID LeRobot `droid_plus_lerobot_640x360_20260412`，train **484** episode（500 − val − 短于 33 帧）。

---

## 目录

0. [影响 iter 时间和显存的三个因素](#0-影响-iter-时间和显存的三个因素)
1. [当前 TND 如何实现、利弊](#1-当前-tnd-如何实现利弊)（Q/KV 图解：[tnd_q_kv.md](tnd_q_kv.md)）
2. [公共配方](#2-公共配方)
3. [256p packing 长跑 vs FSDP2 MP（09/09）](#3-256p-packing-长跑-vs-fsdp2-mp0909)
4. [TND gather 链条（09/10–09/11）](#4-tnd-gather-链条0910-0911)
5. [VAE pack / prefetch / sidestream（09/11）](#5-vae-pack--prefetch--sidestream0911)
6. [nw=8 ablation 与预存 VAE（09/13）](#6-nw8-ablation-与预存-vae0913)
7. [在线 VAE：256p/480p × TND/per_sample（09/14）](#7-256p-vs-480p-与-480p-优化0914)
8. [预存 VAE：指纹 / episode-id / worker-load（09/14）](#8-预存-vae指纹--episode-id--worker-load0914)
9. [40-step 复核（09/14）](#9-40-step-复核0914)
10. [4 卡 max-token：256/480 × TND/per_sample（09/14–09/15）](#10-4-卡-max-token256480--tndper_sample0914-0915)
11. [`encode_chunk_frames`（09/15）](#11-encode_chunk_frames0915)
12. [10k 正式长跑（未跑满）](#12-10k-正式长跑未跑满)

---

## 0. 影响 iter 时间和显存的三个因素

一步墙钟是 DataLoader、Wan VAE、网络串行或重叠之后的结果。显存要分 **host RSS** 和 **NPU HBM**。后面所有对照都是在拆这三块：不要把 AV1 气泡判成网络慢，也不要把 256p 长 T OOM 判成 TND。周报里同一节写得更完整：[weekly_summary.md §2](weekly_summary.md#2-影响-iter-时间和显存的三个因素)。

```text
磁盘（packed AV1 或预存 .pt）          ← 加载下限，受盘限制
    → DataLoader worker 解码 / 组 batch  ← 主要打 host RSS 和气泡时间
        → packing
            → Wan VAE encode             ← 时间 + NPU HBM（跟帧数 T 走）
                → MoT teacher-forcing    ← 网络：激活显存 + 计算下限
```

现行默认 VAE 在 `training_step` 里和网络 **串行**。DataLoader 可以和下一步重叠。

| 因素 | 主要打什么 | 本周结论 |
| --- | --- | --- |
| **1. DataLoader** | host RSS、墙钟气泡（min 不变） | nw=8 藏 AV1；加载受 `/data5T` 限制 |
| **2. VAE** | 稳态时间 **和** NPU HBM | 可预存，−24%；256p 长 T OOM 在这里 |
| **3. 网络** | `min_s`、transformer 激活 | TND 换 mask；kernel 已较满 |

读表：`min_s` ≈ 网络+在线 VAE；spread ≈ DataLoader；`npu-smi` 64 GiB 但 PyTorch ~38 GiB → VAE workspace；RSS 几百 GiB → DataLoader。

### DataLoader：主要影响内存和时间，加载受磁盘限制

整 episode 从盘上读 packed AV1，解成 uint8。**几乎不占 NPU HBM**（nw=4/8 都是 65508 MB），但：

1. **Host RSS** = 像素 × 卡数 × workers × prefetch。仍解 AV1 时 8 卡 ~470–540 GiB；skip-video 只读 `.pt` → ~150 GiB。week1 一条 batch 堆多条整视频 → 主机 1.46 TiB 被 OOM killer 杀掉。
2. **墙钟气泡**：TND 关 pack，`nw=4` 35.19 s（spread 26 s，min ~21 s）vs `nw=8` 23.30 s（spread 7.4 s）。min 不变，变的是 AV1 露不露出来。`nw=16` 只 −2.1%，RSS 898 GiB → drop。
3. **磁盘是加载下限**。8 卡 × 8 worker 并发读 `.pt` 比 packing 单线程更打盘（480p 27.81 vs 24.51 s）。加 worker 只能把盘延迟藏进队列。

256p：`nw=8` `prefetch_factor=2`。480p：解码不是瓶颈，**`pf=1`**。

### VAE：同时影响时间和显存；可以提前计算、省掉很大一部分

1. **时间**。关 pack 时和网络串在同一步：256p「~21 s 下限」含同卡 VAE。预存 `.pt` 后 24.21→**18.36 s（−24%）**，min 21.2→15.0。480p 31.67→23.71（−25%）。256p 预计算 8.9 min / 回本 ~91 步；480p 5.5 min / ~57 步。
2. **NPU HBM**。CANN 3D workspace **跟 T 走**。同样 ~4.8 万 token：256p T=2257 vs 480p T=477。256p TND T=2257 step1 OOM（reserved ~38 GiB，`npu-smi` 64 GiB）——不是 TND。
3. **提前算**。离线编 `.pt`，训练跳过 encode + uint8 H2D。这是还能再抠的最大一块可离线时间。同卡 pack+sidestream 在 nw=8 后是负优化，已关。

### 网络本身的设计

DataLoader 藏住、VAE 预存之后，剩下的是 MoT + Scheme B。双流展开 packed ≈ `UND + 2×(vis+action)`。GEN attention 默认 **TND：不建 Dense Mask，只 gather 每个 block 看得见的 KV，做 packed 无 mask attention**（下一节）。加卡主要省 HBM 不加速；FSDP2 MP 不省激活。profile 里 FA ~3 s、MatMul cube 97%，再抠单算子收益有限。

---

## 1. 当前 TND 如何实现、利弊

**一句话：** TND 不建 Dense Mask，只把每个 causal block 真正看得见的 KV gather 成变长段，用 packed 无 mask 融合 attention 一次算完。

本周对照默认的 GEN attention 是 **`teacher_forcing_dense_mode=tnd`**。可见性仍是 Scheme B teacher forcing，只换 kernel：Dense Mask + masked SDPA → 按 block gather + packed varlen 无 mask attention（Ascend 上是 `npu_fusion_attention` 的 TND 布局）。UND causal attention 三种模式都走 packed varlen，不受这个开关影响。Q / KV 用 `UND C1 C2 N1 N2` 的拼法见 [tnd_q_kv.md](tnd_q_kv.md)。设计背景见 [`docs/omni_mot_causal_training_implementation_zh.md`](../../../../workspace/chenzhi/cosmos-framework/docs/omni_mot_causal_training_implementation_zh.md) §7.9。

### 它解决什么问题

causal teacher forcing 同一次 forward 同时放：

```text
UND（题目） + 完整 clean 视频（答案/历史） + 完整 noisy 视频（要预测）
```

视频按 VAE latent 切成 causal block（`S` 个 latent / block）。noisy block `i` 可以看题目、最多 `K` 个更早的 clean block、以及当前 noisy block 内部；**不能看当前 clean，也不能看未来**。否则训练抄答案，推理没有这份 clean 就会垮。

Naive 实现是给 GEN query 一张 `[Q, KV]` bool mask，一次 masked SDPA。packing 到 ~4.8 万 pre-TF token 后，mask 和被挡住的 QK 都按二次增长。TND 的目标：**规则不变，但不分配 Dense Mask，也不算被挡住的 QK。**

### 可见性（三种 kernel 必须一致）

`lo = max(0, i - history_blocks)`。

| Query | 可以读 | 禁止 |
| --- | --- | --- |
| clean `Ci` | UND + clean `[lo, i]`（含当前 clean） | 所有 noisy、窗口外 / 未来 clean |
| noisy `Ni` | UND + clean `[lo, i)` + 当前 noisy `Ni` | **当前 clean `Ci`**、未来、其他 noisy block |

多样本之间完全隔离。TND 的 group 由 [`build_teacher_forcing_block_attention_groups()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/data/generator/sequence_packing/teacher_forcing.py#L545) 生成；dense / per_sample mask 是同一套规则的 oracle。CPU 测试要求 TND 的 output 与 QKV 梯度对齐 mask。

### 三种 GEN kernel

`model.config.teacher_forcing_dense_mode`，默认 **`tnd`**（`244068c`）。只换 GEN attention；Norm / QKV / out proj / MLP 仍对完整 packing 跑一次。

```text
tnd:         每个 (sample, block, stream) 一个 varlen segment
             Q_j 只对 gather 出的同样本可见 KV_j 做 unmasked attention
             多样本拼进同一次（或按预算切开的几次）TND 调用

per_sample:  每条 packed sample 一次 masked SDPA（不跨样本，但仍算样本内历史空洞）

global:      整包一张 Dense Mask，一次 masked SDPA（跨样本 QK 也算，靠 bit 遮）
```

主计算量（`Ui` = UND，`Vi` = 原始视觉 token，`P` ≈ 一个 block 的 GEN token）：

```text
global:     2 * sum(Vi) * [sum(Ui) + 2*sum(Vi)]
per_sample: sum(2*Vi * [Ui + 2*Vi])
tnd:        近似 sum(2*Vi * [Ui + (K+1)*P])   ← 再去掉样本内被历史挡住的空洞
```

`global` 的 `[Q, KV]` bool 在 48k packing 上是 HBM 风险，本周没测、不当默认。对照只用 `tnd` vs `per_sample`。

### 实现（block-gather + packed varlen，`2a451d3`）

```text
TeacherForcingLayout（每条 sample 的 block span / S / K）
        │
        ▼
build_teacher_forcing_block_attention_groups
  CLEAN i : query=该 block 的 GEN 切片
            kv_slices = (UND) + (clean[lo .. i])
  NOISY i : kv_slices = (UND) + (clean[lo .. i) 若非空) + (noisy i)
  先发完一条 sample 的全部 CLEAN group，再发 NOISY（拼接 GEN 连续分区）
        │
        ▼
build_teacher_forcing_packed_kv_metadata
  kv_gather_index      重叠的 UND/history 会按 group 复制
  cumulative_seqlen_q  /  cumulative_seqlen_kv
        │
        ▼
teacher_forcing_block_gather_attention
  NPU:  K/V.index_select(gather) → packed varlen unmasked attention
        （Ascend: npu_fusion_attention 的 TND 布局）
        chunk：262144 gathered KV token（~1 GiB bf16 K+V）或 1024 segment
  CPU/fp64 测试: 按 (Q_len, KV_len) 分桶，batched unmasked SDPA，对齐 oracle
```

#### 具体例子（`UND C1 C2 N1 N2`）

完整图解：[tnd_q_kv.md](tnd_q_kv.md)。Q 和 KV **分开组装**。一条样本、2 个 causal block：

```text
packed KV 源序列:   [ UND | C1 C2 | N1 N2 ]
GEN query (full_q): [ C1 C2 | N1 N2 ]
```

| segment | Q（view） | gathered KV | 不进 softmax |
| --- | --- | --- | --- |
| 1 | `C1` | `UND C1` | `C2`、全部 N |
| 2 | `C2` | `UND C1 C2` | 全部 N |
| 3 | `N1` | `UND N1` | 当前答案 `C1` |
| 4 | `N2` | `UND C1 N2` | 当前答案 `C2`、`N1` |

```text
packed Q:   [ C1 | C2 | N1 | N2 ]
packed KV:  [ UND C1 | UND C1 C2 | UND N1 | UND C1 N2 ]
```

`N2` 读得了历史 C1 和当前 N2，slice 停在 C2 之前。UND 复制 4 次。再加 C3/N3 且 `K=1`：`C3 → UND C2 C3`（C1 出窗），`N3 → UND C2 N3`。样本 B 的 group 只接在后面，A 的 Q 不会 gather 到 B。

看不见的 token 不在 gathered KV 里，所以这次 attention **不再带 mask**。noisy 读不到当前 clean，是因为 `kv_slices` 停在 `clean_start + gen_start`。GQA 在 gather 之后 `repeat_interleave` 扩 KV。分发入口：[`teacher_forcing_attention()`](../../../../workspace/chenzhi/cosmos-framework/cosmos_framework/model/generator/mot/attention.py#L289)。

这不是 Scheme A（先 clean forward 存逐层可微 KV），也不是多次 attention 再 LSE 合并。仍是 Scheme B 一次 joint forward，只是 softmax 作用在 gather 出来的可见 KV 上。与 dense SDPA **不要求 bitwise** 一致（tiling / softmax 归约 / BF16 累加不同）。

### 利

| 点 | 本周数字 |
| --- | --- |
| 不算空洞 QK、不分配 Dense Mask | `per_sample` 480p 扫到 6.4 万时稠密 mask 一次申请 **11.30 GiB**，step 1 进不去 |
| 更少 FA launch | 4 卡 profile FA 5.66 s / 27% → **3.06 s / 18%** |
| 墙钟 | 8 卡 nw=4 15-step **48.67 vs 54.81 s（−11.2%）**；nw=8 关 pack **23.30 vs 29.85 s（−21.9%）** |
| 480p 更长序列 | 4 卡 S=4 K=32：TND **68041** vs per_sample **55796** |
| 语义可测 | 与 mask 同一 layout；泄漏当前 clean（`j <= i`）测得出来 |

### 弊

| 点 | 说明 |
| --- | --- |
| Gather 复制 KV | 每个 block 再拷一份 UND + 滑动 clean 窗；用带宽换掉 mask / 空洞 FLOP，所以要 262144 token chunk |
| 临时 buffer | `index_select` + varlen workspace 是额外 HBM。8 卡短测 TND 峰值常贴 655xx，per_sample 15-step 反而 63557 |
| 浮点不等价 | 不能拿单 step loss 逐位对 dense |
| 救不了 VAE | 256p 同 ~4.8 万 token 是 T=2257（480p 的 ~4.7× 帧）。OOM 在 Wan VAE / CANN 3D（跟 **T** 走）。256p TND 上限 **47452** 低于 per_sample **49972**，因为 VAE 先炸。**不要把 256p 长 T OOM 判成 TND gather 不够用** |
| 仍是 Scheme B | 展开 packed ≈ `UND + 2×(vis+action)`；按 block 复用 clean KV 的 Scheme A 还没做 |
| 实现复杂 | group / gather index / cu_seqlen / chunk / GQA 必须与 mask 一致，切错会静默错 attention |

**怎么选：** 要墙钟、要 480p 长序列 → TND（默认）。对齐旧 dense 或怀疑 gather 梯度 → `per_sample`。256p 想加 token → 先封 T / 降 `encode_chunk_frames`，再谈 TND。不要开 `global`。

---

## 2. 公共配方

除非某档另注，短测都从这条基线改 **一个** knob：

| 项 | 值 |
| --- | --- |
| 启动 | `launch_sft_action_policy_droid_edge_local.sh`，`model=mot_causal_fsdp` |
| 模型 | Cosmos3-Edge，`causal_training_strategy=teacher_forcing` |
| 并行 | FSDP，`grad_accum_iter=1`（global batch = 卡数；token packing 下每卡样本数可变） |
| 精度 | bf16 + FSDP2 mixed precision（`fsdp_master_dtype=fp32`） |
| AC | selective，`save_ops_regex=["fmha"]` |
| EMA / compile / viz mask | 关 |
| TF 几何 | S∈[1,4]、K∈[1,32] Uniform（max-token 扫参钉死 S=4 K=32） |
| 分辨率 | **256**（concat 320×256，每 latent 帧 **K=80**）；480p 另注（736×544，**K=391**） |
| packing | `max_samples_per_batch=null`，`max_sequence_length=48000`（pre-TF `UND+vis+action`） |
| 数据集 | `chunk_length=-1`，`max_episode_blocks=-1`，`max_pre_tf_tokens=48000`（index 丢掉超长，不截断） |
| 优化 | AdamW `lr=2e-4`，action 头 `lr_multipliers=5` |
| HCCL | 默认 **AI_CPU**（09-10 起不再 export `HCCL_OP_EXPANSION_MODE=AIV`）；多卡 `HCCL_NPU_SOCKET_PORT_RANGE=35100-35200` |

pre-TF token 公式：`500 + K + N*(K+4)`，`T = 1+4N`。256p 48000 ≈ N=564 / T=2257；480p 48000 ≈ N=119 / T=477。

Scheme-B 展开后 packed ≈ `UND + 2×(vis+action)`。TOML 的 `max_sequence_length` 卡的是 **展开前**。

---

## 3. 256p packing 长跑 vs FSDP2 MP（09/09）

目的：确认 256p + `max_sequence_length=48000` 能稳态跑，以及默认打开 FSDP2 MP 会不会省 HBM。

公共：8 卡、`nw=4`、`prefetch_factor=1`、`lookahead_limit=1`、keep_ranges / 增强 **关**、VAE 仍在 `training_step`。无 packing discard。

| | 无 MP（189-step，手动停） | **有 MP（200/200）** |
| --- | --- | --- |
| run / 文档 | [`docs/droid_edge_res256_seqlen48000.md`](../../../../workspace/chenzhi/cosmos-framework/docs/droid_edge_res256_seqlen48000.md) | [`20260909_154933`](../../../../workspace/chenzhi/cosmos-framework/experiments/20260909_154933_8npu_droid_tf_persample_mp_200step/) |
| `fsdp_mixed_precision_enabled` | 未打开 | **true** |
| GEN attention | per_sample dense mask | 同左 |
| first_s | 99.64 | 99.22 |
| **avg_skip1** | **56.16** | **57.15（+1.8%）** |
| min / max | 47.40 / 71.03 | 45.59 / 72.23 |
| last20 mean loss | 1.48 | 1.36 |
| iter>20 mean loss | 1.627 | 1.621 |
| 峰值 HBM | 最高 **63.7 GiB**（无 655xx） | **64.0 GiB**（NPU0 65522） |
| 峰值 RSS | 319 GiB | 318 GiB |
| 墙钟 | ~2.9 h / 189 step | ~3.16 h / 200 step |

**判定**：MP **没有省显存**（fp32 master shard 多一份拷贝），时间 +1.8%，loss 对齐。收益在优化器/reduce 走 fp32，不在 64 GiB 贴边。要省 HBM 得减 token 或改 AC。

同日更早的 480p packing 1000-step（week1 跨到 09-09，iter 656 手动停）：avg **67.2 s**，6/8 卡贴满 64 GiB，RSS ~340 GiB。256p 同 cap **快约 16%、HBM/RSS 都更松**。相对 256p `max_samples_per_batch=1`（week1 ~9.7 s、HBM 43 GiB），packing 把每步填满 48000，step 时间约 **5.8×**。

---

## 4. TND gather 链条（09/10–09/11）

目的：把 GEN 从 per-sample dense mask SDPA 换成 packed varlen TND，再叠 packing 向量化和 fused RMS。8 卡、256p packing 48000、`nw=4`、`pf=1`、VAE 仍在 training_step。`max_iter=21`（TND vs per_sample 15-step 那对是 15）。

### 4.1 逐步墙钟

| 档 | run | 改动 | avg_skip1 | min / max | HBM | RSS GiB |
| --- | --- | --- | ---: | --- | ---: | ---: |
| persample 200-step 前 21 | `20260909_154933` | dense mask | 58.67（前 21） / 57.15（全 200） | 47.6 / 66.5 | 65522 | 318 |
| **TND gather** | `20260910_024618` | block-span TND | **51.46** | 45.16 / 66.69 | 65520 | 277 |
| packopt | `20260910_055435` | 去掉 per-token `.item()` | **51.13** | — | 65517 | 276 |
| fused RMS | `20260910_073155` | `npu_rms_norm` | **49.86** | 43.65 / 62.90 | 65507 | 279 |
| VAE pack r1 | `20260910_093911` | pack 线程把视频放到 `npu:0` | **oom** 第 0 step | — | 17228 | 149 |
| **VAE pack r2** | `20260910_094336` | device = `parameters().device` | **40.37** | 21.98 / 53.87 | 65510 | 282 |

TND vs 200-step 前 21：**−7.21 s / −12.3%**。packopt 几乎可忽略。fused RMS **−1.27 s**。VAE pack **−9.49 s / −19%**（相对 fused RMS）。

VAE pack r1 失败原因：prefetch 线程把 uint8 放到 `npu:0`，VAE 权重在 `npu:LOCAL_RANK`。r2 按 parameter device 放置后 21/21 ok。iter 1 loss 39.51 与 fused-RMS 逐位对齐（`episode_shuffle_seed=42`）。

### 4.2 同配方 15-step：只改 `teacher_forcing_dense_mode`

8 卡、nw=4、TND 已进代码、**未**开 VAE pack。git `4ea988a`。

| 指标 | tnd `20260911_083350` | per_sample `20260911_085350` | Δ |
| --- | ---: | ---: | ---: |
| n_logs | 15 | 15 | |
| first_s | 96.70 | 100.38 | −3.7% |
| **avg_skip1** | **48.67** | **54.81** | **−11.2%** |
| median | 49.93 | 55.73 | −10.4% |
| min / max | 39.59 / 58.39 | 46.43 / 62.35 | |
| last loss | 3.271 | 3.428 | |
| skip1 mean loss | 7.353 | 7.278 | +1.0% |
| peak HBM | 65531 | 63557 | TND 更高 |
| peak RSS | 264 | 273 | |

**判定：keep TND。** loss 噪声内。TND 稳态 HBM 略高（~64 vs ~57 GiB mean），墙钟明显更好。

---

## 5. VAE pack / prefetch / sidestream（09/11）

目的：profile 显示 Preparing（解 AV1 + packing + H2D）和同卡 VAE 堵 HCCL。措施都在 **训练卡自己** 上做，没有第 9 张卡。

### 5.1 4 卡 profile（驱动措施）

`20260911_093705`：4 卡、TND、**未**开 VAE pack，profile iter 6。对照旧 2 卡 dense-mask（负载不可直接比）。

| CANN / 算子 | 时间 | 说明 |
| --- | ---: | --- |
| Preparing | **19.88 s** | 第一颗 kernel 前的 host |
| Computing | 16.91 s | |
| FA 合计 | 3.06 s | TND varlen；旧 dense 2 卡 5.66 s / 27% → 18% |
| MatMulV3 | 3.44 s | cube ~97% |
| Conv3DV2 | 2.79 s | 当时还在 `get_data_and_condition` 同步 encode |
| fused RMS | 0.47 s | |
| HCCL 非重叠 | **0.07 s** | |

开 VAE pack 后 Preparing → 0.89 s，dump 仍有 Conv3DV2（packing 线程在同一张 NPU 编下一包），HCCL 非重叠升到 **7.5 s** → 做 sidestream。

### 5.2 4 卡 8-step 方案对照

基线：`20260911_103432` VAE pack + `pf=1` + `depth=1`，avg **32.54 s**，spread 29.8 s（19.13–48.93）。有效门槛当时：avg ≤ 30.9 s（−5%），或 avg 不差过 5% 且 spread ≤ 21.8 s。

| 方案 | 改动 | run | avg_skip1 | spread | 判定 |
| --- | --- | --- | ---: | ---: | --- |
| 基线 | pack depth=1 pf=1 | `20260911_103432` | 32.54 | 29.8 | |
| S1 | `prefetch_factor=2` | `20260911_145121` | 31.20 | **21.21** | **keep**（抹平气泡） |
| S3a | `encoded_prefetch_depth=0` | `20260911_150637` | **44.64** | 12.61 | **drop**（同步 encode 更慢） |
| S3b | 本卡 `torch.npu.Stream` | `20260911_151751` | **26.50** | 11.13 | **keep** |
| combo | S1 + S3b + depth=1 | `20260911_153256` | **25.26** | 17.75 | 叠加优于单项 |

### 5.3 8 卡终测（当时默认）

`20260911_154447`：VAE pack + pf=2 + sidestream，`nw=4`，21-step。

| | 8 卡 TND 无 pack（15-step） | 8 卡仅 VAE pack | **终测 combo** |
| --- | ---: | ---: | ---: |
| run | `20260911_083350` | `20260910_094336` | `20260911_154447` |
| **avg_skip1** | 48.67 | 40.37 | **28.17** |
| vs 仅 pack | — | — | **−12.20 s / −30%** |
| vs 无 pack TND | — | — | **−20.50 s / −42%** |
| first_s | — | 142.67 | 93.05 |
| skip1 min / max | — | 21.98 / 53.87 | 20.38 / 41.13 |
| last loss | 3.27 @15 | 2.58 @21 | 2.76 @21 |
| peak HBM | 65531 | 65510 | 65529 |
| peak RSS | 264 | 282 | 283 |
| n_discard | — | 0 | 0 |

当时写进默认：`encode_vision_latents=true`，`encoded_prefetch_depth=1`，`prefetch_factor=2`。**09-13 用 nw=8 复核后这项被工作区回退**（见 §6）。

---

## 6. nw=8 ablation 与预存 VAE（09/13）

公共 knobs（本表所有行除非另注）：8 卡，`max_sequence_length=48000`，resolution=256，FSDP2 MP，HCCL AI_CPU，无 profiler，`max_iter=21`，`prefetch_factor=2`。git `6565bf5`。F 的 git 是 `4ea988a`。

### 6.1 一张表

| 档 | run | attn | VAE pack | sidestream | VAE cache | nw | depth | avg_skip1 | min / max / spread | first | HBM | RSS |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| A persample 无 pack | `20260913_161144` | per_sample | 关 | — | 关 | 8 | 0 | **29.85** | 24.48 / 35.54 / 11.06 | 75.83 | 65465 | 496 |
| **B TND 无 pack** | `20260913_162502` | tnd | 关 | — | 关 | 8 | 0 | **23.30** | 20.51 / 27.86 / 7.35 | 71.49 | 65508 | 491 |
| B4 TND 无 pack | `20260913_234421` | tnd | 关 | — | 关 | **4** | 0 | **35.19** | 20.97 / 47.31 / 26.34 | 90.20 | 65508 | 278 |
| C TND+pack 无 stream | `20260913_163610` | tnd | 开 | 关 | 关 | 8 | 1 | **28.63** | 24.25 / 33.51 / 9.26 | 101.67 | 65507 | 497 |
| D TND+pack+stream | `20260913_134951` | tnd | 开 | 开 | 关 | 8 | 1 | **24.21** | 21.18 / 28.79 / 7.61 | 85.06 | 65530 | 499 |
| E D+预存 VAE | `20260913_144719` | tnd | 开 | 开 | 开 | 8 | 1 | **18.36** | 14.95 / 23.13 / 8.18 | 70.41 | 65209 | 505 |
| F 09-11 终测 | `20260911_154447` | tnd | 开 | 开 | 关 | 4 | 1 | 28.17 | 20.38 / 41.13 / 20.75 | 93.05 | 65529 | 283 |
| G D 配方 nw=16 | `20260913_151133` | tnd | 开 | 开 | 关 | **16** | 1 | 23.71 | 19.43 / 28.43 / 9.00 | 81.52 | 65525 | 898 |

无 pack 时 sidestream 不适用，`encoded_prefetch_depth=0`。

### 6.2 各措施单独贡献

| 措施 | 对照 | Δ avg_skip1 |
| --- | --- | --- |
| TND（vs persample，均 nw=8 关 pack） | B − A | **−6.55 s（−21.9%）** |
| nw=8 vs nw=4（同 TND 无 pack） | B − B4 | **−11.90 s（−33.8%）** |
| VAE pack 无 sidestream | C − B | **+5.33 s（+22.9%）** 负优化 |
| sidestream | D − C | −4.42 s（−15.4%）把 pack 救回 24.21 |
| 预存 VAE | E − D | **−5.85 s（−24.2%）** |
| nw=16 vs nw=8（同 D） | G − D | −0.50 s（−2.1%），RSS 翻倍 |

B4 怎么读：min 几乎一样（20.97 vs 20.51）→ 计算下限没变；max 47.31 vs 27.86、spread 26.3 vs 7.4 → nw=4 挡不住整段 AV1。HBM 两边都是 65508，workers 不改显存。

```text
persample nw=8                    29.85 s
  → TND 无 pack nw=8              23.30 s    −6.55 s   ← 现行方向
  → TND 无 pack nw=4              35.19 s    AV1 露出来
  → +VAE pack（无 sidestream）    28.63 s    负优化
  → +sidestream                   24.21 s    仍慢于关 pack
  → +预存 VAE cache               18.36 s    可选
```

**判定**：TND keep；nw=8 keep（不当全局默认写进 09-11 代码，已写进工作区 recipe）；nw=16 drop；在线 VAE pack **关**（工作区已改 `encode_vision_latents=false`）；预存 VAE 作可选 env。

### 6.3 8 卡 profile（#1）

`20260913_130109` r1：conda `libstdc++` 不在 `LD_LIBRARY_PATH` 最前，torchcodec 挂（`GLIBCXX_3.4.32`）。`20260913_130731` r2：iter 6 dump，CSV 齐，wrapper 20 min hang 杀掉 `trace_view.json`。

| | 4 卡无 VAE pack（09-11） | 8 卡 pack+stream（09-13 r2） |
| --- | ---: | ---: |
| Preparing | 19.88 s | **0.002 s** |
| Computing | 16.91 s | 21.07 s |
| HCCL 非重叠 | 0.07 s | **6.62 s** |
| Conv3DV2 | 2.79 s | 4.27 s |
| FA 合计 | 3.06 s | 4.42 s |
| host `aclrtSynchronizeStream` | 6.00 s | 17.57 s |

VAE pack 砍掉 Preparing。sidestream **没有**把 8 卡 HCCL 非重叠压回 0.07 s。这驱动了预存 VAE。

### 6.4 预存 VAE 占用

cache：`/data5T/chenzhi/vae_latent_cache_droid_edge_256`。原始 success 视频 5.575 GiB。

| | 原始 success | VAE latent cache |
| --- | ---: | ---: |
| 合计 | 5.941 GiB | **2.061 GiB**（原视频的 37%） |
| 条目 | 500 episode（info） | 484 × `.pt`（`max_pre_tf_tokens=48000` 丢掉 16 条） |
| 平均每条 | 视频约 11.4 MiB | 4.37 MiB |
| 预计算 | — | 8 卡 **532.9 s**（8.9 min），484/484 |
| 回本 vs D | — | 532.9 / 5.85 ≈ **91 步** |
| 测速 cache hit | — | 1338/1338 |

预存只跳过 NPU encode 和 uint8 H2D；DataLoader **仍解 AV1**（spread 仍 ~8 s）。

### 6.5 nw=8 `prefetch_factor=1`（11-step）

对照 D 档 pf=2 的 iter 2..11 ≈ 24.43 s、RSS 到 iter 11 ≈ 472 GiB。

| run | res | nw | pf | avg_skip1 | min / max / spread | first | HBM | RSS |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| D 的 2..11 | 256 | 8 | 2 | 24.43 | 21.18 / 28.79 / 7.61 | 85.06 | 65530 | 472 |
| `20260914_012422` | 256 | 8 | **1** | **24.79** | 22.45 / 29.24 / 6.79 | 87.46 | 65524 | **472** |
| `20260914_013436` | **480** | 8 | 1 | **34.11** | 24.76 / 41.82 / **17.06** | 80.65 | 65512 | 418 |

256p pf 2→1：墙钟几乎不变，RSS 也没少。480p 同 nw=8 pf=1 能跑完；spread 17 s，数据气泡比 256p 大。

---

## 7. 256p vs 480p 与 480p 优化（09/14）

### 7.1 在线 VAE（不预存）：256p / 480p × TND / per_sample

全部 **`encode_vision_latents=false`**，VAE 在 `training_step`。无预存 cache、无 packing 侧预编。

#### 8 卡 packing 48000，S/K 随机

S∈[1,4]、K∈[1,32] Uniform。256p `nw=8 pf=2`，480p `nw=8 pf=1`。git `6565bf5`。

| 分辨率 | kernel | avg_skip1 | min–max | 峰值 HBM | RSS | n | run |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 256p | **TND** | **23.30**（21-step）；2..11 **24.61** | 20.5–27.9 | **65508** | 491 | 21 | `162502` |
| 256p | per_sample | **29.85**；2..11 **29.29** | 24.5–35.5 | **65465** | 496 | 21 | `161144` |
| 480p | **TND** | **31.67**（11-step） | 25.2–40.6 | **63959** | **379** | 11 | `020622` |
| 480p | per_sample | **未测** | — | — | — | — | 没有 8 卡 480p + 在线 VAE + per_sample |

256p TND vs per_sample：**−6.55 s（−22%）**，HBM/RSS 几乎一样。480p TND vs 256p TND（同 11-step）：**1.29×**，HBM 480p 略低（63959 vs 65508）。

#### 4 卡同约 48000 token：唯一完整 2×2

VAE 仍在网络里。一条强制长度样本，**钉死 S=4、K=32**。5 step，avg = iter 2..5。

| 分辨率 | kernel | T | pre-TF | status | 2–5 均 | 峰值 HBM | RSS | run |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 256p | TND | 2257 | 47956 | **oom** step1 | — | 64489 | 41 | `140920` |
| 256p | TND | 2233 | 47452 | ok | **43.61** | **65512** | 49 | `n558` |
| 256p | per_sample | 2257 | 47956 | ok | **76.45** | **58444** | 49 | `002513` |
| 480p | TND | 477 | 47896 | ok | **35.97** | **65513** | 73 | `003428` |
| 480p | per_sample | 477 | 47896 | ok | **73.23** | **64127** | 72 | `004012` |

- 480p TND vs per_sample：**35.97 vs 73.23（−51%）**，两边都装得下。
- 256p：同 T=2257 TND **VAE OOM**、per_sample 76.45 s / 58 GiB → 墙在 VAE/T，不在 TND。T=2233 时 TND 43.61 vs per_sample 76.45（−43%）。
- 钉死最大 history 后 480p TND 比 256p TND **更快**（36 vs 44 s）；8 卡日常 packing 相反（24.6 vs 31.7 s）。

### 7.1b 仅 TND：256p vs 480p 其它路径（含预存）

S/K Uniform，8 卡 TND、cap 48000。256p `pf=2`，480p 关 pack `pf=1`。256p 关 pack 21-step，480p 11-step；「同 11-step」用 256p iter 2..11。

| 路径 | 卡 | 256p avg | 256p HBM / RSS | 480p avg | 480p HBM / RSS | 480÷256 | run 256 / 480 |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| **关 pack 在线** | 8 | **23.30**（21-step）；2..11 **24.61** | **65508** / 491 | **31.67** | **63959** / **379** | **1.29×** | `162502` / `020622` |
| pack+sidestream | 8 | 24.21 | 65530 / 499 | 34.11 | 65512 / 418 | 1.41× | `134951` / `013436` |
| 指纹 cache 11-step | 8 | **19.22** | 65501 / 478 | **23.71** | **62982** / 471 | **1.23×** | `050033` / `043731` |
| episode-id 11-step | 8 | 20.93 | 63785 / **152** | 24.51 | 63766 / **153** | 1.17× | `090054` / `072856` |
| episode-id 40-step | 8 | 19.81 | 65531 / 153 | 27.95 | 65510 / 155 | 1.41× | `093801` / `101314` |
| worker-load 40-step | 4 | 18.93 | 65527 / 77 | 25.59 | 65506 / 77 | 1.35× | `121811` / `125719` |

关 pack 256p min/max 20.5–27.9（spread 7.4）；480p 25.2–40.6（spread 15.4）。指纹 cache 256p 14.2–23.5；480p 14.7–32.4。

- **时间**：日常 480p 是 256p 的 **1.2–1.4×**。慢在 VAE 空间和 480p 长包，不是 token 数。预存后收到 1.23× / 1.17×。
- **HBM**：packing 48000 两边都常贴 64 GiB。480p 关 pack **63959 < 256p 65508**（T 短，VAE workspace 轻）。不要和 §10 把 256p 拉到 T=2233 去撞 VAE 墙混读。
- **RSS**：解 AV1 时 480p 更低（帧少）。skip-video 后两边都 ~150 GiB（8 卡）/ ~77 GiB（4 卡）。

### 7.2 480p 11-step 优化 1/2/3/5

对照 baseline `20260914_013436`：nw=8 pf=1 pack+sidestream，avg **34.105 s**。每档 `max_iter=11`。不落盘 VAE cache。git `6565bf5`。

| 档 | run | pack | depth | pf | avg_skip1 | min / max / spread | first | HBM | RSS | status |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| baseline pack+stream | `20260914_013436` | 开 | 1 | 1 | 34.10 | 24.76 / 41.82 / 17.06 | 80.65 | 65512 | 418 | ok |
| **1 关 VAE pack** | `20260914_020622` | 关 | 0 | 1 | **31.67** | 25.21 / 40.57 / 15.36 | 64.49 | 63959 | 379 | ok |
| 2 encode_chunk 24→8 | `20260914_021448` | 开 | 1 | 1 | — | — | — | 65530 | 302 | **oom** |
| 3 packing 像素 bf16 | `20260914_021713` | 开 | 1 | 1 | 33.52 | 28.91 / 40.76 / 11.85 | 83.45 | 65527 | 409 | ok |
| 5 prefetch_factor=2 | `20260914_022715` | 开 | 1 | 2 | 32.06 | 22.48 / 36.80 / 14.32 | 86.65 | 65073 | 482 | ok |
| 1+5 关 pack + pf=2 | `20260914_025812` | 关 | 0 | 2 | 32.21 | 25.78 / 44.68 / 18.90 | 66.09 | 65355 | 460 | ok |

相对 baseline 34.10 s：

| 档 | Δ | 判定 |
| --- | --- | --- |
| 1 关 pack | **−2.44 s（−7.2%）** | **keep**：480p 重叠也是负优化 |
| 2 chunk=8 | OOM 第 0/11，HBM 65530 | **drop**：更小窗反而峰值更高 |
| 3 bf16 像素 | −0.59 s | **drop**：`COSMOS_VAE_ENCODE_DTYPE=bf16`，Wan 计算已是 bf16 |
| 5 pf=2 | −2.04 s | 单独略快；RSS +64 GiB |
| 1+5 | −1.89 s | **不叠加**：比关 pack pf=1 慢 0.5 s |

### 7.3 关 pack 时怎么配 prefetch

在线 VAE 的 2×2 见 [§7.1](#71-在线-vae不预存256p--480p--tnd--per_sample)。这里只留 pf 结论：

| res | nw | pf | depth | run | avg_skip1 | min / max | HBM | RSS |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| 256 | 8 | **2** | 0 | `20260913_162502`（21-step） | **23.30** | 20.51 / 27.86 | 65508 | 491 |
| 480 | 8 | **1** | 0 | `20260914_020622` | **31.67** | 25.21 / 40.57 | 63959 | 379 |
| 480 | 8 | 2 | 0 | `20260914_025812` | 32.21 | 25.78 / 44.68 | 65355 | 460 |

- **256p**：`nw=8` **`pf=2`**。AV1 要 8 worker + prefetch 才能藏住（nw=4 变成 35.2 s）。
- **480p**：`nw=8` **`pf=1`**。关 pack 后再加 pf=2 不加速，RSS 379→460。VAE 长包才是瓶颈。

同 11-step 关 pack：**31.67 / 24.61 ≈ 1.29×**（不要拿 21-step 的 23.30 去除 11-step 的 31.67 得到 1.36×）。

---

## 8. 预存 VAE：指纹 / episode-id / worker-load（09/14）

全部 8 卡、11-step（另注除外）、TND、token cap 48000。`depth` = packing 预编队列；cache 命中时 depth=1 只 overlap 读 `.pt` / encode skip。

指纹路径：cache key 是像素指纹，**仍解 AV1**，跳过 encode。episode-id：key = episode + 后处理 `(T,H,W)`，`COSMOS_VAE_SKIP_VIDEO=1` **不解视频**。worker-load：DataLoader worker `torch.load` `.pt`（bf16 盘占用减半）。

| res | 路径 | 解视频 | 谁读 `.pt` | nw | pf | depth | avg | min / max | RSS | run |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 256 | 关 pack 在线 | 解+编 | — | 8 | 2 | 0 | 23.30 | 20.5 / 27.9 | 491 | `162502`（21-step） |
| 256 | 指纹 cache | 解、不编 | packing | 8 | 2 | 1 | **19.22** | 14.2 / 23.5 | 478 | `050033` |
| 256 | worker+bf16 | 不解 | worker | 8 | 2 | 1 | 21.80 | 17.8 / 26.9 | 151 | `083819` |
| 256 | worker+bf16 | 不解 | worker | 4 | 2 | 1 | 20.07 | 17.8 / 24.8 | **89** | `084916` |
| 256 | episode-id | 不解 | packing | 8 | 2 | 1 | 20.93 | 17.4 / 28.9 | 152 | `090054` |
| 480 | 关 pack 在线 | 解+编 | — | 8 | 1 | 0 | 31.67 | 25.2 / 40.6 | 379 | `020622` |
| 480 | 指纹 cache | 解、不编 | packing | 8 | 1 | 1 | 25.90 | 17.6 / 32.7 | 414 | `034024` |
| 480 | 指纹 cache | 解、不编 | packing | 8 | 2 | 1 | **23.71** | 14.7 / 32.4 | 471 | `043731` |
| 480 | episode-id | 不解 | packing | 8 | 2 | 1 | 24.51 | 18.8 / 32.9 | 153 | `072856` |
| 480 | worker+bf16 | 不解 | worker | 8 | 2 | 1 | 27.81 | 16.9 / 42.3 | 151 | `075631` |

失败档（不进表）：`060659` / `072125` episode-id 因 skip-video 误触发失败；`035925` pf=2 指纹 ENOSPC，方向与 r2 的 23.71 一致。

含 iter 1 的全程平均（11-step 里 first 权重大）：

| res | 路径 | first | avg_skip1 | **mean_all** |
| --- | --- | ---: | ---: | ---: |
| 256 | 指纹 cache | 69.96 | 19.22 | 23.84 |
| 256 | episode-id packing | **51.10** | 20.93 | **23.67** |
| 256 | worker+bf16 nw=4 | **48.17** | 20.07 | **22.63** |
| 480 | 指纹 cache pf=2 | 61.54 | 23.71 | **27.15** |
| 480 | episode-id packing | **55.39** | 24.51 | 27.32 |

**判定**：稳态墙钟指纹最好；skip-video 赢在 first 和 RSS。8 卡 × 8 worker 并发读 `/data5T` 比 packing 单线程更打盘（480p worker 27.81，max 42 s）。减 worker 变快（256p nw=4 20.07 vs nw=8 21.80）支持「抢盘」假说。没有测过 cache + `encoded_prefetch_depth=0`。

480p 预计算：8 卡 **330 s** / **404** 条 / **6.55 GiB**（256p 是 2.06 GiB / 484 条）。相对关 pack 31.67→25.90，回本 ≈ 57 步。

---

## 9. 40-step 复核（09/14）

### 9.1 指纹 vs episode-id（8 卡，packing 读 `.pt`）

`nw=8` `pf=2` `depth=1` TND token cap 48000。指纹仍解 AV1；epid 不解视频。指纹前两次 skip-video 误触发作废；r3 关掉 `COSMOS_VAE_SKIP_VIDEO` 后 256p 跑满，480p 指纹 iter 27 **HBM OOM**。

| 档 | run | first | avg_skip1 | mean_all | min / max / spread | HBM | RSS | status |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 256p 指纹 | `20260914_104714` | 64.60 | **19.63** | 20.76 | 13.65 / 31.40 / 17.75 | 65512 | 542 | ok |
| 256p episode-id | `20260914_093801` | 48.66 | **19.81** | 20.53 | 15.80 / 29.77 / 13.97 | 65531 | **153** | ok |
| 480p 指纹 | `20260914_110348` | 61.98 | 22.75 | 24.21 | 16.17 / 31.83 / 15.66 | 65535 | 567 | **oom** iter 27 |
| 480p episode-id | `20260914_101314` | 59.35 | **27.95** | 28.73 | 16.50 / 43.10 / 26.60 | 65510 | **155** | ok |

11-step 的「指纹略快」在 40-step 256p 上变成打平（19.63 vs 19.81）。480p 指纹长跑会被解视频的 HBM/RSS 打死；epid 能跑满但稳态 28 s，和 11-step 的 24.5 同量级、spread 更大。

作废：`093218` / `103933`（256p 指纹 skip-video）；`095433`（480p 指纹中途失败）。

### 9.2 packing 线程 load vs worker load（4 卡）

phy 0–3，episode-id v2，`COSMOS_VAE_SKIP_VIDEO=1`，`nw=8` `pf=2` `depth=1` TND 48000。不解 AV1。`avg_skip1` = iter 2..40。

**256p**

| 档 | run | first | avg_skip1 | mean_all | min / max | HBM | RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| packing 线程 | `20260914_120209` | 45.92 | 19.20 | 19.86 | 13.03 / 29.47 | 65535 | 77.73 |
| worker load | `20260914_121811` | 46.68 | **18.93** | 19.63 | 13.96 / 29.45 | 65527 | 76.95 |

**480p**

| 档 | run | first | avg_skip1 | mean_all | min / max | HBM | RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| packing 线程 | `20260914_123642` | 50.98 | 26.16 | 26.78 | 13.46 / 39.14 | 65532 | 78.73 |
| worker load | `20260914_125719` | 48.95 | **25.59** | 26.17 | 12.05 / 40.42 | 65506 | 77.30 |

**判定**：4 卡上 worker 略快（−0.3 / −0.6 s），噪声内。RSS 都在 77 GiB（4 卡、skip-video）。8 卡 worker 抢盘的结论不能外推到 4 卡。

---

## 10. 4 卡 max-token：256/480 × TND/per_sample（09/14–09/15）

这是 **钉死 `history_blocks=32`** 的天花板。日常在线 VAE 的 256p / 480p × TND / per_sample 在 [§7.1](#71-在线-vae不预存256p--480p--tnd--per_sample)。

### 10.1 扫参方法

phy **0–3**，FSDP 4 卡，**一条样本填满预算**：

| 项 | 值 |
| --- | --- |
| `max_samples_per_batch` | **1** |
| `max_sequence_length` | null |
| `encode_vision_latents` | **false**（VAE 在 `training_step`） |
| TF | **钉死** `block_size=4`、`history_blocks=32` |
| 帧数 | `COSMOS_FORCE_VIDEO_FRAMES=T=1+4N` |
| pre-TF | `500 + K + N*(K+4)` |
| K | 80（256p 320×256）/ 391（480p 736×544） |
| 每档 | 5 step |
| 停搜 | token 间隔 ≤ **1000** 或 N 相邻 |
| 离 OOM 远 | 480p TND 曾用 +3000，接近上限改 +1000 |

`20260914_140743` 启动失败（HBM 3426，几乎没起来）；`140920` 是 256p TND max-T 的 OOM。

### 10.2 同约 48000 tokens 四档

| 分辨率 | 模式 | N | T | pre-TF | status | n_logs | iter1 | 2–5 均 | 峰值 HBM | run |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 256p | TND | 564 | 2257 | 47956 | **oom** | 0 | — | — | 64489 | `20260914_140920` |
| 256p | per_sample | 564 | 2257 | 47956 | ok | 5 | 123.69 | 76.45 | 58444 | `20260915_002513` |
| 480p | TND | 119 | 477 | 47896 | ok | 5 | 79.25 | **35.97** | 65513 | `20260915_003428` |
| 480p | per_sample | 119 | 477 | 47896 | ok | 5 | 115.66 | 73.23 | 64127 | `20260915_004012` |

同样 ~4.8 万 token：256p T 是 480p 的约 **4.7 倍**（2257 vs 477），TF block 约 141 vs 30。256p TND 在 step 1 前 OOM：`Tried to allocate 1.47 GiB`，PyTorch reserved ~38 GiB，`npu-smi` 64 GiB。

### 10.3 256p TND 二分（token 间隔 1000）

已知 oom N=564 tok=47956。从短序列探起：

| N | T | tok | status | first | avg_skip1 | HBM | run |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 8 | 33 | — | ok | — | 8.72 | 17325 | `srch_256_tnd_n8` |
| 469 | 1877 | — | ok | — | 32.88 | 56342 | `n469` |
| 516 | 2065 | — | ok | — | 37.70 | 63611 | `n516` |
| 540 | 2161 | — | ok | — | 39.70 | 65181 | `n540` |
| 552 | 2209 | — | ok | — | 44.76 | 65509 | `n552` |
| **558** | **2233** | **47452** | **ok** | 99.86 | **43.61** | 65512 | `n558` |
| 564 | 2257 | 47956 | oom | — | — | 64489 | `max_256_tnd` |

最大 ok：**T=2233 / 47452**。夹缝 N=559–563 未测（均 &lt; 1000 tokens）。

### 10.4 256p per_sample 从 48000 往上 +1000

同 T=2257 时 TND 已因 VAE 死，per_sample 还能跑。

| target | N | T | tok | status | first | avg_skip1 | HBM | run |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 48000 | 564 | 2257 | 47956 | ok | 123.69 | 76.45 | 58444 | `max_256_per_sample` |
| 49000 | 576 | 2305 | 48964 | ok | 129.41 | 80.16 | 65016 | `step_256_per_sample_t49000` |
| **50000** | **588** | **2353** | **49972** | **ok** | 133.51 | **82.89** | 61792 | `t50000` |
| 51000 | 600 | 2401 | 50980 | oom | 133.13 | — | 65515 | `t51000`（step1 过、再申请 7.79 GiB） |

### 10.5 480p TND 从 48000 往上

| target | N | T | tok | status | first | avg_skip1 | HBM | run |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 48000 | 119 | 477 | 47896 | ok | 79.25 | 35.97 | 65513 | `max_480_tnd` |
| 49000 | 121 | 485 | 48686 | ok | 79.63 | 36.86 | 65508 | `t49000` |
| 50000 | 124 | 497 | 49871 | ok | 82.62 | 37.67 | 59987 | `t50000` |
| 51000 | 126 | 505 | 50661 | ok | 81.71 | 38.83 | 55754 | `t51000` |
| 52000 | 129 | 517 | 51846 | ok | 87.48 | 41.24 | 57039 | `t52000` |
| 53000 | 131 | 525 | 52636 | ok | 89.62 | 42.37 | 65533 | `t53000` |
| 54000 | 134 | 537 | 53821 | ok | 89.17 | 42.96 | 60495 | `t54000` |
| 55000 | 136 | 545 | 54611 | ok | 90.32 | 43.29 | 62586 | `t55000` |
| 56000 | 139 | 557 | 55796 | ok | 91.25 | 47.73 | 64028 | `t56000` |
| 57000 | 142 | 569 | 56981 | ok | 92.42 | 47.23 | 62090 | `t57000` |
| 58000 | 144 | 577 | 57771 | ok | 96.57 | 49.52 | 64737 | `t58000` |
| 59000 | 147 | 589 | 58956 | ok | 97.82 | 51.32 | 64893 | `t59000` |
| 60000 | 149 | 597 | 59746 | ok | 98.55 | 52.83 | 64667 | （progress；未单独建 run 目录） |
| 61000 | 152 | 609 | 60931 | ok | 98.93 | 52.85 | 65409 | `t61000` |
| 62000 | 154 | 617 | 61721 | ok | 103.34 | 54.32 | 65347 | `t62000` |
| 63000 | 157 | 629 | 62906 | ok | 107.25 | 58.03 | 65503 | `t63000` |
| 64000 | 159 | 637 | 63696 | ok | 111.48 | 60.10 | 65419 | `t64000` |
| 70000 | 174 | 697 | 69621 | oom | 118.62 | — | 65514 | `t70000` |

64000–70000 二分：

| N | T | tok | status | first | avg_skip1 | HBM | run |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 166 | 665 | 66461 | ok | 112.51 | 61.31 | 64697 | `srch_480_tnd_n166` |
| **170** | **681** | **68041** | **ok** | 116.21 | **72.82** | 65513 | `n170` |
| 172 | 689 | 68831 | oom | 114.27 | — | 64688 | `n172`（step1 过、backward 再申请 4.89 GiB） |

### 10.6 480p per_sample 二分

| N | T | tok | status | first | avg_skip1 | HBM | run |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 119 | 477 | 47896 | ok | 115.66 | 73.23 | 64127 | `max_480_per_sample` |
| 159 | 637 | 63696 | oom | — | — | 65253 | `n159`（稠密 mask 11.30 GiB，step1 进不去） |
| **139** | **557** | **55796** | **ok** | 150.82 | **100.79** | 65528 | `n139` |
| 149 | 597 | 59746 | oom | — | — | 59377 | `n149` |
| 144 | 577 | 57771 | oom | — | — | 56665 | `n144` |
| 141 | 565 | 56586 | oom | 149.23 | 102.72 | 65533 | `n141`（step2 后炸） |

### 10.7 上限怎么读

```
480p TND (68041)  >  480p per_sample (55796)  >  256p per_sample (49972)  >  256p TND (47452)
```

- **256p TND 最短**：爆的是长序列 **Wan VAE / CANN 3D workspace（跟 T 走）**，不是 TND gather。
- **256p per_sample 反而更能撑 token**：同 T=2257 时 TND 因 VAE 先死。
- **480p TND 最长**：T 短，VAE 过得去；transformer 反传到 ~6.8 万才顶满 64 GiB。
- **480p per_sample 比同分辨率 TND 少约 1.2 万 token**：`build_per_sample_teacher_forcing_gen_masks` 稠密 mask。

峰值 HBM 经常顶到 655xx，15 s 采样里有单卡尖峰，不等于全程 64 GiB 稳态。

---

## 11. `encode_chunk_frames`（09/15）

只改 `model.config.tokenizer.encode_chunk_frames.256`（必须是 4 的倍数）。其余与 §10 相同：4 卡、mspb=1、VAE 在 training_step、TF S=4 K=32、`COSMOS_FORCE_VIDEO_FRAMES=T`。

### 11.1 固定 T=2233（先前 max-ok）改窗长

| T_chunk | T | N | pre-TF | status | first | avg_skip1 | HBM | run |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 68（对照） | 2233 | 558 | 47452 | ok | 99.86 | 43.61 | 65512 | `srch_256_tnd_n558` |
| 32 | 2233 | 558 | 47452 | ok | 90.75 | **42.94** | 65379 | `t2233_c32` |
| **24** | 2233 | 558 | 47452 | ok | 93.99 | 45.36 | **63015** | `t2233_c24` |
| 68 @ T=2257 | 2257 | 564 | 47956 | oom | — | — | 64489 | `max_256_tnd` |

chunk=24 把峰值从贴顶 65512 降到 63015，稳态时间略差（45.4 vs 43.6）。窗越小 encode 次数越多（T=2233：68→32 次、24→93 次）。

### 11.2 chunk=24 再搜 256p TND 上限

先探 50000 / 60000；60000 炸则中间二分，token 间隔 ≤ 1000 停。

| slug | target | N | T | tok | status | n_logs | first | avg_skip1 | HBM |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `c24_t50000_n588` | 50000 | 588 | 2353 | 49972 | ok | 5 | 96.28 | 49.21 | 65528 |
| `c24_t60000_n707` | 60000 | 707 | 2829 | 59968 | oom | 0 | — | — | 64465 |
| `c24_t54928_n647` | 54928 | 647 | 2589 | 54928 | oom | 1 | 102.2 | — | 65510 |
| **`c24_t52408_n617`** | **52408** | **617** | **2469** | **52408** | **ok** | 5 | 101.93 | **53.13** | 65528 |
| `c24_t53668_n632` | 53668 | 632 | 2529 | 53668 | oom | 1 | 101.64 | — | 65216 |
| `c24_t52996_n624` | 52996 | 624 | 2497 | 52996 | oom | 3 | 99.87 | 53.64 | 65535 |

最大 ok：**52408**（相对 chunk=68 的 47452 **+4956 tok / +236 帧**）。下一档 52996 第 3 步 OOM。

### 11.3 同样 T_chunk=24：256p vs 480p（仍是钉死 K=32）

日常在线 VAE 的 2×2 见 [§7.1](#71-在线-vae不预存256p--480p--tnd--per_sample)。下面是上限扫参、history=32。

480p 的 `encode_chunk_frames.480` 默认已是 24。

**各自上限**

| 档 | T_chunk | N | T | pre-TF | iter1 | 2–5 均 | 峰值 HBM | 下一档 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256p TND | 68 | 558 | 2233 | 47452 | 99.86 | 43.61 | 65512 | 47956 step1 前 OOM |
| 256p TND | **24** | 617 | 2469 | **52408** | 101.93 | 53.13 | 65528 | 52996 第 3 步 OOM |
| 480p TND | **24** | 170 | 681 | **68041** | 116.21 | 72.82 | 65513 | 68831 backward OOM |

**同约 5 万 tokens**

| 档 | T_chunk | N | T | pre-TF | iter1 | 2–5 均 | 峰值 HBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256p TND | 24 | 588 | 2353 | 49972 | 96.28 | **49.21** | 65528 |
| 480p TND | 24 | 124 | 497 | 49871 | 82.62 | **37.67** | 59987 |
| 256p TND（T=2233） | 24 | 558 | 2233 | 47452 | 93.99 | 45.36 | 63015 |
| 480p TND（~48000） | 24 | 119 | 477 | 47896 | 79.25 | 35.97 | 65513 |

同 ~5 万 tokens：256p 慢约 **30%**（49.2 vs 37.7 s），已经贴 64 GiB；480p 还有余量所以能继续加到 6.8 万（那时 2–5 均 72.8 s，T 仍只有 681）。

**判定**：减小 T_chunk 能把 256p VAE workspace 上限抬一点，换的是更多次 3D conv、稳态更慢。不能靠它让 256p 在同 token 上追上 480p。不要在 256p 上只加 `max_sequence_length` 却不封 T。

---

## 12. 10k 正式长跑（未跑满）

目标：8 卡 256p packing + TND + AI_CPU，`max_iter=10000`，`save_iter=1000`，wandb offline。当时配方仍是 09-11 的 VAE pack + sidestream + nw=4。

| run | 启动 | 结果 | last |
| --- | --- | --- | --- |
| `20260910_155517` r1 | 前台 | rank0 **SIGKILL @ iter 93**；无 dmesg OOM；NPU 正常 | loss 1.478 @ 93 |
| `20260910_171645` r2 | `setsid` 脱离 Grok 进程组 | **HCCL 默认 16666 bind 冲突** | 0 step |
| `20260910_235029` r3 | setsid + `HCCL_NPU_SOCKET_PORT_RANGE=35100-35200` | 跑到 iter 120 后 SIGINT | loss 1.446 @ 120 |

要再开 10k：AI_CPU（不要 AIV）+ 端口区间 + `export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH`。工作区默认已经关 pack、nw=8，墙钟会和 r3 不同（预期更快、RSS 更高）。
