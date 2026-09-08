# 周报：Cosmos3 DROID Edge 整 episode SFT（2026/09/02 – 2026/09/08）

仓库：`cosmos-framework` 分支 `feat/omni-mot-causal-action-training`（相对 origin **超前 5 个 commit，未 push**）。
`cosmos` 本体跟了一笔：本地 launch 打开 keep_ranges concat（`c83d99e`，9/4）。

本周主线：把 DROID Edge **整 episode teacher-forcing SFT** 在 910B3 上跑通，并扫清 **HBM / iter 时间 / host RSS** 边界。

---

## 目录

1. [一句话总结](#1-一句话总结)
2. [代码改动](#2-代码改动)
3. [训练显存与 iter 时间](#3-训练显存与-iter-时间)
   - 3.1 [结论](#31-结论)
   - 3.2 [方法与数据](#32-方法与数据)
   - 3.3 [2 / 4 / 8 卡主表（480p）](#33-2--4--8-卡主表480p)
   - 3.4 [8 卡 `num_workers` 搜索](#34-8-卡-num_workers-搜索)
   - 3.5 [480p OOM token 边界（100–120 / `-1`）](#35-480p-oom-token-边界100120--1)
   - 3.6 [256p](#36-256p)
   - 3.7 [full activation checkpointing](#37-full-activation-checkpointing)
   - 3.8 [固定 TF S=8 / K=10，以及有没有漏](#38-固定-tf-s8--k10以及有没有漏)
   - 3.9 [只设 `max_sequence_length=48000` 的 packing](#39-只设-max_sequence_length48000-的-packing)
   - 3.10 [910C 单卡 / 多卡对照](#310-910c-单卡--多卡对照)
4. [Host RSS / dataloader 探测](#4-host-rss--dataloader-探测)
5. [未提交与待办](#5-未提交与待办)
6. [附录：原始 log](#6-附录原始-log)

---

## 1. 一句话总结

整 episode + `keep_ranges` 拼接已经接到训练；**480p 8 卡能稳定跑到 `max_episode_blocks=120`（481 帧，~77 s/step，HBM 贴满 64 GiB），128（513 帧）activation OOM**。加卡主要省显存不加速；8 卡必须 `num_workers=4`。HBM 无逐步泄漏，host RSS 是 DataLoader prefetch 灌队列。

---

## 2. 代码改动

| 日期 | 提交 | 说明 |
|------|------|------|
| 9/4 | `c2b3045` | 整 episode SFT 接 `keep_ranges`：有效段拼成一条 sample，空段建索引时丢掉；视频按连续段解码，避开 idle gap。配套 `cosmos` `c83d99e`。 |
| 9/8 | `068adef` | 去掉 packed-token / teacher-forcing 序列长度上限，不再按 token budget 截断。Packing 仍要求 `max_sequence_length` 与 `max_samples_per_batch` 二选一。 |
| 9/8 | `970836b` | 丢掉不足 33 帧的 episode；超长样本直接丢而不是截断；连续 keep_ranges 一次 `get_frames_at`；packed-mp4 decoder 用完即关（`cache_size=0`）；shard 不足时回绕。 |
| 9/8 | `423b1f9` | policy 模式的 action JSON 不再带 clip timeline，避免 UND 文本泄露 episode horizon。 |
| 9/8 | `085635d` | wandb 0.19+ `generate_id` 兼容；`torch_npu` 钉到 `2.10.0.post4`（对齐 CANN 9.1）。 |

配方侧：`_PRE_TF_EXTRA_TOKENS` 从 155 提到 500（480p 帧帽约 477）；DROID Edge 默认 `max_samples_per_batch=1`。

环境：conda `cosmos-framework`（Python 3.11 + `torch_npu==2.10.0.post4`，CANN 9.1）。多卡启动需要 `HCCL_OP_EXPANSION_MODE=AIV`。

---

## 3. 训练显存与 iter 时间

主扫测机器：`liteserver-8da9-2`，Ascend **910B3 × 8**，每卡 HBM 64 GiB。
启动脚本：`cosmos/cookbooks/cosmos3/generator/action/finetune/launch_sft_action_policy_droid_edge_local.sh`
（`model=mot_causal_fsdp`，整 episode `chunk_length=-1`，`max_samples_per_batch=1`，`grad_accum_iter=1`；global batch = 卡数）。

扫测日期：2026-09-03 起（本周）。单卡对照在另一台 910C（Phy-14，8/31，见 [3.10](#310-910c-单卡--多卡对照)）。

### 3.1 结论

- **卡数加多主要省显存，不加速。** 每卡仍是 1 条整 episode，compute 是 activation-bound。32 档 2/4/8 卡都是 ~10 s，64 档都是 ~27–28 s。
- **480p 能稳定跑的上限是 `max_episode_blocks=120`（481 帧）；128（513 帧）activation OOM，再申请 7.35 GiB。**
- **8 卡必须 `num_workers=4`**，否则数据是卡点（nw=1 要 45 s，nw=2 要 21 s）。
- HBM **没有逐步泄漏**；host RSS 是 DataLoader prefetch 先灌满再按 episode 长短振荡。
- 256p 可以把 128 跑满（峰值仅 20.5 GiB），但 decode 跟不上，avg 被气泡拉高。
- full AC **压不下来** 96 档峰值，也救不了 128 的 7.35 GiB。
- 去掉 `max_samples_per_batch=1`、只按 48000 token packing，会把主机打到 **1462 GiB** 被 OOM killer 杀掉。

### 3.2 方法与数据

- 覆盖 `dataloader_train.dataloader.datasets.droid.dataset.max_episode_blocks`。
- 观测帧数上限 = `1 + 4 * max_episode_blocks`（Wan VAE 的 `4N+1`）。
- IterSpeed 每步打一条 rank0 `Time:`（`hit_thres=50`）。2/4/8 卡用 `trainer.max_iter=21`，凑齐 21 条后杀掉（收尾 checkpoint 不计入）。
- **主指标 `avg_skip1_s` 丢掉第 1 条**（含首次 forward / NPU 预热），对第 2–21 条（共 20 个 `Time`）算术平均。`min_s` / `max_s` 也只在这 20 条里取。
- `peak_hbm_mb` 来自该 run 期间 `npu-smi info` 各可见卡峰值。开测前空载约 Phy-0/1 **3411 MB**，Phy-2/3 **3203 MB**。
- 每个 (卡数, N) 独立 `job.name` / `OUTPUT_ROOT`，避免 auto-resume。
- `model.config.teacher_forcing_visualize_sdpa_mask=false`。默认 `HCCL_OP_EXPANSION_MODE=AIV`。

**数据集（9/3 复核）**：`/data5T/Embodied-AI/datasets/droid_plus_lerobot_640x360_20260412` → `Cosmos3-DROID`，磁盘 3.9 GiB。

| 项 | 数量 |
| --- | --- |
| meta / episodes parquet 行数 | 200 |
| 帧数（`sum(length)`） | 86398 |
| episode 长度 min / p50 / mean / p90 / max | 1 / 413 / 432 / 604 / 1122 |
| `split_val_ratio=0.03` 后 train episode | 194 |
| 训练实际 shuffle blocks（`chunk_length=-1`） | **193** |

不回绕上限：`num_workers ≤ floor(193 / 卡数)` → 8 卡最多 24。实际卡点是 AV1 解码，不是 shard 数。早先 25-episode subset 上 8 卡 `nw=4` 会 32 shard > 25 而 hang，已修 `ActionIterableShuffleDataset` 回绕；当前 193 条下不再回绕。

### 3.3 2 / 4 / 8 卡主表（480p）

#### 2 卡（Phy 0+1，`num_workers=4`）

| max_episode_blocks | 帧数上限 (4N+1) | 状态 | n_logs | first_s | avg_skip1_s（20 次） | min_s | max_s | peak_hbm_mb | last_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 129 | ok | 21 | 78.48 | **9.97** | 9.84 | 10.61 | 32809 / 32809 | 2.277 |
| 64 | 257 | ok | 21 | 135.97 | **27.18** | 26.83 | 29.06 | 45495 / 43577 | 2.252 |
| 128 | 513 | **oom**（第 2 step） | 1 | 242.25 | — | — | — | 64150 / 63280 | 49.920 |

峰值约 32.0 / 44.4 / 62.6 GiB。128 只有第 1 条 Time，没有 20 次平均。

#### 4 卡（Phy 0–3，`num_workers=4`）

| max_episode_blocks | 帧数上限 (4N+1) | 状态 | n_logs | first_s | avg_skip1_s（20 次） | min_s | max_s | peak_hbm_mb | last_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 129 | ok | 21 | 80.29 | **10.05** | 9.86 | 10.63 | 25240 / 25243 / 25242 / 25242 | 1.892 |
| 64 | 257 | ok | 21 | 143.04 | **27.34** | 26.86 | 29.63 | 37948 / 37949 / 37949 / 36033 | 2.045 |
| 128 | 513 | **oom**（第 4 step） | 3 | 264.49 | — | — | — | 64913 / 58892 / 63247 / 65033 | 24.270 |

峰值约 24.7 / 37.1 / 63.5 GiB。128 只打出 3 条 Time（第 2、3 步约 92–94 s），不是 20 次平均。

#### 8 卡（Phy 0–7，`num_workers=4`）

| max_episode_blocks | 帧数上限 (4N+1) | 状态 | n_logs | first_s | avg_skip1_s（20 次） | min_s | max_s | peak_hbm_mb | last_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 129 | ok | 21 | 84.98 | **10.36** | 9.85 | 14.11 | 21697–21698 | 2.002 |
| 64 | 257 | ok | 21 | 148.21 | **28.03** | 26.97 | 29.14 | 33418–33420 / phy3 **42201** / phy5 39315 | 2.002 |
| 96 | 385 | ok | 21 | 250.17 | **55.95** | 52.49 | 60.74 | phy0 **65517** / phy7 65020 / 其余 46–63 GiB | 1.840 |
| 128 | 513 | **oom**（第 14 step） | 13 | 248.93 | — | 68.56 | 96.01 | 65529（多卡贴满） | 2.615 |

32 / 64 的 avg 已对齐 2/4 卡（10.36 vs 9.97；28.03 vs 27.18）。96 档 min/max 只有 16% 散度，数据不是卡点；HBM 峰值贴满 Phy-0 但仍跑满 21 step。128 第 2–13 步 68.6–96.0 s（12 条，**不是 20 次平均**）。

#### 对照

| max_episode_blocks | 2 卡时间 / HBM | 4 卡时间 / HBM | 8 卡时间 / HBM |
| --- | --- | --- | --- |
| 32 | 9.97 s / 32.0 GiB | 10.05 s / 24.7 GiB | **10.36 s / 21.2 GiB** |
| 64 | 27.18 s / 44.4 GiB | 27.34 s / 37.1 GiB | **28.03 s / 32.6 GiB**（尖峰 41.2） |
| 96 | — | — | **55.95 s / 64.0 GiB**（贴满，未 OOM） |
| 128 | oom 第 2 step / 62.6 | oom 第 4 step / 63.5 | oom 第 14 step / 64.0 |

64 / 32 稳态时间比约 **2.73×**，比帧数翻倍更陡，符合 attention 随序列变长。

128 各卡数都是同一块 **7.35 GiB** activation 申请失败。多卡只是 FSDP 把参数占用从 ~49 GiB 降到 ~40 GiB，513 帧 TF 的这块激活仍然装不下。8 卡比 2 卡多撑几步，是因为 193 条里短 episode 更多，不是 128 档突然能跑满。

### 3.4 8 卡 `num_workers` 搜索

判定：丢掉第 1 条后，`avg` 落在 2 卡 GPU 下限 9.97 s 的 1.25× 以内，且 `max/min < 1.5`，则数据不是卡点。recipe 默认 `num_workers=4`。blocks=32，193 train episode。

| num_workers | shards（8×nw） | 状态 | avg_skip1_s | min / max | 相对 2 卡 9.97 s | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | ok（旧 25-ep 子集） | 45.08（20 次） | 33.96 / 62.60 | 4.5× | 数据卡死 |
| 2 | 16 | ok（旧 25-ep 子集） | 20.97（20 次） | 9.87 / 40.52 | 2.1× | min 已贴 GPU，avg 仍有气泡 |
| **4** | **32** | **ok** | **10.36（20 次）** | **9.85 / 14.11** | **1.04×** | **数据不是卡点** |
| 8 | 64 | ok | 10.52（10 次） | 10.27 / 11.15 | 1.06× | 不比 4 快；无回绕 |

`num_workers=4` 的 20 条稳态里，只有第 4 步 14.11 s 偏高，其余 9.85–10.53 s。`num_workers=8` 已是 GPU 下限，再加 worker 不会缩短 compute。Pre-warm（nw=4）：各 rank 43–47 s 出第一条，barrier 一次通过。

25-ep 子集历史对照（当时 `nw=4` 会 hang，只用了 1 和 2）：

| blocks | nw | avg_skip1_s | min / max | 峰值 HBM |
| --- | --- | --- | --- | --- |
| 32 | 1 | 45.08 | 33.96 / 62.60 | ~21.2 GiB |
| 32 | 2 | 20.97 | 9.87 / 40.52 | ~21.2 GiB |
| 64 | 1 | 90.09 | 69.75 / 119.29 | ~32.6 GiB |
| 64 | 2 | 40.58 | 26.91 / 62.15 | 32.6 / phy3 41.9 |
| 128 | 2 | oom 第 6 step | — | 64.0 GiB |

### 3.5 480p OOM token 边界（100–120 / `-1`）

在 96（ok、贴满 64 GiB）和 128（OOM、再申请 7.35 GiB）之间按 4 步进。8 卡、`num_workers=4`、selective AC、TF S/K 随机、21 step。packed 按双流

`UND + 2 × ((1+N)×391 + (1+4N))`，UND≈151。

100–120 仍是旧 200-episode subset。`-1` 是新进程，吃到 **500 episode**（train 485，keep_ranges 留下 483 / 129784 帧）。

| max_episode_blocks | 帧 cap | vis / packed | 状态 | first_s | avg_skip1_s | min / max | 峰值 HBM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 96 | 385 | 37927 / ~7.7 万 | ok | 250.17 | **55.95** | 52.49 / 60.74 | phy0 **65517** |
| 100 | 401 | 39491 / 79935 | ok | 214.92 | **59.03** | 56.44 / 66.70 | 65518 |
| 104 | 417 | 41055 / 83095 | ok | 226.84 | **62.51** | 56.45 / 68.68 | 65527 |
| 108 | 433 | 42619 / 86255 | ok | 218.59 | **67.31** | 64.28 / 74.15 | 65517 |
| 112 | 449 | 44183 / 89415 | ok | 218.06 | **70.61** | 65.22 / 77.10 | 65515 |
| 116 | 465 | 45747 / 92575 | ok | 237.82 | **74.04** | 67.05 / 81.71 | 65530 |
| **120** | **481** | **47311 / 95735** | **ok** | **242.77** | **76.92** | **66.62 / 84.35** | phy4 **65527** |
| 128 | 513 | 50439 / ~10.2 万 | oom 第 14 step | 248.93 | — | 68.56 / 96.01 | 65529；再申请 **7.35 GiB** |
| **-1** | 不截断 | 最坏 concat 1100 帧 / ~21.8 万 | **oom 第 10 step** | 104.15 | — | 34.61 / 91.27（8 条稳态） | 65513；再申请 **9.79 GiB** |

**480 档 cap 边界：120 能跑满，128 爆。** packed 大约 9.6 万–10.2 万。100–120 峰值都贴 64 GiB，但 PyTorch 没有再要那块 7.35 GiB。

**`-1` 会爆，说明不是「数据不够长」。** keep_ranges 拼接后 500 条里仍有 51 条 >513 帧、最长 1100。keep_ranges 平均只丢掉约 10% 帧，不是 OOM 扫不到的原因。

128 OOM 原文（8 卡 nw=4，第 14 step）：

```text
NPU out of memory. Tried to allocate 7.35 GiB
(NPU 1; 60.96 GiB total capacity; 40.18 GiB already allocated;
 40.18 GiB current active; 6.78 GiB free;
 52.64 GiB reserved in total by PyTorch)
```

`-1` OOM 原文（第 10 step）：

```text
NPU out of memory. Tried to allocate 9.79 GiB
(NPU 0; 60.96 GiB total capacity; 40.79 GiB already allocated;
 40.79 GiB current active; 8.93 GiB free;
 50.70 GiB reserved in total by PyTorch)
```

### 3.6 256p

concat 640×540 等比缩到 303×256 再 pad 到 **320×256**。每 latent 帧 vision 格子 391→**80**。其余与 480 档相同：8 卡、`num_workers=4`、TF S/K 随机、`trainer.max_iter=21`。

| max_episode_blocks | 帧数上限 | 状态 | first_s | avg_skip1_s（20 次） | min / max | 峰值 HBM | 480 对照 avg / HBM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 129 | ok | 97.72 | **9.70** | **3.07** / 31.54 | **14.8 GiB** | 10.36 s / 21.2 GiB |
| 64 | 257 | ok | 121.69 | **19.17** | **3.64** / 56.49 | **15.7 GiB** | 28.03 s / 32.6 GiB |
| 96 | 385 | ok | 170.16 | **27.46** | **5.18** / 76.66 | **18.3 GiB** | 55.95 s / 64.0 GiB |
| **128** | **513** | **ok 21 step** | **186.36** | **34.64** | **6.68** / 106.74 | **20.5 GiB** | **oom** / 64.0 GiB |

**128 在 256 能跑满**，峰值约 21 GiB，离 64 GiB 还很远。平均时间被 DataLoader 气泡拉高：GPU 下限是 `min_s`（32 档 3.1 s、128 档 6.7 s），`avg` 常到 3–5×，因为整 episode AV1 解码没变、compute 变快后 GPU 在等 CPU。480 档 32/64 的 min≈avg，那时 compute 更慢、prefetch 跟得上。

### 3.7 full activation checkpointing

对照 `selective`（`save_ops_regex=["fmha"]`）。其余不变：8 卡、`num_workers=4`、TF S/K 随机 1–4 / 1–32。

| max_episode_blocks | AC | 状态 | n_logs | first_s | avg_skip1_s（20 次） | min / max | 峰值 HBM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 96 | selective | ok | 21 | 250.17 | **55.95** | 52.49 / 60.74 | phy0 **65517** |
| **96** | **full** | **ok** | **21** | **255.02** | **54.29** | **52.64 / 58.76** | phy2 **65527** |
| 128 | selective | oom 第 14 step | 13 | 248.93 | — | 68.56 / 96.01（12 条） | 65529 |
| 128 | full | oom 第 12 step | 11 | 257.97 | — | 52.80 / 93.86（10 条） | 64548 |

96 + full 跑满 21 step。时间和 selective 同一档（54.3 vs 56.0 s），HBM 峰值仍贴满 64 GiB，**full 没有把 96 的显存峰值压下来**。128 + full 仍 OOM，原文与 selective 几乎逐字相同（再申请 7.35 GiB）。`full` 只 checkpoint transformer block，这块 7.35 GiB 不在层激活里。

### 3.8 固定 TF S=8 / K=10，以及有没有漏

覆盖 `teacher_forcing_block_size_{min,max}=8`、`teacher_forcing_history_blocks_{min,max}=10`（不再每步随机 1–4 / 1–32）。21 step。

| max_episode_blocks | 帧数上限 | 状态 | first_s | avg_skip1_s（20 次） | min / max | 峰值 HBM | 对照随机 S/K avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 129 | ok | 84.45 | **10.29** | 9.92 / 11.31 | 21.2 GiB（全卡 21700） | 10.36 |
| 64 | 257 | ok | 147.43 | **27.90** | 26.92 / 29.11 | 32.6 GiB 典型 / phy4+7 **41.2** | 28.03 |
| 96 | 385 | ok | 227.74 | **56.80** | 53.73 / 63.83 | phy0/2/7 贴满 **65513** | 55.95 |

时间和显存峰值与随机 S/K 几乎一样。S=8 时 32 档只有 `ceil(33/8)=5` 个因果 block，K=10 等于看全文，但 21 step 平均并不因此变快/变慢。

**HBM（`npu-smi`，逐步采样）没有泄漏。**

- 32：warmup 后 `hbm_max` 钉在 **21700–21701 MB**，21 步 delta **0 MB**。
- 64：`hbm_max` **42205 → 42207**（+2 MB）。八卡之和在 iter 9 一次跳 +0.8 GiB（allocator 多 reserve），之后完全持平。
- 96：iter 2–3 碰到长 episode，单卡峰值 62.8 GiB，随后稳定在 **49438 MB**。iter 13–21 之和 +5.2 GiB，但 `hbm_max` 不变——是偏低的卡把 cache 补齐，不是逐步堆积 activation。

**Host RSS 没有逐步泄漏，是 DataLoader prefetch 先灌满再按 episode 长短振荡。**

采样所有 `cosmos_framework.scripts.train` 进程 VmRSS（含 spawn worker；8 rank × (1+4 worker) ≈ 42 个进程）：

- 64 档：3 分钟内 733 → ~1048 GiB（prefetch 灌队列），随后 8 分钟只到 **1054 GiB**（+6 GiB），结束立刻回到 ~23 GiB。
- 96 档：同样先灌到 ~1050 GiB，之后在 **990–1149 GiB** 之间随 385 帧整段视频进出队列振荡，没有单调爬升；run 结束同样回到 ~25 GiB。

机器总内存 1.5 TiB，96 档 Available 最低约 374 GiB，没有 OOM killer。

### 3.9 只设 `max_sequence_length=48000` 的 packing

目标：整 episode、超长丢掉、短的拼到 48000 展开前 token，跑 100 step。两次都在 **主机 OOM killer** 杀掉 `pt_data_worker` 后挂，不是 NPU 再申请 7.35 GiB 那种爆。机器约 1.5 TiB，RSS 到 **1462 GiB**。

| 次 | prefetch | 状态 | n_logs | first_s | avg | min / max | 峰值 HBM | 峰值 RSS | discard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4（recipe） | worker Killed 第 5 step 前 | 4 | 522.6 | 71.5 | 71.3 / 72.0 | 65516 | **1462 GiB** | 4 |
| 2 | **1** | worker Killed 第 10 step 前 | 9 | 471.2 | **110.3** | 61.8 / 222.0 | phy7 **65535** | **1462 GiB** | 19 |

丢掉超长是生效的（51107–91027 token 的样本被 Discard）。拼 batch 后每步都接近 120 档的 GPU 负载，HBM 峰值贴满，但 **HBM 逐步不单调爬升**。主机 RSS 9 步里 1117→1421 GiB，是 DataLoader 队列里堆了多条整段视频，不是 NPU leak。

`prefetch_factor=1` 只多撑了几步，同样撞上 1.46 TiB。这个 packing 模式（一条 batch 多条整 episode × 8 卡 × 4 worker）在这台机器上跑不长。要长跑应加回 `max_samples_per_batch=1`，或再降 `num_workers`。

### 3.10 910C 单卡 / 多卡对照

机器：`liteserver-mi-dev-00001`，Ascend 910C。单卡 Phy-14 扫测日期 2026-08-31（上周 week4 已记一版）；本周把多卡 HCCL 问题钉死，并作为 910B3 扫测的基线。

#### 单卡 Phy-14（50 step，丢掉第 1 条后 49 次平均）

| max_episode_blocks | 帧数上限 (4N+1) | 状态 | first_s | avg_skip1_s | min_s | max_s | peak_hbm_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 65 | ok | 19.40 | **3.21** | 3.16 | 3.28 | 47988 |
| 32 | 129 | ok | 51.79 | **7.90** | 7.83 | 8.08 | 49092 |
| 64 | 257 | ok | 89.37 | **22.04** | 21.96 | 22.31 | 62525 |
| 128 | 513 | **oom** 第 1 step | — | — | — | — | 65140 |

开测前卡上约有 ~10.8 GiB 残留占用，峰值含这部分。128 原文：`Tried to allocate 1.30 GiB`（61.27 GiB capacity，49.92 already allocated）。

单卡稳态比 910B3 2 卡更快（32 档 7.90 vs 9.97 s），但 128 同样爆。丢掉第 1 条后，16→32→64 时间约 2.46× / 2.79× 增长。

#### 多卡 HCCL

默认路径（AICPU KFC `RunAicpuKfcResInitV2`）在这台机上 **普遍失败 `errorCode=0x2a`**，与卡对拓扑无关（同板 SIO / 跨板 HCCS_SW 都炸）。最小复现：`torchrun --nproc_per_node=2` + `hccl` + `all_reduce`。

| 设置 | 结果 |
| --- | --- |
| 默认 HCCL | `0x2a`，0 条 IterSpeed |
| `HCCL_OP_EXPANSION_MODE=AIV` | 小消息 `all_reduce` / `broadcast` / ≤1 MiB `all_gather` **ok** |
| AIV + 大 `all_gather`（64/512 MiB）或训练 `sync_model_states` | 仍 `0x2a` |
| `HCCL_OP_EXPANSION_MODE=Host` | 910C 非法（只允许 `AI_CPU \| AIV`） |

所以 2/4/8 卡数字全部来自 910B3；910C 多卡训练没进入 loss。以后多卡启动带：

```bash
export HCCL_OP_EXPANSION_MODE=AIV
```

---

## 4. Host RSS / dataloader 探测

正式训练之外，本周后半专门测了 DROID Edge dataloader 的 host RSS（不加载模型 / VAE / NPU）。脚本：`cosmos_framework/scripts/probe_droid_edge_dataloader_rss.py`（未提交）。

条件：keep_ranges、uint8、PyAV / torchcodec、256/480、开关增强、`num_workers=1`、`prefetch_factor=1`、`max_samples_per_batch=1`、`video_decoder_cache_size=0`。

结论：

- 单 rank dataloader（训练路径 `nw=4`）灌满后约 **73 GiB 封顶，无明显泄漏**，像素体积对得上 uint8。
- torchcodec 把前 20 个 episode 循环 70–100 step：第一遍后 worker 钉在约 **8.7–9.4 GiB 平台期，无逐步泄漏**。
- 8 卡训练曾被幽灵 HBM / HCCL 挡住，所以完整 8 卡长跑的 RSS 以 [3.8](#38-固定-tf-s8--k10以及有没有漏) 的 sidecar 为准（prefetch 灌到 ~1.05 TiB 后振荡，结束回到 ~25 GiB）。
- `worker_restart_every_n=0` 非法（必须 `None` 或 ≥1）。epoch 重启当时尚未实现，训练仍只跟 `max_iter`；工作区已接 `worker_restart_every_epoch` + `begin_new_iterator`，未提交、未在正式训练里验过。

解码改动（已进 `970836b`）：packed-mp4 按 `from_timestamp` 整数帧号只解当前 episode；连续 keep_ranges 一次读完，不再按块 open/decode/close。

---

## 5. 未提交与待办

工作区脏文件（`cosmos-framework`）：

- dataloader：按 epoch 重启 worker（`worker_restart_every_epoch` + `begin_new_iterator`），以及 probe 用的 `max_episodes`
- 探测脚本 `probe_droid_edge_dataloader_rss.py`
- DROID dataflow 图、若干 `max_episode_blocks` 扫参笔记、teacher-forcing block token 说明
- `AGENTS.md` / cosmos3 skill：本机 conda、CANN、`LD_LIBRARY_PATH`、hf-mirror 下载约定

待办：

1. 5 个训练相关 commit **还没 push**。
2. 480p 8 卡长跑应把 cap 放在 **120（481 帧）**，不要开 128 / `-1`。
3. 8 卡训练仍可能被幽灵 HBM / HCCL 挡住，需要干净卡面。
4. epoch worker 重启接到 dataloader 后，还没合进提交、也还没在正式训练里验过。
5. 多 episode packing（`max_sequence_length=48000`）在 1.5 TiB 主机上跑不长，长跑保持 `max_samples_per_batch=1`。

---

## 6. 附录：原始 log

均在 `cosmos/cookbooks/cosmos3/generator/action/finetune/outputs/bench_max_episode_blocks/`：

| 内容 | 路径 |
| --- | --- |
| 2 卡 | `logs_dp2/blocks_<N>.log`、`results_dp2_dp4.tsv` |
| 4 卡 | `logs_dp4/blocks_<N>.log` |
| 8 卡 nw 搜索 | `logs_dp8_nwsweep/`、`results_dp8_nwsweep.tsv` |
| 8 卡 S=8/K=10 + 内存采样 | `logs_dp8_s8k10/`、`mem_s8k10/`、`results_dp8_s8k10.tsv` |
| 8 卡 480 OOM token | `logs_dp8_oom_tokens/`、`results_dp8_oom_tokens.tsv` |
| 8 卡 `-1` | `logs_dp8_unbounded/blocks_neg1.log` |
| 8 卡 256p | `logs_dp8_res256/`、`results_dp8_res256.tsv` |
| 8 卡 full AC | `logs_dp8_acfull/`、`results_dp8_acfull.tsv` |
| packing 48000 | `logs_dp8_seq48000/`、`mem_seq48000/` |
| 扫测脚本 | `run_bench_dp2_dp4.py`、`run_bench_dp8_nw.py`、`run_bench_dp8_s8k10.py`、`run_bench_dp8_acfull_128.py`、`run_bench_dp8_res256.py`、`run_bench_dp8_oom_tokens.py`、`run_bench_dp8_unbounded.py` |

工作区未提交的扫参笔记：`cosmos-framework/docs/max_episode_blocks_npu0_3_dp2_dp4_iter_time.md`（本周主表原文）。
