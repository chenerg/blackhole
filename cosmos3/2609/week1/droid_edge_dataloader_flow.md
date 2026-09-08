# DROID Edge 整 episode：Dataset Loader 载入流程与组 Batch 参数

针对当前训练配方 `action_policy_droid_edge`（本机入口 `launch_sft_action_policy_droid_edge_local.sh`）。
窗口模式（`chunk_length>0`，如 Nano `chunk_length=32`）只在对照处提及。

仓库：`cosmos-framework`。本文描述的是 **DataLoader 产出、交给 `train_step` 的 packed batch**；VAE 编码和 teacher-forcing token 交错发生在模型 GPU 侧，见文末。

---

## 目录

1. [总览：一条 sample 怎么变成训练 batch](#1-总览一条-sample-怎么变成训练-batch)
2. [调用栈（谁包谁）](#2-调用栈谁包谁)
3. [阶段 0：磁盘源](#3-阶段-0磁盘源)
4. [阶段 1：建扁平索引（只读 meta，不解码）](#4-阶段-1建扁平索引只读-meta不解码)
5. [阶段 2：rank × worker 分片与打乱](#5-阶段-2rank--worker-分片与打乱)
6. [阶段 3：整 episode 解码](#6-阶段-3整-episode-解码)
7. [阶段 4：组成 raw sample（concat_view + 8D action）](#7-阶段-4组成-raw-sampleconcat_view--8d-action)
8. [阶段 5：ActionTransformPipeline](#8-阶段-5actiontransformpipeline)
9. [阶段 6：PackingDataLoader 组最终 batch](#9-阶段-6packingdataloader-组最终-batch)
10. [关键参数一览](#10-关键参数一览)
11. [长度预算怎么算（4N+1 / token / HBM）](#11-长度预算怎么算4n1--token--hbm)
12. [模型侧（不在 loader 里）](#12-模型侧不在-loader-里)

---

## 1. 总览：一条 sample 怎么变成训练 batch

```
DROID LeRobot 磁盘
  info.json + episodes.parquet + packed mp4
        │  建索引（丢短 / 丢超长 / keep_ranges 拼接）
        ▼
  _episode_records  每条 episode 一个 sample（valid_len=1）
        │  rank × worker 打乱 episode 顺序，集内顺序读
        ▼
  _fetch_whole_episode
        │  15 fps 抽帧 → 可选 keep_ranges 拼接 → 4N+1 尾填充
        │  三路相机 uint8 + joint/gripper 表
        ▼
  concat_view 视频 [C,T,H,W] uint8  +  8D action [T,8]（含初始 state）
        │  ActionTransformPipeline
        ▼
  736×544 反射 pad、JSON prompt tokenize、SequencePlan、action pad 到 64D
        │  DataLoader batch_size=1，num_workers=4
        ▼
  PackingDataLoader（max_samples_per_batch=1）
        │
        ▼
  训练 batch：video / action / text_token_ids 都是 list[list[Tensor]]，
              每卡每步恰好 1 条整 episode
```

当前默认：**每卡每步 1 条整 episode**。加卡只增加 global batch（= 卡数 × `grad_accum_iter`），不把多条 episode 拼进同一个 micro-batch。

本机 launch 会覆盖配方里关掉的 keep_ranges：

```text
dataloader_train.datasets.droid.dataset.use_filter_dict=True
dataloader_train.datasets.droid.dataset.filter_dict_path=$DROID_KEEP_RANGES
```

配方文件本身 `use_filter_dict=False`。下文按 **本机实际训练路径（keep_ranges 开）** 写，并标明关 filter 时的差异。

---

## 2. 调用栈（谁包谁）

从外到内：

| 层 | 类 | 文件 |
| --- | --- | --- |
| 训练配方 | `action_policy_droid_edge` | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_edge.py` |
| 组 batch | `PackingDataLoader` | `cosmos_framework/data/generator/joint_dataloader.py` |
| 按卡分数据 | `RankPartitionedDataLoader` | 同上 |
| PyTorch worker | `DataLoader(batch_size=1, num_workers=4, collate=custom_collate_fn)` | 同上 |
| episode 打乱流 | `ActionIterableShuffleDataset` | `.../action/datasets/action_sft_dataset.py` |
| 变换 | `ActionSFTDataset` → `ActionTransformPipeline` | `action_sft_dataset.py` / `action/transforms.py` |
| 数据本体 | `DROIDLeRobotDataset` | `.../action/datasets/droid_lerobot_dataset.py` |
| 索引 / 整集解码 | `BaseActionLeRobotDataset` | `.../action/datasets/cosmos3_action_lerobot.py` |

工厂：`get_action_droid_sft_dataset(...)`。Hydra 覆盖路径形如：

```text
dataloader_train.dataloader.datasets.droid.dataset.<参数>=...
dataloader_train.max_samples_per_batch=1
```

内层 `dataloader=` 是 `RankPartitionedDataLoader`，所以 dataset 参数要写 `dataloader_train.dataloader.datasets.droid.dataset.*`。

---

## 3. 阶段 0：磁盘源

| 项 | 本机默认 |
| --- | --- |
| `DROID_ROOT` | `/data5T/Embodied-AI/datasets/droid_plus_lerobot_640x360_20260412` |
| 版本键 | 父目录 **basename** 必须是 `LEROBOT_ROOTS` 的 key，不要指到 `success/` |
| `use_success_only=True` | 只注册 `success/` shard |
| 相机 | 640×360 AV1 packed mp4：wrist + 左右肩 |
| 原生 FPS | 通常 30；训练 `fps=15` → 整数 stride=2 |
| keep_ranges JSON | `$DROID_ROOT/keep_ranges_1_0_1_whole_episode.json` |

`LeRobotDataset` 在 `__getitem__` 时才真正打开视频（LRU）。建索引只读 `info.json` + `episodes.parquet` + `tasks.parquet`。

---

## 4. 阶段 1：建扁平索引（只读 meta，不解码）

`DROIDLeRobotDataset.__init__` 立刻 `_register_sources()`，填两张表：

- `_episode_records[i] = (ds_idx, sample_start, valid_len, episode_id)`
- `_episode_cum_ends[i]`：到第 i 个 span 为止的样本数，给 `__getitem__` 做 `bisect`

整 episode 模式（`chunk_length=-1`）每个留下的 episode **恰好 1 个 sample**（`valid_len=1`），锚在该集第一帧。

### 4.1 先选哪些 episode

1. `split_episode_ids(total_episodes, seed=42, val_ratio=0.03, split=train)`  
   当前 200 条 meta → train **194**（再经过后面的丢弃）。
2. `build_episode_spans(..., whole_episode=True)`  
   丢掉只有 1 帧、没有 action step 的集。
3. 最短长度：DROID 固定 `_DROID_MIN_EPISODE_FRAMES=33`（对齐窗口配方 `chunk_length=32` 的 33 帧观测）。  
   - **关 keep_ranges**：看原始 `length`。  
   - **开 keep_ranges**：看拼接后的 concat 长度。
4. 超长丢掉（**截断的反面**）：`max_episode_length_frames`。  
   配方不直接设帧帽，而是 `max_pre_tf_tokens=48000` + `resolution=480` → 算出 **477 帧**（见 [§11](#11-长度预算怎么算4n1--token--hbm)）。  
   超过的 episode **不进索引**，worker 永远不会解码它们。  
   这和 `max_episode_blocks` 不同：后者是解码时 **截断**。

### 4.2 keep_ranges（本机开）

JSON：`{episode_id 或 KarlP GCS key: [[start, end), ...]}`，帧号是该集内部 `[0, length)`。

对每个 train episode：

1. 查 ranges；没有 key → **整集丢弃**。
2. `clip_droid_keep_ranges` 裁到 `[0, length)`，丢掉空区间。
3. `concat_len = sum(end-start)`；`< 33` → 丢。
4. 把 concat 长度按 15 fps、4N+1 估一把 padded 帧数；`> max_episode_length_frames` → 丢。
5. 留下的写入 `_keep_ranges_by_episode[ep_id] = clipped`，并追加一条 index record。

窗口模式（`chunk_length>0`）开 filter 时，每个 range 各自滑窗，一条 episode 会变成多条 sample。整 episode 模式是 **多段拼成一条**。

---

## 5. 阶段 2：rank × worker 分片与打乱

### 5.1 RankPartitionedDataLoader

当前只有 `droid` 一个 dataset、`ratio=1`，所以 **所有卡都跑 DROID**。  
写入 `dataset.shard_world_size = 卡数`、`dataset.shard_rank = 本卡`。  
再包一层 `torch.utils.data.DataLoader`：

| 参数 | 配方默认 | 作用 |
| --- | --- | --- |
| `batch_size` | 1 | 每个 worker 吐 1 条 sample；真正组 batch 在 Packing |
| `num_workers` | 4 | 每卡 4 个解码进程。8 卡 = 32 shard |
| `prefetch_factor` | 2 | 每个 worker 预取 2 个 batch；整集 RGB 很大，4 会把 host 顶到 ~1 TiB |
| `persistent_workers` | True | worker 进程跨 epoch 活着（FFmpeg 状态不随 iterator 结束释放） |
| `pin_memory` | True |  |
| `in_order` | False |  |

### 5.2 ActionIterableShuffleDataset

`iterable_shuffle=True` 时工厂返回这个 Iterable，而不是 map-style。

每个 `(rank, worker)`：

```
global_shard = shard_rank * num_workers + worker_id
total_shards = shard_world_size * num_workers
stride = min(total_shards, n_blocks)
my_shard = global_shard % stride
```

- `get_shuffle_blocks()` 从 `_episode_cum_ends` 得到每集一块 `(start, length)`。整 episode 时 `length=1`。
- 所有 shard 用同一个 `seed + epoch` 做 `randperm`，再按 `order[my_shard::stride]` 取互斥子集 → **跨卡、跨 worker 不重复**（除非 shard 数 > episode 数，才会 wrap 并打 warning）。
- 块内按下标顺序 `__getitem__`。整 episode 每块只有 1 个 idx，等价于按打乱后的 episode 顺序读。
- 默认 **无限循环**，训练靠 `trainer.max_iter` 停。`worker_restart_every_n` / `worker_restart_every_epoch` 可以提前结束 iterator，让 `persistent_workers=False` 的 worker 带着 FFmpeg 一起死掉；**当前配方未开**。

8 卡 × 4 worker = 32 shard。train episode 大约 193（无 filter）或更少（有 keep_ranges）。只要 `32 < n_blocks` 就不会回绕。

---

## 6. 阶段 3：整 episode 解码

`__getitem__(idx)` → `_fetch_sample` → 因 `chunk_length=-1` 走 `_fetch_whole_episode`，**绕开** `LeRobotDataset.__getitem__` 的固定 `delta_timestamps` 窗口。

### 6.1 行号：`_whole_episode_rows`

DROID 覆盖了基类：

1. 若该集有 keep_ranges：把各 `[start,end)` 的绝对行号 **拼成一条 native 序列**（idle gap 不在序列里）。
2. 否则：`range(dataset_from_index, dataset_to_index)`。
3. 按 `native_fps / fps` 整数 stride 抽到 15 fps。
4. `_finalize_whole_episode_obs_rows`：
   - 若 `max_episode_blocks != -1`：**截断**到 `1 + 4 * max_episode_blocks` 帧（当前配方 **-1 = 不截断**；超长已在索引阶段丢掉）。
   - 要求至少 2 帧。
   - 帧数 **向上取整到 4N+1**（Wan VAE），不够的用 **最后一行重复**。
   - `obs_rows` 长度 `target_t`；`act_rows = obs_rows[:target_t-1]`。

### 6.2 表与视频

- 表：`ds._query_hf_dataset`，joint action / gripper action 走 act 网格，joint/gripper state 走 obs 网格（`use_state=True`）。
- 视频：把 `obs_rows[:n_real]` 按 stride 切成 **连续 run**，每个 run 一次 `get_frames_at`（packed mp4 的整数帧号，夹在该集 `from_timestamp`/`to_timestamp` 里）。keep_ranges 的 gap **不会被解码**。
- 后端：`video_backend` 默认 torchcodec，可改 `pyav`。
- `video_decoder_cache_size=0`：每个 packed mp4 用完即关，不做 64 槽 LRU（整集流几乎不复用同一文件，LRU 只会堆 FFmpeg 状态）。
- 尾填充在 tensor 上重复最后一帧到 `target_t`。输出仍是 **uint8 `[T,C,H,W]`**。

---

## 7. 阶段 4：组成 raw sample（concat_view + 8D action）

### 7.1 视频 `concat_view`

每帧拼成：

```
┌──────────────┐
│    wrist     │   H×W（640×360）
├───────┬──────┤
│ left  │ right│   各 H/2 × W/2（320×180）
└───────┴──────┘
```

输出 `[T,C,540,640]` uint8。`use_image_augmentation=True` 时在 concat 前对三路一起做 RandomCrop(95%) + Resize 回去 + ColorJitter。

`_convert_video` 再 permute 成训练约定 **`[C,T,H,W]` uint8**。归一化到 [-1,1] 和 VAE 编码都在模型 GPU 上，不在 loader。

### 7.2 Action（整 episode + `joint_pos` + `use_state`）

- gripper action `[T-1,1]` + joint action `[T-1,7]` → `[T-1,8]`（raw，不归一化；`action_normalization=None`）。
- `use_state=True`：把 **第 0 帧** 的 joint state + gripper state 插到最前面 → **`action` 形状 `[T, 8]`**，与视频帧数对齐。  
  SequencePlan 把第 0 个 action 当初始状态 / conditioning，后面 `T-1` 步才监督。

`mode="policy"`：每条 sample 固定 policy，不会随机抽 FD/ID。

`__getitem__` 最后 `_build_result` 给出：

```text
video, action, ai_caption, conditioning_fps=15, mode, domain_id, viewpoint,
additional_view_description（concat_view 说明）
```

---

## 8. 阶段 5：ActionTransformPipeline

`ActionSFTDataset.__getitem__` = 上一节 raw sample 再过 pipeline（`resolution` 来自配方，默认 `"480"`）。

顺序：

1. **空间**：按 `VIDEO_RES_SIZE_INFO["480"]` 找最近长宽比。640×540 ≈ 16:13.5，落到 **4:3 = 736×544**（均为 16 的倍数）。等比缩放 + 反射 pad。`image_size` 记下 pad 前尺寸。
2. **JSON prompt**（`format_prompt_as_json=True`）：任务文本 + 视角 / 分辨率 / 时长 / FPS 等进结构化 JSON，再 tokenize。policy 模式 **不写 clip timeline**（`duration` / `actions[].time`），避免 UND 泄露 episode 长度。
3. **CFG dropout** `cfg_dropout_rate=0.1`：10% 概率把 caption 换成空串。
4. **SequencePlan**（policy）：`has_text/has_vision/has_action=True`；vision 只把第 0 帧当 condition；action 第 0 行（初始 state）是 condition，其余监督。
5. **Action pad**：8D → `max_action_dim=64`，`raw_action_dim=8` 供 loss 把 pad 通道 mask 掉。

这一步之后，单条 sample 已经是模型能吃的 dict，但还不是 packed batch。

---

## 9. 阶段 6：PackingDataLoader 组最终 batch

内层 DataLoader 已经 `batch_size=1` + `custom_collate_fn`（`video`/`action`/`text_token_ids`/`sequence_plan` 不 stack，保持 list）。  
`PackingDataLoader._get_next_sample` 再拆成单样本，然后贪心往一个 output_batch 里装。

当前配方：

| 参数 | 值 | 效果 |
| --- | --- | --- |
| `max_samples_per_batch` | **1** | 装满 1 条就停 → 每步 1 条整 episode |
| `max_sequence_length` | **None** | 与上面二选一；不在 packing 里按 token 丢样本 |
| `lookahead_limit` | **1** | 装不下时最多再看 1 条。`max_samples_per_batch=1` 时几乎用不到 |
| `patch_spatial` | 2 | 只用于 **数 token**（`H/16/2 × W/16/2`） |
| `tokenizer_temporal_compression_factor` | 4 | 数 vision latent 帧 |
| `recycle_inner_iterator` | False | inner StopIteration 就结束，不自动 `iter()` 重生 |

`_update_output_batch` 把每个 key 收成 list。最终 **train_step 看到的 batch**：

| key | 形状（当前 msb=1） | 含义 |
| --- | --- | --- |
| `video` | `list[list[Tensor]]`，内层 1 个 `[3, T, 544, 736]` uint8 | 整集 RGB |
| `action` | 内层 1 个 `[T, 64]` | 含初始 state，后 56 维是 pad |
| `action_raw` | `[T, 8]` | pad 前 |
| `raw_action_dim` | 标量 8 | loss mask |
| `text_token_ids` | 1D token 序列 | JSON prompt |
| `sequence_plan` | 1 个 plan | 谁是 condition / 谁监督 |
| `dataset_name` | `"action_droid"` |  |
| `image_size` / `domain_id` / `mode` / … |  |  |

`T` 随 episode 变（4N+1，且 ≤ 477）。**不会**在 packing 里把短集拼到同一条序列——那需要改成 `max_samples_per_batch=None` + `max_sequence_length=48000`，host RSS 会爆（见 week1 周报）。

每卡独立 pack，global batch = `max_samples_per_batch × world_size × grad_accum_iter`。当前 `msb=1`、`grad_accum_iter=1` → global batch = 卡数。

---

## 10. 关键参数一览

控制「这条 sample 有多长、进不进训练、一 step 装几条」的旋钮：

### 10.1 索引：进不进、拼不拼

| 参数 | 默认（配方 / 本机 launch） | 控制什么 |
| --- | --- | --- |
| `chunk_length` | **-1** | `-1` 整集一条；`>0` 滑窗，每集 `len-chunk` 条 |
| `use_filter_dict` | 配方 False / **launch True** | 是否按 keep_ranges 过滤/拼接 |
| `filter_dict_path` | launch 指向 `keep_ranges_1_0_1_whole_episode.json` | ranges 文件 |
| `use_success_only` | True | 只 `success/` |
| `split` / `split_val_ratio` / `split_seed` | train / 0.03 / 42 | 划分 |
| `min_episode_length_frames` | DROID 写死 33 | 短于 33（或 concat&lt;33）丢弃 |
| `max_pre_tf_tokens` | **48000** | 换成帧帽，超长 **丢弃**（不截断） |
| `max_episode_length_frames` | 由上面算出，480p→**477** | 同上；也可显式覆盖 |
| `max_episodes` | None（仅 probe） | 截断索引前 N 条，方便 RSS 探测 |

### 10.2 解码：一条有多长、怎么读盘

| 参数 | 默认 | 控制什么 |
| --- | --- | --- |
| `fps` | 15 | 相对 native 的整数抽帧 |
| `max_episode_blocks` | **-1** | 解码时截到 `1+4N` 帧；-1 不截。超长应靠 `max_pre_tf_tokens` 丢掉 |
| `viewpoint` | `concat_view` | 腕+双肩拼接；可改 `wrist_view` / `third_person_view` |
| `use_image_augmentation` | True | CPU RandomCrop + ColorJitter |
| `video_backend` | None→torchcodec | `pyav` 可换 |
| `video_decoder_cache_size` | **0** | 0=用完即关；64=LRU（整集流不建议） |
| `action_space` | `joint_pos` | 整集只支持这个 |
| `use_state` | True | action 前面插初始 8D，使 `T_action=T_video` |
| `action_normalization` | None | raw 8D |
| `mode` | `policy` | SequencePlan；`joint` 会每条随机 FD/ID/policy |

### 10.3 变换：空间档与文本

| 参数 | 默认 | 控制什么 |
| --- | --- | --- |
| `resolution` | `"480"` | pad 到 736×544；也决定 `max_pre_tf_tokens`→帧帽（256 档 K=80，同样 48000 token 能装更长） |
| `format_prompt_as_json` | True | JSON UND |
| `cfg_dropout_rate` | 0.1 | 空 caption |
| `max_action_dim` | 64（跟 model） | action 通道 pad |
| `tokenizer_config` | 跟 VLM tokenizer | 文本 token 数，进入 packing 计数 |

### 10.4 流与组 batch

| 参数 | 默认 | 控制什么 |
| --- | --- | --- |
| `iterable_shuffle` | True | episode 打乱流；关了变 map-style + RandomSampler（整集随机访问很伤） |
| `episode_shuffle_seed` | 42 | 打乱种子 |
| `num_workers` | 4 | 每卡解码进程；8 卡必须 ≥4 才不把 GPU 饿死 |
| `prefetch_factor` | 2 | host 上同时活着的整集 RGB 份数 ≈ workers × prefetch |
| `persistent_workers` | True | worker 是否常驻 |
| `max_samples_per_batch` | **1** | 一个 packed batch 装几条 episode |
| `max_sequence_length` | None | 按 token 预算拼多条；与 msb **互斥** |
| `lookahead_limit` | 1 | packing 装不下时再看几条 |
| `recycle_inner_iterator` | False | inner 耗尽是否重启 |
| `worker_restart_every_n` | None | 每 N 条结束 iterator（要 `persistent_workers=False` 才释放 FFmpeg） |
| `grad_accum_iter` | 1 | 不改 packed batch 形状，只改几步才 optimizer.step |

Hydra 覆盖例子（扫 `max_episode_blocks` 时用过）：

```bash
EXTRA_TAIL_OVERRIDES="dataloader_train.dataloader.datasets.droid.dataset.max_episode_blocks=64 trainer.max_iter=21"
```

注意：只设 `max_episode_blocks=64` 是 **截断**；要「超过就丢」应动 `max_pre_tf_tokens` / `max_episode_length_frames`。两者同时存在时，索引先按 padded 长度丢超长，解码再按 blocks 截断（当前 blocks=-1，等于只丢不截）。

---

## 11. 长度预算怎么算（4N+1 / token / HBM）

Wan VAE：`T = 4N+1` 像素帧 → `T_lat = 1+N = 1+(T-1)/4` 个 latent 帧。

480p concat_view pad 到 736×544：

```
latent 空间 = (544/16) × (736/16) = 34 × 46
patch_spatial=2 → 17 × 23 = 391 = K
```

Packing 数 token（pre-TF）：

```
UND(text+eos+vision_start/end) + T_lat × K + T_action
```

`use_state` 时 `T_action = T`。配方用 `extra=500` 兜住 JSON 长短：

```
max_pre_tf_tokens=48000, K=391
→ n_blocks = 119, T_max = 1+4×119 = 477 帧
```

对应大约 `max_episode_blocks=119` 的截断帽；扫参里 **120（481 帧）HBM 还能跑满、128（513 帧）OOM**，所以 477 帧是按 token 丢弃，比 HBM 硬顶（481）略紧一档。

256p（320×256）`K=80`，同样 48000 token 能装更长；HBM 上也是 128 档（513 帧）还能跑，这时卡点变成 AV1 解码而不是激活。

---

## 12. 模型侧（不在 loader 里）

Loader 交出的 batch 仍是 **像素 + raw action + text ids**。`train_step` 里：

1. GPU 上把 uint8 video 归一化，Wan VAE 编成 latent（`T_lat` 帧 × 391 token）。
2. `causal_training_strategy=teacher_forcing`：按 TOML 的 S/K（默认每步随机 `block_size∈[1,4]`、`history_blocks∈[1,32]`）把 vision/action 按 block 交错成因果序列，再算 SDPA mask。
3. 这步会把序列拉得比 packing 统计的 pre-TF token **更长**（双流 + 历史 block），HBM 峰值由这里决定，不是 DataLoader。

所以调 loader 参数改变的是 **每步有多少像素、多少 action step 进 VAE**；调 TF S/K 改变的是 **同一条 episode 在 transformer 里怎么切块**。两边不要混用一个旋钮。
