# DROID Edge 整 episode `max_episode_blocks` 单卡 iter 时间扫测

机器：`liteserver-mi-dev-00001`，Ascend 910，**Phy-ID 14**（`ASCEND_RT_VISIBLE_DEVICES=14`，`npu-smi -i 7 -c 0`）。HBM 64 GiB（65536 MB）。

启动脚本：[launch_sft_action_policy_droid_edge_local.sh](../cosmos/cookbooks/cosmos3/generator/action/finetune/launch_sft_action_policy_droid_edge_local.sh)
（`model=mot_causal_fsdp`，整 episode `chunk_length=-1`，micro-batch 1）。

扫测日期：2026-08-31。

## 方法

- 依次覆盖 `dataloader_train.dataloader.datasets.droid.dataset.max_episode_blocks` = 16, 32, 64, 128, …（每次翻倍），直到某一次因序列过长 OOM。
- 观测帧数上限为 `1 + 4 * max_episode_blocks`（Wan VAE 的 `4N+1` 尾帧填充）。
- IterSpeed 打出前 50 条 `Time:` log，形如：

```text
[RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 0.2515 | Time: 120.42s
```

  对应实现：[iter_speed.py](../cosmos_framework/callbacks/iter_speed.py#L44)（`hit_thres=50`，前 50 step 每步打一条）。
- **平均执行时间（主指标 `avg_skip1_s`）丢掉第 1 条**。第 1 条含模型首次前进、NPU 分配/预热，不能代表稳态 step。对第 2–50 条（共 49 个 `Time`）做算术平均。`first_s` 单独列出，不参与平均。
- 为避免跑满 recipe 的 10000 iter，额外覆盖 `trainer.max_iter=50`，并在凑齐 50 条 Time log 后结束该 run（收尾 checkpoint 不计入时间）。
- 每个 N 使用独立 `job.name` / `OUTPUT_ROOT`，避免 auto-resume。
- `peak_hbm_mb` 是该 run 期间 `npu-info -p 14` 读到的 HBM 占用峰值（MB）。开测前卡上约有 **~10.8 GiB 残留占用**（已退出进程留下的 NPU context，PID 2363484 / 7517 MB），峰值含这部分。

## 结果

| max_episode_blocks | 帧数上限 (4N+1) | 状态 | n_logs | first_s | avg_skip1_s | min_s | max_s | peak_hbm_mb | last_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 65 | ok | 50 | 19.40 | 3.2094 | 3.1600 | 3.2800 | 47988 | 1.8081 |
| 32 | 129 | ok | 50 | 51.79 | 7.8973 | 7.8300 | 8.0800 | 49092 | 2.2111 |
| 64 | 257 | ok | 50 | 89.37 | 22.0384 | 21.9600 | 22.3100 | 62525 | 2.8460 |
| 128 | 513 | **oom** | 0 | — | — | — | — | 65140 | — |

结论：单卡 Phy-ID 14 上，整 episode causal SFT 在 `max_episode_blocks=64`（257 帧）仍能稳定跑满 50 step；`max_episode_blocks=128`（513 帧）在 **第 1 个 training step** 爆显存。

## 补充

### 为何丢掉第 1 条

第 1 条 `Time` 含首次 forward / NPU 内存预热，比稳态慢一个数量级（16/32/64 分别是 19.40s / 51.79s / 89.37s）。主表的 `avg_skip1_s` 是第 2–50 条平均；`min_s` / `max_s` 也只在这 49 条里取。若把第 1 条算进去，50 条全平均分别是 3.5332s / 8.7752s / 23.3850s。

丢掉第 1 条后，稳态时间随 `max_episode_blocks` 约 2.46×（16→32）和 2.79×（32→64）增长，比纯线性更陡，符合 attention 随序列变长的开销。

### 128 OOM 原文

第一条 training step 分配失败（0 条 IterSpeed Time log）：

```text
NPU out of memory. Tried to allocate 1.30 GiB
(NPU 0; 61.27 GiB total capacity; 49.92 GiB already allocated;
 49.92 GiB current active; 829.83 MiB free;
 52.31 GiB reserved in total by PyTorch)
```

`npu-info` 峰值 65140 / 65536 MB（约 99.4%）。PyTorch 侧看到的 61.27 GiB 容量小于整卡 64 GiB，差额主要是 driver / 残留 context。

### 2 卡（Phy-ID 8 + 10）

2026-08-31 试过当时仅有的两张占用 <15 GiB 的卡。HCCL 集合通信在 AICPU KFC 初始化失败（`RunAicpuKfcResInitV2 errorCode=0x2a`），没有 iter 时间。详见 [max_episode_blocks_npu8_10_2card_iter_time.md](./max_episode_blocks_npu8_10_2card_iter_time.md)。

### 原始 log

目录：`cosmos/cookbooks/cosmos3/generator/action/finetune/outputs/bench_max_episode_blocks/logs/blocks_<N>.log`
