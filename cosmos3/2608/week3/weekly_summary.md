# 周报总结（2026/08/24 – 2026/08/26）

本周主线：**在 Cosmos-Framework 中支持 Edge 模型的 Action Policy（动作策略）训练**。共 6 个提交，约 1700 行改动。

## 1. Edge 模型的 action policy 训练支持（8/24）

- 新增针对 action policy 训练的 Edge 模型支持及对应配置（`action_policy_robotwin_edge.py`）。
- 将因果 teacher forcing 机制扩展到 **vision + action 双流**，为本周最大改动（约 720 行，涉及 `cosmos3_vfm_network.py` 等模型代码）。

## 2. RoboTwin 数据集接入（8/25 – 8/26）

- 新增 **RoboTwin LeRobot 数据集**加载器和 Edge action-policy 的 SFT 训练 recipe（约 540 行）。
- 支持**整段 episode 加载**（`chunk_length=-1`），完善 RoboTwin action 数据集的灵活性。

## 3. 可视化与测试（8/25）

- Teacher-forcing mask 可视化中按 **vision / action 模态分色**（hue 拆分）显示，并补充对应单元测试。

## 提交记录

| 日期 | 提交 | 说明 |
| ---- | ---- | ---- |
| 08/26 | `0eed89b` | feat: support whole-episode loading (chunk_length=-1) in RoboTwin action dataset |
| 08/25 | `4191d8b` | test: cover vision/action hue split in teacher-forcing mask visualization |
| 08/25 | `cb072c6` | feat: split vision/action modalities in teacher-forcing mask visualization |
| 08/25 | `8604012` | feat: add RoboTwin LeRobot dataset and Edge action-policy SFT recipe |
| 08/24 | `30bf9eb` | feat: extend causal teacher forcing to vision+action dual streams |
| 08/24 | `2dd3449` | 增加对于 action policy 训练的 edge model 的支持 |

## 附图（attention mask 可视化）

- `attn_mask_base.png` — 基础 attention mask
- `attn_mask_teacher_forcing.png` — teacher-forcing attention mask
- `tf_mask_demo.png` — teacher-forcing mask（vision/action 分色演示）
- `teacher_forcing_sdpa_bool_mask.png` — 训练 smoke run 中导出的 SDPA bool mask

## 一句话总结

打通了 Edge 模型基于 RoboTwin 数据的 vision+action 双流 teacher-forcing 训练链路，从模型、数据、配置到可视化调试工具都已就位。
