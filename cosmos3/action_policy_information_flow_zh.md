# Cosmos3-Edge Action Policy 信息流：从输入、Embedding 到 Attention

本文以仓库根目录的 [`run_policy_batch_inference.sh`](../run_policy_batch_inference.sh#L1) 为入口，沿真实代码路径解释 Cosmos3-Edge Action Policy 批量推理中的信息流。重点不是罗列函数，而是回答以下问题：

1. 文本、视频和动作分别如何变成 Transformer 可以处理的 token；
2. 不同模态的 token 如何排列进同一条 packed sequence；
3. condition、noise、timestep embedding 和 mRoPE 各自表达什么；
4. `two_way_attention` 如何规定文本、视觉和动作之间的信息可见性；
5. 两个样本物理拼接后为什么不会互相污染；
6. [`frontend.py`](../cosmos_framework/model/attention/frontend.py#L49) 和 MoT [`attention.py`](../cosmos_framework/model/generator/mot/attention.py#L143) 各自负责哪一层抽象；
7. Attention 输出如何重新变成最终 action chunk。

本文讨论的是当前脚本与当前 Edge 配置对应的标准非自回归 policy 推理：

```text
joint_attn_implementation = "two_way"
video_temporal_causal     = false
action_gen                = true
sound_gen                 = false
```

配置来源见 `cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml:15`、`cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml:51`、`cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml:113-140`。

## 1. 先建立正确的 token 心智模型

阅读这条链路时，“token”至少有三种含义。如果不先分清，后面很容易把 VAE、sequence packing 和 Transformer embedding 混在一起。

| 名称 | 典型形状 | 含义 |
| --- | --- | --- |
| 文本 token ID | `[P]` | tokenizer 产生的离散词表 ID |
| 模态 latent/token payload | vision `[C,T,H,W]`；action `[T,D]` | VAE latent 或连续动作向量，还不在 LLM hidden space |
| Transformer sequence token | `[N,hidden_size]` | 各模态投影到统一 hidden size 后，真正进入 Transformer 的序列表示 |

因此完整过程不是简单的“所有输入都查 embedding 表”：

```text
文本字符串 ──tokenizer──> token IDs ──embedding lookup──────────┐
                                                               │
视频像素 ──Wan VAE──> vision latent ──patchify + Linear────────┼─> 统一 hidden sequence
                                                               │
动作噪声 ──DomainAwareLinear + modality embedding──────────────┘
```

只有文本使用词表 embedding。Vision 和 action 是连续量，分别使用 `vae2llm` 和 `action2llm` 投影到统一的 2048 维空间，见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:153-175`。

## 2. 全局调用链

从 shell 到一次 Transformer forward 的主链如下：

```text
run_policy_batch_inference.sh
  └─ python -m cosmos_framework.scripts.inference
      └─ scripts.inference.inference(...)
          ├─ 读取两个输入 JSON
          ├─ 创建并加载 OmniInference / OmniMoTModel
          └─ pipe.generate(...)
              └─ OmniInference.create_batches(...)
                  ├─ SampleDataset.__getitem__
                  └─ get_action_sample_data(...)
              └─ OmniInference.generate_batch(...)
                  └─ OmniMoTModel.generate_samples_from_batch(...)
                      ├─ _prepare_inference_data(...)
                      │   ├─ 构造 SequencePlan
                      │   ├─ VAE encode vision
                      │   ├─ tokenize conditional text
                      │   ├─ pack text / vision / action metadata
                      │   └─ 初始化 vision/action noise
                      └─ UniPC 去噪循环（默认 30 steps）
                          └─ velocity_fn(...)
                              └─ _get_velocity(...)
                                  ├─ 用当前噪声刷新 packed payload
                                  └─ denoise(...)
                                      └─ Cosmos3VFMNetwork.forward(...)
                                          ├─ 各模态 embedding
                                          ├─ build_packed_sequence
                                          ├─ 多层 MoT Transformer
                                          │   └─ two_way_attention
                                          │       └─ frontend.attention
                                          │           └─ npu_fa / sdpa
                                          └─ llm2vae / llm2action decode
```

顶层入口和批处理调用分别位于 `cosmos_framework/scripts/inference.py:36-67`、`cosmos_framework/inference/inference.py:1391-1458` 和 `cosmos_framework/inference/inference.py:1483-1518`。

## 3. 脚本配置决定了什么

[`run_policy_batch_inference.sh`](../run_policy_batch_inference.sh#L14) 的关键参数包括：

```bash
export COSMOS_DEVICE=npu
MAX_NUM_SEQS=2

python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --max-num-seqs="${MAX_NUM_SEQS}" \
  -i "${INPUT_FILES[@]}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-file "${CONFIG_FILE}" \
  --seed 0 \
  --no-use-torch-compile
```

这里需要注意两点。

第一，`--max-num-seqs=2` 只是批次预算：最多把两个独立 sample 放进一次 model forward。它不会把两个任务合成一个语义样本。greedy batching 逻辑位于 `cosmos_framework/inference/inference.py:237-263`。

第二，当前 policy 默认参数为：

```text
num_steps = 30
guidance = 1.0
shift = 10.0
sigma_max = 80.0
action_chunk_size = 16
```

来源是 `cosmos_framework/inference/defaults/policy/sample_args.json:1-23`。因为 `guidance=1.0`，正常情况下每个去噪 step 只做 conditional forward，不额外执行 unconditional CFG forward；对应 fast path 在 `cosmos_framework/model/generator/omni_mot_model.py:2787-2789`。

## 4. 原始 Policy 样本如何构造

### 4.1 视频长度和动作长度的对应关系

每个输入任务设置：

```text
model_mode       = policy
action_chunk_size = 16
fps              = 5
vision_path      = vision.mp4
```

Policy 输入需要一个观测起点以及之后 16 个预测步，因此目标视频长度是：

```text
target_frames = action_chunk_size + 1 = 17
```

`build_action_batch()` 会截断过长视频，或重复最后一帧补齐过短视频，见 `cosmos_framework/inference/action.py:94-102`。

### 4.2 Policy 模式不会把 action.json 当作条件动作

这是理解 policy 信息流最重要的细节之一。

在 `forward_dynamics` 模式中，`action_path` 会被读取并作为已知动作条件；但在 `policy` 和 `inverse_dynamics` 模式中，输入动作直接初始化为零槽位：

```python
case ModelMode.POLICY | ModelMode.INVERSE_DYNAMICS:
    return torch.zeros(action_chunk_size, max_action_dim, dtype=torch.float32)
```

源码位于 `cosmos_framework/inference/action.py:29-50`。这些零并不是最终预测结果，它们只是 clean data/template 阶段的动作容器；进入扩散采样后，未 condition 的 action 位置会被随机噪声替换。

当前 Edge 配置的 `max_action_dim=64`，所以每个样本的 action payload 是 `[16,64]`，见 `cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml:63`。

`bridge_orig_lerobot` 的真实动作宽度是 10、domain ID 是 7，见 `cosmos_framework/data/generator/action/domain_utils.py:6-35`。这意味着模型内部统一处理 64 维 action，但该 embodiment 只允许前 10 维有效。

### 4.3 Prompt 会先被结构化

原始自然语言指令在 tokenize 前会被改造成包含视角、动作描述、时长、FPS、分辨率和宽高比的 JSON prompt：

```json
{
  "cinematography": {"framing": "..."},
  "actions": [{"time": "0:00-0:03", "description": "..."}],
  "duration": "3s",
  "fps": 5.0,
  "resolution": {"H": 480, "W": 640},
  "aspect_ratio": "4,3"
}
```

构造入口位于 `cosmos_framework/inference/action.py:54-75`，字段规则位于 `cosmos_framework/data/generator/action/json_formatter.py:55-96`。

这说明模型看到的文本条件不只是用户的短句，还包含与视觉/action 时间轴对齐所需的结构化元数据。

## 5. SequencePlan：先定义“哪些是条件，哪些要生成”

`SequencePlan` 是数据预处理与 sequence packing 之间的协议。它不保存真实 token，而是规定当前 sample 有哪些模态，以及每种模态的 condition indexes。

对于当前 policy 样本，`build_sequence_plan_from_mode()` 产生：

```text
has_text                       = true
has_vision                     = true
has_action                     = true
condition_frame_indexes_vision = [0]
condition_frame_indexes_action = []
action_start_frame_offset      = 1
```

源码位于 `cosmos_framework/data/generator/action/transforms.py:284-363`。

各字段的含义是：

- `condition_frame_indexes_vision=[0]`：第 0 个 vision latent frame 是已知观测；其他 latent frame 是生成目标。
- `condition_frame_indexes_action=[]`：没有已知 action，16 个 action step 都是生成目标。
- `action_start_frame_offset=1`：action[0] 描述观测帧之后的第一个动作间隔，因此在 mRoPE 时间坐标上与后续视觉状态对齐。

这里必须区分两个概念：

```text
condition token：输入值保持固定，但仍参与 Transformer attention
generated/noisy token：随 diffusion timestep 变化，并由网络预测 velocity
```

Condition 只影响噪声注入和最终 velocity mask，不会把 token 从网络序列中删除。实际上 condition vision token 仍然会被归到 generation/diffusion pathway，参与 `q_proj_moe_gen/k_proj_moe_gen/v_proj_moe_gen`。源码对此有明确说明，见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:989-1004`。

## 6. Vision 信息流：像素到 Transformer token

### 6.1 Resize、padding 和有效内容区域

视频先根据输入 `image_size` 和原始宽高比选择预定义 canvas，再保持宽高比缩放并在右侧/底部 padding。处理逻辑位于：

- target size 选择：`cosmos_framework/data/generator/action/transforms.py:43-81`
- resize/padding：`cosmos_framework/data/generator/action/transforms.py:84-183`
- 分辨率表：`cosmos_framework/data/generator/utils.py:42-59`

`image_size` metadata 记录：

```text
[target_h, target_w, resized_content_h, resized_content_w]
```

VAE encode 后，模型会根据后两项移除 padding 对应的 latent 区域，见 `cosmos_framework/model/generator/omni_mot_model.py:2348-2397`。

### 6.2 Wan VAE 编码

视频进入模型后先归一化到 `[-1,1]`，然后经过 Wan VAE：

```text
pixel video: [1,3,T_pixel,H,W]
       ↓ spatial compression = 16
       ↓ temporal compression = 4
vision latent: [1,48,T_latent,H/16,W/16]
```

当前模型配置使用 48 个 latent channels，见 `cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml:116-138`。VAE 调用入口是 `cosmos_framework/model/generator/omni_mot_model.py:5083-5089`，完整的数据抽取与 encode 流程位于 `cosmos_framework/model/generator/omni_mot_model.py:3415-3504`。

17 个 pixel frames 在时间压缩 4 下对应：

```text
T_latent = 1 + floor((17 - 1) / 4) = 5
```

### 6.3 Policy 推理的 causal VAE prefix 优化

Policy 只保留第 0 个 latent frame 为视觉条件。对于 causal VAE，模型不必编码后面仅用于生成的 pixel frames：它只编码产生 condition latent 所需的 pixel prefix，再为其余 latent frame 分配零槽位。

```text
完整输入 17 pixel frames
  → 实际 VAE encode condition prefix
  → 得到 latent frame 0
  → latent frame 1..4 先补零
  → diffusion 初始化时用随机噪声替换 frame 1..4
```

该优化及其适用条件位于 `cosmos_framework/model/generator/omni_mot_model.py:3262-3349`，prefix scatter 位于 `cosmos_framework/model/generator/omni_mot_model.py:3373-3411`。

零槽位不会泄漏进最终输入，因为初始化噪声时会执行：

```python
noise_i = cond_mask * x0_token + (1.0 - cond_mask) * pure_noise_i
```

见 `cosmos_framework/model/generator/omni_mot_model.py:1857-1872`。

### 6.4 Patchify 与 vision embedding

VAE latent 还不是一个“每行对应一个 Transformer token”的矩阵。网络使用 2×2 spatial patch：

```text
[C=48,T,H,W]
  → reshape/einsum
[T × ceil(H/2) × ceil(W/2), 2×2×48]
  = [N_vision,192]
```

源码位于 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:310-355`。

之后通过：

```text
vae2llm: Linear(192, 2048)
```

将每个 vision patch 映射为一个 2048 维 Transformer token。对于 noisy vision frames，还会叠加 timestep embedding，最后 scatter 到全局序列的 `vision.sequence_indexes`，见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:583-669`。

## 7. Text 信息流：Prompt 到 embedding

结构化 JSON prompt 会先经过 chat template，再由 VLM tokenizer 生成 ragged `list[list[int]]`：

```text
JSON prompt
  → chat template
  → per-sample token IDs，不做 batch padding
```

入口位于 `cosmos_framework/model/generator/omni_mot_model.py:1714-1748`，实际逐条 tokenize 位于 `cosmos_framework/model/generator/omni_mot_model.py:4316-4350`。

Sequence packer 会为文本补充结构 token：

```text
[可选 BOS] + caption/chat token IDs + [EOS] + [start_of_generation]
```

`start_of_generation` 对应 `<|vision_start|>`，即使后面同时包含 vision 和 action，它仍作为统一 generation block 的起点。特殊 token 注册位于 `cosmos_framework/data/generator/sequence_packing/modality.py:191-225`，文本打包逻辑位于 `cosmos_framework/data/generator/sequence_packing/sequence.py:182-237`。

进入网络后，文本是唯一使用词表查表的模态：

```python
packed_text_embedding = self.language_model.model.embed_tokens(packed_seq.text_ids)
packed_sequence[packed_seq.text_indexes] = packed_text_embedding
```

见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:549-568`。

## 8. Action 信息流：噪声到 embedding

### 8.1 Action 是 diffusion state，不是离散词表 ID

Policy action 不经过 tokenizer，也不存在 action vocabulary。每个 action step 是一个连续向量：

```text
16 steps × 64-dimensional unified action space
```

在 diffusion 初始化时，每个未 condition 的 action step 被替换为架构无关的随机噪声：

```text
pure_noise_action: [16,64]
```

并将超过真实动作宽度的维度清零。相关代码位于 `cosmos_framework/model/generator/omni_mot_model.py:1874-1904`。

### 8.2 Domain-aware action embedding

Action embedding 包含三部分：

```text
action2llm(action, domain_id)
  + action_modality_embed
  + timestep_embedding（仅 noisy action）
```

其中：

- `action2llm` 将 `[64]` 投影到 `[2048]`；
- `domain_id=7` 告诉投影层当前是 `bridge_orig_lerobot` embodiment；
- `action_modality_embed` 显式告诉主干“这是 action token”；
- timestep embedding 告诉网络当前 token 位于哪个 diffusion noise level。

模块定义见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:169-175`，实际编码见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:732-778`。

需要注意：vision token 没有额外的 vision modality embedding；它通过独立的 `vae2llm` 投影和后续 generation pathway 权重自然区分。Action 则显式加了可学习的 `action_modality_embed`。

## 9. Timestep embedding 与 mRoPE 不是一回事

这是多模态 diffusion Transformer 中最容易混淆的一组知识点。

### 9.1 Timestep embedding：当前有多“吵”

Diffusion timestep 表达当前去噪阶段/噪声强度。它只加到 noisy generation token 上：

```text
vision/action payload embedding + time_embedder(diffusion_t)
```

`TimestepEmbedder` 的调用位于 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:570-581`；将它应用到 noisy frame/token 的逻辑位于 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1188-1245`。

### 9.2 mRoPE：token 在序列、空间和物理时间中的位置

mRoPE position IDs 是 `[3,N_total]`，三轴分别描述 temporal、height、width。它不会直接加到 hidden state，而是在每层 attention 内旋转 Q/K。

不同模态的位置定义为：

```text
text:   (i, i, i)
vision: (temporal_offset + t, h, w)
action: (temporal_offset + aligned_t, 0, 0)
```

文本位置生成位于 `cosmos_framework/data/generator/sequence_packing/mrope.py:41-72`，vision/action 共用的 3D grid 逻辑位于 `cosmos_framework/data/generator/sequence_packing/mrope.py:75-224`。

当前配置在 text 与 generation modality 之间加入 15000 的 temporal margin，避免两类位置范围重叠，配置见 `cosmos_framework/inference/configs/model/Cosmos3-Edge.yaml:40-41`，应用位置见 `cosmos_framework/data/generator/sequence_packing/packers.py:182-200`。

### 9.3 FPS modulation 如何对齐 vision 和 action

当前 `enable_fps_modulation=true`。mRoPE 的时间轴不是简单使用 latent index，而是按 FPS 和时间压缩率缩放：

```text
scaled_time =
  (frame_index + start_frame_offset)
  / (fps / temporal_compression_factor)
  × (base_fps / base_temporal_compression_factor)
```

公式实现位于 `cosmos_framework/data/generator/sequence_packing/mrope.py:168-182`。

对 fps=5、base_fps=24、vision temporal compression=4：

```text
vision latent 时间步长 = 24 / 5 = 4.8
action 时间步长        = (24 / 4) / 5 = 1.2
```

因此 4 个 action step 跨越的 mRoPE 时间大致等于 1 个 vision latent timestep。这正对应 vision VAE 时间压缩 4 倍、action 仍保持原始控制频率的关系。

Action 不会推进全局 temporal cursor，而是复用 vision 起始 offset，在同一时间空间中并行排列，见 `cosmos_framework/data/generator/sequence_packing/sequence.py:400-417`。

## 10. PackedSequence：逻辑上分样本，物理上连续存储

### 10.1 单个 policy 样本的顺序

因为当前 `video_temporal_causal=false`，走标准 pack 路径，单样本顺序为：

```text
[text tokens] [all vision tokens] [all action tokens]
```

Vision 和 action 分开 append 的顺序位于 `cosmos_framework/data/generator/sequence_packing/packers.py:245-383`。

设文本段长度为 `L_text`、vision token 数为 `L_vision`、action token 数为 16，则：

```text
split_lens = [L_text, L_vision + 16]
attn_modes = ["causal", "full"]
sample_len = L_text + L_vision + 16
```

注意：虽然物理内存里 vision 在 action 前面，但二者属于同一个 `full` generation split。它们之间不存在由拼接顺序产生的 causal 约束。

### 10.2 两个样本的 batch 布局

批次中两个样本会连续存储：

```text
[text₁ | vision₁ | action₁ | text₂ | vision₂ | action₂]
```

相应 metadata 为：

```text
sample_lens = [S₁, S₂]
split_lens  = [T₁, G₁, T₂, G₂]
attn_modes  = [causal, full, causal, full]
```

`PackedSequenceBuilder.finalize()` 会生成：

- 全局 `text_indexes`；
- vision/action 的 `sequence_indexes`；
- 每条样本的 `sample_lens`；
- 每个 attention split 的 `split_lens/attn_modes`；
- `[3,N_total]` 的 mRoPE position IDs。

见 `cosmos_framework/data/generator/sequence_packing/sequence.py:778-857`。

### 10.3 Payload 与 sequence indexes 分离

PackedSequence 并不会立刻把原始 latent 数据复制成一条大 tensor。它同时保存：

```text
tokens            # 每个样本/模态原始 payload tensor 列表
sequence_indexes  # 该模态对应统一序列中的位置
token_shapes      # 如何还原每个样本
condition_mask    # 哪些 frame/step 是条件
noisy_frame_indexes
mse_loss_indexes  # 哪些序列位置需要输出 velocity
```

这些字段定义和语义见 `cosmos_framework/data/generator/sequence_packing/modality.py:71-106`。真正进入网络时，各模态才完成 embedding，并通过 `sequence_indexes` scatter 到统一 `[N_total,hidden_size]` tensor。

## 11. 统一 Embedding 序列如何建立

`Cosmos3VFMNetwork.forward()` 首先创建全局 hidden tensor：

```text
packed_sequence = zeros([N_total, hidden_size=2048])
```

然后依次填入：

```text
text indexes   ← embed_tokens(text_ids)
vision indexes ← vae2llm(patchified_vision) + timestep embedding
action indexes ← action2llm(action, domain) + modality embedding + timestep embedding
```

调用顺序见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:968-981`。

Edge backbone 的关键维度是：

```text
vocab_size          = 131072
hidden_size         = 2048
num_hidden_layers   = 28
num_attention_heads = 16
num_key_value_heads = 8
head_dim             = 128
```

来源是 `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json:2-10`。

这里采用 GQA：Q 有 16 个 heads，K/V 只有 8 个 heads。底层 attention backend 原生接受 `H_Q != H_KV`，无需调用方手工 repeat K/V heads。

## 12. 两层 attention 抽象不要混淆

仓库中与当前链路最相关的两个 `attention` 文件处于不同层级。

### 12.1 MoT attention：决定信息可见性

文件：`cosmos_framework/model/generator/mot/attention.py`

它负责：

- 从 SequencePack 拆出 causal text 与 full generation stream；
- 决定每个 Query 可以使用哪些 K/V；
- 使用 sample offsets 保持 batch 样本隔离；
- 调用统一 attention frontend；
- 把 causal/full 输出重新合回原序列顺序。

当前调用的是 `two_way_attention()`，见 `cosmos_framework/model/generator/mot/attention.py:143-244`。

### 12.2 Attention frontend：选择算子后端

文件：`cosmos_framework/model/attention/frontend.py`

它接收已经完成以下工作的 Q/K/V：

```text
embedding
→ layer norm
→ und/gen 专用 QKV projections
→ reshape heads
→ QK norm
→ mRoPE
```

因此 frontend 不负责 tokenizer、embedding、sequence packing 或 mRoPE。它只负责：

1. tensor/参数合法性检查；
2. varlen metadata 标准化；
3. 默认 scale `head_dim ** -0.5`；
4. 选择兼容 backend；
5. 调用 `npu_fa`、`sdpa`、FlashAttention 等实际算子。

对应代码为 `cosmos_framework/model/attention/frontend.py:165-253`。

## 13. 每层 MoT Attention 之前发生什么

### 13.1 UND 与 GEN 使用不同参数路径

`PackedAttentionMoT` 为 understanding/text 和 generation/vision+action 定义两套 projection：

```text
UND:
  q_proj, k_proj, v_proj, o_proj

GEN:
  q_proj_moe_gen, k_proj_moe_gen,
  v_proj_moe_gen, o_proj_moe_gen
```

定义位于 `cosmos_framework/model/generator/mot/unified_mot.py:502-549`。

这表示 vision 和 action 虽然来源不同，但 embedding 后共同进入 GEN tower；text 则进入 UND tower。模态差异主要来自输入投影、action modality embedding、mRoPE 位置以及共享 GEN tower 中的上下文互动。

### 13.2 Q/K/V 的形状

每层 projection 后：

```text
Q_und: [N_und, 16, 128]
K_und: [N_und,  8, 128]
V_und: [N_und,  8, 128]

Q_gen: [N_gen, 16, 128]
K_gen: [N_gen,  8, 128]
V_gen: [N_gen,  8, 128]
```

代码位于 `cosmos_framework/model/generator/mot/unified_mot.py:612-633`。

Q/K 随后使用 packed mRoPE cos/sin 旋转，再重新封装为 SequencePack，见 `cosmos_framework/model/generator/mot/unified_mot.py:635-655`。

## 14. Two-way Attention 的精确可见性

### 14.1 Text/UND Query

Text Query 只进入 causal self-attention：

```text
Q = text
K = text
V = text
mask = causal
```

所以文本 token `i` 只能看到本样本中不晚于自己的文本 token：

```text
text[i] → text[0:i]
```

它看不到 vision 或 action。调用位于 `cosmos_framework/model/generator/mot/attention.py:165-218`。

### 14.2 Vision/Action GEN Query

Generation Query 进入 full attention：

```text
Q = all generation tokens
K = text + all generation tokens
V = text + all generation tokens
mask = none/full
```

源码位于 `cosmos_framework/model/generator/mot/attention.py:223-241`。

由于 vision 和 action 属于同一个 generation split，精确关系是：

```text
vision Query 可以看：全部 text + 全部 vision + 全部 action
action Query 可以看：全部 text + 全部 vision + 全部 action
```

这带来两个重要结论：

1. Action 不是只对观测帧做一次 cross-attention，而是与所有当前 vision latent 联合建模。
2. `[vision | action]` 的物理顺序不等于 causal 顺序；full attention 下，早先排列的 vision 也能读取后面的 action K/V。

### 14.3 可见性矩阵

对单个样本，可以把逻辑 attention mask 理解为：

| Query \ Key | Text | Vision | Action |
| --- | --- | --- | --- |
| Text | causal | 不可见 | 不可见 |
| Vision | 全可见 | 全可见 | 全可见 |
| Action | 全可见 | 全可见 | 全可见 |

因此 `two_way` 的“两路”不是 vision 与 action 两路，而是：

```text
UND/text causal 路径
GEN/vision+action full 路径
```

## 15. 两个样本为什么不会互相 Attention

物理存储虽然是：

```text
sample1 tokens || sample2 tokens
```

但 `sequence_pack_from_packed_sequence()` 会根据 `sample_lens` 生成：

```text
sample_offsets = [0, S₁, S₁+S₂]
```

并分别生成 causal/full stream 的 cumulative offsets，见 `cosmos_framework/data/generator/sequence_packing/runtime.py:143-172` 和 `cosmos_framework/data/generator/sequence_packing/runtime.py:226-283`。

`two_way_attention()` 把这些 offsets 传给 varlen attention：

```text
text causal:
  cumulative_seqlen_Q  = causal_q_offsets
  cumulative_seqlen_KV = causal_k_offsets

generation full:
  cumulative_seqlen_Q  = full_q_offsets
  cumulative_seqlen_KV = sample_offsets
```

见 `cosmos_framework/model/generator/mot/attention.py:200-237`。

底层算子将每个 offset 区间视为独立 sequence。因此它在一次 kernel 中处理多个 sample，但不会计算 sample1 Query 与 sample2 K/V 的点积。

这是一种“无 padding 的变长批处理”：

```text
优点：不必把较短样本 padding 到最长样本
约束：必须可靠维护 cumulative sequence lengths
```

## 16. Ascend NPU Attention 后端

在非 CUDA 设备上，backend 优先级为：

```text
npu_fa → sdpa
```

见 `cosmos_framework/model/attention/backends.py:131-139`。

### 16.1 Frontend 的 backend 选择

`frontend.attention()` 根据以下条件选择兼容后端：

- device；
- dtype；
- 是否需要 backward；
- causal type；
- 是否 varlen；
- Q/K/V heads 和 head dimension。

选择与 dispatch 位于 `cosmos_framework/model/attention/frontend.py:200-253`。

### 16.2 NPU fused attention 的 TND 布局

`npu_fa` 将 frontend 的 heads-last 输入：

```text
[B,S,H,D]
```

展平为 Ascend fused operator 使用的：

```text
TND = [total_sequence,H,D]
```

然后把 cumulative lengths 去掉开头的 0，作为 `actual_seq_qlen` 和 `actual_seq_kvlen` 传入 `torch_npu.npu_fusion_attention`，见 `cosmos_framework/model/attention/npu_fa/functions.py:131-156`。

Text causal 路径使用 `sparse_mode=2` 和压缩 causal mask；generation full 路径使用 `sparse_mode=0`、`atten_mask=None`，见 `cosmos_framework/model/attention/npu_fa/functions.py:40-53` 和 `cosmos_framework/model/attention/npu_fa/functions.py:122-127`。

## 17. Attention 输出之后的信息流

### 17.1 输出投影与 Transformer block

Attention 输出重新拆为 UND/GEN：

```text
UND attention output → o_proj
GEN attention output → o_proj_moe_gen
```

见 `cosmos_framework/model/generator/mot/unified_mot.py:737-740`。

随后分别执行 residual、post-attention norm 和各自的 MLP/MoE 路径，见 `cosmos_framework/model/generator/mot/unified_mot.py:1141-1194`。

因此每层都遵循：

```text
统一 packed hidden
  → 按 UND/GEN 拆分并做各自 norm
  → 各自 QKV projection
  → two-way attention 交换信息
  → 各自 output projection
  → residual
  → 各自 MLP/MoE
  → residual
```

### 17.2 Action head 只读取 noisy action 位置

最后一层 hidden state 中，模型通过 `action.mse_loss_indexes` 取出所有需要预测的 action token：

```text
action_hidden_states: [total_noisy_action_tokens,2048]
  → llm2action(hidden, domain_id)
  → [total_noisy_action_tokens,64]
  → 按样本还原为 [16,64]
```

对应代码位于 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:780-833`。

Condition action 位置不会进入真实 action velocity head；如果某种模式下所有 action 都是条件，代码只走 dummy branch 以保持参数图一致。

### 17.3 Velocity 回到采样器

网络预测的是 rectified-flow velocity，不是一步得到最终 action。`_get_velocity()` 会：

1. 对 condition 位置的 velocity 置零；
2. 对超过 `raw_action_dim` 的 action channels 置零；
3. 将 vision/action velocity 重新 flatten 并交回 UniPC sampler。

代码位于 `cosmos_framework/model/generator/omni_mot_model.py:2256-2298` 和 `cosmos_framework/model/generator/omni_mot_model.py:2322-2346`。

经过默认 30 个去噪 step 后，最终 `[16,64]` action 只保留当前 domain 的前 10 维并写入输出 JSON，见 `cosmos_framework/inference/inference.py:1762-1770`。

## 18. 以当前两个任务估算 token 数

设经过 chat template 和特殊 token 后的文本长度分别为 `T₁`、`T₂`。

原始视频是 4:3；根据两个任务的 `image_size`，可以得到如下近似/确定布局：

| 样本 | canvas | 有效内容 | VAE latent 空间 | patch grid | vision tokens | action tokens | generation tokens |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| task1 / 480 | 736×544 | 640×480 | 40×30 | 20×15 | 5×20×15 = 1500 | 16 | 1516 |
| task2 / 256 | 320×256 | 320×240 | 20×15 | 10×8 | 5×10×8 = 400 | 16 | 416 |

task2 的 latent 高度 15 不能整除 patch size 2，所以网络 patchify 前补到 16，高度方向得到 8 个 patches；padding 逻辑见 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:326-350`。

总 packed 长度为：

```text
N_total = T₁ + 1516 + T₂ + 416
```

这些 token 不会 pad 到相同样本长度，而是通过 varlen offsets 在一个 fused attention batch 中运行。

## 19. 最容易误解的知识点

### 19.1 `action_path` 存在，不代表 policy 会读取 GT action

Policy 分支用全零 action template，并在采样开始时替换为噪声；只有 forward dynamics 把 action 文件作为条件。参考 `cosmos_framework/inference/action.py:37-50`。

### 19.2 Condition vision 仍属于 GEN tower

“condition”表示该 latent 不加噪、采样时保持固定，不表示它归入 text/UND tower。所有 vision/action token 都使用 generation QKV/MLP 权重。参考 `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:989-1004`。

### 19.3 Sequence 中 vision 在 action 前，不代表 vision 看不到 action

它们属于同一个 full split；两者互相全可见。只有 text split 内部是 causal。

### 19.4 `frontend.py` 不构造 attention 语义

“text causal、gen full、样本隔离”在 MoT attention 和 SequencePack metadata 中已经确定。Frontend 只是统一 kernel API 与后端选择层。

### 19.5 Diffusion timestep 与 mRoPE temporal position 不同

- diffusion timestep：表示噪声强度，每个采样 step 会变化；
- mRoPE temporal position：表示 token 在文本/视频/action 时间轴的位置，采样过程中基本固定。

### 19.6 `max_num_seqs=2` 不等于 batch dimension 直接变成 2

在 attention kernel 看来，外层 tensor 常是 `[1,total_tokens,H,D]`，不同样本由 cumulative offsets 表示。这是 sequence-packed varlen batch，而不是传统 `[B,max_len,H,D]` padded batch。Frontend 对该约定的说明位于 `cosmos_framework/model/attention/frontend.py:73-94`。

## 20. 推荐调试断点

如果要在 VS Code 中验证本文信息流，建议按下列顺序设置断点。

| 观察目标 | 断点位置 | 建议检查的变量 |
| --- | --- | --- |
| 原始 policy batch | `cosmos_framework/inference/action.py:113` | `video.shape`、`action.shape`、`sequence_plan` |
| VAE 后 latent | `cosmos_framework/model/generator/omni_mot_model.py:3493` | `x0_tokens_vision` |
| 文本 token IDs | `cosmos_framework/model/generator/omni_mot_model.py:1819` | `cond_text_tokens` |
| 初始 condition/noise | `cosmos_framework/model/generator/omni_mot_model.py:1861` | `condition_mask`、`noise_vision_list`、`noise_action_list` |
| Packed metadata | `cosmos_framework/model/generator/omni_mot_model.py:2224` | `sample_lens`、`split_lens`、`attn_modes`、各 `sequence_indexes` |
| 三模态 embedding 后 | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:983` | `packed_sequence.shape` 与各模态 index 对应区域 |
| SequencePack offsets | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1049` | `input_pack`、`attention_meta` |
| Q/K/V + mRoPE 后 | `cosmos_framework/model/generator/mot/unified_mot.py:653` | `q_und_`、`q_gen_`、`k_und_`、`k_gen_` |
| Two-way 两次 attention | `cosmos_framework/model/generator/mot/attention.py:211`、`:233` | causal/full Q/K/V shape 与 offsets |
| Frontend backend 选择 | `cosmos_framework/model/attention/frontend.py:211` | `is_varlen`、`compatible_backend` |
| NPU fused kernel | `cosmos_framework/model/attention/npu_fa/functions.py:145` | `q_tnd`、`actual_seq_qlen`、`actual_seq_kvlen` |
| Action velocity head | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:821` | `action_hidden_states`、`per_token_domain_id` |

## 21. 一张图总结完整信息流

```text
原始 instruction
  → ActionPromptJsonFormatter
  → chat template + tokenizer
  → text IDs
  → embed_tokens
  → text hidden [T_text,2048] ───────────────────────────────┐
                                                            │
17 帧观测视频                                               │
  → resize/pad                                              │
  → causal Wan VAE（policy 只需 encode condition prefix）   │
  → latent [48,5,H/16,W/16]                                │
  → 2×2 patchify                                            │
  → [N_vision,192]                                          │
  → vae2llm + noisy timestep embedding                      │
  → vision hidden [N_vision,2048] ──────────────────────────┤
                                                            ├─> packed hidden
policy action slots [16,64]                                 │   [sample1 || sample2]
  → 未 condition 位置替换为 noise                           │
  → 无效 channels 清零                                      │
  → action2llm(domain=7)                                    │
  → + action modality embedding                             │
  → + noisy timestep embedding                              │
  → action hidden [16,2048] ────────────────────────────────┘
                         │
                         ├─ packed mRoPE position IDs [3,N]
                         └─ split/sample offsets
                                  ↓
                         28 层 MoT Transformer
                                  ↓
                ┌─────────────────┴──────────────────┐
                │                                    │
          UND/text pathway                     GEN pathway
          独立 QKV projection                  独立 QKV projection
          causal text→text                     full gen→(text+gen)
                │                                    │
                └──────── two-way information flow ──┘
                                  ↓
                       action hidden states
                                  ↓
                    llm2action(domain=7)
                                  ↓
                     action velocity [16,64]
                                  ↓
                       UniPC 更新 diffusion state
                                  ↓ × 30 steps
                       最终 action [16,64]
                                  ↓ 截取 raw_action_dim
                       输出 action [16,10]
```

## 22. 源码阅读顺序

如果需要继续深入实现，建议按以下顺序阅读：

1. `run_policy_batch_inference.sh:30-56`：确认本次推理参数。
2. `cosmos_framework/inference/action.py:29-192`：理解 policy 原始 batch。
3. `cosmos_framework/data/generator/action/transforms.py:235-364`：理解 SequencePlan。
4. `cosmos_framework/model/generator/omni_mot_model.py:1750-2011`：理解 condition/noise 初始化。
5. `cosmos_framework/data/generator/sequence_packing/packers.py:40-430`：理解 token 顺序。
6. `cosmos_framework/data/generator/sequence_packing/sequence.py:182-434`：理解每种模态的 indexes、mask 和 mRoPE。
7. `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:549-669`、`:732-833`、`:937-1185`：理解 embedding、主 forward 与 decode。
8. `cosmos_framework/model/generator/mot/unified_mot.py:468-740`：理解每层 UND/GEN QKV 和 mRoPE。
9. `cosmos_framework/model/generator/mot/attention.py:143-244`：理解 two-way attention 语义。
10. `cosmos_framework/model/attention/frontend.py:49-253`：理解统一 attention API。
11. `cosmos_framework/model/attention/npu_fa/functions.py:56-158`：理解 Ascend fused backend。

读完上述文件后，可以把整条链路概括为一句话：Cosmos3-Edge Policy 将结构化文本、带条件的视觉 latent 和待生成的连续 action noise 投影到统一 hidden space，用 packed varlen metadata 隔离样本，以 text-causal / generation-full 的 two-way MoT attention 联合推理视觉状态和动作，再通过 rectified-flow 多步去噪得到最终 action chunk。
