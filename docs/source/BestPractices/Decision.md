# Decision 决策模型训练

SWIFT 通过 `decision` task_type 支持**typed-decision（System-1）打分模型**的训练与推理。与生成 token 的模型不同，决策模型执行一次前向，对每个问题的每个选项输出校准概率，并在该问题自己的选项集合上做 softmax。

适用于分类、是非判断和有序打分等任务——模型需要在每个问题的固定选项集中做出选择，而非生成自由文本。

## 环境准备

```shell
pip install "transformers>=5.2.0" "qwen_vl_utils>=0.0.14" decord
pip install "ms-swift>=4.6.0"
```

如需使用 OmniJev 视频功能，还需安装系统依赖 `ffmpeg` 和 `ffprobe`。

## 支持模型

1. JEV-27B-VL：Qwen3-VL + PEFT adapter
  - [ModelScope](https://www.modelscope.cn/models/autotrust/JEV-27B-VL)
2. Clef：已合并 LoRA 的完整 Qwen3-VL
  - [ModelScope](https://www.modelscope.cn/models/Cloudflare/clef)
3. OmniJev：Qwen3.5-4B（hybrid gated-deltanet + VL）+ PEFT adapter
  - [ModelScope](https://modelscope.cn/models/AI-ModelScope/OmniJev)


| 模型         | `--model_type` | 底座                               | 头制品                                                                            | `--loss_type`  |
| ---------- | -------------- | -------------------------------- | ------------------------------------------------------------------------------ | -------------- |
| JEV-27B-VL | `jev`          | Qwen3-VL + PEFT adapter          | `decision_head.json` + `calibration.json`（24 槽 verbalizer bias）                | `distill_rps`  |
| Clef       | `clef`         | 已合并 LoRA 的 Qwen3-VL              | `joint_head.safetensors` + `joint_head_config.json`                            | `smooth_brier` |
| OmniJev    | `omnijev`      | Qwen3.5-4B hybrid + PEFT adapter | `head.pt`（FiLM）+ `ord.pt`（有序回归）+ `new_tok_emb.pt`（2 新 token）+ `head_meta.json` | `ce_rps`       |


开发者可以将任意 CausalLM 转为 JEV 风格的决策模型，只需指定：

```shell
swift sft --task_type decision
```



## 损失函数

所有 decision 损失都是 **proper scoring rule**，按问题在各问题自己的选项集上计算。

### 符号约定

- $Q$：一个 micro-batch 中的问题总数（所有 record 展开后）
- $K_q$：问题 q 的选项数（`noul`=2, `score`=6, `choice`=数据集给定）
- $z_q$：头部输出的 per-option logit（softmax 前），shape $\mathbb{R}^{K_q}$
- $p_q = \mathrm{softmax}(z_q)$：模型预测概率（masked softmax，仅在真实选项上归一化），shape $\mathbb{R}^{K_q}$
- $t_q$：目标分布（硬 one-hot 或软标签蒸馏），shape $\Delta^{K_q}$
- $\hat{p}_q, \hat{t}_q$：$p_q, t_q$ 的累积分布（CDF），其中 $\hat{p}_q^{(k)} = \sum_{j \le k} p_q^{(j)}$



### 公共项：Masked Soft Cross-Entropy

所有损失共享一个基础项——masked soft cross-entropy，即 **log_score**（一种 proper scoring rule）：

$$
\mathcal{L}_{\mathrm{CE}} = \frac{1}{|Q|} \sum_{q \in Q} \left( -\sum_{k=1}^{K_q} t_q^{(k)} \log p_q^{(k)} \right)
$$

当 $t_q$ 为软标签时，此式即 $\mathrm{KL}(t_q \| p_q)$（去掉不影响梯度的常数 $H(t_q)$）；当 $t_q$ 为 one-hot 时，退化为标准交叉熵。各模型专属损失在此基础上叠加校准项。

### 各损失详解

**distill_rps**（JEV）：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_r \cdot \mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}}
$$

- 基础项使用软标签蒸馏目标（teacher 的概率分布），此时 $\mathcal{L}_{\mathrm{CE}} \equiv \mathrm{KL}(t \| p)$。
- RPS 附加项仅作用于 `score` 类问题（有序打分），权重 $\lambda_r = 0.5$。

$$
\mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}} = \frac{1}{|Q|} \sum_{q \in Q_{\mathrm{score}}} \frac{1}{K_q - 1} \sum_{k=1}^{K_q-1} (\hat{p}_q^{(k)} - \hat{t}_q^{(k)})^2
$$

**smooth_brier**（Clef）：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\mathrm{LS}} + \lambda_b \cdot \mathcal{L}_{\mathrm{Brier}}
$$

- 标签平滑后的 CE：将 one-hot 标签 $t_q$ 与均匀分布混合，$\tilde{t}_q^{(k)} = (1-\varepsilon) \cdot t_q^{(k)} + \varepsilon / K_q$，$\varepsilon$ = `CLEF_LABEL_SMOOTHING`（默认 0.1）。
- Brier 分数（所有问题类型均参与），权重 $\lambda_b$ = `CLEF_BRIER_WEIGHT`（默认 1.0）。

$$
\mathcal{L}_{\mathrm{Brier}} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{K_q} \sum_{k=1}^{K_q} (p_q^{(k)} - \tilde{t}_q^{(k)})^2
$$

**ce_rps**（OmniJev）：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_r \cdot \mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}}
$$

- 基础项为无标签平滑的 plain log_score（`noul` 2 列二分类 log-score，`choice` $K{+}1$ 列含 abstain 列，`score` $L$ 列有序）。
- RPS 仅作用于 `score` 类问题，权重 $\lambda_r$ = `OMNIJEV_RPS_WEIGHT`（默认 0.5）。
- 与 `distill_rps` 的 RPS 不同，此处不做 per-row 归一化（不除以 $K_q{-}1$）：

$$
\mathcal{L}_{\mathrm{RPS}} = \frac{1}{|Q|} \sum_{q \in Q_{\mathrm{score}}} \sum_{k=1}^{K_q} (\hat{p}_q^{(k)} - \hat{t}_q^{(k)})^2
$$

**scoring**：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}
$$

- 仅公共项，无附加校准项。适用于不需要模型专属损失的通用场景。



## 准确率

`ScoringTrainer` 在训练和验证期间自动计算两个 accuracy 指标，均记录在训练日志中，无需额外配置：

- `acc`**（per-question）**：对每个问题的选项集做 masked argmax 得到预测，与 gold（`target_probs` 的 argmax）比对——每个问题独立计分。
- `record_acc`**（per-record）**：全对才算对的 record 级 accuracy——一个 record 的所有问题均预测正确时该 record 才计为正确，否则计为错误。对多问题联合解码模型（如 Clef）尤其有意义。



## 数据集格式

每个模型有自己的原始 JSONL schema，预处理器将其归一化为统一内部契约。

### JEV

JEV **每个前向只评分一个问题**。含多个问题的 record 会被展开为每问题一行。

```json lines
{"state": "The sky is blue.", "questions": [{"kind": "noul", "question": "Is the sky blue?", "gold": 1}]}
{"state": "A review: the product arrived late.", "questions": [{"kind": "choice", "question": "Classify sentiment.", "options": ["positive", "negative", "neutral"], "gold": 1}]}
{"state": "some text to rate", "questions": [{"kind": "score", "question": "Rate quality.", "gold": [0.05, 0.1, 0.15, 0.4, 0.2, 0.1]}]}
```

- `state`：字符串（待评估的文本状态），也可以是 dict 或 JSON 对象。
- `questions`：问题 dict 列表。每项含 `kind`、`question`（文本）、`options`（仅 `choice` 需要；`noul` 和 `score` 使用固定 canonical 选项）、`gold`。
- `gold`：`noul` — int（0 = 否，1 = 是）；`choice` — int 选项索引；`score` — int 档位索引或完整软分布（float 列表，用于蒸馏）。

> **Canonical 选项**：JEV 对 `noul`/`score` 强制固定 canonical 选项并忽略调用方传入的选项文本，只有 `choice` 使用调用方选项。预处理器完整遵循此行为。



### Clef

Clef **在单次前向中联合解码一个 record 的所有问题**——三者中唯一的多问题单前向模型。

```json lines
{"id": "clef_noul", "state": {"text": "The sky is blue.", "lang": "en"}, "questions": {"is_blue": {"type": "noul", "instructions": "Is the sky blue?"}}, "answers": {"is_blue": true}}
{"id": "clef_choice", "state": "A review: the product arrived late.", "questions": {"sentiment": {"type": "choice", "instructions": "Classify sentiment.", "criteria": {"positive": "Good", "negative": "Bad", "neutral": "Mixed"}}}, "answers": {"sentiment": "negative"}}
{"id": "clef_score", "state": {"doc": "some text"}, "questions": {"quality": {"type": "score", "instructions": "Rate quality.", "criteria": ["terrible", "poor", "ok", "good", "great"]}}, "answers": {"quality": 3}}
```

- `state`：字符串或 dict（渲染为 SCHEMA FIELDS 布局）。
- `questions`：以问题 ID 为键的 dict。每个值含 `type`（`noul`/`choice`/`score`）、`instructions`、`criteria`（选项定义：`choice` 为 dict，`score` 为 list）。
- `answers`：以问题 ID 为键的 dict，给出 gold 答案（`noul` — bool；`choice` — key 字符串；`score` — int 索引）。



### OmniJev

OmniJev **每个前向只评分一个问题**，在**多模态**状态（图像或预合成的视频 mosaic）上进行。

```json lines
{"state": "Look at the image.", "images": ["http://example.com/animal.png"], "questions": [{"type": "noul", "instructions": "Is there an animal?", "gold": true}]}
{"images": ["http://example.com/animal.png"], "questions": [{"type": "choice", "instructions": "What is in the image?", "options": [{"key": "cat", "text": "a cat"}, {"key": "dog", "text": "a dog"}], "gold": "cat"}]}
{"images": ["http://example.com/animal.png"], "questions": [{"type": "score", "instructions": "How clear?", "levels": ["blurry", "ok", "sharp"], "gold": "sharp"}]}
{"images": ["http://example.com/animal.png"], "questions": [{"type": "choice", "instructions": "Which region?", "options": [{"key": "left", "region": {"box": [0, 0, 100, 100]}}, {"key": "none", "abstain": true}], "gold": "left"}]}
```

- `state`（可选）：文本状态。省略时图像即为状态。
- `images`：图像路径或 URL 列表。
- `questions`：问题 dict 列表。每项含 `type`、`instructions` 及类型相关字段：
  - `noul`：`gold` 为 bool。
  - `choice`：`options` 为 `{key, text}` 或 `{key, region}` 或 `{key, abstain: true}` 列表。`gold` 为 key 字符串。
  - `score`：`levels` 为档位标签列表。`gold` 为档位标签字符串。
- `videos`：视频路径列表。**注意**：训练/推理模板不会在线解码原始视频文件，必须先离线调用 `swift.model.omnijev_media.video_state(path)` 把原始视频转成 `{"images": ["mosaic.jpg"], "video": {...}}` 状态再传入。`video_state` 调用 `ffmpeg`/`ffprobe` 将视频均匀采样 16 帧、合成 4×4 mosaic 图像 + 时间戳元数据。直接传入原始视频路径会报错。



## 训练

ms-swift 提供的训练脚本（使用**本地 mini 数据集**，无需联网下载）：

- [JEV](https://github.com/modelscope/ms-swift/blob/main/examples/train/decision/jev.sh) — `--external_plugins examples/train/decision/dataset.py --dataset jev_decision`
- [Clef](https://github.com/modelscope/ms-swift/blob/main/examples/train/decision/clef.sh) — `--external_plugins examples/train/decision/dataset.py --dataset clef_decision`
- [OmniJev](https://github.com/modelscope/ms-swift/blob/main/examples/train/decision/omnijev.sh) — `--external_plugins examples/train/decision/dataset.py --dataset omnijev_decision`

同时支持使用**远程数据集**（如下所示 `--dataset open_jev_jev`，需在线下载 Open-Jev 语料）。两种方式仅数据来源不同，其余参数完全一致。详见[数据集](#数据集)章节。

### JEV

JEV 的决策能力存在于**发布 adapter**（`adapter_vllm/`）中，因此训练是续训该 adapter：

```bash
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
JEV_ADAPTER="$("$SWIFT_PY" -c "import os; from modelscope import snapshot_download; print(os.path.join(snapshot_download('autotrust/JEV-27B-VL'), 'adapter_vllm'))")"

USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model autotrust/JEV-27B-VL \
    --model_type jev \
    --task_type decision \
    --adapters "$JEV_ADAPTER" \
    --loss_type distill_rps \
    --dataset open_jev_jev:all \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

关键参数：

- `--model_type jev` — 解析 JEV 底座和头制品
- `--task_type decision` — 路由到 `ScoringTrainer` + decision data collator
- `--adapters "$JEV_ADAPTER"` — 续训发布 adapter（is_trainable=True）
- `--loss_type distill_rps` — KL(target‖model) + 0.5 × RPS（`score`）



### Clef

Clef 发布为**已合并 backbone**，因此创建新 LoRA：

```bash
USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Cloudflare/clef \
    --model_type clef \
    --task_type decision \
    --loss_type smooth_brier \
    --dataset open_jev_clef:all \
    --tuner_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

关键参数：

- `--model_type clef`
- `--loss_type smooth_brier` — 标签平滑 CE + Brier
- `--tuner_type lora --target_modules all-linear` — 在已合并 backbone 上创建新 LoRA



### OmniJev

OmniJev 发布为**仅 adapter**（底座权重需单独获取），头制品位于 adapter 目录中：

```bash
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
OMNIJEV_CKPT="$("$SWIFT_PY" -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/OmniJev'))")"
export OMNIJEV_HEAD="$OMNIJEV_CKPT/head.pt"
export OMNIJEV_ORD="$OMNIJEV_CKPT/ord.pt"
export OMNIJEV_HEAD_META="$OMNIJEV_CKPT/head_meta.json"
export OMNIJEV_NEW_TOK_EMB="$OMNIJEV_CKPT/new_tok_emb.pt"
export MSO_BRANCH_CKPT=1

USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --model_type omnijev \
    --task_type decision \
    --adapters "$OMNIJEV_CKPT" \
    --loss_type ce_rps \
    --dataset open_jev_omnijev:all \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --freeze_vit true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

关键参数：

- `--model Qwen/Qwen3.5-4B` — 提供 hybrid 底座
- `--model_type omnijev`
- `--adapters "$OMNIJEV_CKPT"` — 续训出厂 adapter
- `--loss_type ce_rps` — 仅 proper scoring rules（log_score + RPS on `score`）
- `--gradient_checkpointing false` — 见[下方说明](#omnijev-msol_branch_ckpt-与-gradient_checkpointing-互斥说明)



### 从零训练任意 CausalLM

`generic_decision` 模型类型允许将任意文本 CausalLM 转为 JEV 风格的决策模型。与三个模型专属路径不同，此路径从 tokenizer 构建随机初始化的 `JevVerbalizerHead`，无需预置头权重：

> `JevVerbalizerHead` **工作原理**：该头是一种轻量级 verbalizer 打分头，不引入额外的神经网络层，而是直接复用底座 CausalLM 的 `lm_logits`。它在每个问题 prompt 的最后一个 token 位置读取 `lm_logits`，做 `log_softmax` 后，按 `kind` 从中抽取对应 verbalizer token 的 log-probability。三种 kind 对应 **24 个固定 verbalizer token**：
>
> - `noul`（2 个）：`false`, `true`
> - `score`（6 个）：`0`, `1`, `2`, `3`, `4`, `5`
> - `choice`（16 个）：`A` … `P`
>
> 每个 verbalizer token 拥有一个可训练的 fp32 bias 标量（共 24 个），再加上 3 个 per-kind 温度参数。前向计算为 `z = (log_prob + bias) / temperature`，随后在问题自己的选项集上做 softmax。`from_tokenizer` 构建路径会从 tokenizer 解析这 24 个 token id，bias 零初始化、温度单位初始化，使其可以直接从随机状态开始与 LoRA adapter 联合训练。

```bash
USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type generic_decision \
    --task_type decision \
    --loss_type distill_rps \
    --dataset open_jev_jev:all \
    --tuner_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

关键参数：

- `--model <任意 CausalLM>` — 任何 `AutoModelForCausalLM.from_pretrained` 能加载的模型
- `--model_type generic_decision` — 选取 `GenericDecisionLoader`，通过 `JevVerbalizerHead.from_tokenizer` 构建头（从 tokenizer 解析 24 个 verbalizer token id，零初始化 bias，单位温度）
- `--loss_type distill_rps` — KL(target‖model) + 0.5 × RPS（`score`）；或用 `--loss_type scoring` 走无 RPS 项的纯 soft CE
- `--tuner_type lora --target_modules all-linear` — `lm_head` 含于 `all-linear`，适配头所读的 verbalizer 行

头的 24 个 fp32 bias 标量与 LoRA adapter 联合训练——`ScoringTrainer.__init__` 自动重新激活 `scoring_head` 参数（设置 `requires_grad=True`，fp16→fp32）。训练后 checkpoint 含 `adapter_model.safetensors` + `scoring_head.safetensors` + `scoring_head.json`，部署命令：

```bash
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type generic_decision \
    --adapters output/checkpoint-xxx
```

数据格式与 JEV 相同。

## 数据集

Decision 模型的数据集有两种来源：**本地数据集**（通过 `--external_plugins` 注册，使用对应的 `ScoringPreprocessor` 子类）和**远程数据集**（SWIFT 内置的 Open-Jev 语料，直接用 `--dataset <名称>` 加载）。

> **重要**：Decision 数据**不能**通过 `--dataset /path/to/file.jsonl` 直接加载。swift 的标准数据加载方法（直接传文件路径，使用 `AutoPreprocessor`）不适用于 decision 数据——`AutoPreprocessor` 不认识 `{state, questions: [{kind, question, options, gold}]}` 这种 schema，会静默走错路径或报错。必须通过 `--external_plugins` 注册一个 Python 插件，用对应的 `ScoringPreprocessor` 子类处理数据。



### 远程数据集（Open-Jev）

SWIFT 将 [Open-Jev](https://www.modelscope.cn/datasets/ZefanCai/Open-Jev) 和 [Open-Jev-v1.1](https://modelscope.cn/datasets/ZefanCai/Open-Jev-v1.1)语料注册为内置远程数据集。无需 `--external_plugins`，直接用 `--dataset <名称>` 即可。


| 数据集名称                  | 来源                       | 模型适配器                 |
| ---------------------- | ------------------------ | --------------------- |
| `open_jev_jev`         | `ZefanCai/Open-Jev` (v1) | `JevPreprocessor`     |
| `open_jev_clef`        | `ZefanCai/Open-Jev` (v1) | `ClefPreprocessor`    |
| `open_jev_omnijev`     | `ZefanCai/Open-Jev` (v1) | `OmniJevPreprocessor` |
| `open_jev_jev_v11`     | `ZefanCai/Open-Jev-v1.1` | `JevPreprocessor`     |
| `open_jev_clef_v11`    | `ZefanCai/Open-Jev-v1.1` | `ClefPreprocessor`    |
| `open_jev_omnijev_v11` | `ZefanCai/Open-Jev-v1.1` | `OmniJevPreprocessor` |


使用 v1.1 语料时替换为带 `_v11` 后缀的名称（如 `--dataset open_jev_jev_v11`）。

本文档训练命令中的 `--dataset` 使用远程数据集名称（如 `open_jev_jev`），需要在线下载 `ZefanCai/Open-Jev` 语料。

#### Open-Jev 原始行样例

Open-Jev 数据集中的一行包含一个独立问题记录，字段包括 `kind`、`question`、`state`、`options`、`target`：

```json lines
{"kind": "choice", "question": "What is the capital of France?", "state": "France is a country in Western Europe. Its capital city is Paris.", "options": ["London", "Paris", "Berlin", "Madrid"], "target": [0.1, 0.7, 0.15, 0.05]}
{"kind": "noul", "question": "Is the sky blue during daytime?", "state": "The sky appears blue due to Rayleigh scattering.", "target": [0.1, 0.9]}
{"kind": "score", "question": "Rate the quality of this response.", "state": "The response is mostly accurate but lacks detail.", "target": [0.0, 0.0, 0.2, 0.5, 0.2, 0.1]}
```

- `kind`：问题类型（`noul` / `choice` / `score`）。
- `question`：问题文本。
- `state`：待评估的文本状态（字符串或 JSON 对象）。
- `options`：选项列表（仅 `choice` 需要；`noul` 和 `score` 使用固定 canonical 选项，可省略）。
- `target`：教师软分布（float 列表），用于蒸馏。长度与选项数对齐。



#### 经过预处理器后的输出行

`OpenJevPreprocessor(JevPreprocessor())` 将每行归一化为统一内部契约。`target` 会被重映射到与标准选项对齐的顺序：

choice（`options` 保留原始顺序，`target` 与之对齐）：

```json
{"messages": [{"role": "user", "content": "France is a country in Western Europe. Its capital city is Paris."}], "questions": ["What is the capital of France?"], "kinds": ["choice"], "options": [["London", "Paris", "Berlin", "Madrid"]], "target_probs": [[0.1, 0.7, 0.15, 0.05]]}
```

noul（`options` 被强制设为标准 `["false", "true"]`，`target` 重映射）：

```json
{"messages": [{"role": "user", "content": "The sky appears blue due to Rayleigh scattering."}], "questions": ["Is the sky blue during daytime?"], "kinds": ["noul"], "options": [["false", "true"]], "target_probs": [[0.1, 0.9]]}
```

score（`options` 被强制设为标准 `["0", "1", "2", "3", "4", "5"]`）：

```json
{"messages": [{"role": "user", "content": "The response is mostly accurate but lacks detail."}], "questions": ["Rate the quality of this response."], "kinds": ["score"], "options": [["0", "1", "2", "3", "4", "5"]], "target_probs": [[0.0, 0.0, 0.2, 0.5, 0.2, 0.1]]}
```

> **注意**：JEV 每行只渲染一个问题（`JevPreprocessor` 将记录拆为每问题一行），`noul`/`score` 的选项被强制设为标准 verbalizer 集合，`target` 被重映射到与标准选项对齐。



### 本地数据集

本地数据集需要编写 Python 插件文件，通过 `--external_plugins` 传入。插件使用 `register_dataset` 配合对应的 `ScoringPreprocessor` 子类。

#### 现成示例

仓库提供了一套完整可用的本地 mini 数据集，位于 `examples/train/decision/`：


| 文件                                  | 说明                                                                  |
| ----------------------------------- | ------------------------------------------------------------------- |
| `dataset.py`                        | 插件注册文件，注册了 `jev_decision`、`clef_decision`、`omnijev_decision` 三个数据集名 |
| `data/jev.jsonl`                    | JEV 风格 mini 示例（4 行，含 noul / choice / score / 多问题联合展开）               |
| `data/clef.jsonl`                   | Clef 风格 mini 示例（4 行，含 noul / choice / score / 多问题联合解码）              |
| `data/omnijev.jsonl`                | OmniJev 风格 mini 示例（4 行，含 noul / choice / score / region+abstain）    |
| `jev.sh` / `clef.sh` / `omnijev.sh` | 训练脚本，使用本地数据集                                                        |


直接运行训练脚本即可：

```bash
sh examples/train/decision/jev.sh
```

脚本内部通过 `--external_plugins examples/train/decision/dataset.py --dataset jev_decision` 加载本地数据，无需下载远程数据集。而本文档正文训练命令用的是远程数据集（如 `--dataset open_jev_jev`），两者仅数据来源不同，其余参数完全一致。

#### 编写自定义插件

```python
# my_decision_dataset.py
import os
from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.preprocessor.decision import JevPreprocessor, ClefPreprocessor, OmniJevPreprocessor

register_dataset(
    DatasetMeta(
        dataset_name='my_jev_decision',
        dataset_path='/path/to/my_jev_data.jsonl',
        preprocess_func=JevPreprocessor(),
    ),
    exist_ok=True,
)
```

训练时使用：

```bash
swift sft --external_plugins my_decision_dataset.py --dataset my_jev_decision ...
```

根据数据 schema 选择预处理器：

- `JevPreprocessor` — JEV 风格 `{state, questions: [{kind, question, options, gold}]}` 记录（每问题一行）
- `ClefPreprocessor` — Clef 风格 `{state, questions: {qid: {type, instructions, criteria}}, answers}` 记录（全部问题一行）
- `OmniJevPreprocessor` — OmniJev 风格多模态记录（含 `images`/`videos` 和选项块）
- `GenericDecisionPreprocessor` — 与 `JevPreprocessor` 行为相同，配套 `generic_decision` 模型类型



## 数据增强

决策数据集支持训练时 schema 增强，通过 CLI 参数控制。全部默认关闭——当所有参数为默认值时，增强为零开销透传。


| CLI 参数                         | 类型      | 默认值     | 说明                                             |
| ------------------------------ | ------- | ------- | ---------------------------------------------- |
| `--decision_drop_instructions` | `float` | `0.0`   | 丢弃问题指令文本的概率（布局仍渲染，但文本置空）                       |
| `--decision_subsample_options` | `int`   | `0`     | 将 `choice` 选项采样至至多 N 个（保留 gold + N-1 干扰项）。0 关闭 |
| `--decision_state_as_text`     | `float` | `0.0`   | 将 dict `state` 展平为人类可读文本的概率                    |
| `--decision_shuffle_fields`    | `bool`  | `False` | 打乱 dict `state` 字段顺序及多问题记录的问题顺序                |


> 增强随机种子复用 `--data_seed`，无需单独指定。

示例——训练 JEV 时启用 30% 指令丢弃、选项采样至 8、字段打乱：

```bash
swift sft \
    --model_type jev \
    --task_type decision \
    --loss_type distill_rps \
    --dataset open_jev_jev:all \
    --adapters .../adapter_vllm \
    --decision_drop_instructions 0.3 \
    --decision_subsample_options 8 \
    --decision_shuffle_fields true \
    --torch_dtype bfloat16 \
    --output_dir output \
    ...
```

增强在预处理阶段由 `augment_decision_row` 应用，从 CLI 参数自动注入，无需插件代码。

## 推理

Decision 模型仅使用 transformers 后端（`decision` task_type 不支持 vLLM）。推理走标准 `swift infer` 路径——训好的头通过 guarded hook 自动加载。

### 命令行

```bash
# JEV
swift infer \
    --model autotrust/JEV-27B-VL \
    --model_type jev \
    --adapters output/checkpoint-xxx

# Clef
swift infer \
    --model Cloudflare/clef \
    --model_type clef \
    --adapters output/checkpoint-xxx

# OmniJev
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
OMNIJEV_CKPT="$("$SWIFT_PY" -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/OmniJev'))")"
export OMNIJEV_HEAD="$OMNIJEV_CKPT/head.pt"
export OMNIJEV_ORD="$OMNIJEV_CKPT/ord.pt"
export OMNIJEV_HEAD_META="$OMNIJEV_CKPT/head_meta.json"
export OMNIJEV_NEW_TOK_EMB="$OMNIJEV_CKPT/new_tok_emb.pt"
export MSO_BRANCH_CKPT=1

swift infer \
    --model Qwen/Qwen3.5-4B \
    --model_type omnijev \
    --adapters output/checkpoint-xxx

# Generic Decision
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type generic_decision \
    --adapters output/checkpoint-xxx
```

模型返回 per-option 概率（在各问题自己的选项集上 softmax），而非生成文本。每条记录的推理输出是一个列表，每问题一项：

```json
[{"index": 1, "probs": [0.1, 0.7, 0.2]}]
```

其中 `index` 为模型预测的选项索引（argmax of `probs`），`probs` 为各选项的 softmax 概率。

### 推理评测（accuracy）

推理时传入 `--metric acc`，swift 会自动对 decision 数据集计算 per-question accuracy：

```bash
swift infer \
    --model autotrust/JEV-27B-VL \
    --model_type jev \
    --adapters output/checkpoint-xxx \
    --metric acc
```

计算逻辑：取 `target_probs` 的 argmax 作为 gold 答案，取 `response` 中每问的 `index` 作为预测，逐问比对后输出 `{'acc': ...}`。

## Head 存盘 / 读回机制

Decision 模型携带**可插拔打分头**，与 backbone（或 LoRA adapter）联合训练。头权重**单独存储**，避开「config 声明生成模型但权重含分类器」的不一致问题：

- **存盘**：`ScoringTrainer._save` 先通过标准路径存 adapter（或全量模型），然后在旁边写入头制品。对于 adapter（LoRA）训练，写入 `scoring_head.safetensors`（头参数）和 `scoring_head.json`（头类型 + 标定元数据）。对于全量微调训练，只写入 `scoring_head.json`——头参数已在模型的 safetensors 分片中（`from_pretrained` 因 `scoring_head` 尚未挂载而忽略了它们）。
- **读回**：`maybe_load_trained_scoring_head(model, adapter_dir)` 在 `Swift.from_pretrained` 之后自动调用，覆盖续训、断点恢复、推理三条路径。处理两种场景：当 `scoring_head.safetensors` 存在时（adapter checkpoint），从中加载权重；当仅 `scoring_head.json` 存在时（全量微调 checkpoint），从模型的 safetensors 分片中提取头权重。对非 decision 模型是**受守卫的空操作**。

训完的 checkpoint 可直接部署——`swift infer` 或 `--adapters <checkpoint>` 会自动读回训好的头。

## OmniJev：`MSO_BRANCH_CKPT` 与 `gradient_checkpointing` 互斥说明

OmniJev 的 hybrid Qwen3.5-4B backbone 使用线性注意力层，沿序列携带递归状态。block-diagonal 注意力 mask 无法隔离选项，因为递归状态会在选项块之间泄漏信息。该问题用**两段缓存 branch**机制解决（`swift/model/omnijev_branch.py`）：

1. 共享前缀（chat template + 图像 token + instruction）只前向一次，保留 KV/recurrent cache。
2. 每个 (问题, 选项) 对是一行，通过 `expand_cache` 复用前缀 cache，选项之间物理隔离。

`--gradient_checkpointing` **与此机制互斥。** HF 在 `gradient_checkpointing=True` 且 `training=True` 时强制 `use_cache=False`，但 branch 的前缀 pass 必须返回非 None cache。branch forward 内部会自动关闭 gradient_checkpointing，因此该 flag 对 OmniJev 实际上是**空操作**。

**改用** `MSO_BRANCH_CKPT=1` 节省显存（实测：14.3 GiB vs 不开 16.2 GiB）。

```bash
export MSO_BRANCH_CKPT=1
# ...
swift sft --gradient_checkpointing false ...
```

OmniJev adapter 还需通过环境变量指向四个头制品（位于 adapter 目录而非底座目录）：


| 环境变量                  | 文件               | 用途                |
| --------------------- | ---------------- | ----------------- |
| `OMNIJEV_HEAD`        | `head.pt`        | FiLM 双 MLP 打分头    |
| `OMNIJEV_ORD`         | `ord.pt`         | 有序回归（CORAL/CORN）头 |
| `OMNIJEV_HEAD_META`   | `head_meta.json` | 头元数据（各 kind 温度等）  |
| `OMNIJEV_NEW_TOK_EMB` | `new_tok_emb.pt` | 2 个新 token（`<     |




### JEV / Clef 头制品环境变量覆盖

当头制品与底座不在同一目录时（如自定义部署路径），JEV 和 Clef 各有两个环境变量可覆盖头配置/权重的加载路径：


| 环境变量                      | 适用模型 | 覆盖文件                     | 用途                            |
| ------------------------- | ---- | ------------------------ | ----------------------------- |
| `JEV_DECISION_HEAD`       | JEV  | `decision_head.json`     | 覆盖 verbalizer bias / ids 加载路径 |
| `JEV_DECIDE_CALIBRATION`  | JEV  | `calibration.json`       | 覆盖 per-kind 温度标定路径            |
| `CLEF_JOINT_HEAD_CONFIG`  | Clef | `joint_head_config.json` | 覆盖头配置加载路径                     |
| `CLEF_JOINT_HEAD_WEIGHTS` | Clef | `joint_head.safetensors` | 覆盖头权重加载路径                     |


未设置时，加载器会自动在 `model_dir`（及 JEV 的 `adapter_vllm/` 子目录）下查找对应文件。文件缺失时报错信息中会提示设置对应环境变量。

## 已知限制

- **多卡 FSDP / DeepSpeed-zero3 头存盘**：单卡正确，分片下 rank0 可能只拿到分片。
- `--target_modules all-linear` **注入 head**：Clef/OmniJev head 含 Linear，all-linear 可能叠加 LoRA。
- **JEV choice > 16 选项**：仅支持 `single` 模式（≤16 一次前向）。
- **OmniJev 音频频谱图**：不在推理路径内，需用户自备转换器。
- `tie_word_embeddings=True` **adapter 合并**：仅影响合并/转格式，训练与推理不受影响。



## 验证报告

> **验证环境**：transformers 5.16.1 / ms-swift 4.6.0.dev0 / 8× L20X (143 GiB) / ffmpeg + ffprobe 已安装
>
> **验证方式**：4 个模型逐一执行「训练前推理 → 本地数据集训练 → 训练后推理」全流程，采用的CLI均为上述示例
>
> **Generic Decision**:采用的是Qwen2.5-0.5B 底座



### Benchmark — 训练后推理对比

| #   | 模型               | before acc | after acc  | Δ acc   | before top prob | after top prob | 结果      |
| --- | ------------ | ---------------- | ---------- | ---------- | ------- | --------------- | -------------- | ------- |
| 1   | JEV              | 0.6667     | **1.0000** | +0.3333 | 0.9855          | **1.0000**     | ✅ +50%  |
| 2   | Clef             | 0.5000     | **0.8333** | +0.3333 | 0.9712          | 0.9311         | ✅ +67%  |
| 3   | OmniJev          | 0.5500     | 0.7500     | +0.2884 | 0.8465          | 0.8196         | ✅ +37%  |
| 4   | Generic Decision | 0.1667     | **0.6667** | +0.5000 | 0.5682          | 0.9363         | ✅ +300% |
