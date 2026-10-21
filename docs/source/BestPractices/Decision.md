# Decision 决策模型训练

SWIFT 通过 `decision` task_type 支持**typed-decision（System-1）打分模型**的训练与推理。与生成 token 的模型不同，决策模型执行一次前向，对每个问题的每个选项输出校准概率，并在该问题自己的选项集合上做 softmax。

适用于分类、是非判断和有序打分等任务——模型需要在每个问题的固定选项集中做出选择，而非生成自由文本。

## 环境准备

```shell
pip install "transformers>=5.2.0" "qwen_vl_utils>=0.0.14" decord
pip install "ms-swift>=4.6.0"
```

如需使用视频功能（多模态决策模型），还需安装系统依赖 `ffmpeg` 和 `ffprobe`。

## 支持模型

1. JEV-27B-VL：Qwen3-VL + PEFT adapter
  - [ModelScope](https://www.modelscope.cn/models/autotrust/JEV-27B-VL)
2. Clef：已合并 LoRA 的完整 Qwen3-VL
  - [ModelScope](https://www.modelscope.cn/models/Cloudflare/clef)
3. OmniJev：Qwen3.5-4B（hybrid gated-deltanet + VL）+ PEFT adapter
  - [ModelScope](https://modelscope.cn/models/AI-ModelScope/OmniJev)
4. Kev：Qwen3.5-{0.8,4,9}B-Base（hybrid gated-deltanet）+ PEFT adapter，纯文本
  - [kev-0.8b](https://www.modelscope.cn/models/jaredpalmer/kev-0.8b) / [kev-4b](https://www.modelscope.cn/models/jaredpalmer/kev-4b) / [kev-9b](https://www.modelscope.cn/models/jaredpalmer/kev-9b)


| 模型         | `--model_type` | 底座                               | 头制品                                                                            | `--loss_type`  |
| ---------- | -------------- | -------------------------------- | ------------------------------------------------------------------------------ | -------------- |
| JEV-27B-VL | `jev`          | Qwen3-VL + PEFT adapter          | `decision_head.json` + `calibration.json`（24 槽 verbalizer bias）                | `distill_rps`  |
| Clef       | `clef`         | 已合并 LoRA 的 Qwen3-VL              | `joint_head.safetensors` + `joint_head_config.json`                            | `smooth_brier` |
| OmniJev    | `omnijev`      | Qwen3.5-4B hybrid + PEFT adapter | `head.pt`（FiLM）+ `ord.pt`（有序回归）+ `new_tok_emb.pt`（2 新 token）+ `head_meta.json` | `ce_rps`       |
| Kev        | `kev`          | Qwen3.5-{0.8,4,9}B-Base hybrid + PEFT adapter | `head.pt`（pointer head，fp32）+ 拟合 `temperature`（仅 eval 生效）                 | `scoring`      |


开发者可以将任意 CausalLM 转为决策模型，只需指定：

```shell
swift sft --task_type decision
```



## 损失函数

所有 decision 损失都是 **proper scoring rule**，按问题在各问题自己的选项集上计算。各模型默认使用的 `--loss_type` 见[支持模型](#支持模型)表格。

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

公共交叉熵项使用 FP32 masked `log_softmax` 计算，不对概率做截断，因此即使预测非常自信但错误（包括 BF16 logits），仍保留纠正梯度。Padding 和没有有效选项的行不贡献 loss 或梯度。

### 各损失详解

**distill_rps**：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_r \cdot \mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}}
$$

- 基础项使用软标签蒸馏目标（teacher 的概率分布），此时 $\mathcal{L}_{\mathrm{CE}} \equiv \mathrm{KL}(t \| p)$。
- RPS 附加项仅作用于 `score` 类问题（有序打分），权重 $\lambda_r = 0.5$。

$$
\mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}} = \frac{1}{|Q|} \sum_{q \in Q_{\mathrm{score}}} \frac{1}{K_q - 1} \sum_{k=1}^{K_q-1} (\hat{p}_q^{(k)} - \hat{t}_q^{(k)})^2
$$

**smooth_brier**：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\mathrm{LS}} + \lambda_b \cdot \mathcal{L}_{\mathrm{Brier}}
$$

- 标签平滑后的 CE：将 one-hot 标签 $t_q$ 与均匀分布混合，$\tilde{t}_q^{(k)} = (1-\varepsilon) \cdot t_q^{(k)} + \varepsilon / K_q$，$\varepsilon$ = `CLEF_LABEL_SMOOTHING`（默认 0.1）。
- Brier 分数（所有问题类型均参与），权重 $\lambda_b$ = `CLEF_BRIER_WEIGHT`（默认 1.0）。

$$
\mathcal{L}_{\mathrm{Brier}} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{K_q} \sum_{k=1}^{K_q} (p_q^{(k)} - \tilde{t}_q^{(k)})^2
$$

**ce_rps**：

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
- `record_acc`**（per-record）**：全对才算对的 record 级 accuracy——一个 record 的所有问题均预测正确时该 record 才计为正确，否则计为错误。对多问题联合解码的模型尤其有意义。



## 温度校准

`temperature_calibration` 回调在训练结束后、最终保存前，于**验证集**上对 decision 模型做**后训练温度校准**（post-training temperature scaling）——让模型输出的置信度数字变得"诚实"（说 80% 把握的题，实际正确率约八成）。

### 为什么需要校准

微调后的模型普遍存在**过度自信**或**信心不足**：softmax 概率偏离真实正确率。温度缩放用一个标量 $T$ 修正 logits：$p = \mathrm{softmax}(z/T)$。由于 $T$ 不改变 logits 间的相对大小，**argmax（预测选项、accuracy）完全不变**，只重塑概率分布的"尖锐程度"——T > 1 拉平分布（降信心），T < 1 拉尖分布（增信心）。下游若拿 confidence 做阈值、排序或风险加权，校准是这些数字可用的前提。

### 触发与跳过条件

**自动启用**：`task_type=decision`时自动挂载，无需手动传参。也可显式指定：

```bash
--callbacks temperature_calibration
```

**静默跳过**（温度保持出厂/加载值，不报错）：

- head 不携带温度 buffer（如 joint schema 类头）；
- 训练器没有 eval_dataset；
- 验证集收集到的总题数 < 10（`SWIFT_CALIBRATION_MIN_ITEMS` 可调）；
- logits 收集阶段抛出异常。

**前提**：需要验证集。各示例脚本的 `--split_dataset_ratio 0` 关闭了自动切分，如需校准请改为 `--split_dataset_ratio 0.1`（从训练集按整 record 切 10% 做验证），或显式传 `--val_dataset`。

### 校准原理（两阶段拟合）

温度不是单一标量，而是**全局 × 每题型**的分解结构：

1. **Stage 1**：把全部问题（noul / choice / score 池化）合在一起拟合一个全局温度 `head_temperature`（$T_h$），捕捉模型整体的"锐度"偏差；
2. **Stage 2**：logits 先除以 $T_h$，再对每个题型单独拟合相对温度 $T_r^{\mathrm{kind}}$，捕捉题型特有的残差；
3. 线上生效温度为二者乘积：$T^{\mathrm{kind}} = T_h \cdot T_r^{\mathrm{kind}}$，最终被夹取到 0.05~20 之间。

拟合目标（最小化对象）由 `SWIFT_CALIBRATION_LOSS` 选择。设 $p = \mathrm{softmax}(z/T)$ 为校准后概率（masked，仅真实选项参与归一化），目标分布为 $t$，两种损失的目标分别为：

$$
\mathcal{L}_{\mathrm{Brier}} = \sum_k \left( p_k - t_k \right)^2 \qquad \mathcal{L}_{\mathrm{NLL}} = -\sum_k t_k \log p_k
$$

| 损失            | 求解器                            | 特点                                |
| ------------- | ------------------------------ | --------------------------------- |
| `brier`（默认）  | 粗网格（0.05~10, 50 步）+ 黄金分割         | 无 scipy/LBFGS 依赖，有界梯度，对噪声稳健       |
| `nll`         | LBFGS + strong_wolfe 线搜索       | 对"高置信但答错"惩罚更重，更激进地压制过度自信         |

**样本门槛**：总题数不足 10 条不启动校准（`SWIFT_CALIBRATION_MIN_ITEMS` 可调）；启动后某题型单独不足 10 条，该题型保持 $T_r = 1$（只校准够数的题型）。少量样本上拟合温度会被偶然模式主导，保持默认 T = 1（不修正）比错误修正更安全。

**报告指标（2-fold 交叉验证）**：为避免"用拟合数据给自己打分"的乐观偏差，日志里的 accuracy / ECE / Brier / NLL 是 **out-of-fold** 的——验证 record 按奇偶分两折，每折的温度在对面那折上拟合，再用来给本折打分。注意：**写进 checkpoint 供线上使用的温度仍是用全部验证数据拟合的**（生产估计），CV 只影响日志口径。同一条 record 的多道题总是落在同一折，避免信息泄漏。

### 日志解读

训练开始时先输出 base 基线（`on_train_begin`，用出厂/加载温度在验证集上评估，纯报告不写入）：

```
temperature_calibration: base metrics (pre-training, n=520): accuracy=0.6231, ece=0.1523, brier=0.2045, nll=1.1023
```

训练结束、校准完成后输出：

```
temperature_calibration done (loss=brier, head_T=1.412): noul: T=1.523 (Brier 0.1823 -> 0.1102, n=180); choice: T=1.298 (Brier 0.2151 -> 0.1534, n=220); score: T=1.611 (Brier 0.2401 -> 0.1620, n=120)
temperature_calibration CV (out-of-fold): accuracy=0.8077, ece=0.0311, brier=0.1420, nll=0.5621
temperature_calibration: base (pre-training, n=520) vs tuned (out-of-fold): accuracy 0.6231 -> 0.8077, ece 0.1523 -> 0.0311, brier 0.2045 -> 0.1420, nll 1.1023 -> 0.5621
temperature_calibration: written to ['output/scoring_head.json', ...]
```

- 第一行：每题型的最终 T、校准前后 in-sample Brier、样本数；
- 第二行：out-of-fold 指标。**accuracy 校准前后必然不变**（温度不改变 argmax），它是对照项；**ECE 是校准是否生效的最直接指标**（纯校准误差，不参与温度拟合）；
- 第三行：训练前 base vs 训练+校准后对比，量化整次微调的收益。

### 持久化与生效

拟合好的温度写入 head 的非持久化 buffer，并通过 `save_pretrained` 序列化到 `output_dir` 和最后 checkpoint 目录的 **`scoring_head.json`**（不进 safetensors，JSON 人类可读）。加载 checkpoint 时由 `maybe_load_trained_scoring_head` → `_load_calibration_meta` 自动恢复，线上推理的概率即为**校准后**温度下的输出，无需任何额外部署步骤。

### 多卡（DDP）下的行为

DDP 训练时验证集会被分片到各 rank（每个 rank 只看到约 1/world_size 的题），回调对此做了两层处理：

- **收集阶段**：各 rank 在自己的分片上前向收集 logits/target/mask 后，通过 `all_gather_object` 合并成**全量验证集**（record_id 按 rank 偏移防止 CV 分折碰撞）——保证温度在全量数据上拟合，而非某个分片；
- **拟合与写盘阶段**：仅主进程（rank 0）拟合温度、写 `scoring_head.json` 并输出日志，其余 rank 在 gather 后直接返回——保证落盘值与日志值一致、且结果确定。

### 环境变量

| 环境变量                        | 取值            | 默认     | 说明                            |
| --------------------------- | ------------- | ------ | ----------------------------- |
| `SWIFT_CALIBRATION_LOSS`    | `brier` / `nll` | `brier` | 拟合目标（非法值回退 brier 并告警）         |
| `SWIFT_CALIBRATION_MIN_ITEMS` | 正整数           | `10`   | 启动校准 / 单题型拟合的最少样本门槛           |

环境变量为进程级，多机训练需保证每个节点都已 export；其取值不会记录进 `args.json`，复现实验时请注意留存。

```bash
# 示例：切到 NLL 拟合目标，降低样本门槛
export SWIFT_CALIBRATION_LOSS=nll
export SWIFT_CALIBRATION_MIN_ITEMS=5
swift sft --task_type decision ... --split_dataset_ratio 0.1
```



## 数据集格式

每个模型有自己的原始 JSONL schema，预处理器将其归一化为统一内部契约。

### 单问题逐行 schema

**适用 `--model_type`：`jev`、`generic_decision`（`kev` 为其纯文本变体，见下文「变体」）**

此类模型**每个前向只评分一个问题**。含多个问题的 record 会被展开为每问题一行。

```json lines
{"state": "The sky is blue.", "questions": [{"kind": "noul", "question": "Is the sky blue?", "gold": 1}]}
{"state": "A review: the product arrived late.", "questions": [{"kind": "choice", "question": "Classify sentiment.", "options": ["positive", "negative", "neutral"], "gold": 1}]}
{"state": "some text to rate", "questions": [{"kind": "score", "question": "Rate quality.", "gold": [0.05, 0.1, 0.15, 0.4, 0.2, 0.1]}]}
```

- `state`：字符串（待评估的文本状态），也可以是 dict 或 JSON 对象。
- `questions`：问题 dict 列表。每项含 `kind`、`question`（文本）、`options`（仅 `choice` 需要；`noul` 和 `score` 使用固定 canonical 选项）、`gold`。
- `gold`：`noul` — int（0 = 否，1 = 是）；`choice` — int 选项索引；`score` — int 档位索引或完整软分布（float 列表，用于蒸馏）。

> **Canonical 选项**：JEV 对 `noul`/`score` 强制固定 canonical 选项并忽略调用方传入的选项文本，只有 `choice` 使用调用方选项。预处理器完整遵循此行为。

#### `kev` 变体

`kev` 使用同一结构（每前向一题、逐行展开，每行 = 状态 + 该问题的分支，普通因果序列，无需 branch mask），但为**纯文本**模型（`state` 不支持图像/视频），问题字段有原生别名：

```json lines
{"state": "The sky is blue.", "questions": [{"instr": "Is the sky blue?", "options": ["false", "true"], "label": 1}]}
{"state": "A review: late and damaged.", "questions": [{"instr": "Sentiment?", "options": ["positive", "negative", "neutral"], "label": 1}]}
{"state": "some text to rate", "questions": [{"instr": "Rate quality.", "options": ["0", "1", "2", "3", "4", "5"], "label": 4}]}
```

- `questions`：问题 dict 列表。原生字段为 `instr`（问题文本）、`options`（选项文本列表）、`label`（正确选项的**索引**）；统一 schema 的 `kind`/`question`/`gold` 也被接受。
- 未提供 `kind` 时按选项集推断：true/false 风格二元 → `noul`；全数字选项 → `score`；其余 → `choice`。
- **adapter-only 仓库**：仓库不含 config.json / 底座权重，`--model` 传底座、`--adapters` 传该仓库，并通过 `KEV_HEAD` 指向 adapter 目录下的 `head.pt`（见 `examples/train/decision/kev.sh`）。



### 多问题联合 schema

**适用 `--model_type`：`clef`**（全部 decision 模型中唯一的多问题单前向路径）

此类模型在**单次前向中联合解码一个 record 的所有问题**。

```json lines
{"id": "clef_noul", "state": {"text": "The sky is blue.", "lang": "en"}, "questions": {"is_blue": {"type": "noul", "instructions": "Is the sky blue?"}}, "answers": {"is_blue": true}}
{"id": "clef_choice", "state": "A review: the product arrived late.", "questions": {"sentiment": {"type": "choice", "instructions": "Classify sentiment.", "criteria": {"positive": "Good", "negative": "Bad", "neutral": "Mixed"}}}, "answers": {"sentiment": "negative"}}
{"id": "clef_score", "state": {"doc": "some text"}, "questions": {"quality": {"type": "score", "instructions": "Rate quality.", "criteria": ["terrible", "poor", "ok", "good", "great"]}}, "answers": {"quality": 3}}
```

- `state`：字符串或 dict（渲染为 SCHEMA FIELDS 布局）。
- `questions`：以问题 ID 为键的 dict。每个值含 `type`（`noul`/`choice`/`score`）、`instructions`、`criteria`（选项定义：`choice` 为 dict，`score` 为 list）。
- `answers`：以问题 ID 为键的 dict，给出 gold 答案（`noul` — bool；`choice` — key 字符串；`score` — int 索引）。



### 多模态 schema

**适用 `--model_type`：`omnijev`**

此类模型**每个前向只评分一个问题**，在**多模态**状态（图像或预合成的视频 mosaic）上进行。

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

ms-swift 为每个已接入模型提供训练脚本 `examples/train/decision/<model_type>.sh`（使用**本地 mini 数据集**，无需联网下载），通过 `--external_plugins examples/train/decision/dataset.py --dataset <model_type>_decision` 加载本地数据；同时支持使用**远程数据集**（`--dataset open_jev_<model_type>:all`，需在线下载 Open-Jev 语料）。两种方式仅数据来源不同，其余参数完全一致，详见[数据集](#数据集)章节。各模型的完整可运行命令（含 `--adapters` 路径获取、头制品环境变量导出等）见对应脚本。

### 命令骨架

所有 decision 模型共用同一条命令骨架——`--model` / `--model_type` / `--loss_type` 的取值见[支持模型](#支持模型)表格：

```bash
USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model <模型或底座，见支持模型表格> \
    --model_type <model_type> \
    --task_type decision \
    --loss_type <损失，默认值见支持模型表格> \
    --dataset open_jev_<model_type>:all \
    --torch_dtype bfloat16 \
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

- `--model` / `--model_type` / `--loss_type` — 取值及各模型默认损失见[支持模型](#支持模型)表格
- `--task_type decision` — 路由到 `ScoringTrainer` + decision data collator
- `--adapters` — 发布为 adapter 的模型传该 adapter 路径以续训（is_trainable=True）
- 头制品不在底座目录的模型（adapter-only 仓库）需先导出环境变量指向头制品，见[头制品环境变量覆盖](#头制品环境变量覆盖)
- 使用 branch 缓存机制的模型还需 `export MSO_BRANCH_CKPT=1` 且 `--gradient_checkpointing false`，见[下方说明](#omnijev-msol_branch_ckpt-与-gradient_checkpointing-互斥说明)
- `--split_dataset_ratio 0` — 示例脚本默认关闭验证集切分；如需训练后温度校准，改为 `0.1` 或显式传 `--val_dataset`



### 从零训练任意 CausalLM

`generic_decision` 模型类型允许将任意文本 CausalLM 转为决策模型。与各模型专属路径不同，此路径从 tokenizer 构建随机初始化的 `JevVerbalizerHead`，无需预置头权重：

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

数据格式使用单问题逐行 schema（见[数据集格式](#数据集格式)）。

## 数据集

Decision 模型的数据集有两种来源：**本地数据集**（通过 `--external_plugins` 注册，使用对应的 `ScoringPreprocessor` 子类）和**远程数据集**（SWIFT 内置的 Open-Jev 语料，直接用 `--dataset <名称>` 加载）。

> **重要**：Decision 数据**不能**通过 `--dataset /path/to/file.jsonl` 直接加载。swift 的标准数据加载方法（直接传文件路径，使用 `AutoPreprocessor`）不适用于 decision 数据——`AutoPreprocessor` 不认识 `{state, questions: [{kind, question, options, gold}]}` 这种 schema，会静默走错路径或报错。必须通过 `--external_plugins` 注册一个 Python 插件，用对应的 `ScoringPreprocessor` 子类处理数据。



### 远程数据集（Open-Jev）

SWIFT 将 [Open-Jev](https://www.modelscope.cn/datasets/ZefanCai/Open-Jev) 和 [Open-Jev-v1.1](https://modelscope.cn/datasets/ZefanCai/Open-Jev-v1.1)语料注册为内置远程数据集。无需 `--external_plugins`，直接用 `--dataset <名称>` 即可。


数据集名按 **`open_jev_<model_type>`** 规则命名——每个已接入模型（见[支持模型](#支持模型)表格）都按此约定注册了对应的远程数据集，预处理使用该模型专属的 `ScoringPreprocessor` 子类。使用 v1.1 语料时替换为带 `_v11` 后缀的名称（如 `--dataset open_jev_<model_type>_v11`）。

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

> **注意**：每行只渲染一个问题（`JevPreprocessor` 将记录拆为每问题一行），`noul`/`score` 的选项被强制设为标准 verbalizer 集合，`target` 被重映射到与标准选项对齐。



### 本地数据集

本地数据集需要编写 Python 插件文件，通过 `--external_plugins` 传入。插件使用 `register_dataset` 配合对应的 `ScoringPreprocessor` 子类。

#### 现成示例

仓库提供了一套完整可用的本地 mini 数据集，位于 `examples/train/decision/`：`dataset.py` 为插件注册文件，按 **`<model_type>_decision`** 规则为各模型注册本地数据集名；`data/<model_type>.jsonl` 为各 schema 的 mini 示例（4 行）；`<model_type>.sh` 为对应的训练脚本。


直接运行训练脚本即可：

```bash
sh examples/train/decision/<model_type>.sh
```

脚本内部通过 `--external_plugins examples/train/decision/dataset.py --dataset <model_type>_decision` 加载本地数据，无需下载远程数据集。而本文档正文训练命令用的是远程数据集（如 `--dataset open_jev_jev`），两者仅数据来源不同，其余参数完全一致。

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

根据数据 schema 选择预处理器（与[数据集格式](#数据集格式)的三类 schema 一一对应）：

- `JevPreprocessor` — 单问题逐行 schema（`{state, questions: [{kind, question, options, gold}]}`，每问题一行）
- `KevPreprocessor` — 单问题逐行 schema 的纯文本变体（`{state, questions: [{instr, options, label}]}`）
- `ClefPreprocessor` — 多问题联合 schema（`{state, questions: {qid: {type, instructions, criteria}}, answers}`，全部问题一行）
- `OmniJevPreprocessor` — 多模态 schema（含 `images`/`videos` 和选项块）
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

示例——启用 30% 指令丢弃、选项采样至 8、字段打乱：

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

所有 decision 模型共用同一条命令骨架——`--model` / `--model_type` 的取值见[支持模型](#支持模型)表格：

```bash
swift infer \
    --model <模型或底座，见支持模型表格> \
    --model_type <model_type> \
    --adapters output/checkpoint-xxx
```

- 头制品不在底座目录的模型（adapter-only 仓库）需先导出环境变量指向头制品，见[头制品环境变量覆盖](#头制品环境变量覆盖)；
- 使用 branch 缓存机制的模型还需 `export MSO_BRANCH_CKPT=1`，见[下方说明](#omnijev-msol_branch_ckpt-与-gradient_checkpointing-互斥说明)；
- 各模型完整可运行推理命令见 `examples/train/decision/` 下的对应示例脚本。

模型返回 per-option 概率（在各问题自己的选项集上 softmax），而非生成文本。每条记录的推理输出是一个列表，每问题一项：

```json
[{"index": 1, "probs": [0.1, 0.7, 0.2]}]
```

其中 `index` 为模型预测的选项索引（argmax of `probs`），`probs` 为各选项的 softmax 概率。

### 推理评测（accuracy）

推理时传入 `--metric acc`，swift 会自动对 decision 数据集计算 per-question accuracy：

```bash
swift infer \
    --model <模型或底座，见支持模型表格> \
    --model_type <model_type> \
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




### 头制品环境变量覆盖

当头制品与底座不在同一目录时（adapter-only 仓库、自定义部署路径），各模型通过环境变量覆盖头制品的加载路径，命名遵循 `<MODEL>_<制品>` 约定。已接入模型的环境变量：


| 环境变量                      | 适用 `--model_type` | 覆盖文件                     | 用途                            |
| ------------------------- | ---- | ------------------------ | ----------------------------- |
| `JEV_DECISION_HEAD`       | `jev`  | `decision_head.json`     | 覆盖 verbalizer bias / ids 加载路径 |
| `JEV_DECIDE_CALIBRATION`  | `jev`  | `calibration.json`       | 覆盖 per-kind 温度标定路径            |
| `CLEF_JOINT_HEAD_CONFIG`  | `clef` | `joint_head_config.json` | 覆盖头配置加载路径                     |
| `CLEF_JOINT_HEAD_WEIGHTS` | `clef` | `joint_head.safetensors` | 覆盖头权重加载路径                     |
| `KEV_HEAD`                | `kev`  | `head.pt`                | 指向 adapter 目录下的 pointer head 权重（kev 仓库为 adapter-only，必设） |


未设置时，加载器会自动在 `model_dir`（及 `jev` 的 `adapter_vllm/` 子目录）下查找对应文件。文件缺失时报错信息中会提示设置对应环境变量。

## 已知限制

- **多卡 FSDP / DeepSpeed-zero3 头存盘**：单卡正确，分片下 rank0 可能只拿到分片。
- `--target_modules all-linear` **注入 head**：`clef` / `omnijev` 的 head 含 Linear，all-linear 可能叠加 LoRA。
- **`jev` choice > 16 选项**：仅支持 `single` 模式（≤16 一次前向）。
- **`omnijev` 音频频谱图**：不在推理路径内，需用户自备转换器。
- `tie_word_embeddings=True` **adapter 合并**：仅影响合并/转格式，训练与推理不受影响。



## 验证报告

> **验证环境**：transformers 5.16.1 / ms-swift 4.6.0.dev0 / 8× L20X (143 GiB) / ffmpeg + ffprobe 已安装
>
> **验证方式**：4 个模型逐一执行「训练前推理 → 本地数据集训练 → 训练后推理」全流程，采用的CLI均为上述示例
>
> **Generic Decision**:采用的是Qwen2.5-0.5B 底座



### Benchmark — 训练后推理对比

| #   | 模型               | before acc | after acc  | Δ acc   | before top prob | after top prob | 结果      |
| --- | ------------ | ---------------- | ---------- | ---------- | ------- | --------------- | -------------- |
| 1   | JEV              | 0.6667     | **1.0000** | +0.3333 | 0.9855          | **1.0000**     | ✅ +50%  |
| 2   | Clef             | 0.5000     | **0.8333** | +0.3333 | 0.9712          | 0.9311         | ✅ +67%  |
| 3   | OmniJev          | 0.5500     | 0.7500     | +0.2884 | 0.8465          | 0.8196         | ✅ +37%  |
| 4   | Generic Decision | 0.1667     | **0.6667** | +0.5000 | 0.5682          | 0.9363         | ✅ +300% |

### 温度模块消融实验：同 checkpoint 有/无温度后处理

用**同一份 checkpoint**（权重与 bias 完全一致）在训练时真正留出的 80 条 held-out 数据上单次前向，对同一份 raw logits 分别按有无温度缩放计算指标（温度是 softmax 前的单调缩放，argmax 不变，故 acc 必须恒等——这也是实验的有效性校验）。

实验配置：JEV-27B-VL + 发布 adapter 续训（`open_jev_jev:all#800`，`split_dataset_ratio 0.1`，1 epoch，checkpoint-23）；脚本 `examples/train/decision/ablation_jev_confidence.py`。

| 变体 | T | acc | conf | gap(conf−acc) | ECE | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| **无温度** | 1.000 | 0.7500 | 0.7023 | −0.048 | 0.1650 | 0.0882 |
| checkpoint 落盘 T | 0.903 | 0.7500 | 0.7204 | −0.030 | 0.1304 | 0.0860 |
| 训练日志 T（分片拟合） | 0.528 | 0.7500 | 0.8132 | +0.063 | 0.0830 | 0.0829 |
| 全量重拟合（in-sample） | 0.618 | 0.7500 | 0.7871 | +0.037 | 0.0961 | 0.0823 |
| **2-fold out-of-fold** | ~0.645 | 0.7500 | 0.7873 | +0.037 | 0.1210 | **0.0798** |

分题型（无温度 → out-of-fold 温度）：

| 题型 | n | acc | conf 变化 | ECE 变化 | 说明 |
| --- | --- | --- | --- | --- | --- |
| noul | 50 | 0.900 | 0.781 → 0.862 | 0.130 → 0.089 | 欠自信主因，T<1 提信心方向正确 |
| choice | 22 | 0.636 | 0.613 → 0.705 | 0.185 → 0.228 | 中性偏正 |
| score | 8 | 0.125 | 0.456 → 0.550 | 0.379 → 0.455 | **反例**：样本少且 acc 极低，统一 T<1 有害，需 per-kind 温度 |

结论：

1. **方向与量级合理**：模型欠自信（conf 0.702 < acc 0.750，noul 差 −0.119），拟合出 T<1 正确；out-of-fold（未参与拟合的一半数据）Brier 0.0882→0.0798（−9.5%）、ECE 0.1650→0.1210，改善可泛化而非 in-sample 假象；
2. **per-kind 温度有价值**：score 题型 acc 仅 0.125 且样本少（8 < 10），统一温度对它有害——两阶段 per-kind 拟合的设计是必要的，样本不足的题型保持 T=1 是安全兜底；
3. **拟合样本量决定噪声**：10 条分片拟合出 0.528/0.903 两个互异值，全量 80 条拟合 0.618，40 条子集分别为 0.446/0.844——校准数据越多，温度越稳定。
