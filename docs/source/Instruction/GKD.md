# GKD

GKD（Generalized Knowledge Distillation，广义知识蒸馏）训练算法由论文 [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/pdf/2306.13649) 提出。该算法通过结合离线（off-policy）和在线（on-policy）学习策略，将教师模型的知识迁移到学生模型中。

## 损失函数

当给定输入序列 $x$ 与输出序列 $y$，GKD 的损失函数可以写为：

$$
\mathcal{L}_{\text{GKD}}(x, y) = \sum_{t=1}^{|y|} D(P_{\text{teacher}}(\cdot | x, y_{<t}), P_{\text{student}}(\cdot | x, y_{<t}))
$$

其中：
- $y_{<t} = (y_1, y_2, \ldots, y_{t-1})$：前 $t-1$ 个 token 的序列
- $P_{\text{teacher}}(\cdot | x, y_{<t})$：教师模型在给定上下文 $x, y_{<t}$ 时的输出概率分布
- $P_{\text{student}}(\cdot | x, y_{<t})$：学生模型在给定上下文 $x, y_{<t}$ 时的输出概率分布
- $D(\cdot, \cdot)$：散度函数，用于度量两个概率分布之间的差异性

## 散度度量函数

### KL 散度（Kullback-Leibler Divergence）

KL 散度是衡量两个概率分布 $P$ 和 $Q$ 之间差异的非对称度量：

$$
\text{KL}(P \| Q) = \sum_v P(v) \log \frac{P(v)}{Q(v)} = \mathbb{E}_{v \sim P}\left[\log \frac{P(v)}{Q(v)}\right]
$$

### Forward KL 与 Reverse KL

在知识蒸馏中，根据 KL 散度中两个分布的顺序不同，有两种选择：

#### Forward KL（前向 KL）

$$
\text{KL}(P_{\text{teacher}} \| P_{\text{student}}) = \sum_v P_{\text{teacher}}(v) \log \frac{P_{\text{teacher}}(v)}{P_{\text{student}}(v)}
$$

**特性**：Mode-covering
- 期望在教师分布下计算
- 学生模型倾向于覆盖教师的整个分布（包括低概率区域）


#### Reverse KL（反向 KL）
$$
\text{KL}(P_{\text{student}} \| P_{\text{teacher}}) = \sum_v P_{\text{student}}(v) \log \frac{P_{\text{student}}(v)}{P_{\text{teacher}}(v)}
$$

**特性**：Mode-seeking
- 期望在学生分布下计算
- 学生模型倾向于集中在教师模型的峰值区域（高概率区域）

### 广义 Jensen-Shannon 散度（Generalized JSD）

GKD 使用广义 JSD 作为核心度量，通过参数 $\beta \in [0, 1]$ 在 Forward KL 和 Reverse KL 之间进行**平滑插值**。

对于两个概率分布 $P$ 和 $Q$，广义 JSD 定义为：

$$
D_{\text{JSD}(\beta)}(P, Q) = \beta \cdot \text{KL}(P \| M) + (1-\beta) \cdot \text{KL}(Q \| M)
$$

其中混合分布 $M$ 定义为：

$$
M = \beta \cdot P + (1-\beta) \cdot Q
$$

- 当 $\beta = 0.5$ 时，退化为标准的对称 JSD
- 通过调节 $\beta$，可以在 Mode-seeking 和 Mode-covering 之间权衡

在 GKD 中，我们令 $P = P_{\text{teacher}}$，$Q = P_{\text{student}}$，因此：

$$
D_{\text{JSD}(\beta)}(P_{\text{teacher}}, P_{\text{student}}) = \beta \cdot \text{KL}(P_{\text{teacher}} \| M) + (1-\beta) \cdot \text{KL}(P_{\text{student}} \| M)
$$

其中 $M = \beta \cdot P_{\text{teacher}} + (1-\beta) \cdot P_{\text{student}}$

> 对极端情况（$\beta = 0$ 或 $\beta = 1$），直接计算单个 KL 散度：
> - 当 $\beta = 0$ 时：直接定义 $D = \text{KL}(P_{\text{teacher}} \| P_{\text{student}})$（Forward KL，Mode-covering）
> - 当 $\beta = 1$ 时：直接定义 $D = \text{KL}(P_{\text{student}} \| P_{\text{teacher}})$（Reverse KL，Mode-seeking）
> - 当 $0 < \beta < 1$ 时：使用上述混合分布公式进行插值

通过调节 $\beta$ 参数，可以在不同的散度度量之间进行插值，当 $\beta = 0.5$ 时，散度为标准的对称 JSD。

## 三种训练模式

GKD训练具有三种训练模式，区别在于输出序列 $y$ 的来源。

### 模式选择逻辑

训练时，每个样本按照以下优先级选择模式：

```python
# 伪代码：模式选择逻辑
if random() < lmbda:
    # Mode 1: On-Policy 学习，由学生模型采样输出序列
    y = student.generate(x)
    source = "student"
elif seq_kd:
    # Mode 2: Sequential KD，由教师模型采样输出序列
    y = teacher.generate(x)
    source = "teacher"
else:
    # Mode 3: 使用数据集中的输出序列
    y = y_ground_truth
    source = "dataset"

# 相同的损失函数
loss = D_JSD(P_teacher(·|x,y), P_student(·|x,y))
```

### Mode 1: On-Policy 学习
设置参数`lambda`, 以概率 $\lambda$ 触发，使用学生模型采样 $y \sim P_{\text{student}}(\cdot | x)$

- 学生模型从**自己生成的序列**中学习
- 暴露在自己可能犯的错误中，学会**自我纠正和错误恢复**
- 对齐训练分布与推理分布
- 提升模型的鲁棒性和实际应用表现

**适用场景**：
- 学生模型已有一定生成能力
- 希望提升模型在真实推理场景下的表现

### Mode 2: Sequential KD（`seq_kd=True` 且未触发 on-policy）
设置参数 `seq_kd=True`, 当未触发 on-policy 时，使用教师模型采样

**数据来源**：$y \sim P_{\text{teacher}}(\cdot | x)$

### Mode 3: 离线学习（其他情况）

**数据来源**：$y = y^* \sim \text{Dataset}$

- 学生模型从**数据集的标注序列**中学习


## 参数设置

我们可以通过设置以下参数进行 GKD 训练：

### 基础参数

| 参数 | 类型 | 默认值 | 取值范围 | 说明 |
|------|------|--------|---------|------|
| `--teacher_model` | str | None | - | 教师模型路径或模型 ID<br>*使用 `teacher_model_server` 时可省略 |
| `--beta` | float | 0.5 | [0.0, 1.0] | 散度插值系数<br>• 0.0: Forward KL <br>• 0.5: JSD (平衡)<br>• 1.0: Reverse KL |
| `--lmbda` | float | 0.5 | [0.0, 1.0] | On-Policy 学习触发概率<br>• 0.0: 离线学习<br>• 0.5: 混合策略<br>• 1.0: 纯 On-Policy |
| `--seq_kd` | bool | False | True/False | 是否使用教师生成序列<br>• False: 非 on-policy 时使用数据集<br>• True: 非 on-policy 时使用教师生成 |
| `--temperature` | float | 0.9 | > 0 | 生成采样温度，控制随机性 |
| `--sft_alpha` | float | 0 | >= 0 | 混合一定比例的sft loss，对非student生成结果生效 |
| `--max_completion_length` | int | 512 | > 0 | 生成时的最大 token 数 |

### Top-K KL 计算

默认情况下，GKD 使用完整词表计算 KL 散度，容易造成 OOM，这种情况下可以使用 **Top-K** 模式来减少显存占用和计算量。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gkd_logits_topk` | int | None | Top-K logits 数量<br>• None: 使用完整词表（默认）<br>• 正整数: 仅使用教师模型概率最高的 K 个 token 计算 KL |

**Top-K 模式原理**：

在 Top-K 模式下，选取**教师模型**输出概率最高的 K 个 token，在这个子集上计算两个模型分布的 KL 散度。
$$
D_{\text{JSD}(\beta)}^{\text{top-k}}(P_T, P_S) = \beta \cdot \text{KL}(\tilde{P}_T \| \tilde{M}) + (1-\beta) \cdot \text{KL}(\tilde{P}_S \| \tilde{M})
$$

其中 Top-K 索引来自教师模型：$\text{Top-K} = \text{argtop}_K(P_T)$，$\tilde{P}_T$ 和 $\tilde{P}_S$ 是在 Top-K 子集上**重新归一化**的概率分布：

$$
\tilde{P}_T(v) = \frac{P_T(v)}{\sum_{v' \in \text{Top-K}} P_T(v')}, \quad \tilde{P}_S(v) = \frac{P_S(v)}{\sum_{v' \in \text{Top-K}} P_S(v')}, \quad v \in \text{Top-K}
$$

**使用示例**：

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B-Instruct \
    --teacher_model Qwen/Qwen2.5-72B-Instruct \
    --gkd_logits_topk 64 \
    --dataset your_dataset \
    ...
```

> **注意**：Top-K 模式不能与 liger kernel 同时使用（`--use_liger_kernel`）。

### 外部教师模型 API

当设置 `gkd_logits_topk` 时，可以使用外部教师模型 API 服务来获取 logprobs，这样可以避免在训练进程中加载教师模型。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--teacher_model_server` | str | None | 教师模型服务地址<br>如：`http://localhost:8000` |
| `--gkd_logits_topk` | int | **必需** | 使用外部 API 时必须设置，对应 API 返回的 top_logprobs 数量 |

**支持的后端**：
- `swift deploy`（vLLM backend）
- 独立 vLLM 服务（`vllm serve`）

**步骤 1：部署教师模型服务**

```bash
# 使用 swift deploy 部署教师模型
CUDA_VISIBLE_DEVICES=0,1 swift deploy \
    --model Qwen/Qwen2-72B-Instruct \
    --infer_backend vllm \
    --port 8000 \
    --vllm_engine_kwargs '{"max_logprobs": 64}'

# 或使用独立 vLLM 服务
vllm serve Qwen/Qwen2-72B-Instruct --max-logprobs 64 --port 8000
```

**步骤 2：启动 GKD 训练**

```bash
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2-7B-Instruct \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 20 \
    --dataset your_dataset \
    --lmbda 1.0 \
    --beta 0.5 \
    ...
```

> **vLLM max_logprobs 限制**：
> - vLLM 默认 `max_logprobs=20`，可通过 `--vllm_engine_kwargs '{"max_logprobs": N}'` 参数调整
> - `gkd_logits_topk` 不能超过服务端的 `max_logprobs` 设置

## 采样加速

在 GKD 训练中，涉及到两种在线采样的情况：

1. **学生模型采样**（当 `lmbda > 0`）：以 $\lambda$ 概率触发学生模型采样
2. **教师模型采样**（当 `seq_kd=True`）：以 $1-\lambda$ 概率触发教师模型采样

由于采样过程会显著减慢训练速度，可参考以下两种加速方案：

### 方案 1：学生模型采样加速

**要求**：swift >= 3.10.dev

使用 vLLM 作为推理后端来加速学生模型采样，支持两种部署模式，与 GRPO 一致，参考[GRPO文档](./GRPO/GetStarted/GRPO.md#集群支持), 相关参数参考[GRPO vLLM 参数](./Command-line-parameters.md#vllm_mode)

> **注意**：vLLM 加速仅适用于学生模型的 on-policy 采样（`lmbda > 0`）。教师模型的 sequential KD 采样（`seq_kd=True`）目前仍使用 Transformers，建议使用预采样方案。

训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/rlhf/gkd/vllm_server.sh)

### 方案 2：教师模型预采样

对于教师模型采样（`seq_kd=True`），推荐使用 **预采样** 方式：先用教师模型离线生成高质量数据，再进行训练。

**步骤 1：使用教师模型生成数据**
```bash
export teacher_model='OpenGVLab/InternVL3-8B'

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model $teacher_model \
    --infer_backend vllm \
    --val_dataset 'modelscope/coco_2014_caption:validation#5000' \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --write_batch_size 1000 \
    --result_path teacher_generated_data.jsonl
```

**步骤 2：使用预生成数据训练**
```bash
swift rlhf \
    --rlhf_type gkd \
    --model OpenGVLab/InternVL3-2B-Pretrained \
    --teacher_model $teacher_model \
    --dataset 'teacher_generated_data.jsonl' \
    --seq_kd false \
    ...
```

训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/rlhf/gkd/fast.sh)

## On-Policy Distillation

我们可以通过设置以下参数实现 Thinking Machine Lab blog 中的[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)训练。
```bash
--lmbda 1 # on-policy
--beta 1 # reverse
```

相关脚本可以参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/on_policy_distillation.sh)
