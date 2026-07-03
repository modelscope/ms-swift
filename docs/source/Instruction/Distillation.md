# 知识蒸馏（Knowledge Distillation）

知识蒸馏是一种将教师模型（teacher model）的能力迁移到学生模型（student model）的训练方法。其核心思想是：让学生在每个 token 位置上向教师的输出分布靠拢，从而获得比单纯模仿标注答案更丰富的监督信号——教师不仅告诉学生「哪个 token 是对的」，还告诉学生「其他 token 有多好 / 多差」。

本文档自顶向下地介绍：蒸馏为什么有效（第一节）、蒸馏方法的统一设计框架（第二节），最后落到 swift 中三种具体的蒸馏训练方法 GKD / OPD-RL / OPSD（第三节）。

---

## 一、为什么需要蒸馏：从稀疏信号到稠密信号

一个语言模型的能力，通常由一连串训练阶段堆叠而成：

- **预训练（Pre-training）**：习得语言、世界知识、基础推理等通用能力。
- **中训练（Mid-training）**：注入领域知识，如代码、医学、公司内部文档等。
- **后训练（Post-training）**：激发目标行为，如指令遵循、数学推理、对话风格等。

蒸馏主要发生在**后训练**阶段。要理解它的价值，需要从两个彼此独立的维度来看待后训练方法：

1. **采样方式（数据从哪来）**：训练序列是由学生自己生成（on-policy），还是来自外部固定数据（off-policy）。
2. **反馈密度（每条序列能学到多少）**：是整条序列只有一个奖励（per-sequence，稀疏），还是每个 token 都有信号（per-token，稠密）。


**SFT / 离线蒸馏**（off-policy + 稠密）：在固定数据上对齐标注或教师分布。信号稠密，但训练时见到的都是教师/标注的状态，和学生推理时自己会进入的状态不一致。学生一旦在推理早期犯了教师不会犯的错，就会进入训练中从未见过的状态，误差不断累积，这被称为 **exposure bias（曝光偏差）**。

**RL**（on-policy + 稀疏）：学生自己采样，按最终结果给奖励。分布与学生推理一致，但奖励通常是**序列级**标量，一般不指明具体哪个 token 出错。

**On-policy 蒸馏**（on-policy + 稠密）：学生自己采样轨迹，再由教师对轨迹的**每一个 token** 打分。训练分布与学生推理分布一致，且反馈为 per-token 级别。

### SFT 是蒸馏的一个特例

理解蒸馏的一个自然切入点是 SFT 的损失函数。SFT 的 cross-entropy loss，等价于以标注 token 的 one-hot 分布 $\delta_{y^*}$ 为「教师」的 KL 散度：

$$-\log P_S(y^*_t) = \text{KL}(\delta_{y^*} \,\|\, P_S)$$

知识蒸馏只是把这个确定性的 one-hot「教师」换成了真实教师模型的软分布 $P_T$，在每个 token 上优化 $\text{KL}(P_T \,\|\, P_S)$，从而提供比 one-hot 丰富的监督信号。

| 方法 | 采样方式 | 反馈密度 | 教师分布 |
|------|----------|----------|----------|
| SFT | off-policy（固定数据） | 稠密（per-token） | one-hot $\delta_{y^*}$ |
| 离线（off-policy）蒸馏 | off-policy（固定数据或教师生成） | 稠密（per-token） | 教师软分布 |
| RL | on-policy（学生采样） | 稀疏（per-sequence） | 无 |
| **On-policy 蒸馏** | **on-policy（学生采样）** | **稠密（per-token）** | **教师软分布** |


---

## 二、蒸馏的两个核心选择

不同蒸馏方法的差异，几乎都可以归结为两个问题。理解了这两个维度，后面的方法都只是它们的不同组合。

### 2.1 教师信号怎么算

在每个 token 位置上，量化教师分布 $P_T$ 与学生分布 $P_S$ 的差异（我们称之为 **Teacher KL**），有两个子选择。

**(a) 散度方向**

| 散度 | 定义 | 优化时的行为（信息论含义） |
|------|------|------|
| Forward KL | $\text{KL}(P_T \,\|\, P_S)$ | Mode-covering：学生需对教师概率较高的区域都赋予足够概率 |
| Reverse KL | $\text{KL}(P_S \,\|\, P_T)$ | Mode-seeking：学生主要拟合教师的众数（高概率）区域 |
| 广义 JSD($\beta$) | $\beta\,\text{KL}(P_T\|M) + (1-\beta)\,\text{KL}(P_S\|M)$，其中 $M=\beta P_T+(1-\beta)P_S$ | 在两者之间插值 |

> 其中 $\beta=0$ 退化为 Forward KL，$\beta=1$ 退化为 Reverse KL。SFT 等价于 Forward KL（教师为 one-hot）。

在 swift 中：
- GKD 默认 $\beta=0.5$（JSD），可通过 `--beta` 在 Forward / JSD / Reverse 之间选择；
- OPD-RL 的实现固定使用 Reverse KL 的 k1 估计量 $\log\pi_{\text{teacher}}(y_t)-\log\pi_{\text{student}}(y_t)$ 作为 per-token advantage。

**(b) 计算粒度**

| 计算粒度 | 需要的教师信息 | 说明 |
|----------|---------------|------|
| 全词表 | 教师完整的 next-token 分布 | 可计算散度的精确值；显存开销大 |
| Top-K | 教师概率最高的 K 个 token | 在 top-K 子集上重新归一化后的近似；适合外部 API（受 `max_logprobs` 限制） |
| 采样 token | 教师在学生实际采样 token 上的单个 logp | Reverse KL 的单样本蒙特卡洛估计；通信开销最低 |

> **精度 vs 开销**：全词表需要物化完整 logits；采样 token 只需教师在已采样 token 上的 logp（可走远程 API）。[DeepSeek-V4](https://arxiv.org/abs/2606.19348)技术报告指出，仅用采样 token 的 log-ratio 作 advantage 时梯度估计方差较大，因此其全词表 OPD 采用完整 logit 蒸馏

### 2.2 信号怎么传给学生

|  | **路径 A：GKD（直接损失）** | **路径 B：OPD-RL（RL Advantage）** |
|---|---|---|
| 训练范式 | `--rlhf_type gkd` | `--rlhf_type grpo` + 教师 |
| 信号传递 | 把信号作为 loss | 把信号当 advantage，走 policy gradient |
| 梯度流经 | 学生**全词表** logits（或 top-k） | 仅学生**采样 token** 的 $\nabla\log\pi(y_t)$ |
| 教师信息需求 | 全词表分布（或 top-k logits） | 采样 token 上的单个 logp |
| 散度选择 | Forward / Reverse / JSD（`--beta`） | Reverse KL（k1 log-ratio） |
| 与任务奖励组合 | 通过 `sft_alpha` 混合 SFT loss | 可与 GRPO reward 叠加为 advantage |

两者**共享同一套教师基础设施**（见下文），区别只在如何使用 teacher KL 信号。

> **蒸馏的常见用法**
> 1. **能力融合**：多个专家模型蒸馏到统一模型。
> 2. **强到弱**：大模型向小模型传递能力。
> 3. **防遗忘**：用旧 checkpoint 作教师，在多阶段训练后恢复先前能力。


---

## 三、swift 中的蒸馏方法

swift 提供三种蒸馏训练方法，它们共享同一套教师基础设施，差异落在第二节的框架里：

| 方法 | 信号传递路径 | 启用方式 | 一句话 |
|------|--------------|----------|--------|
| **GKD** | 直接损失（路径 A） | `--rlhf_type gkd` | 教师散度作为 loss 反向传播；支持全词表 / top-k 散度 |
| **OPD-RL** | RL advantage（路径 B） | `--rlhf_type grpo` + 教师 | 教师 log-ratio 注入 GRPO advantage，可与任务奖励叠加 |
| **OPSD** | A 或 B 均可 | 在上面基础上提供 `teacher_prompt` | 单模型自蒸馏：教师输入含特权信息（如参考解答） |

**教师的三种来源**（GKD 与 OPD-RL 通用）：

- `--teacher_model`：在训练进程中加载一个独立的冻结教师模型。
- `--teacher_model_server`：连接一个外部教师服务（`swift deploy --infer_backend vllm`），不在训练卡上加载教师。GKD 使用 API 时需同时设置 `--gkd_logits_topk`。支持多 teacher 配置（见 [3.4 节](#34-multi-teacher-蒸馏)）。
- **自蒸馏**：教师与学生同源。LoRA 训练且 `--teacher_model` 与 `--model` 相同时，自动用 `disable_adapter()` 以基座为固定教师，无需额外加载；GKD 下不传 `--teacher_model` 则以「学生当前权重」为动态教师。

**教师相关参数**（GKD 与 OPD-RL 共享，完整说明见[命令行参数](./Command-line-parameters.md)）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teacher_model` | None | 教师模型路径；GKD 下不传则为动态自蒸馏 |
| `--teacher_model_server` | None | 教师 API 地址（与 `teacher_model` 互斥）；支持多 teacher JSON |
| `--teacher_tag_key` | `"dataset"` | 多 teacher 路由的字段名 |
| `--teacher_deepspeed` | None | 教师模型的 DeepSpeed 配置（如 `zero3`） |
| `--offload_teacher_model` | False | 非前向阶段将教师卸载到 CPU |

---

### 3.1 GKD：散度作为直接损失

GKD（[Generalized Knowledge Distillation](https://arxiv.org/pdf/2306.13649)）直接把教师-学生间的散度作为损失函数反向传播。

**损失函数**

$$
\mathcal{L}_{\text{GKD}}(x, y) = \sum_{t=1}^{|y|} D_{\text{JSD}(\beta)}\big(P_{\text{teacher}}(\cdot|x,y_{<t}),\, P_{\text{student}}(\cdot|x,y_{<t})\big)
$$

其中散度 $D$ 由 `--beta` 选择（见 2.1）：$\beta=0$ 为 Forward KL，$\beta=1$ 为 Reverse KL，$0<\beta<1$ 为广义 JSD（默认 $0.5$）。

**On-Policy vs Off-Policy：`lmbda`**

GKD 通过 `lmbda` 控制每个 batch 用学生在线采样的概率：

```python
if random() <= lmbda:
    y = student.generate(x)   # on-policy：学生自己采样
else:
    y = y_ground_truth        # off-policy：使用数据集标注
loss = D(P_teacher(·|x, y), P_student(·|x, y))
```

- `lmbda=0`：纯离线（传统 SFT 蒸馏）。
- `lmbda=1`：纯在线（学生从自身错误中学习，即 on-policy 蒸馏）。
- `0<lmbda<1`：混合。

> 若希望走teacher生成数据路径，需先用教师离线生成响应写入数据集，再设 `lmbda=0` 训练

**GKD参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--beta` | float | 0.5 | 散度插值：0=Forward KL，0.5=JSD，1=Reverse KL |
| `--lmbda` | float | 0.5 | 在线采样概率：0=离线，1=纯在线 |
| `--sft_alpha` | float | 0 | 混合 SFT loss 比例，最终 `loss = gkd_loss + sft_alpha * sft_loss`（仅对**非学生生成**的数据生效） |
| `--gkd_logits_topk` | int | None | 仅用教师 top-K logits 计算 KL；使用 `teacher_model_server` 时为必填 |

**Top-K 蒸馏（省显存）**

默认用完整词表计算 KL，词表很大时容易 OOM，可使用`--gkd_logits_topk`参数。

**外部教师 API**

设置 `--teacher_model_server` 时需同时设置 `--gkd_logits_topk`（API 仅返回 top-k logprobs）。示例如下：

```bash
# 步骤 1：部署教师模型（max_logprobs 需 >= gkd_logits_topk）
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen3.5-9B \
    --infer_backend vllm \
    --port 8000 \
    --max_logprobs 64

# 步骤 2：启动 GKD 训练
CUDA_VISIBLE_DEVICES=1,2,3,4 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-2B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 64 \
    --lmbda 1.0 \
    --beta 1.0 \
    --dataset xxx
```

**在线采样加速**

`lmbda > 0` 时学生需在线生成序列，建议用 vLLM 加速采样（colocate / server 两种模式，与 GRPO 一致），参考 [GRPO 文档](./GRPO/GetStarted/GRPO.md#集群支持)。

**参考脚本**

- 基础训练：[examples/train/rlhf/gkd/](https://github.com/modelscope/ms-swift/tree/main/examples/train/rlhf/gkd/)
- 多模态：[examples/train/multimodal/rlhf/gkd/](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/rlhf/gkd/)
- Megatron：[examples/megatron/rlhf/gkd/](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/rlhf/gkd/)


---

### 3.2 OPD-RL：KL 作为 RL Advantage

OPD（On-Policy Distillation）RL 把教师 KL 信号注入 GRPO 的 **per-token advantage**，通过 policy gradient 更新学生。

**原理**

标准 GRPO 的 advantage 来自组内归一化的任务奖励（per-sequence 标量）。OPD-RL 在 advantage 归一化**之后**逐 token 注入教师信号：

$$
A_t = A_t^{\text{base}} + \alpha \cdot \big(\log \pi_{\text{teacher}}(y_t|x,y_{<t}) - \log \pi_{\text{student}}(y_t|x,y_{<t})\big)
$$

- $A_t^{\text{base}}$：GRPO 归一化后的任务奖励 advantage（无奖励函数时为 0）。
- $\alpha$：`--teacher_kl_coef`，教师信号强度。
- $\log\pi_{\text{teacher}}(y_t) - \log\pi_{\text{student}}(y_t)$：教师 log-ratio，即 Reverse KL 梯度中 $\nabla_\theta\log\pi_\theta(y_t)$ 系数对应的 k1 估计量（见 `compute_teacher_logratio`）。

**纯蒸馏模式**：不设 `--reward_funcs` 时 base advantage 为 0，教师信号是唯一驱动力，此时 $A_t = \alpha\cdot(\log\pi_{\text{teacher}}(y_t)-\log\pi_{\text{student}}(y_t))$。

**监控指标**：日志中的 `teacher_kl` 是 k3 估计量 $e^{d}-d-1$（$d=\log\pi_{\text{teacher}}-\log\pi_{\text{student}}$），衡量学生与教师的距离。

**启用方式**：在 `--rlhf_type grpo` 下设置 `--teacher_model` 或 `--teacher_model_server` 即自动启用 OPD-RL，无需额外开关。教师相关参数见上文共享参数表。

**OPD-RL 特有参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teacher_kl_coef` | 1.0 | 教师 log-ratio 注入 advantage 的系数 $\alpha$ |

**参考脚本**

- HF：[examples/train/grpo/opd_rl.sh](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/opd_rl.sh)
- Megatron：[examples/megatron/grpo/opd_rl.sh](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/grpo/opd_rl.sh)
- Ray：[examples/ray/grpo/run_opd.sh](https://github.com/modelscope/ms-swift/tree/main/examples/ray/grpo/run_opd.sh)


---

### 3.3 OPSD：On-Policy Self-Distillation

OPSD（[On-Policy Self-Distillation](https://arxiv.org/abs/2601.18734)）是一种单模型自蒸馏方法：同一模型分别构造学生输入与教师输入，教师侧额外接收**特权信息**（如参考解答），再对齐两者在学生采样响应上的输出分布。

**核心机制**

- **学生**：仅看到问题，正常推理。
- **教师**：看到问题 + 参考解答（通过 `teacher_prompt` 列提供特权信息）。
- **训练目标**：用散度（JSD / KL）对齐学生与教师在同一份学生采样响应上的输出分布。

OPSD 既可走 GKD 路径，也可走 OPD-RL 路径：

- **GKD + OPSD**：`--rlhf_type gkd`，教师 KL 作为直接损失。
- **OPD-RL + OPSD**：`--rlhf_type grpo` + `--teacher_model`（与 `--model` 相同），教师 KL 作为 advantage。

**两种自蒸馏权重模式**

| 模式 | 参数配置 | 教师权重 | 说明 |
|------|---------|---------|------|
| **Dynamic（动态）** | 不传 `--teacher_model` | 学生当前权重 | 教师随训练同步更新 |
| **Fixed（固定）** | `--teacher_model` 设为与 `--model` 相同 | 初始教师权重 | 教师权重固定 |

**数据格式**

OPSD 数据集需包含 `teacher_prompt` 列，可通过 `--external_plugins` 加载数据处理插件来构建。以数学推理数据集 `open-r1/OpenThoughts-114k-math` 为例：

```python
from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset

class OpenThoughtsOPSDPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        if not row.get('correct', True):
            return None
        problem = row.get('problem', '')
        solution = row.get('solution', '')
        teacher_prompt = f'{problem}\n\nReference solution:\n{solution}\n\nNow articulate your own reasoning.'
        messages = [
            {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
            {'role': 'user', 'content': problem},
        ]
        return {'messages': messages, 'teacher_prompt': teacher_prompt}

register_dataset(DatasetMeta(
    ms_dataset_id='open-r1/OpenThoughts-114k-math',
    preprocess_func=OpenThoughtsOPSDPreprocessor(),
    tags=['math', 'opsd'],
))
```

**参考脚本**

- HF：[examples/train/rlhf/opsd/](https://github.com/modelscope/ms-swift/tree/main/examples/train/rlhf/opsd/)
- Megatron：[examples/megatron/rlhf/gkd/opsd.sh](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/rlhf/gkd/opsd.sh)


---

### 3.4 Multi-Teacher 蒸馏

Multi-Teacher OPD 允许同时使用多个外部教师 API 服务指导一个学生模型，按数据集或样本级别将不同数据路由到不同的教师，实现领域专家蒸馏。

**核心机制**

`--teacher_model_server` 参数同时支持单 teacher（URL 字符串）和多 teacher（JSON 数组）：

```bash
# 单 teacher
--teacher_model_server http://localhost:8000

# 多 teacher
--teacher_model_server '[{"url":"http://t1:8000","tags":["math"]},{"url":"http://t2:8001","tags":["code"]}]'
```

每个 teacher 配置项：

| 字段 | 说明 |
|------|------|
| `url` | 教师 API 地址 |
| `tags` | 该教师负责的标签列表；多 teacher 时**必须非空且各教师之间不重叠**（每条数据只归一个教师） |

所有教师共用全局 `--teacher_kl_coef` 作为 KL 系数。

**两种路由模式**

**模式一：数据集级路由（自动注入）**

当加载多个数据集时，Swift 自动为每条数据注入 `dataset` 列（值为数据集名），无需修改数据文件：

```bash
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --dataset AI-ModelScope/alpaca-gpt4-data-en AI-ModelScope/alpaca-gpt4-data-zh \
    --teacher_model_server '[{"url":"http://t1:8000","tags":["alpaca-gpt4-data-en"]},{"url":"http://t2:8001","tags":["alpaca-gpt4-data-zh"]}]' \
    ...
```

数据集名为 `--dataset` 参数中传入的值。`--teacher_tag_key` 默认为 `"dataset"`。

**模式二：样本级路由（用户指定列）**

用户在数据文件中添加自定义列（如 `teacher_tag`），每条数据指定使用哪个教师：

```jsonl
{"messages": [{"role": "user", "content": "math problem..."}], "teacher_tag": "math"}
{"messages": [{"role": "user", "content": "code problem..."}], "teacher_tag": "code"}
```

配置 `--teacher_tag_key teacher_tag` 即可按该列路由。

**路由规则**

- `tags=["math"]` = 只处理 tag 为 `"math"` 的样本
- 每条数据按 tag 命中**唯一一个**教师；各教师 tags 不重叠，故不会重复路由
- 未被任何教师匹配的样本：直接报错（fail-fast），提示配置未覆盖该 tag

**KL 注入**

每条数据用其所属教师的 logp，配合全局系数 $\alpha$（`--teacher_kl_coef`）逐 token 注入 advantage：

$$
A_t = A_t^{\text{base}} + \alpha \cdot \big(\log \pi_{T_i}(y_t) - \log \pi_S(y_t)\big)
$$

其中 $T_i$ 为该样本路由到的教师。

**实验示例**

```bash
# 步骤 1：启动两个教师服务
CUDA_VISIBLE_DEVICES=1 swift deploy --model Qwen/Qwen3.5-4B --port 8000 --max_logprobs 1
CUDA_VISIBLE_DEVICES=2 swift deploy --model Qwen/Qwen3.5-1.7B --port 8001 --max_logprobs 1

# 步骤 2：启动多 teacher GRPO 训练
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-0.6B \
    --teacher_model_server '[{"url":"http://localhost:8000","tags":["math"]},{"url":"http://localhost:8001","tags":["code"]}]' \
    --dataset data/math.jsonl data/code.jsonl \
    --use_vllm true --vllm_mode colocate \
    ...
```

支持 HF GRPO、Megatron GRPO（含 Ray pipeline）三条路径。

**参考脚本**：[scripts/test_multi_teacher.sh](https://github.com/modelscope/ms-swift/tree/main/scripts/test_multi_teacher.sh)


---

## Reference

- Kevin Lu & Thinking Machines Lab. [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/). 2025.
- Agarwal et al. [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/pdf/2306.13649). 2023.
- Gu et al. [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543). 2023.
- DeepSeek-AI. [DeepSeek-V4](https://arxiv.org/html/2606.19348v1). 2026.
- Qwen Team. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). 2025.
- Zhipu AI. [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/html/2602.15763). 2026.
- Kimi Team. [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276). 2026.
