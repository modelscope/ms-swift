# GRPO

论文地址

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

环境安装
```bash
pip install math_verify==0.5.2 # reward function
pip install -U trl
```

GRPOTrainer在swift3.5.dev进行了代码重构，如果你使用的swift版本<3.5, 请参考[stable文档](https://github.com/modelscope/ms-swift/blob/v3.4.1/docs/source/Instruction/GRPO.md)

**更新日志**
- **2025-05-29** — 支持了padding_free(--padding_free true)和序列并行(--sequence_parallel_size N)。
- **2025-05-23** — 支持自定义采样批量大小，参考 generation_batch_size / steps_per_generation 参数。
- **2025-05-22** — swift rollout 支持 data_parallel_size 参数。
- **2025-05-16** - 增加 ref_model 同步逻辑，参考参数 sync_ref_model。
- **2025-05-13** — 为了代码的可读性和维护性， GRPOTrainer代码重构，Internal mode 支持vLLM>=0.8。
- **2025-05-11** — 支持生成式奖励模型，通过 reward_model_plugin 自定义奖励模型逻辑。有关更多详细信息，请参阅[自定义奖励模型](#自定义奖励模型)部分。
- **2025-04-30** — external vllm server 的启动命令改为 `swift rollout`。

## 集群支持

![](../../resources/grpo.png)

GRPO 训练框架支持集成高性能推理引擎（如 vLLM）来加速采样过程，提供以下两种部署模式：

### 1. Colocate(Internal) Mode

- 训练与推理共享GPU资源，在 Trainer 内部启动推理服务，

启动参数
```bash
--use_vllm true \
--vllm_mode colocate
```

#### Colocate 模式下的显存优化方案
在 Colocate 模式下运行时，容易出现显存不足（OOM）的情况。以下是几种有效的显存优化方法和参数配置：

1. 在训练阶段，释放 vLLM 占用的显存：

```bash
--sleep_level 1
```

2. 在vLLM 推理阶段，释放模型和优化器占用的显存：

```bash
--offload_optimizer true \
--offload_model true \
--gc_collect_after_offload true \
```

3. 在vLLM中使用 Tensor Parallel 技术：

```bash
--vllm_tensor_parallel_size [tp_size]
```

4. 分批 Gather 模型权重（zero3下同步 vLLM 权重时）：

```bash
--move_model_batches [批次数量]
```

### 2. Async(External) Mode

- 训练与推理资源分离，启动单独的推理服务器

使用`swift rollout`命令部署vLLM 服务器, 现仅支持vLLM backend
```bash
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor_parallel_size 2 \
  --data_parallel_size 1

CUDA_VISIBLE_DEVICES=0,1 \
swift rollout \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor_parallel_size 2 \
  --data_parallel_size 1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rollout \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor_parallel_size 2 \
  --data_parallel_size 2
```

对于更多 vLLM 参数，你可以参考[vLLM参数](./命令行参数.md#vllm参数)

训练使用以下参数配置外部 vLLM 服务器
```bash
--use_vllm true \
--vllm_mode server \
--vllm_server_host <服务器IP> \
--vllm_server_port <服务端口> \
--vllm_server_timeout <超时时间> \
```

## 奖励函数
### 自定义奖励函数
奖励函数接受模型生成的文本 completions 以及其他数据集中的列作为参数(kwargs)，并对模型生成的文本进行打分。以下是一个示例，展示了如何实现一个简单的长度奖励函数。该函数会在模型生成的文本长度超过 1024 时，给予 1.0 的奖励信号；否则，奖励信号为 0.0。

```python
from swift.plugin import ORM, orms
class DummyLengthRewardFunction(ORM)
    def __call__(completions, **kwargs):
        return [1.0 if len(completion) > 1024 else 0.0 for completion in completions]

orms['dummy']= DummyLengthRewardFunction
```

可以在`swift/examples/train/grpo/plugin/plugin.py`中加入该奖励函数，使用参数`--external_plugins examples/train/grpo/plugin/plugin.py`进行注册，并通过 reward_funcs 参数进行指定

执行脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/run_external_reward_func.sh)

### 内置奖励函数
swift内置了五种基于规则的奖励函数(代码见swift/plugin/orm.py)

| 奖励函数       | 论文                                                                 |
|----------------|----------------------------------------------------------------------------|
| accuracy       | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) |
| format         | 同上                                                                        |
| cosine         | [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373) |
| repetition     | 同上                                                                        |
| soft_overlong  | [Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476)    |

#### 1. **accuracy**

该函数将模型的生成结果与数据集中的 solution 列进行比较，计算准确率分数。如果生成结果与标准答案一致，则得分为 1.0；否则为 0.0。

注意：该奖励函数使用`math_verify`库解析生成结果和solution中的答案，可能只适用于特定的数学数据集。

#### 2. **format**

论文中使用以下system prompt要求模型按照固定格式进行返回
```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
```

该函数检查模型是否按照 `<think>think content</think><answer>answer content</answer>` 的格式进行生成。如果生成文本符合格式要求，则得分为 1.0；否则为 0.0。

#### 3. **cosine**

论文发现，仅使用 accuracy 奖励函数进行训练会导致模型的生成长度趋于超长，从而影响训练效果。cosine 奖励函数通过控制模型的生成长度来优化训练过程：

- 对于生成正确答案的文本，奖励值随长度增加而递减，鼓励模型生成简洁的回答。
- 对于生成错误答案的文本，奖励值随长度增加而递增，鼓励模型进行更深入的思考。

使用余弦函数平滑地调整奖励值，确保奖励变化在合理范围内。余弦函数的参数包括生成文本的长度、最大长度限制以及奖励的最小值和最大值。

参数
- cosine_min_len_value_wrong（默认值：-0.5）：生成错误答案时，最小长度对应的奖励值。
- cosine_max_len_value_wrong（默认值：0.0）：生成错误答案时，最大长度对应的奖励值。
- cosine_min_len_value_correct（默认值：1.0）：生成正确答案时，最小长度对应的奖励值。
- cosine_max_len_value_correct（默认值：0.5）：生成正确答案时，最大长度对应的奖励值。
- cosine_max_len（默认值等于模型生成的最大程度）：生成文本的最大长度限制。


#### 4. **repetition**

惩罚模型生成文本中的重复内容，通过检测生成文本中的重复 n-gram 模式来评估重复程度，并给予相应的惩罚。

函数将生成文本分割为单词，并提取指定大小的 n-gram（默认为 3-gram）。通过统计不同 n-gram 的数量与总 n-gram 数量的比例，计算重复比例。如果生成文本中重复的 n-gram 比例较高，则给予较大的负奖励（惩罚）。惩罚值通过重复比例和最大惩罚值（默认为 -1.0）计算得出。

参数
- repetition_n_grams（默认值：3）：用于检测重复的 n-gram 大小。
- repetition_max_penalty（默认值：-1.0）：最大惩罚值，用于控制惩罚的强度。

#### 5. **soft overlong punishment**
定义长度惩罚区间。在这个区间内，给予[-1,0]的线性惩罚。

参数
- soft_max_length: 论文中的L_max，模型的最大生成长度，默认等于max_completion_length
- soft_cache_length: 论文中的L_cache，控制长度惩罚区间，区间为[soft_max_length-soft_cache_length, soft_max_length]


论文原文
> a length-aware penalty mechanism designed to shape the reward for truncated samples. Specifically, when the response length exceeds the predefined maximum value, we define a punishment interval. Within this interval, the longer the response, the greater the punishment it receives. This penalty is added to the original rule-based correctness reward, thereby signaling to the model to avoid excessively long responses.

6. **奖励模型**

除了基于规则的奖励函数外，本框架还支持使用奖励模型作为奖励函数。在使用奖励模型时，需要指定 reward_model 参数，该参数与 model 参数类似，用于指定奖励模型的路径或名称。需要注意的是，reward_model 和 reward_funcs 至少需要指定一个。


## 参数与运行脚本
参数
- per_device_train_batch_size: 每个设备训练批量大小，在GRPO中，指 completion 的批次大小。
- per_device_eval_batch_size: 每个设备评估批量大小，在GRPO中，指 completion 的批次大小。
- generation_batch_size: 采样completion批量大小，需要是 num_processes * per_device_train_batch_size 的倍数，默认等于 per_device_batch_size * gradient_accumulation_steps * num_processes
- steps_per_generation: 每轮生成的优化步数，默认等于gradient_accumulation_steps。与generation_batch_size 只能同时设置一个
- num_generations: 每个prompt采样的数量，论文中的G值，需要被 generation_batch_size 或 per_device_batch_size * steps_per_generation * num_processes 整除，默认为8
- max_completion_length: 采样生成的最大长度，默认为512
- ds3_gather_for_generation: 该参数适用于DeepSpeed ZeRO-3。如果启用，策略模型权重将被收集用于生成，从而提高生成速度。然而，禁用此选项允许训练超出单个GPU VRAM的模型，尽管生成速度会变慢。禁用此选项与vLLM生成不兼容。默认为True
- reward_funcs: 奖励函数，根据模型生成结果进行打分，内置accuracy、format、cosine和repetition四个rule-based函数，详细见 swift/plugin/orm.py 文件
- reward_weights: 每个奖励函数的权重。必须与奖励函数和奖励模型的总数量匹配。如果为 None，则所有奖励的权重都相等，为`1.0`
  - 提示：如果GRPO训练中包含`--reward_model`，则其加在奖励函数的最后位置
- reward_model: 同model, 使用奖励模型作为奖励函数，与reward_funcs至少需要指定一个。
- reward_model_plugin: 奖励模型逻辑，默认为orm逻辑, 详细见[自定义奖励模型](#自定义奖励模型)。
- dataset_shuffle: 是否对dataset进行随机操作，默认为True
- loss_type: loss 归一化的类型，可选项为['grpo', 'bnpo', 'dr_grpo'], 默认为'grpo', 具体查看该[pr](https://github.com/huggingface/trl/pull/3256#discussion_r2033213348)
- log_completions: 是否记录训练中的模型生成内容，搭配 `--report_to wandb` 使用。默认为False
  - 提示：若没有设置`--report_to wandb`，则会在checkpoint中创建`completions.jsonl`来存储生成内容
- use_vllm: 是否使用 vLLM 作为 GRPO 生成的 infer_backend，默认为False。
- vllm_mode: vLLM 集成模式，可选项为 `server` 和 `colocate`。server 模式使用 `swift rollout` 拉起的 vLLM 服务器进行采样，colocate 模式在程序内部署 vLLM。
- vllm_mode server 参数
  - vllm_server_base_url: vLLM server的Base URL(比如 http://local_host:8000), 默认为None。设置后，忽略host和port设置。
  - vllm_server_host：vLLM server host地址，默认为None，使用外部vLLM server时使用.
  - vllm_server_port vLLM server 服务端口，默认为8000.
  - vllm_server_timeout 连接vLLM server的超时时间，默认为 240s.
  - async_generate: 异步rollout以提高训练速度，注意开启时采样会使用上一轮更新的模型进行采样，不支持多轮场景。默认`false`.
- vllm_mode colocate 参数
  - vllm_gpu_memory_utilization: vllm透传参数，默认为0.9.
  - vllm_max_model_len: vllm透传参数，默认为None.
  - vllm_enforce_eager: vllm透传参数，默认为False.
  - vllm_limit_mm_per_prompt: vllm透传参数，默认为None.
  - vllm_enable_prefix_caching: vllm透传参数，默认为True.
  - sleep_level: 训练时释放 vLLM 显存，可选项为[0, 1], 默认为0，不释放.
  - offload_optimizer: 是否在vLLM推理时offload optimizer参数，默认为False。
  - offload_model: 是否在vLLM推理时 offload 模型，默认为False。
  - gc_collect_after_offload: 是否在offload结束时进行gc（python gc和GPU gc），默认为False。
  - completion_length_limit_scope: 在多轮对话中，`max_completion_length` 的限制范围。
  `total`限制所有对话轮次的总输出长度不超过`max_completion_length`, `per_round`限制每一轮的输出长度。
  默认为`per_round`, 当前仅对 colocate mode 生效。
- num_iterations: 每个批次代更新次数，默认为1。
- epsilon: clip 系数，默认为0.2。
- epsilon_high: upper clip 系数，默认为None，设置后与epsilon共同构成[epsilon, epsilon_high]裁剪范围。
- delta: [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)中双侧 GRPO 上界裁剪值。若设置，建议大于 1 + epsilon。默认为None。
- sync_ref_model: 是否定期同步ref_model，默认为False。
- ref_model_mixup_alpha: 控制在更新过程中model和先前ref_model之间的混合。更新公式为 $π_{ref} = α * π_θ + (1 - α) * π_{ref_{prev}}$。默认为0.6。
- ref_model_sync_steps：同步频率，默认为512。
- move_model_batches: 在模型向vLLM等快速推理框架移动参数时，将layers分为多少个batch. 默认为None, 代表整个模型不进行拆分，否则拆分为move_model_batches+1(非layer参数)+1(多模态部分参数)个。注意：该参数仅对LoRA(PEFT)训练有意义。
- multi_turn_func: 多轮GRPO参数, 传入对应的plugin名称, 同时在plugin/multi_turn.py中添加好对应的实现。
- dynamic_sample：筛除group内奖励标准差为0的数据，额外采样新数据，默认为False。
- max_resample_times：dynamic_sample设置下限制重采样次数，默认3次。
- overlong_filter：跳过超长截断的样本，不参与loss计算，默认为False。
- padding_free: 去掉所有padding token，并将有效token拼接到一个batch中，仅支持flash_attn.
- sequence_parallel_size: 序列并行段数

奖励函数参数，见[内置奖励函数](#内置奖励函数)

训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo)

## 自定义奖励模型
默认情况下，奖励模型指的是包含数值头的分类模型（通常称为输出奖励模型（ORM））。这些模型对其他模型的输出进行评分，产生一个标量值，表示模型响应的质量。

目前，我们可以利用reward_model_plugin灵活地自定义奖励模型的处理逻辑。这使得实现诸如生成式奖励模型等技术成为可能，包括：
- 自定义模型的系统提示：定义特定的指令和上下文以指导评估过程。
- 处理模型交互历史：管理对话上下文，以提供有意义且具有上下文感知的评估。
- 定义自定义评估标准：设置独特的标准和度量，用于评估模型的响应，超越默认的准确性和相关性衡量标准。

通过reward_model_plugin，开发者可以针对其应用的特定需求定制奖励评估过程。这种灵活性允许更细致和有效的基于奖励的训练策略。

我们在 [rm_plugin.py](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/rm_plugin.py) 中提供了一个简单的生成式奖励模型示例（GenRMPlugin）。

您还可以在 [plugin.py](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/plugin.py) 中自定义您的奖励模型插件，并使用 `external_plugins` 参数进行注册。

以下是一个训练脚本示例，用于使用两个奖励模型，包括一个 ORM 和一个 Gen-RM（此处使用 qwen2.5-3B-Instruct）进行 GRPO 训练：

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs format \
    --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \
    --reward_model_plugin genrm my_rmplugin \
    --reward_weights 0.1 1 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --log_completions true \
    --deepspeed zero2
```

注意：
1. 在 GRPOTrainer 中，reward_model 会依次append到 reward_funcs 中。因此，reward_weights 的顺序对应 [reward_funcs, reward_model]。
2. reward_model_plugin 默认为 default，即使用 ORM 处理逻辑。

## 多任务训练
我们可以在数据集中添加一个用于标识任务类型的列，并在奖励函数/奖励模型插件中根据任务类型进行判断，从而实现多任务训练。假设数据集中包含数学和编程任务，比如：

```
    {"query": "Solve the equation x + 2 = 5", "solution": "3", "task": "math"},
    {"query": "Write a function to calculate the Fibonacci sequence", "solution": "xxx", "task": "code"},
    {"query": "What is the integral of x^2?", "solution": "xxx", "task": "math"},
    {"query": "Implement a sorting algorithm in Python", "solution": "xxx", "task": "code"},
```

下面是针对不同任务的奖励函数的示例：

```python
from swift.plugin import ORM, orms
import random

# Math-specific reward function
class MathRandomReward(ORM):
  def __call__(self, completions, task, **kwargs):
      rewards = []
      for completion, t in zip(completions, task):
          if t == "math":
              import random
              # imple math accuracy logic
              reward = random.random()
              rewards.append(reward)
          else:
              # Return None for non-math tasks
              rewards.append(None)
      return rewards

# Coding-specific reward function
class CodeRandomReward(ORM):
  def __call__(self, completions, task, **kwargs):
      rewards = []
      for prompt, completion, t in zip(prompts, completions, task):
          if t == "code":
              # imple coding accuracy logic
              reward = random.random()
              rewards.append(reward)
          else:
              # Return None for non-coding tasks
              rewards.append(None)
      return rewards

orms['math_reward'] = MathRandomReward
orms['code_reward'] = CodeRandomReward
```
对于非当前任务的数据， 通过返回 None 来处理，从而使得奖励相关仅计算任务内的数据。

## DAPO
[Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476)在GRPO的基础上设置了几种trick，分别是
- Clip Higher
- Dynamic Sampling
- Overlong Filtering
- Token level Loss
- Soft Overlong Punishment

以上trick，我们可以基于GRPOTrainer，设置以下参数实现。

其中Token level Loss是通过使用参数 loss type `bnpo` 实现

| 参数                 | 类型      | 值      |
|----------------------|-----------|-------------|
|`--loss_type`        | `str`      | `bnpo`     |
| `--epsilon_high`     | `float`   | `0.28`      |
| `--dynamic_sample`   | `bool`    | `true`      |
| `--overlong_filter`  | `bool`    | `true`      |
| `--reward_funcs`     | `str`     | `soft_overlong`|
| `--max_resample_times` | `int`    | `3`        |


## FAQ
**1. 训练过程中 loss 等于0 / 接近0 / 小于0**

正常情况， 参考[issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851)

**2. num_generations / 批量大小相关**

在 GRPO 中，batch_size 以 completion（模型生成结果） 为单位。例如，设置 per_device_train_batch_size=8 表示每张 GPU 在训练过程中会同时处理 8 个 completion 的 loss 计算。

训练阶段，在一次完整的梯度累计 batch 中，总的批量大小等于：

```
effective_batch_size = num_processes * per_device_train_batch_size * gradient_accumulation_steps
```

采样阶段，总的批量大小 (completion-level) 数量等于:

1. 设置 generation_batch_size 下，等于 generation_batch_size

2. 设置 steps_per_generation 下， 等于 steps_per_generation * 训练总批量大小

3. 默认等于训练总批量大小(即num_processes * per_device_train_batch_size * gradient_accumulation_steps)

在评估阶段，completion 的数量等于：
```
num_processes * per_device_eval_batch_size
```

参数 `num_generations` 必须能够被以上采样阶段和评估的总批量大小整除，以保证生成任务可以均匀分配到各个设备上。

**示例**

num_processes = 8
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
generation_batch_size = 512
num_generations = 64

1. 采样需要的总数据(prompt)量等于 512 / 64 = 8
2. 每次采样 512 条模型回复
3. 每次更新模型权重批量大小为 8 *4 * 8 = 256


**3. 为什么 KL 出现了NaN**

开启 overlong_filter 后，某一卡上的所有 completion 都被截断

**4. 训练的steps怎么计算?**

参考[issue](https://github.com/modelscope/ms-swift/issues/3912)

**5. clip_ratio为什么总是1?**

Clip机制的核心目的是限制策略更新的幅度，防止因单次更新过大而导致策略性能崩溃（即策略更新后表现急剧下降）。
Clip操作的具体公式如下：

$$
L_{\text{CLIP}}(\theta) = \mathbb{E}_{t} \left[ \min\left(r_{t}(\theta) \hat{A}_{t}, \text{clip}(r_{t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{t} \right) \right]
$$

其中：$r_{t}(\theta) = \frac{\pi_{\theta}(a_{t} \mid s_{t})}{\pi_{\text{old}}(a_{t} \mid s_{t})}$ 是重要性采样比，衡量新旧策略的差异。$\hat{A}_{t}$ 是优势函数（advantage function），表示动作的相对收益。$\epsilon$ 用于限制 $r_{t}(\theta)$ 的偏离范围。

在 on-policy 训练过程中，由于每次更新都使用最新策略生成的数据，新旧策略相同，即 $\pi_{\theta} = \pi_{\text{old}}$

因此重要性采样比恒为 1，此时，clip 操作不会生效。

在设置以下参数情况下，算法为off-policy (near-on-policy)
1. num_iterations > 1
2. steps_per_generation > gradient_accumulation_steps

参考[issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851)

**6. 为什么没有设置val_dataset，仍然有验证过程，如何取消**
当没有显式传入`val_dataset`时，参数`split_dataset_ratio`负责切分部分`dataset`为验证数据集，默认切分1%数据

通过设置`--split_dataset_ratio 0` 来取消验证过程
