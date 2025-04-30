# GRPO

论文地址

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

环境安装
```bash
pip install math_verify==0.5.2 # reward function
pip install -U trl
```

**FAQ**
1. 训练过程中 loss 接近0 是正常情况， 参考[issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851)
2. 训练的steps怎么计算? 参考[issue](https://github.com/modelscope/ms-swift/issues/3912)
3. clip_ratio为什么总是1? 参考[issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851)


## 集群支持

![](../../resources/grpo.png)

GRPO 训练框架支持集成高性能推理引擎（如 vLLM）来加速采样过程，提供以下两种部署模式：

### 1. 内部集成模式 (Internal)

- 在Trainer内部直接启动推理服务
- 提供两种资源分配策略：
  - **协同模式 (Colocate)**: 训练与推理共享GPU资源
  - **异步模式 (Async)**: 训练与推理使用独立GPU资源

### GRPO训练资源配置方案
| 配置场景                 | NPROC_PER_NODE | num_infer_workers | 资源分配说明             |
|--------------------------|----------------|------------------|------------------------|
| **Colocate**   | =总GPU数      | =总GPU数          | 训练和推理共享全部GPU资源              |
| **Async**      | =训练卡数      | =推理卡数         | 必须满足：训练卡数 + 推理卡数 = 总GPU数 |

**注：**
1. 在Colocate模式下推荐设置`sleep_level=1`, 在模型训练时释放vLLM占用显存
2. 总GPU数指可见的GPU设备总数

### 2. 外部服务模式 (External)
连接外部的 vLLM 推理服务器
使用时，使用以下参数配置外部 vLLM 服务器
```bash
--vllm_server_host <服务器IP> \
--vllm_server_port <服务端口> \
--vllm_server_timeout <超时时间> \
```
使用`swift rollout`命令部署vLLM 服务器, 现仅支持vLLM backend
```bash
CUDA_VISIBLE_DEVICES=2 \
swift rollout \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor_parallel_size 2 \
```
完整脚本可以参考[这里](../../../examples/train/grpo/multi_node/Qwen2_5_32B_full.sh)


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

执行脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/run_external_rm.sh)

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
- num_generations: 每个prompt采样的数量，论文中的G值，需要被 per_device_batch_size * gradient_accumulation_steps * nproc_per_node 整除，默认为8
- max_completion_length: 采样生成的最大长度，默认为512
- ds3_gather_for_generation: 该参数适用于DeepSpeed ZeRO-3。如果启用，策略模型权重将被收集用于生成，从而提高生成速度。然而，禁用此选项允许训练超出单个GPU VRAM的模型，尽管生成速度会变慢。禁用此选项与vLLM生成不兼容。默认为True
- reward_funcs: 奖励函数，根据模型生成结果进行打分，内置accuracy、format、cosine和repetition四个rule-based函数，详细见 swift/plugin/orm.py 文件
- reward_weights: 每个奖励函数的权重。必须与奖励函数的数量匹配。如果为 None，则所有奖励的权重都相等，为`1.0`
  - 提示：如果GRPO训练中包含`--reward_model`，则其加在奖励函数的最后位置
- dataset_shuffle: 是否对dataset进行随机操作，默认为True
- loss_type: loss 归一化的类型，可选项为['grpo', 'bnpo', 'dr_grpo'], 默认为'grpo', 具体查看该[pr](https://github.com/huggingface/trl/pull/3256#discussion_r2033213348)
- log_completions: 是否记录训练中的模型生成内容，搭配 `--report_to wandb` 使用。默认为False
  - 提示：若没有设置`--report_to wandb`，则会在checkpoint中创建`completions.jsonl`来存储生成内容
- use_vllm: 是否使用vLLM作为采样的生成后端，默认为False，建议使用加快训练速度
- vllm_device: 设置vLLM部署的设备，默认为`auto`, 即未被使用的第一张显卡，使用`cuda:x`来设置特定的卡。
- vllm_gpu_memory_utilization: vllm透传参数，默认为0.9
- vllm_max_model_len: vllm透传参数，默认为None
- vllm_max_num_seqs: vllm透传参数，默认为256
- vllm_enforce_eager: vllm透传参数，默认为False
- vllm_limit_mm_per_prompt: vllm透传参数，默认为None
- vllm_enable_prefix_caching: vllm透传参数，默认为True
- vllm_server_host：vLLM server host地址，默认为None，使用外部vLLM server时使用
- vllm_server_port vLLM server 服务端口，默认为8000
- vllm_server_timeout 连接vLLM server的超时时间，默认为120s
- reward_model: 同model, 使用奖励模型作为奖励函数，与reward_funcs至少需要指定一个
- num_iterations: 每个批次代更新次数，默认为1.
- epsilon: clip 系数，默认为0.2.
- epsilon_high: upper clip 系数，默认为None，设置后与epsilon共同构成[epsilon, epsilon_high]裁剪范围.
- async_generate: 异步rollout以提高训练速度，默认`false`.
- sleep_level: vllm特有参数，在训练和rollout复用卡的时候，可以选择vllm进行offload.
- move_model_batches: 在模型向vLLM/LMDeploy等快速推理框架移动参数时，将layers分为多少个batch. 默认为None, 代表整个模型不进行拆分，否则拆分为move_model_batches+1(非layer参数)+1(多模态部分参数)个
- offload_optimizer: 是否在vLLM/LMDeploy推理时offload optimizer参数，默认为False
- offload_model: 是否在vLLM/LMDeploy推理时offload 模型本身，默认为False
  - 注意：若该参数设置为True，训练时grad_norm一直为0，请安装`vllm==0.7.3`
- gc_collect_after_offload: 是否在offload结束时进行gc（python gc和GPU gc），默认为False
- multi_turn_func: 多轮GRPO参数, 传入对应的plugin名称, 同时在plugin/multi_turn.py中添加好对应的实现
- dynamic_sample：筛除group内奖励标准差为0的数据，额外采样新数据，默认为False。
- max_resample_times：dynamic_sample设置下限制重采样次数，默认3次。
- overlong_filter：跳过超长截断的样本，不参与loss计算，默认为False。
- vllm_server_host：vLLM server host地址，默认为None，使用外部vLLM server时使用 \
- vllm_server_port vLLM server 服务端口，默认为8000 \
- vllm_server_timeout 连接vLLM server的超时时间，默认为120s \


奖励函数参数，见[内置奖励函数](#内置奖励函数)

可以使用vLLM、LMDeploy作为采样后端加速训练
多卡vLLM
```bash
# async mode
# 要求 num_infer_workers(部署) + NPROC_PER_NODE(训练) = device_count
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=7 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --num_infer_workers 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 7 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true

# colocate mode
# 要求 num_infer_workers(部署) = NPROC_PER_NODE(训练) = device_count
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-1.5B \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 8192 \
    --num_infer_workers 8 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --log_completions true
```


单卡
```bash
# PT backend
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --reward_funcs accuracy format \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true

# vLLM backend
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --vllm_gpu_memory_utilization 0.5 \
    --use_vllm true \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --reward_funcs accuracy format \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true
```
多机训练参考[这里](../../../examples/train/grpo/multi_node/)

注：内部集成模式下，需要不同节点的GPU配置以及训练参数相同




## DAPO
[Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476)在GRPO的基础上设置了几种trick，分别是
- Clip Higher
- Dynamic Sampling
- Overlong Filtering
- Token level Loss
- Soft Overlong Punishment

其中Token level Loss是默认实现，不用额外设置。对于其余trick，我们可以基于GRPOTrainer，设置以下参数实现。

| 参数                 | 类型      | 值      |
|----------------------|-----------|-------------|
| `--epsilon_high`     | `float`   | `0.28`      |
| `--dynamic_sample`   | `bool`    | `true`      |
| `--overlong_filter`  | `bool`    | `true`      |
| `--reward_funcs`     | `str`     | `soft_overlong`|
| `--max_resample_times` | `int`    | `3`        |

参考训练脚本(八卡colocate mode)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
WANDB_API_KEY=xxx \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-1.5B \
    --reward_funcs accuracy soft_overlong \
    --max_completion_length 4096 \
    --soft_cache_length 819 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --overlong_filter true \
    --max_resample_times 3 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.6 \
    --num_infer_workers 8 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --report_to tensorboard wandb \
    --beta 0.0 \
```
