# GRPO

论文地址

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

环境安装
```bash
pip install math_verify # reward function
pip install git+https://github.com/huggingface/trl.git"
```

**注意**：训练过程中 loss 接近0 是正常情况， 参考[issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851)

## 奖励函数
### 自定义奖励函数
奖励函数接受模型生成的文本 completions 以及其他数据集中的列作为参数，并对模型生成的文本进行打分。以下是一个示例，展示了如何实现一个简单的长度奖励函数。该函数会在模型生成的文本长度超过 1024 时，给予 1.0 的奖励信号；否则，奖励信号为 0.0。

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
swift内置了四种基于规则的奖励函数，分别是 accuracy、format、 cosine 和 repetition。(代码见swift/plugin/orm.py)

其中 accuracy 和 format 奖励函数源于论文[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), cosine 和 repetition 奖励函数源于论文[Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373)

1. **accuracy**

该函数将模型的生成结果与数据集中的 solution 列进行比较，计算准确率分数。如果生成结果与标准答案一致，则得分为 1.0；否则为 0.0。

2. **format**

论文中使用以下system prompt要求模型按照固定格式进行返回
```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
```

该函数检查模型是否按照 `<think>think content</think><answer>answer content</answer>` 的格式进行生成。如果生成文本符合格式要求，则得分为 1.0；否则为 0.0。

3. **cosine**

论文发现，仅使用 accuracy 奖励函数进行训练会导致模型的生成长度趋于超长，从而影响训练效果。cosine 奖励函数通过控制模型的生成长度来优化训练过程：

- 对于生成正确答案的文本，奖励值随长度增加而递减，鼓励模型生成简洁的回答。
- 对于生成错误答案的文本，奖励值随长度增加而递增，鼓励模型进行更深入的思考。

使用余弦函数平滑地调整奖励值，确保奖励变化在合理范围内。余弦函数的参数包括生成文本的长度、最大长度限制以及奖励的最小值和最大值。

参数
- cosine_min_len_value_wrong（默认值：0.0）：生成错误答案时，最小长度对应的奖励值。
- cosine_max_len_value_wrong（默认值：-0.5）：生成错误答案时，最大长度对应的奖励值。
- cosine_min_len_value_correct（默认值：1.0）：生成正确答案时，最小长度对应的奖励值。
- cosine_max_len_value_correct（默认值：0.5）：生成正确答案时，最大长度对应的奖励值。
- cosine_max_len（默认值等于模型生成的最大程度）：生成文本的最大长度限制。


4. **repetition**

惩罚模型生成文本中的重复内容，通过检测生成文本中的重复 n-gram 模式来评估重复程度，并给予相应的惩罚。

函数将生成文本分割为单词，并提取指定大小的 n-gram（默认为 3-gram）。通过统计不同 n-gram 的数量与总 n-gram 数量的比例，计算重复比例。如果生成文本中重复的 n-gram 比例较高，则给予较大的负奖励（惩罚）。惩罚值通过重复比例和最大惩罚值（默认为 -1.0）计算得出。

参数
- repetition_n_grams（默认值：3）：用于检测重复的 n-gram 大小。
- repetition_max_penalty（默认值：-1.0）：最大惩罚值，用于控制惩罚的强度。


5. **奖励模型**

除了基于规则的奖励函数外，本框架还支持使用奖励模型作为奖励函数。在使用奖励模型时，需要指定 reward_model 参数，该参数与 model 参数类似，用于指定奖励模型的路径或名称。需要注意的是，reward_model 和 reward_funcs 至少需要指定一个。


## 运行脚本
超参数
- num_generations: 每个prompt采样的数量，论文中的G值，需要被 per_device_eval_batch_size * nproc_per_node 整除
- max_completion_length: 采样生成的最大长度，默认为512
- ds3_gather_for_generation: 该参数适用于DeepSpeed ZeRO-3。如果启用，策略模型权重将被收集用于生成，从而提高生成速度。然而，禁用此选项允许训练超出单个GPU VRAM的模型，尽管生成速度会变慢。禁用此选项与vLLM生成不兼容。默认为True
- reward_funcs: 奖励函数，根据模型生成结果进行打分，内置accuracy、format、cosine和repetition四个rule-based函数，详细见 swift/plugin/orm.py 文件
- reward_weights: 每个奖励函数的权重。必须与奖励函数的数量匹配。如果为 None，则所有奖励的权重都相等，为`1.0`
  - 提示：如果GRPO训练中包含`--reward_model`，则其加在奖励函数的最后位置
- log_completions: 是否记录训练中的模型生成内容，搭配 `--report_to wandb` 使用。默认为False
  - 提示：若没有设置`--report_to wandb`，则会在checkpoint中创建`completions.jsonl`来存储生成内容
- use_vllm: 是否使用vLLM作为采样的生成后端，默认为False，建议使用加快训练速度
- vllm_device: 设置vLLM部署的设备，默认为`auto`, 即未被使用的第一张显卡，使用`cuda:x`来设置特定的卡。
- vllm_gpu_memory_utilization: vLLM透传参数
- vllm_max_model_len: vLLM透传参数
- reward_model: 同model, 使用奖励模型作为奖励函数，与reward_funcs至少需要指定一个
- num_iterations: 每个批次代更新次数，默认为1.
- epsilon: clip 系数

奖励函数超参，见[内置奖励函数](#内置奖励函数)

建议使用vLLM作为采样后端加速训练，多卡环境下，建议单独设置一张显卡用于部署vLLM，此时进程数应等于显卡数减一

多卡vLLM
```bash
# nproc_per_node 比显卡数少一，vLLM默认单独部署于最后一张卡，即卡7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=7 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
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
```

单卡
```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
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
