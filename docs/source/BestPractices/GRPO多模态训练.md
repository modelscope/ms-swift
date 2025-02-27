# GRPO完整实验流程
本文介绍如何使用swift GRPO进行多模态模型和任务的训练。目标是对多个多模态任务进行训练，提升任务精度，任务定义和训练参数等参考了 [R1-V](https://github.com/Deep-Agent/R1-V.git) 和 [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal.git)



## ClevrCount 任务
### 任务与数据集定义
本任务从clevr_cogen_a_train数据集出发，模型的目标是输出图像中包含的物体数量，因此，我们定义数据集如下：

```python
class ClevrPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row.get('query', '')
        query = f"""{query} Output the thinking process in <think> </think> and
 final answer (number) in <answer> </answer> tags."""
        row.update({'query': query})
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='okwinds/clevr_cogen_a_train',
        subsets=[
            SubsetDataset(
                name='default',
                subset='default',
                split=['train'],
            ),
        ],
        preprocess_func=ClevrPreprocessor(),
        tags=['qa', 'math']))

```
这里重新定义dataset preprocessor的目的是修改query。数据集示例样本如下，包含messages,images和solution字段，solution会送入后续的奖励函数中，而messages和images则会作为模型输入。
```json
{
    'images': [{'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xe0\x00\x00\x01@\x08\x06\x00\x00\x00d\xc8\xafB`\x82 ...', 'path': 'CLEVR_trainA_000000.png'}],
    'messages': [{'role': 'user', 'content': 'How many items are there in the image? Output the thinking process in <think> </think> and\n final answer (number) in <answer> </answer> tags.'}, {'role': 'assistant', 'content': '<answer> 3 </answer>'}],
    'solution': '<answer> 3 </answer>'
}
```


## 奖励函数定义：
本任务使用的奖励函数有两个，一个是 Deepseek-R1 中提到的格式奖励函数，另一是 ClevrCount 的准确性奖励函数。前者已经在swift中内置，通过 `--reward_funcs format` 可以直接使用，而后者需要我们自己定义，在这里我们使用 external_plugin 的方式定义准确性奖励函数，将代码放在`swift/examples/train/grpo/plugin/plugin.py`中。

在这里，奖励函数的输入包括 completions和solution两个字段，分别表示模型生成的文本和真值。每个都是list，支持多个 completion 同时计算。注意，在这里，solution字段是数据集中定义的字段透传而来，如果有任务上的变动，可以分别对数据集和奖励函数做对应的改变即可。
```python

class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards
orms['external_r1v_acc'] = MultiModalAccuracyORM
```

## GRPO训练实验记录
### 训练参数：
我们选取 Qwen2.5-VL-3B-Instruct 作为基础模型进行训练，选取 Instruct 而不是基模的主要原因是可以更快地获取 format reward。我们在八卡 GPU 上进行实验。swift GRPO训练已支持多卡部署模型以加速rollout，因此我们设置num_infer_workers为2，进程数为6，即2卡部署，6卡训练。如果遇到vllm部署qwen2.5-vl报错，可以参考[issue](https://github.com/vllm-project/vllm/issues/13285)

由于任务简单，我们设置max_completion_length为1024，奖励函数选择external_r1v_acc和format，学习率和beta分别设置为1e-6和0.001。其他设置如下所示，batch_size和num_generations的设置原则可以参考[GRPO完整流程](./GRPO完整流程.md)。

```bash
WANDB_API_KEY=your_wandb_api_key \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.6 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'okwinds/clevr_cogen_a_train' \
    --max_length 8192 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GRPO_CLEVR_COUNTDOWN \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 24 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt_1.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --vllm_max_model_len 1024 \
    --report_to wandb \
    --num_iterations 1 \
    --num_infer_workers 2 \
    --async_generate false \
    --beta 0.001 \

```
### 实验现象
![image.png](../../resources/grpo_clevr_count.png)
由于数据集和任务比较简单，模型训练了500个epoch已经基本收敛，有以下观察。
1. 自定义的ClevrORM在不断增加，证明模型学习到了如何完成这一任务，最终任务成功率从出事的0.4攀升到1左右。
2. 由于是Instruct模型，Format Reward基本稳定在1，这可能也与数据集所有样本都是一样的query有一定关系。
3. reward_std 稳定到0.1以下。
4. completion length最终稳定在60-80，模型学习到的固定输出范式是逐个物体列举数数。

step 1:
```
"<think> The image shows 5 distinct items: a cyan metallic cylinder, a red metallic cylinder, a small purple metallic cylinder, a large gray metallic sphere, and a small yellow metallic cube. To count the number of items, I add the counts of each type of object: 1 cyan cylinder, 1 red cylinder, 1 purple cylinder, 1 gray sphere, and 1 yellow cube. Therefore, the total number of items is 5. </think>\n<answer>5</answer>"
"Let's count the number of items in the image:\n1. A large cyan metallic cylinder.\n2. A small purple metallic cylinder.\n3. A large red metallic cylinder.\n4. A large gray metallic sphere.\n5. A small yellow metallic cube.\n\n<answer>5</answer>"
"<think> The image shows four distinct objects: a large cyan metallic cylinder, a smaller purple metallic cylinder, a large red metallic cylinder, and a small yellow metallic cube. Counting each of these items individually, there are five items in total. </think>\n<answer>5</answer>"
```
step 400:
```
"<think> Counting the number of items in the image:\n1. Green matte sphere\n2. Large metallic yellow sphere\n3. Small metallic brown cube\n\nThere are three distinct objects in total.\n</think>\n<answer>3</answer>"
"<think> Counting the objects in the image:\n1. Large yellow metallic sphere\n2. Small green rubber sphere\n3. Small brown metallic cube\n\nThere are three distinct items in the image.\n</think>\n<answer>3</answer>"
```
总体来讲，这一任务比较简单，reward的收敛也比较典型。
