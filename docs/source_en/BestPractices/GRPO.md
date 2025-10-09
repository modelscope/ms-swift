# Complete GRPO Experiment Process

This article starts with the relatively simple mathematical task "Countdown Game" and introduces the complete GRPO training process through several steps: dataset definition, reward function definition, and GRPO training. The task definition and training parameters are based on [mini-deepseek-r1](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb).

## Task and Dataset Definition

The goal of the Countdown Game task is to reach a target number using the given numbers and the four basic arithmetic operations. Therefore, we define the dataset as follows:

```python
class CoundownTaskPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        numbers = row['nums']
        target = row.pop('response', None)
        query = f"""
        Using the numbers {numbers}, create an equation that equals {target}.
        You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
        Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags,
        for example <answer> (1 + 2) / 3 * 4 = 4 </answer>.
        """
        row.update({'target': target, 'query': query})
        return super().preprocess(row)

register_dataset(
    DatasetMeta(
        ms_dataset_id='zouxuhong/Countdown-Tasks-3to4',
        subsets=['default'],
        preprocess_func=CoundownTaskPreprocessor(),
        tags=['math']))
```

Through a template, numbers and the target are used to define the task, and a `query` field is provided for model sampling. At the same time, we need to retain the `nums` and `target` fields for subsequent reward function calculation.

## Reward Function Definition

Two reward functions are used for this task: one is the format reward function mentioned in Deepseek-R1, and the other is the accuracy reward function for the Countdown Game. The former is already built into Swift and can be used directly with `--reward_funcs format`, while the latter requires custom definition. Here, we use the `external_plugin` method to define the accuracy reward function, placing the code in `swift/examples/train/grpo/plugin/plugin.py`.

The input to the reward function includes three fields: `completions`, `target`, and `nums`, representing the model-generated text, the target answer, and the available numbers, respectively. Each is a list, supporting simultaneous computation of multiple completions. Note that, except for `completions`, the other parameters are transparently passed from the fields defined in the dataset. If there are changes to the task, adjustments can be made to both the dataset and the reward function as needed.

```python
class CountdownORM(ORM):
    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue
                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception as e:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards
orms['external_countdown'] = CountdownORM
```

## GRPO Training Experiment Record

We first present the GRPO formula:

$$
{\scriptstyle
\begin{aligned}
\mathcal{J}_{G R P O}(\theta) & =\mathbb{E}\left[q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{o l d}}(O \mid q)\right] \\
& \frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\left\{\min \left[\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{o l d}}\left(o_{i, t} \mid q, o_{i,<t}\right)} \hat{A}_{i, t}, \operatorname{clip}\left(\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{o l d}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta| | \pi_{r e f}\right]\right\}
\end{aligned}
}
$$

### Training Parameters

We selected Qwen2.5-3B-Instruct as the base model for training, as using an instruct-tuned model allows for faster acquisition of format rewards. The experiment was conducted on three GPUs, with vLLM inference deployed on the last GPU and two processes set on the remaining GPUs for gradient updates.

Since the task is relatively simple, we set both `max_completion_length` to 1024. For more complex tasks, the model output length can be increased appropriately, but note that **the larger these parameters, the more GPU memory is required, and the slower the training speed**. The training time per step is linearly related to `max_completion_length`.

In our experiment, the total batch size is:

```
num_processes * per_device_train_batch_size * gradient_accumulation_steps = 2 * 8 * 8 = 128
```


Note that the single-GPU batch size is also closely related to GPU memory capacity, so set an appropriate value based on memory limits. Additionally, the total number of steps can be calculated as:

$$
\text{num\_steps} = \text{epochs} \times \text{len(datasets)} \times \text{num\_generations} \div \text{batch\_size}
$$

This formula should guide the planning of learning rate and warmup settings.

Finally, two important parameters are learning rate and $\beta$. The learning rate is straightforward, while $\beta$ is the weight of the KL divergence gradient in the formula. Increasing these parameters accelerates convergence but may lead to instability. After experimentation, we set them to `5e-7` and `0.001`, respectively. During training, adjust these parameters appropriately if instability or oscillations occur.

For KL divergence, the community has extensive discussions, such as [Why GRPO Adheres to KL Divergence](https://zhuanlan.zhihu.com/p/25862547100).

Other parameter settings were not explored in detail and will not be discussed here.

```bash
CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model Qwen/Qwen2.5-3B-Instruct
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_API_KEY=your_wandb_key \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_countdown format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'zouxuhong/Countdown-Tasks-3to4#50000' \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --eval_steps 500 \
    --save_steps 100 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --output_dir output/GRPO_COUNTDOWN \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.001 \
    --num_iterations 1
```

### Experimental Observations

After training for 2000 steps, the following observations were made:

1. **Custom CountdownORM Rewards**: The custom `CountdownORM` reward is steadily increasing, which proves that the model is learning to accomplish the task. The final task success rate is around 50%.
2. **Format Reward**: The format reward remains at a relatively high level, fluctuating between 0.9 and 1.
3. **Reward Standard Deviation**: The reward standard deviation stabilizes around 0.2-0.3 but has not fully converged, indicating that the model still has room to improve its reward.
4. **Failure Cases**: By examining the model's output, many failure cases are due to the model making simple arithmetic errors. Therefore, the model's final convergence direction involves solving the problem step by step. Using a larger model may lead to different convergence directions.
5. **Completion Length**: The completion length initially decreases and then increases, reaching its lowest point around 200 steps and later rising to approximately 350. This suggests the model underwent a change in reasoning approach.
6. **Reasoning Patterns**: The model's reasoning pattern evolves over time. Eventually, the model realizes that performing permutations and step-by-step calculations is more effective for completing this task. The final sampled outputs mostly follow this format. Below are examples of how completions change across different steps:

#### Completion Examples at Different Steps

**Step 1:**

```plaintext
<think>
We need to create an equation using the given numbers [70, 5, 42, 30] exactly once and make it equal to 77.

First, let's consider the possibility of multiplication or division to get closer to 77. Notice that 77 is close to 70 and 42. If we can use these numbers together, it might work.

Multiplying 70 by 1.1 would give us approximately 77, which is very close. So, let's try to make 1.1 using the remaining numbers 5 and 30.
If we take 30/5 = 6, and then 5/6 = 0.8333..., this isn't useful here, so let's look at other possibilities.

Next, let's consider if there's another way to achieve 77 using addition, subtraction, or another combination of our numbers. Given the numbers provided, it seems that multiplication or division will be key.

We can try subtracting 5 from 30 to get 25. Then, if we divide 42 by 25, we don't get a whole number, but we can round it to the nearest whole number, 1. Now, we need to make the equation fit 77. Let's multiply 25 by 3 and add 7 (since 77 = 25*3 + 7). However, we only have one 7 available and we need to find a way to use the remaining numbers to get 7.

Let's try:
70 - 42 = 28
Then, 28 + 5 = 33
Next, 33 + 30 = 63
Finally, 77 - 63 = 14, which doesn't help.

Considering different combinations, one viable way is:
(70 - 42) * (30 / 5) = 28 * 6 = 168 / 2 = 84 / 2 = 42 / 2 = 21 / 3 = 7

So, (70 - 42) * (30 / 5) = 77.
</think>
<answer>(70 - 42) * (30 / 5) = 77</answer>
```

**Step 200:**

```plaintext
<think>
We need to use all the given numbers in a way that the resultant value of the calculation is 99. Given these restrictions, it's clear we should look into how the numbers relate and the mathematical operations that will bring them together to meet our condition.

Firstly, let's evaluate the sums, differences, and multiplication/division among the numbers and their relations to see how this could lead us towards an outcome of 99.
- Summing the three numbers: 96 + 70 + 73 = 239, which is far from 99. This shows addition isn't practical.
- Subtracting numbers: For instance, subtracting the lowest two (i.e., 73 - 70 = 3) and combining with the highest one (96) does not work. Also, there is no other combination of subtraction that would result in the number needing to add up to 99.
- Multiplying or dividing the numbers alone wouldn’t reach any near solution since none of the products or division results of the numbers themselves is very close to 99. We might think more carefully about the arrangement of them using a combination of the operations provided.
Therefore, considering the possible arrangement, a combination like taking (73 - 70 + 96) may produce an effective solution as this will sum the values in such a manner that it aligns with the goal of 99.

So we conclude that the operation: 73 - 70 + 96 gives the right sequence to equal 99.
</think>
<answer>73 - 70 + 96 = 99</answer>
```

**Step 500:**

```plaintext
<think>
Let's begin by exploring various possible combinations of the given numbers using the arithmetic operators we have access to.

- If I add the numbers, I get: 56 + 73 + 51 = 180 which is not equal to our target number (68)
- If I subtract one number from another I will still exceed our target value as well since the maximum difference is: |(difference between max-min):| | (73 - 51) or | (73 - 56), etc.|
- Let's multiply the numbers individually and see if the division operation can bring us closer to the goal:
    - (73 * 51) - 56 = 3723 - 56 = 3667 which isn't the goal either.
    - The only operator left is addition and subtraction; I'll try the combination:
    ((73 - 56) + 51) gives me (17 + 51) = 68.

Thus, our solution fits the provided problem constraints and meets the required outcome.
</think>
<answer>(73 - 56) + 51 = 68</answer>
```

**Step 1000:**

```plaintext
<think>
We need to form an expression using the given numbers (50, 86, and 49) such that it equals 85. Let's try different combinations:

(86 - 50) + 49 = 76 + 49 = 125 (too big)
(86 - 49) + 50 = 37 + 50 = 87 (too big)
(86 - 50) + 49 = 37 + 49 = 86 (too big)
(86 + 50) - 49 = 136 - 49 = 87 (too big)
(86 - 50) + (49 - x)? Not working!
(50 + 86) - 49: this is 136 which isn't our aim.
50 + (86 - 49) = 50 + 37 = 87 which wasn't correct.
50 - 86 + 49 = -16 + 49: This doesn't match up as well.
86 - (50 + 49) = 86 - 99 which doesn't work too.

We try one more:
(86 + 49) - 50 = 135 - 50 = 85!

That's it!
</think>
<answer>(86 + 49) - 50 = 85</answer>
```

**Step 2000:**

```plaintext
<think>
We need to form an expression using the numbers 58, 9, 78, and 44 which evaluates to 73. Let's try different combinations:
(78 - 58) + (9 + 44) = 10 + 53 = 63 (too low)
(78 - 58) + (9 - 44) = 20 - 35 = -15 (too low)
(78 - 58) + (44 - 9) = 20 + 35 = 55 (too low)
(78 + 58) - (9 + 44) = 136 - 53 = 83 (too high)
(78 + 58) - (9 - 44) = 136 + 35 = 171 (too high)
(78 + 58) - (44 + 9) = 136 + 53 = 189 (too high)
(78 + 9) - (58 + 44) = 87 - 102 = -15 (too low)
(78 + 9) - (58 - 44) = 87 - 14 = 73

So our solution is: (78 + 9) - (58 - 44) = 73
</think>
<answer>(78 + 9) - (58 - 44) = 73</answer>
```

---

### Unstable Experiment Records

For learning rate set to `1e-6` and $\beta$ set to `0.04`, the model showed oscillations around step 200. Both the format reward and CountdownORM reward dropped significantly:

[Unstable Experiment Graph](../../resources/grpo_countdown_1.png)
