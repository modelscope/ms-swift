# Complete Multimodal GRPO Experiment Workflow

This document explains how to use SWIFT GRPO for training multimodal models and tasks. The goal is to train on multiple multimodal tasks to improve task accuracy. Task definitions, training parameters, etc., refer to [R1-V](https://github.com/Deep-Agent/R1-V.git) and [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal.git).

---

## **ClevrCount Task**

### **Task and Dataset Definition**

This task is based on the `clevr_cogen_a_train` dataset. The model's goal is to output the number of objects in the image. Therefore, we define the dataset as follows:

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
        ms_dataset_id='AI-ModelScope/clevr_cogen_a_train',
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

The purpose of redefining the dataset preprocessor here is to modify the query. A sample dataset entry is as follows, including `messages`, `images`, and `solution` fields. The `solution` is used in the reward function, while `messages` and `images` serve as model input.
- Note: `{'role': 'assistant', 'content': '<answer> 3 </answer>'}` will be removed in GRPOTrainer and can be ignored. The 'solution' field will be passed directly into the ORM. When creating a custom dataset, the 'images' field should be organized as `["image_path1", "image_path2"]`.

```json
{
    "images": ["image_path1", "image_path2"],
    "messages": [
        {
            "role": "user",
            "content": "How many items are there in the image? Output the thinking process in <think> </think> and\n final answer (number) in <answer> </answer> tags."
        }
    ],
    "solution": "<answer> 3 </answer>"
}

```

---

## **Reward Function Definition**

This task uses two reward functions: one is the format reward function mentioned in `Deepseek-R1`, and the other is the accuracy reward function for ClevrCount. The former is built into SWIFT and can be used directly with `--reward_funcs format`. The latter needs to be custom-defined. Here, we use the `external_plugin` method to define the accuracy reward function by placing the code in `swift/examples/train/grpo/plugin/plugin.py`.

The reward function's input includes `completions` and `solution` fields, representing the model-generated text and ground truth, respectively. Each is a list, allowing the computation of multiple completions simultaneously. Note that the `solution` field is passed through directly from the dataset definition. If there are task changes, corresponding modifications can be made to the dataset and reward function.

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

---

### **GRPO Training Experiment Log**

#### **Training Parameters**

We selected `Qwen2.5-VL-3B-Instruct` as the base model for training. The main reason for choosing the `Instruct` model over the base model is to rapidly achieve format rewards. Experiments were conducted on 8 GPUs. SWIFT GRPO training supports multi-GPU deployment to accelerate rollouts. If you encounter deployment errors for `qwen2.5-vl` on `vllm`, refer to [this issue](https://github.com/vllm-project/vllm/issues/13285).

Since the task is simple, we set `max_completion_length` to 1024 and selected `external_r1v_acc` and `format` as reward functions. The learning rate and beta are set to `1e-6` and `0.001`, respectively. Other configurations are as follows. The settings for `batch_size` and `num_generations` can be referenced from [GRPO Full Workflow](./GRPO.md).

launch external vLLM server using following script
```bash
CUDA_VISIBLE_DEVICES=6,7 \
swift rollout \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --vllm_data_parallel_size 2
```

```shell
WANDB_API_KEY=your_wandb_api_key \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/clevr_cogen_a_train' \
    --load_from_cache_file true \
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
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
```

#### **Experimental Observations**

[image.png](../../resources/grpo_clevr_count.png)

- Given the simplicity of the dataset and task, the model converged after 500 epochs. Key observations:
  1. The custom `ClevrORM` reward steadily increased, proving the model learned how to complete the task. The task success rate climbed from an initial 0.4 to nearly 1.
  2. The `Format Reward` remained stable at 1, likely due to the consistent query format across all dataset samples.
  3. The `reward_std` stabilized below 0.1.
  4. The `completion length` eventually stabilized between 60-80 tokens, with the model learning a fixed output pattern for item-by-item counting.

---
For additional tasks like Geometric QA and Open R1 Multimodal datasets, refer to their respective sections in the full experiment documentation.

## **Geometric QA Task**

### **Task and Dataset Definition**

This task is a Geometric QA task, where the task description is: given a geometric figure, answer mathematical questions related to the figure. The original data comes from [this paper](https://arxiv.org/pdf/2312.11370), and [R1-V](https://github.com/Deep-Agent/R1-V.git) has preprocessed the data into a `problem-solution` format while retaining the images in the `image` field. Therefore, we do not need to redefine the dataset and can directly use `--dataset AI-ModelScope/GEOQA_R1V_Train_8K`.

---

### **Reward Function**

As this is also a mathematical problem, and the answers are already processed into final results, we directly use the previously defined `MultiModalAccuracyORM` reward function.

---

### **GRPO Training Experiment Log**

#### **Training Parameters**

The selected model and most hyperparameters are similar to the previous experiment, with two main differences:
1. **SWIFT now supports the `--num_iteration` parameter**, allowing multiple updates during a single rollout. We set it to 2.
2. During the experiment, we found that training might become unstable in mathematical problems, causing the model to collapse. This is characterized by a sharp drop in all rewards, a rapid increase in loss, `grad_norm`, and KL divergence, with no subsequent recovery. To prevent this, we set `--max_grad_norm 0.5` to ensure stable training. Note that this instability can have some randomness.

```shell
WANDB_API_KEY=your_wandb_api_key \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MAX_PIXELS=401408 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/GEOQA_R1V_Train_8K' \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 400 \
    --save_steps 400 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GRPO_GEOQA \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 2 \
    --async_generate false \
    --beta 0.001 \
    --max_grad_norm 0.5 \
```

---

#### **Experimental Observations**

[image.png](../../resources/grpo_geoqa.png)

The training curve is shown above.

1. Compared to the **Count task**, the **Geometric QA task** is significantly more challenging. As a result, the **Accuracy Reward** converges much more slowly and has larger oscillations.
2. The `completion_length` exhibits a clear trend, eventually stabilizing at around 250 tokens.
3. The reward converges, with the standard deviation stabilizing at around 0.2 after a total of 1300 steps.

Below is an example of the model's completion after training:

```plaintext
User:
"If AB is the tangent to circle O with the tangent point A, and BO intersects circle O at point C, and point D is located on circle O, what is the measure of angle ADC if the degree of angle ABO is 32.0?"
Assistant:
"<think>\nTo find the measure of angle ADC, we first need to understand the properties of the tangent and the circle. Since AB is a tangent to the circle at point A, the line segment OA (which is perpendicular to the tangent AB) is perpendicular to AB. This means that angle OAB is 90 degrees.\n\nGiven that angle ABO is 32 degrees, we can find angle BAO by subtracting angle ABO from 90 degrees:\n∠BAO = 90° - ∠ABO = 90° - 32° = 58°\n\nSince angle BAO is an angle in the minor arc AD, the angle ADC, which is an angle in the minor arc AD, is half the angle BAO. Therefore, angle ADC is:\n∠ADC = 1/2 × ∠BAO = 1/2 × 58° = 29°\n\nSo, the measure of angle ADC is 29 degrees.\n</think>\n<answer>\nThe measure of angle ADC is 29 degrees.\n</answer>"
```

---

## **Multimodal Open R1 Dataset Experiment**

### **Task and Dataset Definition**

This task is based on the experiments in [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal.git), using the dataset: [lmms-lab/multimodal-open-r1-8k-verified](https://www.modelscope.cn/datasets/lmms-lab/multimodal-open-r1-8k-verified). This dataset focuses on multimodal mathematical reasoning tasks, with data generated by GPT4o based on the `Math360K` and `Geo170K` datasets. It includes reasoning paths and verifiable answers. The dataset already contains `image`, `problem`, and `solution` fields, so no additional prompt modifications are required, and there is no need to redefine the dataset.

---

### **Reward Function**

We directly use the previously defined `MultiModalAccuracyORM` reward function.

---

### **GRPO Training Experiment Log**

#### **Training Parameters**

The selected model and most hyperparameters are similar to the previous experiment. Due to an **OOM (Out of Memory) issue**, we set `MAX_PIXELS=262144` to reduce memory usage.

```shell
WANDB_API_KEY=your_wandb_api_key \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MAX_PIXELS=262144 \
MASTER_PORT=29600 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'lmms-lab/multimodal-open-r1-8k-verified' \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 400 \
    --save_steps 400 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GRPO_GEOQA \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 2 \
    --async_generate false \
    --beta 0.001 \
    --max_grad_norm 0.5 \
```

---

#### **Experimental Observations**

[image.png](../../resources/grpo_openr1_multimodal.png)

The training curve is shown above.

- With a total of 8k data samples, the training reached **1268 steps**, with the **Accuracy Reward** converging around 0.5.
- The `completion_length` stabilized around 200 tokens.
- The `reward_std` eventually oscillated around 0.2.

Below is an example of the model's completion after training:

```plaintext
User:
"Based on the map, which state falls into the lowest percentage range of lots, and what is that percentage range?"
Assistant:
"<think>\nThe image provided shows a color-coded map of the USA indicating different ranges of lot percentages across various states. According to the legend at the bottom, the lowest percentage range (6.0% - 6.5%) is represented by white. In the image, Alabama (AL) is shaded in white, which corresponds to the 6.0% - 6.5% category. Therefore, based on the map, the state that falls into the lowest percentage range of lots is Alabama, with the percentage range of 6.0% - 6.5%.\nTherefore, the answer is 6.0% - 6.5%.\n</think>\n<answer>Alabama</answer>"
```
