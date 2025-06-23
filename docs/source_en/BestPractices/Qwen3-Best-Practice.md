# Qwen3 Best Practices
Discussion: [issue 4030](https://github.com/modelscope/ms-swift/issues/4030)

Qwen Documentation: [https://qwen.readthedocs.io/en/latest/training/ms_swift.html](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)

## Inference

Thinking mode:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-8B \
    --infer_backend vllm \
    --stream true \
    --max_new_tokens 2048 \
    --max_model_len 8192
```

```text
<<< who are you?
<think>
Okay, the user is asking "who are you?" Let me start by introducing myself as Qwen, the large language model developed by Alibaba Cloud. I should mention my capabilities, like answering questions, creating content, and engaging in conversations. But I need to keep it concise. Also, the user might want to know how I can assist them. Maybe I should ask how I can help them today. Let me check if there's anything else important to include. Oh, I should make sure the tone is friendly and approachable. Alright, that should cover it.
</think>

Hello! I am Qwen, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, such as answering questions, creating content, writing stories, coding, and more. How can I help you today? 😊
<<< clear
<<< who are you? /no_think
<think>

</think>

I am Qwen, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, including answering questions, creating content, and providing information. How can I help you today?
```

Non-thinking mode:

- `--response_prefix` indicates that the model's output will continue after the prefix. It is equivalent to setting enable_thinking to False.

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-8B \
    --infer_backend vllm \
    --stream true \
    --max_new_tokens 2048 \
    --max_model_len 8192 \
    --response_prefix '<think>\n\n</think>\n\n'
```

```text
<<< who are you?
<think>

</think>

I am Qwen, a large-scale language model developed by Alibaba Cloud. I am designed to assist with a wide range of tasks, including answering questions, creating content, and providing information. How can I assist you today?
```

## Training

Before starting training, please ensure that your environment is properly configured.

```bash
pip install ms-swift -U
pip install transformers -U

pip install deepspeed # for multi-GPU training
pip install liger-kernel # to save GPU memory resources
pip install flash-attn --no-build-isolation  # required for packing
```

## Supervised Fine-Tuning (SFT)

### Data Preparation

When using ms-swift for SFT, the custom dataset format is as follows (the `system` field is optional). You can organize it in JSON, JSONL, or CSV format. Specify `--dataset <dataset_path>` in the training script. For a complete guide on dataset formats, refer to the [Custom Dataset Documentation](../Customization/Custom-dataset.md).

```text
# General format
{"messages": [
    {"role": "system", "content": "<system-prompt>"},
    {"role": "user", "content": "<query1>"},
    {"role": "assistant", "content": "<response1>"}
]}
# Format with thinking process
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang?"},
    {"role": "assistant", "content": "Thought: ...\n\nAnswer:\nThe capital of Zhejiang is Hangzhou."}
]}
```

If you want to train using data without the thinking chain while preserving the model's reasoning ability, you can use one of the following methods to minimize the impact of fine-tuning:

**Option 1**: [Recommended] During training, specify `--loss_scale ignore_empty_think`, which will ignore the loss calculation for `Thought:` and `Answer:` tokens, thus avoiding the loss of reasoning capability. The training script can be found [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh). This method also works for models like DeepSeek-R1. The custom dataset format is as follows:

```json
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang?"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

**Option 2**: Add `/no_think` to the query in the dataset to avoid losing the reasoning capability. The training script can be found [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo2.sh). The custom dataset format is as follows:

```json
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang? /no_think"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

You can use the following command to obtain a distilled reasoning dataset. During training, you can mix it with datasets that do not contain chain-of-thought (CoT) data to further mitigate the loss of reasoning ability:

- The choice of `--val_dataset` is arbitrary. The reasoning results saved to `result_path` can be specified directly in training via `--dataset distill_dataset.jsonl`.
- This approach is also applicable to other reasoning models, such as deepseek-r1.

```shell
# 4 * 80GiB
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model Qwen/Qwen3-32B \
    --infer_backend vllm \
    --val_dataset 'AI-ModelScope/alpaca-gpt4-data-en#5000' 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 2 \
    --max_model_len 8192 \
    --max_new_tokens 4096 \
    --write_batch_size 1000 \
    --result_path distill_dataset.jsonl
```

### 30-Minute Self-Awareness Fine-Tuning

This section demonstrates how to perform self-awareness fine-tuning on Qwen3-8B within 30 minutes. A GPU with at least 22GB of VRAM is required and can run on the free computing resources provided by ModelScope, such as the A10 instance.

After training, the model will no longer identify itself as a "Qwen" trained by "Tongyi Lab," but rather as a "swift-robot" trained by "swift."

If you need to train in an offline environment, you can manually download the model and dataset, and specify `--model <model-path>` and `--dataset <dataset-dir>`. The dataset is available on the [ModelScope Hub](https://modelscope.cn/datasets/swift/self-cognition). You can view the preprocessing function for the `swift/self-cognition` dataset [here](https://github.com/modelscope/ms-swift/blob/36fdf381e5e88cb8a71c9d69c1d8936a989318cc/swift/llm/dataset/dataset/llm.py#L882).

For explanations of the parameters used in the training script, please refer to the [Command Line Arguments Documentation](../Instruction/Command-line-parameters.md).

```bash
# GPU Memory Usage: 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' \
              'swift/self-cognition:qwen3#600' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

After fine-tuning, you can test the results using the following script. Note that the `--adapters` part should be modified to point to the final saved checkpoint directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

```text
<<< who are you?
<think>
Okay, the user asked, "who are you?" I need to introduce myself. Let me start by stating my name, swift-robot. Then, I should mention that I'm an AI assistant developed by swift. I should explain my purpose, which is to provide information and assistance. I should also highlight my capabilities, like answering questions, generating text, and engaging in conversation. It's important to keep the tone friendly and approachable. Maybe add something about being here to help and encourage the user to ask anything. Let me check if I covered all the key points: name, developer, purpose, capabilities, and a welcoming statement. Yeah, that should do it. Now, let me put that into a concise and friendly response.
</think>

Hello! I am swift-robot, an artificial intelligence assistant developed by swift. My purpose is to provide information and assistance to users like you. I can answer questions, generate text, and engage in conversations on a wide range of topics. I am here to help, so feel free to ask me anything you need!
```

By default, ms-swift uses the ModelScope community to download models and datasets. If you want to use the HuggingFace community instead, you need to additionally specify `--use_hf true`.

Merge LoRA weights:
```shell
swift export \
    --adapters output/checkpoint-xxx \
    --merge_lora true
```

Push the model to ModelScope/HuggingFace:

```shell
# If pushing full weights, change `--adapters` to `--model`.
# You can find your ModelScope hub_token here: https://modelscope.cn/my/myaccesstoken
swift export \
    --adapters output/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<hub-model-id>' \
    --hub_token '<hub-token>' \
    --use_hf false
```

If you want to perform training on multiple GPUs, the following example provides a multi-GPU training setup:

```shell
# 4 * 60GB
# You can run the experiment by setting `--dataset AI-ModelScope/alpaca-gpt4-data-en`
# Note: If you specify `--packing true`, you must also set `--attn_impl flash_attn`

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset '<your-dataset>' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
```

## Reinforcement Learning (RL)

ms-swift supports RLHF methods such as DPO, GRPO, DAPO, PPO, KTO, and GKD. This section will focus on using ms-swift for GRPO training on Qwen3-8B. For more information about GRPO, refer to the [GRPO documentation](../Instruction/GRPO/GetStarted/GRPO.md). Additional RLHF training scripts can be found in [examples/train/rlhf](https://github.com/modelscope/ms-swift/tree/main/examples/train/rlhf).

### Environment Setup

In addition to installing the dependencies related to ms-swift mentioned above, you also need to install the following:

```shell
pip install "math_verify==0.5.2"
pip install vllm==0.8.5.post1
```


### Data Preparation

The dataset format used for GRPO training with ms-swift is similar to that of SFT, but it does not require the final assistant's response part. If accuracy is used as the reward, an additional `solution` column is required to calculate accuracy.

Example dataset format:


```jsonl
{"messages": [{"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
{"messages": [{"role": "user", "content": "What is your name?"}]}
```

For data preparation for other RLHF algorithms, please refer to the [Custom Dataset Documentation](../Customization/Custom-dataset.md#rlhf).

Notes on dataset requirements:

- **Reward Function Calculation**: The dataset format depends on the reward function being used. Additional columns may be needed to support specific reward calculations. For example:
  - When using built-in `accuracy` or `cosine` rewards, the dataset must include a `solution` column to calculate the accuracy of responses.
  - Other columns in the dataset will be passed as `**kwargs` to the reward function for further customization.
- **Custom Reward Functions**: To customize the reward function according to your specific needs, refer to: [External Reward Plugin](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin). This plugin provides examples and templates for implementing custom reward functions.

We use AI-MO/NuminaMath-TIR as the dataset and compute the accuracy-based reward for model responses.

During training, we utilize vLLM to accelerate the sampling process. By setting `num_infer_workers=8`, we deploy one vLLM engine per device to speed up sampling.

```bash
# 70G*8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_length 4096 \
    --max_completion_length 4096 \
    --vllm_max_model_len 8192 \
    --reward_funcs accuracy \
    --num_generations 16 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --deepspeed zero3 \
    --num_infer_workers 8 \
    --tensor_parallel_size 1 \
    --temperature 1.0 \
    --top_p 0.85 \
    --log_completions true \
    --overlong_filter true
```

## Megatron-SWIFT

ms-swift introduces Megatron parallelism techniques to accelerate CPT/SFT/DPO for large models. Supported models can be found in the [Supported Models and Datasets Document](../Instruction/Supported-models-and-datasets.md).

For environment setup and conversion between HF and MCore model weights, refer to the [Megatron-SWIFT Training Documentation](../Instruction/Megatron-SWIFT-Training.md).

We will use Alibaba Cloud DLC to launch training. The training environment consists of two nodes equipped with 8x 80GiB A800 GPUs each. For more information on multi-node launching, see [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node).

```bash
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# Ensure that the weight save path `--save` and packing cache path `--packing_cache` are the same and shared across both nodes.
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
megatron sft \
    --load Qwen3-30B-A3B-Base-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 2000 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 100 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Base \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_flash_attn true
```


Training loss chart (partial):

<img width="910" alt="Image" src="https://github.com/user-attachments/assets/9fe393aa-8299-4659-aa2f-be5d44f0730b" />

Effect screenshot:

<img width="1066" alt="Image" src="https://github.com/user-attachments/assets/1a924130-1954-43e9-9093-b019aeef5949" />


The custom dataset format is the same as that used in `swift sft`. For details, see the previous sections. Simply specify `--dataset <dataset_path>`.

A comparison of training speed and GPU memory usage when performing full-parameter fine-tuning of the Qwen3-30B-A3B model using `megatron sft` and `swift sft` is shown below:

|          | Megatron-LM | DeepSpeed-ZeRO2 | DeepSpeed-ZeRO3 |
| -------- | ----------- | --------------- | --------------- |
| 训练速度 | 9.6s/it     | -               | 91.2s/it        |
| 显存使用 | 16 * 60GiB  | OOM             | 16 * 80GiB      |
