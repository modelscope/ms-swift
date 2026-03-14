# Qwen3.5 Best Practices

ms-swift 4.0 supports training [Qwen3.5](https://github.com/QwenLM/Qwen3.5) Dense/MoE models using transformers/Megatron backends. Qwen3.5 is a multimodal model with hybrid thinking, combining linear attention and full attention. This article will introduce how to perform inference, instruction fine-tuning, and reinforcement learning on Qwen3.5 Dense/MoE models.

## Environment Setup

```shell
pip install -U ms-swift
pip install -U "transformers>=5.3.0" "qwen_vl_utils>=0.0.14" peft liger-kernel

# flash-linear-attention
# Please install the fla main branch. If you encounter slow training issues, please refer to: https://github.com/fla-org/flash-linear-attention/issues/758
pip install -U git+https://github.com/fla-org/flash-linear-attention

# causal_conv1d
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

# flash-attention
pip install "flash-attn==2.8.3" --no-build-isolation

# deepspeed training
pip install deepspeed

# vllm (torch2.10) for inference/deployment/RL
pip install -U "vllm>=0.17.0"
# For RL training, need to override vllm's default installation version
# For training errors, refer to this issue: https://github.com/modelscope/ms-swift/issues/8188
pip install -U "transformers>=5.3.0"
```

- Qwen3.5 video data training hangs: Using the decord backend to read videos may cause hanging issues, refer to [this issue](https://github.com/dmlc/decord/issues/269). You can use the torchcodec backend, specifically refer to the [qwen_vl_utils](https://github.com/QwenLM/Qwen3-VL/blob/50068df2334f309979ff05d75f1078c8309c63ed/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L390-L400) library.

## Inference

Using ms-swift's `TransformersEngine` for inference:

- The meaning of model-specific parameters such as `VIDEO_MAX_TOKEN_NUM` environment variables is the same as Qwen3-VL, refer to [Command-line Parameters Documentation](../Instruction/Command-line-parameters.md#qwen3_vl,qwen3_5).

```python
import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

model, processor = get_model_processor('Qwen/Qwen3.5-4B')  # attn_impl='flash_attention_2'
template = get_template(processor, enable_thinking=False)
engine = TransformersEngine(model, template=template)
infer_request = InferRequest(messages=[{
    "role": "user",
    "content": '<video>Describe this video.',
}], videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'])
request_config = RequestConfig(max_tokens=128, temperature=0)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
print(response)

# use stream
request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config=request_config)
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
```

Using command line for inference:

```shell
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --enable_thinking false \
    --stream true
```

## Fine-tuning

This chapter will introduce how to train Qwen3.5 using ms-swift and Megatron-SWIFT. It is recommended to use ms-swift (i.e., transformers backend, more convenient and simple) for Dense models, and Megatron-SWIFT (i.e., megatron backend, faster training speed) for MoE models.

If you need to fine-tune the model with a custom dataset, you can prepare the data in the following format and set `--dataset train.jsonl --val_dataset val.jsonl` in the command line, where the validation set is optional. For more information, please refer to [Multimodal Dataset Documentation](../Customization/Custom-dataset.md#multimodal).

```jsonl
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang?"}, {"role": "assistant", "content": "The capital of Zhejiang is Hangzhou."}]}
{"messages": [{"role": "user", "content": "<image><image>What's the difference between these two images?"}, {"role": "assistant", "content": "The first one is a kitten, the second one is a puppy"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
{"messages": [{"role": "system", "content": "You are a helpful and harmless assistant"}, {"role": "user", "content": "<image>What's in the image, <video>what's in the video?"}, {"role": "assistant", "content": "There's an elephant in the image, and a puppy running on the grass in the video"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```

Qwen3.5's bbox output uses normalized relative coordinates with a scale of 1000. You can use the grounding dataset format provided by ms-swift, where the coordinates in "bbox" are absolute coordinates, and ms-swift will automatically convert absolute coordinates to normalized relative coordinates with a scale of 1000. For more information, please refer to [Grounding Dataset Format Documentation](../Customization/Custom-dataset.md#grounding).

```jsonl
{"messages": [{"role": "user", "content": "<image>Locate the <ref-object> in the image"}, {"role": "assistant", "content": "[\n\t{\"bbox_2d\": <bbox>, \"label\": \"<ref-object>\"},\n\t{\"bbox_2d\": <bbox>, \"label\": \"<ref-object>\"}\n]"}], "images": ["cat.png"], "objects": {"ref": ["sheep", "sheep", "sheep"], "bbox": [[90.9, 160.8, 135, 212.8], [360.9, 480.8, 495, 532.8]]}}
```

### Dense Models

Below is a fine-tuning script for the Qwen3.5-4B model. This example script is for demonstration purposes only. Training memory usage is 4 × 20GiB, with a training time of 12 minutes. Since transformers' GatedDeltaNet does not support packing/padding_free (Megatron does support it, see below), we use the `group_by_length` parameter to accelerate training, ensuring load balancing across data parallelism (DP) and reducing zero-padding in micro batches. However, this may cause fluctuations in the loss curve due to insufficient data shuffling. You can also remove this parameter if preferred.

The fine-tuning script is as follows:

```shell
# 4 * 20GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --group_by_length true \
    --output_dir output/Qwen3.5-4B \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --model_author swift \
    --model_name swift-robot
```

After training, use the following script to perform inference on the validation set:

```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --adapters output/Qwen3.5-4B/vx-xxx/checkpoint-xxx \
    --stream true \
    --enable_thinking false \
    --max_new_tokens 512 \
    --load_data_args true
```


```text
[QUERY] 你好，你是谁？
[RESPONSE] <think>

</think>

你好，我是由swift开发的人工智能语言模型，我的名字叫swift-robot。很高兴能与你交流。
--------------------------------------------------
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] e = \sum _ { k = 0 } ^ { \infty } \frac { 1 } { k ! }
[RESPONSE] <think>

</think>

e = \sum _ { k = 0 } ^ { \infty } \frac { 1 } { k ! }
```

```python
import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

adapter_dir = 'output/Qwen3.5-4B/vx-xxx/checkpoint-xxx'
enable_thinking = False

model, processor = get_model_processor('Qwen/Qwen3.5-4B')  # attn_impl='flash_attention_2'
model = PeftModel.from_pretrained(model, adapter_dir)
template = get_template(processor, enable_thinking=enable_thinking)
engine = TransformersEngine(model, template=template)
infer_request = InferRequest(messages=[{
    "role": "user",
    "content": 'who are you?',
}])
request_config = RequestConfig(max_tokens=128, temperature=0)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
print(response)

# use stream
request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config=request_config)
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
# I am an artificial intelligence assistant named swift-robot, trained by swift. I am designed to understand and generate natural language text in order to provide information, answer questions, and engage in conversation with humans. How can I assist you?
```

For an example of training MoE using the transformers backend, refer to: https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen3_5/transformers.sh

### MoE Models

Qwen3.5-35B-A3B Megatron training. For environment preparation, please refer to [Megatron-SWIFT Quick Start Documentation](../Megatron-SWIFT/Quick-start.md). You can complete the following example in 15 minutes:

```shell
# 4 * 40GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --merge_lora true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --group_by_length true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/Qwen3.5-35B-A3B \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --padding_free false \
    --model_author swift \
    --model_name swift-robot
```

After training, use the following script to perform inference on the validation set:

```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model megatron_output/Qwen3.5-35B-A3B/vx-xxx/checkpoint-xxx-merged \
    --stream true \
    --enable_thinking false \
    --max_new_tokens 512 \
    --load_data_args true
```

Tips for training Qwen3.5 with Megatron-SWIFT:

- Full parameter training: Refer to [this example](https://github.com/modelscope/ms-swift/tree/main/examples/models/qwen3_5/mcore_full.sh).
- Regarding MTP training: ms-swift currently does not support multimodal MTP training. If you are only training on pure text data, please set the `SKIP_MULTIMODAL_MTP_VALIDATION=1` environment variable to skip the validation check.
- TP Limitation Removed: Using `megatron-core>=0.16` removes the `num_query_groups` limitation on TP.
- By default, `GatedDeltaNet` uses the transformers implementation (to ensure stability, the default behavior remains unchanged for now). Using `megatron-core>=0.16` and setting the environment variable `SWIFT_USE_MCORE_GDN=1` switches to the mcore implementation, which supports TP for GDN and reduces memory usage.
- Support for padding_free/packing: Packing can improve training speed. You need to set the `SWIFT_USE_MCORE_GDN=1` environment variable. Refer to [this example](https://github.com/modelscope/ms-swift/tree/main/examples/models/qwen3_5/packing.sh).
- apply_wd_to_qk_layernorm: Apply weight decay to qk layernorm. Default is False.


## Reinforcement Learning (RL)

Using Qwen3.5-2B as an example, we demonstrate GRPO and GKD training on the [GSM8K](https://www.modelscope.cn/datasets/modelscope/gsm8k) dataset and evaluate on the GSM8K test set. To avoid excessively long chain-of-thought outputs, all experiments set `enable_thinking false`.

### GRPO

#### Dense Model
Full-parameter training with GRPO, using `gsm8k_accuracy` and `gsm8k_format` as reward functions. See [gsm8k_plugin.py](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py) for the reward implementation.

```shell
SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --external_plugins examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py \
    --reward_funcs gsm8k_accuracy gsm8k_format \
    --columns '{"answer": "solution"}' \
    --enable_thinking false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 10240 \
    --sleep_level 1 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset 'modelscope/gsm8k' \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --save_steps 10 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --deepspeed zero2 \
    --log_completions true \
    --report_to tensorboard swanlab \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none
```

Evaluate the checkpoints:

```shell
CUDA_VISIBLE_DEVICES=0 swift eval \
    --model output/Qwen3.5-2B/vxx-xxx-xxx/checkpoint-xx \
    --enable_thinking false \
    --eval_dataset gsm8k \
    --eval_backend Native --infer_backend vllm \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
```

GSM8K evaluation results at 10-step intervals for the first 50 steps:

| Model / Steps | GSM8K Accuracy | Improvement |
|---|---|---|
| Qwen3.5-2B (baseline) | 0.7597 | - |
| GRPO 10 steps | 0.7650 | +0.53 |
| GRPO 20 steps | 0.7748 | +1.51 |
| GRPO 30 steps | 0.7779 | +1.82 |
| GRPO 40 steps | 0.7817 | +2.20 |
| GRPO 50 steps | 0.7885 | +2.88 |

### MoE Model

GRPO LoRA training for Qwen3.5-35B-A3B MoE model using the Megatron backend, trained on the [DAPO-Math-17k](https://www.modelscope.cn/datasets/open-r1/DAPO-Math-17k-Processed) dataset with `accuracy` as reward functions.

```shell
SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --enable_thinking false \
    --merge_lora true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 1 \
    --moe_permute_fusion true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --num_train_epochs 1 \
    --global_batch_size 64 \
    --micro_batch_size 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 9192 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --target_modules all-linear \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.00 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 1 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --logging_steps 1 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --save_steps 20 \
    --attention_backend flash \
    --moe_expert_capacity_factor 2 \
    --temperature 1.0 \
    --padding_free false \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard swanlab
```

Evaluate on AIME-2025 and MATH-500:

```shell
CUDA_VISIBLE_DEVICES=0,1 swift eval \
    --model <checkpoint-merged-path> \
    --enable_thinking false \
    --eval_dataset aime25 math_500 \
    --eval_backend Native --infer_backend vllm \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 10000 \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
    --eval_num_proc 8
```

Evaluation results on AIME-2025 and MATH-500:

| Model / Steps | AIME-2025 | MATH-500 |
|---|---|---|
| Qwen3.5-35B-A3B (baseline) | 43.33 | 92.40 |
| Megatron GRPO 20 steps | 53.33 (+10.00) | 95.80 (+3.40) |
| Megatron GRPO 40 steps | 53.33 (+10.00) | 96.60 (+4.20) |

### GKD

LoRA training with GKD (Guided Knowledge Distillation), using Qwen3.5-9B as the teacher model. First, launch the teacher server with vLLM (alternatively, use the `--teacher_model` parameter to load the model directly):

```shell
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3.5-9B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.8 \
    --max-logprobs 64
```

Then start GKD training on the remaining GPUs:

```shell
NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=1,2,3 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-2B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 64 \
    --enable_thinking false \
    --tuner_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 10240 \
    --sleep_level 0 \
    --dataset 'modelscope/gsm8k' \
    --lmbda 1 \
    --seq_kd false \
    --beta 0.5 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 10 \
    --max_length 2048 \
    --max_completion_length 8192 \
    --warmup_ratio 0.1 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --attn_impl flash_attn \
    --report_to tensorboard swanlab
```
Evaluate the checkpoints:

```shell
CUDA_VISIBLE_DEVICES=0 swift eval \
    --model Qwen/Qwen3.5-2B \
    --adapters output/Qwen3.5-2B/vxx-xxx-xxx/checkpoint-xx \
    --merge_lora true \
    --enable_thinking false \
    --eval_dataset gsm8k \
    --eval_backend Native --infer_backend vllm \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
```

GSM8K evaluation results at 100-step intervals for the first 300 steps:

| Model / Steps | GSM8K Accuracy | Improvement |
|---|---|---|
| Qwen3.5-2B (baseline) | 0.7597 | - |
| GKD 100 steps | 0.7968 | +3.71 |
| GKD 200 steps | 0.8188 | +5.91 |
| GKD 300 steps | 0.8332 | +7.35 |
