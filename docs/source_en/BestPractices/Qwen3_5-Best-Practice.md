# Qwen3.5 Best Practices

ms-swift 4.0 supports training [Qwen3.5](https://github.com/QwenLM/Qwen3.5) Dense/MoE models using transformers/Megatron backends. Qwen3.5 is a multimodal model with hybrid thinking, combining linear attention and full attention. This article will introduce how to perform inference, instruction fine-tuning, and reinforcement learning on Qwen3.5 Dense/MoE models. (The reinforcement learning section will be supplemented later)

## Environment Setup

```shell
pip install -U ms-swift
pip install -U "transformers>=5.2.0" "qwen_vl_utils>=0.0.14" peft liger-kernel

# flash-linear-attention
# Please install the fla main branch. If you encounter slow training issues, please refer to: https://github.com/fla-org/flash-linear-attention/issues/758
pip install -U git+https://github.com/fla-org/flash-linear-attention

# causal_conv1d
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

# deepspeed training
pip install deepspeed

# vllm (torch2.10) for inference/deployment/RL
pip install uv
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
# For RL training, need to override vllm's default installation version
# For training errors, refer to this issue: https://github.com/modelscope/ms-swift/issues/8188
pip install -U "transformers>=5.2.0"
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

The following provides a fine-tuning script for the Qwen3.5-4B model. This example script is for demonstration purposes only. Training memory is 4 * 20GiB, and training time is 12 minutes. Since GatedDeltaNet does not support packing/padding_free, we use the group_by_length parameter to accelerate training, ensuring DP load balancing and reducing zero padding in micro batches, but this will cause loss curve fluctuations (due to insufficient data randomization), although you can also remove this parameter.
The script for fine-tuning the model is as follows:

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

- Full parameter training: Refer to [this example](https://github.com/modelscope/ms-swift/tree/main/examples/models/qwen3_5/mcore_full.sh).
- Regarding MTP training: ms-swift currently does not support multimodal MTP training. If you are only training on pure text data, please set the `SKIP_MULTIMODAL_MTP_VALIDATION=1` environment variable to skip the validation check.

## Reinforcement Learning (RL)

Coming soon
