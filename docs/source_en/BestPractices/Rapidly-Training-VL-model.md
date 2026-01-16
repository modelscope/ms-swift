# Best Practices for Rapidly Training Vision-Language (VL) Models

This document provides best practices for quickly training vision-language (VL) models from scratch.

Model Links
- [Qwen2.5-VL-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)

Trained Model Link
- [Simple-VL-8B](https://www.modelscope.cn/models/swift/Simple-VL-8B/summary)


The training workflow builds upon the Qwen2.5-VL-7B-Instruct model architecture by replacing its internal large language model (LLM) component with the weights from Qwen3-8B , thereby enhancing the model's visual understanding capabilities. The process involves the following steps:

1. Modify the original model’s configuration file config.json to align with Qwen3-8B.
2. Initialize and load new model weights, saving them as a new model.
3. Fine-tune the new model in two stages:
    1. Stage 1 : Train only the vision-to-language alignment module (aligner), freezing the ViT and LLM components.
    2. Stage 2 : Unfreeze all modules and perform joint fine-tuning to improve overall performance.


## Model Modification

### Config File (config.json) Update
Due to structural differences between Qwen2.5-7B-Instruct and Qwen3-8B (e.g., number of layers, hidden dimensions), create a new config.json based on the Qwen2.5-VL-7B-Instruct config and update the following parameters to match Qwen3-8B:


```
Modified Parameters
1. hidden_size 3584->4096
2. intermediate_size: 18944->12288
3. num_attention_heads: 28->32
4. num_key_value_heads: 4->8
5. num_hidden_layers: 28->36
6. vocab_size:152064->151936
7. max_window_layers:28->36
8. out_hidden_size: 3584->4096

Newly Added Parameter
1. head_dim： 128
```

### Model Weight Initialization and Replacement
Use the following Python script to initialize, replace, and save the model weights:
```python
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger, Qwen2_5_VLModel
from accelerate import Accelerator

# Load original VL model and Qwen3-8B model
qwen2_5_vl_7b_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
device = qwen2_5_vl_7b_model.device

qwen3_8b_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map=device,
    torch_dtype=torch.bfloat16
)

# Load configurations
old_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
new_config = AutoConfig.from_pretrained("/path/to/new_config_dir")  # Path to new config directory
new_visual_config = new_config.vision_config

# Replace merger (aligner) layer
new_merger = Qwen2_5_VLPatchMerger(
    dim=new_visual_config.out_hidden_size,
    context_dim=new_visual_config.hidden_size,
    spatial_merge_size=new_visual_config.spatial_merge_size,
).to(device).to(torch.bfloat16)
qwen2_5_vl_7b_model.visual.merger = new_merger

# Replace LLM part of the VL model
new_llm_model = Qwen2_5_VLModel(new_config).to(device).to(torch.bfloat16)

for name, param in qwen3_8b_model.model.named_parameters():
    if name in new_llm_model.state_dict():
        new_llm_model.state_dict()[name].copy_(param)

qwen2_5_vl_7b_model.model = new_llm_model
qwen2_5_vl_7b_model.lm_head = qwen3_8b_model.lm_head

# Save modified model
accelerator = Accelerator()
accelerator.save_model(
    model=qwen2_5_vl_7b_model,
    save_directory="/path/to/save/Qwen3-VL-Model",
    max_shard_size="4GB",
    safe_serialization=True
)
```

After saving the weights, copy all files from the original Qwen2.5-VL-7B-Instruct model folder, except for the model weights(including `model.safetensors.index.json`), to the new model weights folder, and replace config.json with the newly modified config.json file.

## Training
To simplify the process, we skip pre-training and proceed directly to supervised fine-tuning (SFT). The training is divided into two stages:

### Stage 1: Train Aligner Layer
Train only the vision-to-language alignment module while freezing the ViT and LLM parts:
```bash
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/new_vl_model \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset xxx \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2
```

### Stage 2: Full Model Training

Unfreeze all modules and jointly train to enhance the model's visual understanding:

```bash
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/stage1_checkpoint \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset xxx \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2
```

## Inference / Deployment / Evaluation

### Inference
Perform inference using `swift infer`:
```bash
swift infer \
    --model /path/to/stage2_checkpoint
```

### Deoloyment
Accelerate model serving with vLLM:
```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift deploy \
    --model /path/to/stage2_checkpoint \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --served_model_name Qwen3-VL
```

### Evaluation
Evaluate the trained VL model using [EvalScope](https://github.com/modelscope/evalscope/).

Example Evaluation Using MMMU Benchmark
```python
from evalscope import TaskConfig, run_task

task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config={
        'data': ['MMMU_DEV_VAL'],
        'mode': 'all',
        'model': [
            {
                'api_base': 'http://localhost:8000/v1/chat/completions',
                'key': 'EMPTY',
                'name': 'CustomAPIModel',
                'temperature': 0.6,
                'type': 'Qwen3-VL',
                'img_size': -1,
                'video_llm': False,
                'max_tokens': 512,
            }
        ],
        'reuse': False,
        'nproc': 64,
        'judge': 'exact_matching'
    },
)

run_task(task_cfg=task_cfg_dict)
```
