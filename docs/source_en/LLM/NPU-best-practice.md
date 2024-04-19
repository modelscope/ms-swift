# NPU Best Practice

## Table of Contents
- [Environment Preparation](#Environment-Preparation)
- [Fine-tuning](#Fine-tuning)
- [Inference](#Inference)

## Environment Preparation

Experimental environment: 8 * Ascend 910B3

```shell
pip install ms-swift -U
pip install torch-npu
```

Verify the installation of the testing environment:
```python
from transformers.utils import is_torch_npu_available
import torch

print(is_torch_npu_available())  # True
print(torch.npu.device_count())  # 8
```

## Fine-tuning
The following introduces the fine-tuning of LoRA. Set the parameter `--sft_type full` for full parameter fine-tuning.


### Single Card Training

Start single card fine-tuning with the following command:

```shell
# Experimental Environment: Ascend 910B3
# GPU Memory Requirement: 25GB
# Runtime: 8 hours
ASCEND_RT_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```


### Training with DDP

```shell
# Experimental Environment: 4 * Ascend 910B3
# GPU Memory Requirement: 4 * 30GB
# Runtime: 2 hours
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```


### Training with DeepSpeed

ZeRO2:
```shell
# Experimental Environment: 4 * Ascend 910B3
# GPU Memory Requirement: 4 * 28GB
# Runtime: 3 hours
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
    --deepspeed default-zero2 \
```

ZeRO3:
```shell
# Experimental Environment: 4 * Ascend 910B3
# GPU Memory Requirement: 4 * 25GB
# Runtime: 8 hours
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
    --deepspeed default-zero3 \
```


## Inference

Original Model:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
```

After LoRA Fine-tuning:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer --ckpt_dir xxx/checkpoint-xxx --load_dataset_config true
```
