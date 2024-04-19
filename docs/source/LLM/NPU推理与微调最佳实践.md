# NPU训练最佳实践

## 目录
- [环境准备](#环境准备)
- [微调](#微调)
- [推理](#推理)


## 环境准备

实验环境：8 * 昇腾910B3

```shell
pip install ms-swift -U
pip install torch-npu
```

测试环境是否安装正确：
```python
from transformers.utils import is_torch_npu_available
import torch

print(is_torch_npu_available())  # True
print(torch.npu.device_count())  # 8
```

## 微调
以下介绍LoRA的微调, 全参数微调设置参数`--sft_type full`即可.


### 单卡训练

通过如下命令启动单卡微调：

```shell
# 实验环境: 昇腾910B3
# 显存需求: 25GB
# 运行时长: 8小时
ASCEND_RT_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```


### 数据并行训练

```shell
# 实验环境: 4 * 昇腾910B3
# 显存需求: 4 * 30GB
# 运行时长: 2小时
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```


### Deepspeed训练

ZeRO2:
```shell
# 实验环境: 4 * 昇腾910B3
# 显存需求: 4 * 28GB
# 运行时长: 3小时
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
# 实验环境: 4 * 昇腾910B3
# 显存需求: 4 * 25GB
# 运行时长: 8小时
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


## 推理

原始模型:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
```

LoRA微调后:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer --ckpt_dir xxx/checkpoint-xxx --load_dataset_config true
```
