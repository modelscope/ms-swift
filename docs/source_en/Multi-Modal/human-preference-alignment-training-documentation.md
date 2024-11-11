# Human Preference Alignment Training Documentation
This document provides training scripts for various human preference alignment algorithms. If you wish to delve deeper into more detailed algorithm information and selection methods, please refer to [documentation](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/M.%E4%BA%BA%E7%B1%BB%E5%81%8F%E5%A5%BD%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83.md)

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [DPO](#dpo)
- [RM](#rm)
- [CPO](#cpo)
- [ORPO](#orpo)
- [SimPO](#simpo)

## Environment Setup
```bash
# Set pip global mirror (for faster downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
# Environment alignment (usually not necessary. If you encounter errors, you can run the following code, the repository uses the latest environment test)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## Dataset
Vision human preference alignment training typically requires data in the format $(x,y_w,y_l)$, where $x$ represents the model input, include textual prompt and images, and $y_w,y_l$ represent the preferred and rejected answers according to human preference, such as ![dpo_data](../../resources/vdpo_data.png)


**Custom Dataset Format**

```jsonl
{"system": "123", "query": "11111", "response": "22222", "rejected_response": "33333", "images": ["image_path"], "history": [["query1", "response1"], ["query2", "response2"]]}
{"system": "123", "query": "aaaaa", "response": "bbbbb", "rejected_response": "ccccc", "images": ["image_path"], "history": [["query1", "response1"], ["query2", "response2"]]}
{"system": "123", "query": "AAAAA", "response": "BBBBB", "rejected_response": "CCCCC", "images": ["image_path"], "history": [["query1", "response1"], ["query2", "response2"]]}
```

Different models have varying support for the number of images. Please refer to the corresponding best practices document for each model.

**Training Tips**:

- The following training scripts use --lora_target_modules DEFAULT to only train the model's QKV matrices, but you can set --lora_target_modules ALL to train all linear layers of the model

## DPO
[paper arvix](https://arxiv.org/abs/2305.18290)

Hyperparameters
- `beta`：KL regularization coefficient, the higher the value, the greater the penalty for deviations from the reference model. Default is 0.1

It is recommended to train with the preferred answer part of the preference dataset before starting DPO training to ensure data fits the distribution requirements of the DPO algorithm.

We also mix sft loss in the DPO loss to stabilize training; you can adjust the sft loss coefficient by setting the hyperparameter `rpo_alpha`, the default is `1.`.

For training script, we provide single card/multi-card device map/multi-card ddp versions, for brevity, only the single card version is given for subsequent algorithms.

```bash
# Experimental environment: A100
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type dpo \
    --model_type llava1_6-mistral-7b-instruct \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --sft_type  lora \
    --dataset rlaif-v#1000 \
    --num_train_epochs  2  \
    --lora_target_modules  DEFAULT  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2

# MP(device map)
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type dpo \
    --model_type llava1_6-mistral-7b-instruct \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --sft_type  lora \
    --dataset rlaif-v#1000 \
    --num_train_epochs  2  \
    --lora_target_modules  DEFAULT  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2

# DDP + MP
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift rlhf \
    --rlhf_type dpo \
    --model_type llava1_6-mistral-7b-instruct \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --sft_type  lora \
    --dataset rlaif-v#1000 \
    --num_train_epochs  2  \
    --lora_target_modules  DEFAULT  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  $(expr 16 / $nproc_per_node)  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```
Model inference and deployment after training can refer to the best practice documentation for the corresponding model, [Mutlimodal Deployment Document](./mutlimodal-deployment.md) and [VLLM Inference Acceleration Document](./vllm-inference-acceleration.md)

## RM
[paper arvix](https://arxiv.org/abs/2203.02155)

Reward Modeling phase in RLHF

```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type rm \
    --model_type  internvl2-2b \
    --sft_type  lora \
    --dataset rlaif-v#1000 \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```

The weights of the added value head will be saved in the `value_head.safetensors` or `value_head.bin` file.

## CPO
[Paper arvix](https://arxiv.org/abs/2401.08417)
Hyperparameters
- beta: The beta factor in CPO loss., default is 0.1
- cpo_alpha: Controls the strength of the BC regularizer in CPO training, default is 1.0

Training script
```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type cpo \
    --model_type  llava1_6-mistral-7b-instruct \
    --beta 0.1 \
    --sft_type  lora \
    --dataset rlaif-v#1000 \
    --num_train_epochs  2  \
    --lora_target_modules  DEFAULT  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```

## ORPO
[paper arvix](https://arxiv.org/abs/2403.07691)
Hyperparameters
- lambda: Coefficient for the Odds Ratio loss

**Note**: ORPO uses the parameter beta to input the hyperparameter lambda

```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type orpo \
    --model_type  llava1_6-mistral-7b-instruct \
    --beta 0.1 \
    --sft_type  lora \
    --dataset rlaif-v#1000 \
    --num_train_epochs  2  \
    --lora_target_modules  DEFAULT  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```


## SimPO
[Paper arvix](https://arxiv.org/abs/2405.14734)
Hyperparameters
- beta: Coefficient before the hidden reward, default is 2.0
- simpo_gamma: Reward margin term, default is 1.0
- cpo_alpha: Controls the strength of the BC regularizer in CPO training, mix nll loss in CPO to enhances training stability, with a default value of 1.0. Setting it to 0.0 uses the original SimPO algorithm.

Training script
```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type simpo \
    --model_type  llava1_6-mistral-7b-instruct \
    --beta 2.0 \
    --simpo_gamma 1.0 \
    --sft_type  lora \
    --dataset rlaif-v#1000 \
    --num_train_epochs  2  \
    --lora_target_modules  DEFAULT  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```
