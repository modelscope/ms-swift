# Human Preference Alignment Training Documentation
This document provides training scripts for various human preference alignment algorithms. If you wish to delve deeper into more detailed algorithm information and selection methods, please refer to [documentation](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/M.%E4%BA%BA%E7%B1%BB%E5%81%8F%E5%A5%BD%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83.md)

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [DPO](#dpo)
- [KTO](#kto)
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
Human preference alignment training typically requires data in the format $(x,y_w,y_l)$, where $x$ represents the model input, and $y_w,y_l$ represent the preferred and rejected answers according to human preference, such as ![dpo_data](../../resources/dpo_data.png)


Data for the KTO algorithm is somewhat special, requiring only data in the format $(x,y,\text{label})$ , where $x$ represents the model input, $y$ represents the model output, and the label indicates whether the answer aligns with human preferences

For example, ![kto_data](../../resources/kto_data.png)

**Training Tips**:

- If you are training a base model with history data, you need to specify a template that supports multi-turn dialogue (base models often do not support multi-turn dialogue); for this situation, we have set the default chatml template, but you can also use --model_type to select the template for the training model
- For training with a custom dataset, please refer to [Customization](../Instruction/Customization.md)
- The following training scripts use --lora_target_modules ALL to train all linear layers of the model, but you can set --lora_target_modules DEFAULT to only train the model's QKV matrices

## DPO
[paper arvix](https://arxiv.org/abs/2305.18290)

Hyperparameters
- `beta`：KL regularization coefficient, the higher the value, the greater the penalty for deviations from the reference model. Default is 0.1

It is recommended to train with the preferred answer part of the preference dataset before starting DPO training to ensure data fits the distribution requirements of the DPO algorithm.

We also mix sft loss in the DPO loss to stabilize training; you can adjust the sft loss coefficient by setting the hyperparameter `rpo_alpha`, the default is `1.`.

For training script, we provide single card/multi-card device map/multi-card ddp versions, for brevity, only the single card version is given for subsequent algorithms.

```bash
# Experimental environment: A100
# Memory usage: 40G
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type dpo \
    --model_type  llama3-8b-instruct \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --sft_type  lora \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2

# MP(device map)
# Memory usage: 2*24G
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type dpo \
    --model_type  llama3-8b-instruct \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --sft_type  lora \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2

# DDP + MP
# Memory usage: 4*24G
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift rlhf \
    --rlhf_type dpo \
    --model_type  llama3-8b-instruct \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --sft_type  lora \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  $(expr 16 / $nproc_per_node)  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```
Model inference and deployment after training can refer to [LLM Inference Document](./LLM-Inference.md) and [VLLM Inference Acceleration and Deployment Document](./VLLM-inference-acceleration-and-deployment.md)

## RM
[paper arvix](https://arxiv.org/abs/2203.02155)

Reward Modeling phase in RLHF

Using the base model or instruct model after SFT as the foundation, add a value head and train it on a preference dataset to obtain the reward model.

```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type rm \
    --model_type  llama3-8b-instruct \
    --sft_type  lora \
    --dataset hh-rlhf-cn-harmless-base-cn \
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

## PPO
[Paper arvix](https://arxiv.org/abs/2203.02155)

In the PPO (Proximal Policy Optimization) phase of RLHF (Reinforcement Learning with Human Feedback), four models are involved:

- model: The training model, which is either the base model after SFT (Supervised Fine-Tuning) or an instruct model.
- ref_model: The reference model, which defaults to the model.
- reward_model: The reward model, obtained from the RM (Reward Modeling) training phase.
- value_model: The value model, initialized from the reward model and updated synchronously during training.

Hyperparameters

- local_rollout_forward_batch_size: Per rank no grad forward pass in the rollout phase, default is 64
- whiten_rewards: Whether to whiten the rewards, default is False
- kl_coef: KL coefficient, default is 0.05
- cliprange: Clip range in the PPO policy loss funtion, default is 0.2
- vf_coef: Coefficient for the value loss function, default is 0.1
- cliprange_value: Clip range in the PPO value loss function, default is 0.2
- gamma: Discount factor for cumulative rewards, default is 1.0
- lam: Lambda value for [GAE](https://arxiv.org/abs/1506.02438), default is 0.95

```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type ppo \
    --model_type  llama3-8b-instruct \
    --sft_type  lora \
    --dataset hh-rlhf-cn-harmless-base-cn \
    --reward_model_id_or_path path/to/reawrd_model \
    --reward_model_type  llama3-8b-instruct \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```

Note: When training the base model, you need to perform SFT first, followed by RLHF. Specify the chat template, and it is recommended to use full for sft_type.

For an explanation of the training metrics, refer to the [documentation]((https://huggingface.co/docs/trl/ppov2_trainer#explanation-of-the-logged-metrics)).


## KTO
[Paper arvix](https://arxiv.org/abs/2402.01306)

Hyperparameters

- beta: KL regularization coefficient, the higher the value, the greater the penalty for deviations from the reference model. Default is 0.1
- desirable_weight: The $\lambda_D$ term in the loss function, the loss weight for preference answer samples. Default is 1.0
- undesirable_weight: The $\lambda_U$ term in the loss function, the loss weight for rejected answer samples. Default is 1.0

Use $n_D$ and $n_U$ to respectively represent the number of preference answers and rejected answers in the dataset. For hyperparameters $\lambda_D$ and $\lambda_U$, the authors recommend setting $\frac{\lambda_Dn_D}{\lambda_Un_U}\in[1,\frac{4}{3}]$

Training script using $(x,y,\text{label})$ format data
```bash
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type kto \
    --model_type  llama3-8b-instruct \
    --beta 0.1 \
    --desirable_weight 1.0 \
    --undesirable_weight 1.0 \
    --sft_type  lora \
    --dataset ultrafeedback-kto \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```

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
    --model_type  llama3-8b-instruct \
    --beta 0.1 \
    --sft_type  lora \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
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
    --model_type  llama3-8b-instruct \
    --beta 0.1 \
    --sft_type  lora \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
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
    --model_type  llama3-8b-instruct \
    --beta 2.0 \
    --simpo_gamma 1.0 \
    --sft_type  lora \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```
