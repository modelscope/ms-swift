# GKD

**版本依赖**：ms-swift >= 3.11

如果你是首次使用 GKD，请先参考 [GKD文档](../Instruction/GKD.md)。

GKD（Generalized Knowledge Distillation，广义知识蒸馏）是一种将教师模型的知识迁移到学生模型的训练方法，通过计算两个模型输出分布之间的 Jensen-Shannon 散度（JSD）损失来实现知识蒸馏。

## 功能支持

Megatron GKD 当前已支持以下功能：

- **训练模式**：全参数训练与 LoRA 微调
- **并行策略**：支持上下文并行（CP）、流水线并行（PP）、张量并行（TP）和专家并行（EP）
- **模型支持**：兼容 Megatron-SWIFT 中的 LLM 及 MLLM
- **Teacher Offload**：支持将教师模型卸载到 CPU 以节省 GPU 显存
- **在线生成**：支持使用 vLLM 进行学生模型的 on-policy 生成

### 当前限制

以下功能在后续版本中将逐步支持：

- **教师模型在线生成**（`seq_kd=True`）：当前 Sequential KD 模式下的教师模型生成暂不支持
- **非vLLM生成**：On-policy 生成当前仅支持 vLLM
- **教师模型使用不同并行参数**

⚠️ 注意事项：
- **On-policy 生成**：需要启用 vLLM（`--use_vllm true --vllm_mode colocate/server`）
- 当 `lmbda > 0` 但未启用 vLLM 时，将自动回退到 off-policy 模式（使用数据集响应）
- 当 `seq_kd=True` 时，由于教师生成暂不支持，将自动回退到 off-policy 模式

## 参数说明

### GKD 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--teacher_model` | str | 必需 | 教师模型路径或模型 ID |
| `--teacher_model_type` | str | None | 教师模型类型，不指定则自动检测 |
| `--teacher_model_revision` | str | None | 教师模型版本 |
| `--beta` | float | 0.5 | JSD 散度插值系数：<br>• 0.0: Forward KL<br>• 0.5: 对称 JSD<br>• 1.0: Reverse KL |
| `--lmbda` | float | 0.5 | On-Policy 学习触发概率：<br>• 0.0: 纯 Off-Policy<br>• 1.0: 纯 On-Policy |
| `--seq_kd` | bool | False | 是否使用教师生成的响应（当前暂不支持） |
| `--temperature` | float | 0.9 | 温度参数，用于采样和损失计算 |
| `--offload_teacher_model` | bool | False | 是否将教师模型卸载到 CPU 以节省显存 |
| `--max_completion_length` | int | 512 | 生成时的最大 token 数 |

### 批量相关参数

与 Megatron SFT 相同，使用以下参数控制批量大小：

| 参数 | 说明 |
|------|------|
| `--micro_batch_size` | 每张 GPU 的训练批次大小 |
| `--global_batch_size` | 全局批次大小：`micro_batch_size × dp_size × gradient_accumulation_steps` |

## 快速开始

### 基础训练（Off-Policy）

使用数据集中的响应进行知识蒸馏：

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-3B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/gkd-qwen2.5-3b \
    --save_interval 100 \
    --max_length 2048 \
    --beta 0.5 \
    --temperature 0.9
```

### On-Policy 训练（使用 vLLM）

启用学生模型在线生成：

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-3B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/gkd-qwen2.5-3b-onpolicy \
    --max_length 2048 \
    --lmbda 0.5 \
    --beta 0.5 \
    --temperature 0.9 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --max_completion_length 512
```

### 节省显存（Teacher Offload）

当 GPU 显存有限时，可以启用教师模型卸载：

```bash
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-3B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --offload_teacher_model true \
    # ... 其他参数
```

## 三种训练模式

GKD 支持三种训练模式，通过 `lmbda` 和 `seq_kd` 参数控制：

### Mode 1: On-Policy 学习
- 触发条件：`random() < lmbda` 且 `use_vllm=True`
- 数据来源：学生模型生成的响应
- 特点：学生从自己的错误中学习，提升鲁棒性

### Mode 2: Sequential KD（当前暂不支持）
- 触发条件：`random() >= lmbda` 且 `seq_kd=True`
- 数据来源：教师模型生成的响应

### Mode 3: Off-Policy 学习
- 触发条件：其他情况
- 数据来源：数据集中的标注响应
- 特点：最稳定的训练模式

## 参考

更多参数请参考[命令行文档](./Command-line-parameters.md)

训练脚本请参考 [Megatron GKD 脚本](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd)
