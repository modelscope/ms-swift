# GKD

**Version Requirement**: ms-swift >= 3.11

If you are new to GKD, please refer to the [GKD Documentation](../Instruction/GKD.md) first.

GKD (Generalized Knowledge Distillation) is a training method that transfers knowledge from a teacher model to a student model by computing the Jensen-Shannon Divergence (JSD) loss between their output distributions.

## Feature Support

Megatron GKD currently supports the following features:

- **Training Modes**: Full parameter training and LoRA fine-tuning
- **Parallelism Strategies**: Context Parallel (CP), Pipeline Parallel (PP), Tensor Parallel (TP), and Expert Parallel (EP)
- **Heterogeneous Models**: Teacher and student models can have different architectures (different `hidden_size`, `num_layers`, `vocab_size`, etc.)
- **Model Support**: Compatible with LLMs and MLLMs (Multimodal Large Models) in Megatron-SWIFT
- **Teacher Offload**: Supports offloading teacher model to CPU to save GPU memory
- **Online Generation**: Supports on-policy generation using vLLM for student model

### Current Limitations

The following features will be gradually supported in future versions:

- **Teacher Model Online Generation** (`seq_kd=True`): Teacher model generation in Sequential KD mode is not yet supported
- **Megatron Native Generation**: On-policy generation currently only supports vLLM, not Megatron native inference

⚠️ Notes:
- **On-policy Generation**: Requires vLLM (`--use_vllm true --vllm_mode colocate/server`)
- When `lmbda > 0` but vLLM is not enabled, it will automatically fall back to off-policy mode (using dataset responses)
- When `seq_kd=True`, since teacher generation is not yet supported, it will automatically fall back to off-policy mode

## Parameters

### GKD-specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--teacher_model` | str | Required | Path or model ID of the teacher model |
| `--teacher_model_type` | str | None | Teacher model type, auto-detected if not specified |
| `--teacher_model_revision` | str | None | Teacher model version |
| `--beta` | float | 0.5 | JSD divergence interpolation coefficient:<br>• 0.0: Forward KL<br>• 0.5: Symmetric JSD<br>• 1.0: Reverse KL |
| `--lmbda` | float | 0.5 | On-Policy learning probability:<br>• 0.0: Pure Off-Policy<br>• 1.0: Pure On-Policy |
| `--seq_kd` | bool | False | Use teacher-generated responses (not yet supported) |
| `--temperature` | float | 0.9 | Temperature for sampling and loss computation |
| `--offload_teacher_model` | bool | False | Offload teacher model to CPU to save memory |
| `--max_completion_length` | int | 512 | Maximum tokens for generation |

### Batch-related Parameters

Same as Megatron SFT, use the following parameters to control batch size:

| Parameter | Description |
|-----------|-------------|
| `--micro_batch_size` | Training batch size per GPU |
| `--global_batch_size` | Global batch size: `micro_batch_size × dp_size × gradient_accumulation_steps` |

## Quick Start

### Basic Training (Off-Policy)

Knowledge distillation using responses from the dataset:

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

### On-Policy Training (with vLLM)

Enable student model online generation:

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

### Memory Saving (Teacher Offload)

Enable teacher model offloading when GPU memory is limited:

```bash
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-3B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --offload_teacher_model true \
    # ... other parameters
```

## Three Training Modes

GKD supports three training modes, controlled by `lmbda` and `seq_kd` parameters:

### Mode 1: On-Policy Learning
- Trigger: `random() < lmbda` and `use_vllm=True`
- Data source: Responses generated by the student model
- Feature: Student learns from its own mistakes, improving robustness

### Mode 2: Sequential KD (Not Yet Supported)
- Trigger: `random() >= lmbda` and `seq_kd=True`
- Data source: Responses generated by the teacher model

### Mode 3: Off-Policy Learning
- Trigger: Other cases
- Data source: Labeled responses from the dataset
- Feature: Most stable training mode

## Reference

For more parameters, please refer to [Command-line Parameters](./Command-line-parameters.md)

For training scripts, please refer to [Megatron GKD Scripts](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd)
