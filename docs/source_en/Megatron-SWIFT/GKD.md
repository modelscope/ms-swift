# GKD

If you are new to GKD/OPD-RL, please refer to the [distillation documentation](../Instruction/Distillation.md) first.

GKD (Generalized Knowledge Distillation) is a training method that transfers knowledge from a teacher model to a student model by computing the Jensen-Shannon Divergence (JSD) loss between their output distributions.

## Feature Support

Megatron GKD currently supports the following features:

- **Training Modes**: Full parameter training and LoRA fine-tuning
- **Parallelism Strategies**: Context Parallel (CP), Pipeline Parallel (PP), Tensor Parallel (TP), and Expert Parallel (EP)
- **Model Support**: Compatible with LLMs and MLLMs in Megatron-SWIFT
- **Teacher Offload**: Supports offloading teacher model to CPU to save GPU memory
- **Online Generation**: Supports on-policy generation using vLLM for student model

## Parameters

### GKD-specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--teacher_model` | str | - | Path or model ID of the teacher model<br>*Can be omitted when using `teacher_model_server` |
| `--teacher_model_server` | str | None | Teacher model service URL (`vllm serve` only), e.g. `http://localhost:8000` |
| `--gkd_logits_topk` | int | None | Number of Top-K logits; required when using external API |
| `--beta` | float | 0.5 | JSD divergence interpolation coefficient:<br>• 0.0: Forward KL<br>• 0.5: Symmetric JSD<br>• 1.0: Reverse KL |
| `--lmbda` | float | 0.5 | On-Policy learning probability:<br>• 0.0: Pure Off-Policy<br>• 1.0: Pure On-Policy |
| `--temperature` | float | 0.9 | Temperature for sampling and loss computation |
| `--sft_alpha` | float | 0 | Mix in a  proportion of SFT loss; applied to non-student-generated completions |
| `--max_completion_length` | int | 512 | Maximum tokens for generation |

### Batch-related Parameters

Same as Megatron SFT, use the following parameters to control batch size:

| Parameter | Description |
|-----------|-------------|
| `--micro_batch_size` | Training batch size per DP group |
| `--global_batch_size` | Global batch size: `micro_batch_size × dp_size × gradient_accumulation_steps` |

## Reference

For more parameters, please refer to [Command-line Parameters](./Command-line-parameters.md)

For training scripts, please refer to [Megatron GKD Scripts](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd)

Training script using Teacher Server reference [here](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/rlhf/gkd/teacher_server.sh)
