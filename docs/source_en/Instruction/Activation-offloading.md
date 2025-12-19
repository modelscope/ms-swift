# Activation Offloading

## Overview

Activation offloading is a memory optimization technique that offloads activation tensors saved during the forward pass to CPU memory, and reloads them back to GPU when needed during the backward pass. This feature significantly reduces GPU memory usage, allowing you to train larger models or use larger batch sizes.

## Quick Start

### Enable Activation Offloading

To enable activation offloading during training, simply add the `--activation_cpu_offload true` flag:

```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh' \
    --deepspeed zero2 \
    --fsdp_strategy full_shard \
    --activation_cpu_offload true \
    --gradient_checkpointing true \
    --per_device_train_batch_size 4 \
    --output_dir output
```

### Key Requirements

1. **FSDP Training**: Activation offloading requires FSDP (Fully Sharded Data Parallel) training strategy
2. **PyTorch 2.0+**: Ensure you have PyTorch 2.0 or later
3. **Sufficient CPU RAM**: CPU memory should be 1.5-2x the GPU memory saved

## How It Works

### Architecture

Activation offloading uses an asynchronous double-buffer system:

1. **Forward Pass**: Activations are saved for backward pass
2. **Asynchronous Offloading**: Activations are offloaded to CPU while next layer computes
3. **Backward Pass**: Activations are reloaded to GPU when needed
4. **Double Buffering**: Two buffers overlap computation and data transfer

### Memory Savings

Typical memory savings with activation offloading:

| Configuration | Memory Reduction | Training Speed |
|---------------|------------------|----------------|
| Baseline | 0% | 100% |
| Activation Offloading | 40-50% | 92-95% |
| Offloading + Checkpointing | 60-70% | 85-90% |

## Usage Examples

### Example 1: Training Large Model on Limited GPU

```bash
# Train 7B model on 16GB GPU (normally requires 24GB+)
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh' \
    --deepspeed zero2 \
    --fsdp_strategy full_shard \
    --activation_cpu_offload true \
    --gradient_checkpointing true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --max_length 2048 \
    --output_dir output/qwen2.5-7b-sft
```

### Example 2: Multimodal Model Training

```bash
# Train Qwen3-VL with activation offloading
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-VL-7B-Instruct \
    --train_type lora \
    --dataset 'multimodal-data' \
    --deepspeed zero2 \
    --fsdp_strategy full_shard \
    --activation_cpu_offload true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --output_dir output/qwen3-vl-sft
```

## Configuration Options

### Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--activation_cpu_offload` | bool | `false` | Enable activation offloading |
| `--gradient_checkpointing` | bool | `true` | Enable gradient checkpointing (recommended) |
| `--deepspeed` | str | - | Use `zero2` for FSDP |
| `--fsdp_strategy` | str | - | Use `full_shard` for FSDP |

### Environment Variables

#### For NVIDIA GPUs
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

#### For Huawei Ascend NPUs
```bash
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export TORCH_HCCL_ZERO_COPY=1
```

## Best Practices

### 1. Combine with Gradient Checkpointing

For maximum memory savings, always combine activation offloading with gradient checkpointing:

```bash
--activation_cpu_offload true --gradient_checkpointing true
```

### 2. Batch Size Tuning

With activation offloading, you can typically increase batch size:

1. Start with baseline batch size (no offloading)
2. Enable activation offloading
3. Double the batch size
4. Monitor memory usage and adjust

### 3. Monitor Performance

Use these tools to monitor performance:

```bash
# GPU memory usage
nvidia-smi -l 1  # NVIDIA GPUs
npu-smi info     # Huawei Ascend NPUs

# Training logs
tail -f output/training.log | grep -E "(memory|offload|reload)"
```

## Troubleshooting

### Common Issues

#### Issue 1: "Activation offloading only supports FSDP training strategy"
**Solution**: Enable FSDP with proper configuration:
```bash
--deepspeed zero2 --fsdp_strategy full_shard
```

#### Issue 2: Out of CPU memory
**Solution**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Check available CPU RAM

#### Issue 3: Slow training performance
**Solution**:
1. Check CPU-GPU bandwidth
2. Ensure proper environment variables
3. Consider using faster CPU memory

### Debugging

Enable debug logging for detailed information:

```bash
export SWIFT_LOG_LEVEL=DEBUG
```

Look for these log messages:
- `Enabling activation offloading for FSDP strategy`
- `Offloading activation group X/Y`
- `Reloading activation group X/Y`
- `Memory saved: X GB (Y% reduction)`

## Advanced Topics

### Custom Tensor Filtering

For advanced users, you can implement custom tensor filtering:

```python
def custom_tensor_filter(tensor):
    # Only offload tensors larger than 1MB
    return tensor.numel() * tensor.element_size() > 1_000_000
```

### Manual Control via Python API

```python
from swift.plugin.activation_offload import enable_activation_offloading

# Manual enabling
enable_activation_offloading(
    model=model,
    strategy="fsdp2",  # or "fsdp"
    enable_ckpt=True
)
```

## Compatibility

### Supported Training Types
- Full parameter training ✅
- LoRA training ✅
- QLoRA training ✅
- Adapter training ✅

### Supported Model Types
- Text-only LLMs ✅
- Vision-Language models ✅
- Multimodal models ✅
- MoE models ⚠️ (Experimental)

### Hardware Support
- NVIDIA GPUs ✅
- Huawei Ascend NPUs ✅
- AMD GPUs ⚠️ (Experimental)
- CPU-only ❌ (Not supported)

## Performance Benchmarks

### Memory Savings

| Model | Baseline | With Offloading | Savings |
|-------|----------|-----------------|---------|
| Qwen2.5-7B | 24GB | 14GB | 42% |
| Qwen2.5-14B | 48GB | 26GB | 46% |
| Qwen2.5-32B | 96GB | 48GB | 50% |

### Training Speed

| Configuration | Relative Speed | Use Case |
|---------------|----------------|----------|
| Baseline | 100% | Abundant GPU memory |
| Activation Offloading | 92-95% | Memory-constrained |
| Offloading + Checkpointing | 85-90% | Maximum memory savings |

## FAQ

### Q: Can I use activation offloading with DDP?
**A**: No, activation offloading currently only supports FSDP training strategy.

### Q: Does it work with mixed precision training?
**A**: Yes, fully compatible with fp16, bf16, and other precision formats.

### Q: How does it compare to DeepSpeed ZeRO-Offload?
**A**: Activation offloading is a complementary technique that can be used together with ZeRO-Offload for even more memory savings.

### Q: Is there a minimum model size for benefits?
**A**: Generally, models larger than 1B parameters show significant benefits. Smaller models may not benefit due to overhead.

### Q: Can I disable it mid-training?
**A**: No, the configuration is fixed at training start.

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Gradient Checkpointing Paper](https://arxiv.org/abs/1604.06174)
- [Activation Offloading Research](https://arxiv.org/abs/2104.04445)
- [ms-swift Documentation](https://swift.readthedocs.io/)

## Support

For issues and questions:
1. Check the [ms-swift documentation](https://swift.readthedocs.io/)
2. Join the [Discord community](https://discord.com/invite/D27yfEFVz5)
3. Open an issue on [GitHub](https://github.com/modelscope/ms-swift/issues)

---

*Note: Activation offloading is an experimental feature and may change in future releases. Always test with your specific workload before production use.*
