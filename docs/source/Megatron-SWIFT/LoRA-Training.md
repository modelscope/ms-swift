# LoRA训练

Qwen3-235B-A22B-Instruct-250718 单机8卡H20 LoRA训练的最佳实践参考：[https://github.com/modelscope/ms-swift/pull/5033](https://github.com/modelscope/ms-swift/pull/5033)。

环境准备请参考Megatron-SWIFT的[快速开始文档](./Quick-start.md)。

## 传统方式

### HF转换Mcore

以下，我们分别介绍使用`swift export`和`megatron export`命令进行权重转换。相比于`swift export`，`megatron export`支持多机和LoRA增量权重转换，但也更加复杂，需要在导出时额外指定并行参数，例如`--tensor_model_parallel_size`, `--export_model_parallel_size`，具体参考[Mcore-Bridge文档](./Mcore-Bridge.md)。若要使用`swift export`命令，参考[快速开始文档](./Quick-start.md)。
- `swift export`使用单进程，将HF权重放置在gpu中，并使用device_map并行；mcore权重放置在cpu中，且不开启并行。这种方式非常易于debug，并测试HF和mcore的精度对齐情况。
- `megatron export`使用torchrun启动多进程，mcore权重放置在gpu中，支持开启各种并行、fp8和mtp等功能。如果需测试精度对齐情况，最后一个rank会加载HF权重，并放置在cpu中。

```shell
# megatron export
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor_model_parallel_size 2 \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen2.5-7B-Instruct-mcore \
    --test_convert_precision true

# swift export
# CUDA_VISIBLE_DEVICES=0 \
# swift export \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --to_mcore true \
#     --torch_dtype bfloat16 \
#     --output_dir Qwen2.5-7B-Instruct-mcore \
#     --test_convert_precision true
```

### LoRA训练

训练脚本：
```bash
# full: 2 * 70GiB 0.61s/it
# lora: 2 * 14GiB 0.45s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --mcore_model Qwen2.5-7B-Instruct-mcore \
    --save_safetensors false \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot
```
- MoE模型的LoRA训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/lora)。

### MCore转换HF

```bash
# megatron export
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --mcore_adapter megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --merge_lora false \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --test_convert_precision true

# swift export
# CUDA_VISIBLE_DEVICES=0 \
# swift export \
#     --mcore_adapter megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
#     --to_hf true \
#     --torch_dtype bfloat16 \
#     --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
#     --test_convert_precision true
```
- 注意：`--mcore_adapter`文件夹中包含`args.json`文件，转换过程会读取文件中`--model/--mcore_model`以及LoRA相关的参数信息。`swift export`暂不支持LoRA增量权重的转换。`megatron export`你可以使用`--merge_lora`参数控制是否进行权重合并。

### 推理
```shell
# 如果是全量权重，请将`--adapters`替换为`--model
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --stream true
```

### Merge-LoRA

如果只想merge-lora，而不希望转成HF格式权重，用于后续DPO训练，可以使用以下脚本：
```shell
# megatron export
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --mcore_adapter megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
    --tensor_model_parallel_size 2 \
    --to_mcore true \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-mcore \
    --test_convert_precision true

# swift export
# CUDA_VISIBLE_DEVICES=0 \
# swift export \
#     --mcore_adapter megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
#     --to_mcore true \
#     --torch_dtype bfloat16 \
#     --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-mcore \
#     --test_convert_precision true
```

## Mcore-Bridge【推荐】

### 训练

```shell
# full: 2 * 70GiB 0.61s/it
# lora: 2 * 14GiB 0.45s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot
```

### 推理

```shell
# 如果是全量权重，请将`--adapters`替换为`--model
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --stream true
```
