# LoRA Training

Best practice reference for single-node 8xH20 LoRA training with Qwen3-235B-A22B-Instruct-250718: https://github.com/modelscope/ms-swift/pull/5033.

For environment setup, please refer to the [Quick Start Guide](./Quick-start.md) of Megatron-SWIFT.


## Traditional Method

### Converting HF to Mcore

Below, we introduce weight conversion using the `swift export` and `megatron export` commands respectively. Compared to `swift export`, `megatron export` supports multi-node and LoRA incremental weight conversion, but is also more complex, requiring additional specification of parallelism parameters during export, such as `--tensor_model_parallel_size` and `--export_model_parallel_size`. For details, refer to the [Mcore-Bridge Documentation](./Mcore-Bridge.md). To use the `swift export` command, refer to the [Quick Start Documentation](./Quick-start.md).
- `swift export` uses a single process, places HF weights on the GPU, and uses device_map for parallelization; mcore weights are placed on the CPU without enabling parallelization. This approach is very easy to debug and test the precision alignment between HF and mcore.
- `megatron export` uses torchrun to launch multiple processes, places mcore weights on the GPU, supports enabling various parallelization methods, fp8, mtp, etc., with comprehensive functionality. If precision alignment testing is needed, the last rank will load HF weights and place them on the CPU.


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

### LoRA Training

Training Script:

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
- For LoRA training scripts of MoE models, please refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/lora).

### Converting MCore to HF

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

- Note: The `--mcore_adapter` folder contains an `args.json` file. The conversion process will read the `--model/--mcore_model` and LoRA-related parameter information from this file. `swift export` does not currently support conversion of LoRA incremental weights. With `megatron export`, you can use the `--merge_lora` parameter to control whether to merge weights.


### Inference

```shell
# If using full weights, replace `--adapters` with `--model`
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --stream true
```


### Merge-LoRA

If you only want to merge the LoRA weights without converting them to Hugging Face format, for subsequent DPO training, you can use the following script:

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


## Mcore-Bridge [Recommended]

### Training

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


### Inference

```shell
# If using full weights, replace `--adapters` with `--model`
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx-hf \
    --stream true
```
