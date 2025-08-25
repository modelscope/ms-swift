# LoRA训练

Qwen3-235B-A22B-Instruct-250718 单机8卡H20 LoRA训练的最佳实践参考：[https://github.com/modelscope/ms-swift/pull/5033](https://github.com/modelscope/ms-swift/pull/5033)。

环境准备请参考Megatron-SWIFT的[快速开始文档](./快速开始.md)。

## HF转换Mcore

转换方式与全参数训练一致，脚本如下：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen2.5-7B-Instruct-mcore \
    --test_convert_precision true
```

## LoRA训练

训练脚本：
```bash
# full: 2 * 70GiB 0.61s/it
# lora: 2 * 14GiB 0.45s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --train_type lora \
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
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot
```
- MoE模型的LoRA训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/lora)。

## MCore转换HF

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf \
    --test_convert_precision true
```
- 注意：`mcore_adapters`文件夹中包含`args.json`文件，转换过程中会读取文件中`mcore_model`和LoRA相关的参数信息，并将`mcore_model`和`mcore_adapters`merge-lora成完整权重，最终转换成HF格式权重。（暂不支持LoRA增量权重的转换）

## Merge-LoRA

如果只想merge-lora，而不希望转成HF格式权重，用于后续DPO训练，可以使用以下脚本：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-mcore \
    --test_convert_precision true
```
