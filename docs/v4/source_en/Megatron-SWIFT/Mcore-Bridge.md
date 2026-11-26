# Mcore Bridge

Megatron is renowned for its exceptional training speed and rich parallel techniques, but this also comes with a relatively high barrier to entry. Therefore, [mcore-bridge](https://github.com/modelscope/mcore-bridge) was created to make Megatron training as simple and user-friendly as transformers. Through Mcore-Bridge, users can:

1. Directly load model weights in safetensors format and seamlessly use Megatron for efficient training. Save training weights directly in safetensors format without additional conversion.
2. Support bidirectional conversion compatible with LoRA incremental weights.
3. Support `Megatron->vLLM` weight synchronization for algorithms like GRPO/GKD.
4. Support multi-machine conversion of ultra-large-scale models.

Mcore-Bridge is compatible with various model architectures including Dense/MoE/multimodal. After training is complete, the converted models can be directly deployed using mainstream inference frameworks such as transformers, vLLM, SGLang, etc.

## Seamless Training

Currently, Mcore-Bridge supports parallel techniques such as TP/PP/EP/ETP/VPP, and the supported models can be found in the [Supported Models Documentation](../Instruction/Supported-models-and-datasets.md). The following introduces Mcore-Bridge's seamless training capabilities.

- When reading models with `--model/--adapters/--ref_model/--ref_adapters`, mcore-bridge is used to load safetensors format weights. With `--mcore_model/--mcore_adapter/--mcore_ref_model/--mcore_ref_adapter`, the default mcore loading method is used.
- `save_safetensors` determines whether weights are saved in safetensors or mcore format. When `--no_save_optim false` is set, mcore weights are always saved additionally for checkpoint resumption.
- Tip: During GKD/GRPO training, if you encounter GPU OOM issues when updating vLLM weights, you can set `--offload_bridge true` to offload tensors to CPU to reduce GPU memory usage.

### Full Parameter

Below is an example of training the multimodal model Qwen3-VL:

```shell
# 2 * 76GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --save_safetensors true \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen3-VL-8B-Instruct \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```

Then we perform inference on the validation set:

```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3-VL-8B-Instruct/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --stream true
```

### LoRA

In addition to full-parameter training, Mcore-Bridge also supports LoRA training.

Below is an example of self-cognition training using LoRA for the text-only model Qwen3-Moe:

- If you want to export merged weights instead of LoRA delta weights, please set `--merge_lora true`. Setting `--merge_lora true` has better compatibility and supports all model series.
- Note: (For transformers>5.0) Transformers 5.0 refactored the MoE model architecture. This new structure does not support MoE LoRA inference and may cause inference anomalies. **It is recommended to merge LoRA for MoE models** (vLLM is not affected; refer to vLLM's support status).
- Note: (For transformers<5.0) Due to structural differences between transformers and Megatron model experts (e.g., the expert components in transformers' Qwen3-VL-MoE are implemented as Parameters rather than Linear layers), some models cannot convert LoRA delta weights (however, Qwen3-VL-MoE does support conversion when LoRA training targets only linear_proj and linear_qkv). Most models support LoRA conversion, such as: Qwen3-MoE, Qwen3-Omni-MoE, GLM4.5-V, etc.

```shell
# 50GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
              'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --expert_model_parallel_size 2 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot
```

Perform inference on the exported LoRA weights using the vLLM inference engine:

```shell
# For specific model LoRA support in vLLM, please refer to the vLLM documentation.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --stream true
```

If you need to manually **Merge-LoRA**, you can use the `megatron export` command. Note: Please do not use the `swift export` command to merge LoRA, as the **MoE model structures** in Megatron and transformers are not necessarily consistent.

```shell
# If the adapter is in mcore format, please use `--mcore_adapter`
# If the final format needs to be in mcore format, use `--to_mcore true`
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --output_dir megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx-merged \
    --merge_lora true \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2
```

Perform inference on the merged full weights using the transformers inference engine:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx-merged \
    --stream true
```

## `megatron export` and Conversion Accuracy Testing

In addition to supporting safetensors conversion and saving during training, Mcore-Bridge also supports the `megatron export` command for standalone weight export. `megatron export` supports conversion precision testing during weight conversion, which is very helpful for verifying accuracy when integrating new models. Typically, models already integrated into Megatron-SWIFT will not have precision misalignment issues, so you can confidently set `--test_convert_precision false`.
- Note: For multimodal models, please focus on the `mean_diff (with loss)` field. The `mean_diff` may show a large difference because it includes image tokens, and loss is not calculated for that portion.

Full parameter weights:

```shell
# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --output_dir Qwen3-30B-A3B-Instruct-2507-mcore \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

```shell
# torch_dist -> safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --mcore_model Qwen3-30B-A3B-Instruct-2507-mcore \
    --output_dir Qwen3-30B-A3B-Instruct-2507-hf \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

LoRA weights:

```shell
# torch_dist -> safetensors
# If you need to perform merge-lora and test precision alignment after merge-lora, simply set `--merge_lora true`
# You can also change `--model safetensors-path` to `--mcore_model torch-dist-path`. These two methods are equivalent, and mcore-bridge will handle it automatically.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --mcore_adapter megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --output_dir megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx-lora \
    --merge_lora false \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

```shell
# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx-lora \
    --output_dir megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx-mcore \
    --merge_lora false \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

## Using Code

Please refer to [mcore-bridge github](https://github.com/modelscope/mcore-bridge/tree/main?tab=readme-ov-file#-quick-Start)
