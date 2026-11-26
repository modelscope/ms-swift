# Mcore-Bridge
Megatron 以其卓越的训练速度和丰富的并行技术而著称，但也因此带来了较高的使用门槛。因此[mcore-bridge](https://github.com/modelscope/mcore-bridge) 应运而生，旨在让 Megatron 训练像 transformers 一样简单易用。通过 Mcore-Bridge，用户可以：

1. 直接加载 safetensors 格式的模型权重，无缝使用 Megatron 进行高效训练。直接保存 训练权重为 safetensors 格式，无需额外转换。
2. 兼容 LoRA 增量权重的双向转换。
3. 兼容GRPO/GKD等算法的`Megatron->vLLM`权重同步。
4. 支持多机转换超大规模模型。

Mcore-Bridge 兼容 Dense/MoE/多模态等多种模型架构。训练完成后，转换后的模型可直接使用 transformers、vLLM、SGLang 等主流推理框架部署。

## 无缝训练
目前Mcore-Bridge已支持TP/PP/EP/ETP/VPP等并行技术，支持的模型参考[支持的模型文档](../Instruction/Supported-models-and-datasets.md)。以下介绍Mcore-Bridge的无缝训练能力。

- 使用`--model/--adapters/--ref_model/--ref_adapters`参数读取模型时，将使用mcore-bridge来读取safetensors格式的模型权重。若使用`--mcore_model/--mcore_adapter/--mcore_ref_model/--mcore_ref_adapter`参数，则使用mcore默认方式读取。
- `save_safetensors`参数决定存储权重为safetensors格式还是mcore格式。如果设置`--no_save_optim false`则总会额外存储一份mcore权重用于断点续训。
- 提示：在GKD/GRPO训练期间，如果在vLLM权重更新时遇到 GPU OOM 问题，您可以设置 `--offload_bridge true` 将张量卸载到 CPU 减少 GPU 内存使用量。

### 全参数

以下为多模态模型Qwen3-VL模型训练的例子:
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

然后我们对验证集部分进行推理：
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

Mcore-Bridge除了支持全参数训练，还支持LoRA训练。

以下为纯文本模型Qwen3-Moe模型使用LoRA自我认知训练的例子：
- 若你希望导出merge后的权重，而不是LoRA增量权重，请设置`--merge_lora true`。设置`--merge_lora true`的兼容性更好，支持所有系列模型。
- 注意：（transformers>5.0的情况）transformers 5.0对Moe的模型组织结构进行了重构，该结构不支持Moe LoRA的推理，可能造成推理异常。**建议对Moe模型进行Merge LoRA**（vLLM不受影响，看vLLM的支持情况）。
- 注意：（transformers<5.0的情况）由于transformers和Megatron模型专家结构并不一定一致（例如transformers的Qwen3-VL-Moe的专家部分并不是Linear实现，而是Parameters），因此部分模型无法转换LoRA增量权重（若Qwen3-VL-Moe只设置linear_proj和linear_qkv训练LoRA也支持转换）。但大多数的模型支持LoRA转换，例如：Qwen3-Moe，Qwen3-Omni-Moe，GLM4.5-V等。

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


对导出的LoRA权重进行推理，这里使用vLLM推理引擎：
```shell
# 具体模型vLLM的LoRA的支持情况请参考vLLM文档。
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --stream true
```

如果你需要手动进行**Merge-LoRA**，你可以使用`megatron export`命令。注意：请不要使用`swift export`导出命令Merge-LoRA，因为Megatron与transformers的**Moe模型结构**并不一定一致。

```shell
# 如果是mcore格式的adapter，请使用`--mcore_adapter`
# 如果最终格式需要是mcore格式，则使用`--to_mcore true`
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

对Merge的全量权重进行推理，这里使用transformers推理引擎：

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx-merged \
    --stream true
```

## `megatron export` 与 转换精度测试

Mcore-Bridge除了支持在训练中进行safetensors的转换和保存，也支持了`megatron export`命令用于单独的权重导出。`megatron export`支持在权重转换时，对转换精度进行测试，这在接入新模型时验证接入准确性很有帮助。通常，Megatron-SWIFT已经接入的模型不会出现精度不对齐的情况，你可以放心设置`--test_convert_precision false`。
- 提示：多模态模型请关注`mean_diff (with loss)`字段，`mean_diff`因包含图像tokens且该部分不计算损失，有较大的diff。


全参数权重：
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



LoRA权重：
```shell
# torch_dist -> safetensors
# 若你需要进行merge-lora，并测试merge-lora后的精度对齐，你只需要设置`--merge_lora true`即可
# 你也可以将`--model safetensors-path`修改为`--mcore_model torch-dist-path`。这两种方式是等价的，mcore-bridge会自动处理。
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



## 使用代码

请查看[mcore-bridge github](https://github.com/modelscope/mcore-bridge/blob/main/README_zh.md#-%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
