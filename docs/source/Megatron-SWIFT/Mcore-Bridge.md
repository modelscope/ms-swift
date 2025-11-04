# Mcore Bridge

Megatron 以其卓越的训练速度和丰富的并行技术而著称，但也因此带来了较高的使用门槛。因此mcore-bridge 应运而生，旨在让 Megatron 训练像 transformers 一样简单易用。通过 Mcore-Bridge，用户可以：
1. 直接加载 safetensors 格式的模型权重，无缝使用 Megatron 进行高效训练。直接保存 训练权重为 safetensors 格式，无需额外转换。
2. 兼容 LoRA 权重的双向转换。
3. 兼容GRPO/GKD等算法的`Megatron->vLLM`权重同步。

Mcore-Bridge 兼容 Dense/MoE/多模态等多种模型架构。训练完成后，转换后的模型可直接使用 transformers、vLLM、SGLang 等主流推理框架部署。

## 无缝训练
目前Mcore-Bridge已支持TP/PP/EP/ETP/VPP等并行技术，支持所有Megatron-SWIFT支持的模型架构，参考[支持的模型文档](../Instruction/支持的模型和数据集.md)。以下介绍Mcore-Bridge的无缝训练能力，分别介绍Dense模型和Moe模型。

### Dense模型
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
    --load_safetensors true \
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
    --max_epochs 1 \
    --save megatron_output/Qwen3-VL-8B-Instruct \
    --save_interval 200 \
    --vit_gradient_checkpointing false \
    --max_length 2048 \
    --num_workers 4 \
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
    --model megatron_output/Qwen3-VL-8B-Instruct/vx-xxx \
    --load_data_args true \
    --stream true
```

### Moe模型
以下为纯文本模型Qwen3-Moe模型CoT训练的例子:

```shell
# 8 * 76GiB, 3s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 25 \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 8192 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash
```

## LoRA导出

Mcore-Bridge除了支持全参数的导入导出，还支持单独对LoRA增量模型进行导入导出。

以下为纯文本模型Qwen3-Moe模型使用LoRA自我认知训练的例子:
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
              'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --train_type lora \
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
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot
```

### Merge-LoRA

## 导出与转换精度测试 



## 单独使用

导入权重：




导出权重：




保存权重：


