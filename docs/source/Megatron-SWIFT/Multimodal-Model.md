# 多模态模型

ms-swift引入了Megatron的并行技术来加速多模态大模型的训练。目前支持Qwen3-VL, Qwen3-Omni, Qwen2.5-VL, Qwen2.5-Omni, InternVL3.5, GLM4.5v, Kimi-VL等模型的CPT/SFT/GRPO/DPO/KTO/RM。完整支持的模型可以参考[支持的模型与数据集文档](../Instruction/Supported-models-and-datasets.md)。

环境准备请参考Megatron-SWIFT的[快速开始文档](./Quick-start.md)。

## Dense模型

这里介绍使用2卡80GiB A100对Qwen2.5-VL-7B-Instruct模型进行Latex-OCR的微调，分别使用全参数和LoRA的方式，以下最佳实践可以在10分钟内完成。

### Full

全参数训练脚本如下：
```shell
# 2 * 72GiB; 4.1s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
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
    --output_dir megatron_output/Qwen2.5-VL-7B-Instruct \
    --save_interval 200 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```


### LoRA

LoRA训练脚本如下：
```shell
# 2 * 23GiB; 2.3s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 1 \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/Qwen2.5-VL-7B-Instruct \
    --save_interval 200 \
    --max_length 2048 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```


最后，我们使用生成的HF格式权重对验证集进行推理：
```shell
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters megatron_output/Qwen2.5-VL-7B-Instruct/vx-xxx/checkpoint-xxx \
    --attn_impl flash_attn \
    --stream true \
    --load_data_args true \
    --temperature 0 \
    --max_new_tokens 512
```

推理结果如下：
```
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] \forall x \in X , ( \alpha f ) ( x ) = \alpha f ( x )
[RESPONSE] \forall x \in X , ( \alpha f ) ( x ) = \alpha f ( x )
--------------------------------------------------
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] \pi \int _ { c } ^ { d } \{ g ( y ) \} ^ { 2 } d y
[RESPONSE] \pi \int _ { c } ^ { d } \{ g ( y ) \} ^ { 2 } d y
--------------------------------------------------
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] [ \frac 2 3 x ^ { \frac 3 2 } ] _ { 0 } ^ { 1 }
[RESPONSE] [ \frac 2 3 x ^ { \frac 3 2 } ] _ { 0 } ^ { 1 }
```


## Moe模型


训练脚本：
```bash
# 2 * 43GiB, 8s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model OpenGVLab/InternVL3_5-30B-A3B \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --split_dataset_ratio 0.01 \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --output_dir megatron_output/InternVL3_5-30B-A3B \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash
```

训练结束后，我们使用生成的HF格式权重对验证集进行推理：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters megatron_output/InternVL3_5-30B-A3B/vx-xxx/checkpoint-xxx \
    --attn_impl flash_attn \
    --stream true \
    --load_data_args true \
    --temperature 0 \
    --max_new_tokens 512
```
