#!/usr/bin/env bash
# 4 x 180 GiB; BF16; peak memory is about 75 GiB/GPU on this image dataset.
# Requires transformers>=5.16.1, Megatron-Core>=0.18, Transformer Engine and mamba-ssm.
# MTP and FP8 are unsupported. Keep attention-only LoRA targets and freeze the ETLB bias.
# EVS is disabled by default; set USE_NEMOTRON_EVS=1 for video training (PP=1, no packing/cache).
# Use short clips: decoded video costs about 128 tokens per frame.

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --model nv-community/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules linear_qkv linear_proj \
    --freeze_vit true \
    --freeze_aligner true \
    --vit_attn_impl sdpa \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --mtp_num_layers 0 \
    --megatron_extra_kwargs '{"moe_router_bias_update_rate":0.0}' \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --recompute_granularity none \
    --padding_free false \
    --attention_backend unfused \
    --logging_steps 1 \
    --save_steps 200 \
    --save_safetensors true \
    --merge_lora false \
    --no_save_optim true \
    --no_save_rng true
