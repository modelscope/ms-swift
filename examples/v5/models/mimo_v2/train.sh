#!/usr/bin/env bash
# Frozen-main MiMo-V2.5/2.6 Megatron LoRA SFT. Not executed in v5.
# Requires mcore-bridge>=1.7.0 in a compatible environment; no base-environment pin change is needed.
# Official FP8/MXFP4 checkpoint weights are dequantized to BF16 by mcore-bridge.
# CP must stay 1; SWA sinks require fused attention. TP>1 requires sequence_parallel.
# TE's adapter-tensor limit can require attention-only targets instead of all-linear.
set -euo pipefail
export USE_SWIFT_V5=1
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
swift sft --backend megatron \
    --model "${MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-RL}" \
    --model_type mimo_v2 \
    --dataset "${DATASET:-swift/self-cognition#500}" \
    --tuner lora --lora_rank 8 --lora_alpha 32 \
    --target_modules all-linear --freeze_vit true --freeze_aligner true \
    --tensor_model_parallel_size 2 --expert_model_parallel_size 4 \
    --sequence_parallel true --context_parallel_size 1 \
    --attention_backend fused --moe_grouped_gemm true --moe_permute_fusion true \
    --micro_batch_size 1 --global_batch_size 16 \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --cross_entropy_loss_fusion true --learning_rate 1e-4 --warmup_ratio 0.05 \
    --min_lr 1e-5 --num_train_epochs 1 --max_length 4096 --padding_free true \
    --torch_dtype bfloat16 --split_dataset_ratio 0.01 \
    --eval_steps 200 --save_steps 200 --save_only_model true \
    --dataset_num_proc 4 --dataloader_num_workers 4 --logging_steps 1 \
    --output_dir "${OUTPUT_DIR:-output/MiMo-V2.6-Flash-RL}" "$@"
