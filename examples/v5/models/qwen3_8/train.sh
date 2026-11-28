#!/usr/bin/env bash
# Ported from frozen main; not executed in v5. Load the model's compatible environment first.
# On NPU, load CANN and set ASCEND_RT_VISIBLE_DEVICES; on GPU, set CUDA_VISIBLE_DEVICES.
# Each node runs this command with its own NODE_RANK and a shared MASTER_ADDR.
set -euo pipefail
export USE_SWIFT_V5=1
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
WORLD=$((NPROC_PER_NODE * NNODES))
# The frozen PLE n-gram table stays on CPU and is excluded from FSDP all-gathers.
swift sft \
    --model "${MODEL:-Qwen/Qwen3.8-Flash-Next}" --model_type qwen4_exp \
    --dataset "${DATASET:-AI-ModelScope/alpaca-gpt4-data-zh#128}" \
    --tuner lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
    --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa \
    --fsdp fsdp2 --parallel_spec "fsdp${WORLD}" \
    --model_kwargs '{"memory_efficient_init":true}' \
    --per_device_train_batch_size "${WORLD}" --gradient_accumulation_steps 1 \
    --gradient_checkpointing false --learning_rate 1e-4 --warmup_ratio 0.05 \
    --max_length 256 --max_steps "${MAX_STEPS:-3}" --num_train_epochs 1 \
    --eval_strategy no --save_steps "${MAX_STEPS:-3}" --logging_steps 1 \
    --dataloader_num_workers 4 --dataset_num_proc 8 --save_only_model true \
    --packing false --report_to none \
    --output_dir "${OUTPUT_DIR:-output/Qwen3.8-Flash-Next-lora-fsdp2}" "$@"
