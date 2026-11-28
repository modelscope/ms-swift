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
swift sft \
    --model "${MODEL:-inclusionAI/Ling-3.0-tiny}" --model_type ling3 \
    --dataset "${DATASET:-AI-ModelScope/alpaca-gpt4-data-zh}" \
    --tuner lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
    --torch_dtype bfloat16 --attn_impl flash_attention_2 \
    --fsdp fsdp2 --parallel_spec "fsdp${WORLD}" \
    --model_kwargs '{"memory_efficient_init":true}' \
    --per_device_train_batch_size "${WORLD}" --gradient_accumulation_steps 2 \
    --gradient_checkpointing false --max_length 2048 --padding_side right \
    --packing true --packing_length 2048 --packing_strategy sequential \
    --num_train_epochs 2 --eval_strategy no --save_steps 500 --logging_steps 1 \
    --dataset_num_proc 8 --dataloader_num_workers 8 --save_total_limit 2 \
    --save_only_model true --output_dir "${OUTPUT_DIR:-output/Ling3.0-Tiny-lora-fsdp2}" "$@"
