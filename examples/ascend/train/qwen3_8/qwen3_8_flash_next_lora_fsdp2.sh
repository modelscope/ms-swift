#!/bin/bash
# Atlas 800T A3 Ascend NPU. FSDP2 + LoRA SFT.
# Run from ms-swift repo root:
#   bash examples/ascend/train/qwen3_8/qwen3_8_flash_next_lora_fsdp2.sh
# Override model path: MODEL=/path/to/Qwen3.8-Flash-Next bash ...

set -euo pipefail

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

MODEL="${MODEL:-/mnt/weight/Qwen3.8-Flash-Next}"
LOG_DIR="${LOG_DIR:-output/Qwen3.8-Flash-Next-lora-fsdp2}"
mkdir -p "${LOG_DIR}"

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 \
swift sft \
    --model "${MODEL}" \
    --model_type qwen4_exp \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --dataset "${DATASET:-AI-ModelScope/alpaca-gpt4-data-zh#128}" \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_length 256 \
    --max_steps "${MAX_STEPS:-3}" \
    --num_train_epochs 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${MAX_STEPS:-3}" \
    --logging_strategy steps \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --output_dir "${LOG_DIR}" \
    --report_to none \
    --attn_impl sdpa \
    --packing false \
    --use_logits_to_keep true \
    --fsdp "./examples/ascend/train/qwen3_8/qwen3_8_flash_next_fsdp2.json"
