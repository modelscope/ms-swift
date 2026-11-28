#!/usr/bin/env bash
# Frozen-main LFM3 LoRA SFT, using the v5 CLI. Not executed in v5.
# MODEL=LiquidAI/LFM3-VL-2B selects the registered VL loader/template.
set -euo pipefail
export USE_SWIFT_V5=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} swift sft \
    --model "${MODEL:-LiquidAI/LFM3-1.7B}" \
    --dataset "${DATASET:-AI-ModelScope/alpaca-gpt4-data-zh#2000}" \
    --tuner lora --target_modules all-linear \
    --max_length 2048 --padding_free false --packing false \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --num_train_epochs 1 \
    --attn_impl flash_attention_2 \
    --output_dir "${OUTPUT_DIR:-output/lfm3_sft}" "$@"
