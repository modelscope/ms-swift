#!/bin/bash
# Ling3-tiny Ascend NPU FSDP2 LoRA SFT. Run from the repository root.
set -e

# Change CANN_PATH when the toolkit is installed elsewhere.
source "${CANN_PATH:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
accelerate launch \
    --config_file examples/ascend/train/ling3/fsdp2.json \
    swift/cli/sft.py \
    --model "${MODEL:-inclusionAI/Ling-3.0-tiny}" \
    --template ling3 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh' \
              'AI-ModelScope/alpaca-gpt4-data-en' \
              'swift/self-cognition' \
    --model_name Ling3 \
    --model_author inclusionAI \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing false \
    --max_length 2048 \
    --padding_side right \
    --packing true \
    --packing_length 2048 \
    --packing_strategy sequential \
    --num_train_epochs 2 \
    --eval_strategy no \
    --save_steps 500 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Ling3.0-Tiny-lora-fsdp2 \
    "$@"
