#!/bin/bash
# Atlas 800T A3 Ascend NPU. FSDP2 + LoRA SFT.
# Run from ms-swift repo root:
#   bash examples/ascend/train/qwen3_8/qwen3_8_27b_lora_fsdp2.sh
# Override model path: MODEL=Qwen/Qwen3.8-27B bash ...

set -euo pipefail

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export USE_MCORE_GDN=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-.triton_cache}"
if [ "${CLEAR_TRITON_CACHE:-0}" = "1" ]; then
    rm -rf "${TRITON_CACHE_DIR}"
fi
mkdir -p "${TRITON_CACHE_DIR}"

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
accelerate launch --config_file "./examples/ascend/train/qwen3_8/qwen3_8_27b_fsdp2.json" \
    swift/cli/sft.py \
    --model "${MODEL}" \
    --model_type qwen3_5 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh' \
              'AI-ModelScope/alpaca-gpt4-data-en' \
              'swift/self-cognition' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --max_steps 50 \
    --num_train_epochs 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --logging_strategy steps \
    --logging_steps 1 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --save_only_model true \
    --output_dir output/Qwen3.8-27B-lora-fsdp2 \
    --report_to tensorboard \
    --attn_impl flash_attention_2 \
    --packing true
