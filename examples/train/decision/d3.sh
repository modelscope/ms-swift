#!/usr/bin/env bash
# Run manually on the Ascend validation host from the repository root.
set -euo pipefail
: "${D3_MODEL_DIR:?Set the local D3 release directory}"
: "${D3_OUTPUT_DIR:?Set a separate output directory}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SWIFT_SINGLE_DEVICE_MODE=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
swift sft \
    --model "$D3_MODEL_DIR" --model_type d3 --task_type decision \
    --external_plugins examples/train/decision/dataset.py \
    --dataset d3_decision --val_dataset d3_decision_val --split_dataset_ratio 0 \
    --strict true --load_from_cache_file false \
    --tuner_type lora --tuner_backend peft \
    --loss_type scoring --padding_side left --attn_impl sdpa \
    --torch_dtype bfloat16 --max_length 2048 --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 2 --dataset_num_proc 1 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 --max_steps "${D3_MAX_STEPS:-4}" \
    --save_steps 2 --eval_steps 2 --save_strategy steps --eval_strategy steps \
    --save_only_model false --optim adamw_torch --logging_steps 1 \
    --report_to none --output_dir "$D3_OUTPUT_DIR"
