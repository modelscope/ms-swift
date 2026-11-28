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
# TUNER=full selects full fine-tuning; MODEL_TYPE=hy_v4_flash_vl selects image/video training.
# Native registration replaces the vendor's v4 custom_register_path and Trainer patches.
MODEL_TYPE=${MODEL_TYPE:-hy_v4_flash}
if [[ "$MODEL_TYPE" == hy_v4_flash_vl ]]; then
    MODEL=${MODEL:-Tencent-Hunyuan/Hy4-Flash-VL}
else
    MODEL=${MODEL:-Tencent-Hunyuan/Hy4-Flash}
fi
swift sft \
    --model "$MODEL" --model_type "$MODEL_TYPE" \
    --dataset "${DATASET:?Set DATASET to the text or VL fine-tuning dataset}" \
    --tuner "${TUNER:-lora}" --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
    --target_modules q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj o_proj \
    --torch_dtype bfloat16 --attn_impl eager \
    --fsdp fsdp2 --parallel_spec "fsdp${WORLD}" \
    --model_kwargs '{"memory_efficient_init":true}' \
    --max_length 1024 --lazy_tokenize true --dataset_num_proc 4 \
    --per_device_train_batch_size "${WORLD}" --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 --max_steps 20 --num_train_epochs 3 --warmup_steps 0 \
    --lr_scheduler cosine --gradient_checkpointing true --max_grad_norm 1 \
    --packing false --padding_free false \
    --save_steps 20 --save_total_limit 3 --logging_steps 1 --report_to none \
    --output_dir "${OUTPUT_DIR:-output/hy4-flash-sft-fsdp2}" "$@"
