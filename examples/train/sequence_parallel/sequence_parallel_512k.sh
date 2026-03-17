#!/usr/bin/env bash

# Original benchmark:
# - Env: 8 * A100
# - Max Length: 512000
# - GPU Memory: 8 * 80GiB, Training Speed 150s/it
#
# Ring attention is only enabled when rp_world_size > 1.
# For 32-head Qwen models, using 6 ranks yields sp_world_size=2, rp_world_size=3.
# The defaults below favor a 1-step ring smoke on machines that already have
# ~/model/Qwen3-4B and ~/dataset/self-cognition available. Override MODEL,
# DATASET, NPROC_PER_NODE, DEEPSPEED_CONFIG, MAX_STEPS, etc. to restore the
# original benchmark profile.

DEFAULT_MODEL=Qwen/QwQ-32B
DEFAULT_DATASET='AI-ModelScope/LongAlpaca-12k'
DEFAULT_NPROC_PER_NODE=8

if [ -d "${HOME}/model/Qwen3-4B" ] && [ -d "${HOME}/dataset/self-cognition" ]; then
    DEFAULT_MODEL="${HOME}/model/Qwen3-4B"
    DEFAULT_DATASET="${HOME}/dataset/self-cognition/self_cognition.jsonl"
    DEFAULT_NPROC_PER_NODE=6
fi

OUTPUT_DIR="${OUTPUT_DIR:-output/sequence_parallel_512k_ring}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0}"
MAX_LENGTH="${MAX_LENGTH:-512000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512000}"
MAX_STEPS="${MAX_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

CMD=(
    swift sft
    --model "${MODEL:-${DEFAULT_MODEL}}"
    --output_dir "${OUTPUT_DIR}"
    --tuner_type lora
    --dataset "${DATASET:-${DEFAULT_DATASET}}"
    --load_from_cache_file true
    --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
    --torch_dtype bfloat16
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --learning_rate 1e-5
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --packing true
    --rope_scaling yarn
    --max_length "${MAX_LENGTH}"
    --max_model_len "${MAX_MODEL_LEN}"
    --max_steps "${MAX_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --warmup_ratio 0.05
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --save_total_limit 2
    --use_liger_kernel "${USE_LIGER_KERNEL}"
    --save_only_model true
    --attn_impl flash_attn
    --padding_free true
    --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE:-${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}}"
)

if [ -n "${DEEPSPEED_CONFIG}" ]; then
    CMD+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}" \
CELOSS_PARALLEL_SIZE="${CELOSS_PARALLEL_SIZE:-2048}" \
"${CMD[@]}"
