#!/usr/bin/env bash
set -euo pipefail

# AgentArk Env Server and Unity runtimes are prepared from the AgentArk repo.
# See: https://github.com/P90-RushB/AgentArk/tree/main/integrations/ms_swift/tutorial
: "${AGENTARK_MODEL:?Set AGENTARK_MODEL to a Swift-supported multimodal model}"
: "${AGENTARK_TICKET_DATASET:?Set AGENTARK_TICKET_DATASET to a generated ticket JSONL file}"
: "${AGENTARK_RUNTIME_CONFIG:?Set AGENTARK_RUNTIME_CONFIG to the AgentArk runtime YAML}"

export AGENTARK_SERVER_URL="${AGENTARK_SERVER_URL:-http://127.0.0.1:18080}"
export AGENTARK_PROTOCOL_VERSION="${AGENTARK_PROTOCOL_VERSION:-v2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
swift rlhf \
    --rlhf_type grpo \
    --model "$AGENTARK_MODEL" \
    --dataset "$AGENTARK_TICKET_DATASET" \
    --split_dataset_ratio 0 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --use_gym_env true \
    --gym_env agentark \
    --multi_turn_scheduler agentark_scheduler \
    --max_turns "${AGENTARK_MAX_TURNS:-6}" \
    --num_generations "${AGENTARK_NUM_GENERATIONS:-4}" \
    --generation_batch_size "${AGENTARK_GENERATION_BATCH_SIZE:-4}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length "${AGENTARK_MAX_LENGTH:-6144}" \
    --max_completion_length "${AGENTARK_MAX_COMPLETION_LENGTH:-512}" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${AGENTARK_VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
    --max_steps "${AGENTARK_MAX_STEPS:-1}" \
    --save_steps 1 \
    --logging_steps 1 \
    --log_completions true \
    --report_to none \
    --output_dir "${AGENTARK_OUTPUT_DIR:-output/agentark-smoke}" \
    "$@"
