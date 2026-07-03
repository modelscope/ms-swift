#!/bin/bash
# Multi-teacher OPD experiment script.
#
# This script demonstrates two routing modes:
# - Mode 1: Dataset-level routing (auto-injected 'dataset' column)
# - Mode 2: Sample-level routing (user-specified 'teacher_tag' column)
#
# Usage:
#   bash scripts/test_multi_teacher.sh          # Mode 1 (default)
#   bash scripts/test_multi_teacher.sh sample    # Mode 2
#
# Prerequisites:
# - swift installed (pip install -e .)
# - At least 3 GPUs available (1 for student, 1 per teacher server)

set -e

MODE="${1:-dataset}"

# === Config ===
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-0.6B}"
TEACHER_MODEL_1="${TEACHER_MODEL_1:-Qwen/Qwen3.5-1.7B}"  # Math specialist
TEACHER_MODEL_2="${TEACHER_MODEL_2:-Qwen/Qwen3.5-0.6B}"  # Code specialist
TEACHER_PORT_1=8000
TEACHER_PORT_2=8001
MAX_LENGTH=2048
MAX_COMPLETION=512
TOTAL_LEN=$((MAX_LENGTH + MAX_COMPLETION))
OUTPUT_DIR="output/multi_teacher_test"
DATA_FILE="data/multi_teacher_test.jsonl"

# === Step 1: Prepare mixed dataset ===
echo "=== Step 1: Prepare mixed dataset (mode: ${MODE}) ==="
mkdir -p data

if [ "${MODE}" = "sample" ]; then
    # Mode 2: Sample-level routing - user adds 'teacher_tag' column
    python -c "
import json, random
random.seed(42)
data = []
for i in range(100):
    tag = 'math' if i % 2 == 0 else 'code'
    if tag == 'math':
        prompt = f'What is {random.randint(1,100)} + {random.randint(1,100)}? Think step by step.'
    else:
        prompt = f'Write a Python function to check if {random.randint(1,1000)} is prime. Think step by step.'
    data.append({
        'messages': [{'role': 'user', 'content': prompt}],
        'teacher_tag': tag,
    })
with open('data/multi_teacher_test.jsonl', 'w') as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + '\\n')
print(f'Written {len(data)} samples with teacher_tag column')
"
    TAG_KEY="teacher_tag"
    DATASET_ARG="data/multi_teacher_test.jsonl"
else
    # Mode 1: Dataset-level routing - two separate files, Swift auto-injects 'dataset' column
    python -c "
import json, random
random.seed(42)
math_data = []
for i in range(50):
    prompt = f'What is {random.randint(1,100)} + {random.randint(1,100)}? Think step by step.'
    math_data.append({'messages': [{'role': 'user', 'content': prompt}]})
with open('data/multi_teacher_math.jsonl', 'w') as f:
    for d in math_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\\n')

code_data = []
for i in range(50):
    prompt = f'Write a Python function to check if {random.randint(1,1000)} is prime. Think step by step.'
    code_data.append({'messages': [{'role': 'user', 'content': prompt}]})
with open('data/multi_teacher_code.jsonl', 'w') as f:
    for d in code_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\\n')
print(f'Written 50 math + 50 code samples (separate files)')
"
    TAG_KEY="dataset"
    DATASET_ARG="data/multi_teacher_math.jsonl data/multi_teacher_code.jsonl"
fi

# === Step 2: Start teacher servers ===
echo "=== Step 2: Start teacher servers ==="

echo "[Teacher 1] Starting ${TEACHER_MODEL_1} on port ${TEACHER_PORT_1} (math specialist)..."
CUDA_VISIBLE_DEVICES=1 swift deploy \
    --model "${TEACHER_MODEL_1}" \
    --infer_backend vllm \
    --port ${TEACHER_PORT_1} \
    --max_logprobs 1 \
    --max_length ${TOTAL_LEN} \
    --vllm_max_model_len ${TOTAL_LEN} \
    --vllm_gpu_memory_utilization 0.5 &
TEACHER1_PID=$!
echo "Teacher 1 PID: ${TEACHER1_PID}"

echo "[Teacher 2] Starting ${TEACHER_MODEL_2} on port ${TEACHER_PORT_2} (code specialist)..."
CUDA_VISIBLE_DEVICES=2 swift deploy \
    --model "${TEACHER_MODEL_2}" \
    --infer_backend vllm \
    --port ${TEACHER_PORT_2} \
    --max_logprobs 1 \
    --max_length ${TOTAL_LEN} \
    --vllm_max_model_len ${TOTAL_LEN} \
    --vllm_gpu_memory_utilization 0.5 &
TEACHER2_PID=$!
echo "Teacher 2 PID: ${TEACHER2_PID}"

# Wait for servers to be ready
echo "Waiting for teacher servers to start..."
sleep 30

# Check if servers are up
for port in ${TEACHER_PORT_1} ${TEACHER_PORT_2}; do
    if ! curl -s http://localhost:${port}/v1/models > /dev/null 2>&1; then
        echo "ERROR: Teacher server on port ${port} is not responding"
        kill ${TEACHER1_PID} ${TEACHER2_PID} 2>/dev/null || true
        exit 1
    fi
    echo "Teacher server on port ${port} is ready."
done

# === Step 3: Run GRPO with multi-teacher ===
echo "=== Step 3: Run GRPO with multi-teacher_model_server ==="

# Multi-teacher JSON config: each sample routes to exactly one teacher by tag.
# Tags must be non-empty and non-overlapping across teachers.
# Mode 1 (dataset): tags match the auto-injected 'dataset' column (the dataset name).
# Mode 2 (sample):  tags match the 'teacher_tag' column values (math / code).
if [ "${MODE}" = "sample" ]; then
    TAG_1="math"; TAG_2="code"
else
    TAG_1="data/multi_teacher_math.jsonl"; TAG_2="data/multi_teacher_code.jsonl"
fi
MULTI_TEACHER_CONFIG="[{\"url\":\"http://localhost:${TEACHER_PORT_1}\",\"tags\":[\"${TAG_1}\"]},{\"url\":\"http://localhost:${TEACHER_PORT_2}\",\"tags\":[\"${TAG_2}\"]}]"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --model "${STUDENT_MODEL}" \
    --teacher_model_server "${MULTI_TEACHER_CONFIG}" \
    --teacher_kl_coef 1.0 \
    --teacher_tag_key "${TAG_KEY}" \
    --dataset ${DATASET_ARG} \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len ${TOTAL_LEN} \
    --sleep_level 0 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --max_length ${MAX_LENGTH} \
    --max_completion_length ${MAX_COMPLETION} \
    --warmup_ratio 0.1 \
    --save_steps 100 \
    --save_total_limit 1 \
    --torch_dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to tensorboard \
    --attn_impl flash_attn \
    || true

# === Cleanup ===
echo "=== Cleanup ==="
echo "Killing teacher servers..."
kill ${TEACHER1_PID} ${TEACHER2_PID} 2>/dev/null || true
echo "Done. Check ${OUTPUT_DIR} for results."
