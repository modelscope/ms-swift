#!/bin/bash
# Smoke test for multi-teacher OPD fixes (short run, max_steps=2).
# Usage: bash scripts/test_multi_teacher_smoke.sh [dataset|sample] [nproc]
set -euo pipefail

# Use hjh_h20 conda env (vLLM + CUDA)
export PATH="/mnt/nas2/anaconda3/envs/hjh_h20/bin:${PATH}"
PYTHON="${PYTHON:-/mnt/nas2/anaconda3/envs/hjh_h20/bin/python}"
SWIFT="${SWIFT:-/mnt/nas2/anaconda3/envs/hjh_h20/bin/swift}"

MODE="${1:-dataset}"
NPROC="${2:-1}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-0.8B}"
TEACHER_MODEL_1="${TEACHER_MODEL_1:-Qwen/Qwen3.5-2B}"
TEACHER_MODEL_2="${TEACHER_MODEL_2:-Qwen/Qwen3.5-0.8B}"
TEACHER_GPU_1="${TEACHER_GPU_1:-5}"
TEACHER_GPU_2="${TEACHER_GPU_2:-6}"
STUDENT_GPU="${STUDENT_GPU:-7}"
TEACHER_VLLM_UTIL="${TEACHER_VLLM_UTIL:-0.25}"
TEACHER_PORT_1=9000
TEACHER_PORT_2=9001
MAX_LENGTH=1024
MAX_COMPLETION=256
TOTAL_LEN=$((MAX_LENGTH + MAX_COMPLETION))
OUTPUT_DIR="output/multi_teacher_smoke_${MODE}_nproc${NPROC}"
DATA_FILE="data/multi_teacher_test.jsonl"

# Free requested ports (swift deploy uses find_free_port and may shift if occupied)
pkill -f "swift/cli/deploy.py" 2>/dev/null || true
for p in ${TEACHER_PORT_1} ${TEACHER_PORT_2}; do
    fuser -k ${p}/tcp 2>/dev/null || true
done
sleep 2

if [ "${MODE}" = "sample" ]; then
    ${PYTHON} -c "
import json, random
random.seed(42)
data = []
for i in range(20):
    tag = 'math' if i % 2 == 0 else 'code'
    if tag == 'math':
        prompt = f'What is {random.randint(1,20)} + {random.randint(1,20)}?'
    else:
        prompt = f'Write a Python function to check if {random.randint(1,100)} is prime.'
    data.append({'messages': [{'role': 'user', 'content': prompt}], 'teacher_tag': tag})
with open('${DATA_FILE}', 'w') as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print(f'Written {len(data)} samples with teacher_tag')
"
    TAG_KEY="teacher_tag"
    DATASET_ARG="${DATA_FILE}"
    TAG_1="math"; TAG_2="code"
else
    ${PYTHON} -c "
import json, random
random.seed(42)
for name, n, kind in [('data/multi_teacher_math.jsonl', 10, 'math'), ('data/multi_teacher_code.jsonl', 10, 'code')]:
    rows = []
    for i in range(n):
        if kind == 'math':
            p = f'What is {random.randint(1,20)} + {random.randint(1,20)}?'
        else:
            p = f'Write a Python function to check if {random.randint(1,100)} is prime.'
        rows.append({'messages': [{'role': 'user', 'content': p}]})
    with open(name, 'w') as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'Written {n} to {name}')
"
    TAG_KEY="dataset"
    DATASET_ARG="data/multi_teacher_math.jsonl data/multi_teacher_code.jsonl"
    TAG_1="data/multi_teacher_math.jsonl"; TAG_2="data/multi_teacher_code.jsonl"
fi

cleanup() {
    echo "=== Cleanup ==="
    kill ${TEACHER1_PID:-} ${TEACHER2_PID:-} 2>/dev/null || true
    wait ${TEACHER1_PID:-} ${TEACHER2_PID:-} 2>/dev/null || true
}
trap cleanup EXIT

# === Step 2: Start teacher servers (sequential — avoid port race & long vLLM init) ===
echo "=== Step 2: Start teacher servers ==="

wait_teacher_ready() {
    local port=$1
    local name=$2
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null; then
            echo "[${name}] ready on port ${port} after ${i}s"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: ${name} on port ${port} not ready after 600s"
    return 1
}

echo "[Teacher 1] Starting ${TEACHER_MODEL_1} on GPU ${TEACHER_GPU_1} port ${TEACHER_PORT_1}..."
CUDA_VISIBLE_DEVICES=${TEACHER_GPU_1} ${SWIFT} deploy \
    --model "${TEACHER_MODEL_1}" \
    --infer_backend vllm \
    --port ${TEACHER_PORT_1} \
    --max_logprobs 1 \
    --max_length ${TOTAL_LEN} \
    --vllm_max_model_len ${TOTAL_LEN} \
    --vllm_gpu_memory_utilization ${TEACHER_VLLM_UTIL} &
TEACHER1_PID=$!
wait_teacher_ready ${TEACHER_PORT_1} "Teacher 1"

echo "[Teacher 2] Starting ${TEACHER_MODEL_2} on GPU ${TEACHER_GPU_2} port ${TEACHER_PORT_2}..."
CUDA_VISIBLE_DEVICES=${TEACHER_GPU_2} ${SWIFT} deploy \
    --model "${TEACHER_MODEL_2}" \
    --infer_backend vllm \
    --port ${TEACHER_PORT_2} \
    --max_logprobs 1 \
    --max_length ${TOTAL_LEN} \
    --vllm_max_model_len ${TOTAL_LEN} \
    --vllm_gpu_memory_utilization ${TEACHER_VLLM_UTIL} &
TEACHER2_PID=$!
wait_teacher_ready ${TEACHER_PORT_2} "Teacher 2"

MULTI_TEACHER_CONFIG="[{\"url\":\"http://localhost:${TEACHER_PORT_1}\",\"tags\":[\"${TAG_1}\"]},{\"url\":\"http://localhost:${TEACHER_PORT_2}\",\"tags\":[\"${TAG_2}\"]}]"

echo "=== GRPO smoke test mode=${MODE} NPROC=${NPROC} GPU=${STUDENT_GPU} ==="
if [ "${NPROC}" -gt 1 ]; then
    STUDENT_GPUS="${STUDENT_GPU},$((STUDENT_GPU + 1))"
else
    STUDENT_GPUS="${STUDENT_GPU}"
fi

CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
NPROC_PER_NODE=${NPROC} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
${SWIFT} rlhf \
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
    --num_generations 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 2 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --max_length ${MAX_LENGTH} \
    --max_completion_length ${MAX_COMPLETION} \
    --save_steps 1000 \
    --torch_dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none \
    --attn_impl flash_attn

echo "=== Smoke test passed: ${OUTPUT_DIR} ==="
