# GKD Training with External Teacher Model Server (vLLM)
#
# This script demonstrates using an external vLLM server as the teacher model
# for knowledge distillation. The teacher server provides prompt_logprobs via
# the /v1/completions endpoint, which requires native vLLM serving (vllm serve).
#
# NOTE: Only `vllm serve` is supported as the teacher server backend, because
# the training code sends raw token IDs via the `prompt` field and uses the
# `prompt_logprobs` parameter in the /v1/completions API. This is a vLLM-native
# feature not available through swift deploy.

# ===================== Step 1: Start Teacher Server =====================
# Run in a separate terminal / GPU:
#
#   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-14B-Instruct \
#       --port 8000 \
#       --max-logprobs 64 \
#       --gpu-memory-utilization 0.9
#
# Wait until the server is ready (shows "Uvicorn running on ...").
# Verify with: curl http://localhost:8000/v1/models
# ========================================================================

TEACHER_SERVER_URL=${TEACHER_SERVER_URL:-"http://localhost:8000"}
GKD_LOGITS_TOPK=${GKD_LOGITS_TOPK:-64}

NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_model_server $TEACHER_SERVER_URL \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 10240 \
    --gkd_logits_topk $GKD_LOGITS_TOPK \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en' \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --max_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_completion_length 2048 \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn
