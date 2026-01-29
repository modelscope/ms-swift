# Megatron GKD Training with External Teacher Model Server
#
# This script demonstrates using an external vLLM server as the teacher model
# for knowledge distillation with Megatron-SWIFT. This approach is useful when:
# - The teacher model is too large to load alongside the student model
# - You want to separate teacher inference from training for better resource utilization
# - You need to use different model parallelism for student vs teacher
#
# Prerequisites:
# 1. Start the teacher model server first (see below)
# 2. Ensure the server is accessible at the specified URL
#
# Teacher Server Setup (run in a separate terminal):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 swift deploy \
#       --model Qwen/Qwen2-72B-Instruct \
#       --infer_backend vllm \
#       --port 8000 \
#       --vllm_engine_kwargs '{"max_logprobs": 64}'
#
# Or using vLLM directly:
#   vllm serve Qwen/Qwen2-72B-Instruct --max-logprobs 64 --port 8000

TEACHER_SERVER_URL=${TEACHER_SERVER_URL:-"http://localhost:8000"}
GKD_LOGITS_TOPK=${GKD_LOGITS_TOPK:-20}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-8B-Base \
    --teacher_model_server $TEACHER_SERVER_URL \
    --gkd_logits_topk $GKD_LOGITS_TOPK \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en#2000' 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 2 \
    --seq_kd false \
    --lmbda 0 \
    --beta 0.5 \
    --torch_dtype bfloat16 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --max_epochs 1 \
    --lr 5e-6 \
    --log_interval 5 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --attention_backend flash \
    --recompute_granularity selective \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --temperature 0.9 \
    --padding_free true \
    --sequence_parallel true
