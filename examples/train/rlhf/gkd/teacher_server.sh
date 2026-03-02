# GKD Training with External Teacher Model Server
#
# This script demonstrates using an external vLLM server as the teacher model
# for knowledge distillation.

# Teacher Server Setup (run in a separate gpu):
# CUDA_VISIBLE_DEVICES=5 swift deploy \
#     --model Qwen/Qwen2.5-14B-Instruct \
#     --infer_backend vllm \
#     --port 8000 \
#     --vllm_engine_kwargs '{"max_logprobs": 64}'

TEACHER_SERVER_URL=${TEACHER_SERVER_URL:-"http://localhost:8001"}
GKD_LOGITS_TOPK=${GKD_LOGITS_TOPK:-64}

NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
