top_k=64
max_prompt_length=2048
max_completion_length=2048
max_total_length=$((max_prompt_length + max_completion_length))

export IMAGE_MAX_TOKEN_NUM=1024

# Teacher server must be running first:

# CUDA_VISIBLE_DEVICES=0 \
# swift deploy \
#     --model Qwen/Qwen3.5-4B \
#     --infer_backend vllm \
#     --port 8000 \
#     --max_logprobs $top_k \
#     --max_length $max_total_length \
#     --vllm_max_model_len $max_total_length

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-4B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk $top_k \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --vllm_server_timeout 600 \
    --dataset 'modelscope/gsm8k' \
    --lmbda 1 \
    --beta 0.5 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --max_length $max_prompt_length \
    --max_completion_length $max_completion_length \
    --warmup_ratio 0.1 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --temperature 1.0 \
    --attn_impl flash_attn \
    --report_to tensorboard swanlab
