# On-Policy Distillation https://thinkingmachines.ai/blog/on-policy-distillation/

# CUDA_VISIBLE_DEVICES=7 \
# swift rollout \
#     --model Qwen/Qwen3-8B-Base \
#     --vllm_max_model_len 24192

NPROC_PER_NODE=7 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-8B-Base \
    --teacher_model Qwen/Qwen3-32B \
    --train_type full \
    --dataset open-thoughts/OpenThoughts3-1.2M#10000 \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16000 \
    --max_completion_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --teacher_deepspeed zero3 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000
