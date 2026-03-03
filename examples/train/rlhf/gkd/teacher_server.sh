# GKD Training with External Teacher Model Server (vLLM)
# ===================== Step 1: Start Teacher Server =====================
# Run in a separate terminal / GPU:
#
#   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
#       --port 8000 \
#       --max-logprobs 64 \
#       --gpu-memory-utilization 0.9

# ========================================================================

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 64 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --sleep_level 0 \
    --dataset 'modelscope/gsm8k' \
    --lmbda 1 \
    --seq_kd false \
    --beta 0.5 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --max_length 2048 \
    --max_completion_length 2048 \
    --warmup_ratio 0.1 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --attn_impl flash_attn \
    --report_to tensorboard swanlab
