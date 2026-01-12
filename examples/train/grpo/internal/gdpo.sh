# 8*80G GPU
# GDPO https://arxiv.org/abs/2601.05242
# hyperparameter
# - scale_rewards = gdpo: Enable Group-wise Direct Preference Optimization logic

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --beta 0.0 \
    --scale_rewards gdpo \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_generations 16 \
    --train_type full \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_model_len 16384 \
    --max_completion_length 8192 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --log_completions true