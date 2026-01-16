# 8*80G GPU
# GSPO https://arxiv.org/pdf/2507.18071
# hyperparameter
# - epsilon = 3e-4 from paper serction 5.1
# - epsilon_high = 4e-4 from paper serction 5.1
# - steps_per_generation = 4 from paper serction 5.1 (each batch of rollout data is partitioned into four minibatches for gradient updates)
# - beta = 0: zero kl regularization https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --beta 0.0 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --steps_per_generation 4 \
    --importance_sampling_level sequence \
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
