# 8*80G GPU
# CHORD https://arxiv.org/abs/2508.11408
# GRPO total batch = 32(prompts)*8(num_generations) = 256 =  8(gpus) * 4(per_device_train_batch_size) * 8(gradient_accumulation_steps)
# SFT total batch = 64 = 8(gpus) * 1(chord_sft_per_device_train_batch_size) * 8(gradient_accumulation_steps)

# NOTE: We use the same dataset for GRPO and SFT, which may cause overlap (i.e., the same examples to be selected).
# You can pre-download the dataset and manually split it to avoid this.

export CHORD_SYSTEM_PROMPT="You are a helpful assistant that solves MATH problems.
You should first think about the reasoning process in mind and then provide the user with the answer.
You should present your reasoning process using the format: <think>\n...your reasoning process here... </think>\n"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-MO/NuminaMath-TIR \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --beta 0.0 \
    --steps_per_generation 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --chord_sft_per_device_train_batch_size 1 \
    --chord_sft_dataset AI-MO/NuminaMath-TIR \
    --chord_enable_phi_function false \
    --chord_mu_warmup_steps 0 \
    --chord_mu_decay_steps 200 \
    --chord_mu_peak 0.9 \
    --chord_mu_valley 0.05 \
    --num_generations 8 \
    --train_type full \
    --reward_funcs accuracy \
    --system "$CHORD_SYSTEM_PROMPT" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 8192 \
    --max_completion_length 4096 \
    --overlong_filter true \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard swanlab
