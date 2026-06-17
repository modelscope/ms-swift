# MoE GRPO Server Mode: Qwen3.5-35B-A3B LoRA
# Rollout server on GPUs 0,1 (vLLM TP=2)
# Megatron trainer on GPUs 2,3,4,5 (EP=4, DP=4)
#
# NOTE: global_batch_size and micro_batch_size are completion-level
# DP size = world_size / (CP * TP * PP) = 4 / (1 * 1 * 1) = 4
# global_batch_size = micro_batch_size * DP size * gradient_accumulation_steps (64)
# generation_batch_size = global_batch_size * steps_per_generation (64 * 2 = 128)
# num_of_prompt_to_rollout = generation_batch_size / num_generations (128 / 8 = 16)
# num_of_prompt_to_train = global_batch_size / num_generations (64 / 8 = 8)

SYSTEM_PROMPT="You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."

# Step 1: Start rollout server
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rollout \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 9192 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_enable_prefix_caching true > moe_server_rollout.log 2>&1 &

# Wait for rollout server to be ready
echo "Waiting for rollout server to start..."
until curl -s http://127.0.0.1:8000/health/ > /dev/null 2>&1; do
    sleep 10
done
echo "Rollout server is ready!"

# Step 2: Start Megatron GRPO trainer
CUDA_VISIBLE_DEVICES=2,3,4,5 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --enable_thinking false \
    --merge_lora true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --moe_permute_fusion true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --num_train_epochs 1 \
    --global_batch_size 64 \
    --micro_batch_size 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --target_modules all-linear \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.00 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --logging_steps 1 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --save_steps 20 \
    --attention_backend flash \
    --moe_expert_capacity_factor 2 \
    --temperature 1.0 \
    --padding_free true \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard swanlab
