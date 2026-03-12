# Qwen3.5-35B-A3B (MoE) Megatron GRPO LoRA Training Example

SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --enable_thinking false \
    --merge_lora true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
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
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 9192 \
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
    --sleep_level 1 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
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
    --padding_free false \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard swanlab
