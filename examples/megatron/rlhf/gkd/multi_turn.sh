# 2 * 73GiB, multi-turn GKD with math_tip_trick scheduler
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-0.8B \
    --teacher_model Qwen/Qwen3.5-2B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/NuminaMath-TIR#2000' \
    --tensor_model_parallel_size 2 \
    --lmbda 1 \
    --beta 0.5 \
    --temperature 1.0 \
    --torch_dtype bfloat16 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --num_train_epochs 1 \
    --lr 5e-6 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --attention_backend flash \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --offload_teacher_model true \
    --recompute_granularity selective \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --enable_thinking false \
    --multi_turn_scheduler math_tip_trick \
    --max_turns 2 \
    --truncation_strategy delete \
    --padding_free true \
    --sequence_parallel true
