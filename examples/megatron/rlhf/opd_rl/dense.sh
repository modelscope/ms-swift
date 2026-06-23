# Megatron On-Policy Distillation as RL (OPD-RL): teacher KL as a GRPO advantage.
#
# Same teacher as Megatron GKD (`--rlhf_type gkd`); the only change is `--rlhf_type grpo`.
# OPD-RL keeps the GRPO policy-gradient pipeline and injects the per-token teacher KL on
# the student-sampled tokens as an *advantage* (post-normalization). With no `--reward_funcs`,
# teacher KL is the sole training signal (pure distillation); add `--reward_funcs` to mix
# task reward with teacher KL. `--teacher_kl_coef` (default 1.0) scales the teacher KL.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B-Base \
    --teacher_model Qwen/Qwen3-32B \
    --tuner_type full \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 2 \
    --advantage_estimator grpo \
    --beta 0.0 \
    --torch_dtype bfloat16 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --num_generations 8 \
    --steps_per_generation 4 \
    --num_train_epochs 1 \
    --lr 1e-6 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 4096 \
    --attention_backend flash \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 16384 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --offload_teacher_model true \
    --recompute_granularity selective \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --temperature 1.0 \
    --padding_free true \
    --sequence_parallel true \
    --log_completions true \
    --train_iters 200 \
    --save_steps 1000
