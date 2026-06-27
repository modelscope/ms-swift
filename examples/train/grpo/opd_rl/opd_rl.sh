# On-Policy Distillation as RL (OPD-RL): the signed teacher log-ratio as an advantage in GRPO.
#
# Unlike GKD (`--rlhf_type gkd`), which back-propagates a supervised JSD/KL loss, OPD-RL
# keeps the GRPO policy-gradient pipeline (PG OPD) and injects the per-token signed teacher
# log-ratio `teacher_logp - student_logp` on the student-sampled tokens as an *advantage*
# (the k1 reverse-KL reward; teacher-preferred tokens get a positive advantage). Because the
# update only flows through the sampled token, a single-token teacher logp is enough (no top-k).
#
# To turn a GKD script into OPD-RL, just change `--rlhf_type gkd` to `--rlhf_type grpo`:
# the same `--teacher_model` (or `--teacher_model_server`) is reused. With no
# `--reward_funcs`, the teacher signal is the sole training signal (pure distillation). Add
# `--reward_funcs` to mix task reward with the teacher signal.
#
# Optional: `--teacher_kl_coef` (default 1.0) scales the teacher advantage.

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --teacher_model Qwen/Qwen3.5-2B \
    --teacher_deepspeed zero3 \
    --advantage_estimator grpo \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 10240 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --offload_teacher_model true \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --beta 0.0 \
    --num_iterations 1 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --log_completions true \
    --report_to tensorboard swanlab
