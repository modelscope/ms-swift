# OPD-RL with a remote teacher served via `swift rollout` / `swift deploy` (teacher API).
#
# The teacher only needs to score the student-sampled tokens, so OPD-RL requests
# prompt_logprobs=0 (single-token logp) — the top-k=0 special case of the GKD teacher
# API fetch. This is much lighter than the GKD top-k server.
#
# Step 1 — start the teacher server on its own GPUs:
#
#   CUDA_VISIBLE_DEVICES=6,7 \
#   swift deploy \
#       --model Qwen/Qwen3-32B \
#       --infer_backend vllm \
#       --vllm_tensor_parallel_size 2 \
#       --port 8000 \
#       --max_length 10240 \
#       --vllm_max_model_len 10240
#
# Step 2 — run OPD-RL training, pointing --teacher_model_server at the server. With no
# --reward_funcs the teacher KL is the sole signal (pure distillation).

max_prompt_length=2048
max_completion_length=8192
max_total_length=$((max_prompt_length + max_completion_length))

NPROC_PER_NODE=6 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --teacher_model_server http://localhost:8000 \
    --advantage_estimator grpo \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len $max_total_length \
    --sleep_level 1 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --max_length $max_prompt_length \
    --max_completion_length $max_completion_length \
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
