top_k=64
max_prompt_length=2048
max_completion_length=2048
max_total_length=$((max_prompt_length + max_completion_length))

export IMAGE_MAX_TOKEN_NUM=1024

# Teacher server must be running first:

# CUDA_VISIBLE_DEVICES=0 \
# swift deploy \
#     --model Qwen/Qwen3.5-4B \
#     --infer_backend vllm \
#     --port 8000 \
#     --max_logprobs $top_k \
#     --max_length $max_total_length \
#     --vllm_max_model_len $max_total_length

CUDA_VISIBLE_DEVICES=1,2 \
NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-4B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk $top_k \
    --dataset 'AI-ModelScope/clevr_cogen_a_train' \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --lmbda 1 \
    --seq_kd false \
    --beta 0.5 \
    --torch_dtype bfloat16 \
    --micro_batch_size 2 \
    --global_batch_size 32 \
    --train_iters 500 \
    --lr 5e-5 \
    --lr_warmup_fraction 0.1 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 10 \
    --max_length $max_prompt_length \
    --max_completion_length $max_completion_length \
    --attention_backend flash \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len $max_total_length \
    --sleep_level 1 \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --temperature 1.0 \
    --padding_free true \
    --recompute_granularity selective
