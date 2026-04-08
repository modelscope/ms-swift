# Teacher server must be running first:
#   CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-logprobs 64

CUDA_VISIBLE_DEVICES=1,2 \
NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 64 \
    --dataset 'modelscope/gsm8k' \
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
    --max_length 2048 \
    --max_completion_length 2048 \
    --attention_backend flash \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --temperature 1.0 \
    --padding_free true \
    --recompute_granularity selective
