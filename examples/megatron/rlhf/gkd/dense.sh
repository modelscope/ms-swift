CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-8B-Base \
    --teacher_model Qwen/Qwen3-32B \
    --tuner_type lora \
    --dataset AI-ModelScope/alpaca-gpt4-data-en#2000 AI-ModelScope/alpaca-gpt4-data-zh#2000 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 2 \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --num_train_epochs 1 \
    --lr 5e-6 \
    --log_interval 1 \
    --max_length 8192 \
    --max_completion_length 8192 \
    --attention_backend flash \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 16384 \
    --sleep_level 1 \
    --offload_teacher_model true \
    --recompute_granularity selective \
    --finetune \
    --no_save_optim \
    --no_save_rng \
    --temperature 1.0 \
    --padding_free true \
    --sequence_parallel true
