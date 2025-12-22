# megatron version of https://github.com/modelscope/ms-swift/blob/main/examples/train/on_policy_distillation.sh
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=3,5,6,7 \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-0.6B-Base \
    --teacher_model Qwen/Qwen3-4B \
    --train_type full \
    --dataset AI-ModelScope/alpaca-gpt4-data-en#2000 AI-ModelScope/alpaca-gpt4-data-zh#2000 \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --lr 1e-6 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16000 \
    --max_completion_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --teacher_deepspeed zero3 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 16384 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --offload_teacher_model true \
    --log_interval 1 \
    --max_epochs 1 \
    --recompute_granularity selective \
    --finetune \
    --num_workers 8 \
    --dataset_num_proc 8
