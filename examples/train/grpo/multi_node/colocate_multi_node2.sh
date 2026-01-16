export CUDA_VISIBLE_DEVICES=0,1,2,3
export NNODES=2
export NODE_RANK=1
export MASTER_ADDR=xxx.xxx.xxx.xxx
export MASTER_PORT=29500
export NPROC_PER_NODE=4

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Math-7B \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true
