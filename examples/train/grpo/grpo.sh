nproc_per_node=7 \
MASTER_PORT=29600 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Math-7B \
    --reward_funcs accuracy format \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.8 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 8 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_num_proc 4 \
    --num_generations 7 \
    --use_vllm true \
    --system 'swift/example/train/grpo/prompt.txt' \
    --deepspeed zero3
