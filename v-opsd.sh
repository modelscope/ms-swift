CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
uv run --no-sync swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-2B \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_max_model_len 64000 \
    --dataset smoke_training/train.jsonl \
    --lmbda 1.0 \
    --beta 0.5 \
    --temperature 1.0 \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 64000 \
    --max_completion_length 4096 \
    --save_only_model true \
    --gradient_checkpointing true \
    --deepspeed zero3 \
    --report_to tensorboard
