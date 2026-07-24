export PYTHONPATH=$PYTHONPATH:<your_local_megatron_lm_path>
export MEGATRON_LM_PATH=<your_local_megatron_lm_path>
export VLLM_ASCEND_ENABLE_NZ=0
export TASK_QUEUE_ENABLE=0
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
accelerate launch --config_file "./examples/ascend/train/qwen3_5/fsdp2.json" \
    swift/cli/rlhf.py \
        --rlhf_type grpo \
        --model Qwen/Qwen3.5-35B-A3B \
        --model_type qwen3_5_moe \
        --reward_funcs accuracy \
        --use_vllm true \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.4 \
        --vllm_enforce_eager false \
        --vllm_tensor_parallel_size 4 \
        --vllm_max_model_len 16384 \
        --tuner_type lora \
        --lora_rank 8 \
        --lora_alpha 32 \
        --torch_dtype bfloat16 \
        --dataset AI-MO/NuminaMath-TIR#1000 \
        --max_length 1024 \
        --max_completion_length 8192 \
        --overlong_filter true \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --learning_rate 1e-6 \
        --gradient_accumulation_steps 1 \
        --save_strategy 'steps' \
        --eval_strategy 'steps' \
        --eval_steps 1000 \
        --save_steps 1000 \
        --save_total_limit 10 \
        --logging_steps 1 \
        --warmup_ratio 0.01 \
        --dataloader_num_workers 4 \
        --num_generations 4 \
        --temperature 1.0 \
        --log_completions true \
        --sleep_level 2 \
        --offload_model true \
        --offload_optimizer true \
        --gradient_checkpointing false \
        --report_to tensorboard \
        --num_iterations 1 \
        --beta 0.001 \
        --move_model_batches 40
