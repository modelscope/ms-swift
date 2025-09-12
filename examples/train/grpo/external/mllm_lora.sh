# For LoRA Training, set following parameters to speed up weight update
# ```bash
#   --vllm_enable_lora true
#   --vllm_max_lora_rank xxx # same as lora_rank in training script
# ```

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# swift rollout \
#     --model Qwen/Qwen2.5-VL-7B-Instruct \
#     --vllm_data_parallel_size 2 \
#     --vllm_tensor_parallel_size 2 \
#     --vllm_enable_lora true \
#     --vllm_max_lora_rank 16


CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/clevr_cogen_a_train' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard swanlab \
    --num_iterations 1 \
    --beta 0.001
