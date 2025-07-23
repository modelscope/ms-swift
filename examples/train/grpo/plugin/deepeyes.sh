# Deploy Qwen2.5-7B-VL-Instruct for verify

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# swift deploy \
#     --model Qwen/Qwen2.5-7B-VL-Instruct \
#     --vllm_tensor_parallel_size 4


CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-VL-Instruct \
    --dataset "../data/data_0.1.2_visual_toolbox_v2.parquet","../data/data_v0.8_visual_toolbox_v2.parquet", "../data/data_thinklite_reasoning_acc.parquet" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 8192 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --external_plugins examples/train/grpo/plugin/deepeyes.py \
    --reward_funcs deepeyes_reward \
    --multi_turn_scheduler deepeyes_scheduler \
    --max_turns 3 \
    --train_type full \
    --torch_dtype bfloat16 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --deepspeed zero2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --log_completions true \
    --report_to tensorboard swanlab
