# 8 * 80G
# docs: https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/deepeyes.html

# First: Deploy Qwen2.5-VL-72B-Instruct for verify
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# swift deploy \
#     --model Qwen/Qwen2.5-VL-72B-Instruct \
#     --vllm_tensor_parallel_size 4

# Second: Run swift rollout to deploy rollout server
# MAX_PIXELS=602112 \
# CUDA_VISIBLE_DEVICES=3 \
# swift rollout \
#     --model Qwen/Qwen2.5-VL-7B-Instruct \
#     --vllm_use_async_engine true \
#     --external_plugins examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py  \
#     --multi_turn_scheduler deepeyes_scheduler \
#     --vllm_max_model_len 8192 \
#     --vllm_gpu_memory_utilization 0.8 \
#     --max_turns 5

# Third: Run swift rlhf to train GRPO model
MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset "path/to/data_0.1.2_visual_toolbox_v2.parquet"\
    "path/to/data_v0.8_visual_toolbox_v2.parquet"\
    "path/to/data_thinklite_reasoning_acc.parquet" \
    --load_from_cache_file true \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8001 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --external_plugins examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py \
    --reward_funcs deepeyes_reward \
    --train_type full \
    --torch_dtype bfloat16 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --deepspeed zero3 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 12 \
    --beta 0 \
    --temperature 0.9 \
    --report_to tensorboard swanlab \
    --log_completions true
