# 8 * 80G

# docs: https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/deepeyes.html
# First: Deploy Qwen2.5-7B-VL-Instruct for verify
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# swift deploy \
#     --model Qwen/Qwen2.5-7B-VL-Instruct \
#     --vllm_tensor_parallel_size 4

MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset "path/to/data_0.1.2_visual_toolbox_v2.parquet"\
    "path/to/data_v0.8_visual_toolbox_v2.parquet"\
    "path/to/data_thinklite_reasoning_acc.parquet" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 8192 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --external_plugins examples/train/grpo/plugin/deepeyes/deepeyes_plugin.py \
    --reward_funcs deepeyes_reward \
    --multi_turn_scheduler deepeyes_scheduler \
    --max_turns 6 \
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
    --num_generations 16 \
    --beta 0 \
    --temperature 0.9 \
    --log_completions true \
