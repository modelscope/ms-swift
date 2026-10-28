# 以下是一个训练脚本示例，用于使用两个奖励模型，包括一个 ORM 和一个 Gen-RM（此处使用 qwen2.5-3B-Instruct）进行 GRPO 训练：

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs format \
    --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \
    --reward_model_plugin genrm my_rmplugin \
    --reward_weights 0.1 1 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --log_completions true \
    --deepspeed zero2


# GRPO 推理vllm * 1 , 训练 * 2
# Batch Size = num_processes * per_device_train_batch_size * gradient_accumulation_steps = 2 * 8 * 8 = 128

CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model Qwen/Qwen2.5-3B-Instruct

CUDA_VISIBLE_DEVICES=0,1 \
WANDB_API_KEY=your_wandb_key \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-3B-Instruct \

    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'zouxuhong/Countdown-Tasks-3to4#50000' \
    --max_length 2048 \
    --max_completion_length 2500 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --output_dir output/GRPO_COUNTDOWN \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 0 \
    --num_generations 8 \
    --temperature 1.0 \

    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.001 \
    --num_iterations 1


    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_countdown format \

    #VLLM
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \



    ## ref
WANDB_API_KEY=your_wandb_api_key \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'okwinds/clevr_cogen_a_train' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GRPO_CLEVR_COUNTDOWN \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 24 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \