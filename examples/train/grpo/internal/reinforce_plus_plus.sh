# Reinforce++-Baseline in https://arxiv.org/abs/2501.03262

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29900 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --advantage_estimator reinforce_plus_plus \
    --scale_rewards batch \
    --kl_in_reward true \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --dataset 'AI-ModelScope/clevr_cogen_a_train' \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 16384 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --overlong_filter false \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero1 \
    --log_completions true \
    --report_to tensorboard swanlab \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --attn_impl flash_attention_2
