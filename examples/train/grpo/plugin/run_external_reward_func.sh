# register customized plugins in plugin.py file

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_PIXELS=602112 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 16384 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/clevr_cogen_a_train' \
    --overlong_filter false \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 4 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero1 \
    --log_completions true \
    --report_to tensorboard swanlab \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --loss_type grpo \
    --vllm_enable_lora false \
    --advantage_estimator grpo
