# SAPO https://arxiv.org/abs/2511.20347
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MAX_PIXELS=602112 \
swift rlhf \
    --rlhf_type grpo \
    --loss_type sapo \
    --tau_pos 1 \
    --tau_neg 1.05 \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --learning_rate 1e-6 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 8192 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset AI-ModelScope/clevr_cogen_a_train \
    --overlong_filter false \
    --importance_sampling_level token \
    --max_length 4096 \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --steps_per_generation 32 \
    --save_steps 1000 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero1 \
    --log_completions true \
    --report_to tensorboard swanlab \
    --beta 0.001 \
    --attn_impl flash_attention_2
