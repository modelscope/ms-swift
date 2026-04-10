# OPSD Training Script
# Paper: https://arxiv.org/abs/2601.18734
#
# ## Configuration
# - **Teacher**: Base model (disable_adapter)
# - **Student**: LoRA-adapted model
# - **Dataset**: open-r1/OpenThoughts-114k-math
# - **Model**: Qwen3-4B
#
# ## Hyperparameters (follow paper)
# ```
# lr=2e-5, lora_r=64, lora_alpha=128, temp=1.2, beta=0.5, lambda=1
# max_completion_length=2048, effective_batch=32 (1×8×4)
# ```
#
# ## AIME2025 Results (OVERALL)
# | Checkpoint | Accuracy | Improvement |
# |------------|----------|-------------|
# | Base       | 0.1667   | -           |
# | 100 steps  | 0.2667   | +60%        |
#
# ## Evaluation
# ```bash
# swift eval --model Qwen/Qwen3-4B \
#     --adapters output/Qwen3-4B/xxx/checkpoint-xxx \
#     --eval_dataset aime25 --eval_backend Native --infer_backend vllm \
#     --vllm_max_lora_rank 64 \
#     --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
# ```


NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-4B \
    --teacher_model Qwen/Qwen3-4B \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 10240 \
    --sleep_level 1 \
    --external_plugins examples/train/rlhf/opsd/opsd_plugin.py \
    --dataset 'open-r1/OpenThoughts-114k-math' \
    --lmbda 1.0 \
    --beta 0.5 \
    --temperature 1.2 \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --max_steps 1000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 2048 \
    --save_only_model true \
    --gradient_checkpointing true \
    --deepspeed zero0 \
    --attn_impl flash_attn \
    --report_to tensorboard swanlab
