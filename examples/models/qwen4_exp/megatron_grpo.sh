# 8*135G
# BestPractices/Qwen3_8-Flash-Next-Best-Practice

SYSTEM_PROMPT="Please reason step by step, and put your final answer within \\boxed{}."

PLE_CPU_OFFLOAD=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.8-Flash-Next \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules in_proj out_proj linear_proj linear_qkv \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 12 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --padding_free true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_enable_lora false \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 9216 \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --offload_bridge false \
    --num_train_epochs 1 \
    --global_batch_size 8 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 4 \
    --reward_funcs accuracy \
    --max_length 1024 \
    --max_completion_length 8192 \
    --temperature 1.0 \
    --loss_type grpo \
    --beta 0.0 \
    --lr 5e-5 \
    --save_safetensors true \
    --merge_lora false \
    --logging_steps 1 \
    --log_completions true \
    --output_dir output
