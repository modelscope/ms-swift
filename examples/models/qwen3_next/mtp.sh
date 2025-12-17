# 8 * 60GiB, 10s/it

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --mtp_num_layers 1 \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
              'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-Next-80B-A3B-Instruct \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# swift infer \
#     --model megatron_output/Qwen3-Next-80B-A3B-Instruct/vx-xxx/checkpoint-xxx \
#     --vllm_tensor_parallel_size 4 \
#     --infer_backend vllm \
#     --vllm_max_model_len 8192 \
#     --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#100 \
#     --vllm_gpu_memory_utilization 0.9 \
#     --vllm_speculative_config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
#     --max_new_tokens 2048
