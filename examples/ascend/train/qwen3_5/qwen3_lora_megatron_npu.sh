# 16 * 64GiB Ascend A3

# NPU stability environment variables
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1
# NPU memory management environment variables
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# NPU performance environment variables
export TASK_QUEUE_ENABLE=2
export USE_MCORE_GDN=0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NPROC_PER_NODE=16 \
megatron sft \
    --model Qwen/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 16 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --sequence_parallel true \
    \
    --micro_batch_size 2 \
    --global_batch_size 32 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --gradient_accumulation_fusion false \
    --masked_softmax_fusion false \
    \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    \
    --output_dir output/Qwen3.5-35B-A3B \
    --save_steps 100 \
    --max_length 1024 \
    --system 'You are a helpful assistant.' \
    \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot \
    --save_total_limit 3
