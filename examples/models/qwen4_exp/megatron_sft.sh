# 8*70G
# BestPractices/Qwen3_8-Flash-Next-Best-Practice

PLE_CPU_OFFLOAD=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron sft \
    --model Qwen/Qwen3.8-Flash-Next \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
    --num_train_epochs 1 \
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
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --padding_free true \
    --max_length 8192 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --lr 1e-4 \
    --save_steps 500 \
    --output_dir output
