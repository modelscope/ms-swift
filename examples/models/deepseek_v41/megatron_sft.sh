# DeepSeek-V4.1-Flash LoRA SFT (Megatron backend).
#
# Requirements / notes:
# - Real DeepSeek-V4.1-Flash is a ~700B MoE model and needs an H200-class multi-node
#   cluster; the numbers below (EP/PP/DP/VPP) are a starting point, adjust to your hardware.
# - Parallelism support: CP / PP / EP / DP / VPP are supported. TP (tensor parallelism) and
#   SP (sequence parallelism) are NOT supported yet for DeepSeek-V4.1, so keep
#   --tensor_model_parallel_size 1 and do NOT pass --sequence_parallel.
# - Keep --bf16: the checkpoint's FP4 weights are dequantized to BF16 on load. Native FP4
#   training (--fp4) requires Blackwell and is not wired up (absorbed_mla asserts), do not use it.
# - --virtual_pipeline_model_parallel_size (VPP / interleaved pipeline) is optional; it needs
#   enough layers per pipeline stage. Remove it if your layout does not divide evenly.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
megatron sft \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
    --num_train_epochs 1 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --bf16 true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 4 \
    --virtual_pipeline_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --recompute_granularity selective \
    --padding_free true \
    --max_length 8192 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --lr 1e-4 \
    --save_safetensors true \
    --merge_lora false \
    --logging_steps 1 \
    --save_steps 500 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --attention_backend flash \
    --output_dir output
