export USE_SWIFT_V5=0
# MiMo-V2.5 / V2.6-Flash / V2.6-Pro Megatron SFT (mcore-bridge backend).
#
# Install order:
#   pip install -U transformers  # remote-code modeling; tested with 5.x
#   pip install git+https://github.com/modelscope/mcore-bridge.git  # main (mimo_v2 support)
#
# The official checkpoints store quantized weights with a fused qkv layout; mcore-bridge
# dequantizes them to BF16 online while loading, so --model can point straight at the official
# checkpoint. Two official layouts are supported: FP8 dense + MXFP4 experts (V2.6-Flash/Pro,
# mxfp4_block_size=32) and all-FP8 (V2.5, store_dtype=fp8); anything else is rejected loudly.
#
# MTP: --mtp_num_layers auto (default) detects model.mtp.layers.* and trains the draft heads
# with --mtp_loss_scaling_factor 0.1 (default). Set --mtp_num_layers 0 to skip MTP.
#
# Parallelism limits: TP <= min(full KV heads, SWA KV heads) — 4 for Flash/V2.5 (4 full / 8 SWA),
# 8 for Pro (8 / 8). context_parallel_size must stay 1; FP8/FP4 training is unsupported (QK head
# dim 192 is not divisible by the 128 FP8 block size). attention_backend must be fused (or auto):
# the SWA learnable sinks are unsupported by every flash-attention backend.
#
# Memory: the dequantized BF16 experts dominate. Flash/V2.5 (~565/564 GiB experts) fit 4x183GB
# cards at tp1 x ep4 (~150-165 GiB/rank incl. dense+MTP). Pro (~1863 GiB experts) does NOT fit
# 8x183GB cards at any topology (ep8 leaves ~233 GiB/rank > 183): full-weight Pro training needs
# more nodes (e.g. 2 nodes ep16 ~117 GiB/rank) or an expert-subset checkpoint
# (64 experts keeps all 70 layers at ~310 GiB experts, tp1 x ep4 ~97 GiB/rank).
# Online loading also accepts a full-width checkpoint for an expert-subset model: experts
# 0..N-1 are read and the router gate/bias is truncated to the same prefix automatically.
# MoE + TP > 1 requires --sequence_parallel true (Megatron MoE layer raises otherwise).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${WORLD_SIZE:-1} \
NODE_RANK=${RANK:-0} \
NPROC_PER_NODE=8 \
megatron sft \
    --model ${MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-RL} \
    --dataset 'swift/self-cognition#500' \
    --model_name swift-robot \
    --model_author swift \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --finetune true \
    --freeze_vit true \
    --freeze_aligner true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --attention_backend fused \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --padding_free true \
    --bf16 true \
    --split_dataset_ratio 0.01 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_safetensors true \
    --no_save_optim true \
    --no_save_rng true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --logging_steps 1 \
    --output_dir megatron_output/MiMo-V2.6-Flash-RL

# all-linear LoRA on 256 routed experts creates ~18k adapter tensors per rank at ep4 and hits
# TE's compile-time NVTETensor registry cap (20321) in the optimizer's multi_tensor_apply
# (observed on TE 2.18). Restrict the targets to the attention projections instead:
#   --target_modules linear_qkv linear_proj
#
# Full-parameter SFT: replace the tuner block with
#   --tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner true --lr 1e-5
# and raise pipeline_model_parallel_size as memory requires.
#
# Merge the adapter back to HF format after training:
#   swift export --adapters megatron_output/MiMo-V2.6-Flash-RL/vx-xxx/checkpoint-xxx \
#       --merge_lora true --to_hf true
