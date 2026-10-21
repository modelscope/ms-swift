# v5 FlashAttention-3 (--attn_impl flash_attention_3).  *** REFERENCE-ONLY -- see the hardware note below. ***
#
# FA3 is the Hopper-generation rewrite of flash-attention: it exploits the H100/H800 tensor-memory + warp
# specialisation (and FP8 attention) for a large throughput jump over FA2 on long sequences. In v5 the attention
# backend is one flag -- `--attn_impl` -- passed straight through to the transformers model, so switching
# FA2 -> FA3 is only this token. Valid values on the dev (transformers) backend are the HF ones
# (`eager` / `sdpa` / `flash_attention_2` / `flash_attention_3`); the legacy spelling `flash_attn` is rejected.
# For a multimodal model the vision tower has its own `--vit_attn_impl`.
#
# WHY THIS IS REFERENCE-ONLY (cannot be smoked on the authoring box): FA3 requires (a) a Hopper GPU -- compute
# capability 9.0 (H100/H800/H20) -- and (b) the flash-attn 3.x package. The machine these examples were
# validated on is an NVIDIA L20X (Ada, compute capability 8.9) with flash-attn 2.8.3, which has FA2 but no FA3
# kernel; asking for flash_attention_3 there fails at kernel dispatch, not in swift. Everything else in this
# script is the ordinary lora SFT that IS smoke-validated (train/lora/train.sh); only the attn token differs.
# On a Hopper box with flash-attn 3 installed, run it as-is and expect the same convergence, faster.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --attn_impl flash_attention_3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --output_dir output \
    --report_to tensorboard
