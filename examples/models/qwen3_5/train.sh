# v5 models / qwen3_5: multimodal LoRA SFT of the Qwen3.5 flagship (Qwen/Qwen3.5-4B).
#
# model_type = `qwen3_5` (swift/dev/model/loader/qwen.py:281; class Qwen3_5ForConditionalGeneration,
# template `qwen3_5`). swift resolves it from --model, so no explicit --model_type/--template is needed.
# Qwen3.5 is a MULTIMODAL family (vision + audio keep-alive); this recipe fine-tunes it on an image-text
# corpus with LoRA on the LLM while --freeze_vit/--freeze_aligner keep the vision tower and projector
# frozen (adapter_config defaults, stated explicitly here for clarity). The MoE sibling is a separate
# model_type `qwen3_5_moe` (Qwen3.5-35B-A3B and up) -- see the mcore/packing/fp8 variants in this dir.
#
# Requires transformers>=5.0.0.dev, qwen_vl_utils, decord (all present on the smoke box; 5.12.1).
# --attn_impl flash_attention_2 is pinned: the default SDPA path resolves to a cuDNN attention kernel whose
# bf16 BACKWARD returns NaN on longer rows (clean for ~18 steps, then grad_norm goes non-finite and loss
# becomes nan); FA2 avoids it and is passed straight through to HF (the token must be `flash_attention_2`,
# not legacy swift's `flash_attn`). The Qwen3.5 vision tower accepts FA2 too -- see the smoke note below.
# SMOKE-VERIFIED: this exact flag set was run on Qwen/Qwen3.5-4B with a bounded shrink only (dataset
# slice #2000 -> #16, --max_steps 3, --logging_steps 1, single GPU); swift resolved model_type qwen3_5
# (agent_template qwen3_5), the vision tower loaded frozen under flash_attention_2, and the loss fell
# 2.69 -> 2.16 -> 0.89 over the 3 steps (eval_loss 0.55) -- a real multimodal SFT descent.
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --dataset 'OK-VQA_train#2000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 4 \
    --output_dir output \
    --report_to tensorboard
