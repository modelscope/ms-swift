# v5 models / gemma4: multimodal LoRA SFT of Google Gemma-4.
#
# model_type = `gemma4` (swift/dev/model/loader/gemma.py:213; Gemma4ForConditionalGeneration, template
# `gemma4_nothinking`, tags vision+audio). Siblings: `gemma4_thinking` (gemma-4-31B / 26B-A4B) and
# `gemma4_unified` (gemma-4-12B, Gemma4UnifiedForConditionalGeneration) -- all resolved from --model.
# gemma-4-E2B-it is the smallest (effective-2B MoE) and the one used here. The family needs a shared
# vision/audio keep-alive on text-only micro-batches (the loader's _Gemma4KeepAlive), handled for you.
# --freeze_vit/--freeze_aligner keep the vision + audio towers and projectors frozen while LoRA tunes the
# language model. --attn_impl flash_attention_2 is pinned: the default SDPA path resolves to a cuDNN
# attention kernel whose bf16 BACKWARD returns NaN on longer rows; FA2 avoids it (verified on the sibling
# multimodal qwen3_5 recipe) and is passed straight through to HF as `flash_attention_2` (not `flash_attn`).
#
# REFERENCE-ONLY: not smoke-run here. google/gemma-4-* is a GATED checkpoint (accept the license on
# ModelScope/HF first) and is not cached on this box; the family also `requires` transformers>=5.10.1 for
# the unified variant. model_type is verified against swift/dev/model/loader/gemma.py; the multimodal LoRA
# flag set mirrors the smoke-verified examples/train/multi-modal/train.sh.
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
USE_SWIFT_V5=1 \
swift sft \
    --model google/gemma-4-E2B-it \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --output_dir output
