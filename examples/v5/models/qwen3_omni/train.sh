# v5 models / qwen3_omni: multimodal LoRA SFT of Qwen3-Omni (Qwen/Qwen3-Omni-30B-A3B-Instruct).
#
# model_type = `qwen3_omni_moe` (swift/dev/model/loader/qwen.py, Qwen3OmniMoeLoader; template
# `qwen3_omni`). swift resolves it from --model, so no explicit --model_type/--template is needed.
# Qwen3-Omni is a vision + video + audio MoE family (thinker/talker split): this recipe fine-tunes the
# thinker's LLM with LoRA while --freeze_vit/--freeze_aligner keep the audio/visual towers and
# projectors frozen. The mixed corpus exercises all three input modalities at once -- text (alpaca),
# image (LaTeX_OCR) and audio (aishell ASR).
#
# Requires transformers>=4.57.dev0, soundfile, decord, qwen_omni_utils>=0.0.9 (all present on this box;
# transformers 5.12.1). The media env vars bound the per-sample visual/audio token budget the same way
# the legacy recipe did. --attn_impl flash_attention_2 is pinned: the default SDPA path lands on a cuDNN
# attention kernel whose bf16 backward returns NaN on longer rows; the token is passed straight through
# to HF, so it must be `flash_attention_2`, not legacy swift's `flash_attn`.
#
# REFERENCE-ONLY: model_type verified against the loader registry; the flag set ports the legacy
# recipe to v5 flag names but was not parse/smoke-verified here. The checkpoint (~70GB, bf16 MoE
# 30B-A3B) is not cached on this box and was not downloaded for a smoke run.
# Needs 2x48GiB+ for the LoRA SFT below (the legacy recipe ran on 2x48GiB); the megatron counterpart for
# larger parallelism lives in examples/v4/megatron/multimodal/omni/.
# MEASURED: not run — FAIL: train LoRA regex bug; infer cu_seqlens_q CUDA error (2026-11-24)
CUDA_VISIBLE_DEVICES=0,1 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#5000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --dataset_num_proc 1 \
    --dataloader_num_workers 4 \
    --output_dir output \
    --report_to tensorboard
