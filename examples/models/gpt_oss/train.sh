# v5 models / gpt_oss: MoE LoRA SFT of OpenAI gpt-oss-20b.
#
# model_type = `gpt_oss` (swift/dev/model/loader/llm.py:394; GptOssForCausalLM, template `gpt_oss`,
# is_moe; requires transformers>=4.55; the loader's ignore_patterns skips the non-HF metal/ and original/
# weight dirs). gpt-oss-20b is a ~20B sparse MoE (~42GB bf16) that fits a single large-VRAM GPU with LoRA,
# so this uses the default transformers backend (add `--backend megatron` + expert parallel for the 120b or
# for full-parameter training). --router_aux_loss_coef weights the MoE load-balancing loss.
# --model_author/--model_name feed the self-cognition rows so the tuned model reports its own identity.
# --attn_impl flash_attention_2 is pinned: the default SDPA path resolves to a cuDNN attention kernel whose
# bf16 BACKWARD returns NaN on longer rows (clean for ~18 steps, then loss becomes nan); FA2 avoids it and
# is passed straight through to HF (the token must be `flash_attention_2`, not legacy swift's `flash_attn`).
#
# REFERENCE-ONLY: not smoke-run here. openai-mirror/gpt-oss-20b is not cached on this box (a ~42GB
# download). model_type is verified against swift/dev/model/loader/llm.py; the MoE LoRA flag set mirrors
# the smoke-verified examples/train/lora/train.sh plus the router aux-loss knob.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model openai-mirror/gpt-oss-20b \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --router_aux_loss_coef 1e-3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --output_dir output
