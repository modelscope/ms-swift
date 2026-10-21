# v5 SFT with GaLore gradient projection (causal_lm, single GPU).
#
# `--use_galore true` upgrades the AdamW base optimizer to GaLoreAdamW, which projects the FULL-parameter
# gradient into a low-rank subspace before the step -- memory savings without an adapter. GaLore is
# therefore full-parameter training: `--tuner full`, and validate._check_galore refuses it alongside a
# LoRA/adapter tuner. `--galore_target_modules` is left unset so twinkle's GaLoreConfig projects the
# attn/mlp Linear (+ embedding) weights it defaults to; `--galore_update_proj_gap` is how often the
# projection is recomputed.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tuner full \
    --optim adamw_torch \
    --use_galore true \
    --galore_rank 128 \
    --galore_update_proj_gap 50 \
    --galore_scale 1.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --output_dir output \
    --report_to tensorboard
