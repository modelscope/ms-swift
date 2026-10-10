# Typed-decision (System-1) fine-tuning of Cloudflare/clef.
#
# Clef decodes ALL of a record's questions jointly in ONE forward (the only multi-question-per-forward
# model of the three), scoring each option of every question and softmaxing within each question's own
# option set. `--task_type decision` routes to ScoringTrainer + the decision data collator;
# `--loss_type clef` is label-smoothing CE + Brier over each question's options.
#
# The scoring head (`scoring_head`) is the decision layer; ScoringTrainer re-activates it for joint
# training AUTOMATICALLY (the `--trainable_parameters scoring_head` flag is a no-op in the LoRA branch,
# so it is omitted). Clef ships a MERGED backbone that already carries the decision ability (no adapter
# to continue), so a fresh `--tuner_type lora --target_modules all-linear` is correct here -- unlike
# JEV/OmniJev, which continue-train their shipped adapter via `--adapters`. `--attn_impl eager` matches
# tests/test_align/test_decision_clef_align.py (removes sdpa/flash kernel divergence).
#
# sh examples/train/decision/clef.sh
USE_HF=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model Cloudflare/clef \
    --model_type clef \
    --task_type decision \
    --loss_type clef \
    --dataset clef_decision \
    --tuner_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
