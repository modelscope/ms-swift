# v5 cached_dataset: tokenize a dataset ONCE, save it to disk, then train straight from the cache.
#
# Two stages. Stage 1 runs `swift export --to_cached_dataset`, which drives the load + encode chain a
# single time and save_to_disk's the result under <output_dir>/train (plus /val when
# --split_dataset_ratio > 0). It is a CPU job (only the tokenizer is loaded -- no model, no distributed
# init). Stage 2 points `--cached_dataset <output_dir>/train` at that SUBDIRECTORY and skips encoding
# entirely: dev's forward sees the rows already carry input_ids and does not re-encode them.
#
# THE CACHE IS TOKENIZER-BOUND: stage 2 must use the SAME --model (tokenizer) as stage 1, or the cached
# token ids mean something different. --template_mode on the export selects the objective ('train' here;
# 'rlhf' / 'kto' encode chosen/rejected shapes instead) and must match the training run that reads it.
# cached_dataset is map-style, so it cannot be combined with --streaming.
#
# The write half of this feature also lives at examples/export/cached_dataset; this script keeps both
# halves together so the consume side is runnable end to end on a small, cached model.

# --- stage 1: build the cache ---------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift export \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset 'swift/self-cognition#2000' \
    --split_dataset_ratio 0.02 \
    --to_cached_dataset true \
    --template_mode train \
    --dataset_num_proc 4 \
    --output_dir ./self_cognition_cached

# --- stage 2: train from the cache ----------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --cached_dataset ./self_cognition_cached/train \
    --cached_val_dataset ./self_cognition_cached/val \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --eval_strategy steps \
    --eval_steps 20 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --output_dir output \
    --report_to tensorboard
