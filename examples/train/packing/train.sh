# v5 sequence packing (--packing true): concatenate many short samples into few dense fixed-length rows.
#
# USE_SWIFT_V5=1 routes `swift sft` to the dev CLI. `--packing true` (DatasetConfig.packing) groups encoded
# samples with a bin-packer into rows of `--packing_length` tokens, so one forward/backward covers several
# logical samples instead of one sample + padding. Packing AUTO-ENABLES padding_free (see
# swift/dev/config/process.py: `Setting padding_free=True because packing is enabled`): the packed row is a
# single [1, total] sequence whose per-sample position_ids restart at 0, and attention is block-diagonal so
# samples cannot see each other. That varlen/block-diagonal attention is why `--attn_impl flash_attention_2`
# is pinned here -- it is the backend that honours the packed layout on this stack.
#
# PACKING IS causal_lm ONLY. Pooled tasks (embedding/reranker/seq_cls) have no legal packed layout -- they
# pool per sequence and cannot recover sub-sequence boundaries after concatenation -- so dev rejects packing
# for them; do not copy this flag into an embedding/reranker/seq_cls script.
#
# --packing_length caps each packed row (keep it <= --max_length). --packing_strategy binpack fills rows
# greedily (fewest rows); 'sequential' concatenates in dataset order. self-cognition rows are short, so
# packing shows its value clearly: ~2000 tiny samples collapse into a few hundred dense rows.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#2000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --packing true \
    --packing_length 1024 \
    --packing_strategy binpack \
    --attn_impl flash_attention_2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
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
