# v5 padding-free training (--padding_free true): drop pad tokens by flattening a batch into one sequence.
#
# Without padding_free, a batch of B variable-length samples is a [B, T] tensor padded to the longest T, and
# the pad tokens are wasted compute masked out of the loss. `--padding_free true` (TemplateConfig.padding_free)
# instead concatenates the B samples into a single [1, sum_len] row and passes per-sample position_ids that
# restart at 0 for each sample; flash-attention's varlen kernel then keeps attention block-diagonal, so the
# result is mathematically identical to the padded batch but with zero pad tokens. Throughput rises with how
# uneven the sample lengths are.
#
# DIFFERENT FROM --packing: padding_free removes padding WITHIN a batch at collate time (samples stay 1:1 with
# the dataset); packing pre-concatenates samples ACROSS the dataset into fixed-length rows (and turns
# padding_free on itself). Use padding_free alone for a normal SFT run with ragged lengths; use packing when
# samples are much shorter than max_length and you want to fill every row.
#
# --attn_impl flash_attention_2 is required: only the varlen flash kernel consumes the packed position_ids and
# enforces the block-diagonal mask. Like packing, padding_free is causal_lm-only -- pooled tasks
# (embedding/reranker/seq_cls) cannot recover per-sequence pooling boundaries from a flattened row.
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
    --padding_free true \
    --attn_impl flash_attention_2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
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
