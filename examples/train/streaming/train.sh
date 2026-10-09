# v5 streaming (--streaming true): train from a dataset read as a lazy stream, not fully downloaded/materialised.
#
# By default dev loads and encodes the whole dataset up front (fine for small sets, impossible for a 100M-row
# corpus). `--streaming true` (DatasetConfig.streaming) switches the source to an IterableDataset: rows are
# pulled, tokenised, and batched on the fly, so memory stays flat and training starts after the first shard is
# fetched rather than after the whole download. `--lazy_tokenize` is implied by streaming (encode per row as it
# arrives); `--shuffle_buffer_size` holds a rolling buffer that is shuffled in place, since a stream cannot be
# globally shuffled without one.
#
# TWO CONSEQUENCES OF STREAMING, both reflected below:
#   1. A stream has no known length, so `--num_train_epochs` cannot size the run -- bound it with `--max_steps`
#      instead. (An epoch over an unbounded stream never ends.)
#   2. Holdout splitting a stream is awkward, so this trains only (`--split_dataset_ratio 0`); use a separate
#      `--val_dataset` if you need eval while streaming.
# The 110k distill set is used to make the point -- streaming pulls only the ~few hundred rows max_steps needs,
# never the full file. Swap in any large hub dataset id.
#
# OVER-LENGTH ROWS: this reasoning-distill corpus has rows longer than max_length, and the default
# truncation_strategy=None FAILS LOUDLY (MaxLengthError) the moment one is reached -- you cannot know up front
# which streamed row is too long. So `--truncation_strategy delete` drops any row exceeding `--max_length`
# instead of crashing (alternatives: 'left'/'right' truncate, 'split' chunks it). max_length is set to 4096 to
# keep most rows and delete only the rare longer one.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT' \
    --streaming true \
    --shuffle_buffer_size 1000 \
    --max_steps 50 \
    --truncation_strategy delete \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --report_to tensorboard
