# v5 legacy-format compatibility: the SAME MOPD run over a legacy `query`/`response` dataset.
#
# This is examples/v5/rl/mopd/mopd.sh with one change -- the dataset. examples/v5/rl/data/
# legacy_query.jsonl uses the flat legacy columns `query` / `response` (and their many aliases:
# `instruction`, `input`, `question`, `problem` -> query; `answer`, `output`, `target` -> response)
# instead of the canonical OpenAI `messages` list.
#
# The dev format converter normalizes those legacy columns into `messages` at load time
# (swift.dev.dataset.format_converter.response / alpaca), so this is ENTRY compatibility only: past the
# converter everything flows as the standard `messages` -> twinkle Trajectory -> InputFeature, exactly as
# in the messages-format examples. New datasets should prefer the `messages` form (see mopd.sh); this file
# exists to show an existing legacy dataset runs unchanged under `swift rl`.
#
# `--lmbda 1.0` samples on-policy every step, so the legacy `response` column is not used as a target
# (only the prompt is). The teacher setup, flags and placement are identical to mopd.sh -- see that file's
# header for what MOPD does.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type mopd \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B \
    --teacher_weights 0.6 0.4 \
    --dataset examples/v5/rl/data/legacy_query.jsonl \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --lmbda 1.0 \
    --max_completion_length 256 \
    --max_length 1024 \
    --max_steps 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --save_steps 20 \
    --save_total_limit 2 \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard
