# v5 text SFT (causal_lm) with LoRA on a single GPU.
#
# USE_SWIFT_V5=1 routes `swift sft` to the v5 dev CLI (swift.dev.cli.sft), which parses argv straight
# into the dev Config dataclasses -- there is no legacy SftArguments bridge. `--tuner lora` selects the
# LoRA adapter path (TunerConfig); drop it (or set `--tuner full`) for full-parameter training.
# The 0.5B model keeps the example light; swap in Qwen/Qwen2.5-7B-Instruct for a real run.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
