# v5 thinking-mode control (--enable_thinking): train a hybrid-reasoning model (Qwen3) in think or non-think mode.
#
# Qwen3 is a hybrid thinking model: the same weights can answer directly, or first emit a <think>...</think>
# rationale. Which behaviour a training run teaches is controlled at the TEMPLATE layer, not by a different
# model. `--enable_thinking` (TemplateConfig.enable_thinking) is the switch:
#   true  -> the template runs in thinking mode; the reasoning span is part of the trained target.
#   false -> non-thinking mode; the model is trained to answer without a visible rationale.
#   unset -> follow the checkpoint's default (Qwen3-0.6B defaults to thinking on).
# Related knobs: `--preserve_thinking` keeps think blocks already present in the data verbatim instead of
# re-deriving them; `--add_non_thinking_prefix` (default true) prepends the empty-think prefix so a
# non-thinking turn is explicit; `--loss_scale ignore_empty_think` (see swift/loss_scale/mapping.py) drops an
# empty <think> from the loss.
#
# self-cognition is used only because it is small and cached so the smoke is quick; it demonstrates that the
# flag is accepted and the run converges. For a real reasoning run point --dataset at data whose responses
# actually contain rationales, and pair --enable_thinking true with --preserve_thinking.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen3-0.6B \
    --task_type causal_lm \
    --enable_thinking true \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --output_dir output \
    --report_to tensorboard
