# v5 plugins: load a custom training callback with --external_plugins and select it with --callbacks.
#
# --external_plugins imports my_plugin.py before anything is built, running its
# @PluginRegistry.register('callback', 'progress_printer') so the name resolves. --callbacks then names
# the callbacks to install (dev's build_callbacks resolves the names and appends the built-ins their own
# knobs warrant). The plugin logs a heartbeat every 10 steps and an eval/train-end summary; watch for
# `[progress_printer] ...` lines interleaved with the normal training log.
#
# `callback` is the training-side extension point. The other dev plugin kinds -- reward / tool / sampler
# (RL and inference) -- load through the same --external_plugins flag; see examples/rl and examples/infer.
# An eval is configured (eval_strategy/eval_steps + a val split) so the callback's on_evaluate fires.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --external_plugins examples/train/plugins/my_plugin.py \
    --callbacks progress_printer \
    --dataset 'swift/self-cognition#600' \
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
    --split_dataset_ratio 0.05 \
    --output_dir output \
    --report_to tensorboard
