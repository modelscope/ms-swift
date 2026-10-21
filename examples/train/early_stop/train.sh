# v5 early stopping (--early_stop_interval N): stop as soon as the eval metric stalls, instead of running all epochs.
#
# SFT on a small set keeps improving for a while, then the eval loss flattens (or turns up = overfitting). Early
# stopping watches the eval metric every evaluation and halts training once it has failed to improve for
# `--early_stop_interval` consecutive evaluations (dev's EarlyStopCallback; the value is a count of EVALS, i.e.
# HF "patience"). The callback is appended automatically only when --early_stop_interval is set.
#
# THE FLAGS MUST AGREE, or it silently does nothing / errors at the end:
#   --eval_strategy steps + --eval_steps 20      run an eval every 20 optimizer steps (the stall is counted in
#                                                evals, so eval cadence sets the granularity);
#   --metric_for_best_model eval_loss            the watched key. NOTE it is the literal eval-metric name: dev
#                                                looks it up verbatim in the eval result (train_loop._track_best)
#                                                and eval emits 'eval_loss', so 'loss' would NOT match and
#                                                best-model tracking is skipped with a warning;
#   --greater_is_better false                    loss improves by going DOWN;
#   --load_best_model_at_end true                reload the best checkpoint when training ends;
#   --save_steps 20 == --eval_steps               the best step must be checkpointed to be reloaded, and
#   --save_total_limit 10 (NOT 2)                 must not have been pruned -- an aggressive limit makes
#                                                load_best_model_at_end raise FileNotFoundError.
# --split_dataset_ratio carves the val set the eval loss is computed on. num_train_epochs is set high on
# purpose: early stopping is expected to cut the run well short of it.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --eval_strategy steps \
    --eval_steps 20 \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --early_stop_interval 3 \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 10 \
    --split_dataset_ratio 0.05 \
    --output_dir output \
    --report_to tensorboard
