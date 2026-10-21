# v5 AdaLoRA (--tuner adalora): LoRA that re-allocates rank across modules during training.
#
# Plain LoRA gives every target module the same fixed rank r. AdaLoRA (adaptive LoRA) instead starts each
# module at a larger `--adalora_init_r` and prunes toward an average `--adalora_target_r` over the course of
# training, spending rank where it matters most (by an importance/sensitivity score) and trimming it where it
# does not. Same memory budget at convergence as fixed-rank LoRA of r=target_r, but the allocation is learned
# rather than uniform.
#
# `--tuner adalora` selects it (v5 tuners are exactly lora/adalora/trainable_tokens/full -- there is no
# qlora/dora/rslora; those are LoRA flags, not tuners). The adalora_* knobs (swift/dev/config/adapter_config.py)
# control the schedule: init_r -> target_r is the pruning range, tinit/tfinal/deltaT set when and how often the
# rank budget is updated, beta1/beta2 smooth the importance estimate, orth_reg_weight penalises redundant
# singular directions. target_modules works the same as LoRA ('all-linear' covers every linear layer).
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner adalora \
    --adalora_init_r 12 \
    --adalora_target_r 8 \
    --adalora_deltaT 1 \
    --target_modules all-linear \
    --num_train_epochs 3 \
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
