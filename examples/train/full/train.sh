# v5 text SFT (causal_lm), FULL-PARAMETER, on a single GPU.
#
# USE_SWIFT_V5=1 routes `swift sft` to the v5 dev CLI (swift.dev.cli.sft). `--tuner full` maps to a None
# TunerConfig -- no adapter is built and every weight trains (swift/dev/cli/parser.py:select_tuner). This
# is the only difference from examples/train/lora/train.sh: drop the LoRA knobs, train the whole model.
# Full-parameter needs a SMALLER learning rate than LoRA (1e-5 here vs 1e-4 for the adapter), otherwise the
# pretrained weights are perturbed too hard. The 0.5B model keeps the example light; a real full run usually
# spans several GPUs (see examples/train/multi-gpu and, for a sharded optimizer, --backend megatron).
#
# DATASET CHOICE = VISIBLE CONVERGENCE. `swift/self-cognition` is new knowledge the model must fit, so the
# loss falls clearly (a real SFT curve) instead of plateauing the way an already-instruct model does on
# alpaca. Keep --num_train_epochs 3 so the descent completes.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --output_dir output \
    --report_to tensorboard
