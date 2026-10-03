# v5 On-Policy Self-Distillation (OPSD) with LoRA on a single GPU.
# Paper: https://arxiv.org/abs/2601.18734
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI (swift.dev.cli.rlhf). `--rlhf_type opsd` dispatches
# to swift.dev.recipe.run_opsd.
#
# What OPSD does: ONE model is both teacher and student, differing only in CONTEXT. The student
# conditions on the question alone; the teacher conditions on a PRIVILEGED view (question + a reference
# solution) supplied per-row by the dataset's `teacher_prompt` column. Each step the student generates an
# on-policy response, the teacher re-scores those SAME response tokens under the privileged prompt, and
# the student is pulled toward the teacher's per-token log-probs with a dense sampled-token k3 surrogate
# (twinkle.loss.opsd.OPSDLoss).
#
# Teacher selection here: with NO --teacher_model and no adapter-disable mode, run_opsd uses DYNAMIC
# self-distillation -- the teacher is the student's CURRENT weights on the privileged view (teacher=None).
# Because teacher and student see different context, the distillation signal is non-zero. Set
# --teacher_model <id> to distil from a separate frozen model instead.
#
# Placement: OPSD generates and scores in-process on the driver (mode='local', the default) -- it is NOT
# flipped to Ray, and needs no vLLM. `--lmbda 1.0` means every step samples on-policy (the dataset's own
# completions, if any, are not reused as targets).
#
# `--opsd_reverse true` sets the divergence direction (r = teacher - student, the paper's on-policy
# direction; it is the default and shown here for discoverability). OPSD v1 applies NO distillation
# temperature and NO reference-KL term, so --temperature and --beta are rejected for --rlhf_type opsd
# rather than silently ignored.
#
# The dataset ships `teacher_prompt` pre-built (examples/v5/rl/data/opsd.jsonl). To DERIVE it from a real
# dataset's `solution` column instead, load the accompanying plugin and select the dataset it registers:
#   swift rl --rlhf_type opsd \
#       --external_plugins examples/v5/rl/opsd/opsd_plugin.py \
#       --dataset opsd_synthetic \
#       ... (all other flags identical)
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type opsd \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset examples/v5/rl/data/opsd.jsonl \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --lmbda 1.0 \
    --opsd_reverse true \
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
