# v5 On-Policy Self-Distillation (OPSD) with LoRA, on 3 GPUs through Ray + vLLM.
# Paper: https://arxiv.org/abs/2601.18734
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI (swift.dev.cli.rlhf). `--rlhf_type opsd` dispatches
# to swift.dev.recipe.run_opsd.
#
# What OPSD does: the student conditions on the question ALONE; a teacher conditions on a PRIVILEGED view
# (question + a reference solution) supplied per-row by the dataset's `teacher_prompt` column. Each step the
# student generates an on-policy response, the teacher re-scores those SAME response tokens under the
# privileged prompt, and the student is pulled toward the teacher's per-token log-probs with a dense,
# logits-free sampled-token k3 surrogate (twinkle.loss.opsd.OPSDLoss). Because the two views differ, the
# distillation signal is non-zero even when the weights are shared.
#
# HOW TO READ THIS RUN: OPSD's loss IS the teacher->student divergence (the k3 surrogate), so -- unlike RL,
# where you watch the REWARD -- here the LOSS FALLING is the convergence signal: the student's distribution
# is collapsing onto the privileged teacher's. There is no reward channel (OPSD scores no ground truth).
# Watch the console head signal (`step N  loss=...`) and the tensorboard scalars `loss` +
# `grad_norm` (LossMetric emits them unprefixed; the `train/`-prefixed keys belong to the reward/policy
# components OPSD does not register); both come from twinkle.metric components merged by the shared record
# path, not hand-folded. There is likewise no --log_completions: completion logging is GRPO-only (it lives in
# GRPOLoop's rollout assembly, which the distillation loops bypass), so on OPSD it is a dead flag the config
# validator rejects.
#
# Teacher selection (every teacher scores the PRIVILEGED view with forward_only; there is no HTTP server):
#   * SEPARATE FROZEN TEACHER (used here): --teacher_model is a DISTINCT, stronger model (a 7B teacher into a
#     1.5B student). It is built as a frozen Ray actor on its OWN 'teacher' DeviceGroup, so it needs +1 GPU
#     beyond the trainer/sampler cards. It MUST share the student's tokenizer (both Qwen2.5 here), since it
#     re-scores the student's own response token ids.
#   * DISABLE-ADAPTER self-teacher: set --teacher_model == --model with a LoRA student -> the teacher is the
#     student's own FROZEN BASE (adapter disabled) on the privileged view. A stable target, the paper-following
#     setup (legacy examples/train/rlhf/opsd reports +60% AIME2025), and it reuses the policy actor (NO extra
#     GPU).
#   * DYNAMIC self-teacher: drop --teacher_model entirely -> the student's CURRENT weights on the privileged
#     view (a moving target; the teacher_model=None branch the v5 e2e suite covers). Also no extra GPU.
#
# Dataset: OPSD is DEFINED by the privileged `teacher_prompt` column. The accompanying plugin derives it from
# a REAL hub math corpus -- `opsd_numina` streams AI-MO/NuminaMath-TIR (the same dataset GRPO/RFT here train
# on) and builds teacher_prompt from each row's problem/solution. `#5000` takes a 5000-row slice. For an
# OFFLINE run with no hub access, select the plugin's 4-row `opsd_synthetic`, or the pre-built
# examples/v5/rl/data/opsd.jsonl (ships teacher_prompt directly, needs no plugin).
#
# num_generations 1: distillation samples ONE on-policy trajectory per prompt -- there is no group-relative
# baseline as in GRPO, so a group of completions would only multiply the teacher's scoring cost. The step
# budget is derived from the prompt set x num_train_epochs (NOT a toy --max_steps).
#
# OPSD v1 applies NO reference-KL term, so --beta is REJECTED (not silently ignored). --sampling_temperature
# sets how hot the student ROLLS OUT (a separate role from any distillation temperature, which the k3
# surrogate does not apply). --opsd_reverse true is the paper's on-policy divergence direction
# (r = teacher - student; the default, shown for discoverability). The plugin already puts a system prompt in
# every row, so --system is NOT passed here (it would only duplicate the row's own).
#
# Placement: OPSD is Ray-only with a weight-syncable vLLM sampler -- the dev CLI forces use_vllm=true,
# vllm_mode=colocate and mode=ray for --rlhf_type opsd. --nproc_per_node IS required because in Ray mode the
# driver orchestrates and holds no GPU. colocate shares the trainer GPUs with the sampler (CUDA-IPC weight
# sync, sleep/wake hand-over each step), so --vllm_gpu_memory_utilization must leave room for training; the
# separate teacher is placed on its own card AFTER the trainer/sampler cards.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3 (trainer+sampler,
# colocate) and 4 (the separate frozen teacher): 2 + 1 = 3 GPUs. Adjust CUDA_VISIBLE_DEVICES,
# --nproc_per_node and --vllm_gpu_memory_utilization to the cards/memory you actually have; a self-teacher
# (disable-adapter / dynamic) drops the third card.
#
# This is a real training scaffold (a full pass over a 5000-row slice, --num_train_epochs 1), not a 2-step
# demo, so it downloads a 7B teacher and a 1.5B student and runs a while. For a fast smoke run, shrink
# everything at once: a tiny student with a tiny separate teacher of the SAME tokenizer family
# (--model Qwen/Qwen2.5-0.5B-Instruct --teacher_model Qwen/Qwen2.5-0.5B), a tiny slice (`opsd_numina#16`, or
# the offline `opsd_synthetic`), and cap `--max_steps 4`.
CUDA_VISIBLE_DEVICES=1,3,4 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type opsd \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --external_plugins examples/v5/rl/opsd/opsd_plugin.py \
    --dataset 'opsd_numina#5000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --lmbda 1.0 \
    --opsd_reverse true \
    --num_generations 1 \
    --sampling_temperature 1.0 \
    --nproc_per_node 2 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --max_completion_length 2048 \
    --max_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_only_model true \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard
