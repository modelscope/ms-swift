# v5 Multi-teacher On-Policy Distillation (MOPD) with LoRA, on 3 GPUs through Ray + vLLM.
# Paper: https://arxiv.org/abs/2606.30406 (Open-MOPD)
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI (swift.dev.cli.rlhf). `--rlhf_type mopd` dispatches to
# swift.dev.recipe.run_mopd.
#
# What MOPD does: the student generates ONE on-policy trajectory; K frozen DOMAIN teachers each re-score that
# SAME trajectory token-by-token; their per-token distributions are fused into a single target by WEIGHTED
# PROBABILITY MIXTURE (log sum_k w_k p_k), and the student is pulled toward it with the same sampled-token k3
# surrogate OPSD uses (twinkle.loss.mopd.MOPDLoss). Compared with single-teacher OPSD/GKD, MOPD reduces the
# cross-domain degradation a student suffers when trained on ONE teacher's distribution alone. With a single
# teacher the fused target is exactly that teacher's log-probs, i.e. MOPD degenerates to OPSD.
#
# HOW TO READ THIS RUN: MOPD's loss IS the fused teacher->student divergence (the k3 surrogate), so -- unlike
# RL, where you watch the REWARD -- here the LOSS FALLING is the convergence signal: the student's distribution
# is collapsing onto the mixture of the K teachers'. There is no reward channel (MOPD scores no ground truth).
# Watch the console head signal (`step N  loss=...`) and the tensorboard scalars `loss` +
# `grad_norm` (LossMetric emits them unprefixed; the `train/`-prefixed keys belong to the reward/policy
# components MOPD does not register); both come from twinkle.metric components merged by the shared record
# path, not hand-folded. There is likewise no --log_completions: completion logging is GRPO-only (it lives in
# GRPOLoop's rollout assembly, which the distillation loops bypass), so on MOPD it is a dead flag the config
# validator rejects.
#
# Teachers: --teacher_model is a LIST, one id per domain expert. EVERY teacher must share the student's
# tokenizer, since each re-scores the student's OWN response token ids. This example mixes two Qwen2.5 domain
# experts of the same tokenizer family into a 1.5B student -- a MATH specialist (Qwen2.5-Math-7B-Instruct) and a
# GENERAL instruct model (Qwen2.5-7B-Instruct) -- so the student learns math from the specialist while retaining
# general reasoning/formatting from the instruct model, instead of degrading on either when trained on one alone.
# --teacher_weights is optional (one entry per teacher, non-negative; normalized to sum 1 inside MOPDLoss); unset
# weights every teacher equally. The weights below favor the math expert on a math corpus.
#
# Teacher placement: all K teachers are built as frozen Ray actors that SHARE ONE 'teacher' DeviceGroup -- a
# single card by default, scoring trajectories SERIALLY (forward_only). That one card must therefore hold the K
# teachers' COMBINED weights (two 7B models here, ~28GB in bf16); use --teacher_parallel_spec to give the group
# more GPUs when the teachers are too big for one card. MOPD REJECTS --offload_teacher_model, the single-model
# disable-adapter self-teacher, and --teacher_adapters (each teacher is a distinct full model). The teacher card
# is placed AFTER the trainer/sampler cards.
#
# Dataset: MOPD has NO privileged `teacher_prompt` view (that is OPSD's single-teacher trick) -- teacher and
# student see the SAME on-policy prompt -- so MOPD needs NO plugin and reads a plain prompts dataset. This example
# uses the real hub dataset AI-MO/NuminaMath-TIR (the same one GRPO/RFT/OPSD here train on); the `#2000` suffix
# takes a 2000-row slice. Its rows carry no system turn, so --system supplies the math answering instruction
# (OPSD's plugin injects one per row, which is why the OPSD example does NOT pass --system).
#
# num_generations 1: distillation samples ONE on-policy trajectory per prompt -- there is no group-relative
# baseline as in GRPO, so a group of completions would only multiply the K teachers' scoring cost. The step budget
# is derived from the prompt set x num_train_epochs (NOT a toy --max_steps).
#
# MOPD v1 applies NO reference-KL term, so --beta is REJECTED (not silently ignored). --sampling_temperature sets
# how hot the student ROLLS OUT (a separate role from any distillation temperature, which the k3 surrogate does
# not apply). --opsd_reverse true is the paper's on-policy divergence direction (r = teacher - student; the
# default, inherited from OPSDLoop, shown for discoverability). --lmbda 1.0 samples on-policy every step.
#
# Placement: MOPD is Ray-only with a weight-syncable vLLM sampler -- the dev CLI forces use_vllm=true,
# vllm_mode=colocate and mode=ray for --rlhf_type mopd. --nproc_per_node IS required because in Ray mode the
# driver orchestrates and holds no GPU. colocate shares the trainer GPUs with the sampler (CUDA-IPC weight sync,
# sleep/wake hand-over each step), so --vllm_gpu_memory_utilization must leave room for training.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3 (trainer+sampler,
# colocate) and 4 (the ONE 'teacher' group hosting BOTH 7B teachers): 2 + 1 = 3 GPUs. Adjust CUDA_VISIBLE_DEVICES,
# --nproc_per_node and --vllm_gpu_memory_utilization to the cards/memory you actually have.
#
# This is a real training scaffold (a full pass over a 2000-row slice, --num_train_epochs 1), not a 2-step demo,
# so it downloads two 7B teachers and a 1.5B student and runs a while. For a fast smoke run, shrink everything at
# once: a tiny student with two tiny teachers of the SAME tokenizer family (--model Qwen/Qwen2.5-0.5B-Instruct
# --teacher_model Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B --teacher_weights 0.6 0.4), a tiny slice
# (`AI-MO/NuminaMath-TIR#8`), and cap `--max_steps 4`.
CUDA_VISIBLE_DEVICES=1,3,4 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type mopd \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model Qwen/Qwen2.5-Math-7B-Instruct Qwen/Qwen2.5-7B-Instruct \
    --teacher_weights 0.7 0.3 \
    --dataset 'AI-MO/NuminaMath-TIR#2000' \
    --system "You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}." \
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
