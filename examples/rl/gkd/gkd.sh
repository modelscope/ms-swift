# v5 Generalized Knowledge Distillation (GKD) with LoRA, on 3 GPUs through Ray + vLLM.
# Paper: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
#        https://arxiv.org/abs/2306.13649
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI (swift.dev.cli.rl). `--rlhf_type gkd` dispatches to
# swift.dev.recipe.distill.run_gkd.
#
# What GKD does: the student is trained to match a frozen teacher ON SEQUENCES THE STUDENT ITSELF GENERATES,
# so it learns to correct its own mistakes instead of only imitating a fixed corpus. Each round the student
# produces a response, the teacher re-scores those SAME tokens with `forward_only(return_logits=True)`, and the
# student is pulled toward the teacher's FULL-VOCAB distribution with a beta-weighted Jensen-Shannon divergence
# (twinkle.loss.gkd.GKDLoss). This is the key contrast with OPSD/MOPD, which use a logits-free sampled-token k3
# surrogate: GKD scores the teacher's whole per-token distribution.
#
# lmbda -- GKD's signature on/off-policy mixing knob (this is what sets GKD apart from the always-on-policy
# OPSD/MOPD):
#   * lmbda=1.0  purely ON-policy: every round rolls out from the student's live weights. Rides the streaming
#                driver and is the only setting async (--async_mode one_step_off) accepts.
#   * lmbda=0.5  (used here, and the config default) each round COIN-FLIPS: with probability lmbda it rolls out
#                on-policy, otherwise it distils on the DATASET'S OWN completions for that round. Off-policy
#                rounds need the dataset to carry assistant completions (alpaca-gpt4-data-zh below does), run on
#                the synchronous driver, and are why --lmbda<1 is rejected together with any async setting.
#   * lmbda=0.0  fully OFF-policy: static-corpus KD on the dataset completions (no student generation at all).
#
# Teacher selection (every teacher scores with forward_only; there is NO HTTP server -- server teachers were
# removed, all-Ray premise):
#   * SEPARATE FROZEN TEACHER (used here): --teacher_model is a DISTINCT, stronger model (a 7B teacher into a
#     1.5B student). It is built as a frozen Ray actor on its OWN 'teacher' DeviceGroup, so it needs +1 GPU
#     beyond the trainer/sampler cards. It MUST share the student's tokenizer (both Qwen2.5 here), since it
#     re-scores the student's own response token ids. A distinct checkpoint also matters for a visible signal:
#     a self-teacher scores near-identical logits at init, so the divergence -- and the gradient -- starts ~0.
#   * DISABLE-ADAPTER self-teacher: set --teacher_model == --model with a LoRA student -> the teacher becomes
#     the student's own FROZEN BASE (adapter disabled), the paper's headline self-distillation setup, reusing
#     the policy actor (NO extra GPU). Because the base and the LoRA student start identical, expect a small
#     initial divergence that grows as lora_B moves.
#   GKD has NO dynamic-self mode: unlike OPSD, dropping --teacher_model entirely (teacher_model=None without a
#   LoRA student) is an error, not a moving-target self-teacher. GKD always needs an explicit teacher.
#
# GKD-only loss knobs (OPSD/MOPD read none of these; --beta in particular is REJECTED there, not ignored):
#   * --beta (default 0.5) is the JSD mixture weight: beta=0 -> forward KL (mean-seeking), beta=1 -> reverse KL
#     (mode-seeking), beta=0.5 -> symmetric JSD. Shown explicitly below for discoverability.
#   * --temperature (default 0.9) softens the teacher/student logits inside the divergence. It is the
#     DISTILLATION temperature, kept apart from --sampling_temperature (how hot the student ROLLS OUT);
#     sampling_temperature defaults to temperature when unset.
#   * --gkd_logits_topk (unset here) scores only the teacher's top-k logits instead of the full vocab, trading a
#     little fidelity for a large memory cut on big-vocab teachers.
#   * --sft_alpha (unset here, 0) adds a supervised cross-entropy term on the OFF-policy dataset rounds only
#     (useful when the student has not seen the corpus style, e.g. distilling a thinking-format teacher).
#
# HOW TO READ THIS RUN: GKD's loss IS the teacher->student divergence (the beta-JSD), so -- unlike RL, where you
# watch the REWARD -- here a FALLING loss over the run is the convergence signal: the student's distribution is
# collapsing onto the teacher's. There is no reward channel (GKD scores no ground truth). The console head signal
# is `step N  loss=...`. swift registers NO GKD-specific algorithm metric, so the record holds only twinkle's
# auto-registered base metrics, all BARE keys (no `train/` prefix):
#   - `loss` + `grad_norm` (LossMetric): the beta-JSD divergence and its gradient. Per-round it JITTERS, because
#     with --lmbda<1 an off-policy dataset round and an on-policy rollout round have different difficulty -- read
#     the trend over the run, not step to step.
#   - `accuracy` / `correct_tokens` / `total_tokens` (the Accuracy metric): the student's per-token argmax match
#     against the labels over the scored completion tokens. A secondary read on how close the student's greedy
#     decode already is to the target; it is NOT the distillation objective (which is the full-distribution JSD).
#   - `iters` and `learning rate(param group N)` (TrainMetric). No `rewards/*`, no policy or value terms.
# There is likewise no --log_completions: completion logging is GRPO-only (it lives in GRPOLoop's
# rollout assembly, which the distillation loops bypass), so on GKD it is a dead flag the config validator
# rejects.
#
# Dataset: with --lmbda 0.5 the off-policy rounds distil on the dataset's OWN completions, so the dataset must
# carry assistant turns. AI-ModelScope/alpaca-gpt4-data-zh (instruction + output) does; `#5000` takes a 5000-row
# slice. A purely on-policy run (--lmbda 1.0) needs only prompts and can use examples/rl/data/prompts.jsonl.
#
# num_generations 1: distillation samples ONE trajectory per prompt -- there is no group-relative baseline as in
# GRPO, so a group of completions would only multiply the teacher's full-vocab scoring cost. The step budget is
# derived from the prompt set x num_train_epochs (NOT a toy --max_steps).
#
# Placement: GKD is Ray-only with a weight-syncable vLLM sampler -- the dev CLI forces
# vllm_mode=colocate and mode=ray for --rlhf_type gkd. --nproc_per_node IS required because in Ray mode the
# driver orchestrates and holds no GPU. colocate shares the trainer GPUs with the sampler (CUDA-IPC weight sync,
# sleep/wake hand-over each step), so --vllm_gpu_memory_utilization must leave room for training; the separate
# teacher is placed on its own card AFTER the trainer/sampler cards.
#
# DEVICES: pick FREE cards -- this box keeps GPU 0/2 busy, so the default below uses 1,3 (trainer+sampler,
# colocate) and 4 (the separate frozen teacher): 2 + 1 = 3 GPUs. Adjust CUDA_VISIBLE_DEVICES, --nproc_per_node
# and --vllm_gpu_memory_utilization to the cards/memory you actually have; a disable-adapter self-teacher drops
# the third card.
#
# This is a real training scaffold (a full pass over a 5000-row slice, --num_train_epochs 1), not a 2-step demo,
# so it downloads a 7B teacher and a 1.5B student and runs a while. For a fast smoke run, shrink everything at
# once: a tiny student with a tiny DISTINCT teacher of the SAME tokenizer family (--model
# Qwen/Qwen2.5-0.5B-Instruct --teacher_model Qwen/Qwen2.5-0.5B), a tiny slice
# (`AI-ModelScope/alpaca-gpt4-data-zh#256`), --nproc_per_node 1, and cap `--max_steps 4` (with --lmbda 0.5 and
# the default seed the first four rounds hit BOTH an off-policy and on-policy round, so the smoke exercises the
# dataset-completion path AND the vLLM rollout + teacher-scoring path).
CUDA_VISIBLE_DEVICES=1,3,4 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --teacher_model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --lmbda 0.5 \
    --beta 0.5 \
    --temperature 1.0 \
    --num_generations 1 \
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
