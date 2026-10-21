# v5 Proximal Policy Optimization (PPO) with LoRA, on 3 GPUs through Ray + vLLM.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type ppo` dispatches to
# swift.dev.recipe.ppo.run_ppo.
#
# What PPO does: it is the online RL method WITH a learned value function -- the piece GRPO drops. Each step:
# sample --num_generations completions per prompt from the CURRENT policy (trained weights are synced into the
# vLLM sampler before every rollout), score each completion with a frozen reward MODEL, then compute a
# PER-TOKEN advantage with Generalized Advantage Estimation (GAE) against a trained critic. The policy is
# updated with a PPO-style CLIPPED surrogate (--cliprange) on that advantage plus a per-token KL-to-reference
# penalty (--kl_coef); the critic is updated in the same step by a CLIPPED value loss (--cliprange_value)
# toward the GAE returns. GRPO replaces the critic with a group-relative baseline; PPO learns the baseline.
#
# The FOUR models PPO assembles (this is why it is the heaviest online path):
#   1. policy   -- the trained model (LoRA here);
#   2. critic   -- a trainable seq_cls num_labels=1 VALUE head, built automatically (task='value' keeps its
#                  per-token output V(s_t) instead of pooling to one scalar). It is initialised from the
#                  FIRST reward model in --orm (TRL inits the value function from the reward model), or from
#                  the policy base if the channel carries none. You do NOT configure it separately;
#   3. reference-- the frozen model the KL penalty anchors to. Under LoRA it is the policy's OWN base with the
#                  adapter DISABLED (no second model, no extra GPU); a full-parameter run loads a separate
#                  frozen copy on its own DeviceGroup;
#   4. reward   -- the frozen scalar reward MODEL (a model-id item of --orm), scoring each completion once.
#
# Reward model (READ THIS): PPO's reward is a trained scalar model passed as a MODEL-ID ITEM of the shared
# --orm channel (rules and models share the one heterogeneous list). It MUST share the policy's
# tokenizer/template (the four forwards line up token-for-token), so the safe choice is the same base
# family. For a MEANINGFUL reward, train one on your preference data first with examples/rl/rm/rm.sh, then
# either point --orm at that full checkpoint, or keep the shared base below and attach the RM LoRA with
# `--orm_adapters <rm_ckpt>` (one entry per model id in the channel). Loading the bare base as the reward
# model (as below) initialises its score head randomly: every PPO mechanism (critic, per-token GAE,
# clipped surrogate + value loss, KL) still runs and trains, but the reward signal is untrained until you
# supply a real RM -- so treat the bare-base form as a plumbing demo, not an alignment run.
#
# HOW TO READ THIS RUN: judge convergence by the REWARD climbing AND the critic's EXPLAINED VARIANCE staying
# high -- the clipped-surrogate loss oscillates around zero by construction and hides the trend. Watch:
#   - console head signal:  `step N  loss=...  value_loss=...  reward=<mean total reward>  explained_variance=...`;
#   - tensorboard scalars:  `train/total_reward` (should climb), `train/completion_length`, the critic's
#                           `train/explained_variance` (how well V predicts the returns -- should stay near/
#                           above 0; a collapse means the critic is not keeping up), `train/value_mean`,
#                           `train/return_mean`, `train/value_clip_ratio`, and the policy's PPOMetric stats
#                           `train/clip_ratio`, `train/approx_kl`, `train/policy_confidence`, plus the bare
#                           `loss`/`grad_norm` (policy) and `value_loss`/`value_grad_norm` (critic).
# reward comes from the driver-side CompletionRewardMetric; explained_variance/value_* from the critic's
# PPOValueMetric; clip_ratio/approx_kl from the policy's PPOMetric (which shares GRPOMetric's token stats).
#
# PPO knobs: --num_ppo_epochs (default 4) replays each rollout's captured advantages/returns/old_logps/
# old_values that many times -- the genuine PPO batch re-use; --gamma (1.0) / --lam (0.95) parameterise GAE;
# --whiten_rewards standardises the advantages across the rollout. --num_generations is completions per prompt.
#
# Placement: PPO is Ray-only with a weight-syncable vLLM (or SGLang) sampler, like GRPO/RFT. The dev CLI
# forces vllm_mode (colocate by default) and mode=ray; --nproc_per_node IS required (the driver
# holds no GPU). Device budget = nproc_per_node (policy + critic share one mesh, sampler colocates on them)
# PLUS one disjoint GPU per full-parameter reward model -- so nproc_per_node 2 below needs 3 cards. PPO runs
# SP=1: the per-token value head does not cover sequence-parallel/packed layouts, and --sequence_parallel_size
# > 1 is rejected. --padding_free/--packing are likewise not used by the critic path.
#
# NO GRPO-only controls: --log_completions (and the GRPO clipping/replay/entropy/FIPO knobs) are rejected for
# ppo -- completion logging lives in GRPOLoop's rollout assembly, which PPO's own loop bypasses, so it is a
# dead flag the validator refuses rather than silently ignores.
#
# Attention kernel: if bf16 training on the default SDPA path selects a cuDNN attention kernel whose BACKWARD
# returns NaN on your torch/cuDNN build (loss goes NaN a few steps in, during warmup, while the lr is still ~0
# -- a kernel NaN, not divergence; reproduced on torch 2.11 / cuDNN 9.19 for the offline-preference family),
# pin `--attn_impl flash_attention_2` (needs the flash-attn package and an Ampere+ GPU).
#
# DEVICES: pick FREE cards -- this box keeps GPU 0 busy, so the default below uses 1,3,5 (2 trainer + 1
# reward). Adjust CUDA_VISIBLE_DEVICES and --vllm_gpu_memory_utilization to the cards/memory you actually have.
#
# This is a real training scaffold, not a 2-step demo. For a fast smoke run, shrink everything at once: a
# smaller policy AND reward model (Qwen/Qwen2.5-0.5B-Instruct for both), a tiny slice (`#256`),
# --nproc_per_node 1 (then 2 cards: 1 trainer + 1 reward), --num_generations 2, and cap `--max_steps 4`.
CUDA_VISIBLE_DEVICES=1,3,5 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type ppo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#20000' \
    --orm Qwen/Qwen2.5-7B-Instruct \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --nproc_per_node 2 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --num_generations 2 \
    --num_ppo_epochs 4 \
    --cliprange 0.2 \
    --cliprange_value 0.2 \
    --gamma 1.0 \
    --lam 0.95 \
    --kl_coef 0.05 \
    --temperature 0.7 \
    --max_completion_length 512 \
    --max_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard
